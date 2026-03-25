"""
compression/qlora/run_qlora.py
===============================
QLoRA (Quantized Low-Rank Adaptation) compression pipeline for VLMs.

Based on: "QLoRA: Efficient Finetuning of Quantized LLMs"
          (Dettmers et al., NeurIPS 2023)

Key idea: Load base model in 4-bit NF4 quantization, attach small LoRA
adapter matrices to attention layers, and fine-tune on a small dataset.
The result is a highly compressed model (~3.5GB for 7B params) that
recovers quality through the learned adapters.

This is a compression method because:
  - The base model is stored in INT4 (4x smaller than fp16)
  - Only the small LoRA adapters (few MB) need full precision
  - A 7B model that won't fit on a Jetson in fp16 can run in QLoRA

Requires: pip install peft bitsandbytes

Usage:
  python compression/qlora/run_qlora.py \
      --model_id Qwen/Qwen2.5-VL-3B-Instruct --lora_rank 16

  python compression/qlora/run_qlora.py \
      --model_id Qwen/Qwen2.5-VL-3B-Instruct --lora_rank 64 --train_samples 500
"""

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.model_loader import load_model, unload_model, detect_family
from evaluation.run_baseline import (
    load_vqav2, run_inference,
    evaluate_dataset, _vqa_accuracy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "qlora"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# LoRA target modules per family (attention projections in the LLM backbone).
# Vision encoder weights are frozen — only LLM layers get adapters.
LORA_TARGETS = {
    "qwen25vl":   ["q_proj", "v_proj", "k_proj", "o_proj"],
    "internvl25": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "gemma3":     ["q_proj", "v_proj", "k_proj", "o_proj"],
    "smolvlm":    ["q_proj", "v_proj", "k_proj", "out_proj"],
    "ovis2":      ["q_proj", "v_proj", "k_proj", "o_proj"],
    "moondream":  ["q_proj", "v_proj"],
    "fastvlm":    ["q_proj", "v_proj", "k_proj", "o_proj"],
    "lfm2vl":     ["q_proj", "v_proj", "k_proj", "o_proj"],
    "nanovlm":    ["q_proj", "v_proj"],
}


def _gpu_mem_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info(torch.cuda.current_device())
    return (total - free) / 1024**2


def _find_lora_targets(model, family: str) -> list[str]:
    """Find which LoRA target module names actually exist in the model."""
    candidates = LORA_TARGETS.get(family, ["q_proj", "v_proj"])
    all_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    targets = [t for t in candidates if t in all_names]
    if not targets:
        # Fallback: scan for common projection layer names
        for fallback in ["q_proj", "v_proj", "query", "value"]:
            if fallback in all_names:
                targets.append(fallback)
    if not targets:
        raise ValueError(
            f"Could not find any LoRA target modules for family '{family}'. "
            f"Available modules: {sorted(all_names)[:20]}..."
        )
    return targets


def load_qlora_model(model_id: str, family: str, lora_rank: int = 16,
                     lora_alpha: int = 32, lora_dropout: float = 0.05,
                     no_quantize: bool = False):
    """
    Load a model in INT4 (NF4) and attach LoRA adapters.

    If no_quantize=True, loads in fp16 instead (pure LoRA, useful on devices
    where BitsAndBytes kernels are not available, e.g. Jetson).

    Returns (model, processor, peft_config, meta).
    """
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError:
        raise ImportError(
            "PEFT not installed. Install with: pip install peft"
        )

    if no_quantize:
        # Pure LoRA on fp16 base (for devices without BnB support)
        logger.info("[LoRA] Loading base model in fp16 (no quantization)...")
        model, processor, meta = load_model(model_id, quant="fp16", family=family)
    else:
        # Load base model in INT4 via the existing model_loader
        model, processor, meta = load_model(model_id, quant="int4", family=family)
        # Prepare model for k-bit training (freeze base, enable gradient checkpointing)
        model = prepare_model_for_kbit_training(model)

    # Find valid target modules
    targets = _find_lora_targets(model, family)
    logger.info(f"[QLoRA] LoRA targets: {targets}")

    # Attach LoRA adapters
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)

    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        f"[QLoRA] Trainable params: {trainable/1e6:.2f}M / "
        f"{total/1e6:.1f}M ({100*trainable/total:.2f}%)"
    )

    return model, processor, peft_config, meta


def finetune_qlora(model, processor, family: str, device: str,
                   train_samples: int = 200, epochs: int = 1,
                   lr: float = 2e-4, max_seq_len: int = 256):
    """
    Quick fine-tune on VQAv2 training split to recover quality after quantization.

    This is intentionally lightweight — just enough to demonstrate that LoRA
    adapters can recover accuracy lost to INT4 quantization.
    """
    from datasets import load_dataset

    # Use a separate portion of VQAv2 validation for training (offset past
    # the eval samples to avoid overlap). lmms-lab/VQAv2 has no train split.
    logger.info(f"[QLoRA] Loading VQAv2 validation for fine-tuning ({train_samples} samples)...")
    ds = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)

    # Skip first 5000 samples (reserved for evaluation) to avoid data leakage
    train_data = []
    skip = 5000
    from collections import Counter
    for idx, item in enumerate(ds):
        if idx < skip:
            continue
        answers = [a["answer"] for a in item["answers"]]
        answer = Counter(answers).most_common(1)[0][0]
        train_data.append({
            "question": item["question"],
            "answer": answer,
        })
        if len(train_data) >= train_samples:
            break

    logger.info(f"[QLoRA] Fine-tuning for {epochs} epoch(s) on {len(train_data)} samples...")

    # Simple text-only fine-tuning (no image — just trains the LLM adapter
    # to produce short VQA-style answers). This is sufficient to demonstrate
    # the compression method; full multimodal fine-tuning would need image
    # preprocessing per family.
    from transformers import AutoTokenizer

    # Get tokenizer from processor
    if hasattr(processor, 'tokenizer'):
        tokenizer = processor.tokenizer
    elif hasattr(processor, 'encode'):
        tokenizer = processor
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model.config._name_or_path, trust_remote_code=True
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )

    total_loss = 0.0
    n_steps = 0

    for epoch in range(epochs):
        for i, sample in enumerate(train_data):
            text = f"Question: {sample['question']} Answer: {sample['answer']}"
            encoding = tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=max_seq_len, padding="max_length",
            )

            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n_steps += 1

            if (i + 1) % 50 == 0:
                avg_loss = total_loss / n_steps
                logger.info(f"  [Epoch {epoch+1}] Step {i+1}/{len(train_data)} loss={avg_loss:.4f}")

    avg_loss = total_loss / n_steps if n_steps > 0 else 0.0
    logger.info(f"[QLoRA] Fine-tuning done. Avg loss: {avg_loss:.4f}")

    model.eval()
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="QLoRA compression pipeline")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha scaling (default: 32)")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--train_samples", type=int, default=200,
                        help="Number of VQAv2 train samples for fine-tuning")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--vqav2_n", type=int, default=1000,
                        help="Number of VQAv2 eval samples")
    parser.add_argument("--no_quantize", action="store_true",
                        help="Use fp16 base instead of INT4 (pure LoRA, for Jetson)")
    parser.add_argument("--skip_finetune", action="store_true",
                        help="Skip fine-tuning (evaluate INT4+LoRA without training)")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_id = args.model_id
    family = detect_family(model_id)
    safe_name = model_id.replace("/", "__")
    lora_tag = "lora" if args.no_quantize else "qlora"
    tag = f"{safe_name}__{lora_tag}_r{args.lora_rank}"
    out_path = RESULTS_DIR / f"{tag}.json"

    if out_path.exists() and not args.force:
        logger.info(f"Result already exists at {out_path}. Skipping.")
        return

    # Load model + attach LoRA adapters
    mem_before = _gpu_mem_mb()
    model, processor, peft_config, meta = load_qlora_model(
        model_id, family,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        no_quantize=args.no_quantize,
    )
    mem_after = _gpu_mem_mb()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = str(next(model.parameters()).device)
    except StopIteration:
        pass

    trainable, total = model.get_nb_trainable_parameters()

    results = {
        "model_id": model_id,
        "family": family,
        "method": "lora" if args.no_quantize else "qlora",
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "trainable_params_M": round(trainable / 1e6, 2),
        "total_params_M": round(total / 1e6, 1),
        "trainable_pct": round(100 * trainable / total, 2),
        "base_quant": "fp16" if args.no_quantize else "int4_nf4",
        "gpu_mem_load_mb": round(mem_after - mem_before, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmarks": {},
    }

    # Fine-tune
    if not args.skip_finetune:
        train_loss = finetune_qlora(
            model, processor, family, device,
            train_samples=args.train_samples,
            epochs=args.epochs,
            lr=args.lr,
        )
        results["train_loss"] = round(train_loss, 4)
        results["train_samples"] = args.train_samples
        results["train_epochs"] = args.epochs

    # Merge LoRA weights into base model for faster inference
    try:
        model = model.merge_and_unload()
        logger.info("[QLoRA] Merged LoRA adapters into base model.")
        results["merged"] = True
    except Exception as e:
        logger.warning(f"[QLoRA] Could not merge adapters: {e}. Using unmerged model.")
        results["merged"] = False

    # Evaluate
    if not args.skip_eval:
        samples = load_vqav2(n_samples=args.vqav2_n)
        results["benchmarks"]["vqav2"] = evaluate_dataset(
            model, processor, samples, family, device,
            "VQAv2", _vqa_accuracy,
        )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"QLoRA results saved to {out_path}")

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
