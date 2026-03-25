"""
compression/ptq/run_gptq.py
=============================
GPTQ (Accurate Post-Training Quantization) pipeline for VLMs.

Based on: "GPTQ: Accurate Post-Training Quantization for Generative
           Pre-trained Transformers" (Frantar et al., ICLR 2023)

Key idea: Layer-wise quantization using approximate second-order information
(Hessian). Quantizes weights one at a time and updates remaining weights to
compensate for quantization error.

Requires: pip install optimum auto-gptq
  (or: pip install gptq  — for newer versions)

Usage:
  python compression/ptq/run_gptq.py --model_id Qwen/Qwen2.5-VL-3B-Instruct
  python compression/ptq/run_gptq.py --model_id OpenGVLab/InternVL2_5-1B --bits 4
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
from models.model_loader import detect_family
from evaluation.run_baseline import (
    load_vqav2, evaluate_dataset, _vqa_accuracy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "gptq"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def quantize_gptq(model_id: str, save_dir: Path,
                  bits: int = 4, group_size: int = 128,
                  n_calib: int = 128, desc_act: bool = False):
    """Quantize model with GPTQ via optimum and save."""
    from optimum.gptq import GPTQQuantizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"[GPTQ] Quantizing {model_id} (bits={bits}, group_size={group_size})...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )

    # Build calibration dataset from tokenizer
    from datasets import load_dataset
    calib_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    calib_texts = [t for t in calib_data["text"] if len(t.strip()) > 0][:n_calib]

    quantizer = GPTQQuantizer(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        dataset=calib_texts,
    )

    quantized_model = quantizer.quantize_model(model, tokenizer)

    save_dir.mkdir(parents=True, exist_ok=True)
    quantized_model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    logger.info(f"[GPTQ] Saved quantized model to {save_dir}")
    return save_dir


def quantize_gptq_autogptq(model_id: str, save_dir: Path,
                            bits: int = 4, group_size: int = 128,
                            n_calib: int = 128):
    """Quantize model with auto-gptq library directly."""
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from transformers import AutoTokenizer

    logger.info(f"[GPTQ/auto-gptq] Quantizing {model_id} "
                f"(bits={bits}, group_size={group_size})...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=False,
    )

    model = AutoGPTQForCausalLM.from_pretrained(
        model_id, quantize_config,
        trust_remote_code=True,
    )

    # Prepare calibration data
    from datasets import load_dataset
    calib_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    calib_texts = [t for t in calib_data["text"] if len(t.strip()) > 0][:n_calib]

    examples = []
    for text in calib_texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        examples.append(enc)

    model.quantize(examples)

    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    logger.info(f"[GPTQ] Saved quantized model to {save_dir}")
    return save_dir


def load_gptq_model(save_dir: Path):
    """Load a pre-quantized GPTQ model for inference."""
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    logger.info(f"[GPTQ] Loading quantized model from {save_dir}...")
    model = AutoModelForCausalLM.from_pretrained(
        str(save_dir), device_map="auto", trust_remote_code=True,
    )

    try:
        processor = AutoProcessor.from_pretrained(str(save_dir), trust_remote_code=True)
    except Exception:
        processor = AutoTokenizer.from_pretrained(str(save_dir), trust_remote_code=True)

    return model, processor


def _gpu_mem_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info(torch.cuda.current_device())
    return (total - free) / 1024**2


def main():
    parser = argparse.ArgumentParser(description="GPTQ quantization pipeline")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8],
                        help="Quantization bits (default: 4)")
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--n_calib", type=int, default=128)
    parser.add_argument("--backend", default="optimum",
                        choices=["optimum", "auto_gptq"],
                        help="GPTQ backend: optimum or auto_gptq")
    parser.add_argument("--vqav2_n", type=int, default=1000)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_id = args.model_id
    family = detect_family(model_id)
    safe_name = model_id.replace("/", "__")
    tag = f"{safe_name}__gptq_w{args.bits}"
    out_path = RESULTS_DIR / f"{tag}.json"

    if out_path.exists() and not args.force:
        logger.info(f"Result already exists at {out_path}. Skipping.")
        return

    # Quantize
    gptq_save_dir = RESULTS_DIR / f"{safe_name}__gptq_w{args.bits}_weights"
    if not gptq_save_dir.exists() or args.force:
        if args.backend == "auto_gptq":
            quantize_gptq_autogptq(model_id, gptq_save_dir,
                                    bits=args.bits, group_size=args.group_size,
                                    n_calib=args.n_calib)
        else:
            quantize_gptq(model_id, gptq_save_dir,
                          bits=args.bits, group_size=args.group_size,
                          n_calib=args.n_calib)

    # Load quantized model
    mem_before = _gpu_mem_mb()
    model, processor = load_gptq_model(gptq_save_dir)
    mem_after = _gpu_mem_mb()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_params = sum(p.numel() for p in model.parameters())

    results = {
        "model_id": model_id,
        "family": family,
        "method": "gptq",
        "bits": args.bits,
        "group_size": args.group_size,
        "backend": args.backend,
        "num_params_M": round(num_params / 1e6, 1),
        "gpu_mem_load_mb": round(mem_after - mem_before, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmarks": {},
    }

    if not args.skip_eval:
        samples = load_vqav2(n_samples=args.vqav2_n)
        results["benchmarks"]["vqav2"] = evaluate_dataset(
            model, processor, samples, family, device,
            "VQAv2", _vqa_accuracy,
        )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"GPTQ results saved to {out_path}")

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
