"""
compression/awq_gptq/run_awq_gptq.py
======================================
AWQ and GPTQ post-training quantization for VLMs.

AWQ: Activation-Aware Weight Quantization (Lin et al., MLSys 2024)
  - Protects salient weights by observing activation magnitudes
  - Per-channel scaling reduces quantization error

GPTQ: Accurate Post-Training Quantization (Frantar et al., ICLR 2023)
  - Layer-wise quantization using approximate second-order (Hessian) information
  - Quantizes weights one at a time, updates remaining weights to compensate

Usage:
  python compression/awq_gptq/run_awq_gptq.py --model_id HuggingFaceTB/SmolVLM-256M-Instruct --method awq
  python compression/awq_gptq/run_awq_gptq.py --model_id Qwen/Qwen2.5-VL-3B-Instruct --method gptq
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.model_loader import load_model, unload_model, detect_family
from evaluation.run_baseline import (
    load_vqav2, load_textvqa, load_pope,
    evaluate_dataset, _vqa_accuracy, _pope_accuracy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "awq_gptq"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── AWQ quantization ─────────────────────────────────────────────────────────

def quantize_awq(model_id: str, save_dir: Path, n_calib: int = 128) -> Path:
    """Quantize model with AutoAWQ (INT4, group_size=128, GEMM kernel)."""
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    logger.info(f"[AWQ] Loading {model_id} for quantization...")
    model = AutoAWQForCausalLM.from_pretrained(model_id, safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }
    logger.info(f"[AWQ] Quantizing with config: {quant_config}")
    model.quantize(tokenizer, quant_config=quant_config)

    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    logger.info(f"[AWQ] Saved to {save_dir}")
    return save_dir


# ── GPTQ quantization ────────────────────────────────────────────────────────

def quantize_gptq(model_id: str, save_dir: Path, n_calib: int = 128) -> Path:
    """Quantize model with AutoGPTQ (INT4, group_size=128)."""
    from transformers import AutoTokenizer, GPTQConfig, AutoModelForCausalLM

    logger.info(f"[GPTQ] Loading {model_id} for quantization...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Build calibration dataset from VQAv2 questions
    calib_samples = load_vqav2(n_samples=n_calib)
    calib_texts = [s["question"] for s in calib_samples]

    gptq_config = GPTQConfig(
        bits=4,
        group_size=128,
        dataset=calib_texts,
        tokenizer=tokenizer,
        desc_act=False,
    )

    logger.info(f"[GPTQ] Quantizing with 4-bit, group_size=128...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=gptq_config,
        device_map="auto",
        trust_remote_code=True,
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    logger.info(f"[GPTQ] Saved to {save_dir}")
    return save_dir


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AWQ/GPTQ quantization pipeline")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--method", required=True, choices=["awq", "gptq"],
                        help="Quantization method: awq or gptq")
    parser.add_argument("--n_calib", type=int, default=128,
                        help="Number of calibration samples")
    parser.add_argument("--vqav2_n", type=int, default=1000)
    parser.add_argument("--skip_vqav2", action="store_true")
    parser.add_argument("--skip_textvqa", action="store_true")
    parser.add_argument("--skip_pope", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_id = args.model_id
    safe_name = model_id.replace("/", "__")
    tag = f"{safe_name}__{args.method}_int4"
    out_path = RESULTS_DIR / f"{tag}.json"

    if out_path.exists() and not args.force:
        logger.info(f"Result exists at {out_path}. Skipping.")
        return

    family = detect_family(model_id)

    # ── Quantize ──────────────────────────────────────────────────────────
    weights_dir = RESULTS_DIR / f"{safe_name}__{args.method}_weights"

    if not weights_dir.exists():
        if args.method == "awq":
            quantize_awq(model_id, weights_dir, args.n_calib)
        else:
            quantize_gptq(model_id, weights_dir, args.n_calib)

    # ── Load quantized model ──────────────────────────────────────────────
    logger.info(f"Loading quantized model from {weights_dir}...")
    model, processor, meta = load_model(str(weights_dir), quant="fp16", family=family)
    device = str(next(model.parameters()).device)
    num_params = sum(p.numel() for p in model.parameters())

    results = {
        "model_id": model_id,
        "family": family,
        "method": args.method,
        "quant": "int4",
        "bits": 4,
        "group_size": 128,
        "num_params_M": round(num_params / 1e6, 1),
        "gpu_mem_load_mb": meta.gpu_mem_delta_mb,
        "compression_ratio": None,
        "benchmarks": {},
    }

    # Compression ratio vs fp16 baseline
    baseline_path = (
        Path(__file__).resolve().parents[2]
        / "results" / "baseline" / f"{safe_name}.json"
    )
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        baseline_mem = baseline.get("gpu_mem_load_mb", 0)
        if baseline_mem > 0 and meta.gpu_mem_delta_mb > 0:
            results["compression_ratio"] = round(baseline_mem / meta.gpu_mem_delta_mb, 2)

    # ── Evaluate ──────────────────────────────────────────────────────────
    if not args.skip_vqav2:
        samples = load_vqav2(n_samples=args.vqav2_n)
        results["benchmarks"]["vqav2"] = evaluate_dataset(
            model, processor, samples, family, device, "VQAv2", _vqa_accuracy,
        )

    if not args.skip_textvqa:
        samples = load_textvqa()
        results["benchmarks"]["textvqa"] = evaluate_dataset(
            model, processor, samples, family, device, "TextVQA", _vqa_accuracy,
        )

    if not args.skip_pope:
        samples = load_pope()
        results["benchmarks"]["pope"] = evaluate_dataset(
            model, processor, samples, family, device, "POPE", _pope_accuracy,
        )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    unload_model(model)


if __name__ == "__main__":
    main()
