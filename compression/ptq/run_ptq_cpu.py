"""
compression/ptq/run_ptq_cpu.py
================================
CPU-compatible Post-Training Quantization using optimum-quanto.

Works on Raspberry Pi 5 and any CPU-only system.
Supports INT8 and INT4 weight quantization via optimum-quanto.

Usage:
  python compression/ptq/run_ptq_cpu.py --model_id HuggingFaceTB/SmolVLM-256M-Instruct --quant int8
  python compression/ptq/run_ptq_cpu.py --model_id HuggingFaceTB/SmolVLM-500M-Instruct --quant int4
  python compression/ptq/run_ptq_cpu.py --model_id vikhyatk/moondream2 --quant int8
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.model_loader import load_model, unload_model
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

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "ptq_cpu"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# RPi5-compatible model families only
RPI5_FAMILIES = {"smolvlm", "nanovlm", "moondream", "florence2"}


def apply_quanto_quantization(model, quant: str):
    """Apply optimum-quanto weight quantization (CPU-compatible)."""
    try:
        from optimum.quanto import quantize, freeze, qint8, qint4
    except ImportError:
        raise ImportError(
            "optimum-quanto is required for CPU quantization. "
            "Install with: pip install optimum-quanto"
        )

    dtype_map = {"int8": qint8, "int4": qint4}
    if quant not in dtype_map:
        raise ValueError(f"Unsupported quant '{quant}'. Use int8 or int4.")

    logger.info(f"Applying quanto {quant} weight quantization...")
    quantize(model, weights=dtype_map[quant])
    freeze(model)
    logger.info("Quantization applied and frozen.")
    return model


def _count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser(description="CPU PTQ via optimum-quanto")
    parser.add_argument("--model_id",     required=True)
    parser.add_argument("--quant",        required=True, choices=["int8", "int4"])
    parser.add_argument("--vqav2_n",      type=int, default=200)
    parser.add_argument("--textvqa_n",    type=int, default=200)
    parser.add_argument("--pope_n",       type=int, default=200)
    parser.add_argument("--skip_vqav2",   action="store_true")
    parser.add_argument("--skip_textvqa", action="store_true")
    parser.add_argument("--skip_pope",    action="store_true")
    parser.add_argument("--force",        action="store_true")
    args = parser.parse_args()

    model_id  = args.model_id
    safe_name = model_id.replace("/", "__")
    tag       = f"{safe_name}__{args.quant}__quanto"
    out_path  = RESULTS_DIR / f"{tag}.json"

    if out_path.exists() and not args.force:
        logger.info(f"Result already exists at {out_path}. Skipping.")
        return

    # Load model in float32 (CPU-safe)
    logger.info(f"Loading {model_id} on CPU (float32)...")
    model, processor, meta = load_model(model_id)
    family = meta.family

    if family not in RPI5_FAMILIES:
        logger.warning(
            f"Family '{family}' may not be RPi5-compatible (>500M params). "
            "Proceeding anyway — check RAM usage carefully."
        )

    device = "cpu"

    # Apply quanto quantization
    model = apply_quanto_quantization(model, args.quant)

    num_params = _count_params(model)

    existing_benchmarks: dict = {}
    if out_path.exists() and args.force:
        with open(out_path) as f:
            existing_benchmarks = json.load(f).get("benchmarks", {})

    results: dict = {
        "model_id":         model_id,
        "family":           family,
        "quant":            args.quant,
        "backend":          "quanto",
        "device":           "cpu",
        "num_params_M":     round(num_params / 1e6, 1),
        "ram_load_mb":      meta.gpu_mem_delta_mb,  # populated with RAM delta on CPU
        "benchmarks":       dict(existing_benchmarks),
    }

    if not args.skip_vqav2:
        samples = load_vqav2(n_samples=args.vqav2_n)
        results["benchmarks"]["vqav2"] = evaluate_dataset(
            model, processor, samples, family, device,
            "VQAv2", _vqa_accuracy,
        )

    if not args.skip_textvqa:
        samples = load_textvqa(n_samples=args.textvqa_n)
        results["benchmarks"]["textvqa"] = evaluate_dataset(
            model, processor, samples, family, device,
            "TextVQA", _vqa_accuracy,
        )

    if not args.skip_pope:
        samples = load_pope(n_samples=args.pope_n)
        results["benchmarks"]["pope"] = evaluate_dataset(
            model, processor, samples, family, device,
            "POPE", _pope_accuracy,
        )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    unload_model(model)


if __name__ == "__main__":
    main()
