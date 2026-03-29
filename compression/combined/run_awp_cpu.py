"""
compression/combined/run_awp_cpu.py
=====================================
AWP: Activation-Aware Weight Pruning + Quantization on CPU.

Based on: "AWP: Activation-Aware Weight Pruning and Quantization" (MERL, ICML 2025)

Strategy: Prune-first then quantize.
  1. Apply Wanda pruning (activation-aware) at target sparsity
  2. Apply INT8 weight quantization via optimum-quanto
  3. Result: sparse + quantized model for maximum memory savings

Pruning-first outperforms quantization-first per the AWP paper.

Usage:
  python compression/combined/run_awp_cpu.py \
      --model_id HuggingFaceTB/SmolVLM-256M-Instruct --sparsity 0.20
  python compression/combined/run_awp_cpu.py \
      --model_id HuggingFaceTB/SmolVLM-256M-Instruct --sparsity 0.40
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.model_loader import load_model, unload_model
from evaluation.run_baseline import (
    load_vqav2, evaluate_dataset, _vqa_accuracy, run_inference,
)
from compression.pruning.run_wanda import ActivationCollector, apply_wanda_pruning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "awp_cpu"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def apply_quanto_int8(model):
    from optimum.quanto import quantize, freeze, qint8
    logger.info("Applying quanto INT8 quantization...")
    quantize(model, weights=qint8)
    freeze(model)
    logger.info("Quantization frozen.")
    return model


def main():
    parser = argparse.ArgumentParser(description="AWP: Wanda pruning + INT8 quantization")
    parser.add_argument("--model_id",     required=True)
    parser.add_argument("--sparsity",     type=float, default=0.20,
                        choices=[0.20, 0.40],
                        help="Wanda pruning sparsity before quantization")
    parser.add_argument("--n_calib",      type=int, default=16,
                        help="Calibration samples for Wanda activation collection")
    parser.add_argument("--vqav2_n",      type=int, default=200)
    parser.add_argument("--skip_vqav2",   action="store_true")
    parser.add_argument("--force",        action="store_true")
    args = parser.parse_args()

    safe_name = args.model_id.replace("/", "__")
    sp_tag    = f"sp{int(args.sparsity * 100)}"
    tag       = f"{safe_name}__awp_wanda{sp_tag}_int8"
    out_path  = RESULTS_DIR / f"{tag}.json"

    if out_path.exists() and not args.force:
        logger.info(f"Result already exists at {out_path}. Skipping.")
        return

    # Step 1: Load model in float32
    logger.info(f"Loading {args.model_id} for AWP compression...")
    model, processor, meta = load_model(args.model_id)
    family = meta.family
    device = "cpu"

    # Step 2: Wanda calibration
    logger.info(f"Wanda calibration pass ({args.n_calib} samples)...")
    calib_samples = load_vqav2(n_samples=args.n_calib)
    collector = ActivationCollector(model)
    skipped = 0
    for sample in calib_samples:
        try:
            _ = run_inference(model, processor, sample, family, device)
        except Exception:
            skipped += 1
            continue
    if skipped:
        logger.warning(f"  Calibration: skipped {skipped}/{args.n_calib} samples")

    input_norms = collector.get_input_norms()
    collector.remove_hooks()
    logger.info(f"  Collected norms for {len(input_norms)} layers")

    # Step 3: Apply Wanda pruning
    logger.info(f"Applying Wanda pruning at sparsity={args.sparsity}...")
    prune_stats = apply_wanda_pruning(model, input_norms, args.sparsity)

    # Step 4: Apply INT8 quantization (pruning-first per AWP paper)
    model = apply_quanto_int8(model)

    num_params = sum(p.numel() for p in model.parameters())

    results = {
        "model_id":       args.model_id,
        "family":         family,
        "method":         "awp_wanda_int8",
        "sparsity":       args.sparsity,
        "quant":          "int8",
        "backend":        "quanto",
        "device":         "cpu",
        "num_params_M":   round(num_params / 1e6, 1),
        "ram_load_mb":    meta.gpu_mem_delta_mb,
        "prune_stats":    prune_stats,
        "benchmarks":     {},
    }

    if not args.skip_vqav2:
        samples = load_vqav2(n_samples=args.vqav2_n)
        results["benchmarks"]["vqav2"] = evaluate_dataset(
            model, processor, samples, family, device, "VQAv2", _vqa_accuracy,
        )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"AWP results saved to {out_path}")

    unload_model(model)


if __name__ == "__main__":
    main()
