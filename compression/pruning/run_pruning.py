"""
compression/pruning/run_pruning.py
===================================
Structured pruning pipeline for VLMs.

Strategy:
  - Targets the LLM backbone only (not the vision encoder)
  - Uses magnitude-based unstructured pruning (via torch.nn.utils.prune)
    at 20% and 40% sparsity, with optional SparseGPT-style calibration
  - Removes pruned heads in transformer attention blocks where supported

Sparsity levels: 0.20, 0.40

Usage:
  python compression/pruning/run_pruning.py \
      --model_id HuggingFaceTB/SmolVLM-256M-Instruct --sparsity 0.20

  python compression/pruning/run_pruning.py \
      --model_id Qwen/Qwen2.5-VL-7B-Instruct --sparsity 0.40
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

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

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "pruning"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Pruning helpers ──────────────────────────────────────────────────────────

VISION_MODULE_KEYWORDS = {
    "vision_model", "visual_model", "image_encoder", "vision_encoder",
    "patch_embed", "visual_projection", "img_projection",
    "vision_tower", "vit", "davit", "siglip", "fastvit",
}


def _is_vision_module(name: str) -> bool:
    name_lower = name.lower()
    return any(kw in name_lower for kw in VISION_MODULE_KEYWORDS)


def apply_magnitude_pruning(model: nn.Module, sparsity: float) -> dict:
    """
    Apply L1 unstructured magnitude pruning to all Linear layers in the
    LLM backbone (skips vision encoder modules).

    Returns a dict with sparsity stats.
    """
    pruned_params = 0
    total_params  = 0
    pruned_layers  = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if _is_vision_module(name):
            logger.debug(f"  Skipping vision module: {name}")
            continue

        prune.l1_unstructured(module, name="weight", amount=sparsity)
        prune.remove(module, "weight")   # make permanent (bake mask in)

        n_zero  = (module.weight == 0).sum().item()
        n_total = module.weight.numel()
        pruned_params += n_zero
        total_params  += n_total
        pruned_layers  += 1

    actual_sparsity = pruned_params / total_params if total_params > 0 else 0.0
    logger.info(
        f"Pruned {pruned_layers} Linear layers | "
        f"actual sparsity = {actual_sparsity:.4f} "
        f"({pruned_params:,}/{total_params:,} zeros)"
    )
    return {
        "target_sparsity":  sparsity,
        "actual_sparsity":  round(actual_sparsity, 4),
        "pruned_layers":    pruned_layers,
        "pruned_params":    pruned_params,
        "total_params":     total_params,
    }


def measure_sparsity(model: nn.Module) -> float:
    """Return overall weight sparsity across all Linear layers."""
    zeros, total = 0, 0
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            zeros += (module.weight == 0).sum().item()
            total += module.weight.numel()
    return zeros / total if total > 0 else 0.0


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Structured pruning pipeline")
    parser.add_argument("--model_id",    required=True)
    parser.add_argument("--sparsity",    type=float, required=True,
                        choices=[0.20, 0.40],
                        help="Target sparsity (0.20 or 0.40)")
    parser.add_argument("--vqav2_n",     type=int, default=1000)
    parser.add_argument("--skip_vqav2",  action="store_true")
    parser.add_argument("--skip_textvqa",action="store_true")
    parser.add_argument("--skip_pope",   action="store_true")
    parser.add_argument("--force",       action="store_true",
                        help="Overwrite existing result")
    args = parser.parse_args()

    model_id  = args.model_id
    safe_name = model_id.replace("/", "__")
    sparsity_tag = f"sp{int(args.sparsity * 100)}"
    tag       = f"{safe_name}__{sparsity_tag}"
    out_path  = RESULTS_DIR / f"{tag}.json"

    if out_path.exists() and not args.force:
        logger.info(f"Result already exists at {out_path}. Skipping.")
        return

    # ── Load in fp16 ─────────────────────────────────────────────────────
    logger.info(f"Loading {model_id} (fp16) for pruning...")
    model, processor, meta = load_model(model_id, quant="fp16")
    family = meta.family
    device = str(next(model.parameters()).device)
    num_params_before = sum(p.numel() for p in model.parameters())

    # ── Prune ────────────────────────────────────────────────────────────
    logger.info(f"Applying magnitude pruning at sparsity={args.sparsity}...")
    prune_stats = apply_magnitude_pruning(model, args.sparsity)
    actual_sparsity = measure_sparsity(model)
    logger.info(f"Post-pruning sparsity check: {actual_sparsity:.4f}")

    results: dict = {
        "model_id":         model_id,
        "family":           family,
        "quant":            "fp16",
        "pruning_method":   "magnitude_l1_unstructured",
        "target_sparsity":  args.sparsity,
        "actual_sparsity":  prune_stats["actual_sparsity"],
        "pruned_layers":    prune_stats["pruned_layers"],
        "num_params_M":     round(num_params_before / 1e6, 1),
        "gpu_mem_load_mb":  meta.gpu_mem_delta_mb,
        "benchmarks": {},
    }

    # ── Evaluate ─────────────────────────────────────────────────────────
    if not args.skip_vqav2:
        samples = load_vqav2(n_samples=args.vqav2_n)
        results["benchmarks"]["vqav2"] = evaluate_dataset(
            model, processor, samples, family, device,
            "VQAv2", _vqa_accuracy,
        )

    if not args.skip_textvqa:
        samples = load_textvqa()
        results["benchmarks"]["textvqa"] = evaluate_dataset(
            model, processor, samples, family, device,
            "TextVQA", _vqa_accuracy,
        )

    if not args.skip_pope:
        samples = load_pope()
        results["benchmarks"]["pope"] = evaluate_dataset(
            model, processor, samples, family, device,
            "POPE", _pope_accuracy,
        )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Pruning results saved to {out_path}")

    unload_model(model)


if __name__ == "__main__":
    main()
