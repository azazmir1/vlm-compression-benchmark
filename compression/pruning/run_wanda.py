"""
compression/pruning/run_wanda.py
================================
Wanda (Weights AND Activations) pruning pipeline for VLMs.

Based on: "A Simple and Effective Pruning Approach for Large Language Models"
          (Sun et al., ICLR 2024)

Prunes by: importance = |W_ij| * ||X_j||_2
  - W_ij is the weight value
  - X_j is the L2 norm of input activations for column j
  - Pruning is done per output row (per-output granularity)

Requires a small calibration pass (~128 samples) to collect activation norms.

Usage:
  python compression/pruning/run_wanda.py \
      --model_id HuggingFaceTB/SmolVLM-256M-Instruct --sparsity 0.20

  python compression/pruning/run_wanda.py \
      --model_id Qwen/Qwen2.5-VL-7B-Instruct --sparsity 0.40
"""

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.model_loader import load_model, unload_model
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

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "wanda"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Vision module detection (skip vision encoder) ───────────────────────────

VISION_MODULE_KEYWORDS = {
    "vision_model", "visual_model", "image_encoder", "vision_encoder",
    "patch_embed", "visual_projection", "img_projection",
    "vision_tower", "vit", "davit", "siglip", "fastvit",
}


def _is_vision_module(name: str) -> bool:
    name_lower = name.lower()
    return any(kw in name_lower for kw in VISION_MODULE_KEYWORDS)


# ── Wanda pruning ───────────────────────────────────────────────────────────

class ActivationCollector:
    """Collects squared L2 norms of inputs to Linear layers via hooks."""

    def __init__(self, model: nn.Module):
        self.norms: dict[str, torch.Tensor] = {}   # name → sum of ||x||^2
        self.counts: dict[str, int] = {}             # name → sample count
        self.hooks = []
        self._register(model)

    def _register(self, model: nn.Module):
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if _is_vision_module(name):
                continue
            hook = module.register_forward_hook(self._make_hook(name))
            self.hooks.append(hook)

    def _make_hook(self, name: str):
        def hook_fn(module, inp, out):
            x = inp[0]                          # (batch, seq_len, in_features)
            if x.dim() == 2:
                x = x.unsqueeze(0)
            # Sum of squared activations per input feature across batch & seq
            # shape: (in_features,)
            sq_sum = x.float().pow(2).sum(dim=(0, 1))
            n_tokens = x.shape[0] * x.shape[1]

            if name in self.norms:
                self.norms[name] += sq_sum
                self.counts[name] += n_tokens
            else:
                self.norms[name] = sq_sum
                self.counts[name] = n_tokens
        return hook_fn

    def get_input_norms(self) -> dict[str, torch.Tensor]:
        """Return RMS activation norm per input feature for each layer."""
        result = {}
        for name in self.norms:
            # RMS = sqrt(sum_sq / count)
            result[name] = (self.norms[name] / self.counts[name]).sqrt()
        return result

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def apply_wanda_pruning(model: nn.Module, input_norms: dict[str, torch.Tensor],
                        sparsity: float) -> dict:
    """
    Apply Wanda pruning: zero out weights with smallest |W| * ||X|| scores.
    Pruning granularity: per output row.

    Returns sparsity stats.
    """
    pruned_params = 0
    total_params = 0
    pruned_layers = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name not in input_norms:
            continue

        W = module.weight.data           # (out_features, in_features)
        X_norm = input_norms[name].to(W.device)  # (in_features,)

        # Skip if activation norm shape doesn't match weight in_features
        if X_norm.shape[0] != W.shape[1]:
            logger.debug(f"  Skipping {name}: X_norm={X_norm.shape[0]} != W.in={W.shape[1]}")
            continue

        # Wanda importance: |W| * ||X||  per element
        importance = W.abs() * X_norm.unsqueeze(0)

        # Per-row pruning: for each output neuron, prune the least important inputs
        n_prune = int(W.shape[1] * sparsity)
        if n_prune == 0:
            continue

        # Get indices of smallest importance per row
        _, idx = torch.topk(importance, n_prune, dim=1, largest=False)
        mask = torch.ones_like(W, dtype=torch.bool)
        mask.scatter_(1, idx, False)

        # Apply pruning
        W[~mask] = 0.0

        n_zero = (W == 0).sum().item()
        n_total = W.numel()
        pruned_params += n_zero
        total_params += n_total
        pruned_layers += 1

    actual_sparsity = pruned_params / total_params if total_params > 0 else 0.0
    logger.info(
        f"Wanda pruned {pruned_layers} Linear layers | "
        f"actual sparsity = {actual_sparsity:.4f} "
        f"({pruned_params:,}/{total_params:,} zeros)"
    )
    return {
        "target_sparsity": sparsity,
        "actual_sparsity": round(actual_sparsity, 4),
        "pruned_layers": pruned_layers,
        "pruned_params": pruned_params,
        "total_params": total_params,
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
    parser = argparse.ArgumentParser(description="Wanda pruning pipeline")
    parser.add_argument("--model_id",    required=True)
    parser.add_argument("--sparsity",    type=float, required=True,
                        choices=[0.20, 0.40, 0.50],
                        help="Target sparsity (0.20, 0.40, or 0.50)")
    parser.add_argument("--n_calib",     type=int, default=128,
                        help="Number of calibration samples for activation collection")
    parser.add_argument("--vqav2_n",     type=int, default=1000)
    parser.add_argument("--skip_vqav2",  action="store_true")
    parser.add_argument("--skip_textvqa", action="store_true")
    parser.add_argument("--skip_pope",   action="store_true")
    parser.add_argument("--force",       action="store_true",
                        help="Overwrite existing result")
    args = parser.parse_args()

    model_id  = args.model_id
    safe_name = model_id.replace("/", "__")
    sparsity_tag = f"sp{int(args.sparsity * 100)}"
    tag       = f"{safe_name}__wanda_{sparsity_tag}"
    out_path  = RESULTS_DIR / f"{tag}.json"

    if out_path.exists() and not args.force:
        logger.info(f"Result already exists at {out_path}. Skipping.")
        return

    # ── Load in fp16 ─────────────────────────────────────────────────────
    logger.info(f"Loading {model_id} (fp16) for Wanda pruning...")
    model, processor, meta = load_model(model_id, quant="fp16")
    family = meta.family
    device = str(next(model.parameters()).device)
    num_params = sum(p.numel() for p in model.parameters())

    # ── Calibration: collect activation norms ────────────────────────────
    logger.info(f"Calibration pass: collecting activation norms ({args.n_calib} samples)...")
    calib_samples = load_vqav2(n_samples=args.n_calib)

    collector = ActivationCollector(model)
    skipped = 0
    for sample in calib_samples:
        try:
            _ = run_inference(model, processor, sample, family, device)
        except Exception:
            skipped += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    if skipped:
        logger.warning(f"  Calibration: skipped {skipped}/{args.n_calib} samples")

    input_norms = collector.get_input_norms()
    collector.remove_hooks()
    logger.info(f"  Collected activation norms for {len(input_norms)} layers")

    # ── Wanda pruning ────────────────────────────────────────────────────
    logger.info(f"Applying Wanda pruning at sparsity={args.sparsity}...")
    prune_stats = apply_wanda_pruning(model, input_norms, args.sparsity)
    actual_sparsity = measure_sparsity(model)
    logger.info(f"Post-pruning sparsity check: {actual_sparsity:.4f}")

    results: dict = {
        "model_id":         model_id,
        "family":           family,
        "quant":            "fp16",
        "pruning_method":   "wanda",
        "target_sparsity":  args.sparsity,
        "actual_sparsity":  prune_stats["actual_sparsity"],
        "pruned_layers":    prune_stats["pruned_layers"],
        "n_calib_samples":  args.n_calib,
        "num_params_M":     round(num_params / 1e6, 1),
        "gpu_mem_load_mb":  meta.gpu_mem_delta_mb,
        "benchmarks": {},
    }

    # ── Evaluate ─────────────────────────────────────────────────────────
    if not args.skip_vqav2:
        eval_samples = load_vqav2(n_samples=args.vqav2_n)
        results["benchmarks"]["vqav2"] = evaluate_dataset(
            model, processor, eval_samples, family, device,
            "VQAv2", _vqa_accuracy,
        )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Wanda results saved to {out_path}")

    unload_model(model)


if __name__ == "__main__":
    main()
