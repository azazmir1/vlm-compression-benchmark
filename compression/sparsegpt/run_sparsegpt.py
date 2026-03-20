"""
compression/sparsegpt/run_sparsegpt.py
=======================================
SparseGPT: one-shot unstructured pruning using approximate Hessian inverse.

Based on: "Massive Language Models Can Be Accurately Pruned in One-Shot"
          (Frantar & Alistarh, ICML 2023)

Key idea: prune weights one at a time, use approximate Hessian inverse to
update remaining weights to compensate for pruning error. Achieves 50-60%
sparsity with minimal accuracy loss.

Usage:
  python compression/sparsegpt/run_sparsegpt.py \
      --model_id HuggingFaceTB/SmolVLM-256M-Instruct --sparsity 0.50
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

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "sparsegpt"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Vision module detection ──────────────────────────────────────────────────

VISION_MODULE_KEYWORDS = {
    "vision_model", "visual_model", "image_encoder", "vision_encoder",
    "patch_embed", "visual_projection", "img_projection",
    "vision_tower", "vit", "davit", "siglip", "fastvit",
}


def _is_vision_module(name: str) -> bool:
    name_lower = name.lower()
    return any(kw in name_lower for kw in VISION_MODULE_KEYWORDS)


# ── Hessian collection ───────────────────────────────────────────────────────

class HessianCollector:
    """Collects H = X^T X (Hessian approximation) for each Linear layer."""

    def __init__(self, model: nn.Module):
        self.H: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = {}
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
            with torch.no_grad():
                x = inp[0].float()
                if x.dim() == 3:
                    x = x.reshape(-1, x.shape[-1])
                elif x.dim() == 1:
                    x = x.unsqueeze(0)
                n_tokens = x.shape[0]

                # Compute X^T X on GPU, then move to CPU to save VRAM
                xtx = (x.t() @ x).cpu()

                if name in self.H:
                    self.H[name].add_(xtx)
                    self.counts[name] += n_tokens
                else:
                    self.H[name] = xtx.clone()
                    self.counts[name] = n_tokens
        return hook_fn

    def get_hessians(self) -> dict[str, torch.Tensor]:
        """Return averaged Hessian H = (X^T X) / n for each layer (on CPU)."""
        result = {}
        for name in self.H:
            result[name] = self.H[name] / self.counts[name]
        return result

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ── SparseGPT pruning ───────────────────────────────────────────────────────

def apply_sparsegpt_pruning(model: nn.Module, hessians: dict[str, torch.Tensor],
                            sparsity: float, blocksize: int = 128,
                            percdamp: float = 0.01) -> dict:
    """
    Apply SparseGPT pruning with Hessian-based weight reconstruction.

    Vectorized implementation: processes blocks of columns at once using
    matrix operations instead of column-by-column loops.

    Returns pruning stats.
    """
    pruned_params = 0
    total_params = 0
    pruned_layers = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name not in hessians:
            continue

        W = module.weight.data.float()  # (out_features, in_features)
        H = hessians[name].to(W.device)
        del hessians[name]  # free CPU memory after moving to GPU

        rows, cols = W.shape

        if H.shape[0] != cols:
            continue

        # Damping for numerical stability
        damp = percdamp * torch.diag(H).mean()
        H_damp = H + damp * torch.eye(cols, device=H.device)

        # Cholesky for H_inv
        try:
            L = torch.linalg.cholesky(H_damp)
            H_inv = torch.cholesky_inverse(L)
        except RuntimeError:
            H_inv = torch.linalg.pinv(H_damp)

        # Process in blocks — vectorized per-row pruning + reconstruction
        for b_start in range(0, cols, blocksize):
            b_end = min(b_start + blocksize, cols)
            b_size = b_end - b_start

            W_block = W[:, b_start:b_end].clone()
            H_inv_block = H_inv[b_start:b_end, b_start:b_end]
            diag_inv = torch.diag(H_inv_block)

            # Per-row: prune smallest-magnitude weights
            n_prune = int(b_size * sparsity)
            if n_prune == 0:
                continue

            _, idx = torch.topk(W_block.abs(), n_prune, dim=1, largest=False)
            prune_mask = torch.zeros_like(W_block, dtype=torch.bool)
            prune_mask.scatter_(1, idx, True)

            # Compute pruning errors: err = W * prune_mask / diag(H_inv)
            err = W_block.clone()
            err[~prune_mask] = 0.0
            safe_diag = diag_inv.clamp(min=1e-10).unsqueeze(0)
            scaled_err = err / safe_diag  # (rows, b_size)

            # Weight update: W_block -= scaled_err @ H_inv_block
            W_block -= scaled_err @ H_inv_block

            # Re-zero pruned positions
            W_block[prune_mask] = 0.0
            W[:, b_start:b_end] = W_block

        module.weight.data = W.to(module.weight.dtype)

        n_zero = (module.weight.data == 0).sum().item()
        n_total = module.weight.data.numel()
        pruned_params += n_zero
        total_params += n_total
        pruned_layers += 1

    actual_sparsity = pruned_params / total_params if total_params > 0 else 0.0
    logger.info(
        f"SparseGPT pruned {pruned_layers} layers | "
        f"actual sparsity = {actual_sparsity:.4f} "
        f"({pruned_params:,}/{total_params:,} zeros)"
    )
    return {
        "target_sparsity": sparsity,
        "actual_sparsity": round(actual_sparsity, 4),
        "pruned_layers": pruned_layers,
        "pruned_params": pruned_params,
        "total_params": total_params,
        "blocksize": blocksize,
        "percdamp": percdamp,
    }


def measure_sparsity(model: nn.Module) -> float:
    zeros, total = 0, 0
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            zeros += (module.weight == 0).sum().item()
            total += module.weight.numel()
    return zeros / total if total > 0 else 0.0


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SparseGPT pruning pipeline")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--sparsity", type=float, required=True,
                        help="Target sparsity (e.g., 0.50)")
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--n_calib", type=int, default=128,
                        help="Calibration samples for Hessian collection")
    parser.add_argument("--vqav2_n", type=int, default=1000)
    parser.add_argument("--skip_vqav2", action="store_true")
    parser.add_argument("--skip_textvqa", action="store_true")
    parser.add_argument("--skip_pope", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_id = args.model_id
    safe_name = model_id.replace("/", "__")
    sparsity_tag = f"sp{int(args.sparsity * 100)}"
    tag = f"{safe_name}__sparsegpt_{sparsity_tag}"
    out_path = RESULTS_DIR / f"{tag}.json"

    if out_path.exists() and not args.force:
        logger.info(f"Result exists at {out_path}. Skipping.")
        return

    # ── Load model ────────────────────────────────────────────────────────
    logger.info(f"Loading {model_id} (fp16) for SparseGPT pruning...")
    model, processor, meta = load_model(model_id, quant="fp16")
    family = meta.family
    device = str(next(model.parameters()).device)
    num_params = sum(p.numel() for p in model.parameters())

    # ── Calibration: collect Hessians ─────────────────────────────────────
    logger.info(f"Calibration: collecting Hessians ({args.n_calib} samples)...")
    calib_samples = load_vqav2(n_samples=args.n_calib)

    collector = HessianCollector(model)
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

    hessians = collector.get_hessians()
    collector.remove_hooks()
    logger.info(f"  Collected Hessians for {len(hessians)} layers")

    # ── SparseGPT pruning ─────────────────────────────────────────────────
    logger.info(f"Applying SparseGPT at sparsity={args.sparsity}...")
    prune_stats = apply_sparsegpt_pruning(
        model, hessians, args.sparsity, blocksize=args.blocksize,
    )
    actual_sparsity = measure_sparsity(model)
    logger.info(f"Post-pruning sparsity check: {actual_sparsity:.4f}")

    results = {
        "model_id": model_id,
        "family": family,
        "quant": "fp16",
        "pruning_method": "sparsegpt",
        "target_sparsity": args.sparsity,
        "actual_sparsity": prune_stats["actual_sparsity"],
        "pruned_layers": prune_stats["pruned_layers"],
        "blocksize": prune_stats["blocksize"],
        "n_calib_samples": args.n_calib,
        "num_params_M": round(num_params / 1e6, 1),
        "gpu_mem_load_mb": meta.gpu_mem_delta_mb,
        "benchmarks": {},
    }

    # ── Evaluate ──────────────────────────────────────────────────────────
    if not args.skip_vqav2:
        eval_samples = load_vqav2(n_samples=args.vqav2_n)
        results["benchmarks"]["vqav2"] = evaluate_dataset(
            model, processor, eval_samples, family, device,
            "VQAv2", _vqa_accuracy,
        )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    unload_model(model)


if __name__ == "__main__":
    main()
