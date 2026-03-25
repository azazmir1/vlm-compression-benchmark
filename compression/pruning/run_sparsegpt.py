"""
compression/pruning/run_sparsegpt.py
=====================================
SparseGPT one-shot pruning pipeline for VLMs.

Based on: "SparseGPT: Massive Language Models Can Be Accurately Pruned
           in One-Shot" (Frantar & Alistarh, ICML 2023)

Key idea: One-shot unstructured pruning using approximate Hessian inverse
for weight reconstruction after pruning. Achieves 50-60% sparsity with
significantly better accuracy than magnitude pruning.

Algorithm per layer:
  1. Collect Hessian H = X^T X from calibration data
  2. For each column (in order), decide whether to prune based on
     reconstruction error
  3. Update remaining weights to compensate for pruning error using
     H^{-1} (approximate inverse Hessian)

Usage:
  python compression/pruning/run_sparsegpt.py \
      --model_id HuggingFaceTB/SmolVLM-256M-Instruct --sparsity 0.50

  python compression/pruning/run_sparsegpt.py \
      --model_id Qwen/Qwen2.5-VL-3B-Instruct --sparsity 0.50 --n_calib 128
"""

import argparse
import gc
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

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

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "sparsegpt"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Vision module detection (skip vision encoder) ───────────────────────────

VISION_MODULE_KEYWORDS = {
    "vision_model", "visual_model", "image_encoder", "vision_encoder",
    "patch_embed", "visual_projection", "img_projection",
    "vision_tower", "vit", "davit", "siglip", "fastvit",
}


def _is_vision_module(name: str) -> bool:
    return any(kw in name.lower() for kw in VISION_MODULE_KEYWORDS)


# ── SparseGPT core algorithm ────────────────────────────────────────────────

class SparseGPTLayer:
    """SparseGPT pruning for a single Linear layer.

    Implements the OBS-based (Optimal Brain Surgeon) one-shot pruning
    with Hessian-based weight reconstruction.
    """

    def __init__(self, layer: nn.Linear):
        self.layer = layer
        self.dev = layer.weight.device
        W = layer.weight.data.clone().float()
        self.rows, self.columns = W.shape
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor):
        """Accumulate Hessian H = X^T X from a batch of inputs.

        inp shape: (batch*seq_len, in_features) or (batch, seq_len, in_features)
        """
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        inp = inp.float().to(self.dev)
        n_tokens = inp.shape[0]
        self.H += inp.T @ inp
        self.nsamples += n_tokens

    def prune(self, sparsity: float, blocksize: int = 128,
              percdamp: float = 0.01) -> dict:
        """Apply SparseGPT pruning to the layer.

        Returns dict with pruning stats.
        """
        W = self.layer.weight.data.clone().float()

        # Normalize Hessian
        H = self.H / self.nsamples
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # Dampening for numerical stability
        damp = percdamp * torch.mean(torch.diag(H))
        diag_idx = torch.arange(self.columns, device=self.dev)
        H[diag_idx, diag_idx] += damp

        # Cholesky decomposition of H
        try:
            H_inv = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(H_inv)
            H_inv = torch.linalg.cholesky(H_inv, upper=True)
        except torch.linalg.LinAlgError:
            # Fallback: add more dampening
            H[diag_idx, diag_idx] += 10 * damp
            H_inv = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(H_inv)
            H_inv = torch.linalg.cholesky(H_inv, upper=True)

        Losses = torch.zeros(self.rows, device=self.dev)
        Err = torch.zeros_like(W)
        mask = torch.zeros_like(W, dtype=torch.bool)

        # Process in blocks for memory efficiency
        for col_start in range(0, self.columns, blocksize):
            col_end = min(col_start + blocksize, self.columns)
            count = col_end - col_start

            W_block = W[:, col_start:col_end].clone()
            Q = W_block.clone()
            Err_block = torch.zeros_like(W_block)
            Losses_block = torch.zeros_like(W_block)
            H_inv_block = H_inv[col_start:col_end, col_start:col_end]

            for j in range(count):
                w = W_block[:, j]
                d = H_inv_block[j, j]

                # Determine which weights to prune in this column
                # Sort by |w| / d (importance score)
                scores = w.abs() / d
                n_prune = int(sparsity * self.rows)
                if n_prune > 0:
                    _, prune_idx = torch.topk(scores, n_prune, largest=False)
                    mask_col = torch.zeros(self.rows, dtype=torch.bool, device=self.dev)
                    mask_col[prune_idx] = True
                else:
                    mask_col = torch.zeros(self.rows, dtype=torch.bool, device=self.dev)

                q = w.clone()
                q[mask_col] = 0

                Q[:, j] = q
                Losses_block[:, j] = ((w - q) ** 2) / (d ** 2)

                err = (w - q) / d
                W_block[:, j:] -= err.unsqueeze(1) * H_inv_block[j, j:].unsqueeze(0)
                Err_block[:, j] = err

            W[:, col_start:col_end] = Q
            Losses += Losses_block.sum(dim=1)

            # Propagate error to remaining columns
            if col_end < self.columns:
                W[:, col_end:] -= (
                    Err_block @ H_inv[col_start:col_end, col_end:]
                )

        # Apply pruned weights
        self.layer.weight.data = W.to(self.layer.weight.dtype)

        n_zeros = (self.layer.weight.data == 0).sum().item()
        n_total = self.layer.weight.data.numel()

        return {
            "actual_sparsity": n_zeros / n_total if n_total > 0 else 0.0,
            "reconstruction_loss": Losses.sum().item(),
        }

    def free(self):
        """Free Hessian memory."""
        del self.H
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ── Calibration + pruning pipeline ──────────────────────────────────────────

class HessianCollector:
    """Collects Hessian data for all Linear layers via forward hooks."""

    def __init__(self, model: nn.Module):
        self.layers: dict[str, SparseGPTLayer] = {}
        self.hooks = []
        self._register(model)

    def _register(self, model: nn.Module):
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if _is_vision_module(name):
                continue
            spg = SparseGPTLayer(module)
            self.layers[name] = spg
            hook = module.register_forward_hook(self._make_hook(name))
            self.hooks.append(hook)

    def _make_hook(self, name: str):
        def hook_fn(module, inp, out):
            x = inp[0]
            if x.dim() == 2:
                x = x.unsqueeze(0)
            self.layers[name].add_batch(x)
        return hook_fn

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def apply_sparsegpt_pruning(model: nn.Module, collector: HessianCollector,
                             sparsity: float, blocksize: int = 128) -> dict:
    """Apply SparseGPT pruning to all collected layers."""
    total_zeros = 0
    total_weights = 0
    total_loss = 0.0
    pruned_layers = 0

    for name, spg in collector.layers.items():
        if spg.nsamples == 0:
            logger.debug(f"  Skipping {name}: no calibration data")
            continue

        stats = spg.prune(sparsity, blocksize=blocksize)
        n = spg.layer.weight.numel()
        n_z = int(stats["actual_sparsity"] * n)
        total_zeros += n_z
        total_weights += n
        total_loss += stats["reconstruction_loss"]
        pruned_layers += 1
        spg.free()

    actual_sparsity = total_zeros / total_weights if total_weights > 0 else 0.0
    logger.info(
        f"SparseGPT pruned {pruned_layers} layers | "
        f"actual sparsity = {actual_sparsity:.4f} "
        f"({total_zeros:,}/{total_weights:,} zeros) | "
        f"reconstruction loss = {total_loss:.2f}"
    )
    return {
        "target_sparsity": sparsity,
        "actual_sparsity": round(actual_sparsity, 4),
        "pruned_layers": pruned_layers,
        "total_zeros": total_zeros,
        "total_weights": total_weights,
        "reconstruction_loss": round(total_loss, 4),
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
    parser.add_argument("--n_calib", type=int, default=128,
                        help="Number of calibration samples for Hessian")
    parser.add_argument("--blocksize", type=int, default=128,
                        help="Block size for SparseGPT (default: 128)")
    parser.add_argument("--vqav2_n", type=int, default=1000)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_id = args.model_id
    family = detect_family(model_id)
    safe_name = model_id.replace("/", "__")
    sparsity_tag = f"sp{int(args.sparsity * 100)}"
    tag = f"{safe_name}__sparsegpt_{sparsity_tag}"
    out_path = RESULTS_DIR / f"{tag}.json"

    if out_path.exists() and not args.force:
        logger.info(f"Result already exists at {out_path}. Skipping.")
        return

    # Load model in fp16
    logger.info(f"Loading {model_id} (fp16) for SparseGPT pruning...")
    model, processor, meta = load_model(model_id, quant="fp16")
    device = str(next(model.parameters()).device)
    num_params = sum(p.numel() for p in model.parameters())

    # Calibration: collect Hessians
    logger.info(f"Calibration pass: collecting Hessians ({args.n_calib} samples)...")
    calib_samples = load_vqav2(n_samples=args.n_calib)

    collector = HessianCollector(model)
    skipped = 0
    for sample in calib_samples:
        try:
            with torch.no_grad():
                _ = run_inference(model, processor, sample, family, device)
        except Exception:
            skipped += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    if skipped:
        logger.warning(f"  Calibration: skipped {skipped}/{args.n_calib} samples")

    collector.remove_hooks()
    logger.info(f"  Collected Hessians for {len(collector.layers)} layers")

    # Apply SparseGPT pruning
    logger.info(f"Applying SparseGPT pruning at sparsity={args.sparsity}...")
    prune_stats = apply_sparsegpt_pruning(model, collector, args.sparsity,
                                           blocksize=args.blocksize)
    actual_sparsity = measure_sparsity(model)
    logger.info(f"Post-pruning sparsity check: {actual_sparsity:.4f}")

    results = {
        "model_id": model_id,
        "family": family,
        "method": "sparsegpt",
        "target_sparsity": args.sparsity,
        "actual_sparsity": prune_stats["actual_sparsity"],
        "pruned_layers": prune_stats["pruned_layers"],
        "reconstruction_loss": prune_stats["reconstruction_loss"],
        "n_calib_samples": args.n_calib,
        "num_params_M": round(num_params / 1e6, 1),
        "gpu_mem_load_mb": meta.gpu_mem_delta_mb,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmarks": {},
    }

    # Evaluate
    if not args.skip_eval:
        eval_samples = load_vqav2(n_samples=args.vqav2_n)
        results["benchmarks"]["vqav2"] = evaluate_dataset(
            model, processor, eval_samples, family, device,
            "VQAv2", _vqa_accuracy,
        )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"SparseGPT results saved to {out_path}")

    unload_model(model)


if __name__ == "__main__":
    main()
