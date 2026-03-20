"""
compression/awp/run_awp.py
===========================
AWP: Activation-Aware Weight Pruning + Quantization.

Based on: "Activation-Aware Weight Pruning and Quantization" (MERL, ICML 2025)

Key idea: Combine Wanda-style pruning with INT4 quantization for maximum
compression. Pruning-first approach: apply Wanda pruning to create sparsity,
then quantize remaining weights to INT4 using BitsAndBytes NF4.

This gives combined compression: 2:4 sparsity + INT4 = ~8x compression.

Usage:
  python compression/awp/run_awp.py \
      --model_id HuggingFaceTB/SmolVLM-256M-Instruct --sparsity 0.50
"""

import argparse
import json
import logging
import sys
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

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "awp"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Vision module detection ──────────────────────────────────────────────────

VISION_MODULE_KEYWORDS = {
    "vision_model", "visual_model", "image_encoder", "vision_encoder",
    "patch_embed", "visual_projection", "img_projection",
    "vision_tower", "vit", "davit", "siglip", "fastvit",
}


def _is_vision_module(name: str) -> bool:
    return any(kw in name.lower() for kw in VISION_MODULE_KEYWORDS)


# ── Wanda pruning (step 1) ──────────────────────────────────────────────────

class ActivationCollector:
    """Collects squared L2 norms of inputs to Linear layers."""

    def __init__(self, model: nn.Module):
        self.norms: dict[str, torch.Tensor] = {}
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
                x = inp[0]
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                sq_sum = x.float().pow(2).sum(dim=(0, 1)).cpu()
                n_tokens = x.shape[0] * x.shape[1]
                if name in self.norms:
                    self.norms[name] += sq_sum
                    self.counts[name] += n_tokens
                else:
                    self.norms[name] = sq_sum
                    self.counts[name] = n_tokens
        return hook_fn

    def get_input_norms(self) -> dict[str, torch.Tensor]:
        result = {}
        for name in self.norms:
            result[name] = (self.norms[name] / self.counts[name]).sqrt()
        return result

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def apply_wanda_pruning(model: nn.Module, input_norms: dict[str, torch.Tensor],
                        sparsity: float) -> dict:
    """Step 1: Wanda pruning — zero out weights with smallest |W| * ||X||."""
    pruned_params = 0
    total_params = 0
    pruned_layers = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name not in input_norms:
            continue

        W = module.weight.data
        X_norm = input_norms[name].to(W.device)

        if X_norm.shape[0] != W.shape[1]:
            continue

        importance = W.abs() * X_norm.unsqueeze(0)
        n_prune = int(W.shape[1] * sparsity)
        if n_prune == 0:
            continue

        _, idx = torch.topk(importance, n_prune, dim=1, largest=False)
        mask = torch.ones_like(W, dtype=torch.bool)
        mask.scatter_(1, idx, False)
        W[~mask] = 0.0

        n_zero = (W == 0).sum().item()
        n_total = W.numel()
        pruned_params += n_zero
        total_params += n_total
        pruned_layers += 1

    actual_sparsity = pruned_params / total_params if total_params > 0 else 0.0
    logger.info(f"Wanda pruning: {pruned_layers} layers, sparsity={actual_sparsity:.4f}")
    return {
        "wanda_sparsity": round(actual_sparsity, 4),
        "wanda_pruned_layers": pruned_layers,
    }


# ── Simulated INT4 quantization (step 2) ─────────────────────────────────────

def apply_simulated_int4_quantization(model: nn.Module) -> dict:
    """
    Step 2: Simulate INT4 quantization on non-zero weights.

    Uses per-channel absmax quantization to INT4 range [-8, 7].
    Simulated = quantize then dequantize in fp16 (shows accuracy impact
    without requiring specialized INT4 kernels).
    """
    quantized_layers = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if _is_vision_module(name):
            continue

        W = module.weight.data.float()
        # Only quantize non-zero weights (pruned weights stay zero)
        nonzero_mask = W != 0

        if nonzero_mask.sum() == 0:
            continue

        # Per-channel (per-row) absmax scaling
        abs_max = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = abs_max / 7.0  # INT4 range: [-8, 7]

        # Quantize to INT4 range and dequantize
        W_scaled = W / scale
        W_quant = W_scaled.round().clamp(-8, 7)
        W_deq = W_quant * scale

        # Keep pruned weights at zero
        W_deq[~nonzero_mask] = 0.0

        module.weight.data = W_deq.to(module.weight.dtype)
        quantized_layers += 1

    logger.info(f"Simulated INT4 quantization on {quantized_layers} layers")
    return {"int4_quantized_layers": quantized_layers}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AWP: pruning + quantization")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--sparsity", type=float, default=0.50,
                        help="Target sparsity for Wanda step (default 0.50)")
    parser.add_argument("--n_calib", type=int, default=128)
    parser.add_argument("--vqav2_n", type=int, default=1000)
    parser.add_argument("--skip_vqav2", action="store_true")
    parser.add_argument("--skip_textvqa", action="store_true")
    parser.add_argument("--skip_pope", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_id = args.model_id
    safe_name = model_id.replace("/", "__")
    sp_tag = f"sp{int(args.sparsity * 100)}"
    tag = f"{safe_name}__awp_{sp_tag}_int4"
    out_path = RESULTS_DIR / f"{tag}.json"

    if out_path.exists() and not args.force:
        logger.info(f"Result exists at {out_path}. Skipping.")
        return

    # ── Load model (fp16) ─────────────────────────────────────────────────
    logger.info(f"Loading {model_id} (fp16) for AWP...")
    model, processor, meta = load_model(model_id, quant="fp16")
    family = meta.family
    device = str(next(model.parameters()).device)
    num_params = sum(p.numel() for p in model.parameters())

    # ── Step 1: Wanda calibration + pruning ───────────────────────────────
    logger.info(f"Calibration pass ({args.n_calib} samples)...")
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
        logger.warning(f"  Calibration: skipped {skipped}/{args.n_calib}")

    input_norms = collector.get_input_norms()
    collector.remove_hooks()

    logger.info(f"Step 1: Wanda pruning at sparsity={args.sparsity}...")
    prune_stats = apply_wanda_pruning(model, input_norms, args.sparsity)

    # ── Step 2: Simulated INT4 quantization ───────────────────────────────
    logger.info("Step 2: Simulated INT4 quantization on remaining weights...")
    quant_stats = apply_simulated_int4_quantization(model)

    # Measure final sparsity
    zeros, total = 0, 0
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            zeros += (module.weight == 0).sum().item()
            total += module.weight.numel()
    final_sparsity = zeros / total if total > 0 else 0.0

    results = {
        "model_id": model_id,
        "family": family,
        "method": "awp",
        "quant": "fp16_simulated_int4",
        "target_sparsity": args.sparsity,
        "final_sparsity": round(final_sparsity, 4),
        **prune_stats,
        **quant_stats,
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
    logger.info(f"AWP results saved to {out_path}")

    unload_model(model)


if __name__ == "__main__":
    main()
