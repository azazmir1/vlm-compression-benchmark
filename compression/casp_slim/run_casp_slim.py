"""
compression/casp_slim/run_casp_slim.py
=======================================
CASP + SLIM: Advanced combined compression methods.

CASP (Gholami et al., CVPR 2025):
  - VLM-specific: Q,K low-rank factorization + optimal bit allocation for
    mixed-precision quantization based on attention sparsity.

SLIM (NVIDIA, ICML 2025):
  - Unified one-shot: quantization + 2:4 sparsity + low-rank decomposition.
  - Combines all three compression axes for maximum compression.

This implementation provides two modes:
  --method casp: Attention-sparsity-aware mixed-precision quantization + KV low-rank
  --method slim: One-shot quantization + pruning + low-rank (triple compression)

Usage:
  python compression/casp_slim/run_casp_slim.py \
      --model_id HuggingFaceTB/SmolVLM-256M-Instruct --method casp
  python compression/casp_slim/run_casp_slim.py \
      --model_id HuggingFaceTB/SmolVLM-256M-Instruct --method slim
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

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "casp_slim"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Vision module detection ──────────────────────────────────────────────────

VISION_MODULE_KEYWORDS = {
    "vision_model", "visual_model", "image_encoder", "vision_encoder",
    "patch_embed", "visual_projection", "img_projection",
    "vision_tower", "vit", "davit", "siglip", "fastvit",
}


def _is_vision_module(name: str) -> bool:
    return any(kw in name.lower() for kw in VISION_MODULE_KEYWORDS)


KV_PROJ_PATTERNS = {"k_proj", "v_proj", "key_proj", "value_proj"}
QK_PROJ_PATTERNS = {"q_proj", "k_proj", "query_proj", "key_proj"}


def _is_kv_proj(name: str) -> bool:
    return any(p in name.lower().split(".")[-1] for p in KV_PROJ_PATTERNS)


def _is_qk_proj(name: str) -> bool:
    return any(p in name.lower().split(".")[-1] for p in QK_PROJ_PATTERNS)


# ── Activation collector for attention sparsity analysis ─────────────────────

class AttentionSparsityAnalyzer:
    """Collects activation statistics to estimate layer sensitivity."""

    def __init__(self, model: nn.Module):
        self.layer_norms: dict[str, float] = {}
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
            # Track output activation magnitude as sensitivity proxy
            with torch.no_grad():
                out_norm = out.float().norm().item()
                if name in self.layer_norms:
                    self.layer_norms[name] = max(self.layer_norms[name], out_norm)
                else:
                    self.layer_norms[name] = out_norm
        return hook_fn

    def get_sensitivity_scores(self) -> dict[str, float]:
        """Return normalized sensitivity scores (higher = more sensitive)."""
        if not self.layer_norms:
            return {}
        max_norm = max(self.layer_norms.values())
        if max_norm == 0:
            return {k: 1.0 for k in self.layer_norms}
        return {k: v / max_norm for k, v in self.layer_norms.items()}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ── CASP: attention-sparsity-aware compression ───────────────────────────────

def apply_casp(model: nn.Module, sensitivity: dict[str, float],
               target_bits: int = 4) -> dict:
    """
    CASP: Mixed-precision quantization based on attention sparsity.

    - Sensitive layers (high activation norm): keep at 8-bit
    - Less sensitive layers: quantize to 4-bit
    - Q,K projections: apply low-rank factorization

    Returns compression stats.
    """
    quantized_layers_4bit = 0
    quantized_layers_8bit = 0
    lowrank_layers = 0
    original_params = 0
    compressed_params = 0

    # Sensitivity threshold: top 25% layers get 8-bit, rest get 4-bit
    if sensitivity:
        sorted_scores = sorted(sensitivity.values(), reverse=True)
        threshold = sorted_scores[len(sorted_scores) // 4] if len(sorted_scores) > 4 else 0.5
    else:
        threshold = 0.5

    module_dict = dict(model.named_modules())
    replacements = []

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if _is_vision_module(name):
            continue

        W = module.weight.data.float()
        out_features, in_features = W.shape
        orig_count = W.numel()
        original_params += orig_count

        score = sensitivity.get(name, 0.5)

        # Q,K projections: apply low-rank factorization
        if _is_qk_proj(name) and min(out_features, in_features) >= 16:
            rank = max(min(out_features, in_features) // 4, 8)
            try:
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                U_trunc = U[:, :rank] * S[:rank].unsqueeze(0)
                Vt_trunc = Vt[:rank, :]

                # Replace with low-rank
                new_count = out_features * rank + rank * in_features
                compressed_params += new_count

                from compression.palu.run_palu import PALULinear
                bias = module.bias.data.clone() if module.bias is not None else None
                new_module = PALULinear(
                    U_trunc.to(module.weight.dtype),
                    Vt_trunc.to(module.weight.dtype),
                    bias,
                )

                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent = module_dict.get(parts[0])
                    attr_name = parts[1]
                else:
                    parent = model
                    attr_name = name

                if parent is not None:
                    replacements.append((parent, attr_name, new_module))
                    lowrank_layers += 1
                continue
            except RuntimeError:
                pass

        # Mixed-precision simulated quantization
        if score >= threshold:
            # Sensitive: 8-bit quantization
            abs_max = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
            scale = abs_max / 127.0
            W_q = (W / scale).round().clamp(-128, 127)
            W_deq = W_q * scale
            module.weight.data = W_deq.to(module.weight.dtype)
            compressed_params += orig_count  # same param count, lower precision
            quantized_layers_8bit += 1
        else:
            # Less sensitive: 4-bit quantization
            abs_max = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
            scale = abs_max / 7.0
            W_q = (W / scale).round().clamp(-8, 7)
            W_deq = W_q * scale
            module.weight.data = W_deq.to(module.weight.dtype)
            compressed_params += orig_count
            quantized_layers_4bit += 1

    # Apply low-rank replacements
    for parent, attr_name, new_module in replacements:
        setattr(parent, attr_name, new_module)

    logger.info(
        f"CASP: {quantized_layers_8bit} layers@8bit, {quantized_layers_4bit} layers@4bit, "
        f"{lowrank_layers} QK layers low-rank"
    )
    return {
        "quantized_8bit_layers": quantized_layers_8bit,
        "quantized_4bit_layers": quantized_layers_4bit,
        "lowrank_qk_layers": lowrank_layers,
        "sensitivity_threshold": round(threshold, 4),
    }


# ── SLIM: one-shot triple compression ───────────────────────────────────────

def apply_slim(model: nn.Module, sensitivity: dict[str, float],
               sparsity: float = 0.50, rank_ratio: float = 0.30) -> dict:
    """
    SLIM: Unified one-shot quantization + sparsity + low-rank.

    For each Linear layer (LLM backbone only):
      1. Low-rank: truncated SVD on large layers (rank_ratio controls truncation)
      2. Sparsity: magnitude pruning at target sparsity on remaining weights
      3. Quantization: simulated INT4 quantization on non-zero weights

    Returns compression stats.
    """
    total_layers = 0
    svd_layers = 0
    pruned_layers = 0
    quantized_layers = 0
    total_zeros = 0
    total_weights = 0

    module_dict = dict(model.named_modules())
    replacements = []

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if _is_vision_module(name):
            continue

        W = module.weight.data.float()
        out_features, in_features = W.shape
        total_layers += 1

        # Step 1: Low-rank for large layers
        if min(out_features, in_features) >= 64:
            try:
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                full_rank = min(out_features, in_features)
                rank = max(int(full_rank * (1 - rank_ratio)), 8)

                W_approx = (U[:, :rank] * S[:rank].unsqueeze(0)) @ Vt[:rank, :]
                W = W_approx
                svd_layers += 1
            except RuntimeError:
                pass

        # Step 2: Magnitude pruning (per-row)
        n_prune = int(in_features * sparsity)
        if n_prune > 0:
            _, idx = torch.topk(W.abs(), n_prune, dim=1, largest=False)
            mask = torch.ones_like(W, dtype=torch.bool)
            mask.scatter_(1, idx, False)
            W[~mask] = 0.0
            pruned_layers += 1

        # Step 3: Simulated INT4 quantization on non-zero weights
        nonzero_mask = W != 0
        if nonzero_mask.sum() > 0:
            abs_max = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
            scale = abs_max / 7.0
            W_q = (W / scale).round().clamp(-8, 7)
            W_deq = W_q * scale
            W_deq[~nonzero_mask] = 0.0
            W = W_deq
            quantized_layers += 1

        module.weight.data = W.to(module.weight.dtype)
        total_zeros += (module.weight.data == 0).sum().item()
        total_weights += module.weight.data.numel()

    actual_sparsity = total_zeros / total_weights if total_weights > 0 else 0.0

    logger.info(
        f"SLIM: {svd_layers} SVD + {pruned_layers} pruned + {quantized_layers} quantized | "
        f"sparsity = {actual_sparsity:.4f}"
    )
    return {
        "svd_layers": svd_layers,
        "pruned_layers": pruned_layers,
        "quantized_layers": quantized_layers,
        "total_layers": total_layers,
        "target_sparsity": sparsity,
        "actual_sparsity": round(actual_sparsity, 4),
        "rank_ratio": rank_ratio,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CASP/SLIM combined compression")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--method", required=True, choices=["casp", "slim"],
                        help="Compression method: casp or slim")
    parser.add_argument("--sparsity", type=float, default=0.50,
                        help="Target sparsity for SLIM (default 0.50)")
    parser.add_argument("--rank_ratio", type=float, default=0.30,
                        help="Rank reduction ratio for SLIM SVD (default 0.30)")
    parser.add_argument("--n_calib", type=int, default=32,
                        help="Calibration samples for sensitivity analysis")
    parser.add_argument("--vqav2_n", type=int, default=1000)
    parser.add_argument("--skip_vqav2", action="store_true")
    parser.add_argument("--skip_textvqa", action="store_true")
    parser.add_argument("--skip_pope", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_id = args.model_id
    safe_name = model_id.replace("/", "__")
    tag = f"{safe_name}__{args.method}"
    if args.method == "slim":
        tag += f"_sp{int(args.sparsity * 100)}_r{int(args.rank_ratio * 100)}"
    out_path = RESULTS_DIR / f"{tag}.json"

    if out_path.exists() and not args.force:
        logger.info(f"Result exists at {out_path}. Skipping.")
        return

    # ── Load model ────────────────────────────────────────────────────────
    logger.info(f"Loading {model_id} (fp16) for {args.method.upper()}...")
    model, processor, meta = load_model(model_id, quant="fp16")
    family = meta.family
    device = str(next(model.parameters()).device)
    num_params_before = sum(p.numel() for p in model.parameters())

    # ── Calibration: sensitivity analysis ─────────────────────────────────
    logger.info(f"Calibration: sensitivity analysis ({args.n_calib} samples)...")
    calib_samples = load_vqav2(n_samples=args.n_calib)

    analyzer = AttentionSparsityAnalyzer(model)
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

    sensitivity = analyzer.get_sensitivity_scores()
    analyzer.remove_hooks()
    logger.info(f"  Collected sensitivity for {len(sensitivity)} layers")

    # ── Apply compression ─────────────────────────────────────────────────
    if args.method == "casp":
        logger.info("Applying CASP: mixed-precision + QK low-rank...")
        method_stats = apply_casp(model, sensitivity)
    else:
        logger.info(f"Applying SLIM: SVD + pruning@{args.sparsity} + INT4...")
        method_stats = apply_slim(model, sensitivity, args.sparsity, args.rank_ratio)

    num_params_after = sum(p.numel() for p in model.parameters())

    results = {
        "model_id": model_id,
        "family": family,
        "method": args.method,
        "quant": "fp16_simulated",
        **method_stats,
        "n_calib_samples": args.n_calib,
        "num_params_before_M": round(num_params_before / 1e6, 1),
        "num_params_after_M": round(num_params_after / 1e6, 1),
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
    logger.info(f"{args.method.upper()} results saved to {out_path}")

    unload_model(model)


if __name__ == "__main__":
    main()
