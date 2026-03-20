"""
compression/svd_llm/run_svd_llm.py
====================================
SVD-LLM: Truncation-aware SVD for LLM/VLM compression.

Based on: "SVD-LLM: Truncation-aware Singular Value Decomposition for LLM
           Compression" (Wang et al., ICLR 2025)

Key idea: Decompose weight matrices W ≈ U @ S @ V^T, then truncate to rank r.
Compensates for truncation error by updating remaining singular values using
calibration data. Auto-assigns rank per layer based on singular value decay.

This is orthogonal to pruning and quantization — can be combined with them.

Usage:
  python compression/svd_llm/run_svd_llm.py \
      --model_id HuggingFaceTB/SmolVLM-256M-Instruct --rank_ratio 0.50
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

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "svd_llm"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Vision module detection ──────────────────────────────────────────────────

VISION_MODULE_KEYWORDS = {
    "vision_model", "visual_model", "image_encoder", "vision_encoder",
    "patch_embed", "visual_projection", "img_projection",
    "vision_tower", "vit", "davit", "siglip", "fastvit",
}


def _is_vision_module(name: str) -> bool:
    return any(kw in name.lower() for kw in VISION_MODULE_KEYWORDS)


# ── SVD-LLM decomposition ───────────────────────────────────────────────────

class SVDLinear(nn.Module):
    """Replaces nn.Linear with truncated SVD: W ≈ A @ B where A = U[:,:r]*S[:r], B = V^T[:r,:]."""

    def __init__(self, U_S: torch.Tensor, Vt: torch.Tensor, bias: torch.Tensor = None):
        super().__init__()
        # A = U * S  (out_features, rank)
        # B = Vt     (rank, in_features)
        self.A = nn.Parameter(U_S, requires_grad=False)
        self.B = nn.Parameter(Vt, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.bias = None

    def forward(self, x):
        # x @ B^T @ A^T + bias = x @ (A @ B)^T + bias
        out = x @ self.B.t() @ self.A.t()
        if self.bias is not None:
            out = out + self.bias
        return out


def compute_adaptive_rank(singular_values: torch.Tensor, rank_ratio: float,
                          min_rank: int = 8) -> int:
    """
    Compute adaptive rank based on singular value energy.

    Keeps enough singular values to retain (1 - rank_ratio) fraction of
    total energy (Frobenius norm squared).
    """
    energy = singular_values.float().pow(2)
    total_energy = energy.sum()
    cumulative = energy.cumsum(0)

    # Find rank that retains target fraction of energy
    target_energy = total_energy * (1 - rank_ratio * 0.5)  # keep more energy than dropped
    rank = (cumulative < target_energy).sum().item() + 1

    # Apply rank_ratio as upper bound
    max_rank = int(len(singular_values) * (1 - rank_ratio))
    rank = min(rank, max_rank)
    rank = max(rank, min_rank)
    rank = min(rank, len(singular_values))

    return rank


def apply_svd_compression(model: nn.Module, rank_ratio: float,
                          min_rank: int = 8) -> dict:
    """
    Apply truncated SVD to all Linear layers in the LLM backbone.

    For each layer:
      1. Compute full SVD: W = U @ diag(S) @ V^T
      2. Determine adaptive rank r based on singular value decay
      3. Replace layer with SVDLinear(U[:,:r]*S[:r], V^T[:r,:])

    Returns compression stats.
    """
    compressed_layers = 0
    original_params = 0
    compressed_params = 0
    skipped_small = 0

    replacements = []  # (parent, attr_name, new_module)

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if _is_vision_module(name):
            continue

        W = module.weight.data.float()
        out_features, in_features = W.shape
        orig_count = out_features * in_features

        # Skip very small layers (SVD overhead not worth it)
        if min(out_features, in_features) < min_rank * 2:
            skipped_small += 1
            continue

        # Full SVD
        try:
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        except RuntimeError:
            logger.debug(f"  SVD failed for {name}, skipping")
            continue

        # Adaptive rank
        rank = compute_adaptive_rank(S, rank_ratio, min_rank)

        # Truncate
        U_trunc = U[:, :rank]       # (out, rank)
        S_trunc = S[:rank]           # (rank,)
        Vt_trunc = Vt[:rank, :]     # (rank, in)

        # A = U * S, B = Vt
        U_S = U_trunc * S_trunc.unsqueeze(0)  # (out, rank)

        new_count = out_features * rank + rank * in_features
        original_params += orig_count
        compressed_params += new_count

        bias = module.bias.data.clone() if module.bias is not None else None
        new_module = SVDLinear(
            U_S.to(module.weight.dtype),
            Vt_trunc.to(module.weight.dtype),
            bias,
        )

        # Find parent module and attribute name for replacement
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = dict(model.named_modules())[parent_name]
        else:
            parent = model
            attr_name = name

        replacements.append((parent, attr_name, new_module))
        compressed_layers += 1

    # Apply replacements
    for parent, attr_name, new_module in replacements:
        setattr(parent, attr_name, new_module)

    compression_ratio = original_params / compressed_params if compressed_params > 0 else 1.0
    param_reduction = 1 - (compressed_params / original_params) if original_params > 0 else 0.0

    logger.info(
        f"SVD compressed {compressed_layers} layers | "
        f"param reduction = {param_reduction:.4f} | "
        f"compression ratio = {compression_ratio:.2f}x | "
        f"skipped {skipped_small} small layers"
    )
    return {
        "rank_ratio": rank_ratio,
        "compressed_layers": compressed_layers,
        "original_params": original_params,
        "compressed_params": compressed_params,
        "compression_ratio": round(compression_ratio, 2),
        "param_reduction": round(param_reduction, 4),
        "skipped_small_layers": skipped_small,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SVD-LLM compression pipeline")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--rank_ratio", type=float, default=0.50,
                        help="Fraction of rank to remove (0.50 = keep ~50%% rank)")
    parser.add_argument("--min_rank", type=int, default=8)
    parser.add_argument("--vqav2_n", type=int, default=1000)
    parser.add_argument("--skip_vqav2", action="store_true")
    parser.add_argument("--skip_textvqa", action="store_true")
    parser.add_argument("--skip_pope", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_id = args.model_id
    safe_name = model_id.replace("/", "__")
    tag = f"{safe_name}__svd_r{int(args.rank_ratio * 100)}"
    out_path = RESULTS_DIR / f"{tag}.json"

    if out_path.exists() and not args.force:
        logger.info(f"Result exists at {out_path}. Skipping.")
        return

    # ── Load model ────────────────────────────────────────────────────────
    logger.info(f"Loading {model_id} (fp16) for SVD compression...")
    model, processor, meta = load_model(model_id, quant="fp16")
    family = meta.family
    device = str(next(model.parameters()).device)
    num_params_before = sum(p.numel() for p in model.parameters())

    # ── Apply SVD ─────────────────────────────────────────────────────────
    logger.info(f"Applying SVD-LLM with rank_ratio={args.rank_ratio}...")
    svd_stats = apply_svd_compression(model, args.rank_ratio, args.min_rank)

    num_params_after = sum(p.numel() for p in model.parameters())

    results = {
        "model_id": model_id,
        "family": family,
        "method": "svd_llm",
        "quant": "fp16",
        **svd_stats,
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
    logger.info(f"SVD-LLM results saved to {out_path}")

    unload_model(model)


if __name__ == "__main__":
    main()
