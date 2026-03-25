"""
compression/palu/run_palu.py
=============================
PALU: KV-Cache Compression with Low-Rank Projection.

Based on: "PALU: Compressing KV-Cache with Low-Rank Projection"
          (Chang et al., ICLR 2025)

Key idea: Apply low-rank decomposition to K and V projection layers to
compress the KV cache at inference time. Uses truncated SVD on the
key/value projection weights, keeping only top-r singular values.

For each attention layer:
  W_k = U_k @ S_k @ V_k^T  →  W_k ≈ A_k @ B_k  (rank r)
  W_v = U_v @ S_v @ V_v^T  →  W_v ≈ A_v @ B_v  (rank r)

KV cache stores B_k @ x and B_v @ x (rank r instead of full dim),
then A_k and A_v are applied during attention. This compresses KV cache
to r/d fraction of original.

Usage:
  python compression/palu/run_palu.py \
      --model_id HuggingFaceTB/SmolVLM-256M-Instruct --rank_ratio 0.25
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
    load_vqav2,
    evaluate_dataset, _vqa_accuracy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Jetson-safe SVD wrapper ──────────────────────────────────────────────────
_LINALG_SAFE = None


def _safe_svd(tensor_cpu: torch.Tensor, full_matrices: bool = False):
    """SVD that works on Jetson by catching cuSOLVER dlopen failures."""
    global _LINALG_SAFE
    if _LINALG_SAFE is None:
        try:
            _test = torch.randn(2, 2)
            torch.linalg.svd(_test, full_matrices=False)
            _LINALG_SAFE = True
        except RuntimeError:
            _LINALG_SAFE = False
            logger.warning("torch.linalg.svd broken (cuSOLVER) — using torch.svd fallback")

    tensor_cpu = tensor_cpu.cpu().float()
    if _LINALG_SAFE:
        return torch.linalg.svd(tensor_cpu, full_matrices=full_matrices)
    else:
        U, S, V = torch.svd(tensor_cpu, some=not full_matrices)
        return U, S, V.t()

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "palu"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Vision module detection ──────────────────────────────────────────────────

VISION_MODULE_KEYWORDS = {
    "vision_model", "visual_model", "image_encoder", "vision_encoder",
    "patch_embed", "visual_projection", "img_projection",
    "vision_tower", "vit", "davit", "siglip", "fastvit",
}


def _is_vision_module(name: str) -> bool:
    return any(kw in name.lower() for kw in VISION_MODULE_KEYWORDS)


# ── PALU KV projection compression ──────────────────────────────────────────

# Common key/value projection layer name patterns across VLM families
KV_PROJ_PATTERNS = {
    "k_proj", "v_proj",           # LLaMA, Qwen, Gemma, InternLM
    "key_proj", "value_proj",     # some architectures
    "k_linear", "v_linear",       # rare
}


def _is_kv_proj(name: str) -> bool:
    """Check if module name corresponds to a K or V projection."""
    name_lower = name.lower().split(".")[-1]
    return any(p in name_lower for p in KV_PROJ_PATTERNS)


class PALULinear(nn.Module):
    """Low-rank approximation of a Linear layer: W ≈ A @ B."""

    def __init__(self, A: torch.Tensor, B: torch.Tensor,
                 bias: torch.Tensor = None):
        super().__init__()
        self.A = nn.Parameter(A, requires_grad=False)  # (out, rank)
        self.B = nn.Parameter(B, requires_grad=False)   # (rank, in)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.bias = None

    def forward(self, x):
        # Two-step: x @ B^T → (batch, seq, rank), then @ A^T → (batch, seq, out)
        dtype = x.dtype
        out = (x @ self.B.to(dtype).t()) @ self.A.to(dtype).t()
        if self.bias is not None:
            out = out + self.bias.to(dtype)
        return out


def apply_palu_compression(model: nn.Module, rank_ratio: float,
                           min_rank: int = 8) -> dict:
    """
    Apply PALU low-rank compression to K and V projection layers.

    rank_ratio: fraction of original rank to keep (e.g., 0.25 = 25% of dim).

    Returns compression stats.
    """
    compressed_layers = 0
    original_kv_params = 0
    compressed_kv_params = 0
    replacements = []

    # Build name->module mapping for parent lookup
    module_dict = dict(model.named_modules())

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if _is_vision_module(name):
            continue
        if not _is_kv_proj(name):
            continue

        W = module.weight.data.float()  # (out_features, in_features)
        out_features, in_features = W.shape
        orig_count = out_features * in_features

        # Target rank — enforce minimum to prevent destroying small KV dims
        full_rank = min(out_features, in_features)
        rank = max(int(full_rank * rank_ratio), min_rank)
        rank = min(rank, full_rank)

        # Safety: skip if compression would save < 10% params (not worth the quality loss)
        new_count_est = out_features * rank + rank * in_features
        if new_count_est >= orig_count * 0.9:
            logger.debug(f"  Skipping {name}: rank={rank} saves <10% params")
            continue

        # Safety: warn if rank is very small relative to dim
        if rank < 32 and full_rank >= 64:
            logger.warning(
                f"  {name}: rank={rank} is very small (full={full_rank}). "
                f"This may destroy model quality. Consider higher rank_ratio."
            )

        # SVD decomposition (Jetson-safe: falls back to torch.svd if cuSOLVER broken)
        try:
            U, S, Vt = _safe_svd(W.cpu(), full_matrices=False)
            U, S, Vt = U.to(W.device), S.to(W.device), Vt.to(W.device)
        except RuntimeError as e:
            logger.debug(f"  SVD failed for {name} ({e}), skipping")
            continue

        # Truncate
        U_trunc = U[:, :rank]       # (out, rank)
        S_trunc = S[:rank]           # (rank,)
        Vt_trunc = Vt[:rank, :]     # (rank, in)

        # A = U * S, B = Vt
        A = (U_trunc * S_trunc.unsqueeze(0)).to(module.weight.dtype)
        B = Vt_trunc.to(module.weight.dtype)

        new_count = out_features * rank + rank * in_features
        original_kv_params += orig_count
        compressed_kv_params += new_count

        bias = module.bias.data.clone() if module.bias is not None else None
        new_module = PALULinear(A, B, bias)

        # Find parent for replacement
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = module_dict.get(parent_name)
        else:
            parent = model
            attr_name = name

        if parent is not None:
            replacements.append((parent, attr_name, new_module))
            compressed_layers += 1

    # Apply replacements
    for parent, attr_name, new_module in replacements:
        setattr(parent, attr_name, new_module)

    kv_compression = original_kv_params / compressed_kv_params if compressed_kv_params > 0 else 1.0
    kv_param_reduction = 1 - (compressed_kv_params / original_kv_params) if original_kv_params > 0 else 0.0

    logger.info(
        f"PALU compressed {compressed_layers} KV projection layers | "
        f"KV param reduction = {kv_param_reduction:.4f} | "
        f"KV compression = {kv_compression:.2f}x"
    )
    return {
        "rank_ratio": rank_ratio,
        "compressed_kv_layers": compressed_layers,
        "original_kv_params": original_kv_params,
        "compressed_kv_params": compressed_kv_params,
        "kv_compression_ratio": round(kv_compression, 2),
        "kv_param_reduction": round(kv_param_reduction, 4),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PALU KV-cache compression")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--rank_ratio", type=float, default=0.25,
                        help="Fraction of rank to keep for KV projections (default 0.25)")
    parser.add_argument("--min_rank", type=int, default=8)
    parser.add_argument("--vqav2_n", type=int, default=1000)
    parser.add_argument("--skip_vqav2", action="store_true")
    parser.add_argument("--skip_textvqa", action="store_true")
    parser.add_argument("--skip_pope", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_id = args.model_id
    safe_name = model_id.replace("/", "__")
    tag = f"{safe_name}__palu_r{int(args.rank_ratio * 100)}"
    out_path = RESULTS_DIR / f"{tag}.json"

    if out_path.exists() and not args.force:
        logger.info(f"Result exists at {out_path}. Skipping.")
        return

    # ── Load model ────────────────────────────────────────────────────────
    logger.info(f"Loading {model_id} (fp16) for PALU KV compression...")
    model, processor, meta = load_model(model_id, quant="fp16")
    family = meta.family
    device = str(next(model.parameters()).device)
    num_params_before = sum(p.numel() for p in model.parameters())

    # ── Apply PALU ────────────────────────────────────────────────────────
    logger.info(f"Applying PALU with rank_ratio={args.rank_ratio}...")
    palu_stats = apply_palu_compression(model, args.rank_ratio, args.min_rank)

    num_params_after = sum(p.numel() for p in model.parameters())

    results = {
        "model_id": model_id,
        "family": family,
        "method": "palu",
        "quant": "fp16",
        **palu_stats,
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
    logger.info(f"PALU results saved to {out_path}")

    unload_model(model)


if __name__ == "__main__":
    main()
