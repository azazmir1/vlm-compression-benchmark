"""
compression/svd_llm/run_svd_llm.py
====================================
SVD-LLM: Truncation-aware SVD for LLM/VLM compression.

Based on: "SVD-LLM: Truncation-aware Singular Value Decomposition for LLM
           Compression" (Wang et al., ICLR 2025)

Key idea: Decompose weight matrices W ≈ U @ S @ V^T, then truncate to rank r.
Only compresses MLP layers (gate/up/down projections) — attention layers are
too sensitive. Uses energy-based rank selection to preserve model quality.

--energy controls quality: 0.99 = keep 99% of singular value energy (gentle),
0.90 = keep 90% (aggressive, more compression but lower accuracy).

Usage:
  python compression/svd_llm/run_svd_llm.py \
      --model_id HuggingFaceTB/SmolVLM-256M-Instruct --energy 0.95
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

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "svd_llm"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Module filtering ─────────────────────────────────────────────────────────

VISION_MODULE_KEYWORDS = {
    "vision_model", "visual_model", "image_encoder", "vision_encoder",
    "patch_embed", "visual_projection", "img_projection",
    "vision_tower", "vit", "davit", "siglip", "fastvit",
}

# Only compress MLP/FFN layers — attention projections are too sensitive
MLP_KEYWORDS = {
    "mlp", "feed_forward", "ffn",
    "gate_proj", "up_proj", "down_proj",       # LLaMA-style
    "fc1", "fc2",                               # GPT-style
    "dense_h_to_4h", "dense_4h_to_h",          # InternLM-style
    "w1", "w2", "w3",                           # compact naming
}


def _is_vision_module(name: str) -> bool:
    return any(kw in name.lower() for kw in VISION_MODULE_KEYWORDS)


def _is_mlp_layer(name: str) -> bool:
    """Check if this is an MLP/FFN layer (safe to compress)."""
    name_lower = name.lower()
    return any(kw in name_lower for kw in MLP_KEYWORDS)


# ── SVD-LLM decomposition ───────────────────────────────────────────────────

class SVDLinear(nn.Module):
    """Low-rank Linear: W ≈ A @ B where A=(out,r), B=(r,in)."""

    def __init__(self, A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor = None):
        super().__init__()
        self.A = nn.Parameter(A, requires_grad=False)
        self.B = nn.Parameter(B, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.bias = None

    def forward(self, x):
        out = (x @ self.B.t()) @ self.A.t()
        if self.bias is not None:
            out = out + self.bias
        return out


def compute_rank_for_energy(singular_values: torch.Tensor, energy_target: float,
                            min_rank: int = 32) -> int:
    """
    Find minimum rank that retains `energy_target` fraction of total energy.
    Energy = sum of squared singular values (= Frobenius norm squared).
    """
    energy = singular_values.float().pow(2)
    total = energy.sum()
    cumulative = energy.cumsum(0) / total

    # First index where cumulative energy >= target
    rank = (cumulative < energy_target).sum().item() + 1
    rank = max(rank, min_rank)
    rank = min(rank, len(singular_values))
    return rank


def apply_svd_compression(model: nn.Module, energy_target: float = 0.95,
                          min_rank: int = 32) -> dict:
    """
    Apply truncated SVD to MLP layers in the LLM backbone.

    Only decomposes MLP/FFN layers (gate_proj, up_proj, down_proj, fc1, fc2).
    Attention projections (q/k/v/o_proj) are left untouched.

    energy_target: fraction of singular value energy to preserve (0.95 = 95%).
    """
    compressed_layers = 0
    original_params = 0
    compressed_params = 0
    skipped_small = 0
    skipped_non_mlp = 0
    energy_retained_sum = 0.0

    replacements = []
    module_dict = dict(model.named_modules())

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if _is_vision_module(name):
            continue
        if not _is_mlp_layer(name):
            skipped_non_mlp += 1
            continue

        W = module.weight.data.float()
        out_features, in_features = W.shape
        orig_count = out_features * in_features

        # Skip small layers where SVD overhead isn't worth it
        if min(out_features, in_features) < min_rank * 2:
            skipped_small += 1
            original_params += orig_count
            compressed_params += orig_count
            continue

        # Full SVD
        try:
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        except RuntimeError:
            logger.debug(f"  SVD failed for {name}, skipping")
            original_params += orig_count
            compressed_params += orig_count
            continue

        # Energy-based rank selection
        rank = compute_rank_for_energy(S, energy_target, min_rank)
        full_rank = min(out_features, in_features)

        # Check if decomposition actually saves parameters
        new_count = out_features * rank + rank * in_features
        if new_count >= orig_count * 0.95:
            original_params += orig_count
            compressed_params += orig_count
            continue

        # Track energy actually retained
        energy_kept = S[:rank].pow(2).sum() / S.pow(2).sum()
        energy_retained_sum += energy_kept.item()

        # Truncate
        U_trunc = U[:, :rank]
        S_trunc = S[:rank]
        Vt_trunc = Vt[:rank, :]

        A = (U_trunc * S_trunc.unsqueeze(0)).to(module.weight.dtype)
        B = Vt_trunc.to(module.weight.dtype)

        original_params += orig_count
        compressed_params += new_count

        bias = module.bias.data.clone() if module.bias is not None else None
        new_module = SVDLinear(A, B, bias)

        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent = module_dict.get(parts[0])
            attr_name = parts[1]
        else:
            parent = model
            attr_name = name

        if parent is not None:
            replacements.append((parent, attr_name, new_module))
            compressed_layers += 1
            logger.debug(f"  {name}: {full_rank} -> {rank} (energy={energy_kept:.4f})")

    # Apply replacements
    for parent, attr_name, new_module in replacements:
        setattr(parent, attr_name, new_module)

    compression_ratio = original_params / compressed_params if compressed_params > 0 else 1.0
    param_reduction = 1 - (compressed_params / original_params) if original_params > 0 else 0.0
    avg_energy = energy_retained_sum / compressed_layers if compressed_layers > 0 else 1.0

    logger.info(
        f"SVD compressed {compressed_layers} MLP layers | "
        f"param reduction = {param_reduction:.4f} | "
        f"compression = {compression_ratio:.2f}x | "
        f"avg energy retained = {avg_energy:.4f} | "
        f"skipped {skipped_non_mlp} attn + {skipped_small} small"
    )
    return {
        "energy_target": energy_target,
        "compressed_layers": compressed_layers,
        "original_params": original_params,
        "compressed_params": compressed_params,
        "compression_ratio": round(compression_ratio, 2),
        "param_reduction": round(param_reduction, 4),
        "avg_energy_retained": round(avg_energy, 4),
        "skipped_non_mlp": skipped_non_mlp,
        "skipped_small_layers": skipped_small,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SVD-LLM compression pipeline")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--energy", type=float, default=0.95,
                        help="Fraction of singular value energy to preserve (default 0.95)")
    parser.add_argument("--min_rank", type=int, default=32)
    parser.add_argument("--vqav2_n", type=int, default=1000)
    parser.add_argument("--skip_vqav2", action="store_true")
    parser.add_argument("--skip_textvqa", action="store_true")
    parser.add_argument("--skip_pope", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_id = args.model_id
    safe_name = model_id.replace("/", "__")
    energy_tag = int(args.energy * 100)
    tag = f"{safe_name}__svd_e{energy_tag}"
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
    logger.info(f"Applying SVD-LLM with energy={args.energy}...")
    svd_stats = apply_svd_compression(model, args.energy, args.min_rank)

    num_params_after = sum(p.numel() for p in model.parameters())

    # Measure actual GPU memory after SVD replacement
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(torch.cuda.current_device())
        gpu_mem_after = (total - free) / 1024**2
    else:
        gpu_mem_after = 0.0

    results = {
        "model_id": model_id,
        "family": family,
        "method": "svd_llm",
        "quant": "fp16",
        **svd_stats,
        "num_params_before_M": round(num_params_before / 1e6, 1),
        "num_params_after_M": round(num_params_after / 1e6, 1),
        "gpu_mem_load_mb": meta.gpu_mem_delta_mb,
        "gpu_mem_after_svd_mb": round(gpu_mem_after, 1),
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
