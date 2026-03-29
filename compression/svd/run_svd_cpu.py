"""
compression/svd/run_svd_cpu.py
================================
SVD-LLM style low-rank decomposition for VLMs on CPU.

Based on: "SVD-LLM: Truncation-aware SVD for LLM Compression" (Wang et al., ICLR 2025)

Applies truncated SVD to every Linear layer in the LLM backbone (skips vision
encoder). Each weight W ≈ U_r * S_r * Vh_r where r = rank_ratio * min(out, in).

The lower the rank_ratio, the more aggressive the compression (more accuracy loss).

Usage:
  python compression/svd/run_svd_cpu.py \
      --model_id HuggingFaceTB/SmolVLM-256M-Instruct --rank_ratio 0.5
  python compression/svd/run_svd_cpu.py \
      --model_id HuggingFaceTB/SmolVLM-256M-Instruct --rank_ratio 0.3
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
    load_vqav2, evaluate_dataset, _vqa_accuracy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "svd_cpu"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

VISION_MODULE_KEYWORDS = {
    "vision_model", "visual_model", "image_encoder", "vision_encoder",
    "patch_embed", "visual_projection", "img_projection",
    "vision_tower", "vit", "davit", "siglip", "fastvit",
}


def _is_vision_module(name: str) -> bool:
    return any(kw in name.lower() for kw in VISION_MODULE_KEYWORDS)


def apply_svd_compression(model: nn.Module, rank_ratio: float) -> dict:
    """
    Replace each LLM-backbone Linear weight W with its truncated SVD approximation:
        W ≈ (U_r * S_r) @ Vh_r
    where r = floor(rank_ratio * min(out, in)).

    Returns compression stats.
    """
    total_params_before = 0
    total_params_after  = 0
    compressed_layers   = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if _is_vision_module(name):
            continue

        W = module.weight.data.float()  # (out, in)
        out_f, in_f = W.shape
        full_rank = min(out_f, in_f)
        rank = max(1, int(full_rank * rank_ratio))

        if rank >= full_rank:
            continue

        # Truncated SVD
        try:
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        except RuntimeError as e:
            logger.warning(f"SVD failed for {name}: {e} — skipping")
            continue

        # Keep top-rank singular values
        U_r  = U[:, :rank]          # (out, rank)
        S_r  = S[:rank]             # (rank,)
        Vh_r = Vh[:rank, :]         # (rank, in)

        W_approx = (U_r * S_r.unsqueeze(0)) @ Vh_r

        module.weight.data = W_approx.to(module.weight.dtype)

        total_params_before += out_f * in_f
        total_params_after  += rank * (out_f + in_f)
        compressed_layers   += 1

    compression_ratio = (
        total_params_before / total_params_after
        if total_params_after > 0 else 1.0
    )
    logger.info(
        f"SVD compressed {compressed_layers} layers | "
        f"rank_ratio={rank_ratio} | "
        f"param reduction: {total_params_before:,} → {total_params_after:,} "
        f"(ratio {compression_ratio:.2f}x)"
    )
    return {
        "rank_ratio":         rank_ratio,
        "compressed_layers":  compressed_layers,
        "params_before":      total_params_before,
        "params_after":       total_params_after,
        "compression_ratio":  round(compression_ratio, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="SVD-LLM style CPU compression")
    parser.add_argument("--model_id",     required=True)
    parser.add_argument("--rank_ratio",   type=float, default=0.5,
                        help="Fraction of singular values to keep (0.3=aggressive, 0.7=mild)")
    parser.add_argument("--vqav2_n",      type=int, default=200)
    parser.add_argument("--skip_vqav2",   action="store_true")
    parser.add_argument("--force",        action="store_true")
    args = parser.parse_args()

    safe_name = args.model_id.replace("/", "__")
    tag       = f"{safe_name}__svd_r{int(args.rank_ratio * 100)}"
    out_path  = RESULTS_DIR / f"{tag}.json"

    if out_path.exists() and not args.force:
        logger.info(f"Result already exists at {out_path}. Skipping.")
        return

    model, processor, meta = load_model(args.model_id)
    family = meta.family
    device = "cpu"
    num_params = sum(p.numel() for p in model.parameters())

    svd_stats = apply_svd_compression(model, args.rank_ratio)

    results = {
        "model_id":        args.model_id,
        "family":          family,
        "method":          "svd_llm",
        "rank_ratio":      args.rank_ratio,
        "device":          "cpu",
        "num_params_M":    round(num_params / 1e6, 1),
        "ram_load_mb":     meta.gpu_mem_delta_mb,
        "svd_stats":       svd_stats,
        "benchmarks":      {},
    }

    if not args.skip_vqav2:
        samples = load_vqav2(n_samples=args.vqav2_n)
        results["benchmarks"]["vqav2"] = evaluate_dataset(
            model, processor, samples, family, device, "VQAv2", _vqa_accuracy,
        )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"SVD results saved to {out_path}")

    unload_model(model)


if __name__ == "__main__":
    main()
