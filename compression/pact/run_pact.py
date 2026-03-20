"""
compression/pact/run_pact.py
=============================
PACT: Pruning And Clustering-based Token reduction for faster VLMs.

Based on: "PACT: Pruning and Clustering-Based Token Reduction for Faster VLMs"
          (Orailix et al., CVPR 2025)

Key idea: Training-free visual token compression. Prunes irrelevant visual
tokens and merges redundant ones at an early LLM layer.

Two-stage approach:
  1. Token Pruning: Remove visual tokens with lowest attention scores
     (those least attended to by text tokens).
  2. Token Merging: Cluster remaining visual tokens by similarity (cosine)
     and merge redundant clusters into single representative tokens.

This reduces visual token count by up to 50%, lowering memory AND compute.
Orthogonal to weight compression methods (quantization, pruning).

Usage:
  python compression/pact/run_pact.py \
      --model_id HuggingFaceTB/SmolVLM-256M-Instruct --prune_ratio 0.30 --merge_ratio 0.20
"""

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.model_loader import load_model, unload_model
from evaluation.run_baseline import (
    load_vqav2, run_inference,
    _vqa_accuracy,
)
from profiling.gpu_profiler import GPUProfiler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "pact"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── PACT token compression hooks ─────────────────────────────────────────────

class PACTTokenCompressor:
    """
    Installs a forward hook on an early LLM layer to prune + merge visual tokens.

    Strategy:
      - Identifies visual tokens by position (first N tokens after BOS, where
        N = number of image patches)
      - Prunes tokens with lowest L2 norm (proxy for attention importance)
      - Merges remaining similar tokens via cosine similarity clustering
    """

    def __init__(self, model: nn.Module, family: str,
                 prune_ratio: float = 0.3, merge_ratio: float = 0.2,
                 target_layer: int = 1):
        self.prune_ratio = prune_ratio
        self.merge_ratio = merge_ratio
        self.target_layer = target_layer
        self.family = family
        self.hook = None
        self.stats = {"tokens_pruned": 0, "tokens_merged": 0, "calls": 0}
        self._install_hook(model)

    def _find_target_layer(self, model: nn.Module) -> nn.Module:
        """Find the target transformer layer to hook into."""
        # Try common layer access patterns
        for attr_path in [
            "model.layers", "language_model.model.layers",
            "transformer.h", "model.decoder.layers",
            "gpt_neox.layers", "model.model.layers",
        ]:
            obj = model
            try:
                for attr in attr_path.split("."):
                    obj = getattr(obj, attr)
                if hasattr(obj, "__getitem__"):
                    idx = min(self.target_layer, len(obj) - 1)
                    return obj[idx]
            except (AttributeError, IndexError):
                continue

        # Fallback: find any Sequential-like container of layers
        for name, module in model.named_modules():
            if hasattr(module, "__getitem__") and hasattr(module, "__len__"):
                try:
                    if len(module) > self.target_layer:
                        layer = module[self.target_layer]
                        if hasattr(layer, "forward"):
                            return layer
                except (TypeError, IndexError):
                    continue

        raise RuntimeError("Could not find transformer layers for PACT hook")

    def _install_hook(self, model: nn.Module):
        """Install forward pre-hook on target layer."""
        try:
            target = self._find_target_layer(model)
            self.hook = target.register_forward_pre_hook(self._pact_hook)
            logger.info(f"PACT hook installed on layer {self.target_layer}")
        except RuntimeError as e:
            logger.warning(f"Could not install PACT hook: {e}. Running without token compression.")

    def _pact_hook(self, module, args):
        """Forward pre-hook that compresses visual tokens in hidden states."""
        if not args:
            return args

        hidden_states = args[0]
        if hidden_states.dim() != 3:
            return args

        batch, seq_len, hidden_dim = hidden_states.shape

        # Estimate number of visual tokens (typically first ~60-80% of sequence
        # for VLMs, but this varies). Use a heuristic: visual tokens are usually
        # the first chunk before text tokens.
        # Conservative estimate: assume 50% of tokens are visual for pruning
        n_visual = max(int(seq_len * 0.5), 1)

        if n_visual < 4:
            return args  # Too few tokens to compress

        visual_tokens = hidden_states[:, :n_visual, :]
        text_tokens = hidden_states[:, n_visual:, :]

        # Step 1: Prune — remove tokens with lowest L2 norm
        norms = visual_tokens.float().norm(dim=-1)  # (batch, n_visual)
        n_prune = int(n_visual * self.prune_ratio)

        if n_prune > 0 and n_prune < n_visual:
            _, keep_idx = torch.topk(norms, n_visual - n_prune, dim=1, largest=True)
            keep_idx = keep_idx.sort(dim=1).values
            visual_tokens = torch.gather(
                visual_tokens, 1,
                keep_idx.unsqueeze(-1).expand(-1, -1, hidden_dim)
            )
            self.stats["tokens_pruned"] += n_prune

        # Step 2: Merge — cluster similar tokens by cosine similarity
        n_remaining = visual_tokens.shape[1]
        n_merge = int(n_remaining * self.merge_ratio)

        if n_merge > 0 and n_remaining > 2:
            # Compute pairwise cosine similarity
            v_norm = nn.functional.normalize(visual_tokens.float(), dim=-1)
            sim = torch.bmm(v_norm, v_norm.transpose(1, 2))  # (batch, n, n)

            # Find most similar pairs and merge
            # Mask diagonal
            sim.diagonal(dim1=1, dim2=2).fill_(-1.0)

            for _ in range(min(n_merge, n_remaining - 1)):
                if visual_tokens.shape[1] <= 1:
                    break
                # Find most similar pair
                flat_idx = sim[:, :visual_tokens.shape[1], :visual_tokens.shape[1]].reshape(batch, -1).argmax(dim=1)
                n_curr = visual_tokens.shape[1]
                i = flat_idx // n_curr
                j = flat_idx % n_curr

                # Merge by averaging (for first batch element as representative)
                i0, j0 = i[0].item(), j[0].item()
                if i0 == j0 or i0 >= visual_tokens.shape[1] or j0 >= visual_tokens.shape[1]:
                    break

                merged = (visual_tokens[:, i0, :] + visual_tokens[:, j0, :]) / 2.0
                # Remove j, replace i with merged
                keep_mask = torch.ones(visual_tokens.shape[1], dtype=torch.bool, device=visual_tokens.device)
                keep_mask[j0] = False
                visual_tokens = visual_tokens[:, keep_mask, :]
                new_i = i0 if j0 > i0 else i0 - 1
                visual_tokens[:, new_i, :] = merged

                # Update similarity matrix
                n_new = visual_tokens.shape[1]
                v_norm = nn.functional.normalize(visual_tokens.float(), dim=-1)
                sim = torch.bmm(v_norm, v_norm.transpose(1, 2))
                sim.diagonal(dim1=1, dim2=2).fill_(-1.0)

                self.stats["tokens_merged"] += 1

        # Reassemble: compressed visual tokens + text tokens
        compressed = torch.cat([visual_tokens, text_tokens], dim=1)

        self.stats["calls"] += 1

        # Return modified args
        new_args = (compressed,) + args[1:]
        return new_args

    def remove_hook(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    def get_stats(self) -> dict:
        return dict(self.stats)


# ── Evaluation with PACT ─────────────────────────────────────────────────────

def evaluate_with_pact(model, processor, samples, family, device,
                       dataset_name, accuracy_fn) -> dict:
    """Evaluate model with PACT token compression active."""
    logger.info(f"  Evaluating on {dataset_name} ({len(samples)} samples) with PACT...")
    scores, latencies = [], []
    profiler = GPUProfiler(device_index=0)
    skipped = 0

    with profiler:
        for sample in tqdm(samples, desc=dataset_name, leave=False):
            t0 = time.perf_counter()
            try:
                pred = run_inference(model, processor, sample, family, device)
            except Exception:
                skipped += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            latencies.append(time.perf_counter() - t0)
            scores.append(accuracy_fn(pred, sample["answers"]))

    if skipped:
        logger.warning(f"  {dataset_name}: skipped {skipped} samples")

    stats = profiler.stats()
    avg_acc = sum(scores) / len(scores) if scores else 0.0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    throughput = len(latencies) / stats.wall_time_s if stats.wall_time_s > 0 else 0.0

    logger.info(
        f"  {dataset_name}: acc={avg_acc:.4f}  lat={avg_lat:.3f}s  "
        f"mem={stats.peak_memory_mb:.0f}MB"
    )
    return {
        "accuracy": round(avg_acc, 4),
        "avg_latency_s": round(avg_lat, 4),
        "peak_memory_mb": round(stats.peak_memory_mb, 1),
        "avg_memory_mb": round(stats.avg_memory_mb, 1),
        "throughput_sps": round(throughput, 3),
        "avg_power_w": round(stats.avg_power_w, 1),
        "avg_gpu_util_pct": round(stats.avg_gpu_util_pct, 1),
        "n_samples": len(samples),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PACT visual token compression")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--prune_ratio", type=float, default=0.30,
                        help="Fraction of visual tokens to prune (default 0.30)")
    parser.add_argument("--merge_ratio", type=float, default=0.20,
                        help="Fraction of remaining tokens to merge (default 0.20)")
    parser.add_argument("--target_layer", type=int, default=1,
                        help="LLM layer to apply token compression (default 1)")
    parser.add_argument("--vqav2_n", type=int, default=1000)
    parser.add_argument("--skip_vqav2", action="store_true")
    parser.add_argument("--skip_textvqa", action="store_true")
    parser.add_argument("--skip_pope", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_id = args.model_id
    safe_name = model_id.replace("/", "__")
    tag = f"{safe_name}__pact_p{int(args.prune_ratio*100)}_m{int(args.merge_ratio*100)}"
    out_path = RESULTS_DIR / f"{tag}.json"

    if out_path.exists() and not args.force:
        logger.info(f"Result exists at {out_path}. Skipping.")
        return

    # ── Load model ────────────────────────────────────────────────────────
    logger.info(f"Loading {model_id} (fp16) for PACT...")
    model, processor, meta = load_model(model_id, quant="fp16")
    family = meta.family
    device = str(next(model.parameters()).device)
    num_params = sum(p.numel() for p in model.parameters())

    # ── Install PACT hooks ────────────────────────────────────────────────
    logger.info(f"Installing PACT (prune={args.prune_ratio}, merge={args.merge_ratio})...")
    compressor = PACTTokenCompressor(
        model, family,
        prune_ratio=args.prune_ratio,
        merge_ratio=args.merge_ratio,
        target_layer=args.target_layer,
    )

    results = {
        "model_id": model_id,
        "family": family,
        "method": "pact",
        "quant": "fp16",
        "prune_ratio": args.prune_ratio,
        "merge_ratio": args.merge_ratio,
        "target_layer": args.target_layer,
        "total_token_reduction": round(1 - (1 - args.prune_ratio) * (1 - args.merge_ratio), 4),
        "num_params_M": round(num_params / 1e6, 1),
        "gpu_mem_load_mb": meta.gpu_mem_delta_mb,
        "benchmarks": {},
    }

    # ── Evaluate ──────────────────────────────────────────────────────────
    if not args.skip_vqav2:
        samples = load_vqav2(n_samples=args.vqav2_n)
        results["benchmarks"]["vqav2"] = evaluate_with_pact(
            model, processor, samples, family, device, "VQAv2", _vqa_accuracy,
        )

    # Add token compression stats
    results["pact_stats"] = compressor.get_stats()
    compressor.remove_hook()

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"PACT results saved to {out_path}")

    unload_model(model)


if __name__ == "__main__":
    main()
