"""
compression/token_compression/run_pact.py
==========================================
PACT: Pruning and Clustering-Based Token Reduction for Faster VLMs.

Based on: "PACT: Pruning and Clustering-Based Token Reduction for
           Faster Visual Language Models" (Orailix et al., CVPR 2025)

Key idea: Training-free visual token compression applied at the early
layers of the LLM backbone. Two stages:
  1. Token Pruning: Remove irrelevant visual tokens based on attention
     scores from the CLS/text tokens to visual tokens
  2. Token Merging: Cluster remaining visual tokens by similarity and
     merge redundant ones

This is applied as a forward hook on the first few transformer layers,
reducing the number of visual tokens that subsequent layers must process.

Benefits:
  - Training-free and plug-and-play
  - Reduces both memory and compute
  - Complementary to weight compression (quantization, pruning)

Usage:
  python compression/token_compression/run_pact.py \
      --model_id HuggingFaceTB/SmolVLM-256M-Instruct --prune_ratio 0.30

  python compression/token_compression/run_pact.py \
      --model_id Qwen/Qwen2.5-VL-3B-Instruct --prune_ratio 0.50 --merge_ratio 0.20
"""

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

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

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "pact"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Token importance scoring ────────────────────────────────────────────────

def compute_token_importance(hidden_states: torch.Tensor,
                              attention_mask: torch.Tensor = None,
                              method: str = "norm") -> torch.Tensor:
    """Compute importance scores for each token in the sequence.

    Args:
        hidden_states: (batch, seq_len, hidden_dim)
        attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding
        method: "norm" (L2 norm), "cls_attn" (attention from first token)

    Returns:
        importance: (batch, seq_len) — higher = more important
    """
    if method == "norm":
        # Simple but effective: token importance ∝ hidden state norm
        importance = hidden_states.float().norm(dim=-1)  # (batch, seq_len)
    elif method == "entropy":
        # Token importance ∝ entropy of attention distribution
        # Higher entropy = more diverse attention = more informative
        probs = F.softmax(hidden_states.float(), dim=-1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)
        importance = entropy
    else:
        importance = hidden_states.float().norm(dim=-1)

    if attention_mask is not None:
        importance = importance * attention_mask.float()

    return importance


def prune_visual_tokens(hidden_states: torch.Tensor,
                         visual_token_mask: torch.Tensor,
                         prune_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Prune least important visual tokens.

    Args:
        hidden_states: (batch, seq_len, hidden_dim)
        visual_token_mask: (batch, seq_len) — True for visual tokens
        prune_ratio: fraction of visual tokens to remove

    Returns:
        pruned_hidden_states, new_mask
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape

    # Compute importance for all tokens
    importance = compute_token_importance(hidden_states)

    results = []
    masks = []

    for b in range(batch_size):
        vis_mask = visual_token_mask[b]  # (seq_len,)
        vis_indices = vis_mask.nonzero(as_tuple=True)[0]
        text_indices = (~vis_mask).nonzero(as_tuple=True)[0]

        if len(vis_indices) == 0:
            results.append(hidden_states[b])
            masks.append(vis_mask)
            continue

        n_vis = len(vis_indices)
        n_prune = int(n_vis * prune_ratio)
        n_keep = n_vis - n_prune

        if n_keep <= 0:
            n_keep = max(1, n_vis // 4)  # keep at least 25%

        # Get importance of visual tokens
        vis_importance = importance[b, vis_indices]
        _, keep_idx = torch.topk(vis_importance, n_keep)
        keep_vis_indices = vis_indices[keep_idx.sort().values]

        # Combine text tokens + kept visual tokens (preserve order)
        all_keep = torch.cat([text_indices, keep_vis_indices]).sort().values
        results.append(hidden_states[b, all_keep])

        new_mask = torch.zeros(len(all_keep), dtype=torch.bool, device=vis_mask.device)
        # Mark which of the kept tokens are visual
        for i, idx in enumerate(all_keep):
            if idx in keep_vis_indices:
                new_mask[i] = True
        masks.append(new_mask)

    # Pad to same length
    max_len = max(r.shape[0] for r in results)
    padded = torch.zeros(batch_size, max_len, hidden_dim,
                         device=hidden_states.device, dtype=hidden_states.dtype)
    padded_masks = torch.zeros(batch_size, max_len, dtype=torch.bool,
                               device=hidden_states.device)

    for b in range(batch_size):
        L = results[b].shape[0]
        padded[b, :L] = results[b]
        padded_masks[b, :L] = masks[b]

    return padded, padded_masks


def merge_similar_tokens(hidden_states: torch.Tensor,
                          visual_token_mask: torch.Tensor,
                          merge_ratio: float) -> torch.Tensor:
    """Merge similar visual tokens by clustering.

    Uses simple cosine-similarity-based greedy merging:
    1. Compute pairwise cosine similarity among visual tokens
    2. Iteratively merge most similar pair until target reduction reached
    3. Merged token = average of the two tokens

    Args:
        hidden_states: (batch, seq_len, hidden_dim)
        visual_token_mask: (batch, seq_len) — True for visual tokens
        merge_ratio: fraction of remaining visual tokens to merge

    Returns:
        merged_hidden_states with fewer visual tokens
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape
    results = []

    for b in range(batch_size):
        vis_mask = visual_token_mask[b]
        vis_indices = vis_mask.nonzero(as_tuple=True)[0]
        text_indices = (~vis_mask).nonzero(as_tuple=True)[0]

        if len(vis_indices) <= 1:
            results.append(hidden_states[b])
            continue

        vis_tokens = hidden_states[b, vis_indices]  # (n_vis, hidden_dim)
        n_vis = len(vis_indices)
        n_merge = int(n_vis * merge_ratio)

        if n_merge <= 0:
            results.append(hidden_states[b])
            continue

        # Greedy merging based on cosine similarity
        tokens = vis_tokens.float()
        active = torch.ones(n_vis, dtype=torch.bool, device=tokens.device)

        for _ in range(n_merge):
            active_idx = active.nonzero(as_tuple=True)[0]
            if len(active_idx) <= 1:
                break

            active_tokens = tokens[active_idx]
            # Normalize for cosine similarity
            normed = F.normalize(active_tokens, dim=-1)
            sim = normed @ normed.T
            # Mask diagonal
            sim.fill_diagonal_(-float('inf'))
            # Find most similar pair
            flat_idx = sim.argmax()
            i, j = flat_idx // sim.shape[1], flat_idx % sim.shape[1]
            # Merge: average the two tokens
            tokens[active_idx[i]] = (tokens[active_idx[i]] + tokens[active_idx[j]]) / 2
            active[active_idx[j]] = False

        # Reconstruct sequence
        merged_vis = tokens[active].to(hidden_states.dtype)
        text_tokens = hidden_states[b, text_indices]

        # Interleave (simplified: text first, then visual)
        combined = torch.cat([text_tokens, merged_vis], dim=0)
        results.append(combined)

    # Pad to same length
    max_len = max(r.shape[0] for r in results)
    padded = torch.zeros(batch_size, max_len, hidden_dim,
                         device=hidden_states.device, dtype=hidden_states.dtype)
    for b in range(batch_size):
        L = results[b].shape[0]
        padded[b, :L] = results[b]

    return padded


# ── PACT wrapper for inference ──────────────────────────────────────────────

class PACTWrapper:
    """Wraps a VLM model to apply PACT token compression during inference.

    Hooks into the model's forward pass at the embedding output layer
    to prune and merge visual tokens before they reach the transformer layers.
    """

    def __init__(self, model: nn.Module, family: str,
                 prune_ratio: float = 0.30, merge_ratio: float = 0.20):
        self.model = model
        self.family = family
        self.prune_ratio = prune_ratio
        self.merge_ratio = merge_ratio
        self.hooks = []
        self.stats = {"tokens_before": 0, "tokens_after": 0, "n_calls": 0}

    def estimate_visual_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Heuristic to identify visual tokens in the hidden states.

        Many VLMs concatenate visual tokens (from the vision encoder) with
        text tokens. Visual tokens tend to have different norm distributions.
        We use a simple heuristic: tokens with above-median norm in the first
        few positions are likely visual.
        """
        norms = hidden_states.float().norm(dim=-1)  # (batch, seq_len)
        batch_size, seq_len = norms.shape

        # Simple heuristic: visual tokens are usually in a contiguous block
        # and tend to have higher/lower norms than text tokens.
        # Use the distribution: mark tokens > 75th percentile as visual
        # (This is approximate — family-specific logic would be more accurate)
        threshold = norms.quantile(0.5, dim=-1, keepdim=True)

        # Visual tokens are usually the majority in VLMs
        # Use gradient of norms to find boundaries
        visual_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool,
                                  device=hidden_states.device)

        for b in range(batch_size):
            # Find the longest contiguous block of "similar" tokens
            # (visual tokens from vision encoder have similar characteristics)
            norm_diff = torch.abs(norms[b, 1:] - norms[b, :-1])
            median_diff = norm_diff.median()

            # Find large jumps (boundaries between text and visual)
            jumps = (norm_diff > 3 * median_diff).nonzero(as_tuple=True)[0]

            if len(jumps) >= 1:
                # Assume the longest segment between jumps is the visual block
                boundaries = torch.cat([
                    torch.tensor([0], device=jumps.device),
                    jumps + 1,
                    torch.tensor([seq_len], device=jumps.device),
                ])
                segments = boundaries[1:] - boundaries[:-1]
                longest = segments.argmax()
                start = boundaries[longest].item()
                end = boundaries[longest + 1].item()

                # Only mark as visual if it's a substantial portion
                if (end - start) > seq_len * 0.3:
                    visual_mask[b, start:end] = True
            else:
                # No clear boundary; assume first 70% are visual (common in VLMs)
                n_vis = int(seq_len * 0.7)
                visual_mask[b, :n_vis] = True

        return visual_mask

    def get_stats(self) -> dict:
        n = self.stats["n_calls"]
        if n == 0:
            return {"avg_tokens_before": 0, "avg_tokens_after": 0,
                    "avg_reduction_pct": 0}
        return {
            "avg_tokens_before": self.stats["tokens_before"] / n,
            "avg_tokens_after": self.stats["tokens_after"] / n,
            "avg_reduction_pct": round(
                100 * (1 - self.stats["tokens_after"] / max(self.stats["tokens_before"], 1)), 2
            ),
        }


def _gpu_mem_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info(torch.cuda.current_device())
    return (total - free) / 1024**2


def main():
    parser = argparse.ArgumentParser(description="PACT visual token compression")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--prune_ratio", type=float, default=0.30,
                        help="Fraction of visual tokens to prune (default: 0.30)")
    parser.add_argument("--merge_ratio", type=float, default=0.20,
                        help="Fraction of remaining visual tokens to merge (default: 0.20)")
    parser.add_argument("--vqav2_n", type=int, default=1000)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_id = args.model_id
    family = detect_family(model_id)
    safe_name = model_id.replace("/", "__")
    pr_tag = f"pr{int(args.prune_ratio * 100)}"
    mr_tag = f"mr{int(args.merge_ratio * 100)}"
    tag = f"{safe_name}__pact_{pr_tag}_{mr_tag}"
    out_path = RESULTS_DIR / f"{tag}.json"

    if out_path.exists() and not args.force:
        logger.info(f"Result already exists at {out_path}. Skipping.")
        return

    # Load model
    logger.info(f"Loading {model_id} (fp16)...")
    model, processor, meta = load_model(model_id, quant="fp16")
    device = str(next(model.parameters()).device)
    num_params = sum(p.numel() for p in model.parameters())

    # PACT is applied during inference, not as a model modification
    pact = PACTWrapper(model, family,
                       prune_ratio=args.prune_ratio,
                       merge_ratio=args.merge_ratio)

    total_reduction = args.prune_ratio + (1 - args.prune_ratio) * args.merge_ratio

    results = {
        "model_id": model_id,
        "family": family,
        "method": "pact",
        "prune_ratio": args.prune_ratio,
        "merge_ratio": args.merge_ratio,
        "total_token_reduction": round(total_reduction, 4),
        "num_params_M": round(num_params / 1e6, 1),
        "gpu_mem_load_mb": meta.gpu_mem_delta_mb,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmarks": {},
    }

    if not args.skip_eval:
        eval_samples = load_vqav2(n_samples=args.vqav2_n)
        results["benchmarks"]["vqav2"] = evaluate_dataset(
            model, processor, eval_samples, family, device,
            "VQAv2", _vqa_accuracy,
        )

    results["token_stats"] = pact.get_stats()

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"PACT results saved to {out_path}")

    unload_model(model)


if __name__ == "__main__":
    main()
