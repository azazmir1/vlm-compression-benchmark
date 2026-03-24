"""
compression/pact/run_pact.py
=============================
PACT: Pruning And Clustering-based Token reduction for faster VLMs.

Based on: "PACT: Pruning and Clustering-Based Token Reduction for Faster VLMs"
          (Dhouib et al., CVPR 2025)
          GitHub: https://github.com/orailix/PACT

Algorithm (applied at an early LLM decoder layer, default layer 4):
  1. EUTI Pruning: Compute per-token self-consistency score = mean_heads(k_i · q_i).
     Tokens with low k-q alignment are unimportant. Prune the bottom fraction.
  2. DBDPC Clustering: Run Distance-Bounded Density Peak Clustering on surviving
     tokens' hidden states (cosine distance). Cluster similar tokens together.
  3. Recovery: Pruned tokens within cutoff*coef_pruned distance of a cluster center
     are re-included in that cluster.
  4. Merge: Average hidden states within each cluster. Zero out non-representative
     positions (keeps sequence length intact for RoPE/attention mask compatibility).

Usage:
  python compression/pact/run_pact.py \
      --model_id AIDC-AI/Ovis2-1B --prune_ratio 0.30 --cutoff 0.3
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.model_loader import load_model, unload_model
from evaluation.run_baseline import (
    load_vqav2, run_inference,
    _vqa_accuracy, _vqa_multi_metric,
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


# ── DBDPC Clustering ─────────────────────────────────────────────────────────

def dbdpc_cluster(features: torch.Tensor, cutoff: float, dc: float = 2.0):
    """Distance-Bounded Density Peak Clustering (DBDPC).

    Args:
        features: (N, D) tensor of token hidden states.
        cutoff: cosine distance threshold for cluster membership.
        dc: bandwidth parameter for density estimation.

    Returns:
        cluster_ids: (N,) tensor of cluster assignments (0-indexed).
        centers: list of center indices.
    """
    N = features.shape[0]
    if N == 0:
        return torch.zeros(0, dtype=torch.long, device=features.device), []

    # Cosine distance matrix
    normed = F.normalize(features.float(), p=2, dim=-1)
    sim = normed @ normed.t()
    dist = 1.0 - sim  # (N, N), range [0, 2]

    # Local density (Gaussian kernel)
    rho = torch.exp(-(dist / dc) ** 2).sum(dim=1)  # (N,)
    # Convert to ranks for numerical stability
    rho_ranks = torch.zeros_like(rho)
    sorted_idx = rho.argsort()
    rho_ranks[sorted_idx] = torch.arange(N, dtype=rho.dtype, device=rho.device)
    rho = rho_ranks

    cluster_ids = torch.full((N,), -1, dtype=torch.long, device=features.device)
    centers = []

    assigned = torch.zeros(N, dtype=torch.bool, device=features.device)

    max_iters = N  # safety bound
    for _ in range(max_iters):
        unassigned = (~assigned).nonzero(as_tuple=True)[0]
        if len(unassigned) == 0:
            break

        # Compute delta for unassigned tokens
        rho_un = rho[unassigned]
        dist_un = dist[unassigned][:, unassigned]  # (n_un, n_un)

        delta = torch.full((len(unassigned),), float('inf'), device=features.device)
        for i in range(len(unassigned)):
            higher = rho_un > rho_un[i]
            if higher.any():
                delta[i] = dist_un[i, higher].min()

        # Select new centers: highest-density token + any with delta > cutoff
        new_center_mask = delta > cutoff
        # Always include the highest-density unassigned token
        max_rho_idx = rho_un.argmax()
        new_center_mask[max_rho_idx] = True

        new_center_local = new_center_mask.nonzero(as_tuple=True)[0]

        if len(new_center_local) == 0:
            # Fallback: assign remaining to nearest existing center
            break

        # Register new centers
        for c_local in new_center_local:
            c_global = unassigned[c_local].item()
            cid = len(centers)
            centers.append(c_global)
            cluster_ids[c_global] = cid
            assigned[c_global] = True

        # Assign unassigned tokens within cutoff of new centers
        still_unassigned = (~assigned).nonzero(as_tuple=True)[0]
        if len(still_unassigned) == 0:
            break

        for c_local in new_center_local:
            c_global = unassigned[c_local].item()
            cid = cluster_ids[c_global].item()
            dists_to_center = dist[c_global, still_unassigned]
            within = dists_to_center <= cutoff
            assignable = still_unassigned[within]
            for idx in assignable:
                if not assigned[idx]:
                    cluster_ids[idx] = cid
                    assigned[idx] = True

        # Check if we made progress
        newly_unassigned = (~assigned).nonzero(as_tuple=True)[0]
        if len(newly_unassigned) == len(still_unassigned):
            # No progress — assign remaining greedily to nearest center
            if len(centers) > 0:
                center_indices = torch.tensor(centers, device=features.device)
                for idx in newly_unassigned:
                    dists_to_centers = dist[idx, center_indices]
                    nearest = dists_to_centers.argmin().item()
                    cluster_ids[idx] = nearest
                    assigned[idx] = True
            break

    # Assign any remaining unassigned tokens to nearest center
    remaining = (~assigned).nonzero(as_tuple=True)[0]
    if len(remaining) > 0 and len(centers) > 0:
        center_indices = torch.tensor(centers, device=features.device)
        for idx in remaining:
            dists_to_centers = dist[idx, center_indices]
            nearest = dists_to_centers.argmin().item()
            cluster_ids[idx] = nearest
            assigned[idx] = True

    return cluster_ids, centers


# ── PACT Token Compressor ────────────────────────────────────────────────────

class PACTTokenCompressor:
    """
    Faithful implementation of PACT (Dhouib et al., CVPR 2025).

    Installs a forward pre-hook on an LLM decoder layer to:
      1. Score visual tokens via EUTI (key-query self-consistency)
      2. Prune low-scoring tokens
      3. Cluster surviving tokens via DBDPC
      4. Recover pruned tokens close to cluster centers
      5. Merge each cluster into a single representative token

    Uses zero-out approach to keep sequence length intact (RoPE/mask safe).
    """

    def __init__(self, model: nn.Module, family: str,
                 prune_ratio: float = 0.3, cutoff: float = 0.3,
                 target_layer: int = 4, coef_pruned: float = 1.5,
                 dc: float = 2.0):
        self.prune_ratio = prune_ratio
        self.cutoff = cutoff
        self.target_layer = target_layer
        self.coef_pruned = coef_pruned
        self.dc = dc
        self.family = family
        self.hook = None
        self._self_attn = None  # reference to the layer's self-attention module
        self.stats = {
            "tokens_pruned": 0, "tokens_recovered": 0,
            "tokens_merged": 0, "clusters_formed": 0, "calls": 0,
        }
        self._install_hook(model)

    def _find_layers(self, model: nn.Module):
        """Find transformer layer list."""
        for attr_path in [
            "model.layers", "language_model.model.layers",
            "transformer.h", "model.decoder.layers",
            "gpt_neox.layers", "model.model.layers",
            "llm.model.layers",
        ]:
            obj = model
            try:
                for attr in attr_path.split("."):
                    obj = getattr(obj, attr)
                if hasattr(obj, "__getitem__") and len(obj) > 0:
                    return obj
            except (AttributeError, IndexError, TypeError):
                continue

        # Fallback
        for name, module in model.named_modules():
            if hasattr(module, "__getitem__") and hasattr(module, "__len__"):
                try:
                    if len(module) > self.target_layer and hasattr(module[0], "forward"):
                        return module
                except (TypeError, IndexError):
                    continue

        raise RuntimeError("Could not find transformer layers for PACT hook")

    def _find_self_attn(self, layer: nn.Module):
        """Find the self-attention submodule within a layer."""
        for name in ["self_attn", "attn", "attention", "self_attention"]:
            if hasattr(layer, name):
                attn = getattr(layer, name)
                # Verify it has q_proj and k_proj
                if hasattr(attn, "q_proj") and hasattr(attn, "k_proj"):
                    return attn
        return None

    def _install_hook(self, model: nn.Module):
        """Install forward pre-hook on target layer."""
        try:
            layers = self._find_layers(model)
            idx = min(self.target_layer, len(layers) - 1)
            target = layers[idx]
            self._self_attn = self._find_self_attn(target)
            if self._self_attn is None:
                logger.warning("Could not find self_attn with q_proj/k_proj. "
                               "Falling back to L2-norm scoring.")
            self.hook = target.register_forward_pre_hook(self._pact_hook)
            logger.info(f"PACT hook installed on layer {idx}"
                        f" (EUTI={'yes' if self._self_attn else 'no'})")
        except RuntimeError as e:
            logger.warning(f"Could not install PACT hook: {e}")

    def _compute_euti_scores(self, hidden_states: torch.Tensor, n_visual: int):
        """Compute EUTI importance scores: mean_heads(k_i · q_i) for visual tokens.

        This measures per-token self-consistency between its key and query
        representations, before RoPE is applied (using raw projections).
        """
        if self._self_attn is None:
            # Fallback: use L2 norm
            return hidden_states[:, :n_visual, :].float().norm(dim=-1)

        attn = self._self_attn
        visual_hs = hidden_states[:, :n_visual, :]  # (B, n_visual, D)

        with torch.no_grad():
            q = attn.q_proj(visual_hs)  # (B, n_visual, num_heads * head_dim)
            k = attn.k_proj(visual_hs)  # (B, n_visual, num_kv_heads * head_dim)

        # Determine head_dim from k_proj output
        k_dim = k.shape[-1]
        q_dim = q.shape[-1]

        # For GQA: num_kv_heads may < num_heads. Use k's head structure.
        if hasattr(attn, 'num_key_value_heads'):
            num_kv_heads = attn.num_key_value_heads
        elif hasattr(attn, 'num_heads'):
            num_kv_heads = attn.num_heads
        else:
            num_kv_heads = 1

        head_dim = k_dim // num_kv_heads
        num_q_heads = q_dim // head_dim

        # Reshape: (B, n_visual, num_heads, head_dim)
        B = hidden_states.shape[0]
        q = q.view(B, n_visual, num_q_heads, head_dim)
        k = k.view(B, n_visual, num_kv_heads, head_dim)

        # For GQA, repeat k heads to match q heads
        if num_kv_heads < num_q_heads:
            repeats = num_q_heads // num_kv_heads
            k = k.repeat_interleave(repeats, dim=2)

        # EUTI score: element-wise dot product k_i · q_i, averaged across heads
        scores = (k.float() * q.float()).sum(dim=-1)  # (B, n_visual, num_q_heads)
        scores = scores.mean(dim=-1)  # (B, n_visual)

        return scores

    def _pact_hook(self, module, args):
        """Forward pre-hook implementing the full PACT algorithm."""
        if not args:
            return args

        hidden_states = args[0]
        if hidden_states.dim() != 3:
            return args

        batch, seq_len, hidden_dim = hidden_states.shape

        # Estimate visual token count (first ~50% for VLMs)
        n_visual = max(int(seq_len * 0.5), 1)
        if n_visual < 4:
            return args

        # Clone to avoid autograd in-place issues
        hidden_states = hidden_states.clone()

        # ── Step 1: EUTI Pruning ──────────────────────────────────────────
        scores = self._compute_euti_scores(hidden_states, n_visual)  # (B, n_visual)
        n_keep = max(int(n_visual * (1 - self.prune_ratio)), 2)
        n_prune = n_visual - n_keep

        pruned_global_indices = []  # track for recovery
        if n_prune > 0:
            _, sorted_idx = scores[0].sort()  # ascending (batch 0)
            prune_idx = sorted_idx[:n_prune]
            keep_idx = sorted_idx[n_prune:]
            pruned_global_indices = prune_idx.tolist()

            # Zero out pruned tokens
            for b in range(batch):
                hidden_states[b, prune_idx, :] = 0.0
            self.stats["tokens_pruned"] += n_prune

        # ── Step 2: DBDPC Clustering on surviving tokens ──────────────────
        # Get surviving visual token indices
        norms = hidden_states[0, :n_visual, :].float().norm(dim=-1)
        alive_mask = norms > 0
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]  # global indices
        n_alive = len(alive_indices)

        if n_alive < 2:
            self.stats["calls"] += 1
            return (hidden_states,) + args[1:]

        alive_features = hidden_states[0, alive_indices, :]  # (n_alive, D)

        cluster_ids, centers = dbdpc_cluster(
            alive_features, cutoff=self.cutoff, dc=self.dc
        )
        n_clusters = len(centers)
        self.stats["clusters_formed"] += n_clusters

        # ── Step 3: Recovery — re-include pruned tokens near cluster centers ──
        if len(pruned_global_indices) > 0 and n_clusters > 0:
            pruned_idx_t = torch.tensor(pruned_global_indices, device=hidden_states.device)
            # Use original (pre-zeroed) hidden states for pruned tokens:
            # we already zeroed them, so re-read from args[0]
            orig_hs = args[0]
            pruned_features = orig_hs[0, pruned_idx_t, :]  # (n_pruned, D)

            center_global = [alive_indices[c].item() for c in centers]
            center_features = hidden_states[0, torch.tensor(center_global, device=hidden_states.device), :]

            # Cosine distance from each pruned token to each center
            pruned_normed = F.normalize(pruned_features.float(), p=2, dim=-1)
            center_normed = F.normalize(center_features.float(), p=2, dim=-1)
            dist_to_centers = 1.0 - pruned_normed @ center_normed.t()  # (n_pruned, n_centers)

            recovery_cutoff = self.cutoff * self.coef_pruned
            for i, g_idx in enumerate(pruned_global_indices):
                min_dist, nearest_center = dist_to_centers[i].min(dim=0)
                if min_dist.item() <= recovery_cutoff:
                    # Recover: restore hidden state and assign to cluster
                    for b in range(batch):
                        hidden_states[b, g_idx, :] = orig_hs[b, g_idx, :]
                    # Map the nearest center (index in centers list) to cluster_id
                    cid = nearest_center.item()
                    # We'll handle merging below by grouping
                    # For now, just mark as recovered
                    alive_indices = torch.cat([alive_indices,
                                               torch.tensor([g_idx], device=hidden_states.device)])
                    cluster_ids = torch.cat([cluster_ids,
                                             torch.tensor([cid], device=hidden_states.device)])
                    self.stats["tokens_recovered"] += 1
                    self.stats["tokens_pruned"] -= 1  # undo prune count

        # ── Step 4: Merge — average tokens within each cluster, keep center ──
        if n_clusters > 0:
            for cid in range(n_clusters):
                members = (cluster_ids == cid).nonzero(as_tuple=True)[0]
                if len(members) <= 1:
                    continue

                # Map local indices back to global positions
                member_globals = alive_indices[members]
                center_global = alive_indices[centers[cid]].item()

                # Compute mean of all member hidden states
                member_hs = hidden_states[:, member_globals, :]  # (B, n_members, D)
                merged = member_hs.mean(dim=1)  # (B, D)

                # Place merged representation at the center position
                hidden_states[:, center_global, :] = merged

                # Zero out non-center members
                for m_global in member_globals:
                    m_global = m_global.item()
                    if m_global != center_global:
                        hidden_states[:, m_global, :] = 0.0
                        self.stats["tokens_merged"] += 1

        self.stats["calls"] += 1

        # Return modified hidden states (same shape)
        new_args = (hidden_states,) + args[1:]
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
    """Evaluate model with PACT token compression active (multi-metric)."""
    logger.info(f"  Evaluating on {dataset_name} ({len(samples)} samples) with PACT...")
    scores, latencies = [], []
    multi_scores = []
    use_multi = (accuracy_fn is _vqa_accuracy)
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
            if use_multi:
                m = _vqa_multi_metric(pred, sample["answers"])
                multi_scores.append(m)
                scores.append(m["exact_match"])
            else:
                scores.append(accuracy_fn(pred, sample["answers"]))

    if skipped:
        logger.warning(f"  {dataset_name}: skipped {skipped} samples")

    n_evaluated = len(scores)
    stats = profiler.stats()
    avg_acc = sum(scores) / n_evaluated if n_evaluated else 0.0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    throughput = len(latencies) / stats.wall_time_s if stats.wall_time_s > 0 else 0.0

    metric_avgs = {}
    if use_multi and multi_scores:
        metric_names = list(multi_scores[0].keys())
        metric_avgs = {
            name: round(sum(m[name] for m in multi_scores) / len(multi_scores), 4)
            for name in metric_names
        }
        logger.info(
            f"  {dataset_name}: "
            + "  ".join(f"{k}={v:.4f}" for k, v in metric_avgs.items())
            + f"  lat={avg_lat:.3f}s  mem={stats.peak_memory_mb:.0f}MB"
        )
    else:
        logger.info(
            f"  {dataset_name}: acc={avg_acc:.4f}  lat={avg_lat:.3f}s  "
            f"mem={stats.peak_memory_mb:.0f}MB"
        )

    result = {
        "accuracy": round(avg_acc, 4),
        "avg_latency_s": round(avg_lat, 4),
        "peak_memory_mb": round(stats.peak_memory_mb, 1),
        "avg_memory_mb": round(stats.avg_memory_mb, 1),
        "throughput_sps": round(throughput, 3),
        "avg_power_w": round(stats.avg_power_w, 1),
        "avg_gpu_util_pct": round(stats.avg_gpu_util_pct, 1),
        "n_samples": len(samples),
        "n_evaluated": n_evaluated,
        "n_skipped": skipped,
    }
    if metric_avgs:
        result["metrics"] = metric_avgs

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PACT visual token compression")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--prune_ratio", type=float, default=0.30,
                        help="Fraction of visual tokens to prune (default 0.30)")
    parser.add_argument("--cutoff", type=float, default=0.3,
                        help="DBDPC cosine distance cutoff (default 0.3)")
    parser.add_argument("--target_layer", type=int, default=4,
                        help="LLM layer to apply compression (default 4, per paper)")
    parser.add_argument("--coef_pruned", type=float, default=1.5,
                        help="Recovery cutoff multiplier (default 1.5)")
    parser.add_argument("--vqav2_n", type=int, default=1000)
    parser.add_argument("--skip_vqav2", action="store_true")
    parser.add_argument("--skip_textvqa", action="store_true")
    parser.add_argument("--skip_pope", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_id = args.model_id
    safe_name = model_id.replace("/", "__")
    tag = f"{safe_name}__pact_p{int(args.prune_ratio*100)}_c{int(args.cutoff*100)}"
    out_path = RESULTS_DIR / f"{tag}.json"

    if out_path.exists() and not args.force:
        logger.info(f"Result exists at {out_path}. Skipping.")
        return

    logger.info(f"Loading {model_id} (fp16) for PACT...")
    model, processor, meta = load_model(model_id, quant="fp16")
    family = meta.family
    device = str(next(model.parameters()).device)
    num_params = sum(p.numel() for p in model.parameters())

    logger.info(f"Installing PACT (prune={args.prune_ratio}, cutoff={args.cutoff}, "
                f"layer={args.target_layer})...")
    compressor = PACTTokenCompressor(
        model, family,
        prune_ratio=args.prune_ratio,
        cutoff=args.cutoff,
        target_layer=args.target_layer,
        coef_pruned=args.coef_pruned,
    )

    results = {
        "model_id": model_id,
        "family": family,
        "method": "pact",
        "quant": "fp16",
        "prune_ratio": args.prune_ratio,
        "cutoff": args.cutoff,
        "target_layer": args.target_layer,
        "coef_pruned": args.coef_pruned,
        "num_params_M": round(num_params / 1e6, 1),
        "gpu_mem_load_mb": meta.gpu_mem_delta_mb,
        "benchmarks": {},
    }

    if not args.skip_vqav2:
        samples = load_vqav2(n_samples=args.vqav2_n)
        results["benchmarks"]["vqav2"] = evaluate_with_pact(
            model, processor, samples, family, device, "VQAv2", _vqa_accuracy,
        )

    results["pact_stats"] = compressor.get_stats()
    compressor.remove_hook()

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"PACT results saved to {out_path}")

    unload_model(model)


if __name__ == "__main__":
    main()
