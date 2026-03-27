#!/usr/bin/env python3
"""
Benchmark pruned models from HuggingFace on Jetson.
These are standard FP16 weights with zeros (from pruning on A6000).
Model weights from pruned HF repo, processor from base model.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from profiling.gpu_profiler import GPUProfiler
from evaluation.run_baseline import (
    load_vqav2, run_inference, _vqa_accuracy, _vqa_multi_metric
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def benchmark_pruned_model(hf_repo: str, n_samples: int = 50, force: bool = False):
    """Load a pruned model from HuggingFace and benchmark it."""
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from huggingface_hub import hf_hub_download

    results_dir = Path(__file__).resolve().parents[1] / "results" / "jetson" / "pruning_hf"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load prune config
    try:
        prune_cfg_path = hf_hub_download(hf_repo, "prune_config.json")
        with open(prune_cfg_path) as f:
            prune_cfg = json.load(f)
        method = prune_cfg.get("method", "unknown")
        sparsity = prune_cfg.get("sparsity", 0.0)
        base_model = prune_cfg.get("model_id", "HuggingFaceTB/SmolVLM-Instruct")
        family = prune_cfg.get("family", "smolvlm")
    except Exception:
        method = "unknown"
        sparsity = 0.0
        base_model = "HuggingFaceTB/SmolVLM-Instruct"
        family = "smolvlm"

    safe_name = f"{base_model.replace('/', '__')}__{method}_sp{int(sparsity*100)}"
    out_path = results_dir / f"{safe_name}.json"

    if out_path.exists() and not force:
        logger.info(f"Result already exists: {out_path}. Use --force to overwrite.")
        return

    logger.info(f"=== Benchmarking: {hf_repo} ===")
    logger.info(f"  Method: {method}, Sparsity: {sparsity}, Base: {base_model}")

    # Record memory before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1e6

    # Load processor from BASE model (pruned repo has broken tokenizer config)
    logger.info(f"  Loading processor from {base_model}...")
    processor = AutoProcessor.from_pretrained(base_model)

    # Load model weights from PRUNED repo
    logger.info(f"  Loading model from {hf_repo}...")
    t_load = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        hf_repo,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    load_time = time.perf_counter() - t_load

    if torch.cuda.is_available():
        mem_after = torch.cuda.memory_allocated() / 1e6
    else:
        mem_after = 0

    logger.info(f"  Loaded in {load_time:.1f}s, GPU mem: {mem_before:.0f} -> {mem_after:.0f} MB")

    # Count parameters and sparsity
    total_params = sum(p.numel() for p in model.parameters())
    zero_params = sum((p == 0).sum().item() for p in model.parameters())
    actual_sparsity = zero_params / total_params if total_params > 0 else 0

    logger.info(f"  Params: {total_params/1e6:.1f}M, Zeros: {zero_params/1e6:.1f}M, "
                f"Actual sparsity: {actual_sparsity:.2%}")

    # Load VQAv2
    samples = load_vqav2(n_samples)

    # Evaluate with multi-metric
    device = "cuda" if torch.cuda.is_available() else "cpu"
    profiler = GPUProfiler(device_index=0)
    scores = []
    multi_scores = []
    latencies = []
    skipped = 0

    logger.info(f"  Running VQAv2 evaluation ({n_samples} samples)...")
    with profiler:
        for sample in tqdm(samples, desc=f"{method}_sp{int(sparsity*100)}", leave=False):
            t0 = time.perf_counter()
            try:
                pred = run_inference(model, processor, sample, family, device)
            except Exception as e:
                logger.warning(f"  Skipping sample: {e}")
                skipped += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            latencies.append(time.perf_counter() - t0)

            m = _vqa_multi_metric(pred, sample["answers"])
            multi_scores.append(m)
            scores.append(m["exact_match"])

    n_evaluated = len(scores)
    stats = profiler.stats()
    avg_acc = sum(scores) / n_evaluated if n_evaluated else 0.0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    throughput = len(latencies) / stats.wall_time_s if stats.wall_time_s > 0 else 0.0

    # Aggregate multi-metric
    if multi_scores:
        metric_names = list(multi_scores[0].keys())
        metric_avgs = {
            name: round(sum(m[name] for m in multi_scores) / len(multi_scores), 4)
            for name in metric_names
        }
    else:
        metric_avgs = {}

    logger.info(f"  Results: EM={avg_acc:.4f}, Lat={avg_lat:.2f}s, "
                f"Mem={stats.peak_memory_mb:.0f}MB, Tput={throughput:.3f} samp/s")
    if metric_avgs:
        logger.info(f"  Multi-metric: {metric_avgs}")

    # Save result
    result = {
        "model_id": base_model,
        "hf_repo": hf_repo,
        "family": family,
        "method": method,
        "sparsity": sparsity,
        "actual_sparsity": round(actual_sparsity, 4),
        "num_params_M": round(total_params / 1e6, 1),
        "zero_params_M": round(zero_params / 1e6, 1),
        "gpu_mem_before_mb": round(mem_before, 1),
        "gpu_mem_after_mb": round(mem_after, 1),
        "gpu_mem_load_mb": round(mem_after - mem_before, 1),
        "load_time_s": round(load_time, 1),
        "benchmarks": {
            "vqav2": {
                "accuracy": round(avg_acc, 4),
                "avg_latency_s": round(avg_lat, 4),
                "peak_memory_mb": round(stats.peak_memory_mb, 1),
                "avg_memory_mb": round(stats.avg_memory_mb, 1),
                "throughput_sps": round(throughput, 3),
                "avg_power_w": round(stats.avg_power_w, 1),
                "avg_gpu_util_pct": round(stats.avg_gpu_util_pct, 1),
                "n_samples": n_samples,
                "n_evaluated": n_evaluated,
                "n_skipped": skipped,
                "all_failed": n_evaluated == 0,
                "zero_accuracy_warning": avg_acc == 0.0 and n_evaluated >= 5,
                "metrics": metric_avgs,
            }
        },
        "device": "jetson_orin_nano_8gb",
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"  Saved to {out_path}")

    # Cleanup
    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="HuggingFace repo ID")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    benchmark_pruned_model(args.repo, args.n_samples, args.force)
