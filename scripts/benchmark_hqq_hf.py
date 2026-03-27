#!/usr/bin/env python3
"""
Benchmark HQQ INT4 model from HuggingFace on Jetson.
HQQ stores weights as packed uint8 (4bit_u8) + per-group scale/zero.
We pre-dequantize to FP16 at load time using pure PyTorch.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from profiling.gpu_profiler import GPUProfiler
from evaluation.run_baseline import (
    load_vqav2, run_inference, _vqa_multi_metric
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def dequantize_hqq_layer(W_q, scale, zero, shape, group_size=64, nbits=4):
    """Dequantize HQQ packed 4bit_u8 layer to FP16 weight [out, in]."""
    out_feat, in_feat = shape
    n_elements = out_feat * in_feat
    n_groups = n_elements // group_size

    # HQQ 4bit_u8 packing: high nibble first, then low nibble, sequential
    wq_flat = W_q.reshape(-1)
    high = ((wq_flat >> 4) & 0x0F).to(torch.float16)
    low = (wq_flat & 0x0F).to(torch.float16)
    w_unpacked = torch.cat([high, low])  # [n_elements]

    # Reshape into groups and dequantize
    w_grouped = w_unpacked.reshape(n_groups, group_size)
    w_dequant = (w_grouped - zero) * scale  # broadcasting [n_groups, group_size]

    return w_dequant.reshape(out_feat, in_feat)


def load_hqq_model(hf_repo: str, base_model: str):
    """Load HQQ model: base architecture + dequantized HQQ weights."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    logger.info(f"Loading processor from {base_model}...")
    processor = AutoProcessor.from_pretrained(base_model)

    logger.info(f"Loading base model from {base_model} on CPU...")
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    logger.info(f"Loading HQQ weights from {hf_repo}...")
    hqq_path = hf_hub_download(hf_repo, "model.safetensors")
    hqq_state = load_file(hqq_path, device="cpu")

    # Find all HQQ-quantized layers (those with W_q)
    wq_keys = [k for k in hqq_state if k.endswith('.W_q')]
    logger.info(f"Found {len(wq_keys)} HQQ-quantized layers")

    replaced = 0
    for wq_key in wq_keys:
        layer_prefix = wq_key.rsplit('.W_q', 1)[0]

        W_q = hqq_state[f"{layer_prefix}.W_q"]
        scale = hqq_state[f"{layer_prefix}.scale"]
        zero = hqq_state[f"{layer_prefix}.zero"]
        shape = hqq_state[f"{layer_prefix}.shape"].tolist()
        group_size = hqq_state[f"{layer_prefix}.group_size"].item()
        nbits = hqq_state[f"{layer_prefix}.nbits"].item()

        # Dequantize to FP16
        w_fp16 = dequantize_hqq_layer(W_q, scale, zero, shape, group_size, nbits)

        # Navigate to the existing nn.Linear and overwrite weight
        parts = layer_prefix.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        linear = getattr(parent, parts[-1])

        linear.weight.data = w_fp16.contiguous()
        replaced += 1

    logger.info(f"Pre-dequantized {replaced} HQQ layers into standard nn.Linear (FP16)")

    # Load non-HQQ weights (embeddings, layernorms, vision encoder)
    # These are the keys that are just plain weight tensors (not HQQ metadata)
    hqq_metadata_suffixes = [
        '.W_q', '.scale', '.zero', '.shape', '.group_size', '.nbits',
        '.axis', '.channel_wise', '.compute_dtype', '.encoded_state_dict',
        '.offload_meta', '.optimize', '.packing', '.quant_scale',
        '.quant_zero', '.round_zero', '.stores_quant_config',
        '.unpack_view_dtype', '.view_as_float',
    ]
    non_hqq_keys = [k for k in hqq_state if not any(k.endswith(s) for s in hqq_metadata_suffixes)]
    if non_hqq_keys:
        non_hqq_state = {k: hqq_state[k] for k in non_hqq_keys}
        missing, unexpected = model.load_state_dict(non_hqq_state, strict=False)
        logger.info(f"Loaded {len(non_hqq_keys)} non-HQQ tensors (missing={len(missing)}, unexpected={len(unexpected)})")

    logger.info("Moving model to GPU...")
    model = model.to("cuda")
    model.eval()

    return model, processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Azaz666/SmolVLM-Instruct-HQQ-INT4")
    parser.add_argument("--base_model", default="HuggingFaceTB/SmolVLM-Instruct")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    results_dir = Path(__file__).resolve().parents[1] / "results" / "jetson" / "hqq_hf"
    results_dir.mkdir(parents=True, exist_ok=True)

    safe_name = f"{args.base_model.replace('/', '__')}__hqq_int4"
    out_path = results_dir / f"{safe_name}.json"

    if out_path.exists() and not args.force:
        logger.info(f"Result exists: {out_path}. Use --force.")
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1e6

    t_load = time.perf_counter()
    model, processor = load_hqq_model(args.repo, args.base_model)
    load_time = time.perf_counter() - t_load

    mem_after = torch.cuda.memory_allocated() / 1e6
    logger.info(f"Loaded in {load_time:.1f}s, GPU mem: {mem_before:.0f} -> {mem_after:.0f} MB")

    total_params = sum(p.numel() for p in model.parameters())

    samples = load_vqav2(args.n_samples)

    device = "cuda"
    family = "smolvlm"
    profiler = GPUProfiler(device_index=0)
    scores, multi_scores, latencies = [], [], []
    skipped = 0

    logger.info(f"Running VQAv2 evaluation ({args.n_samples} samples)...")
    with profiler:
        for sample in tqdm(samples, desc="hqq_int4"):
            t0 = time.perf_counter()
            try:
                pred = run_inference(model, processor, sample, family, device)
            except Exception as e:
                logger.warning(f"Skipping sample: {e}")
                skipped += 1
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

    metric_avgs = {}
    if multi_scores:
        metric_names = list(multi_scores[0].keys())
        metric_avgs = {
            name: round(sum(m[name] for m in multi_scores) / len(multi_scores), 4)
            for name in metric_names
        }

    logger.info(f"Results: EM={avg_acc:.4f}, Lat={avg_lat:.2f}s, Mem={stats.peak_memory_mb:.0f}MB")
    logger.info(f"Multi-metric: {metric_avgs}")

    result = {
        "model_id": args.base_model,
        "hf_repo": args.repo,
        "family": family,
        "method": "hqq_int4",
        "num_params_M": round(total_params / 1e6, 1),
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
                "n_samples": args.n_samples,
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
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
