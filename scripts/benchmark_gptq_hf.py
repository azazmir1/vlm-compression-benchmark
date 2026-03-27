#!/usr/bin/env python3
"""
Benchmark GPTQ INT4 model from HuggingFace on Jetson.
GPTQ stores weights as packed int32 (qweight) + scales + qzeros + g_idx.
We dequantize to FP16 on-the-fly using pure PyTorch (no custom CUDA kernels).
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


def dequantize_gptq_layer(qweight, qzeros, scales, g_idx, bits=4):
    """Dequantize a GPTQ packed layer to FP16 weight tensor [out, in]."""
    pack_factor = 32 // bits  # 8 for 4-bit

    # Unpack qweight: [in//8, out] int32 -> [in, out] uint4
    # Key: must interleave with stack+reshape, NOT cat
    unpacked = []
    for i in range(pack_factor):
        unpacked.append((qweight >> (bits * i)) & ((1 << bits) - 1))
    # stack along dim=1 then reshape to interleave correctly
    w_int = torch.stack(unpacked, dim=1).reshape(-1, qweight.shape[1]).to(torch.int32)

    # Unpack qzeros: [n_groups, out//8] int32 -> [n_groups, out] uint4
    z_unpacked = []
    for i in range(pack_factor):
        z_unpacked.append((qzeros >> (bits * i)) & ((1 << bits) - 1))
    zeros = torch.stack(z_unpacked, dim=2).reshape(qzeros.shape[0], -1).to(torch.int32)

    # Dequantize: w_fp = (w_int - zeros[group]) * scales[group]
    g = g_idx.long()
    zeros_per_row = zeros[g]
    scales_per_row = scales[g]
    w_fp = (w_int.to(scales_per_row.dtype) - zeros_per_row.to(scales_per_row.dtype)) * scales_per_row

    # Transpose: GPTQ stores [in, out], PyTorch Linear expects [out, in]
    return w_fp.T


def load_gptq_model(hf_repo: str, base_model: str):
    """Load GPTQ model: base architecture + dequantized GPTQ weights."""
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    # Load processor from base model
    logger.info(f"Loading processor from {base_model}...")
    processor = AutoProcessor.from_pretrained(base_model)

    # Load base model config (without quantization_config to avoid GPTQ loader)
    logger.info(f"Loading base model architecture from {base_model}...")
    config = AutoConfig.from_pretrained(base_model)

    # Load model in FP16 with base config (no quantization)
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        config=config,
        torch_dtype=torch.float16,
        device_map="cpu",  # Load on CPU first
    )

    # Load GPTQ weights
    logger.info(f"Loading GPTQ weights from {hf_repo}...")
    gptq_path = hf_hub_download(hf_repo, "model.safetensors")
    gptq_state = load_file(gptq_path, device="cpu")

    # Find all GPTQ-quantized layers (those with qweight)
    qweight_keys = [k for k in gptq_state if k.endswith('.qweight')]
    logger.info(f"Found {len(qweight_keys)} GPTQ-quantized layers")

    # Read quantization config
    try:
        qcfg_path = hf_hub_download(hf_repo, "config.json")
        with open(qcfg_path) as f:
            qcfg = json.load(f).get("quantization_config", {})
        bits = qcfg.get("bits", 4)
        group_size = qcfg.get("group_size", 128)
    except Exception:
        bits = 4
        group_size = 128

    logger.info(f"GPTQ config: bits={bits}, group_size={group_size}")

    # Pre-dequantize all GPTQ layers and inject as standard nn.Linear weights
    # This is done once at load time — no per-forward-pass dequantization overhead
    replaced = 0
    for qw_key in qweight_keys:
        layer_prefix = qw_key.rsplit('.qweight', 1)[0]

        qweight = gptq_state[f"{layer_prefix}.qweight"]
        qzeros = gptq_state[f"{layer_prefix}.qzeros"]
        scales = gptq_state[f"{layer_prefix}.scales"]
        g_idx = gptq_state[f"{layer_prefix}.g_idx"]
        bias_key = f"{layer_prefix}.bias"
        bias = gptq_state.get(bias_key, None)

        # Dequantize to FP16 [out, in]
        w_fp16 = dequantize_gptq_layer(qweight, qzeros, scales, g_idx, bits=bits)

        # Navigate to the existing nn.Linear and overwrite its weight
        parts = layer_prefix.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        linear = getattr(parent, parts[-1])

        # Overwrite weight data in-place
        linear.weight.data = w_fp16.contiguous()
        if bias is not None:
            linear.bias = nn.Parameter(bias)
        replaced += 1

    logger.info(f"Pre-dequantized {replaced} GPTQ layers into standard nn.Linear (FP16)")

    # Also load non-GPTQ weights (embeddings, layernorms, vision encoder, etc.)
    non_gptq_keys = [k for k in gptq_state if not any(
        k.endswith(s) for s in ['.qweight', '.qzeros', '.scales', '.g_idx']
    )]
    if non_gptq_keys:
        non_gptq_state = {k: gptq_state[k] for k in non_gptq_keys}
        missing, unexpected = model.load_state_dict(non_gptq_state, strict=False)
        logger.info(f"Loaded {len(non_gptq_keys)} non-GPTQ tensors (missing={len(missing)}, unexpected={len(unexpected)})")

    # Move to GPU
    logger.info("Moving model to GPU...")
    model = model.to("cuda")
    model.eval()

    return model, processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Azaz666/SmolVLM-Instruct-GPTQ-INT4")
    parser.add_argument("--base_model", default="HuggingFaceTB/SmolVLM-Instruct")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    results_dir = Path(__file__).resolve().parents[1] / "results" / "jetson" / "gptq_hf"
    results_dir.mkdir(parents=True, exist_ok=True)

    safe_name = f"{args.base_model.replace('/', '__')}__gptq_int4"
    out_path = results_dir / f"{safe_name}.json"

    if out_path.exists() and not args.force:
        logger.info(f"Result exists: {out_path}. Use --force.")
        return

    # Memory before
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1e6

    t_load = time.perf_counter()
    model, processor = load_gptq_model(args.repo, args.base_model)
    load_time = time.perf_counter() - t_load

    mem_after = torch.cuda.memory_allocated() / 1e6
    logger.info(f"Loaded in {load_time:.1f}s, GPU mem: {mem_before:.0f} -> {mem_after:.0f} MB")

    # Count params
    total_params = sum(p.numel() for p in model.parameters()) + sum(b.numel() for b in model.buffers())

    # Load dataset
    samples = load_vqav2(args.n_samples)

    # Evaluate
    device = "cuda"
    family = "smolvlm"
    profiler = GPUProfiler(device_index=0)
    scores, multi_scores, latencies = [], [], []
    skipped = 0

    logger.info(f"Running VQAv2 evaluation ({args.n_samples} samples)...")
    with profiler:
        for sample in tqdm(samples, desc="gptq_int4"):
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
        "method": "gptq_int4",
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
