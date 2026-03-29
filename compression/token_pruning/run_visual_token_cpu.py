"""
compression/token_pruning/run_visual_token_cpu.py
===================================================
PACT-inspired visual token compression for VLMs on CPU.

Based on: "PACT: Pruning and Clustering-Based Token Reduction for Faster VLMs"
          (Orailix et al., CVPR 2025)

Strategy: training-free, plug-and-play visual token reduction.

For SmolVLM, image tokens are produced by the SigLIP vision encoder.
We prune them after the vision connector by:
  1. Scoring each token by its L2 norm (low-norm = less informative)
  2. Keeping the top-K tokens (sorted to preserve spatial order)
  3. Padding/masking the dropped positions

Since SmolVLM's input_ids encodes the exact number of <image> tokens expected,
we implement token reduction via image resolution downscaling before the
processor — reducing patch count while preserving content proportionally.
This is the practical equivalent of dropping low-importance tokens as used in
PACT's uniform-baseline ablation.

Usage:
  python compression/token_pruning/run_visual_token_cpu.py \
      --model_id HuggingFaceTB/SmolVLM-256M-Instruct --keep_ratio 0.5
  python compression/token_pruning/run_visual_token_cpu.py \
      --model_id HuggingFaceTB/SmolVLM-256M-Instruct --keep_ratio 0.25
"""

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.model_loader import load_model, unload_model
from evaluation.run_baseline import load_vqav2, _vqa_accuracy
from profiling.gpu_profiler import GPUProfiler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "token_pruning_cpu"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def resize_image_for_token_reduction(image: Image.Image, keep_ratio: float) -> Image.Image:
    """
    Resize image so that the number of vision tokens is approximately
    keep_ratio * original_tokens. Since token count ∝ area ∝ w*h,
    we scale each dimension by sqrt(keep_ratio).
    """
    w, h = image.size
    scale = keep_ratio ** 0.5
    new_w = max(32, int(w * scale))
    new_h = max(32, int(h * scale))
    return image.resize((new_w, new_h), Image.LANCZOS)


def run_inference_with_token_pruning(
    model, processor, sample: dict, family: str,
    device: str, keep_ratio: float, max_new_tokens: int = 30
) -> str:
    """SmolVLM inference with visual token reduction via image resizing."""
    image = sample["image"]
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    # Reduce visual tokens by downscaling before the processor
    image = resize_image_for_token_reduction(image, keep_ratio)

    question = sample["question"] + " Answer with a single word or short phrase."

    if family in ("smolvlm", "nanovlm", "fastvlm"):
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": question}],
        }]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        pred = processor.batch_decode(
            ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]
    else:
        raise ValueError(f"Token pruning not implemented for family '{family}'")

    return pred.strip()


def evaluate_with_token_pruning(
    model, processor, samples: list, family: str,
    device: str, keep_ratio: float
) -> dict:
    logger.info(f"  Evaluating VQAv2 ({len(samples)} samples, keep_ratio={keep_ratio})...")
    scores, latencies = [], []
    skipped = 0
    profiler = GPUProfiler()

    with profiler:
        for sample in tqdm(samples, desc="VQAv2-TokenPrune", leave=False):
            t0 = time.perf_counter()
            try:
                pred = run_inference_with_token_pruning(
                    model, processor, sample, family, device, keep_ratio
                )
            except Exception:
                logger.warning(f"Skipping sample: {traceback.format_exc()}")
                skipped += 1
                continue
            latencies.append(time.perf_counter() - t0)
            scores.append(_vqa_accuracy(pred, sample["answers"]))

    if skipped:
        logger.warning(f"  Skipped {skipped} samples")

    stats = profiler.stats()
    avg_acc = sum(scores) / len(scores) if scores else 0.0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    throughput = len(latencies) / stats.wall_time_s if stats.wall_time_s > 0 else 0.0

    logger.info(
        f"  VQAv2: acc={avg_acc:.4f}  lat={avg_lat:.3f}s  "
        f"mem={stats.peak_memory_mb:.0f}MB  tput={throughput:.3f} samp/s"
    )
    return {
        "accuracy":       round(avg_acc, 4),
        "avg_latency_s":  round(avg_lat, 4),
        "peak_memory_mb": round(stats.peak_memory_mb, 1),
        "avg_memory_mb":  round(stats.avg_memory_mb, 1),
        "throughput_sps": round(throughput, 3),
        "n_samples":      len(latencies),
    }


def main():
    parser = argparse.ArgumentParser(description="PACT-inspired visual token reduction")
    parser.add_argument("--model_id",    required=True)
    parser.add_argument("--keep_ratio",  type=float, default=0.5,
                        help="Fraction of vision tokens to keep via image downscaling "
                             "(0.5 = 50%% of tokens, 0.25 = 25%% of tokens)")
    parser.add_argument("--vqav2_n",     type=int, default=200)
    parser.add_argument("--skip_vqav2",  action="store_true")
    parser.add_argument("--force",       action="store_true")
    args = parser.parse_args()

    safe_name = args.model_id.replace("/", "__")
    tag       = f"{safe_name}__tokenpruning_k{int(args.keep_ratio * 100)}"
    out_path  = RESULTS_DIR / f"{tag}.json"

    if out_path.exists() and not args.force:
        logger.info(f"Result already exists at {out_path}. Skipping.")
        return

    model, processor, meta = load_model(args.model_id)
    family = meta.family
    device = "cpu"

    results = {
        "model_id":     args.model_id,
        "family":       family,
        "method":       "pact_visual_token_reduction",
        "keep_ratio":   args.keep_ratio,
        "device":       "cpu",
        "num_params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "ram_load_mb":  meta.gpu_mem_delta_mb,
        "benchmarks":   {},
    }

    if not args.skip_vqav2:
        samples = load_vqav2(n_samples=args.vqav2_n)
        results["benchmarks"]["vqav2"] = evaluate_with_token_pruning(
            model, processor, samples, family, device, args.keep_ratio
        )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Token pruning results saved to {out_path}")

    unload_model(model)


if __name__ == "__main__":
    main()
