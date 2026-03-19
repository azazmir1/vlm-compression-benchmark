"""
Evaluate moondream/moondream-2b-2025-04-14-4bit on VQAv2.
Saves results in the same format as run_ptq.py for comparison.
"""

import json
import logging
import sys
import time
import traceback
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from profiling.gpu_profiler import GPUProfiler
from evaluation.run_baseline import load_vqav2, _vqa_accuracy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_ID = "moondream/moondream-2b-2025-04-14-4bit"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "ptq"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _gpu_mem_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info(torch.cuda.current_device())
    return (total - free) / 1024**2


def main():
    out_path = RESULTS_DIR / "moondream__moondream-2b-2025-04-14-4bit__int4__native.json"

    # ── Load model ────────────────────────────────────────────────────────
    torch.cuda.empty_cache()
    mem_before = _gpu_mem_mb()

    logger.info(f"Loading {MODEL_ID} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map={"": "cuda"},
    )
    torch.cuda.synchronize()
    mem_after = _gpu_mem_mb()
    gpu_mem_delta = round(mem_after - mem_before, 1)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Params: {num_params / 1e6:.1f}M  |  VRAM delta: {gpu_mem_delta:.0f} MB")

    # ── Load VQAv2 ────────────────────────────────────────────────────────
    samples = load_vqav2(n_samples=1000)

    # ── Evaluate ──────────────────────────────────────────────────────────
    logger.info(f"Evaluating on VQAv2 ({len(samples)} samples)...")
    scores, latencies = [], []
    skipped = 0
    profiler = GPUProfiler(device_index=0)

    with profiler:
        for sample in tqdm(samples, desc="VQAv2", leave=False):
            image = sample["image"]
            question = sample["question"] + " Answer with a single word or short phrase."

            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            image = image.convert("RGB")

            t0 = time.perf_counter()
            try:
                result = model.query(image, question)
                pred = result["answer"].strip()
            except Exception:
                logger.warning(f"  Skipping sample: {traceback.format_exc()}")
                skipped += 1
                torch.cuda.empty_cache()
                continue

            latencies.append(time.perf_counter() - t0)
            scores.append(_vqa_accuracy(pred, sample["answers"]))

    if skipped:
        logger.warning(f"  Skipped {skipped} samples due to errors")

    stats = profiler.stats()
    avg_acc = sum(scores) / len(scores) if scores else 0.0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    throughput = len(latencies) / stats.wall_time_s if stats.wall_time_s > 0 else 0.0

    logger.info(
        f"  VQAv2: acc={avg_acc:.4f}  lat={avg_lat:.3f}s  "
        f"mem={stats.peak_memory_mb:.0f}MB  tput={throughput:.2f} samp/s"
    )

    # ── Compression ratio vs fp16 baseline ────────────────────────────────
    baseline_path = (
        Path(__file__).resolve().parents[1]
        / "results" / "baseline" / "vikhyatk__moondream2.json"
    )
    compression_ratio = None
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        baseline_mem = baseline.get("gpu_mem_load_mb", 0)
        if baseline_mem > 0 and gpu_mem_delta > 0:
            compression_ratio = round(baseline_mem / gpu_mem_delta, 2)

    # ── Save ──────────────────────────────────────────────────────────────
    results = {
        "model_id": MODEL_ID,
        "family": "moondream",
        "quant": "int4",
        "backend": "native (torchao QAT)",
        "num_params_M": round(num_params / 1e6, 1),
        "gpu_mem_load_mb": gpu_mem_delta,
        "compression_ratio": compression_ratio,
        "benchmarks": {
            "vqav2": {
                "accuracy": round(avg_acc, 4),
                "avg_latency_s": round(avg_lat, 4),
                "peak_memory_mb": round(stats.peak_memory_mb, 1),
                "avg_memory_mb": round(stats.avg_memory_mb, 1),
                "throughput_sps": round(throughput, 3),
                "avg_power_w": round(stats.avg_power_w, 1),
                "avg_gpu_util_pct": round(stats.avg_gpu_util_pct, 1),
                "n_samples": len(samples),
            }
        },
    }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    # ── Cleanup ───────────────────────────────────────────────────────────
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Model unloaded and GPU freed.")


if __name__ == "__main__":
    main()
