"""
scripts/ceiling_scan.py
========================
Safe ceiling scan for all 26 VLM models on Jetson.

For each model:
  1. Load on CPU (real load, uses swap — safe)
  2. Measure actual model size in bytes
  3. Check free GPU memory
  4. If fits: move to GPU, try 1 inference, record results
  5. If doesn't fit: stop, mark as ceiling model, unload
  6. If loads but <500 MB remaining: mark as memory-critical

Categories:
  - RUNNABLE: loads on GPU + inference works + >500 MB remaining
  - MEM_CRITICAL: loads on GPU but <500 MB remaining (can't inference safely)
  - CEILING: can't load onto GPU (model size > free GPU memory)
  - ERROR: loading fails for non-memory reasons (code bugs, missing deps)
"""

import gc
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "ceiling_scan"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Memory threshold: if less than this remains after loading, mark MEM_CRITICAL
MEM_CRITICAL_THRESHOLD_MB = 500

# All 22 models organized by family (smallest to largest)
# Excluded: nanoVLM (undertrained, 0.0 accuracy), Florence-2 (grounding model, not VQA)
ALL_MODELS = [
    # SmolVLM
    ("HuggingFaceTB/SmolVLM-256M-Instruct", "smolvlm"),
    ("HuggingFaceTB/SmolVLM-500M-Instruct", "smolvlm"),
    ("HuggingFaceTB/SmolVLM-Instruct", "smolvlm"),
    # InternVL2.5
    ("OpenGVLab/InternVL2_5-1B", "internvl25"),
    ("OpenGVLab/InternVL2_5-2B", "internvl25"),
    ("OpenGVLab/InternVL2_5-4B", "internvl25"),
    ("OpenGVLab/InternVL2_5-8B", "internvl25"),
    # Qwen2.5-VL
    ("Qwen/Qwen2.5-VL-3B-Instruct", "qwen25vl"),
    ("Qwen/Qwen2.5-VL-7B-Instruct", "qwen25vl"),
    # LFM2-VL
    ("LiquidAI/LFM2-VL-450M", "lfm2vl"),
    ("LiquidAI/LFM2-VL-1.6B", "lfm2vl"),
    ("LiquidAI/LFM2-VL-3B", "lfm2vl"),
    # Moondream
    ("vikhyatk/moondream2", "moondream"),
    # FastVLM
    ("apple/FastVLM-0.5B", "fastvlm"),
    ("apple/FastVLM-1.5B", "fastvlm"),
    ("apple/FastVLM-7B", "fastvlm"),
    # Gemma3
    ("google/gemma-3-4b-it", "gemma3"),
    ("google/gemma-3-12b-it", "gemma3"),
    # Ovis2
    ("AIDC-AI/Ovis2-1B", "ovis2"),
    ("AIDC-AI/Ovis2-2B", "ovis2"),
    ("AIDC-AI/Ovis2-4B", "ovis2"),
    ("AIDC-AI/Ovis2-8B", "ovis2"),
]


def get_free_gpu_mb():
    """Get free GPU memory in MB."""
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info(0)
    return free / 1024**2


def get_used_gpu_mb():
    """Get used GPU memory in MB."""
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info(0)
    return (total - free) / 1024**2


def get_model_size_mb(model):
    """Calculate actual model size in MB from loaded parameters and buffers."""
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.nelement() * p.element_size()
    for b in model.buffers():
        total_bytes += b.nelement() * b.element_size()
    return total_bytes / 1024**2


def cleanup():
    """Aggressive cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def scan_model(model_id: str, family: str) -> dict:
    """Scan a single model. Returns result dict."""
    logger.info(f"{'='*60}")
    logger.info(f"SCANNING: {model_id} (family={family})")
    logger.info(f"{'='*60}")

    result = {
        "model_id": model_id,
        "family": family,
        "status": None,
        "model_size_mb": None,
        "gpu_used_before_mb": None,
        "gpu_used_after_mb": None,
        "gpu_free_after_mb": None,
        "num_params_M": None,
        "inference_ok": None,
        "inference_latency_s": None,
        "error": None,
    }

    gpu_used_before = get_used_gpu_mb()
    gpu_free_before = get_free_gpu_mb()
    result["gpu_used_before_mb"] = round(gpu_used_before, 1)
    logger.info(f"  GPU before: {gpu_used_before:.0f} MB used, {gpu_free_before:.0f} MB free")

    # ── Step 1: Load model on CPU ──────────────────────────────────────
    logger.info(f"  Step 1: Loading on CPU...")
    t0 = time.time()
    try:
        from models.model_loader import load_model
        model, processor, meta = load_model(model_id, device_map="cpu")
    except Exception as e:
        err_msg = str(e)[:200]
        logger.error(f"  FAILED to load on CPU: {err_msg}")
        result["status"] = "ERROR"
        result["error"] = err_msg
        cleanup()
        return result

    cpu_load_time = time.time() - t0
    logger.info(f"  CPU load done in {cpu_load_time:.1f}s")

    # ── Step 2: Measure real model size ────────────────────────────────
    model_size_mb = get_model_size_mb(model)
    n_params = sum(p.numel() for p in model.parameters())
    result["model_size_mb"] = round(model_size_mb, 1)
    result["num_params_M"] = round(n_params / 1e6, 1)
    logger.info(f"  Model size: {model_size_mb:.0f} MB ({n_params/1e6:.0f}M params)")

    # ── Step 3: Check if it fits on GPU ────────────────────────────────
    gpu_free = get_free_gpu_mb()
    # Need model_size + some headroom for CUDA overhead during transfer
    transfer_overhead_mb = 200  # conservative overhead for .to(cuda)
    needed = model_size_mb + transfer_overhead_mb

    logger.info(f"  GPU free: {gpu_free:.0f} MB, need: {needed:.0f} MB (model {model_size_mb:.0f} + {transfer_overhead_mb} overhead)")

    if needed > gpu_free:
        logger.warning(f"  CEILING: model needs {needed:.0f} MB but only {gpu_free:.0f} MB free")
        result["status"] = "CEILING"
        result["error"] = f"Model needs ~{needed:.0f} MB, only {gpu_free:.0f} MB free"
        # Unload from CPU
        del model, processor
        cleanup()
        return result

    # ── Step 4: Move to GPU ────────────────────────────────────────────
    logger.info(f"  Step 2: Moving to GPU...")
    try:
        model = model.to("cuda").eval()
        cleanup()
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        err_msg = str(e)[:200]
        logger.error(f"  OOM moving to GPU: {err_msg}")
        result["status"] = "CEILING"
        result["error"] = f"OOM during GPU transfer: {err_msg}"
        del model, processor
        cleanup()
        return result

    gpu_used_after = get_used_gpu_mb()
    gpu_free_after = get_free_gpu_mb()
    result["gpu_used_after_mb"] = round(gpu_used_after, 1)
    result["gpu_free_after_mb"] = round(gpu_free_after, 1)
    gpu_delta = gpu_used_after - gpu_used_before

    logger.info(f"  GPU after load: {gpu_used_after:.0f} MB used, {gpu_free_after:.0f} MB free (delta: {gpu_delta:.0f} MB)")

    # ── Step 5: Check memory-critical ──────────────────────────────────
    if gpu_free_after < MEM_CRITICAL_THRESHOLD_MB:
        logger.warning(f"  MEM_CRITICAL: only {gpu_free_after:.0f} MB free (<{MEM_CRITICAL_THRESHOLD_MB} MB)")
        result["status"] = "MEM_CRITICAL"
        result["inference_ok"] = False
        # Still try inference to see if it crashes or works
        # But mark it as critical regardless

    # ── Step 6: Run 10-sample evaluation ─────────────────────────────────
    logger.info(f"  Step 3: Running 10-sample evaluation...")
    try:
        from evaluation.run_baseline import load_vqav2, evaluate_dataset, _vqa_accuracy
        samples = load_vqav2(n_samples=50)
        if samples:
            vqa_result = evaluate_dataset(
                model, processor, samples, family, "cuda",
                "vqav2", _vqa_accuracy,
            )
            result["inference_ok"] = True
            result["inference_latency_s"] = round(vqa_result.get("avg_latency_s", 0), 2)
            result["peak_memory_mb"] = vqa_result.get("peak_memory_mb", None)
            result["benchmarks"] = {"vqav2": vqa_result}
            logger.info(f"  Eval done: exact_match={vqa_result.get('exact_match', 'N/A')} "
                        f"contains={vqa_result.get('contains', 'N/A')} "
                        f"token_f1={vqa_result.get('token_f1', 'N/A')} "
                        f"lat={vqa_result.get('avg_latency_s', 'N/A'):.2f}s "
                        f"peak_mem={vqa_result.get('peak_memory_mb', 'N/A')}MB")

            if result["status"] is None:
                result["status"] = "RUNNABLE"
    except torch.cuda.OutOfMemoryError:
        logger.warning(f"  OOM during inference!")
        result["inference_ok"] = False
        if result["status"] is None:
            result["status"] = "MEM_CRITICAL"
        result["error"] = "OOM during inference"
        cleanup()
    except Exception as e:
        err_msg = str(e)[:200]
        logger.warning(f"  Inference error: {err_msg}")
        result["inference_ok"] = False
        if result["status"] is None:
            result["status"] = "ERROR"
        result["error"] = err_msg

    # ── Cleanup ────────────────────────────────────────────────────────
    logger.info(f"  Cleaning up...")
    del model, processor
    cleanup()
    time.sleep(1)  # let GPU memory settle
    cleanup()

    gpu_after_cleanup = get_used_gpu_mb()
    logger.info(f"  GPU after cleanup: {gpu_after_cleanup:.0f} MB used")
    logger.info(f"  RESULT: {result['status']}")

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Safe ceiling scan for all VLMs")
    parser.add_argument("--start", type=int, default=0, help="Start from model index")
    parser.add_argument("--end", type=int, default=len(ALL_MODELS), help="End at model index")
    parser.add_argument("--family", type=str, default=None, help="Only scan this family")
    parser.add_argument("--force", action="store_true", help="Re-run even if results exist")
    args = parser.parse_args()

    models_to_scan = ALL_MODELS[args.start:args.end]
    if args.family:
        models_to_scan = [(m, f) for m, f in models_to_scan if f == args.family]

    logger.info(f"Ceiling scan: {len(models_to_scan)} models")
    logger.info(f"GPU total: {(torch.cuda.mem_get_info(0)[1])/1e9:.2f} GB")
    logger.info(f"GPU free:  {get_free_gpu_mb()/1024:.2f} GB")

    all_results = []

    # Load existing results to allow resuming
    out_path = RESULTS_DIR / "ceiling_scan_results.json"
    if out_path.exists():
        with open(out_path) as f:
            all_results = json.load(f)
    done_ids = {r["model_id"] for r in all_results}

    for i, (model_id, family) in enumerate(models_to_scan):
        if model_id in done_ids and not args.force:
            logger.info(f"\n[{i+1}/{len(models_to_scan)}] {model_id} — already done, skipping")
            continue
        logger.info(f"\n[{i+1}/{len(models_to_scan)}] {model_id}")
        result = scan_model(model_id, family)
        # Replace if re-running with --force
        all_results = [r for r in all_results if r["model_id"] != model_id]
        all_results.append(result)

        # Save after each model (in case of crash)
        out_path = RESULTS_DIR / "ceiling_scan_results.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

        # Also save individual result
        safe_name = model_id.replace("/", "__")
        ind_path = RESULTS_DIR / f"{safe_name}.json"
        with open(ind_path, "w") as f:
            json.dump(result, f, indent=2)

    # ── Print summary ──────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("CEILING SCAN SUMMARY")
    logger.info(f"{'='*70}")

    current_family = None
    for r in all_results:
        if r["family"] != current_family:
            current_family = r["family"]
            logger.info(f"\n  Family: {current_family}")

        status = r["status"]
        model_short = r["model_id"].split("/")[-1]
        size = f"{r['model_size_mb']:.0f}MB" if r["model_size_mb"] else "?"
        gpu = f"{r['gpu_used_after_mb']:.0f}MB" if r["gpu_used_after_mb"] else "N/A"
        free = f"{r['gpu_free_after_mb']:.0f}MB" if r["gpu_free_after_mb"] else "N/A"
        lat = f"{r['inference_latency_s']:.1f}s" if r["inference_latency_s"] else "N/A"

        icon = {"RUNNABLE": "✓", "MEM_CRITICAL": "⚠", "CEILING": "✗", "ERROR": "!"}
        logger.info(f"    {icon.get(status, '?')} {model_short:40s} {status:14s} size={size:>8s} gpu={gpu:>8s} free={free:>8s} lat={lat}")

    logger.info(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
