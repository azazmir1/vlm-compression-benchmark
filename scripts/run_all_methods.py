"""
scripts/run_all_methods.py
===========================
Run all 14 compression methods on the 7 RUNNABLE models from ceiling scan.
Each method+model is run in a separate subprocess to avoid GPU memory leaks.

Methods:
  2. BnB INT4 (NF4)           - quantization
  3. Pruning 20%              - magnitude L1 unstructured
  4. Pruning 40%              - magnitude L1 unstructured
  5. Wanda 20%                - weight × activation pruning
  6. Wanda 40%                - weight × activation pruning
  7. AWQ INT4 (simulated)     - activation-aware quantize→dequantize
  8. GPTQ INT4 (simulated)    - Hessian-based quantize→dequantize
  9. SparseGPT 50%            - Hessian-compensated pruning
 10. AWP (sp50 + INT4)        - Wanda 50% + simulated INT4
 11. PACT (p30 + m20)         - visual token pruning + merging
 12. SVD-LLM (rank_ratio=0.5) - truncated SVD on MLP weights
 13. PALU (rank_ratio=0.25)   - low-rank SVD on K/V projections
 14. CASP                     - mixed-precision + QK low-rank
 15. SLIM (sp50 + r30)        - SVD + pruning + simulated INT4
"""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_BASE = PROJECT_ROOT / "results" / "paper"

# 7 runnable models from ceiling scan
MODELS = [
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    "HuggingFaceTB/SmolVLM-500M-Instruct",
    "OpenGVLab/InternVL2_5-1B",
    "LiquidAI/LFM2-VL-450M",
    "vikhyatk/moondream2",
    "apple/FastVLM-0.5B",
    "AIDC-AI/Ovis2-1B",
]

N_SAMPLES = 10

# Method definitions: (method_id, name, script, args_fn)
# args_fn takes model_id and returns list of args
#
# EXCLUDED (blocked on Jetson):
#   2  - BnB INT4: CUDA kernel crash (Error named symbol not found)
#   9  - SparseGPT: cusolverDnXsyevBatched_bufferSize missing
#   10 - AWP: Wanda works but BnB INT4 reload crashes
#   12 - SVD-LLM: torch.linalg.svd dlopen failure (cusolver)
SKIP_EXTRA = ["--skip_textvqa", "--skip_pope"]

METHODS = [
    (3,  "pruning_sp20", "compression/pruning/run_pruning.py",
     lambda m: ["--model_id", m, "--sparsity", "0.20", "--vqav2_n", str(N_SAMPLES), "--force"] + SKIP_EXTRA),

    (4,  "pruning_sp40", "compression/pruning/run_pruning.py",
     lambda m: ["--model_id", m, "--sparsity", "0.40", "--vqav2_n", str(N_SAMPLES), "--force"] + SKIP_EXTRA),

    (5,  "wanda_sp20",   "compression/pruning/run_wanda.py",
     lambda m: ["--model_id", m, "--sparsity", "0.20", "--vqav2_n", str(N_SAMPLES), "--force"] + SKIP_EXTRA),

    (6,  "wanda_sp40",   "compression/pruning/run_wanda.py",
     lambda m: ["--model_id", m, "--sparsity", "0.40", "--vqav2_n", str(N_SAMPLES), "--force"] + SKIP_EXTRA),

    (7,  "awq_sim",      "compression/awq_gptq/run_awq_gptq.py",
     lambda m: ["--model_id", m, "--method", "awq", "--vqav2_n", str(N_SAMPLES), "--force"] + SKIP_EXTRA),

    (8,  "gptq_sim",     "compression/awq_gptq/run_awq_gptq.py",
     lambda m: ["--model_id", m, "--method", "gptq", "--vqav2_n", str(N_SAMPLES), "--force"] + SKIP_EXTRA),

    (11, "pact_p30_m20", "compression/pact/run_pact.py",
     lambda m: ["--model_id", m, "--prune_ratio", "0.30", "--vqav2_n", str(N_SAMPLES), "--force"] + SKIP_EXTRA),

    (13, "palu_rr25",    "compression/palu/run_palu.py",
     lambda m: ["--model_id", m, "--rank_ratio", "0.25", "--vqav2_n", str(N_SAMPLES), "--force"] + SKIP_EXTRA),

    (14, "casp",         "compression/casp_slim/run_casp_slim.py",
     lambda m: ["--model_id", m, "--method", "casp", "--vqav2_n", str(N_SAMPLES), "--force"] + SKIP_EXTRA),

    (15, "slim_sp50_r30", "compression/casp_slim/run_casp_slim.py",
     lambda m: ["--model_id", m, "--method", "slim", "--sparsity", "0.50",
                "--rank_ratio", "0.30", "--vqav2_n", str(N_SAMPLES), "--force"] + SKIP_EXTRA),
]


def run_experiment(method_id, method_name, script, args, model_id, log_dir):
    """Run one experiment in a subprocess. Returns (status, result_dict)."""
    model_short = model_id.split("/")[-1]
    log_file = log_dir / f"m{method_id:02d}_{method_name}__{model_short}.log"

    cmd = [sys.executable, str(PROJECT_ROOT / script)] + args
    logger.info(f"  [{method_id:2d}] {method_name} on {model_short}")
    logger.info(f"       cmd: {' '.join(cmd[-6:])}")

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=2400,  # 40 min max per experiment
        )
        elapsed = time.time() - start
        # Save log
        with open(log_file, "w") as f:
            f.write(f"=== CMD: {' '.join(cmd)}\n")
            f.write(f"=== EXIT CODE: {result.returncode}\n")
            f.write(f"=== ELAPSED: {elapsed:.1f}s\n\n")
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)

        if result.returncode == 0:
            logger.info(f"       PASS ({elapsed:.0f}s)")
            return "PASS", elapsed
        else:
            # Extract error
            err = result.stderr.strip().split("\n")[-1][:120] if result.stderr else "unknown"
            logger.warning(f"       FAIL ({elapsed:.0f}s): {err}")
            return "FAIL", elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        logger.warning(f"       TIMEOUT ({elapsed:.0f}s)")
        with open(log_file, "w") as f:
            f.write(f"=== CMD: {' '.join(cmd)}\n")
            f.write(f"=== TIMEOUT after {elapsed:.0f}s\n")
        return "TIMEOUT", elapsed
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"       ERROR: {e}")
        return "ERROR", elapsed


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Run only this model (substring match)")
    parser.add_argument("--method", type=int, default=None,
                        help="Run only this method number (2-15)")
    parser.add_argument("--start_method", type=int, default=2,
                        help="Start from this method number")
    parser.add_argument("--start_model", type=int, default=0,
                        help="Start from this model index")
    args = parser.parse_args()

    log_dir = RESULTS_BASE / "step2_compression_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Filter models
    models = MODELS
    if args.model:
        models = [m for m in MODELS if args.model.lower() in m.lower()]
        if not models:
            logger.error(f"No models matching '{args.model}'")
            return

    # Filter methods
    methods = METHODS
    if args.method is not None:
        methods = [m for m in METHODS if m[0] == args.method]
    else:
        methods = [m for m in METHODS if m[0] >= args.start_method]

    total = len(models) * len(methods)
    logger.info(f"Running {len(methods)} methods × {len(models)} models = {total} experiments")
    logger.info(f"Results: {log_dir}")
    logger.info("")

    # Summary tracking
    summary = []
    done = 0

    for mi, model_id in enumerate(models):
        if mi < args.start_model:
            continue
        model_short = model_id.split("/")[-1]
        logger.info(f"{'='*60}")
        logger.info(f"MODEL [{mi+1}/{len(models)}]: {model_short}")
        logger.info(f"{'='*60}")

        for method_id, method_name, script, args_fn in methods:
            done += 1
            method_args = args_fn(model_id)
            status, elapsed = run_experiment(
                method_id, method_name, script, method_args,
                model_id, log_dir,
            )
            summary.append({
                "model": model_short,
                "model_id": model_id,
                "method_id": method_id,
                "method": method_name,
                "status": status,
                "elapsed_s": round(elapsed, 1),
            })

        logger.info("")

    # Print summary
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    passed = [s for s in summary if s["status"] == "PASS"]
    failed = [s for s in summary if s["status"] != "PASS"]
    logger.info(f"  PASS: {len(passed)}/{len(summary)}")
    logger.info(f"  FAIL: {len(failed)}/{len(summary)}")
    if failed:
        logger.info("  Failed experiments:")
        for s in failed:
            logger.info(f"    [{s['method_id']:2d}] {s['method']:<20} {s['model']:<30} {s['status']}")

    # Save summary JSON
    summary_file = log_dir / "run_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()
