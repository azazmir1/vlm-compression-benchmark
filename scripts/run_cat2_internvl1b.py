#!/usr/bin/env python3
"""
scripts/run_cat2_internvl1b.py
===============================
Master runner: Apply ALL Category 2 compression methods to InternVL2.5-1B
on Jetson Orin Nano 8GB and evaluate with multi-metric scoring on VQAv2.

Category 2 methods (make loaded models more usable):
  1. Magnitude Pruning  — sp20, sp40
  2. Wanda Pruning      — sp20, sp40
  3. PALU (KV-cache)    — rank_ratio 0.25
  4. PACT (token comp)  — prune 0.30, merge 0.20

Each method runs in its own subprocess for OOM safety.
Results saved to: results/jetson/cat2_internvl1b/
Full log saved to: results/jetson/cat2_internvl1b/run.log

Metrics per sample: exact_match, contains, token_f1, bleu, rouge_l
Plus: latency, peak memory, throughput, power, GPU utilization

Usage:
    python3 scripts/run_cat2_internvl1b.py
    python3 scripts/run_cat2_internvl1b.py --n_samples 50   # quick test
    python3 scripts/run_cat2_internvl1b.py --force           # re-run all
"""

import argparse
import json
import logging
import multiprocessing
import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results" / "jetson" / "cat2_internvl1b"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "OpenGVLab/InternVL2_5-1B"
FAMILY = "internvl25"
PARAM_M = 938.2

# ── Logging setup ────────────────────────────────────────────────────────────
log_path = RESULTS_DIR / "run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_path, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("cat2_runner")

# Use 'spawn' so child gets fresh CUDA context
_mp_ctx = multiprocessing.get_context("spawn")

# ── Method definitions ───────────────────────────────────────────────────────

METHODS = [
    {
        "name": "baseline_fp16",
        "tag": "baseline_fp16",
        "description": "FP16 baseline (no compression) — reference point",
    },
    {
        "name": "magnitude_pruning_sp20",
        "tag": "magnitude_sp20",
        "description": "Magnitude L1 unstructured pruning at 20% sparsity",
    },
    {
        "name": "magnitude_pruning_sp40",
        "tag": "magnitude_sp40",
        "description": "Magnitude L1 unstructured pruning at 40% sparsity",
    },
    {
        "name": "wanda_sp20",
        "tag": "wanda_sp20",
        "description": "Wanda (activation-aware) pruning at 20% sparsity",
    },
    {
        "name": "wanda_sp40",
        "tag": "wanda_sp40",
        "description": "Wanda (activation-aware) pruning at 40% sparsity",
    },
    {
        "name": "palu_r25",
        "tag": "palu_r25",
        "description": "PALU KV-cache low-rank compression (rank_ratio=0.25)",
    },
    {
        "name": "pact_p30_m20",
        "tag": "pact_p30_m20",
        "description": "PACT visual token pruning (30%) + merging (20%)",
    },
]


# ═════════════════════════════════════════════════════════════════════════════
#  SUBPROCESS ENTRY POINTS — each method runs in isolation
# ═════════════════════════════════════════════════════════════════════════════

def _child_baseline(n_samples, result_path):
    """FP16 baseline — load, evaluate, save."""
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_root))

    import json, time, torch
    from models.model_loader import load_model, unload_model
    from evaluation.run_baseline import (
        load_vqav2, evaluate_dataset, _vqa_accuracy,
    )

    try:
        from jetson.safety import make_self_oom_preferred
        make_self_oom_preferred()
    except Exception:
        pass

    logger_c = _setup_child_logger()
    logger_c.info(f"[baseline_fp16] Loading {MODEL_ID} in FP16...")

    model, processor, meta = load_model(MODEL_ID, quant="fp16")
    device = str(next(model.parameters()).device)
    num_params = sum(p.numel() for p in model.parameters())
    logger_c.info(f"[baseline_fp16] Loaded: {num_params/1e6:.1f}M params, "
                  f"mem_delta={meta.gpu_mem_delta_mb:.0f}MB")

    samples = load_vqav2(n_samples=n_samples)
    bench = evaluate_dataset(model, processor, samples, FAMILY, device,
                             "VQAv2", _vqa_accuracy)

    result = {
        "model_id": MODEL_ID, "family": FAMILY,
        "method": "baseline_fp16", "quant": "fp16",
        "num_params_M": round(num_params / 1e6, 1),
        "gpu_mem_load_mb": meta.gpu_mem_delta_mb,
        "benchmarks": {"vqav2": bench},
        "status": "PASS",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    unload_model(model)


def _child_magnitude_pruning(sparsity, n_samples, result_path):
    """Magnitude pruning subprocess."""
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_root))

    import json, time, torch
    from models.model_loader import load_model, unload_model
    from evaluation.run_baseline import (
        load_vqav2, evaluate_dataset, _vqa_accuracy,
    )
    from compression.pruning.run_pruning import apply_magnitude_pruning, measure_sparsity

    try:
        from jetson.safety import make_self_oom_preferred
        make_self_oom_preferred()
    except Exception:
        pass

    tag = f"magnitude_sp{int(sparsity*100)}"
    logger_c = _setup_child_logger()
    logger_c.info(f"[{tag}] Loading {MODEL_ID} in FP16...")

    model, processor, meta = load_model(MODEL_ID, quant="fp16")
    device = str(next(model.parameters()).device)
    num_params = sum(p.numel() for p in model.parameters())

    logger_c.info(f"[{tag}] Applying magnitude pruning at sparsity={sparsity}...")
    prune_stats = apply_magnitude_pruning(model, sparsity)
    actual_sp = measure_sparsity(model)
    logger_c.info(f"[{tag}] Post-pruning sparsity: {actual_sp:.4f}")

    # Free pruning temporaries
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    samples = load_vqav2(n_samples=n_samples)
    bench = evaluate_dataset(model, processor, samples, FAMILY, device,
                             "VQAv2", _vqa_accuracy)

    result = {
        "model_id": MODEL_ID, "family": FAMILY,
        "method": tag, "quant": "fp16",
        "pruning_method": "magnitude_l1_unstructured",
        "target_sparsity": sparsity,
        "actual_sparsity": prune_stats["actual_sparsity"],
        "pruned_layers": prune_stats["pruned_layers"],
        "num_params_M": round(num_params / 1e6, 1),
        "gpu_mem_load_mb": meta.gpu_mem_delta_mb,
        "benchmarks": {"vqav2": bench},
        "status": "PASS",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    unload_model(model)


def _child_wanda(sparsity, n_calib, n_samples, result_path):
    """Wanda pruning subprocess."""
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_root))

    import json, time, torch
    from models.model_loader import load_model, unload_model
    from evaluation.run_baseline import (
        load_vqav2, run_inference, evaluate_dataset, _vqa_accuracy,
    )
    from compression.pruning.run_wanda import (
        ActivationCollector, apply_wanda_pruning, measure_sparsity,
    )

    try:
        from jetson.safety import make_self_oom_preferred
        make_self_oom_preferred()
    except Exception:
        pass

    tag = f"wanda_sp{int(sparsity*100)}"
    logger_c = _setup_child_logger()
    logger_c.info(f"[{tag}] Loading {MODEL_ID} in FP16...")

    model, processor, meta = load_model(MODEL_ID, quant="fp16")
    device = str(next(model.parameters()).device)
    num_params = sum(p.numel() for p in model.parameters())

    # Calibration pass
    logger_c.info(f"[{tag}] Calibration pass ({n_calib} samples)...")
    calib_samples = load_vqav2(n_samples=n_calib)
    collector = ActivationCollector(model)
    skipped = 0
    for sample in calib_samples:
        try:
            _ = run_inference(model, processor, sample, FAMILY, device)
        except Exception:
            skipped += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if skipped:
        logger_c.warning(f"[{tag}] Calibration: skipped {skipped}/{n_calib}")
    input_norms = collector.get_input_norms()
    collector.remove_hooks()
    logger_c.info(f"[{tag}] Collected norms for {len(input_norms)} layers")

    # Apply Wanda pruning
    logger_c.info(f"[{tag}] Applying Wanda pruning at sparsity={sparsity}...")
    prune_stats = apply_wanda_pruning(model, input_norms, sparsity)
    actual_sp = measure_sparsity(model)
    logger_c.info(f"[{tag}] Post-pruning sparsity: {actual_sp:.4f}")

    # Free calibration data
    del calib_samples, input_norms
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    samples = load_vqav2(n_samples=n_samples)
    bench = evaluate_dataset(model, processor, samples, FAMILY, device,
                             "VQAv2", _vqa_accuracy)

    result = {
        "model_id": MODEL_ID, "family": FAMILY,
        "method": tag, "quant": "fp16",
        "pruning_method": "wanda",
        "target_sparsity": sparsity,
        "actual_sparsity": prune_stats["actual_sparsity"],
        "pruned_layers": prune_stats["pruned_layers"],
        "n_calib_samples": n_calib,
        "num_params_M": round(num_params / 1e6, 1),
        "gpu_mem_load_mb": meta.gpu_mem_delta_mb,
        "benchmarks": {"vqav2": bench},
        "status": "PASS",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    unload_model(model)


def _child_palu(rank_ratio, n_samples, result_path):
    """PALU KV-cache compression subprocess."""
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_root))

    import json, time, torch
    from models.model_loader import load_model, unload_model
    from evaluation.run_baseline import (
        load_vqav2, evaluate_dataset, _vqa_accuracy,
    )
    from compression.palu.run_palu import apply_palu_compression

    try:
        from jetson.safety import make_self_oom_preferred
        make_self_oom_preferred()
    except Exception:
        pass

    tag = f"palu_r{int(rank_ratio*100)}"
    logger_c = _setup_child_logger()
    logger_c.info(f"[{tag}] Loading {MODEL_ID} in FP16...")

    model, processor, meta = load_model(MODEL_ID, quant="fp16")
    device = str(next(model.parameters()).device)
    num_params_before = sum(p.numel() for p in model.parameters())

    logger_c.info(f"[{tag}] Applying PALU with rank_ratio={rank_ratio}...")
    palu_stats = apply_palu_compression(model, rank_ratio)
    num_params_after = sum(p.numel() for p in model.parameters())

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    samples = load_vqav2(n_samples=n_samples)
    bench = evaluate_dataset(model, processor, samples, FAMILY, device,
                             "VQAv2", _vqa_accuracy)

    result = {
        "model_id": MODEL_ID, "family": FAMILY,
        "method": tag, "quant": "fp16",
        **palu_stats,
        "num_params_before_M": round(num_params_before / 1e6, 1),
        "num_params_after_M": round(num_params_after / 1e6, 1),
        "gpu_mem_load_mb": meta.gpu_mem_delta_mb,
        "benchmarks": {"vqav2": bench},
        "status": "PASS",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    unload_model(model)


def _child_pact(prune_ratio, merge_ratio, n_samples, result_path):
    """PACT visual token compression subprocess."""
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_root))

    import json, time, torch
    from models.model_loader import load_model, unload_model
    from evaluation.run_baseline import (
        load_vqav2, _vqa_accuracy,
    )
    from compression.pact.run_pact import PACTTokenCompressor, evaluate_with_pact

    try:
        from jetson.safety import make_self_oom_preferred
        make_self_oom_preferred()
    except Exception:
        pass

    tag = f"pact_p{int(prune_ratio*100)}_m{int(merge_ratio*100)}"
    logger_c = _setup_child_logger()
    logger_c.info(f"[{tag}] Loading {MODEL_ID} in FP16...")

    model, processor, meta = load_model(MODEL_ID, quant="fp16")
    device = str(next(model.parameters()).device)
    num_params = sum(p.numel() for p in model.parameters())

    logger_c.info(f"[{tag}] Installing PACT (prune={prune_ratio}, merge={merge_ratio})...")
    compressor = PACTTokenCompressor(model, FAMILY,
                                     prune_ratio=prune_ratio,
                                     merge_ratio=merge_ratio,
                                     target_layer=1)

    samples = load_vqav2(n_samples=n_samples)
    bench = evaluate_with_pact(model, processor, samples, FAMILY, device,
                               "VQAv2", _vqa_accuracy)

    pact_stats = compressor.get_stats()
    compressor.remove_hook()

    result = {
        "model_id": MODEL_ID, "family": FAMILY,
        "method": tag, "quant": "fp16",
        "prune_ratio": prune_ratio,
        "merge_ratio": merge_ratio,
        "total_token_reduction": round(1 - (1 - prune_ratio) * (1 - merge_ratio), 4),
        "pact_stats": pact_stats,
        "num_params_M": round(num_params / 1e6, 1),
        "gpu_mem_load_mb": meta.gpu_mem_delta_mb,
        "benchmarks": {"vqav2": bench},
        "status": "PASS",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    unload_model(model)


# ── Child logger helper ──────────────────────────────────────────────────────

def _setup_child_logger():
    """Set up logging in child process."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    return logging.getLogger("cat2_child")


# ═════════════════════════════════════════════════════════════════════════════
#  MASTER RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def _memory_cleanup():
    """Aggressive memory reclaim between methods."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        os.system("sync")
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
    except (PermissionError, IOError):
        pass
    # Wait for memory to settle
    time.sleep(5)


def run_method_isolated(method_name, target_fn, args, timeout=900):
    """Run a method in a subprocess, return result dict or failure."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                     delete=False, dir="/tmp") as tf:
        result_path = tf.name

    full_args = args + (result_path,)
    proc = _mp_ctx.Process(target=target_fn, args=full_args)

    logger.info(f"  Spawning subprocess for {method_name} (timeout={timeout}s)...")
    t0 = time.time()
    proc.start()
    proc.join(timeout=timeout)
    elapsed = time.time() - t0

    if proc.is_alive():
        logger.warning(f"  [{method_name}] Timeout ({timeout}s) — killing subprocess")
        proc.kill()
        proc.join(5)

    # Read result
    try:
        with open(result_path) as f:
            result = json.load(f)
        result["wall_time_s"] = round(elapsed, 1)
        logger.info(f"  [{method_name}] Completed in {elapsed:.0f}s — status={result.get('status', '?')}")
        return result
    except (FileNotFoundError, json.JSONDecodeError):
        exit_code = proc.exitcode
        if exit_code is not None and exit_code < 0:
            err_msg = f"Killed by signal {-exit_code} (likely OOM-killer)"
        else:
            err_msg = f"Exited with code {exit_code}, no result written"
        logger.error(f"  [{method_name}] FAILED: {err_msg}")
        return {
            "model_id": MODEL_ID, "family": FAMILY,
            "method": method_name,
            "status": "FAILED",
            "error": err_msg,
            "wall_time_s": round(elapsed, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    finally:
        try:
            os.unlink(result_path)
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Run all Category 2 methods on InternVL2.5-1B (Jetson)")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="VQAv2 eval samples per method (default 100)")
    parser.add_argument("--n_calib", type=int, default=32,
                        help="Wanda calibration samples (default 32)")
    parser.add_argument("--timeout", type=int, default=900,
                        help="Subprocess timeout in seconds (default 900)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if results exist")
    parser.add_argument("--methods", type=str, default=None,
                        help="Comma-separated method names to run (default: all)")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Category 2 Compression Benchmark — InternVL2.5-1B on Jetson")
    logger.info("=" * 70)
    logger.info(f"Model:      {MODEL_ID}")
    logger.info(f"Params:     {PARAM_M}M")
    logger.info(f"Eval:       VQAv2 ({args.n_samples} samples)")
    logger.info(f"Metrics:    exact_match, contains, token_f1, bleu, rouge_l")
    logger.info(f"Calib:      {args.n_calib} samples (Wanda)")
    logger.info(f"Timeout:    {args.timeout}s per method")
    logger.info(f"Results:    {RESULTS_DIR}")
    logger.info(f"Log:        {log_path}")
    logger.info("=" * 70)

    # Filter methods if requested
    if args.methods:
        selected = set(args.methods.split(","))
        methods = [m for m in METHODS if m["name"] in selected]
    else:
        methods = METHODS

    summary = {
        "model_id": MODEL_ID,
        "family": FAMILY,
        "n_samples": args.n_samples,
        "n_calib": args.n_calib,
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "methods": {},
    }

    total_start = time.time()

    for i, method in enumerate(methods, 1):
        name = method["name"]
        tag = method["tag"]
        out_path = RESULTS_DIR / f"{tag}.json"

        logger.info("")
        logger.info(f"{'─' * 60}")
        logger.info(f"[{i}/{len(methods)}] {name}: {method['description']}")
        logger.info(f"{'─' * 60}")

        # Check cache
        if out_path.exists() and not args.force:
            try:
                with open(out_path) as f:
                    cached = json.load(f)
                logger.info(f"  [CACHED] {out_path.name} — skipping (use --force to re-run)")
                summary["methods"][name] = {
                    "status": cached.get("status", "?"),
                    "result_file": str(out_path),
                    "cached": True,
                }
                continue
            except (json.JSONDecodeError, KeyError):
                pass

        # Dispatch to correct subprocess
        if name == "baseline_fp16":
            result = run_method_isolated(
                name, _child_baseline,
                (args.n_samples,), args.timeout)

        elif name.startswith("magnitude_pruning"):
            sp = 0.20 if "sp20" in name else 0.40
            result = run_method_isolated(
                name, _child_magnitude_pruning,
                (sp, args.n_samples), args.timeout)

        elif name.startswith("wanda"):
            sp = 0.20 if "sp20" in name else 0.40
            result = run_method_isolated(
                name, _child_wanda,
                (sp, args.n_calib, args.n_samples), args.timeout)

        elif name == "palu_r25":
            result = run_method_isolated(
                name, _child_palu,
                (0.25, args.n_samples), args.timeout)

        elif name == "pact_p30_m20":
            result = run_method_isolated(
                name, _child_pact,
                (0.30, 0.20, args.n_samples), args.timeout)

        else:
            logger.error(f"  Unknown method: {name}")
            continue

        # Save individual result
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"  Result saved to {out_path}")

        # Extract key metrics for summary
        bench = result.get("benchmarks", {}).get("vqav2", {})
        metrics = bench.get("metrics", {})
        summary["methods"][name] = {
            "status": result.get("status", "?"),
            "result_file": str(out_path),
            "wall_time_s": result.get("wall_time_s", 0),
            "exact_match": metrics.get("exact_match", bench.get("accuracy", "?")),
            "contains": metrics.get("contains", "?"),
            "token_f1": metrics.get("token_f1", "?"),
            "bleu": metrics.get("bleu", "?"),
            "rouge_l": metrics.get("rouge_l", "?"),
            "avg_latency_s": bench.get("avg_latency_s", "?"),
            "peak_memory_mb": bench.get("peak_memory_mb", "?"),
            "throughput_sps": bench.get("throughput_sps", "?"),
        }

        # Memory cleanup between methods
        logger.info("  Cleaning up memory before next method...")
        _memory_cleanup()

    total_elapsed = time.time() - total_start
    summary["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    summary["total_time_s"] = round(total_elapsed, 1)

    # Save summary
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print final report
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL REPORT — Category 2 Methods on InternVL2.5-1B")
    logger.info("=" * 70)
    logger.info(f"Total time: {total_elapsed/60:.1f} minutes")
    logger.info("")
    logger.info(f"{'Method':<25s} {'Status':<10s} {'ExMatch':<9s} {'Contains':<9s} "
                f"{'F1':<9s} {'BLEU':<9s} {'ROUGE-L':<9s} {'Lat(s)':<8s} {'Mem(MB)':<8s}")
    logger.info("-" * 96)

    for name, m in summary["methods"].items():
        status = str(m.get("status", "?"))[:8]
        em = m.get("exact_match", "?")
        co = m.get("contains", "?")
        f1 = m.get("token_f1", "?")
        bl = m.get("bleu", "?")
        rl = m.get("rouge_l", "?")
        lat = m.get("avg_latency_s", "?")
        mem = m.get("peak_memory_mb", "?")

        em_s = f"{em:.4f}" if isinstance(em, (int, float)) else str(em)
        co_s = f"{co:.4f}" if isinstance(co, (int, float)) else str(co)
        f1_s = f"{f1:.4f}" if isinstance(f1, (int, float)) else str(f1)
        bl_s = f"{bl:.4f}" if isinstance(bl, (int, float)) else str(bl)
        rl_s = f"{rl:.4f}" if isinstance(rl, (int, float)) else str(rl)
        lat_s = f"{lat:.2f}" if isinstance(lat, (int, float)) else str(lat)
        mem_s = f"{mem:.0f}" if isinstance(mem, (int, float)) else str(mem)

        logger.info(f"{name:<25s} {status:<10s} {em_s:<9s} {co_s:<9s} "
                    f"{f1_s:<9s} {bl_s:<9s} {rl_s:<9s} {lat_s:<8s} {mem_s:<8s}")

    logger.info("")
    logger.info(f"Summary saved to {summary_path}")
    logger.info(f"Full log at {log_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
