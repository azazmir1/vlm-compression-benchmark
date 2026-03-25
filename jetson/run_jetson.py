#!/usr/bin/env python3
"""
jetson/run_jetson.py
====================
Jetson Orin Nano 8 GB — VLM ceiling-detection and compression benchmark.

Workflow
--------
Phase 1  **Baseline scan** — try every model at FP16, smallest→largest per
         family.  Record PASS / OOM / TIMEOUT / TOO_SLOW.  When a model
         fails, mark the family's FP16 ceiling.

Phase 2  **Compression recovery** — for models that failed at FP16, retry
         with INT8 (BitsAndBytes), then INT4 (BnB NF4).  This tells us
         whether quantisation pushes the ceiling higher.

Phase 3  **Pruning pass** — for models that already pass at FP16, apply
         magnitude pruning (20 % / 40 %) and re-evaluate to measure the
         latency impact on Jetson.

Phase 4  **Report** — write ``results/jetson/ceiling_report.json`` with
         per-family ceilings and a full per-model breakdown.

Usage
-----
    # Full benchmark (all families, all phases)
    python3 jetson/run_jetson.py

    # Specific families only
    python3 jetson/run_jetson.py --families smolvlm,lfm2vl

    # Quick ceiling scan — skip compression & pruning
    python3 jetson/run_jetson.py --scan_only

    # Fewer evaluation samples (faster but noisier accuracy)
    python3 jetson/run_jetson.py --n_samples 50
"""

import argparse
import json
import logging
import multiprocessing
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

# ── Ensure project root is importable ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from jetson.safety import (
    Status,
    CACHEABLE_STATUSES,
    CRITICAL_FREE_MB,
    INFERENCE_TIMEOUT_S,
    WARMUP_TIMEOUT_S,
    LATENCY_CEILING_S,
    LATENCY_CHECK_WINDOW,
    DEPLOYABLE_LATENCY_S,
    get_available_memory_mb,
    get_gpu_memory_used_mb,
    is_memory_critical,
    run_with_timeout,
    safe_load_model,
    safe_unload,
    _emergency_cleanup,
    make_self_oom_preferred,
    MemoryWatchdog,
)
from evaluation.run_baseline import (
    load_vqav2,
    run_inference,
    _vqa_accuracy,
)
from profiling.gpu_profiler import GPUProfiler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jetson")

# ── Results directories ───────────────────────────────────────────────────────
RESULTS_BASE = PROJECT_ROOT / "results" / "jetson"
RESULTS_BASELINE  = RESULTS_BASE / "baseline"
RESULTS_PTQ       = RESULTS_BASE / "ptq"
RESULTS_PRUNING   = RESULTS_BASE / "pruning"
RESULTS_AWQ       = RESULTS_BASE / "awq"
RESULTS_GPTQ      = RESULTS_BASE / "gptq"
RESULTS_SPARSEGPT = RESULTS_BASE / "sparsegpt"
RESULTS_AWP       = RESULTS_BASE / "awp"
RESULTS_SVDLLM    = RESULTS_BASE / "svd_llm"
for d in (RESULTS_BASELINE, RESULTS_PTQ, RESULTS_PRUNING,
          RESULTS_AWQ, RESULTS_GPTQ, RESULTS_SPARSEGPT,
          RESULTS_AWP, RESULTS_SVDLLM):
    d.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
#  SUBPROCESS ISOLATION — survive OOM-kills and CUDA crashes
# ═════════════════════════════════════════════════════════════════════════════

# Use 'spawn' so child gets a fresh CUDA context (fork fails after CUDA init)
_mp_ctx = multiprocessing.get_context("spawn")

# ── Memory baseline & full-reclaim logic ─────────────────────────────────────
# Captured once at startup (after dataset is loaded, before any model).
# After each model, we wait until memory stabilises near the baseline.
_baseline_memory_mb: float = 0.0       # set in main()
_RECLAIM_POLL_S = 2                    # poll interval while waiting
_RECLAIM_TIMEOUT_S = 60                # give up after this many seconds
_STABLE_READINGS = 3                   # memory must be stable for this many consecutive readings
_STABLE_DELTA_MB = 50                  # readings within this range count as "stable"


def set_memory_baseline():
    """Capture the available-memory baseline (call once before any model)."""
    global _baseline_memory_mb
    _baseline_memory_mb = get_available_memory_mb()
    logger.info(f"Memory baseline captured: {_baseline_memory_mb:.0f} MB available")


def wait_for_full_memory_reclaim():
    """Block until memory stabilises (stops changing), meaning reclaim is done.

    Strategy: aggressive cleanup, then poll until readings are stable
    (3 consecutive readings within 50MB of each other).  This avoids
    hardcoding a ratio that breaks when the subprocess held extra buffers.
    """
    # Aggressive cleanup
    _emergency_cleanup()
    if torch.cuda.is_available():
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    # Ask the kernel to drop filesystem caches (helps on Jetson unified mem)
    try:
        os.system("sync")
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
    except (PermissionError, IOError):
        pass

    # Wait for memory to stabilise
    deadline = time.time() + _RECLAIM_TIMEOUT_S
    readings: list[float] = []
    while time.time() < deadline:
        _emergency_cleanup()
        avail = get_available_memory_mb()
        readings.append(avail)

        # Check if last N readings are stable
        if len(readings) >= _STABLE_READINGS:
            recent = readings[-_STABLE_READINGS:]
            if max(recent) - min(recent) < _STABLE_DELTA_MB:
                final = recent[-1]
                logger.info(
                    f"  Memory stabilised: {final:.0f} MB available "
                    f"(baseline: {_baseline_memory_mb:.0f} MB)")
                if final < CRITICAL_FREE_MB:
                    logger.warning(
                        f"  WARNING: Stable memory ({final:.0f} MB) is below "
                        f"critical threshold ({CRITICAL_FREE_MB} MB)!")
                return

        time.sleep(_RECLAIM_POLL_S)

    final = get_available_memory_mb()
    logger.warning(
        f"  Memory reclaim timeout ({_RECLAIM_TIMEOUT_S}s): "
        f"{final:.0f} MB free (baseline: {_baseline_memory_mb:.0f} MB). "
        f"Proceeding anyway — watchdog will protect.")


def _subprocess_trial_entry(model_id, family, param_M, precision,
                            n_samples, results_dir_str, force, result_path):
    """Entry point for spawned child — imports everything fresh."""
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_root))

    from jetson.safety import make_self_oom_preferred
    make_self_oom_preferred()

    from jetson.run_jetson import run_trial, _fail_result
    from jetson.safety import Status
    from evaluation.run_baseline import load_vqav2

    try:
        samples = load_vqav2(n_samples=n_samples)
        result = run_trial(model_id, family, param_M, precision,
                           samples, Path(results_dir_str), force)
    except Exception:
        import traceback, time
        result = {
            "model_id": model_id, "family": family,
            "param_M": param_M, "precision": precision,
            "status": Status.ERROR,
            "metrics": _fail_result(Status.ERROR, 0, traceback.format_exc()),
            "device": "jetson_orin_nano_8gb",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    import json
    with open(result_path, "w") as f:
        json.dump(result, f, default=str)


def run_trial_isolated(model_id, family, param_M, precision, samples,
                       results_dir, force=False, timeout=600):
    """Run a model trial in a spawned subprocess for OOM/crash isolation.

    If the child is killed (OOM-killer, CUDA crash), the parent logs the
    failure and returns a synthetic result dict so the benchmark continues.
    """
    # Check cache — every result is a valid data point.  Use --force to
    # re-run a model if conditions have changed.
    safe_name = model_id.replace("/", "__")
    tag = safe_name if precision == "fp16" else f"{safe_name}__{precision}__bnb"
    out_path = results_dir / f"{tag}.json"
    if out_path.exists() and not force:
        try:
            with open(out_path) as f:
                cached = json.load(f)
            if cached.get("status"):
                logger.info(f"  [CACHED] {out_path.name} ({cached['status']})")
                return cached
        except (json.JSONDecodeError, KeyError):
            pass

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                     delete=False, dir="/tmp") as tf:
        result_path = tf.name

    # Pass only picklable primitives — child reloads dataset itself
    proc = _mp_ctx.Process(
        target=_subprocess_trial_entry,
        args=(model_id, family, param_M, precision,
              len(samples), str(results_dir), force, result_path),
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        logger.warning(f"  Subprocess timed out ({timeout}s) — killing")
        proc.kill()
        proc.join(5)

    # Read result from temp file
    try:
        with open(result_path) as f:
            result = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Child was killed before writing (OOM-killer, segfault, etc.)
        exit_code = proc.exitcode
        if exit_code is not None and exit_code < 0:
            status = Status.OOM_LOAD
            err_msg = f"Child killed by signal {-exit_code} (likely OOM-killer)"
        else:
            status = Status.ERROR
            err_msg = f"Child exited with code {exit_code}, no result written"
        logger.warning(f"  {status}: {err_msg}")
        result = {
            "model_id": model_id, "family": family,
            "param_M": param_M, "precision": precision,
            "status": status,
            "metrics": _fail_result(status, len(samples), err_msg),
            "device": "jetson_orin_nano_8gb",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        # Save killed-child result too — it's a real data point
        _save_result(out_path, result)
    finally:
        try:
            os.unlink(result_path)
        except OSError:
            pass

    # Wait for the OS to reclaim the dead child's GPU/unified memory,
    # then verify it actually came back before starting the next model.
    wait_for_full_memory_reclaim()

    return result


def _subprocess_pruning_entry(model_id, family, param_M, sparsity,
                              n_samples, force, result_path):
    """Entry point for spawned pruning child — imports everything fresh."""
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_root))

    from jetson.safety import make_self_oom_preferred
    make_self_oom_preferred()

    from jetson.run_jetson import run_pruning_trial, _fail_result
    from jetson.safety import Status
    from evaluation.run_baseline import load_vqav2

    try:
        samples = load_vqav2(n_samples=n_samples)
        result = run_pruning_trial(model_id, family, param_M, sparsity,
                                   samples, force)
    except Exception:
        import traceback, time
        result = {
            "model_id": model_id, "family": family,
            "param_M": param_M, "precision": "fp16",
            "method": "magnitude_l1_unstructured",
            "sparsity": sparsity,
            "status": Status.ERROR,
            "metrics": _fail_result(Status.ERROR, 0, traceback.format_exc()),
            "device": "jetson_orin_nano_8gb",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    import json
    with open(result_path, "w") as f:
        json.dump(result, f, default=str)


def run_pruning_trial_isolated(model_id, family, param_M, sparsity, samples,
                               force=False, timeout=600):
    """Run a pruning trial in a spawned subprocess for OOM/crash isolation."""
    safe_name = model_id.replace("/", "__")
    tag = f"{safe_name}__sp{int(sparsity * 100)}"
    out_path = RESULTS_PRUNING / f"{tag}.json"
    if out_path.exists() and not force:
        try:
            with open(out_path) as f:
                cached = json.load(f)
            if cached.get("status"):
                logger.info(f"  [CACHED] {out_path.name} ({cached['status']})")
                return cached
        except (json.JSONDecodeError, KeyError):
            pass

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                     delete=False, dir="/tmp") as tf:
        result_path = tf.name

    proc = _mp_ctx.Process(
        target=_subprocess_pruning_entry,
        args=(model_id, family, param_M, sparsity,
              len(samples), force, result_path),
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        logger.warning(f"  Subprocess timed out ({timeout}s) — killing")
        proc.kill()
        proc.join(5)

    try:
        with open(result_path) as f:
            result = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        exit_code = proc.exitcode
        if exit_code is not None and exit_code < 0:
            status = Status.OOM_LOAD
            err_msg = f"Child killed by signal {-exit_code} (likely OOM-killer)"
        else:
            status = Status.ERROR
            err_msg = f"Child exited with code {exit_code}, no result written"
        logger.warning(f"  {status}: {err_msg}")
        result = {
            "model_id": model_id, "family": family,
            "param_M": param_M, "precision": "fp16",
            "method": "magnitude_l1_unstructured",
            "sparsity": sparsity,
            "status": status,
            "metrics": _fail_result(status, len(samples), err_msg),
            "device": "jetson_orin_nano_8gb",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        _save_result(out_path, result)
    finally:
        try:
            os.unlink(result_path)
        except OSError:
            pass

    wait_for_full_memory_reclaim()

    return result


# ═════════════════════════════════════════════════════════════════════════════
#  GENERIC COMPRESSION SUBPROCESS — runs any compression method in isolation
# ═════════════════════════════════════════════════════════════════════════════

def _subprocess_compression_entry(model_id, family, param_M, method,
                                   method_kwargs, n_samples, force, result_path):
    """Generic entry point for spawned compression child."""
    import sys, json, time, traceback
    from pathlib import Path
    _root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_root))

    from jetson.safety import make_self_oom_preferred
    make_self_oom_preferred()

    from jetson.run_jetson import run_compression_trial, _fail_result
    from jetson.safety import Status
    from evaluation.run_baseline import load_vqav2

    try:
        samples = load_vqav2(n_samples=n_samples)
        result = run_compression_trial(model_id, family, param_M, method,
                                        method_kwargs, samples, force)
    except Exception:
        result = {
            "model_id": model_id, "family": family,
            "param_M": param_M, "method": method,
            "status": Status.ERROR,
            "metrics": _fail_result(Status.ERROR, 0, traceback.format_exc()),
            "device": "jetson_orin_nano_8gb",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    with open(result_path, "w") as f:
        json.dump(result, f, default=str)


def run_compression_trial(model_id, family, param_M, method,
                           method_kwargs, samples, force=False):
    """Run a compression method trial: load → compress → evaluate → unload.

    Supported methods: awq, gptq, sparsegpt, awp, svd_llm
    """
    result = {
        "model_id": model_id, "family": family,
        "param_M": param_M, "method": method,
        "precision": method_kwargs.get("precision", "mixed"),
        "device": "jetson_orin_nano_8gb",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    try:
        if method == "sparsegpt":
            from compression.pruning.run_sparsegpt import (
                HessianCollector, apply_sparsegpt_pruning, measure_sparsity,
            )
            sparsity = method_kwargs.get("sparsity", 0.50)
            n_calib = method_kwargs.get("n_calib", 64)

            model, processor, meta, load_status, load_err = safe_load_model(
                model_id, quant=None, family=family)
            if load_status != "ok":
                status = Status.OOM_LOAD if load_status == "oom" else Status.ERROR
                result["status"] = status
                result["metrics"] = _fail_result(status, len(samples), load_err)
                _emergency_cleanup()
                return result

            device = str(next(model.parameters()).device)

            # Calibration
            collector = HessianCollector(model)
            calib_samples = samples[:n_calib]
            for s in calib_samples:
                try:
                    with torch.no_grad():
                        run_inference(model, processor, s, family, device)
                except Exception:
                    pass
            collector.remove_hooks()

            # Prune
            prune_stats = apply_sparsegpt_pruning(model, collector, sparsity)
            result["actual_sparsity"] = prune_stats["actual_sparsity"]
            result["pruned_layers"] = prune_stats["pruned_layers"]

            # Evaluate
            metrics = evaluate_model_safe(model, processor, family, device, samples)
            result["status"] = metrics["status"]
            result["metrics"] = metrics

            safe_unload(model)
            del model, processor
            _emergency_cleanup()

        elif method == "svd_llm":
            from compression.lowrank.run_svd_llm import (
                apply_svd_compression, InputCovCollector,
            )
            rank_ratio = method_kwargs.get("rank_ratio", 0.50)
            truncation_aware = method_kwargs.get("truncation_aware", False)
            n_calib = method_kwargs.get("n_calib", 64)

            model, processor, meta, load_status, load_err = safe_load_model(
                model_id, quant=None, family=family)
            if load_status != "ok":
                status = Status.OOM_LOAD if load_status == "oom" else Status.ERROR
                result["status"] = status
                result["metrics"] = _fail_result(status, len(samples), load_err)
                _emergency_cleanup()
                return result

            device = str(next(model.parameters()).device)

            input_covs = None
            if truncation_aware:
                collector = InputCovCollector(model)
                for s in samples[:n_calib]:
                    try:
                        with torch.no_grad():
                            run_inference(model, processor, s, family, device)
                    except Exception:
                        pass
                input_covs = collector.get_covariances()
                collector.remove_hooks()

            svd_stats = apply_svd_compression(model, rank_ratio,
                                               truncation_aware=truncation_aware,
                                               input_covs=input_covs)
            result["rank_ratio"] = rank_ratio
            result["replaced_layers"] = svd_stats["replaced_layers"]
            result["compression_ratio"] = svd_stats["compression_ratio"]

            metrics = evaluate_model_safe(model, processor, family, device, samples)
            result["status"] = metrics["status"]
            result["metrics"] = metrics

            safe_unload(model)
            del model, processor
            _emergency_cleanup()

        elif method == "wanda":
            from compression.pruning.run_wanda import (
                ActivationCollector, apply_wanda_pruning, measure_sparsity,
            )
            sparsity = method_kwargs.get("sparsity", 0.50)
            n_calib = method_kwargs.get("n_calib", 64)

            model, processor, meta, load_status, load_err = safe_load_model(
                model_id, quant=None, family=family)
            if load_status != "ok":
                status = Status.OOM_LOAD if load_status == "oom" else Status.ERROR
                result["status"] = status
                result["metrics"] = _fail_result(status, len(samples), load_err)
                _emergency_cleanup()
                return result

            device = str(next(model.parameters()).device)

            collector = ActivationCollector(model)
            for s in samples[:n_calib]:
                try:
                    with torch.no_grad():
                        run_inference(model, processor, s, family, device)
                except Exception:
                    pass
            input_norms = collector.get_input_norms()
            collector.remove_hooks()

            prune_stats = apply_wanda_pruning(model, input_norms, sparsity)
            result["actual_sparsity"] = prune_stats["actual_sparsity"]
            result["pruned_layers"] = prune_stats["pruned_layers"]

            metrics = evaluate_model_safe(model, processor, family, device, samples)
            result["status"] = metrics["status"]
            result["metrics"] = metrics

            safe_unload(model)
            del model, processor
            _emergency_cleanup()

        else:
            result["status"] = Status.ERROR
            result["metrics"] = _fail_result(
                Status.ERROR, len(samples), f"Unknown method: {method}")

    except Exception:
        import traceback
        result["status"] = Status.ERROR
        result["metrics"] = _fail_result(
            Status.ERROR, len(samples), traceback.format_exc())
        _emergency_cleanup()

    return result


def run_compression_isolated(model_id, family, param_M, method,
                              method_kwargs, samples,
                              results_dir, force=False, timeout=600):
    """Run a compression trial in a spawned subprocess for OOM/crash isolation."""
    safe_name = model_id.replace("/", "__")
    method_tag = method_kwargs.get("tag", method)
    tag = f"{safe_name}__{method_tag}"
    out_path = results_dir / f"{tag}.json"

    if out_path.exists() and not force:
        try:
            with open(out_path) as f:
                cached = json.load(f)
            if cached.get("status"):
                logger.info(f"  [CACHED] {out_path.name} ({cached['status']})")
                return cached
        except (json.JSONDecodeError, KeyError):
            pass

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                     delete=False, dir="/tmp") as tf:
        result_path = tf.name

    proc = _mp_ctx.Process(
        target=_subprocess_compression_entry,
        args=(model_id, family, param_M, method,
              method_kwargs, len(samples), force, result_path),
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        logger.warning(f"  Subprocess timed out ({timeout}s) — killing")
        proc.kill()
        proc.join(5)

    try:
        with open(result_path) as f:
            result = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        exit_code = proc.exitcode
        if exit_code is not None and exit_code < 0:
            status = Status.OOM_LOAD
            err_msg = f"Child killed by signal {-exit_code} (likely OOM-killer)"
        else:
            status = Status.ERROR
            err_msg = f"Child exited with code {exit_code}, no result written"
        logger.warning(f"  {status}: {err_msg}")
        result = {
            "model_id": model_id, "family": family,
            "param_M": param_M, "method": method,
            "status": status,
            "metrics": _fail_result(status, len(samples), err_msg),
            "device": "jetson_orin_nano_8gb",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        _save_result(out_path, result)
    finally:
        try:
            os.unlink(result_path)
        except OSError:
            pass

    wait_for_full_memory_reclaim()
    return result


# ═════════════════════════════════════════════════════════════════════════════
#  PRE-DOWNLOAD MODELS — separate download from GPU loading
# ═════════════════════════════════════════════════════════════════════════════

def download_models(models: list[tuple], max_param_M: float = None):
    """Download model weights without loading into GPU memory.

    Uses huggingface_hub.snapshot_download() for each model in the registry.
    """
    from huggingface_hub import snapshot_download

    for model_id, family, param_M in models:
        if max_param_M is not None and param_M > max_param_M:
            logger.info(f"  SKIP {model_id} ({param_M:.0f}M > {max_param_M:.0f}M cap)")
            continue
        logger.info(f"  Downloading {model_id} ({param_M:.0f}M) ...")
        try:
            snapshot_download(model_id)
            logger.info(f"    OK")
        except Exception as e:
            logger.warning(f"    FAILED: {e}")


# ═════════════════════════════════════════════════════════════════════════════
#  MODEL REGISTRY — ordered smallest → largest within each family
# ═════════════════════════════════════════════════════════════════════════════

JETSON_MODELS = [
    # (model_id, family, approx_params_M)
    # ── nanoVLM ───────────────────────────────────────────────────────────
    ("lusxvr/nanoVLM-222M",                    "nanovlm",    222),
    ("lusxvr/nanoVLM-450M",                    "nanovlm",    450),
    # ── SmolVLM ───────────────────────────────────────────────────────────
    ("HuggingFaceTB/SmolVLM-256M-Instruct",    "smolvlm",    256),
    ("HuggingFaceTB/SmolVLM-500M-Instruct",    "smolvlm",    500),
    ("HuggingFaceTB/SmolVLM-Instruct",         "smolvlm",   2200),
    # ── LFM2-VL ──────────────────────────────────────────────────────────
    ("LiquidAI/LFM2-VL-450M",                  "lfm2vl",     450),
    ("LiquidAI/LFM2-VL-1.6B",                  "lfm2vl",    1600),
    ("LiquidAI/LFM2-VL-3B",                    "lfm2vl",    3000),
    # ── Moondream ─────────────────────────────────────────────────────────
    ("vikhyatk/moondream2",                      "moondream",  2000),
    # ── FastVLM ──────────────────────────────────────────────────────────
    ("apple/FastVLM-0.5B",                      "fastvlm",    500),
    ("apple/FastVLM-1.5B",                      "fastvlm",   1500),
    ("apple/FastVLM-7B",                        "fastvlm",   7000),
    # ── InternVL2.5 ──────────────────────────────────────────────────────
    ("OpenGVLab/InternVL2_5-1B",                "internvl25", 1000),
    ("OpenGVLab/InternVL2_5-2B",                "internvl25", 2000),
    ("OpenGVLab/InternVL2_5-4B",                "internvl25", 4000),
    ("OpenGVLab/InternVL2_5-8B",                "internvl25", 8000),
    # ── Qwen2.5-VL ──────────────────────────────────────────────────────
    ("Qwen/Qwen2.5-VL-3B-Instruct",            "qwen25vl",   3000),
    ("Qwen/Qwen2.5-VL-7B-Instruct",            "qwen25vl",   7000),
    # ── Gemma 3 (only 4B+ are VLMs) ─────────────────────────────────────
    ("google/gemma-3-4b-it",                    "gemma3",     4000),
    # ── Ovis2 ────────────────────────────────────────────────────────────
    ("AIDC-AI/Ovis2-1B",                        "ovis2",      1000),
    ("AIDC-AI/Ovis2-2B",                        "ovis2",      2000),
    ("AIDC-AI/Ovis2-4B",                        "ovis2",      4000),
    ("AIDC-AI/Ovis2-8B",                        "ovis2",      8000),
]


def models_for_families(families: list[str] | None) -> list[tuple]:
    """Filter JETSON_MODELS to requested families (None = all)."""
    if families is None:
        return list(JETSON_MODELS)
    fset = set(families)
    return [(m, f, p) for m, f, p in JETSON_MODELS if f in fset]


# ═════════════════════════════════════════════════════════════════════════════
#  EVALUATION CORE — safe, timeout-wrapped, memory-monitored
# ═════════════════════════════════════════════════════════════════════════════

def _safe_single_inference(model, processor, sample, family, device,
                           timeout_s=INFERENCE_TIMEOUT_S):
    """Run one inference with timeout + OOM protection.

    Returns (prediction_text | None, status_str, error_msg | None).
    """
    def _do():
        return run_inference(model, processor, sample, family, device)

    pred, status, err = run_with_timeout(_do, timeout_s)
    return pred, status, err


def evaluate_model_safe(
    model, processor, family: str, device: str,
    samples: list, n_warmup: int = 2,
) -> dict:
    """Evaluate a model on VQAv2 samples with full safety guards.

    Returns a dict with keys:
        status, accuracy, avg_latency_s, peak_memory_mb, avg_memory_mb,
        throughput_sps, n_evaluated, n_total, latencies, error_msg
    """
    profiler = GPUProfiler(device_index=0)
    scores: list[float] = []
    latencies: list[float] = []
    error_msg = None

    # ── Warmup (uses longer timeout; catches JIT compile overhead) ────────
    for i in range(min(n_warmup, len(samples))):
        pred, wstatus, werr = _safe_single_inference(
            model, processor, samples[i], family, device,
            timeout_s=WARMUP_TIMEOUT_S,
        )
        if wstatus == "oom":
            return _fail_result(Status.OOM_INFER, len(samples),
                                f"OOM during warmup sample {i}: {werr}")
        if wstatus == "timeout":
            return _fail_result(Status.TIMEOUT, len(samples),
                                f"Timeout during warmup sample {i}")
        if wstatus == "error":
            return _fail_result(Status.ERROR, len(samples),
                                f"Error during warmup: {werr}")

    # ── Main evaluation loop ─────────────────────────────────────────────
    with profiler:
        for idx, sample in enumerate(samples):
            # Memory guard
            if is_memory_critical():
                error_msg = (f"System memory critically low "
                             f"({get_available_memory_mb():.0f} MB free) "
                             f"after {idx} samples")
                return _result(Status.MEM_CRITICAL, scores, latencies,
                               profiler, len(samples), error_msg)

            t0 = time.perf_counter()
            pred, status, err = _safe_single_inference(
                model, processor, sample, family, device,
            )
            elapsed = time.perf_counter() - t0

            if status == "oom":
                error_msg = f"OOM at sample {idx}: {err}"
                return _result(Status.OOM_INFER, scores, latencies,
                               profiler, len(samples), error_msg)
            if status == "timeout":
                error_msg = f"Timeout at sample {idx} ({elapsed:.0f}s)"
                return _result(Status.TIMEOUT, scores, latencies,
                               profiler, len(samples), error_msg)
            if status == "error":
                # Log but continue — some samples may have bad images
                logger.warning(f"  Sample {idx} error (skipped): {err}")
                continue

            latencies.append(elapsed)
            scores.append(_vqa_accuracy(pred, sample["answers"]))

            # ── Latency-ceiling early abort ───────────────────────────────
            if (len(latencies) >= LATENCY_CHECK_WINDOW
                    and len(latencies) <= LATENCY_CHECK_WINDOW + 1):
                avg_lat = sum(latencies) / len(latencies)
                if avg_lat > LATENCY_CEILING_S:
                    error_msg = (f"Avg latency {avg_lat:.1f}s > ceiling "
                                 f"{LATENCY_CEILING_S}s after "
                                 f"{len(latencies)} samples — aborting")
                    return _result(Status.TOO_SLOW, scores, latencies,
                                   profiler, len(samples), error_msg)

    return _result(Status.PASS, scores, latencies, profiler, len(samples))


def _result(status: str, scores, latencies, profiler, n_total,
            error_msg=None) -> dict:
    stats = profiler.stats()
    avg_acc = sum(scores) / len(scores) if scores else 0.0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    throughput = (len(latencies) / stats.wall_time_s
                  if stats.wall_time_s > 0 else 0.0)
    deployable = (status == Status.PASS and avg_lat <= DEPLOYABLE_LATENCY_S)
    return {
        "status":           status,
        "deployable":       deployable,
        "accuracy":         round(avg_acc, 4),
        "avg_latency_s":    round(avg_lat, 4),
        "peak_memory_mb":   round(stats.peak_memory_mb, 1),
        "avg_memory_mb":    round(stats.avg_memory_mb, 1),
        "avg_power_w":      round(stats.avg_power_w, 1),
        "avg_gpu_util_pct": round(stats.avg_gpu_util_pct, 1),
        "throughput_sps":   round(throughput, 3),
        "n_evaluated":      len(latencies),
        "n_total":          n_total,
        "error_msg":        error_msg,
    }


def _fail_result(status: str, n_total: int, error_msg: str) -> dict:
    return {
        "status":           status,
        "deployable":       False,
        "accuracy":         0.0,
        "avg_latency_s":    0.0,
        "peak_memory_mb":   0.0,
        "avg_memory_mb":    0.0,
        "avg_power_w":      0.0,
        "avg_gpu_util_pct": 0.0,
        "throughput_sps":   0.0,
        "n_evaluated":      0,
        "n_total":          n_total,
        "error_msg":        error_msg,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  SINGLE-MODEL TRIAL — load → evaluate → unload → save
# ═════════════════════════════════════════════════════════════════════════════

def run_trial(
    model_id: str,
    family: str,
    param_M: float,
    precision: str,       # "fp16", "int8", "int4"
    samples: list,
    results_dir: Path,
    force: bool = False,
) -> dict:
    """Run a single model trial (load → eval → unload).

    Returns the full result dict (also saved to disk).
    """
    safe_name = model_id.replace("/", "__")
    if precision == "fp16":
        tag = safe_name
        out_path = results_dir / f"{tag}.json"
    else:
        tag = f"{safe_name}__{precision}__bnb"
        out_path = results_dir / f"{tag}.json"

    # ── Resumability — only cache PASS and TOO_SLOW ────────────────────
    if out_path.exists() and not force:
        try:
            with open(out_path) as f:
                cached = json.load(f)
            if cached.get("status") in CACHEABLE_STATUSES:
                logger.info(f"  [CACHED] {out_path.name} ({cached['status']})")
                return cached
            else:
                logger.info(f"  Retrying {out_path.name} (prev status: {cached.get('status')})")
        except (json.JSONDecodeError, KeyError):
            pass

    result = {
        "model_id":    model_id,
        "family":      family,
        "param_M":     param_M,
        "precision":   precision,
        "device":      "jetson_orin_nano_8gb",
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # ── Load model (with preflight check + memory watchdog) ────────────
    quant = None if precision == "fp16" else precision
    logger.info(f"  Loading {model_id} @ {precision} ...")
    model, processor, meta, load_status, load_err = safe_load_model(
        model_id, quant=quant, family=family, param_M=param_M,
    )

    if load_status != "ok":
        status = Status.OOM_LOAD if load_status == "oom" else Status.ERROR
        logger.warning(f"  LOAD FAILED ({status}): {load_err}")
        result["status"] = status
        result["metrics"] = _fail_result(status, len(samples),
                                         load_err or "Unknown load error")
        _emergency_cleanup()
        _save_result(out_path, result)
        return result

    # ── Record model metadata ────────────────────────────────────────────
    try:
        device = str(next(model.parameters()).device)
    except StopIteration:
        device = "cuda:0"

    num_params = sum(p.numel() for p in model.parameters())
    result["num_params_M"] = round(num_params / 1e6, 1)
    result["gpu_mem_load_mb"] = round(meta.gpu_mem_delta_mb, 1)

    logger.info(
        f"  Loaded: {num_params/1e6:.1f}M params | "
        f"mem delta: {meta.gpu_mem_delta_mb:.0f} MB | "
        f"avail: {get_available_memory_mb():.0f} MB"
    )

    # ── Evaluate ─────────────────────────────────────────────────────────
    logger.info(f"  Evaluating on {len(samples)} VQAv2 samples ...")
    metrics = evaluate_model_safe(model, processor, family, device, samples)
    result["status"] = metrics["status"]
    result["metrics"] = metrics

    # ── Log result summary ───────────────────────────────────────────────
    s = metrics["status"]
    if s == Status.PASS:
        logger.info(
            f"  PASS  acc={metrics['accuracy']:.4f}  "
            f"lat={metrics['avg_latency_s']:.2f}s  "
            f"mem={metrics['peak_memory_mb']:.0f}MB  "
            f"deploy={'YES' if metrics['deployable'] else 'NO'}"
        )
    else:
        logger.warning(f"  {s}: {metrics.get('error_msg', '')}")

    # ── Unload ───────────────────────────────────────────────────────────
    safe_unload(model)
    del model, processor, meta
    _emergency_cleanup()
    time.sleep(5)  # Let the OS reclaim unified memory between loads

    # Save every result — each outcome is a real data point
    _save_result(out_path, result)
    return result


# ═════════════════════════════════════════════════════════════════════════════
#  PRUNING TRIAL — load FP16 → prune → evaluate → unload
# ═════════════════════════════════════════════════════════════════════════════

VISION_KEYWORDS = {
    "vision_model", "visual_model", "image_encoder", "vision_encoder",
    "patch_embed", "visual_projection", "img_projection",
    "vision_tower", "vit", "davit", "siglip", "fastvit",
}

def _is_vision_module(name: str) -> bool:
    lo = name.lower()
    return any(kw in lo for kw in VISION_KEYWORDS)


def run_pruning_trial(
    model_id: str,
    family: str,
    param_M: float,
    sparsity: float,
    samples: list,
    force: bool = False,
) -> dict:
    """Load FP16, prune LLM backbone, evaluate."""
    safe_name = model_id.replace("/", "__")
    tag = f"{safe_name}__sp{int(sparsity * 100)}"
    out_path = RESULTS_PRUNING / f"{tag}.json"

    if out_path.exists() and not force:
        logger.info(f"  [CACHED] {out_path.name}")
        with open(out_path) as f:
            return json.load(f)

    result = {
        "model_id":   model_id,
        "family":     family,
        "param_M":    param_M,
        "precision":  "fp16",
        "method":     "magnitude_l1_unstructured",
        "sparsity":   sparsity,
        "device":     "jetson_orin_nano_8gb",
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Load (no preflight — subprocess isolation handles OOM)
    model, processor, meta, load_status, load_err = safe_load_model(
        model_id, quant=None, family=family,
    )
    if load_status != "ok":
        status = Status.OOM_LOAD if load_status == "oom" else Status.ERROR
        result["status"] = status
        result["metrics"] = _fail_result(status, len(samples), load_err)
        _emergency_cleanup()
        return result

    try:
        device = str(next(model.parameters()).device)
    except StopIteration:
        device = "cuda:0"

    # Prune
    logger.info(f"  Pruning {model_id} at {sparsity*100:.0f}% sparsity ...")
    pruned_layers = 0
    pruned_zeros = 0
    total_weights = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if _is_vision_module(name):
            continue
        prune.l1_unstructured(module, name="weight", amount=sparsity)
        prune.remove(module, "weight")
        pruned_zeros += (module.weight == 0).sum().item()
        total_weights += module.weight.numel()
        pruned_layers += 1

    actual_sparsity = pruned_zeros / total_weights if total_weights > 0 else 0.0
    result["actual_sparsity"] = round(actual_sparsity, 4)
    result["pruned_layers"] = pruned_layers
    logger.info(
        f"  Pruned {pruned_layers} layers, "
        f"actual sparsity = {actual_sparsity:.4f}"
    )

    # Evaluate
    logger.info(f"  Evaluating pruned model on {len(samples)} samples ...")
    metrics = evaluate_model_safe(model, processor, family, device, samples)
    result["status"] = metrics["status"]
    result["metrics"] = metrics

    if metrics["status"] == Status.PASS:
        logger.info(
            f"  PASS (pruned {sparsity*100:.0f}%)  "
            f"acc={metrics['accuracy']:.4f}  "
            f"lat={metrics['avg_latency_s']:.2f}s"
        )

    safe_unload(model)
    del model, processor, meta
    _emergency_cleanup()
    time.sleep(5)  # Let the OS reclaim unified memory between loads

    _save_result(out_path, result)
    return result


# ═════════════════════════════════════════════════════════════════════════════
#  CEILING REPORT GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def generate_ceiling_report(all_results: list[dict]) -> dict:
    """Build per-family ceiling report from all trial results."""
    families: dict[str, dict] = {}

    for r in all_results:
        fam   = r["family"]
        mid   = r["model_id"]
        prec  = r.get("precision", "fp16")
        pM    = r["param_M"]
        st    = r["status"]
        method = r.get("method")         # None for baseline/ptq, "magnitude_l1_…" for pruning
        sp     = r.get("sparsity")       # pruning sparsity

        if fam not in families:
            families[fam] = {
                "models_tested": [],
                "fp16_ceiling":  None,
                "int8_ceiling":  None,
                "int4_ceiling":  None,
                "fp16_pass":     [],
                "int8_pass":     [],
                "int4_pass":     [],
                "pruning_results": [],
                "all_trials":    [],
            }

        trial_summary = {
            "model_id": mid,
            "param_M":  pM,
            "precision": prec,
            "status":   st,
        }
        if method:
            trial_summary["method"] = method
            trial_summary["sparsity"] = sp
        if st == Status.PASS:
            m = r.get("metrics", {})
            trial_summary["accuracy"]     = m.get("accuracy", 0)
            trial_summary["avg_latency_s"]= m.get("avg_latency_s", 0)
            trial_summary["peak_memory_mb"]= m.get("peak_memory_mb", 0)
            trial_summary["deployable"]   = m.get("deployable", False)

        families[fam]["all_trials"].append(trial_summary)

        if mid not in families[fam]["models_tested"]:
            families[fam]["models_tested"].append(mid)

        # Track ceilings (excluding pruning — pruning doesn't change memory)
        if method is None and st == Status.PASS:
            bucket = f"{prec}_pass"
            families[fam][bucket].append({"model_id": mid, "param_M": pM})

    # Determine ceilings (largest passing model per precision)
    for fam, data in families.items():
        for prec in ("fp16", "int8", "int4"):
            bucket = data[f"{prec}_pass"]
            if bucket:
                largest = max(bucket, key=lambda x: x["param_M"])
                data[f"{prec}_ceiling"] = largest

    # Compression recovery summary
    report = {"families": families, "summary": {}}
    for fam, data in families.items():
        fp16_max = data["fp16_ceiling"]["param_M"] if data["fp16_ceiling"] else 0
        int8_max = data["int8_ceiling"]["param_M"] if data["int8_ceiling"] else 0
        int4_max = data["int4_ceiling"]["param_M"] if data["int4_ceiling"] else 0
        report["summary"][fam] = {
            "fp16_ceiling_M": fp16_max,
            "int8_ceiling_M": int8_max,
            "int4_ceiling_M": int4_max,
            "compression_extends_ceiling": (int8_max > fp16_max
                                            or int4_max > fp16_max),
            "max_usable_M": max(fp16_max, int8_max, int4_max),
            "max_usable_precision": (
                "int4" if int4_max == max(fp16_max, int8_max, int4_max) and int4_max > fp16_max
                else "int8" if int8_max == max(fp16_max, int8_max, int4_max) and int8_max > fp16_max
                else "fp16"
            ),
        }

    return report


# ═════════════════════════════════════════════════════════════════════════════
#  PRETTY PRINTER
# ═════════════════════════════════════════════════════════════════════════════

def print_report(report: dict):
    """Print a human-readable ceiling report."""
    print("\n" + "=" * 72)
    print("  JETSON ORIN NANO 8 GB — VLM CEILING REPORT")
    print("=" * 72)

    for fam, info in report["summary"].items():
        print(f"\n{'─' * 60}")
        print(f"  Family: {fam}")
        print(f"{'─' * 60}")
        fp16 = info["fp16_ceiling_M"]
        int8 = info["int8_ceiling_M"]
        int4 = info["int4_ceiling_M"]
        print(f"  FP16 ceiling : {fp16:>6.0f} M params" if fp16 else "  FP16 ceiling :   NONE (no model passed)")
        print(f"  INT8 ceiling : {int8:>6.0f} M params" if int8 else "  INT8 ceiling :   NONE")
        print(f"  INT4 ceiling : {int4:>6.0f} M params" if int4 else "  INT4 ceiling :   NONE")
        print(f"  Max usable   : {info['max_usable_M']:>6.0f} M @ {info['max_usable_precision']}")
        if info["compression_extends_ceiling"]:
            print(f"  ** Compression EXTENDS the ceiling beyond FP16 **")

    # Detailed trial table
    print(f"\n{'=' * 72}")
    print("  DETAILED TRIAL RESULTS")
    print(f"{'=' * 72}")
    print(f"  {'Model':<42} {'Prec':<6} {'Status':<14} {'Acc':>6} {'Lat':>7} {'Mem':>7} {'Deploy':>6}")
    print(f"  {'─'*42} {'─'*6} {'─'*14} {'─'*6} {'─'*7} {'─'*7} {'─'*6}")

    for fam, fdata in report["families"].items():
        for trial in fdata["all_trials"]:
            model_short = trial["model_id"].split("/")[-1][:40]
            prec = trial["precision"]
            if trial.get("method"):
                prec = f"prn{int(trial.get('sparsity',0)*100)}"
            status = trial["status"]
            acc = f"{trial.get('accuracy', 0):.4f}" if status == Status.PASS else "  -"
            lat = f"{trial.get('avg_latency_s', 0):.2f}s" if status == Status.PASS else "   -"
            mem = f"{trial.get('peak_memory_mb', 0):.0f}MB" if status == Status.PASS else "   -"
            dep = "YES" if trial.get("deployable") else "no"
            if status != Status.PASS:
                dep = "-"
            print(f"  {model_short:<42} {prec:<6} {status:<14} {acc:>6} {lat:>7} {mem:>7} {dep:>6}")

    print(f"\n{'=' * 72}\n")


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _save_result(path: Path, result: dict):
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"  Saved → {path}")


def _model_short(model_id: str) -> str:
    return model_id.split("/")[-1]


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Jetson Orin Nano 8 GB — VLM ceiling & compression benchmark")
    parser.add_argument("--families", type=str, default=None,
                        help="Comma-separated families (default: all)")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of VQAv2 samples per evaluation (default: 100)")
    parser.add_argument("--scan_only", action="store_true",
                        help="Only run FP16 baseline ceiling scan")
    parser.add_argument("--skip_pruning", action="store_true",
                        help="Skip pruning experiments")
    parser.add_argument("--skip_advanced", action="store_true",
                        help="Skip advanced compression methods (SparseGPT, Wanda, SVD-LLM)")
    parser.add_argument("--methods", type=str, default=None,
                        help="Comma-separated list of advanced methods to run. "
                             "Options: sparsegpt,wanda,svd_llm (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing results")
    parser.add_argument("--timeout", type=int, default=INFERENCE_TIMEOUT_S,
                        help=f"Per-inference timeout in seconds (default: {INFERENCE_TIMEOUT_S})")
    parser.add_argument("--max_param_M", type=float, default=None,
                        help="Skip models larger than this (in millions of params)")
    parser.add_argument("--fp16_max_param_M", type=float, default=3000,
                        help="Skip FP16 baseline for models larger than this (default: 3000M). "
                             "Compression methods (INT8/INT4/pruning) still run for these models.")
    parser.add_argument("--download_only", action="store_true",
                        help="Only download model weights (no GPU loading or evaluation)")
    args = parser.parse_args()

    # Parse families
    families = args.families.split(",") if args.families else None
    models = models_for_families(families)

    if not models:
        logger.error("No models to test. Check --families argument.")
        sys.exit(1)

    # ── Download-only mode ───────────────────────────────────────────────
    if args.download_only:
        logger.info("Download-only mode — fetching model weights without GPU loading")
        download_models(models, max_param_M=args.max_param_M)
        logger.info("Downloads complete.")
        return

    # Filter by max_param_M for FP16 baseline (compression phases handle this per-model)
    if args.max_param_M is not None:
        n_before = len(models)
        models = [(m, f, p) for m, f, p in models if p <= args.max_param_M]
        logger.info(f"  --max_param_M {args.max_param_M:.0f}: {n_before} → {len(models)} models")

    # Override timeout if specified
    import jetson.safety as safety_mod
    safety_mod.INFERENCE_TIMEOUT_S = args.timeout

    logger.info(f"Jetson VLM Benchmark")
    logger.info(f"  Models to test : {len(models)}")
    logger.info(f"  VQAv2 samples  : {args.n_samples}")
    logger.info(f"  Scan only      : {args.scan_only}")
    logger.info(f"  Max param M    : {args.max_param_M or 'unlimited'}")
    logger.info(f"  FP16 max param : {args.fp16_max_param_M or 'unlimited'}")
    logger.info(f"  Timeout        : {args.timeout}s per inference")
    logger.info(f"  Available mem  : {get_available_memory_mb():.0f} MB")
    logger.info("")

    # ── Load VQAv2 dataset once ──────────────────────────────────────────
    logger.info("Loading VQAv2 evaluation dataset ...")
    samples = load_vqav2(n_samples=args.n_samples)
    logger.info(f"  {len(samples)} samples loaded.\n")

    # ── Capture clean-state memory baseline AFTER dataset is loaded ──────
    # The parent process keeps `samples` in memory permanently, so the
    # baseline must reflect that.  Subprocesses reload the dataset into
    # their own address space, but that memory is freed when they exit.
    _emergency_cleanup()
    time.sleep(2)
    set_memory_baseline()

    all_results: list[dict] = []

    # ── Sort ALL models globally by param_M (smallest → largest) ─────────
    models_sorted = sorted(models, key=lambda x: x[2])  # sort by param_M

    # Parse advanced methods
    advanced_methods = None
    if args.methods:
        advanced_methods = set(args.methods.split(","))
    elif not args.skip_advanced:
        advanced_methods = {"sparsegpt", "wanda", "svd_llm"}

    logger.info("=" * 60)
    logger.info("MODEL-BY-MODEL BENCHMARK (smallest → largest)")
    logger.info("  For each model: FP16 → INT8 → INT4 → Advanced → Pruning")
    if advanced_methods:
        logger.info(f"  Advanced methods: {', '.join(sorted(advanced_methods))}")
    logger.info("  No preflight rejection — let timeout/OOM be the signal")
    logger.info("=" * 60 + "\n")

    for i, (model_id, family, param_M) in enumerate(models_sorted, 1):
        # ── Wait for full memory reclaim before starting each model ──────
        wait_for_full_memory_reclaim()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        logger.info("=" * 60)
        logger.info(f"MODEL {i}/{len(models_sorted)}: {_model_short(model_id)} "
                     f"({param_M:.0f}M) — family: {family}")
        logger.info(f"  Available memory: {get_available_memory_mb():.0f} MB "
                     f"(baseline: {_baseline_memory_mb:.0f} MB)")
        logger.info("=" * 60)

        # ── FP16 Baseline ────────────────────────────────────────────────
        skip_fp16 = (args.fp16_max_param_M is not None
                     and param_M > args.fp16_max_param_M)
        if skip_fp16:
            logger.info(f"\n  [1/5] FP16 baseline — SKIPPED "
                        f"({param_M:.0f}M > --fp16_max_param_M {args.fp16_max_param_M:.0f}M)")
            r_fp16 = {
                "model_id": model_id, "family": family,
                "param_M": param_M, "precision": "fp16",
                "status": "SKIPPED_TOO_LARGE",
                "metrics": _fail_result("SKIPPED_TOO_LARGE", len(samples),
                                        f"Skipped: {param_M:.0f}M > fp16_max_param_M"),
                "device": "jetson_orin_nano_8gb",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        else:
            logger.info(f"\n  [1/5] FP16 baseline")
            r_fp16 = run_trial_isolated(model_id, family, param_M, "fp16",
                                         samples, RESULTS_BASELINE, args.force)
        all_results.append(r_fp16)
        logger.info(f"  → FP16 status: {r_fp16['status']}")

        if args.scan_only:
            logger.info(f"  (scan_only — skipping compression & pruning)\n")
            continue

        # ── INT8 ─────────────────────────────────────────────────────────
        logger.info(f"\n  [2/5] INT8 quantization")
        r_int8 = run_trial_isolated(model_id, family, param_M, "int8",
                                     samples, RESULTS_PTQ, args.force)
        all_results.append(r_int8)
        logger.info(f"  → INT8 status: {r_int8['status']}")

        # ── INT4 ─────────────────────────────────────────────────────────
        logger.info(f"\n  [3/5] INT4 quantization")
        r_int4 = run_trial_isolated(model_id, family, param_M, "int4",
                                     samples, RESULTS_PTQ, args.force)
        all_results.append(r_int4)
        logger.info(f"  → INT4 status: {r_int4['status']}")

        # ── Advanced compression methods ─────────────────────────────────
        # These require loading FP16 first (for calibration), so only
        # attempt if FP16 at least fits in memory
        fp16_loadable = r_fp16["status"] in (Status.PASS, Status.TOO_SLOW)
        step_num = 4   # track step numbering dynamically

        if advanced_methods and not args.scan_only and fp16_loadable:
            # SparseGPT: 50% sparsity
            if "sparsegpt" in advanced_methods:
                logger.info(f"\n  [{step_num}] SparseGPT 50% sparsity")
                r_sgpt = run_compression_isolated(
                    model_id, family, param_M, "sparsegpt",
                    {"sparsity": 0.50, "n_calib": 64, "tag": "sparsegpt_sp50"},
                    samples, RESULTS_SPARSEGPT, args.force)
                all_results.append(r_sgpt)
                logger.info(f"  → SparseGPT 50% status: {r_sgpt['status']}")
                step_num += 1

            # Wanda: 50% sparsity (higher than the existing 20%/40% magnitude pruning)
            if "wanda" in advanced_methods:
                logger.info(f"\n  [{step_num}] Wanda 50% sparsity")
                r_wanda = run_compression_isolated(
                    model_id, family, param_M, "wanda",
                    {"sparsity": 0.50, "n_calib": 64, "tag": "wanda_sp50"},
                    samples, RESULTS_PRUNING, args.force)
                all_results.append(r_wanda)
                logger.info(f"  → Wanda 50% status: {r_wanda['status']}")
                step_num += 1

            # SVD-LLM: 50% rank ratio
            if "svd_llm" in advanced_methods:
                logger.info(f"\n  [{step_num}] SVD-LLM rank_ratio=0.50")
                r_svd = run_compression_isolated(
                    model_id, family, param_M, "svd_llm",
                    {"rank_ratio": 0.50, "truncation_aware": False,
                     "tag": "svdllm_rr50"},
                    samples, RESULTS_SVDLLM, args.force)
                all_results.append(r_svd)
                logger.info(f"  → SVD-LLM status: {r_svd['status']}")
                step_num += 1

        elif advanced_methods and not args.scan_only and not fp16_loadable:
            logger.info(f"\n  [{step_num}] Advanced methods — SKIPPED (FP16 couldn't load)")

        # ── Magnitude Pruning (only if FP16 loaded successfully) ─────────
        if not args.skip_pruning and not args.scan_only and fp16_loadable:
            logger.info(f"\n  [{step_num}] Magnitude Pruning 20% sparsity")
            r_p20 = run_pruning_trial_isolated(
                model_id, family, param_M, 0.20, samples, args.force)
            all_results.append(r_p20)
            logger.info(f"  → Prune 20% status: {r_p20['status']}")
            step_num += 1

            logger.info(f"\n  [{step_num}] Magnitude Pruning 40% sparsity")
            r_p40 = run_pruning_trial_isolated(
                model_id, family, param_M, 0.40, samples, args.force)
            all_results.append(r_p40)
            logger.info(f"  → Prune 40% status: {r_p40['status']}")
        elif not args.skip_pruning and not args.scan_only:
            logger.info(f"\n  [{step_num}] Pruning — SKIPPED (FP16 couldn't load)")
        elif not args.scan_only:
            logger.info(f"\n  [{step_num}] Pruning — SKIPPED (--skip_pruning)")

        # ── Summary for this model ───────────────────────────────────────
        logger.info(f"\n  ── Summary for {_model_short(model_id)} ({param_M:.0f}M) ──")
        logger.info(f"     FP16: {r_fp16['status']}")
        if not args.scan_only:
            logger.info(f"     INT8: {r_int8['status']}")
            logger.info(f"     INT4: {r_int4['status']}")
        logger.info(f"  Available memory after cleanup: {get_available_memory_mb():.0f} MB")
        logger.info("")

    # ══════════════════════════════════════════════════════════════════════
    #  REPORT
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING REPORT")
    logger.info("=" * 60 + "\n")

    report = generate_ceiling_report(all_results)
    print_report(report)

    report_path = RESULTS_BASE / "ceiling_report.json"
    _save_result(report_path, report)

    full_path = RESULTS_BASE / "all_trials.json"
    _save_result(full_path, all_results)

    logger.info(f"Ceiling report : {report_path}")
    logger.info(f"Full results   : {full_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
