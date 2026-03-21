"""
jetson/safety.py
================
Safety mechanisms for running VLM benchmarks on the memory-constrained
Jetson Orin Nano 8 GB (unified CPU+GPU memory).

Provides
--------
- Unified-memory monitoring via /proc/meminfo
- Pre-flight model-size estimation (will it fit?)
- Timeout-wrapped inference (per-sample hard timeout)
- OOM-safe model loading and unloading
- Latency-ceiling early abort (stop if model is impractically slow)
"""

import gc
import logging
import threading
import traceback
from typing import Any, Callable, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ── Tunable thresholds ────────────────────────────────────────────────────────

CRITICAL_FREE_MB = 700           # Abort if MemAvailable drops below this
INFERENCE_TIMEOUT_S = 150        # Hard timeout per single inference (seconds)
WARMUP_TIMEOUT_S = 200           # First inference (includes JIT/compilation)
LATENCY_CEILING_S = 30           # Flag model as TOO_SLOW if avg latency > this
LATENCY_CHECK_WINDOW = 5         # Evaluate avg latency after this many samples
DEPLOYABLE_LATENCY_S = 3.0       # Jetson deployability threshold (from config)

# Statuses that should be cached — every outcome is a real data point
CACHEABLE_STATUSES = {"PASS", "TOO_SLOW", "OOM_LOAD", "OOM_INFER", "TIMEOUT",
                      "MEM_CRITICAL", "ERROR", "PREFLIGHT_FAIL"}


# ── Status codes ──────────────────────────────────────────────────────────────

class Status:
    PASS           = "PASS"           # Completed successfully
    OOM_LOAD       = "OOM_LOAD"       # Out of memory during model loading
    OOM_INFER      = "OOM_INFER"      # Out of memory during inference
    TIMEOUT        = "TIMEOUT"        # Inference exceeded hard timeout
    TOO_SLOW       = "TOO_SLOW"       # Avg latency above ceiling
    MEM_CRITICAL   = "MEM_CRITICAL"   # System memory critically low mid-run
    SKIPPED        = "SKIPPED"        # Skipped (larger model in same family+precision failed)
    ERROR          = "ERROR"          # Unexpected error


# ── Memory monitoring ─────────────────────────────────────────────────────────

def get_available_memory_mb() -> float:
    """Return available system memory in MB from /proc/meminfo.

    On Jetson this reflects the unified CPU+GPU pool.
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024  # kB → MB
    except (IOError, ValueError):
        pass
    # Fallback: CUDA free memory
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        return free / 1024**2
    return 0.0


def get_gpu_memory_used_mb() -> float:
    """Return GPU-allocated memory via torch (MB)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def is_memory_critical() -> bool:
    """True if system memory is dangerously low."""
    return get_available_memory_mb() < CRITICAL_FREE_MB


# ── Timeout-wrapped execution ─────────────────────────────────────────────────

def _worker(fn: Callable, args: tuple, kwargs: dict,
            result: dict, done: threading.Event):
    """Thread target for timeout-wrapped execution."""
    try:
        result["value"] = fn(*args, **kwargs)
        result["status"] = "ok"
    except torch.cuda.OutOfMemoryError:
        result["status"] = "oom"
        result["error"] = "CUDA out of memory"
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            result["status"] = "oom"
            result["error"] = str(e)
        else:
            result["status"] = "error"
            result["error"] = traceback.format_exc()
    except Exception:
        result["status"] = "error"
        result["error"] = traceback.format_exc()
    finally:
        done.set()


def run_with_timeout(fn: Callable, timeout_s: float,
                     *args, **kwargs) -> Tuple[Any, str, Optional[str]]:
    """Run *fn* with a hard timeout.

    Returns ``(result_value, status, error_msg)``.

    *status* is one of ``"ok"``, ``"oom"``, ``"timeout"``, ``"error"``.
    """
    result: dict = {}
    done = threading.Event()
    t = threading.Thread(target=_worker,
                         args=(fn, args, kwargs, result, done),
                         daemon=True)
    t.start()
    completed = done.wait(timeout=timeout_s)

    if not completed:
        return None, "timeout", f"Timed out after {timeout_s:.0f}s"

    status = result.get("status", "error")
    if status == "ok":
        return result["value"], "ok", None
    elif status == "oom":
        return None, "oom", result.get("error")
    else:
        return None, "error", result.get("error")


# ── Pre-load memory estimation ────────────────────────────────────────────────

MEMORY_SAFETY_MARGIN = 0.85  # Only use up to 85% of available memory


def estimate_model_memory_mb(param_M: float, quant: Optional[str] = None) -> float:
    """Estimate memory needed to load a model (MB), from param count (millions).

    Includes rough overhead for optimizer state, activations, framework buffers.
    """
    overhead_mb = 500  # framework, processor, dataset, etc.
    if quant in ("int4", "nf4"):
        return param_M * 0.5 + overhead_mb
    elif quant == "int8":
        return param_M * 1.0 + overhead_mb
    else:
        # FP16 / BF16
        return param_M * 2.0 + overhead_mb


def preflight_memory_check(param_M: float,
                           quant: Optional[str] = None) -> Tuple[bool, str]:
    """Check if a model is likely to fit in available memory BEFORE loading.

    Returns ``(ok, message)``.
    """
    available = get_available_memory_mb()
    estimated = estimate_model_memory_mb(param_M, quant)
    usable = available * MEMORY_SAFETY_MARGIN

    if estimated > usable:
        msg = (f"PREFLIGHT_FAIL: model needs ~{estimated:.0f}MB but only "
               f"{available:.0f}MB available ({usable:.0f}MB usable at "
               f"{MEMORY_SAFETY_MARGIN:.0%} margin)")
        logger.warning(msg)
        return False, msg

    logger.info(f"Preflight OK: ~{estimated:.0f}MB needed, "
                f"{available:.0f}MB available")
    return True, ""


# ── Safe model loading / unloading ────────────────────────────────────────────

def safe_load_model(model_id: str,
                    quant: Optional[str] = None,
                    family: Optional[str] = None,
                    param_M: Optional[float] = None,
                    ) -> Tuple[Any, Any, Any, str, Optional[str]]:
    """Load a model with OOM protection.

    If *param_M* is given, runs a preflight memory check before attempting
    the load — avoids the costly OOM crash entirely.

    Returns ``(model, processor, meta, status, error_msg)``.
    On failure *model*, *processor*, *meta* are all ``None``.
    """
    # Preflight: refuse to even try if the model clearly won't fit
    if param_M is not None:
        ok, msg = preflight_memory_check(param_M, quant)
        if not ok:
            return None, None, None, "oom", msg

    # Import here to keep safety.py importable without the full project on sys.path
    from models.model_loader import load_model

    try:
        model, processor, meta = load_model(model_id, quant=quant, family=family)

        # Post-load check: did loading leave the system in a critical state?
        free_after = get_available_memory_mb()
        if free_after < CRITICAL_FREE_MB:
            logger.warning(f"Only {free_after:.0f} MB free after load — unloading")
            try:
                from models.model_loader import unload_model
                unload_model(model)
            except Exception:
                pass
            _emergency_cleanup()
            return None, None, None, "oom", (
                f"MEM_CRITICAL: Only {free_after:.0f} MB free after load"
            )

        return model, processor, meta, "ok", None
    except torch.cuda.OutOfMemoryError:
        _emergency_cleanup()
        return None, None, None, "oom", "CUDA OOM during model loading"
    except RuntimeError as e:
        _emergency_cleanup()
        if "out of memory" in str(e).lower():
            return None, None, None, "oom", str(e)
        return None, None, None, "error", traceback.format_exc()
    except Exception:
        _emergency_cleanup()
        return None, None, None, "error", traceback.format_exc()


def safe_unload(model: Any) -> None:
    """Unload model and aggressively free memory."""
    try:
        from models.model_loader import unload_model
        unload_model(model)
    except Exception:
        pass
    _emergency_cleanup()


def _emergency_cleanup():
    """Aggressively free GPU and system memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    gc.collect()
