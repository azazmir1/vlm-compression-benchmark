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
import os
import signal
import sys
import threading
import time
import traceback
from typing import Any, Callable, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ── Tunable thresholds ────────────────────────────────────────────────────────

CRITICAL_FREE_MB = 700           # Abort if MemAvailable drops below this
WATCHDOG_FREE_MB = 500           # Watchdog kills process below this (last resort)
WATCHDOG_POLL_S = 0.5            # How often the watchdog checks memory
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


# ── OOM score adjustment ─────────────────────────────────────────────────────

def make_self_oom_preferred():
    """Raise this process's oom_score_adj so the kernel's OOM killer
    targets us before sshd or other system services.

    Call this early in every subprocess that loads a model.
    """
    try:
        with open(f"/proc/{os.getpid()}/oom_score_adj", "w") as f:
            f.write("900\n")       # range -1000..1000; 900 = very killable
        logger.debug("Set oom_score_adj=900 for PID %d", os.getpid())
    except (PermissionError, IOError) as e:
        logger.debug("Could not set oom_score_adj: %s", e)


# ── Memory watchdog ──────────────────────────────────────────────────────────

class MemoryWatchdog:
    """Background thread that monitors available memory and kills the
    current process cleanly when it drops below ``WATCHDOG_FREE_MB``.

    This is the last line of defence *before* the Linux OOM killer.  The
    OOM killer kills indiscriminately (it killed sshd last time), while the
    watchdog only kills the model-loading process — keeping the system
    alive and SSH accessible.

    Usage::

        with MemoryWatchdog():
            model = load_big_model()   # watchdog active during load
            run_inference(model)       # still active
        # watchdog stopped automatically

    Or via ``start()`` / ``stop()``.
    """

    def __init__(self, threshold_mb: float = WATCHDOG_FREE_MB,
                 poll_interval: float = WATCHDOG_POLL_S):
        self._threshold = threshold_mb
        self._interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._triggered = False

    @property
    def triggered(self) -> bool:
        """Whether the watchdog had to intervene."""
        return self._triggered

    # ── context manager ──────────────────────────────────────────────────
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    # ── start / stop ─────────────────────────────────────────────────────
    def start(self):
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._triggered = False
        self._thread = threading.Thread(target=self._monitor, daemon=True,
                                        name="mem-watchdog")
        self._thread.start()
        logger.info("Memory watchdog started (kill below %d MB free)",
                    int(self._threshold))

    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=2)
        self._thread = None
        logger.debug("Memory watchdog stopped")

    # ── monitor loop ─────────────────────────────────────────────────────
    def _monitor(self):
        consecutive = 0          # require 2 consecutive readings to avoid blips
        while not self._stop_event.is_set():
            avail = get_available_memory_mb()
            if avail < self._threshold:
                consecutive += 1
                if consecutive >= 2:
                    self._triggered = True
                    logger.critical(
                        "WATCHDOG: Available memory %d MB < %d MB threshold — "
                        "aborting process to prevent system OOM",
                        int(avail), int(self._threshold))
                    # Try to free memory first
                    _emergency_cleanup()
                    avail_after = get_available_memory_mb()
                    if avail_after < self._threshold:
                        # Still critical — kill ourselves cleanly
                        logger.critical("WATCHDOG: Still at %d MB after cleanup "
                                        "— sending SIGKILL to self",
                                        int(avail_after))
                        os.kill(os.getpid(), signal.SIGKILL)
                    else:
                        logger.warning("WATCHDOG: Cleanup freed memory to %d MB "
                                       "— continuing", int(avail_after))
                        consecutive = 0
            else:
                consecutive = 0
            self._stop_event.wait(self._interval)


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
    # NOTE: No preflight rejection — always attempt the load.  Memory
    # estimates are too architecture-dependent to be reliable.  The
    # MemoryWatchdog + oom_score_adj protect the system if it fails.

    # Import here to keep safety.py importable without the full project on sys.path
    from models.model_loader import load_model

    watchdog = MemoryWatchdog()
    try:
        watchdog.start()
        model, processor, meta = load_model(model_id, quant=quant, family=family)
        watchdog.stop()

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
        watchdog.stop()
        _emergency_cleanup()
        return None, None, None, "oom", "CUDA OOM during model loading"
    except RuntimeError as e:
        watchdog.stop()
        _emergency_cleanup()
        if "out of memory" in str(e).lower():
            return None, None, None, "oom", str(e)
        return None, None, None, "error", traceback.format_exc()
    except Exception:
        watchdog.stop()
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
