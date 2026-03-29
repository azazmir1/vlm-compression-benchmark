"""
profiling/gpu_profiler.py
=========================
Context-manager profiler — GPU (pynvml) with psutil RAM fallback.

On Raspberry Pi / CPU-only systems pynvml is unavailable; the profiler
transparently falls back to tracking RSS process memory via psutil.

Tracks:
  - Peak memory (GPU MB if CUDA, else RSS RAM MB)
  - Average power draw (W) — GPU only; 0.0 on CPU
  - Average utilization (%) — GPU only; 0.0 on CPU
  - Wall-clock time (s)
  - Time-to-first-token (set manually via .mark_first_token())

Usage:
    from profiling.gpu_profiler import GPUProfiler

    profiler = GPUProfiler(device_index=0, poll_interval_ms=50)
    with profiler:
        output = model.generate(...)
        profiler.mark_first_token()

    stats = profiler.stats()
    print(stats)
"""

import time
import threading
from dataclasses import dataclass, field, asdict
from typing import Optional

try:
    import pynvml
    _PYNVML_OK = True
except ImportError:
    _PYNVML_OK = False

try:
    import psutil
    import os as _os
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False


@dataclass
class ProfilerStats:
    wall_time_s: float = 0.0
    time_to_first_token_s: Optional[float] = None
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    peak_power_w: float = 0.0
    avg_power_w: float = 0.0
    avg_gpu_util_pct: float = 0.0
    peak_gpu_util_pct: float = 0.0
    num_samples: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class GPUProfiler:
    """
    Thread-based GPU profiler.  Use as a context manager.

    Parameters
    ----------
    device_index    : GPU index (default 0)
    poll_interval_ms: sampling interval in milliseconds (default 50 ms)
    """

    def __init__(self, device_index: int = 0, poll_interval_ms: int = 50):
        self.device_index = device_index
        self.poll_interval_s = poll_interval_ms / 1000.0
        self._handle = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Accumulated data
        self._mem_samples: list[float] = []
        self._power_samples: list[float] = []
        self._util_samples: list[float] = []

        self._t_start: float = 0.0
        self._t_end: float = 0.0
        self._t_first_token: Optional[float] = None

        # Determine mode
        self._use_gpu = _PYNVML_OK
        self._use_cpu = (not _PYNVML_OK) and _PSUTIL_OK
        if not self._use_gpu and not self._use_cpu:
            import warnings
            warnings.warn("Neither pynvml nor psutil available; profiling disabled.")
        if self._use_cpu:
            self._process = psutil.Process(_os.getpid())

    # ── Context manager ────────────────────────────────────────────────────

    def __enter__(self):
        self._reset()
        if self._use_gpu:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        self._t_start = time.perf_counter()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._t_end = time.perf_counter()
        if self._use_gpu:
            pynvml.nvmlShutdown()

    # ── Public helpers ─────────────────────────────────────────────────────

    def mark_first_token(self):
        """Call this immediately after the first token is produced."""
        self._t_first_token = time.perf_counter()

    def stats(self) -> ProfilerStats:
        n = len(self._mem_samples)
        if n == 0:
            return ProfilerStats(wall_time_s=self._t_end - self._t_start)

        ttft = None
        if self._t_first_token is not None:
            ttft = self._t_first_token - self._t_start

        return ProfilerStats(
            wall_time_s=self._t_end - self._t_start,
            time_to_first_token_s=ttft,
            peak_memory_mb=max(self._mem_samples),
            avg_memory_mb=sum(self._mem_samples) / n,
            peak_power_w=max(self._power_samples) if self._power_samples else 0.0,
            avg_power_w=sum(self._power_samples) / n if self._power_samples else 0.0,
            avg_gpu_util_pct=sum(self._util_samples) / n if self._util_samples else 0.0,
            peak_gpu_util_pct=max(self._util_samples) if self._util_samples else 0.0,
            num_samples=n,
        )

    # ── Internal polling ───────────────────────────────────────────────────

    def _reset(self):
        self._mem_samples.clear()
        self._power_samples.clear()
        self._util_samples.clear()
        self._t_first_token = None

    def _poll_loop(self):
        while not self._stop_event.is_set():
            if self._use_gpu and self._handle is not None:
                try:
                    mem   = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                    power = pynvml.nvmlDeviceGetPowerUsage(self._handle)   # mW
                    util  = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                    self._mem_samples.append(mem.used / 1024**2)
                    self._power_samples.append(power / 1000.0)             # → W
                    self._util_samples.append(util.gpu)
                except pynvml.NVMLError:
                    pass
            elif self._use_cpu:
                try:
                    rss_mb = self._process.memory_info().rss / 1024**2
                    self._mem_samples.append(rss_mb)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self._stop_event.wait(timeout=self.poll_interval_s)


# ── Convenience wrapper ────────────────────────────────────────────────────

def profile_inference(fn, *args, device_index: int = 0, **kwargs) -> tuple:
    """
    Run fn(*args, **kwargs) inside a GPUProfiler and return (result, stats).

    Example:
        result, stats = profile_inference(model.generate, input_ids, max_new_tokens=50)
    """
    profiler = GPUProfiler(device_index=device_index)
    with profiler:
        result = fn(*args, **kwargs)
    return result, profiler.stats()
