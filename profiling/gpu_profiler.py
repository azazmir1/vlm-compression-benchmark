"""
profiling/gpu_profiler.py
=========================
Context-manager GPU profiler using pynvml.

Tracks:
  - Peak GPU memory (MB)
  - Average & peak power draw (W)   [0 on Jetson — NVML power unsupported]
  - Average GPU utilization (%)
  - Wall-clock time (s)
  - Time-to-first-token (set manually via .mark_first_token())
  - Total inference time

Jetson Orin Nano notes:
  - Memory is unified (CPU+GPU share the same 8 GB pool).
    pynvml reports the GPU-side allocation; torch.cuda.memory_allocated()
    is used as a complementary measure.
  - nvmlDeviceGetPowerUsage() is not supported on Jetson iGPUs; power
    samples will be empty and reported as 0.0 W.

Usage:
    from profiling.gpu_profiler import GPUProfiler

    profiler = GPUProfiler(device_index=0, poll_interval_ms=50)
    with profiler:
        output = model.generate(...)
        profiler.mark_first_token()   # call right after first token ready

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
    import torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


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

        if not _PYNVML_OK:
            import warnings
            warnings.warn("pynvml not available; GPU profiling disabled.")

    # ── Context manager ────────────────────────────────────────────────────

    def __enter__(self):
        self._reset()
        if _PYNVML_OK:
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
        if _PYNVML_OK:
            pynvml.nvmlShutdown()

    # ── Public helpers ─────────────────────────────────────────────────────

    def mark_first_token(self):
        """Call this immediately after the first token is produced."""
        self._t_first_token = time.perf_counter()

    def stats(self) -> ProfilerStats:
        n_mem  = len(self._mem_samples)
        n_pow  = len(self._power_samples)
        n_util = len(self._util_samples)
        n      = max(n_mem, n_pow, n_util)

        if n == 0:
            return ProfilerStats(wall_time_s=self._t_end - self._t_start)

        ttft = None
        if self._t_first_token is not None:
            ttft = self._t_first_token - self._t_start

        return ProfilerStats(
            wall_time_s=self._t_end - self._t_start,
            time_to_first_token_s=ttft,
            peak_memory_mb=max(self._mem_samples)           if n_mem  else 0.0,
            avg_memory_mb=sum(self._mem_samples) / n_mem    if n_mem  else 0.0,
            peak_power_w=max(self._power_samples)           if n_pow  else 0.0,
            avg_power_w=sum(self._power_samples) / n_pow    if n_pow  else 0.0,
            avg_gpu_util_pct=sum(self._util_samples)/n_util if n_util else 0.0,
            peak_gpu_util_pct=max(self._util_samples)       if n_util else 0.0,
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
            # Memory: prefer NVML (whole-pool view on Jetson unified memory);
            # fall back to torch allocator if NVML unavailable.
            mem_mb: Optional[float] = None
            if self._handle is not None and _PYNVML_OK:
                try:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                    mem_mb = mem.used / 1024**2
                except pynvml.NVMLError:
                    pass
            if mem_mb is None and _TORCH_OK:
                mem_mb = torch.cuda.memory_allocated() / 1024**2
            if mem_mb is not None:
                self._mem_samples.append(mem_mb)

            # Power: not supported on Jetson iGPU — silently skip on NVMLError
            if self._handle is not None and _PYNVML_OK:
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(self._handle)   # mW
                    self._power_samples.append(power / 1000.0)             # → W
                except pynvml.NVMLError:
                    pass  # unsupported on Jetson; power stays empty → reported as 0.0 W

            # Utilisation
            if self._handle is not None and _PYNVML_OK:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                    self._util_samples.append(util.gpu)
                except pynvml.NVMLError:
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
