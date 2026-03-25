"""
profiling/cpu_profiler.py
=========================
Context-manager CPU/RAM profiler for CPU execution paths.

Mirrors the GPUProfiler interface so callers can swap ProfilerCls
without changing evaluation code.

Tracks:
  - Peak RSS memory (MB) via psutil
  - Average RSS memory (MB)
  - Wall-clock time (s)
  - Time-to-first-token (set manually via .mark_first_token())

Usage:
    from profiling.cpu_profiler import CPUProfiler

    profiler = CPUProfiler(poll_interval_ms=100)
    with profiler:
        output = session.run(...)
        profiler.mark_first_token()

    stats = profiler.stats()
"""

import os
import time
import threading
from dataclasses import dataclass, asdict
from typing import Optional

try:
    import psutil
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


class CPUProfiler:
    """
    Thread-based CPU/RAM profiler.  Use as a context manager.

    Parameters
    ----------
    poll_interval_ms : sampling interval in milliseconds (default 100 ms)
    """

    def __init__(self, poll_interval_ms: int = 100, **kwargs):
        self.poll_interval_s = poll_interval_ms / 1000.0
        self._process = psutil.Process(os.getpid()) if _PSUTIL_OK else None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._mem_samples: list[float] = []
        self._t_start: float = 0.0
        self._t_end: float = 0.0
        self._t_first_token: Optional[float] = None

    def __enter__(self):
        self._reset()
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

    def mark_first_token(self):
        """Call this immediately after the first token is produced."""
        self._t_first_token = time.perf_counter()

    def stats(self) -> ProfilerStats:
        n = len(self._mem_samples)
        ttft = None
        if self._t_first_token is not None:
            ttft = self._t_first_token - self._t_start

        return ProfilerStats(
            wall_time_s=self._t_end - self._t_start,
            time_to_first_token_s=ttft,
            peak_memory_mb=max(self._mem_samples) if n else 0.0,
            avg_memory_mb=sum(self._mem_samples) / n if n else 0.0,
            num_samples=n,
        )

    def _reset(self):
        self._mem_samples.clear()
        self._t_first_token = None

    def _poll_loop(self):
        while not self._stop_event.is_set():
            if self._process is not None:
                try:
                    rss = self._process.memory_info().rss / 1024**2
                    self._mem_samples.append(rss)
                except Exception:
                    pass
            self._stop_event.wait(timeout=self.poll_interval_s)
