"""
Timing Tracker for Component-Level Profiling

Uses CUDA events to accurately time GPU operations with minimal overhead (~1-2%).
Avoids synchronization during inference - only synchronizes once at the end.
"""

import torch
import time
from typing import Dict, Optional, List
from collections import defaultdict


class TimingTracker:
    """
    Tracks timing for module forward passes using CUDA events.

    Usage:
        tracker = TimingTracker()
        tracker.record_start("module_name", token_idx=0)
        # ... module forward pass ...
        tracker.record_end("module_name", token_idx=0)
        tracker.compute_timings()  # Synchronize and compute
        timings = tracker.timings
    """

    def __init__(self):
        self.events = {}  # key -> [start, end, token_idx]
        self.timings = {}  # key -> {'elapsed_ms': float, 'token_idx': int}
        self.use_cuda = torch.cuda.is_available()

    def record_start(self, module_name: str, token_idx: Optional[int] = None):
        if not self.use_cuda:
            key = self._make_key(module_name, token_idx)
            self.events[key] = [time.perf_counter(), None, token_idx]
            return

        start = torch.cuda.Event(enable_timing=True)
        start.record()
        key = self._make_key(module_name, token_idx)
        self.events[key] = [start, None, token_idx]

    def record_end(self, module_name: str, token_idx: Optional[int] = None):
        if not self.use_cuda:
            key = self._make_key(module_name, token_idx)
            if key in self.events:
                self.events[key][1] = time.perf_counter()
            return

        end = torch.cuda.Event(enable_timing=True)
        end.record()
        key = self._make_key(module_name, token_idx)
        if key in self.events:
            self.events[key][1] = end

    def compute_timings(self):
        """Synchronize and compute all elapsed times. Call once after inference."""
        if not self.use_cuda:
            for key, (start, end, token_idx) in self.events.items():
                if start is not None and end is not None:
                    elapsed_ms = (end - start) * 1000
                    self.timings[key] = {
                        'elapsed_ms': elapsed_ms,
                        'token_idx': token_idx
                    }
            return

        torch.cuda.synchronize()

        for key, (start, end, token_idx) in self.events.items():
            if start is not None and end is not None:
                try:
                    elapsed_ms = start.elapsed_time(end)
                    self.timings[key] = {
                        'elapsed_ms': elapsed_ms,
                        'token_idx': token_idx
                    }
                except RuntimeError:
                    pass

    def clear(self):
        self.events.clear()
        self.timings.clear()

    def get_timing(self, module_name: str, token_idx: Optional[int] = None) -> Optional[float]:
        key = self._make_key(module_name, token_idx)
        timing_data = self.timings.get(key)
        return timing_data['elapsed_ms'] if timing_data else None

    def get_all_timings(self) -> Dict[str, Dict]:
        return self.timings.copy()

    def get_summary(self) -> Dict[str, Dict]:
        """Get summary statistics grouped by module name (ignoring token index)."""
        if not self.timings:
            return {}

        module_timings = defaultdict(list)
        for key, data in self.timings.items():
            if '_t' in key and key.split('_t')[-1].isdigit():
                module_name = key.rsplit('_t', 1)[0]
            else:
                module_name = key
            module_timings[module_name].append(data['elapsed_ms'])

        summary = {}
        for module_name, times in module_timings.items():
            summary[module_name] = {
                'total_ms': sum(times),
                'avg_ms': sum(times) / len(times),
                'min_ms': min(times),
                'max_ms': max(times),
                'count': len(times)
            }
        return summary

    def _make_key(self, module_name: str, token_idx: Optional[int]) -> str:
        if token_idx is not None:
            return f"{module_name}_t{token_idx}"
        return module_name

    def __len__(self) -> int:
        return len(self.timings)

    def __repr__(self) -> str:
        return f"TimingTracker(events={len(self.events)}, timings={len(self.timings)})"
