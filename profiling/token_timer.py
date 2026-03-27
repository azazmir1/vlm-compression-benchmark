"""
TokenTimingProcessor — Accurate per-token timing via LogitsProcessor

Uses HuggingFace's LogitsProcessor API to record a CUDA event at every
token generation step. This gives exact prefill vs decode separation
without requiring streaming or TextIteratorStreamer.

Works universally with all HuggingFace models that use .generate().

Usage:
    timer = TokenTimingProcessor()
    output = model.generate(**inputs, logits_processor=[timer])
    timer.finalize()  # sync CUDA and compute timings

    print(f"Prefill: {timer.prefill_ms:.1f} ms")
    print(f"Per-token: {timer.per_token_ms}")
"""

import torch
import time
from typing import List, Optional
from transformers import LogitsProcessor


class TokenTimingProcessor(LogitsProcessor):
    """
    Records CUDA events at each token generation step.

    After generation completes, call finalize() to synchronize CUDA
    and compute per-token timings.
    """

    def __init__(self):
        self.events: List[torch.cuda.Event] = []
        self.cpu_times: List[float] = []  # fallback for CPU
        self.use_cuda = torch.cuda.is_available()
        self._finalized = False
        self._end_event = None
        self._end_time = None
        self._wall_total_ms = 0.0

        # Results (populated after finalize())
        self.per_token_ms: List[float] = []
        self.prefill_ms: float = 0.0
        self.decode_ms: float = 0.0
        self.total_ms: float = 0.0
        self.num_tokens: int = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Called by generate() after each forward pass, before sampling."""
        if self.use_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.events.append(event)
        else:
            self.cpu_times.append(time.perf_counter())
        return scores  # pass through unchanged

    def record_start(self):
        """Record the start time (call this right before model.generate())."""
        if self.use_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.events.insert(0, event)
        else:
            self.cpu_times.insert(0, time.perf_counter())

    def record_end(self):
        """Record wall-clock end time (for total elapsed, NOT counted as a token)."""
        if self.use_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._end_event = event
        else:
            self._end_time = time.perf_counter()

    def finalize(self):
        """Synchronize CUDA and compute per-token timings.

        Event layout:
          events[0] = record_start() — before generate()
          events[1] = __call__ after prefill forward pass
          events[2] = __call__ after decode step 1
          ...
          events[N] = __call__ after decode step N-1

        Intervals:
          [0→1] = prefill time (first token)
          [1→2] = decode token 1
          ...
          [(N-1)→N] = decode token N-1

        record_end() is stored separately — NOT appended to events —
        to avoid creating a phantom last "token" that is just generate()
        exit overhead.
        """
        if self._finalized:
            return

        if self.use_cuda:
            torch.cuda.synchronize()
            for i in range(1, len(self.events)):
                try:
                    dt = self.events[i - 1].elapsed_time(self.events[i])
                    self.per_token_ms.append(dt)
                except RuntimeError:
                    self.per_token_ms.append(0.0)
            # Compute wall-clock total from start to end (not sum of tokens)
            if hasattr(self, '_end_event') and self._end_event and len(self.events) > 0:
                try:
                    self._wall_total_ms = self.events[0].elapsed_time(self._end_event)
                except RuntimeError:
                    self._wall_total_ms = 0.0
        else:
            for i in range(1, len(self.cpu_times)):
                dt = (self.cpu_times[i] - self.cpu_times[i - 1]) * 1000.0
                self.per_token_ms.append(dt)
            if hasattr(self, '_end_time') and self._end_time and self.cpu_times:
                self._wall_total_ms = (self._end_time - self.cpu_times[0]) * 1000.0

        self.num_tokens = len(self.per_token_ms)
        if self.num_tokens > 0:
            self.prefill_ms = self.per_token_ms[0]
            self.decode_ms = sum(self.per_token_ms[1:])
            self.total_ms = sum(self.per_token_ms)

        self._finalized = True

    def reset(self):
        """Reset for reuse across multiple runs."""
        self.events.clear()
        self.cpu_times.clear()
        self.per_token_ms.clear()
        self.prefill_ms = 0.0
        self.decode_ms = 0.0
        self.total_ms = 0.0
        self.num_tokens = 0
        self._finalized = False
        self._end_event = None
        self._end_time = None
        self._wall_total_ms = 0.0

    def summary(self) -> str:
        if not self._finalized:
            self.finalize()
        lines = [
            f"TokenTimer: {self.num_tokens} tokens",
            f"  Prefill:     {self.prefill_ms:8.2f} ms",
        ]
        if self.num_tokens > 1:
            avg_decode = self.decode_ms / (self.num_tokens - 1)
            lines.append(f"  Decode avg:  {avg_decode:8.2f} ms/tok ({self.num_tokens-1} tokens)")
            lines.append(f"  Decode tot:  {self.decode_ms:8.2f} ms")
        lines.append(f"  Total:       {self.total_ms:8.2f} ms")
        if self.total_ms > 0:
            throughput = self.num_tokens / (self.total_ms / 1000.0)
            lines.append(f"  Throughput:  {throughput:8.2f} tok/s")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        if not self._finalized:
            self.finalize()
        result = {
            'num_tokens': self.num_tokens,
            'prefill_ms': round(self.prefill_ms, 4),
            'decode_ms': round(self.decode_ms, 4),
            'total_ms': round(self.total_ms, 4),
            'per_token_ms': [round(t, 4) for t in self.per_token_ms],
        }
        if self.num_tokens > 1:
            result['avg_decode_ms'] = round(self.decode_ms / (self.num_tokens - 1), 4)
            result['decode_throughput_tok_s'] = round(
                (self.num_tokens - 1) / (self.decode_ms / 1000.0), 2
            ) if self.decode_ms > 0 else 0.0
        return result
