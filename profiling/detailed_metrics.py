"""
Detailed Stage-Level and Component-Level Metrics

Data classes for storing granular profiling results:
- Per-stage timing (vision encoding, projection, prefill, decode)
- Per-component timing (attention, feedforward, normalization, etc.)
- Per-token timing with category breakdowns
- Memory and power measurements
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class DetailedMetrics:
    """Stage-level profiling metrics for a single VLM inference run."""

    # Image processing
    image_preprocessing_ms: float = 0.0
    image_size: Tuple[int, int] = (0, 0)

    # Text tokenization
    text_tokenization_ms: float = 0.0

    # Vision pipeline
    vision_encoder_ms: float = 0.0
    projection_ms: float = 0.0

    # Generation stages
    prefill_ms: float = 0.0      # First token (full forward pass)
    decode_ms: float = 0.0       # All subsequent tokens
    per_token_ms: List[float] = field(default_factory=list)
    avg_token_ms: float = 0.0
    total_generation_ms: float = 0.0

    # Token counts
    num_input_tokens: int = 0
    num_generated_tokens: int = 0

    # Post-processing
    decoding_ms: float = 0.0     # Token ID -> text

    # Memory (MB)
    gpu_mem_before_mb: float = 0.0
    gpu_mem_after_mb: float = 0.0
    peak_memory_mb: float = 0.0

    # Power/thermal (Jetson)
    avg_power_w: float = 0.0
    peak_power_w: float = 0.0
    avg_gpu_temp_c: float = 0.0
    peak_gpu_temp_c: float = 0.0

    # Total
    total_ms: float = 0.0

    # Model info
    model_id: str = ""
    method: str = ""    # "fp16", "pytorch_int8", "hqq_int4", etc.
    family: str = ""

    # Output
    generated_text: str = ""

    def compute_derived(self):
        """Compute derived metrics from raw timings."""
        if self.per_token_ms:
            self.prefill_ms = self.per_token_ms[0] if self.per_token_ms else 0.0
            self.decode_ms = sum(self.per_token_ms[1:]) if len(self.per_token_ms) > 1 else 0.0
            self.total_generation_ms = sum(self.per_token_ms)
            self.num_generated_tokens = len(self.per_token_ms)
            if self.num_generated_tokens > 0:
                self.avg_token_ms = self.total_generation_ms / self.num_generated_tokens

    def throughput_tok_s(self) -> float:
        if self.total_generation_ms > 0 and self.num_generated_tokens > 0:
            return self.num_generated_tokens / (self.total_generation_ms / 1000.0)
        return 0.0

    def decode_throughput_tok_s(self) -> float:
        """Decode-only throughput (excludes prefill)."""
        n_decode = self.num_generated_tokens - 1
        if self.decode_ms > 0 and n_decode > 0:
            return n_decode / (self.decode_ms / 1000.0)
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, float):
                data[key] = round(value, 4)
            elif isinstance(value, list) and value and isinstance(value[0], float):
                data[key] = [round(v, 4) for v in value]
        data['throughput_tok_s'] = round(self.throughput_tok_s(), 2)
        data['decode_throughput_tok_s'] = round(self.decode_throughput_tok_s(), 2)
        return data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            f.write(self.to_json())

    def summary(self) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append(f"Profiling: {self.model_id} [{self.method}]")
        lines.append("=" * 70)
        lines.append(f"  Input tokens:       {self.num_input_tokens}")
        lines.append(f"  Generated tokens:   {self.num_generated_tokens}")
        lines.append("")
        lines.append(f"  Preprocessing:      {self.image_preprocessing_ms:8.2f} ms")
        lines.append(f"  Tokenization:       {self.text_tokenization_ms:8.2f} ms")
        lines.append(f"  Prefill (1st tok):  {self.prefill_ms:8.2f} ms")
        if self.num_generated_tokens > 1:
            lines.append(f"  Decode ({self.num_generated_tokens-1} tok):    {self.decode_ms:8.2f} ms")
            lines.append(f"  Avg decode tok:     {self.avg_token_ms:8.2f} ms")
        lines.append(f"  Total generation:   {self.total_generation_ms:8.2f} ms")
        lines.append(f"  Decoding (detok):   {self.decoding_ms:8.2f} ms")
        lines.append(f"  Total:              {self.total_ms:8.2f} ms")
        lines.append("")
        lines.append(f"  Throughput:         {self.throughput_tok_s():8.2f} tok/s")
        lines.append(f"  Decode throughput:  {self.decode_throughput_tok_s():8.2f} tok/s")
        lines.append(f"  Peak memory:        {self.peak_memory_mb:8.1f} MB")
        if self.avg_power_w > 0:
            lines.append(f"  Avg power:          {self.avg_power_w:8.1f} W")
            lines.append(f"  Avg GPU temp:       {self.avg_gpu_temp_c:8.1f} C")
        lines.append("=" * 70)
        return "\n".join(lines)


@dataclass
class CategoryTiming:
    """Aggregated timing for a category of modules."""
    category: str
    total_ms: float
    avg_ms: float
    percentage: float
    call_count: int
    modules: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'category': self.category,
            'total_ms': round(self.total_ms, 2),
            'avg_ms': round(self.avg_ms, 2),
            'percentage': round(self.percentage, 2),
            'call_count': self.call_count,
            'num_modules': len(self.modules)
        }


@dataclass
class ComponentMetrics:
    """Complete component-level profiling metrics."""

    category_timings: Dict[str, CategoryTiming] = field(default_factory=dict)
    per_token_categories: List[Dict[str, float]] = field(default_factory=list)
    total_inference_ms: float = 0.0
    num_runs: int = 1

    def summary_table(self) -> str:
        lines = []
        lines.append("\nComponent-Level Breakdown:")
        lines.append("=" * 80)
        lines.append(f"{'Category':<25} {'Time (ms)':<12} {'Percentage':<12} {'Calls':<8}")
        lines.append("-" * 80)

        sorted_cats = sorted(
            self.category_timings.items(),
            key=lambda x: x[1].total_ms,
            reverse=True
        )
        for category, timing in sorted_cats:
            lines.append(
                f"{category:<25} {timing.total_ms:<12.2f} "
                f"{timing.percentage:<12.1f}% {timing.call_count:<8}"
            )

        lines.append("=" * 80)
        lines.append(f"Total inference time: {self.total_inference_ms:.2f}ms")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            'category_timings': {
                cat: timing.to_dict()
                for cat, timing in self.category_timings.items()
            },
            'per_token_categories': self.per_token_categories,
            'total_inference_ms': round(self.total_inference_ms, 2),
            'num_runs': self.num_runs
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            f.write(self.to_json())
