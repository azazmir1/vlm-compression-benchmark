"""
compression/ptq/run_pytorch_int8_cpu.py
========================================
CPU-only INT8 quantization — pure PyTorch, no GPU required.

Per-channel symmetric INT8 quantization of Linear layers:
  1. Load model on CPU
  2. For each Linear layer (skipping vision encoder):
     a. Compute per-channel scale = abs_max / 127
     b. Round weights to INT8
     c. Replace with Int8Linear (stores int8 weights + fp32 scale)
  3. Vision encoder stays FP32 for accuracy

~2x compression vs FP32 (1 byte/param vs 4 bytes/param).

Usage:
  python compression/ptq/run_pytorch_int8_cpu.py --model_id HuggingFaceTB/SmolVLM2-2.2B-Instruct --vqav2_n 20
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import psutil
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.model_loader import detect_family
from evaluation.run_baseline import (
    load_vqav2,
    evaluate_dataset,
    _vqa_accuracy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "pytorch_int8_cpu"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

VISION_KEYWORDS = [
    "vision_model", "vision_tower", "visual", "vit", "image_encoder",
    "img_encoder", "vision_encoder", "patch_embed", "pixel_shuffle",
    "connector", "multi_modal_projector", "mlp1",
]


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def _is_vision(name: str) -> bool:
    return any(kw in name.lower() for kw in VISION_KEYWORDS)


# ── Int8Linear: quantized linear layer ──────────────────────────────────────

class Int8Linear(nn.Module):
    """Linear layer with INT8 weight storage (per-channel symmetric).

    Stores weights as int8 + fp32 scale. Dequantizes on the fly during forward.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight_int8", torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        self.register_buffer(
            "scale", torch.zeros(out_features, 1, dtype=torch.float32)
        )
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None

    def forward(self, x):
        dtype = x.dtype
        w = self.weight_int8.to(dtype) * self.scale.to(dtype)
        return nn.functional.linear(x, w, self.bias.to(dtype) if self.bias is not None else None)

    def quantize(self, weight: torch.Tensor):
        """Symmetric per-channel INT8 quantization on CPU."""
        w = weight.float()
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        w_int8 = (w / scale).round().clamp(-128, 127).to(torch.int8)
        self.weight_int8 = w_int8
        self.scale = scale.to(torch.float32)

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, bias={self.bias is not None}"


# ── Loader ───────────────────────────────────────────────────────────────────

def load_model_int8_cpu(model_id: str, family: str = None, skip_vision: bool = True):
    """Load model with CPU INT8 quantization.

    Returns: (model, processor, meta_dict)
    """
    from models.model_loader import load_model as _load_model_standard

    if family is None:
        family = detect_family(model_id)

    mem_before = _rss_mb()
    logger.info(f"Loading {model_id} with CPU INT8 quantization...")
    logger.info(f"  Family: {family}")
    logger.info(f"  RAM before: {mem_before:.1f} MB")

    t0 = time.time()

    # Step 1: Load on CPU in FP32
    logger.info("  Step 1: Loading model on CPU...")
    t_load = time.time()
    model, processor, load_meta = _load_model_standard(model_id)
    t_load = time.time() - t_load
    logger.info(f"  Loaded in {t_load:.1f}s. RAM: {_rss_mb():.1f} MB")

    # Step 2: Quantize Linear layers to INT8
    logger.info("  Step 2: Quantizing Linear layers to INT8...")
    t_quant = time.time()
    n_replaced = 0

    for mod_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full_name = f"{mod_name}.{child_name}" if mod_name else child_name
            if skip_vision and _is_vision(full_name):
                continue

            int8_layer = Int8Linear(
                child.in_features, child.out_features,
                bias=child.bias is not None,
            )
            int8_layer.quantize(child.weight.data)

            if child.bias is not None:
                int8_layer.bias = child.bias.data.float()

            setattr(parent, child_name, int8_layer)
            n_replaced += 1
            del child

            if n_replaced % 50 == 0:
                gc.collect()
                logger.info(f"    ... quantized {n_replaced} layers ({_rss_mb():.0f} MB RAM)")

    t_quant = time.time() - t_quant
    gc.collect()
    logger.info(f"  Quantized {n_replaced} layers in {t_quant:.1f}s")

    model.eval()
    load_time = time.time() - t0
    mem_after = _rss_mb()

    logger.info(f"  RAM after: {mem_after:.1f} MB (delta: {mem_after - mem_before:.1f} MB)")
    logger.info(f"  Total load time: {load_time:.1f}s")

    n_params = sum(p.numel() for p in model.parameters())
    for name, buf in model.named_buffers():
        if "weight_int8" in name:
            n_params += buf.numel()

    meta = {
        "model_id": model_id,
        "family": family,
        "method": "pytorch_int8_cpu",
        "quant_bits": 8,
        "skip_vision_quant": skip_vision,
        "n_linear_quantized": n_replaced,
        "num_params_M": round(n_params / 1e6, 1),
        "ram_before_mb": round(mem_before, 1),
        "ram_after_mb": round(mem_after, 1),
        "ram_delta_mb": round(mem_after - mem_before, 1),
        "load_time_s": round(load_time, 1),
        "timing": {
            "model_load_s": round(t_load, 1),
            "quantization_s": round(t_quant, 1),
        },
    }

    return model, processor, meta


def unload_model(model):
    del model
    gc.collect()
    logger.info("Model unloaded.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CPU-only PyTorch INT8 quantization"
    )
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--vqav2_n", type=int, default=20)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_id = args.model_id
    family = detect_family(model_id)
    safe_name = model_id.replace("/", "__")

    out_path = RESULTS_DIR / f"{safe_name}__pytorch_int8_cpu.json"
    if out_path.exists() and not args.force:
        logger.info(f"Result exists: {out_path}")
        return

    model, processor, meta = load_model_int8_cpu(model_id, family)

    if not args.skip_eval:
        samples = load_vqav2(args.vqav2_n)
        vqa_result = evaluate_dataset(
            model, processor, samples, family, "cpu",
            "vqav2", _vqa_accuracy,
        )
        meta["benchmarks"] = {"vqav2": vqa_result}
    else:
        meta["benchmarks"] = {}

    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    unload_model(model)


if __name__ == "__main__":
    main()
