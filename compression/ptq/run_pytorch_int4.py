"""
compression/ptq/run_pytorch_int4.py
====================================
Pure-PyTorch INT4 quantization — the heaviest compression we can do.

This is a **Category 1** method: it reduces model memory at load time so that
models which OOM in FP16 (and even INT8) can fit on memory-constrained devices.

Approach:
  1. Load model on CPU using standard model_loader (handles all VLM families)
  2. Replace nn.Linear layers with Int4Linear (packs 2 weights per byte)
  3. Move quantized model to GPU
  4. Vision encoder stays in FP16 for accuracy
  5. Expected savings: ~75% vs FP16 (0.5 bytes/param vs 2 bytes/param for LLM backbone)

INT4 uses per-group quantization (group_size=128 by default) for better accuracy
than per-channel quantization at this low bit width.

Usage:
  python compression/ptq/run_pytorch_int4.py --model_id Qwen/Qwen2.5-VL-7B-Instruct --vqav2_n 30
"""

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.model_loader import detect_family, _gpu_mem_mb
from evaluation.run_baseline import (
    load_vqav2,
    evaluate_dataset,
    _vqa_accuracy,
    run_inference,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "pytorch_int4"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Int4Linear: quantized linear layer ──────────────────────────────────────

class Int4Linear(nn.Module):
    """Linear layer with INT4 weight storage using per-group quantization.

    Weights are packed: 2 x int4 values per int8 byte.
    Per-group quantization (default group_size=128) for better accuracy.

    Memory: ~0.5 bytes/param (int4 packed) + small scale/zero overhead
    vs 2 bytes/param (fp16) = ~4x compression.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 group_size: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        # Pad in_features to be divisible by group_size
        self.in_features_padded = ((in_features + group_size - 1) // group_size) * group_size
        n_groups = self.in_features_padded // group_size

        # Packed INT4 weights: 2 values per byte
        # Shape: (out_features, in_features_padded // 2)
        packed_size = self.in_features_padded // 2
        self.register_buffer(
            "weight_packed", torch.zeros(out_features, packed_size, dtype=torch.uint8)
        )
        # Per-group scale and zero point
        # Shape: (out_features, n_groups)
        self.register_buffer(
            "scale", torch.zeros(out_features, n_groups, dtype=torch.float16)
        )
        self.register_buffer(
            "zero_point", torch.zeros(out_features, n_groups, dtype=torch.float16)
        )
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def _unpack_weights(self):
        """Unpack INT4 weights from packed uint8 storage."""
        # Each uint8 stores 2 int4 values: high nibble and low nibble
        # Low nibble: weight_packed & 0x0F
        # High nibble: (weight_packed >> 4) & 0x0F
        low = (self.weight_packed & 0x0F).to(torch.int8) - 8   # unsigned [0,15] -> signed [-8,7]
        high = (self.weight_packed >> 4).to(torch.int8) - 8
        # Interleave: [high0, low0, high1, low1, ...]
        unpacked = torch.stack([high, low], dim=-1).reshape(
            self.weight_packed.shape[0], -1
        )
        # Trim padding
        return unpacked[:, :self.in_features_padded]

    def forward(self, x):
        dtype = x.dtype
        # Unpack int4 -> int8
        w_int = self._unpack_weights()  # (out, in_padded)

        # Dequantize per group
        w_int = w_int.reshape(self.out_features, -1, self.group_size)  # (out, n_groups, group_size)
        scale = self.scale.unsqueeze(-1).to(dtype)  # (out, n_groups, 1)
        zero = self.zero_point.unsqueeze(-1).to(dtype)  # (out, n_groups, 1)
        w = (w_int.to(dtype) - zero) * scale  # dequantize
        w = w.reshape(self.out_features, -1)  # (out, in_padded)

        # Trim to actual in_features
        w = w[:, :self.in_features]

        return nn.functional.linear(x, w, self.bias.to(dtype) if self.bias is not None else None)

    def set_weight_from_fp16(self, weight_fp16: torch.Tensor):
        """Quantize a FP16/FP32 weight tensor to INT4 with per-group scale/zero."""
        w = weight_fp16.float()
        out_features, in_features = w.shape

        # Pad if needed
        if in_features < self.in_features_padded:
            w = torch.nn.functional.pad(w, (0, self.in_features_padded - in_features))

        # Reshape into groups
        w_grouped = w.reshape(out_features, -1, self.group_size)  # (out, n_groups, group_size)

        # Per-group min/max for asymmetric quantization
        w_min = w_grouped.amin(dim=-1, keepdim=True)  # (out, n_groups, 1)
        w_max = w_grouped.amax(dim=-1, keepdim=True)

        # Scale and zero point for mapping to [-8, 7] (signed int4)
        scale = (w_max - w_min).clamp(min=1e-8) / 15.0  # 15 = 2^4 - 1
        zero = w_min / scale + 8  # offset so that w_min maps to -8

        # Quantize: q = round(w / scale + 8 - zero) -> range [-8, 7]
        w_q = ((w_grouped / scale) + 8 - zero).round().clamp(-8, 7)

        # Store scale and zero
        self.scale.copy_(scale.squeeze(-1).to(torch.float16))
        self.zero_point.copy_(zero.squeeze(-1).to(torch.float16))

        # Pack: convert signed [-8,7] to unsigned [0,15], then pack 2 per byte
        w_unsigned = (w_q + 8).to(torch.uint8)  # [0, 15]
        w_flat = w_unsigned.reshape(out_features, -1)  # (out, in_padded)

        # Pack pairs: high nibble = even indices, low nibble = odd indices
        even = w_flat[:, 0::2]  # high nibble
        odd = w_flat[:, 1::2]   # low nibble
        packed = (even << 4) | odd
        self.weight_packed.copy_(packed.to(torch.uint8))

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"group_size={self.group_size}, bias={self.bias is not None}")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _count_params(model: nn.Module) -> int:
    total = 0
    for p in model.parameters():
        total += p.numel()
    for name, buf in model.named_buffers():
        if "weight_packed" in name:
            total += buf.numel() * 2  # each byte = 2 params
    return total


VISION_KEYWORDS = [
    "vision_model", "vision_tower", "visual", "vit", "image_encoder",
    "img_encoder", "vision_encoder", "patch_embed", "pixel_shuffle",
    "connector", "multi_modal_projector", "mlp1",
]


def load_model_int4(model_id: str, family: str = None, skip_vision: bool = True,
                    device: str = "cuda", group_size: int = 128):
    """Load a model with INT4 quantization applied during loading.

    Strategy:
      1. Load model on CPU with from_pretrained (proper initialization)
      2. Replace nn.Linear with Int4Linear (quantize weights in-place)
      3. Move quantized model to GPU

    Returns: (model, processor, meta_dict)
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM

    if family is None:
        family = detect_family(model_id)

    mem_before = _gpu_mem_mb()
    logger.info(f"Loading {model_id} with PyTorch INT4 quantization...")
    logger.info(f"  Family: {family}, Device: {device}, Group size: {group_size}")
    logger.info(f"  GPU memory before: {mem_before:.1f} MB")

    t0 = time.time()

    # Step 1: Load model on CPU using standard loader
    from models.model_loader import load_model as _load_model_standard
    logger.info("  Step 1: Loading model in FP16 on CPU...")
    model, processor, load_meta = _load_model_standard(model_id, device_map="cpu")

    mem_after_cpu = _gpu_mem_mb()
    logger.info(f"  CPU load done. GPU mem: {mem_after_cpu:.1f} MB")

    # Step 2: Replace Linear layers with Int4Linear (quantize on CPU)
    logger.info("  Step 2: Quantizing Linear layers to INT4...")
    n_replaced = 0
    for mod_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full_name = f"{mod_name}.{child_name}" if mod_name else child_name
            if skip_vision and any(kw in full_name.lower() for kw in VISION_KEYWORDS):
                continue
            int4_layer = Int4Linear(
                child.in_features, child.out_features,
                bias=child.bias is not None,
                group_size=group_size,
            )
            int4_layer.set_weight_from_fp16(child.weight.data)
            if child.bias is not None:
                int4_layer.bias = child.bias.data.to(torch.float16)
            setattr(parent, child_name, int4_layer)
            n_replaced += 1
            del child
            # Periodic GC to keep CPU memory in check during quantization
            if n_replaced % 50 == 0:
                gc.collect()
                logger.info(f"    ... quantized {n_replaced} layers so far")

    logger.info(f"  Quantized {n_replaced} Linear layers to INT4")

    # Force GC before GPU transfer
    gc.collect()

    # Step 3: Move to GPU
    logger.info("  Step 3: Moving quantized model to GPU...")
    model = model.to(device).eval()

    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    load_time = time.time() - t0
    mem_after = _gpu_mem_mb()

    logger.info(f"  GPU memory after: {mem_after:.1f} MB (delta: {mem_after - mem_before:.1f} MB)")
    logger.info(f"  Load time: {load_time:.1f}s")

    # Count params
    n_params = sum(p.numel() for p in model.parameters())
    for name, buf in model.named_buffers():
        if "weight_packed" in name:
            n_params += buf.numel() * 2

    meta = {
        "model_id": model_id,
        "family": family,
        "method": "pytorch_int4",
        "quant_bits": 4,
        "group_size": group_size,
        "skip_vision_quant": skip_vision,
        "n_linear_quantized": n_replaced,
        "num_params_M": round(n_params / 1e6, 1),
        "gpu_mem_before_mb": round(mem_before, 1),
        "gpu_mem_after_mb": round(mem_after, 1),
        "gpu_mem_load_mb": round(mem_after - mem_before, 1),
        "load_time_s": round(load_time, 1),
    }

    return model, processor, meta


# ── Unload ───────────────────────────────────────────────────────────────────

def unload_model(model):
    """Unload model and free GPU memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model unloaded and GPU cache cleared.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pure-PyTorch INT4 quantization — maximum compression"
    )
    parser.add_argument("--model_id", required=True, help="HuggingFace model ID")
    parser.add_argument("--vqav2_n", type=int, default=30,
                        help="Number of VQAv2 samples for evaluation")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip evaluation, just load and report memory")
    parser.add_argument("--quantize_vision", action="store_true",
                        help="Also quantize vision encoder (default: keep FP16)")
    parser.add_argument("--group_size", type=int, default=128,
                        help="Quantization group size (default: 128)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing results")
    args = parser.parse_args()

    model_id = args.model_id
    family = detect_family(model_id)
    safe_name = model_id.replace("/", "__")

    out_path = RESULTS_DIR / f"{safe_name}__pytorch_int4.json"
    if out_path.exists() and not args.force:
        logger.info(f"Result already exists: {out_path}")
        return

    # Load model with INT4 quantization
    model, processor, meta = load_model_int4(
        model_id, family,
        skip_vision=not args.quantize_vision,
        group_size=args.group_size,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.skip_eval:
        # Evaluate on VQAv2
        samples = load_vqav2(args.vqav2_n)
        vqa_result = evaluate_dataset(
            model, processor, samples, family, device,
            "vqav2", _vqa_accuracy,
        )
        meta["benchmarks"] = {"vqav2": vqa_result}
    else:
        meta["benchmarks"] = {}

    # Save results
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    # Cleanup
    unload_model(model)


if __name__ == "__main__":
    main()
