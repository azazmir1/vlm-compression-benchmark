"""
compression/ptq/run_pytorch_int8_gpu.py
========================================
GPU-accelerated INT8 quantization — quantizes on GPU instead of CPU.

Instead of:  CPU load FP16 → CPU quantize → move to GPU  (slow CPU math)
This does:   CPU load FP16 → move each layer to GPU → GPU quantize  (fast GPU math)

Layer-by-layer approach:
  1. Load full model on CPU (swap-backed, handles OOM models)
  2. For each Linear layer:
     a. Move FP16 weight to GPU (~50-200 MB temporary)
     b. Quantize on GPU (scale + round — massively parallel, microseconds)
     c. Store INT8 result on GPU, free FP16 copy
  3. Move remaining non-Linear params to GPU
  4. Vision encoder stays FP16

This should be significantly faster than CPU quantization since GPU does
the scale computation and rounding in parallel across all weight elements.

Usage:
  python compression/ptq/run_pytorch_int8_gpu.py --model_id Qwen/Qwen2.5-VL-3B-Instruct --vqav2_n 10
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
    load_vqav2, evaluate_dataset, _vqa_accuracy,
)
from compression.ptq.run_pytorch_int8 import Int8Linear, unload_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "pytorch_int8_gpu"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

VISION_KEYWORDS = [
    "vision_model", "vision_tower", "visual", "vit", "image_encoder",
    "img_encoder", "vision_encoder", "patch_embed", "pixel_shuffle",
    "connector", "multi_modal_projector", "mlp1",
]


def _is_vision(name: str) -> bool:
    return any(kw in name.lower() for kw in VISION_KEYWORDS)


class Int8LinearGPU(nn.Module):
    """Same as Int8Linear but quantizes on GPU for speed."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight_int8", torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        self.register_buffer(
            "scale", torch.zeros(out_features, 1, dtype=torch.float16)
        )
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def forward(self, x):
        w = self.weight_int8.to(x.dtype) * self.scale.to(x.dtype)
        return nn.functional.linear(x, w, self.bias.to(x.dtype) if self.bias is not None else None)

    def quantize_on_gpu(self, weight_fp16: torch.Tensor):
        """Move FP16 weight to GPU, quantize there, store INT8 on GPU."""
        # Move to GPU for fast quantization
        w_gpu = weight_fp16.to("cuda").float()
        scale = w_gpu.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        w_int8 = (w_gpu / scale).round().clamp(-128, 127).to(torch.int8)

        # Store on GPU
        self.weight_int8 = w_int8
        self.scale = scale.to(torch.float16)

        # Free the FP32 temp
        del w_gpu

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, bias={self.bias is not None}"


def load_model_int8_gpu(model_id: str, family: str = None,
                        skip_vision: bool = True, device: str = "cuda"):
    """Load model with GPU-accelerated INT8 quantization.

    1. Load FP16 on CPU (swap-backed)
    2. For each Linear: move weight to GPU → quantize on GPU → store INT8
    3. Move remaining params (vision, embeddings, norms) to GPU
    """
    from models.model_loader import load_model as _load_model_standard

    if family is None:
        family = detect_family(model_id)

    mem_before = _gpu_mem_mb()
    logger.info(f"Loading {model_id} with GPU-accelerated INT8 quantization...")
    logger.info(f"  Family: {family}, Device: {device}")
    logger.info(f"  GPU memory before: {mem_before:.1f} MB")

    t0 = time.time()

    # Step 1: Load on CPU
    logger.info("  Step 1: Loading model in FP16 on CPU...")
    t_cpu_start = time.time()
    model, processor, load_meta = _load_model_standard(model_id, device_map="cpu")
    t_cpu_end = time.time()
    logger.info(f"  CPU load done in {t_cpu_end - t_cpu_start:.1f}s. GPU mem: {_gpu_mem_mb():.1f} MB")

    # Step 2: Layer-by-layer GPU quantization
    logger.info("  Step 2: Quantizing Linear layers on GPU (layer-by-layer)...")
    t_quant_start = time.time()
    n_replaced = 0

    for mod_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full_name = f"{mod_name}.{child_name}" if mod_name else child_name
            if skip_vision and _is_vision(full_name):
                continue

            # Create Int8 layer (empty, on CPU)
            int8_layer = Int8LinearGPU(
                child.in_features, child.out_features,
                bias=child.bias is not None,
            )

            # Quantize on GPU: sends FP16 weight to GPU, quantizes, stores INT8
            int8_layer.quantize_on_gpu(child.weight.data)

            # Handle bias
            if child.bias is not None:
                int8_layer.bias = child.bias.data.to(torch.float16).to(device)

            setattr(parent, child_name, int8_layer)
            n_replaced += 1
            del child

    t_quant_end = time.time()
    logger.info(f"  Quantized {n_replaced} layers on GPU in {t_quant_end - t_quant_start:.1f}s")

    # Step 3: Move remaining params (vision, embeddings, norms) to GPU
    logger.info("  Step 3: Moving remaining params to GPU...")
    t_move_start = time.time()
    model = model.to(device).eval()
    gc.collect()
    torch.cuda.empty_cache()
    t_move_end = time.time()
    logger.info(f"  Remaining params moved in {t_move_end - t_move_start:.1f}s")

    load_time = time.time() - t0
    mem_after = _gpu_mem_mb()

    logger.info(f"  GPU memory after: {mem_after:.1f} MB (delta: {mem_after - mem_before:.1f} MB)")
    logger.info(f"  Total load time: {load_time:.1f}s")
    logger.info(f"    CPU load: {t_cpu_end - t_cpu_start:.1f}s | "
                f"GPU quant: {t_quant_end - t_quant_start:.1f}s | "
                f"GPU move: {t_move_end - t_move_start:.1f}s")

    # Count params
    n_params = sum(p.numel() for p in model.parameters())
    for name, buf in model.named_buffers():
        if "weight_int8" in name:
            n_params += buf.numel()

    meta = {
        "model_id": model_id,
        "family": family,
        "method": "pytorch_int8_gpu",
        "quant_bits": 8,
        "skip_vision_quant": skip_vision,
        "n_linear_quantized": n_replaced,
        "num_params_M": round(n_params / 1e6, 1),
        "gpu_mem_before_mb": round(mem_before, 1),
        "gpu_mem_after_mb": round(mem_after, 1),
        "gpu_mem_load_mb": round(mem_after - mem_before, 1),
        "load_time_s": round(load_time, 1),
        "timing": {
            "cpu_load_s": round(t_cpu_end - t_cpu_start, 1),
            "gpu_quant_s": round(t_quant_end - t_quant_start, 1),
            "gpu_move_s": round(t_move_end - t_move_start, 1),
        },
    }

    return model, processor, meta


def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated PyTorch INT8 quantization"
    )
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--vqav2_n", type=int, default=10)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_id = args.model_id
    family = detect_family(model_id)
    safe_name = model_id.replace("/", "__")

    out_path = RESULTS_DIR / f"{safe_name}__pytorch_int8_gpu.json"
    if out_path.exists() and not args.force:
        logger.info(f"Result exists: {out_path}")
        return

    model, processor, meta = load_model_int8_gpu(model_id, family)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.skip_eval:
        samples = load_vqav2(args.vqav2_n)
        vqa_result = evaluate_dataset(
            model, processor, samples, family, device,
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
