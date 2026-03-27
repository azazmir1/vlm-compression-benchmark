"""
compression/ptq/run_pytorch_int4.py
====================================
GPU-accelerated INT4 quantization — maximum compression for any device.

This is a **Category 1** method: it reduces model memory at load time so that
models which OOM in FP16 (and even INT8) can fit on memory-constrained devices.

Layer-by-layer GPU-streaming approach:
  1. Load full model on CPU (swap-backed, handles OOM models)
  2. For each Linear layer:
     a. Move FP16 weight to GPU (~50-200 MB temporary)
     b. Quantize on GPU (per-group min/max + round + pack — massively parallel)
     c. Store INT4 packed result on GPU, free FP16 copy
  3. Move remaining non-Linear params (vision, embeddings, norms) to GPU
  4. Vision encoder stays FP16 for accuracy
  5. Save quantized state_dict to disk cache for instant reload

INT4 uses per-group quantization (group_size=128) for better accuracy than
per-channel at this low bit width. ~4x compression vs FP16.

Cache: Quantized weights are saved to ~/.cache/vlm_int4/{model_safe_name}/
so subsequent runs skip the FP16 load + quantization entirely.

Usage:
  python compression/ptq/run_pytorch_int4.py --model_id Qwen/Qwen2.5-VL-7B-Instruct --vqav2_n 30
  python compression/ptq/run_pytorch_int4.py --model_id Qwen/Qwen2.5-VL-7B-Instruct --no_cache
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
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "pytorch_int4"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = Path.home() / ".cache" / "vlm_int4"


# ── Int4Linear: quantized linear layer ──────────────────────────────────────

class Int4Linear(nn.Module):
    """Linear layer with INT4 weight storage using per-group quantization.

    Weights are packed: 2 x int4 values per uint8 byte.
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
        packed_size = self.in_features_padded // 2
        self.register_buffer(
            "weight_packed", torch.zeros(out_features, packed_size, dtype=torch.uint8)
        )
        # Per-group scale and zero point
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
        """Unpack INT4 weights from packed uint8 storage to unsigned [0, 15]."""
        low = self.weight_packed & 0x0F
        high = self.weight_packed >> 4
        unpacked = torch.stack([high, low], dim=-1).reshape(
            self.weight_packed.shape[0], -1
        )
        return unpacked[:, :self.in_features_padded]

    def forward(self, x):
        dtype = x.dtype
        w_uint = self._unpack_weights()  # unsigned [0, 15]
        w_uint = w_uint.reshape(self.out_features, -1, self.group_size)
        scale = self.scale.unsqueeze(-1).to(dtype)
        zero = self.zero_point.unsqueeze(-1).to(dtype)
        # Dequantize: w = (q - zero_point) * scale
        w = (w_uint.to(dtype) - zero) * scale
        w = w.reshape(self.out_features, -1)
        w = w[:, :self.in_features]
        return nn.functional.linear(x, w, self.bias.to(dtype) if self.bias is not None else None)

    def quantize_on_gpu(self, weight_fp16: torch.Tensor):
        """Move FP16 weight to GPU, quantize there, store INT4 packed on GPU.

        Asymmetric unsigned INT4 quantization [0, 15]:
          scale = (max - min) / 15
          zero_point = round(-min / scale)   # maps min to ~0
          q = round(w / scale + zero_point)  # in [0, 15]
          w_recon = (q - zero_point) * scale
        """
        w_gpu = weight_fp16.to("cuda").float()
        out_features, in_features = w_gpu.shape

        # Pad if needed
        if in_features < self.in_features_padded:
            w_gpu = torch.nn.functional.pad(w_gpu, (0, self.in_features_padded - in_features))

        # Reshape into groups: (out, n_groups, group_size)
        w_grouped = w_gpu.reshape(out_features, -1, self.group_size)

        # Per-group min/max for asymmetric quantization
        w_min = w_grouped.amin(dim=-1, keepdim=True)
        w_max = w_grouped.amax(dim=-1, keepdim=True)

        # Unsigned INT4 [0, 15]
        scale = (w_max - w_min).clamp(min=1e-8) / 15.0
        zero_point = (-w_min / scale).round().clamp(0, 15)

        # Quantize to [0, 15]
        w_q = (w_grouped / scale + zero_point).round().clamp(0, 15)

        # Store scale and zero_point on GPU
        self.scale = scale.squeeze(-1).to(torch.float16)
        self.zero_point = zero_point.squeeze(-1).to(torch.float16)

        # Pack: 2 unsigned [0,15] values per byte
        w_flat = w_q.to(torch.uint8).reshape(out_features, -1)

        even = w_flat[:, 0::2]  # high nibble
        odd = w_flat[:, 1::2]   # low nibble
        packed = (even << 4) | odd
        self.weight_packed = packed.to(torch.uint8)

        # Free FP32 temp
        del w_gpu, w_grouped, w_q, w_flat, even, odd, packed

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"group_size={self.group_size}, bias={self.bias is not None}")


# ── Helpers ──────────────────────────────────────────────────────────────────

VISION_KEYWORDS = [
    "vision_model", "vision_tower", "visual", "vit", "image_encoder",
    "img_encoder", "vision_encoder", "patch_embed", "pixel_shuffle",
    "connector", "multi_modal_projector", "mlp1",
]


def _is_vision(name: str) -> bool:
    return any(kw in name.lower() for kw in VISION_KEYWORDS)


def _count_params(model: nn.Module) -> int:
    total = 0
    for p in model.parameters():
        total += p.numel()
    for name, buf in model.named_buffers():
        if "weight_packed" in name:
            total += buf.numel() * 2  # each byte = 2 params
    return total


# ── Cache helpers ────────────────────────────────────────────────────────────

def _cache_path(model_id: str, group_size: int, skip_vision: bool) -> Path:
    """Return the cache directory for a given model + quantization config."""
    safe_name = model_id.replace("/", "__")
    tag = f"g{group_size}" + ("_novision" if not skip_vision else "")
    return CACHE_DIR / f"{safe_name}__{tag}"


def save_int4_cache(model: nn.Module, cache_dir: Path, meta: dict):
    """Save quantized model state_dict + metadata to disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"  Saving INT4 cache to {cache_dir}...")
    t0 = time.time()

    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, cache_dir / "state_dict.pt")
    del state_dict

    int4_config = {}
    for name, module in model.named_modules():
        if isinstance(module, Int4Linear):
            int4_config[name] = {
                "in_features": module.in_features,
                "out_features": module.out_features,
                "group_size": module.group_size,
                "has_bias": module.bias is not None,
            }

    cache_meta = {**meta, "int4_layers": int4_config}
    with open(cache_dir / "meta.json", "w") as f:
        json.dump(cache_meta, f, indent=2)

    elapsed = time.time() - t0
    size_mb = sum(f.stat().st_size for f in cache_dir.iterdir()) / 1024**2
    logger.info(f"  Cache saved: {size_mb:.0f} MB on disk ({elapsed:.1f}s)")


def load_int4_from_cache(model_id: str, cache_dir: Path,
                         device: str = "cuda") -> tuple:
    """Load a previously cached INT4 model."""
    from models.model_loader import load_model as _load_model_standard

    with open(cache_dir / "meta.json") as f:
        cache_meta = json.load(f)

    family = cache_meta["family"]
    int4_config = cache_meta["int4_layers"]

    mem_before = _gpu_mem_mb()
    logger.info(f"Loading {model_id} from INT4 cache: {cache_dir}")
    logger.info(f"  GPU memory before: {mem_before:.1f} MB")

    t0 = time.time()

    logger.info("  Step 1: Loading model architecture on CPU...")
    t_cpu_start = time.time()
    model, processor, _ = _load_model_standard(model_id, device_map="cpu")
    t_cpu_end = time.time()
    logger.info(f"  Architecture loaded in {t_cpu_end - t_cpu_start:.1f}s")

    logger.info(f"  Step 2: Replacing {len(int4_config)} layers with Int4Linear...")
    module_dict = dict(model.named_modules())
    for layer_name, cfg in int4_config.items():
        int4_layer = Int4Linear(
            cfg["in_features"], cfg["out_features"],
            bias=cfg["has_bias"],
            group_size=cfg["group_size"],
        )
        parts = layer_name.rsplit(".", 1)
        if len(parts) == 2:
            parent = module_dict[parts[0]]
            attr_name = parts[1]
        else:
            parent = model
            attr_name = layer_name
        setattr(parent, attr_name, int4_layer)

    logger.info("  Step 3: Loading cached INT4 weights...")
    t_load_start = time.time()
    state_dict = torch.load(cache_dir / "state_dict.pt", map_location="cpu",
                            weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    del state_dict
    gc.collect()
    t_load_end = time.time()
    logger.info(f"  Weights loaded in {t_load_end - t_load_start:.1f}s")

    logger.info("  Step 4: Moving model to GPU...")
    t_move_start = time.time()
    model = model.to(device).eval()
    gc.collect()
    torch.cuda.empty_cache()
    t_move_end = time.time()

    load_time = time.time() - t0
    mem_after = _gpu_mem_mb()

    logger.info(f"  GPU memory after: {mem_after:.1f} MB (delta: {mem_after - mem_before:.1f} MB)")
    logger.info(f"  Total load time: {load_time:.1f}s (from cache)")

    n_params = _count_params(model)

    meta = {
        "model_id": model_id,
        "family": family,
        "method": "pytorch_int4",
        "quant_bits": 4,
        "group_size": cache_meta.get("group_size", 128),
        "skip_vision_quant": cache_meta.get("skip_vision_quant", True),
        "n_linear_quantized": len(int4_config),
        "num_params_M": round(n_params / 1e6, 1),
        "gpu_mem_before_mb": round(mem_before, 1),
        "gpu_mem_after_mb": round(mem_after, 1),
        "gpu_mem_load_mb": round(mem_after - mem_before, 1),
        "load_time_s": round(load_time, 1),
        "loaded_from_cache": True,
    }

    return model, processor, meta


# ── GPU-streaming loader ─────────────────────────────────────────────────────

def load_model_int4(model_id: str, family: str = None, skip_vision: bool = True,
                    device: str = "cuda", group_size: int = 128,
                    use_cache: bool = True):
    """Load model with GPU-accelerated INT4 quantization.

    If a cached version exists (and use_cache=True), loads from cache directly.
    Otherwise: quantize from scratch and save to cache for next time.

    Returns: (model, processor, meta_dict)
    """
    from models.model_loader import load_model as _load_model_standard

    if family is None:
        family = detect_family(model_id)

    # Check cache first
    cache_dir = _cache_path(model_id, group_size, skip_vision)
    if use_cache and (cache_dir / "state_dict.pt").exists():
        logger.info(f"INT4 cache found at {cache_dir}")
        return load_int4_from_cache(model_id, cache_dir, device)

    mem_before = _gpu_mem_mb()
    logger.info(f"Loading {model_id} with GPU-accelerated INT4 quantization...")
    logger.info(f"  Family: {family}, Device: {device}, Group size: {group_size}")
    logger.info(f"  GPU memory before: {mem_before:.1f} MB")

    t0 = time.time()

    # Step 1: Load on CPU
    logger.info("  Step 1: Loading model in FP16 on CPU...")
    t_cpu_start = time.time()
    model, processor, load_meta = _load_model_standard(model_id, device_map="cpu")
    t_cpu_end = time.time()
    logger.info(f"  CPU load done in {t_cpu_end - t_cpu_start:.1f}s. GPU mem: {_gpu_mem_mb():.1f} MB")

    # Step 2: Layer-by-layer GPU quantization
    logger.info("  Step 2: Quantizing Linear layers to INT4 on GPU (layer-by-layer)...")
    t_quant_start = time.time()
    n_replaced = 0

    for mod_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full_name = f"{mod_name}.{child_name}" if mod_name else child_name
            if skip_vision and _is_vision(full_name):
                continue

            int4_layer = Int4Linear(
                child.in_features, child.out_features,
                bias=child.bias is not None,
                group_size=group_size,
            )

            # Quantize on GPU: sends FP16 weight to GPU, quantizes, stores INT4 packed
            int4_layer.quantize_on_gpu(child.weight.data)

            # Handle bias
            if child.bias is not None:
                int4_layer.bias = child.bias.data.to(torch.float16).to(device)

            setattr(parent, child_name, int4_layer)
            n_replaced += 1
            del child

            if n_replaced % 50 == 0:
                gc.collect()
                logger.info(f"    ... quantized {n_replaced} layers so far ({_gpu_mem_mb():.0f} MB GPU)")

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

    n_params = _count_params(model)

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
        "loaded_from_cache": False,
    }

    # Save to cache for next time
    if use_cache:
        save_int4_cache(model, cache_dir, meta)

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
        description="GPU-accelerated PyTorch INT4 quantization — maximum compression"
    )
    parser.add_argument("--model_id", required=True, help="HuggingFace model ID")
    parser.add_argument("--vqav2_n", type=int, default=20,
                        help="Number of VQAv2 samples for evaluation")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip evaluation, just load and report memory")
    parser.add_argument("--quantize_vision", action="store_true",
                        help="Also quantize vision encoder (default: keep FP16)")
    parser.add_argument("--group_size", type=int, default=128,
                        help="Quantization group size (default: 128)")
    parser.add_argument("--no_cache", action="store_true",
                        help="Don't use or save disk cache (quantize from scratch)")
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

    model, processor, meta = load_model_int4(
        model_id, family,
        skip_vision=not args.quantize_vision,
        group_size=args.group_size,
        use_cache=not args.no_cache,
    )

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
