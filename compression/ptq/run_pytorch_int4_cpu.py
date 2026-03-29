"""
compression/ptq/run_pytorch_int4_cpu.py
========================================
CPU-only INT4 quantization — maximum compression, no GPU required.

Per-group asymmetric INT4 quantization of Linear layers:
  1. Load model on CPU
  2. For each Linear layer (skipping vision encoder):
     a. Reshape weights into groups (default 128)
     b. Compute per-group min/max → scale + zero_point
     c. Quantize to unsigned [0, 15] and pack 2 values per uint8 byte
     d. Replace with Int4Linear
  3. Vision encoder stays FP32 for accuracy

~8x compression vs FP32 (0.5 bytes/param vs 4 bytes/param).

Cache: Quantized weights saved to ~/.cache/vlm_int4/{model_safe_name}/
for instant reload on subsequent runs.

Usage:
  python compression/ptq/run_pytorch_int4_cpu.py --model_id HuggingFaceTB/SmolVLM2-2.2B-Instruct --vqav2_n 20
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

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "pytorch_int4_cpu"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = Path.home() / ".cache" / "vlm_int4"

VISION_KEYWORDS = [
    "vision_model", "vision_tower", "visual", "vit", "image_encoder",
    "img_encoder", "vision_encoder", "patch_embed", "pixel_shuffle",
    "connector", "multi_modal_projector", "mlp1",
]


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def _is_vision(name: str) -> bool:
    return any(kw in name.lower() for kw in VISION_KEYWORDS)


def _count_params(model: nn.Module) -> int:
    total = sum(p.numel() for p in model.parameters())
    for name, buf in model.named_buffers():
        if "weight_packed" in name:
            total += buf.numel() * 2  # each byte = 2 int4 params
    return total


# ── Int4Linear: quantized linear layer ──────────────────────────────────────

class Int4Linear(nn.Module):
    """Linear layer with INT4 weight storage using per-group quantization.

    Weights packed: 2 x int4 values per uint8 byte.
    Per-group quantization (default group_size=128) for better accuracy.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 group_size: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        self.in_features_padded = ((in_features + group_size - 1) // group_size) * group_size
        n_groups = self.in_features_padded // group_size

        packed_size = self.in_features_padded // 2
        self.register_buffer(
            "weight_packed", torch.zeros(out_features, packed_size, dtype=torch.uint8)
        )
        self.register_buffer(
            "scale", torch.zeros(out_features, n_groups, dtype=torch.float32)
        )
        self.register_buffer(
            "zero_point", torch.zeros(out_features, n_groups, dtype=torch.float32)
        )
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None

    def _unpack_weights(self):
        """Unpack INT4 weights from packed uint8 to unsigned [0, 15]."""
        low = self.weight_packed & 0x0F
        high = self.weight_packed >> 4
        unpacked = torch.stack([high, low], dim=-1).reshape(
            self.weight_packed.shape[0], -1
        )
        return unpacked[:, :self.in_features_padded]

    def forward(self, x):
        dtype = x.dtype
        w_uint = self._unpack_weights()
        w_uint = w_uint.reshape(self.out_features, -1, self.group_size)
        scale = self.scale.unsqueeze(-1).to(dtype)
        zero = self.zero_point.unsqueeze(-1).to(dtype)
        w = (w_uint.to(dtype) - zero) * scale
        w = w.reshape(self.out_features, -1)
        w = w[:, :self.in_features]
        return nn.functional.linear(x, w, self.bias.to(dtype) if self.bias is not None else None)

    def quantize(self, weight: torch.Tensor):
        """Asymmetric unsigned INT4 quantization on CPU.

        scale = (max - min) / 15
        zero_point = round(-min / scale)
        q = round(w / scale + zero_point), clamped to [0, 15]
        """
        w = weight.float()
        out_features, in_features = w.shape

        if in_features < self.in_features_padded:
            w = torch.nn.functional.pad(w, (0, self.in_features_padded - in_features))

        w_grouped = w.reshape(out_features, -1, self.group_size)

        w_min = w_grouped.amin(dim=-1, keepdim=True)
        w_max = w_grouped.amax(dim=-1, keepdim=True)

        scale = (w_max - w_min).clamp(min=1e-8) / 15.0
        zero_point = (-w_min / scale).round().clamp(0, 15)

        w_q = (w_grouped / scale + zero_point).round().clamp(0, 15)

        self.scale = scale.squeeze(-1).to(torch.float32)
        self.zero_point = zero_point.squeeze(-1).to(torch.float32)

        # Pack: 2 unsigned [0,15] values per byte
        w_flat = w_q.to(torch.uint8).reshape(out_features, -1)
        even = w_flat[:, 0::2]  # high nibble
        odd = w_flat[:, 1::2]   # low nibble
        self.weight_packed = ((even << 4) | odd).to(torch.uint8)

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"group_size={self.group_size}, bias={self.bias is not None}")


# ── Cache helpers ────────────────────────────────────────────────────────────

def _cache_path(model_id: str, group_size: int, skip_vision: bool) -> Path:
    safe_name = model_id.replace("/", "__")
    tag = f"g{group_size}" + ("_novision" if not skip_vision else "")
    return CACHE_DIR / f"{safe_name}__{tag}"


def save_int4_cache(model: nn.Module, cache_dir: Path, meta: dict):
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


def load_int4_from_cache(model_id: str, cache_dir: Path) -> tuple:
    from models.model_loader import load_model as _load_model_standard

    with open(cache_dir / "meta.json") as f:
        cache_meta = json.load(f)

    family = cache_meta["family"]
    int4_config = cache_meta["int4_layers"]

    mem_before = _rss_mb()
    logger.info(f"Loading {model_id} from INT4 cache: {cache_dir}")
    logger.info(f"  RAM before: {mem_before:.1f} MB")
    t0 = time.time()

    logger.info("  Step 1: Loading model architecture on CPU...")
    model, processor, _ = _load_model_standard(model_id)

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
            setattr(parent, parts[1], int4_layer)
        else:
            setattr(model, layer_name, int4_layer)

    logger.info("  Step 3: Loading cached INT4 weights...")
    state_dict = torch.load(cache_dir / "state_dict.pt", map_location="cpu",
                            weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    del state_dict
    gc.collect()

    model.eval()
    load_time = time.time() - t0
    mem_after = _rss_mb()

    logger.info(f"  RAM after: {mem_after:.1f} MB (delta: {mem_after - mem_before:.1f} MB)")
    logger.info(f"  Total load time: {load_time:.1f}s (from cache)")

    meta = {
        "model_id": model_id,
        "family": family,
        "method": "pytorch_int4_cpu",
        "quant_bits": 4,
        "group_size": cache_meta.get("group_size", 128),
        "skip_vision_quant": cache_meta.get("skip_vision_quant", True),
        "n_linear_quantized": len(int4_config),
        "num_params_M": round(_count_params(model) / 1e6, 1),
        "ram_before_mb": round(mem_before, 1),
        "ram_after_mb": round(mem_after, 1),
        "ram_delta_mb": round(mem_after - mem_before, 1),
        "load_time_s": round(load_time, 1),
        "loaded_from_cache": True,
    }

    return model, processor, meta


# ── Loader ───────────────────────────────────────────────────────────────────

def load_model_int4_cpu(model_id: str, family: str = None, skip_vision: bool = True,
                        group_size: int = 128, use_cache: bool = True):
    """Load model with CPU INT4 quantization.

    Returns: (model, processor, meta_dict)
    """
    from models.model_loader import load_model as _load_model_standard

    if family is None:
        family = detect_family(model_id)

    # Check cache
    cache_dir = _cache_path(model_id, group_size, skip_vision)
    if use_cache and (cache_dir / "state_dict.pt").exists():
        logger.info(f"INT4 cache found at {cache_dir}")
        return load_int4_from_cache(model_id, cache_dir)

    mem_before = _rss_mb()
    logger.info(f"Loading {model_id} with CPU INT4 quantization...")
    logger.info(f"  Family: {family}, Group size: {group_size}")
    logger.info(f"  RAM before: {mem_before:.1f} MB")

    t0 = time.time()

    # Step 1: Load on CPU
    logger.info("  Step 1: Loading model on CPU...")
    t_load = time.time()
    model, processor, load_meta = _load_model_standard(model_id)
    t_load = time.time() - t_load
    logger.info(f"  Loaded in {t_load:.1f}s. RAM: {_rss_mb():.1f} MB")

    # Step 2: Quantize Linear layers to INT4
    logger.info("  Step 2: Quantizing Linear layers to INT4...")
    t_quant = time.time()
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
            int4_layer.quantize(child.weight.data)

            if child.bias is not None:
                int4_layer.bias = child.bias.data.float()

            setattr(parent, child_name, int4_layer)
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

    n_params = _count_params(model)

    meta = {
        "model_id": model_id,
        "family": family,
        "method": "pytorch_int4_cpu",
        "quant_bits": 4,
        "group_size": group_size,
        "skip_vision_quant": skip_vision,
        "n_linear_quantized": n_replaced,
        "num_params_M": round(n_params / 1e6, 1),
        "ram_before_mb": round(mem_before, 1),
        "ram_after_mb": round(mem_after, 1),
        "ram_delta_mb": round(mem_after - mem_before, 1),
        "load_time_s": round(load_time, 1),
        "loaded_from_cache": False,
        "timing": {
            "model_load_s": round(t_load, 1),
            "quantization_s": round(t_quant, 1),
        },
    }

    # Save cache
    if use_cache:
        save_int4_cache(model, cache_dir, meta)

    return model, processor, meta


def unload_model(model):
    del model
    gc.collect()
    logger.info("Model unloaded.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CPU-only PyTorch INT4 quantization — maximum compression"
    )
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--vqav2_n", type=int, default=20)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--quantize_vision", action="store_true")
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_id = args.model_id
    family = detect_family(model_id)
    safe_name = model_id.replace("/", "__")

    out_path = RESULTS_DIR / f"{safe_name}__pytorch_int4_cpu.json"
    if out_path.exists() and not args.force:
        logger.info(f"Result already exists: {out_path}")
        return

    model, processor, meta = load_model_int4_cpu(
        model_id, family,
        skip_vision=not args.quantize_vision,
        group_size=args.group_size,
        use_cache=not args.no_cache,
    )

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
