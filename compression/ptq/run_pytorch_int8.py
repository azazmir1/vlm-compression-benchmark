"""
compression/ptq/run_pytorch_int8.py
====================================
Pure-PyTorch INT8 quantization with layer-by-layer loading.

This is a **Category 1** method: it reduces model memory at load time so that
models which OOM in FP16 can fit on memory-constrained devices like Jetson.

Approach:
  1. Create model skeleton with zero GPU memory (accelerate empty weights)
  2. Replace all nn.Linear layers with Int8Linear (stores weights as int8 + fp16 scale)
  3. Load safetensors one tensor at a time, quantize to INT8, store on GPU
  4. Vision encoder stays in FP16 for accuracy
  5. Expected savings: ~40-50% vs FP16 (LLM backbone INT8, vision FP16)

Usage:
  python compression/ptq/run_pytorch_int8.py --model_id HuggingFaceTB/SmolVLM-256M-Instruct
  python compression/ptq/run_pytorch_int8.py --model_id Qwen/Qwen2.5-VL-3B-Instruct --vqav2_n 30
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

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "pytorch_int8"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Int8Linear: quantized linear layer ──────────────────────────────────────

class Int8Linear(nn.Module):
    """Linear layer with INT8 weight storage and FP16 scale factors.

    Weights are stored as int8 with per-channel (per-output) scale.
    Forward pass dequantizes on the fly: w_fp16 = weight_int8 * scale.
    Memory: ~1 byte/param (int8) + tiny scale overhead vs 2 bytes/param (fp16).
    """

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

    def set_weight_from_fp16(self, weight_fp16: torch.Tensor):
        """Quantize a FP16/FP32 weight tensor to INT8 with per-channel scale."""
        w = weight_fp16.float()
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        self.weight_int8.copy_((w / scale).round().clamp(-128, 127).to(torch.int8))
        self.scale.copy_(scale.to(torch.float16))

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, bias={self.bias is not None}"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _count_params(model: nn.Module) -> int:
    total = 0
    for p in model.parameters():
        total += p.numel()
    for name, buf in model.named_buffers():
        if "weight_int8" in name:
            total += buf.numel()
    return total


def _should_skip_quantize(name: str, skip_vision: bool) -> bool:
    """Decide if a layer should stay in FP16 (not be quantized)."""
    if not skip_vision:
        return False
    # Skip vision encoder layers — keep them FP16 for accuracy
    vision_keywords = [
        "vision_model", "vision_tower", "visual", "vit", "image_encoder",
        "img_encoder", "vision_encoder", "patch_embed", "pixel_shuffle",
        "connector", "multi_modal_projector", "mlp1",  # InternVL projector
    ]
    name_lower = name.lower()
    return any(kw in name_lower for kw in vision_keywords)


def _replace_linear_with_int8(model: nn.Module, prefix: str = "",
                               skip_vision: bool = True) -> int:
    """Replace nn.Linear modules with Int8Linear (in-place). Returns count."""
    count = 0
    for name, child in list(model.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            if _should_skip_quantize(full_name, skip_vision):
                continue
            int8_layer = Int8Linear(
                child.in_features, child.out_features,
                bias=child.bias is not None,
            )
            setattr(model, name, int8_layer)
            count += 1
        else:
            count += _replace_linear_with_int8(child, full_name, skip_vision)
    return count


def _load_and_quantize_weights(model: nn.Module, model_id: str, device: str):
    """Load weights from safetensors one tensor at a time, quantizing Linear layers."""
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    import glob as glob_mod

    # Find safetensors files (local cache)
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir_name = f"models--{model_id.replace('/', '--')}"
    model_cache = cache_dir / model_dir_name

    if model_cache.exists():
        snapshot_dirs = sorted(
            (model_cache / "snapshots").iterdir(),
            key=lambda d: d.stat().st_mtime, reverse=True,
        )
        if snapshot_dirs:
            st_files = sorted(snapshot_dirs[0].glob("*.safetensors"))

    if not st_files:
        # Download if not cached
        logger.info("Downloading model safetensors...")
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(model_id)
        st_files = sorted(Path(local_dir).glob("*.safetensors"))

    logger.info(f"  Found {len(st_files)} safetensors files")

    # Build name→module mapping for the model
    param_to_module = {}
    for mod_name, mod in model.named_modules():
        if isinstance(mod, Int8Linear):
            param_to_module[f"{mod_name}.weight"] = (mod, "weight")
            if mod.bias is not None:
                param_to_module[f"{mod_name}.bias"] = (mod, "bias")
        elif isinstance(mod, nn.Linear):
            # Vision/skip layers that stayed as nn.Linear
            param_to_module[f"{mod_name}.weight"] = (mod, "weight")
            if mod.bias is not None:
                param_to_module[f"{mod_name}.bias"] = (mod, "bias")

    loaded = 0
    quantized = 0
    skipped = 0

    for sf_path in st_files:
        with safe_open(str(sf_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)

                if key in param_to_module:
                    mod, attr = param_to_module[key]
                    if isinstance(mod, Int8Linear) and attr == "weight":
                        mod.set_weight_from_fp16(tensor)
                        # Move buffers to GPU
                        mod.weight_int8 = mod.weight_int8.to(device)
                        mod.scale = mod.scale.to(device)
                        quantized += 1
                    elif isinstance(mod, Int8Linear) and attr == "bias":
                        mod.bias = nn.Parameter(tensor.to(torch.float16).to(device), requires_grad=False)
                        # Actually it's a buffer, set directly
                        mod.bias = tensor.to(torch.float16).to(device)
                    else:
                        # Regular nn.Linear (vision layers kept FP16)
                        if attr == "weight":
                            mod.weight.data.copy_(tensor.to(torch.float16).to(device))
                        elif attr == "bias":
                            mod.bias.data.copy_(tensor.to(torch.float16).to(device))
                else:
                    # Non-linear parameters (embeddings, norms, etc.)
                    # Find in state dict
                    parts = key.split(".")
                    target = model
                    try:
                        for part in parts[:-1]:
                            target = getattr(target, part)
                        param_name = parts[-1]
                        param = getattr(target, param_name, None)
                        if param is not None:
                            if isinstance(param, nn.Parameter):
                                param.data.copy_(tensor.to(param.dtype).to(device))
                            elif isinstance(param, torch.Tensor):
                                setattr(target, param_name, tensor.to(param.dtype).to(device))
                            else:
                                skipped += 1
                                continue
                        else:
                            skipped += 1
                            continue
                    except (AttributeError, IndexError):
                        skipped += 1
                        continue

                loaded += 1

                # Free CPU tensor immediately
                del tensor

    logger.info(f"  Loaded {loaded} tensors ({quantized} quantized to INT8, {skipped} skipped)")
    return loaded, quantized


# ── Model-specific skeleton loaders ──────────────────────────────────────────

def _create_model_skeleton(model_id: str, family: str, device: str):
    """Create empty model + processor using accelerate init_empty_weights."""
    from accelerate import init_empty_weights
    from transformers import AutoProcessor, AutoConfig

    logger.info(f"Creating model skeleton for {model_id} (family={family})...")

    # Load processor normally (small, no GPU needed)
    trust_remote = family in ("moondream", "ovis2", "fastvlm", "internvl25")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote)

    # Determine model class
    if family in ("smolvlm", "lfm2vl"):
        from transformers import AutoModelForImageTextToText
        model_cls = AutoModelForImageTextToText
    elif family in ("qwen25vl",):
        from transformers import Qwen2_5_VLForConditionalGeneration
        model_cls = Qwen2_5_VLForConditionalGeneration
    elif family in ("internvl25",):
        from transformers import AutoModelForImageTextToText
        model_cls = AutoModelForImageTextToText
    elif family in ("moondream",):
        from transformers import AutoModelForCausalLM
        model_cls = AutoModelForCausalLM
    elif family in ("gemma3",):
        from transformers import AutoModelForImageTextToText
        model_cls = AutoModelForImageTextToText
    else:
        from transformers import AutoModelForImageTextToText
        model_cls = AutoModelForImageTextToText

    # Create model skeleton with empty weights
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote)

    with init_empty_weights():
        model = model_cls.from_config(config, trust_remote_code=trust_remote)

    # Load correct generation config from the hub (from_config uses wrong defaults)
    from transformers import GenerationConfig
    try:
        model.generation_config = GenerationConfig.from_pretrained(model_id)
    except Exception:
        pass  # Fall back to config defaults if no generation_config.json

    model.eval()

    return model, processor, config


def load_model_int8(model_id: str, family: str = None, skip_vision: bool = True,
                    device: str = "cuda"):
    """Load a model with INT8 quantization applied during loading.

    Strategy:
      1. Load model on CPU with from_pretrained (proper initialization)
      2. Replace nn.Linear with Int8Linear (quantize weights in-place)
      3. Move quantized model to GPU

    On Jetson (unified memory), CPU load uses the same physical RAM as GPU.
    The INT8 conversion reduces total memory before moving to GPU, so the
    final GPU footprint is ~50% of FP16.

    Returns: (model, processor, meta_dict)
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM

    if family is None:
        family = detect_family(model_id)

    mem_before = _gpu_mem_mb()
    logger.info(f"Loading {model_id} with PyTorch INT8 quantization...")
    logger.info(f"  Family: {family}, Device: {device}")
    logger.info(f"  GPU memory before: {mem_before:.1f} MB")

    t0 = time.time()

    # Step 1+2: Load model using the standard model_loader (handles all families)
    # Load on CPU first so we can quantize before moving to GPU.
    from models.model_loader import load_model as _load_model_standard
    logger.info("  Step 1: Loading model in FP16 (standard loader)...")
    model, processor, load_meta = _load_model_standard(model_id, device_map="cpu")

    mem_after_cpu = _gpu_mem_mb()
    logger.info(f"  CPU load done. GPU mem: {mem_after_cpu:.1f} MB")

    # Step 3: Replace Linear layers with Int8Linear (quantize on CPU)
    logger.info("  Step 2: Quantizing Linear layers to INT8...")
    n_replaced = 0
    vision_keywords = [
        "vision_model", "vision_tower", "visual", "vit", "image_encoder",
        "img_encoder", "vision_encoder", "patch_embed", "pixel_shuffle",
        "connector", "multi_modal_projector", "mlp1",
    ]
    for mod_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full_name = f"{mod_name}.{child_name}" if mod_name else child_name
            if skip_vision and any(kw in full_name.lower() for kw in vision_keywords):
                continue
            int8_layer = Int8Linear(
                child.in_features, child.out_features,
                bias=child.bias is not None,
            )
            int8_layer.set_weight_from_fp16(child.weight.data)
            if child.bias is not None:
                int8_layer.bias = child.bias.data.to(torch.float16)
            setattr(parent, child_name, int8_layer)
            n_replaced += 1
            del child  # Free FP16 weight immediately

    logger.info(f"  Quantized {n_replaced} Linear layers to INT8")

    # Step 4: Move to GPU
    logger.info("  Step 3: Moving quantized model to GPU...")
    model = model.to(device).eval()

    # Force garbage collection to free any remaining CPU copies
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
        if "weight_int8" in name:
            n_params += buf.numel()

    meta = {
        "model_id": model_id,
        "family": family,
        "method": "pytorch_int8",
        "quant_bits": 8,
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
        description="Pure-PyTorch INT8 quantization with layer-by-layer loading"
    )
    parser.add_argument("--model_id", required=True, help="HuggingFace model ID")
    parser.add_argument("--vqav2_n", type=int, default=30,
                        help="Number of VQAv2 samples for evaluation")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip evaluation, just load and report memory")
    parser.add_argument("--quantize_vision", action="store_true",
                        help="Also quantize vision encoder (default: keep FP16)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing results")
    args = parser.parse_args()

    model_id = args.model_id
    family = detect_family(model_id)
    safe_name = model_id.replace("/", "__")

    out_path = RESULTS_DIR / f"{safe_name}__pytorch_int8.json"
    if out_path.exists() and not args.force:
        logger.info(f"Result already exists: {out_path}")
        return

    # Load model with INT8 quantization
    model, processor, meta = load_model_int8(
        model_id, family,
        skip_vision=not args.quantize_vision,
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
