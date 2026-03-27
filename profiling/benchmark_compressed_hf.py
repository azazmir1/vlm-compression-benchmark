#!/usr/bin/env python3
"""
Universal benchmark script for compressed models from HuggingFace.
Supports: PyTorch INT8, PyTorch INT4, HQQ INT4, GPTQ INT4.
Detects format automatically from repo contents.
"""

import argparse
import json
import logging
import sys
import time
import gc
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from profiling.gpu_profiler import GPUProfiler
from evaluation.run_baseline import (
    load_vqav2, run_inference, _vqa_multi_metric
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── INT8 Linear ──────────────────────────────────────────────────────────────

class Int8Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, group_size=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.register_buffer("weight_int8", torch.zeros(out_features, in_features, dtype=torch.int8))
        if group_size:
            n_groups = (in_features + group_size - 1) // group_size
            self.register_buffer("scale", torch.zeros(out_features, n_groups, dtype=torch.float16))
        else:
            self.register_buffer("scale", torch.zeros(out_features, 1, dtype=torch.float16))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def forward(self, x):
        dtype = x.dtype
        if self.group_size and self.scale.shape[1] > 1:
            # Per-group dequantization
            w_int = self.weight_int8.reshape(self.out_features, -1, self.group_size)
            w = (w_int.to(dtype) * self.scale.unsqueeze(-1).to(dtype)).reshape(self.out_features, -1)
            w = w[:, :self.in_features]
        else:
            w = self.weight_int8.to(dtype) * self.scale.to(dtype)
        return nn.functional.linear(x, w, self.bias.to(dtype) if self.bias is not None else None)


# ── HQQ INT4 Linear (on-the-fly dequant) ────────────────────────────────────

class HqqInt4Linear(nn.Module):
    """Keeps HQQ INT4 packed weights on GPU, dequantizes per forward pass."""
    def __init__(self, out_features, in_features, group_size=64, bias_data=None):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.group_size = group_size
        packed_numel = out_features * in_features // 2
        n_groups = out_features * in_features // group_size
        self.register_buffer("W_q", torch.zeros(packed_numel, dtype=torch.uint8))
        self.register_buffer("scale", torch.zeros(n_groups, 1, dtype=torch.float16))
        self.register_buffer("zero", torch.zeros(n_groups, 1, dtype=torch.float16))
        if bias_data is not None:
            self.register_buffer("bias", bias_data)
        else:
            self.bias = None

    def forward(self, x):
        dtype = x.dtype
        # HQQ packing: first half of values = high nibbles, second half = low nibbles
        wq_flat = self.W_q.reshape(-1)
        high = (wq_flat >> 4) & 0x0F
        low = wq_flat & 0x0F
        w_uint = torch.cat([high, low]).to(dtype)
        # Dequantize per group
        w_grouped = w_uint.reshape(-1, self.group_size)
        w_fp = ((w_grouped - self.zero.to(dtype)) * self.scale.to(dtype)).reshape(
            self.out_features, self.in_features)
        return nn.functional.linear(x, w_fp,
                                    self.bias.to(dtype) if self.bias is not None else None)


# ── INT4 Linear ──────────────────────────────────────────────────────────────

class Int4Linear(nn.Module):
    # dequant_mode: "sub" = (w - zero) * scale, "add" = w * scale + zero
    # unpack_order: "high_low" = [high, low], "low_high" = [low, high]
    def __init__(self, in_features, out_features, group_size=128, bias=True, dequant_mode="sub", unpack_order="high_low"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.dequant_mode = dequant_mode
        self.unpack_order = unpack_order
        # Pad in_features to be divisible by group_size (matches quantizer)
        self.in_features_padded = ((in_features + group_size - 1) // group_size) * group_size
        n_groups = self.in_features_padded // group_size
        packed_size = self.in_features_padded // 2
        self.register_buffer("weight_packed", torch.zeros(out_features, packed_size, dtype=torch.uint8))
        self.register_buffer("scale", torch.zeros(out_features, n_groups, dtype=torch.float16))
        self.register_buffer("zero_point", torch.zeros(out_features, n_groups, dtype=torch.float16))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def _unpack_weights(self):
        low = self.weight_packed & 0x0F
        high = self.weight_packed >> 4
        if self.unpack_order == "low_high":
            w = torch.stack([low, high], dim=-1).reshape(self.out_features, -1)
        else:
            w = torch.stack([high, low], dim=-1).reshape(self.out_features, -1)
        return w[:, :self.in_features_padded]

    def forward(self, x):
        dtype = x.dtype
        w_uint = self._unpack_weights()
        w_uint = w_uint.reshape(self.out_features, -1, self.group_size)
        scale = self.scale.unsqueeze(-1).to(dtype)
        zero = self.zero_point.unsqueeze(-1).to(dtype)
        if self.dequant_mode == "add":
            w_fp = (w_uint.to(dtype) * scale + zero).reshape(self.out_features, -1)
        else:
            w_fp = ((w_uint.to(dtype) - zero) * scale).reshape(self.out_features, -1)
        w_fp = w_fp[:, :self.in_features]
        return nn.functional.linear(x, w_fp, self.bias.to(dtype) if self.bias is not None else None)


# ── Format Detection ─────────────────────────────────────────────────────────

def detect_format(hf_repo):
    """Detect compression format from HF repo contents."""
    from huggingface_hub import list_repo_files, hf_hub_download
    files = list_repo_files(hf_repo)

    if "quant_config.json" in files:
        cfg = json.load(open(hf_hub_download(hf_repo, "quant_config.json")))
        method = cfg.get("method", "")
        if "int8" in method:
            return "pytorch_int8", cfg
        elif "int4" in method:
            return "pytorch_int4", cfg

    # Check quant_info.json (alternative config name)
    if "quant_info.json" in files:
        cfg = json.load(open(hf_hub_download(hf_repo, "quant_info.json")))
        quant = cfg.get("quantization", "")
        if "int8" in quant:
            return "pytorch_int8", cfg
        elif "int4" in quant:
            return "pytorch_int4", cfg

    # Detect from state_dict.pt keys
    if "state_dict.pt" in files:
        path = hf_hub_download(hf_repo, "state_dict.pt")
        keys = list(torch.load(path, map_location="cpu", weights_only=True).keys())
        if any(k.endswith(".weight_int8") for k in keys):
            return "pytorch_int8", {}
        elif any(k.endswith(".weight_packed") for k in keys):
            return "pytorch_int4", {}

    # Check safetensors for HQQ vs GPTQ markers
    if "model.safetensors" in files:
        from safetensors import safe_open
        path = hf_hub_download(hf_repo, "model.safetensors")
        with safe_open(path, framework="pt") as f:
            keys = list(f.keys())

        if any(k.endswith(".W_q") for k in keys):
            return "hqq_int4", {}
        elif any(k.endswith(".qweight") for k in keys):
            return "gptq_int4", {}
        elif any(k.endswith(".weight_packed") for k in keys):
            return "pytorch_int4", {}
        elif any(k.endswith(".weight_int8") for k in keys):
            return "pytorch_int8", {}
        else:
            return "fp16_pruned", {}

    raise ValueError(f"Cannot detect format for {hf_repo}: no model.safetensors or state_dict.pt found")


# ── Base model loader (family-aware) ─────────────────────────────────────────

def _load_base_model(base_model, family):
    """Load base model + processor, handling family-specific quirks."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if family == "internvl25":
        from transformers import AutoModel, AutoTokenizer, PreTrainedModel
        # Patch for transformers 5.x compatibility
        _patched = not hasattr(PreTrainedModel, 'all_tied_weights_keys')
        if _patched:
            class _Desc:
                ATTR = '_all_tied_weights_keys_storage'
                def __get__(self, obj, objtype=None):
                    if obj is None: return self
                    val = obj.__dict__.get(self.ATTR)
                    if val is not None: return val
                    val = getattr(obj, '_tied_weights_keys', None)
                    return val if val is not None else {}
                def __set__(self, obj, value):
                    obj.__dict__[self.ATTR] = value
            PreTrainedModel.all_tied_weights_keys = _Desc()

        # Patch torch.linspace for meta device
        _orig_linspace = torch.linspace
        def _safe_linspace(*args, **kw):
            try: return _orig_linspace(*args, **kw)
            except RuntimeError:
                kw.pop("device", None)
                return _orig_linspace(*args, device="cpu", **kw)
        torch.linspace = _safe_linspace

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            base_model, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True)

        torch.linspace = _orig_linspace
        if _patched:
            try: del PreTrainedModel.all_tied_weights_keys
            except AttributeError: pass

        # Patch GenerationMixin
        from transformers import GenerationMixin, GenerationConfig
        lm = model.language_model
        if not isinstance(lm, GenerationMixin):
            lm.__class__ = type(lm.__class__.__name__, (lm.__class__, GenerationMixin), {})
        if getattr(lm, "generation_config", None) is None:
            lm.generation_config = GenerationConfig()
        lm.__class__._supports_default_dynamic_cache = lambda *args: False
        model.tokenizer = tokenizer
        return model, tokenizer
    elif family == "ovis2":
        from transformers import AutoModelForCausalLM, PreTrainedModel
        import transformers, importlib.metadata, importlib.machinery
        # Import patching utilities from model_loader
        from models.model_loader import _patch_ovis2_remote_code, _patch_all_tied_weights_keys, _unpatch_all_tied_weights_keys

        # Monkey-patch AutoConfig.register to skip duplicates (aimv2 re-registration)
        _orig_register = transformers.AutoConfig.register
        is_classmethod = hasattr(_orig_register, '__func__')
        raw_func = _orig_register.__func__ if is_classmethod else _orig_register
        def _safe_register(*args, exist_ok=False, **kw):
            try: return raw_func(*args, exist_ok=exist_ok, **kw)
            except ValueError as e:
                if "already used" not in str(e): raise
        if is_classmethod:
            transformers.AutoConfig.register = classmethod(_safe_register)
        else:
            transformers.AutoConfig.register = _safe_register

        # Patch cached remote code
        _patch_ovis2_remote_code(base_model)

        # Fake flash_attn module
        import types
        _had_flash = 'flash_attn' in sys.modules
        if not _had_flash:
            _fake = types.ModuleType('flash_attn')
            _fake.__version__ = '2.7.0'
            _fake.__spec__ = importlib.machinery.ModuleSpec('flash_attn', None)
            sys.modules['flash_attn'] = _fake
        _orig_version = importlib.metadata.version
        def _patched_version(name):
            if name == 'flash_attn': return '2.7.0'
            return _orig_version(name)
        importlib.metadata.version = _patched_version

        # Patch is_parallelizable
        _had_is_par = hasattr(PreTrainedModel, 'is_parallelizable')
        if not _had_is_par:
            PreTrainedModel.is_parallelizable = False

        # Patch all_tied_weights_keys
        _patched_tied = not hasattr(PreTrainedModel, 'all_tied_weights_keys')
        if _patched_tied:
            _patch_all_tied_weights_keys(PreTrainedModel)

        try:
            model = AutoModelForCausalLM.from_pretrained(
                base_model, torch_dtype=torch.bfloat16, device_map="cpu",
                trust_remote_code=True, attn_implementation="eager")
        finally:
            importlib.metadata.version = _orig_version
            if not _had_flash and 'flash_attn' in sys.modules:
                del sys.modules['flash_attn']
            if not _had_is_par and hasattr(PreTrainedModel, 'is_parallelizable'):
                del PreTrainedModel.is_parallelizable
            if _patched_tied:
                _unpatch_all_tied_weights_keys(PreTrainedModel)
            transformers.AutoConfig.register = _orig_register

        processor = model.get_text_tokenizer()
        return model, processor

    elif family == "fastvlm":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from models.model_loader import _FastVLMProcessor

        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        vt = model.get_vision_tower()
        model.to(dtype=torch.float16)
        image_processor = vt.image_processor
        processor = _FastVLMProcessor(image_processor, tokenizer)
        return model, processor

    else:
        # Gemma3 requires bfloat16 — float16 causes NaN/overflow
        dtype = torch.bfloat16 if family == "gemma3" else torch.float16
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            base_model, torch_dtype=dtype, device_map="cpu", trust_remote_code=True)
        return model, processor


# ── Weight loading helper ─────────────────────────────────────────────────────

def _load_quant_state(hf_repo, model):
    """Load quantized state dict, handling safetensors vs pt and key remapping."""
    from huggingface_hub import hf_hub_download, list_repo_files
    files = list_repo_files(hf_repo)

    if "model.safetensors" in files:
        from safetensors.torch import load_file
        state = load_file(hf_hub_download(hf_repo, "model.safetensors"), device="cpu")
    elif "state_dict.pt" in files:
        state = torch.load(hf_hub_download(hf_repo, "state_dict.pt"), map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No model.safetensors or state_dict.pt in {hf_repo}")

    # Check if keys need remapping (backbone-relative → full model path)
    # e.g. "model.layers.0..." → "language_model.model.layers.0..." for VLMs
    sample_key = next((k for k in state if ".weight_int8" in k or ".weight_packed" in k), None)
    if sample_key:
        # Try to find the layer in the model directly
        parts = sample_key.split(".")
        try:
            obj = model
            for p in parts:
                obj = getattr(obj, p)
            # Keys work directly, no remapping needed
        except AttributeError:
            # Keys don't match — try to find a prefix that works
            # Search named modules for one that has the right submodule
            first_part = parts[0]  # e.g. "model"
            for name, mod in model.named_modules():
                if name.endswith(f".{first_part}") or name == first_part:
                    # Found it — but we need the prefix before first_part
                    prefix = name[:-len(first_part)] if name != first_part else ""
                    break
                # Also check if the module itself is named to match
                try:
                    getattr(mod, first_part)
                    prefix = f"{name}." if name else ""
                    break
                except AttributeError:
                    continue
            else:
                prefix = ""

            if prefix:
                logger.info(f"Remapping state dict keys with prefix '{prefix}'")
                state = {f"{prefix}{k}": v for k, v in state.items()}

    return state, files


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_pytorch_int8(hf_repo, base_model, family):
    """Load our custom PyTorch INT8 format."""
    from huggingface_hub import hf_hub_download, list_repo_files

    model, processor = _load_base_model(base_model, family)
    logger.info("Base model loaded on CPU")

    logger.info(f"Loading INT8 weights from {hf_repo}...")
    state, files = _load_quant_state(hf_repo, model)

    # Detect group_size from scale shape or config
    group_size = None
    files = list_repo_files(hf_repo)
    for cfg_name in ["quant_config.json", "quant_info.json"]:
        if cfg_name in files:
            quant_cfg = json.load(open(hf_hub_download(hf_repo, cfg_name)))
            gs = quant_cfg.get("group_size")
            if gs and gs != "per_channel":
                group_size = int(gs)
            break

    # Also detect from scale tensor shape
    if group_size is None:
        for k in state:
            if k.endswith(".scale"):
                s = state[k]
                if len(s.shape) == 2 and s.shape[1] > 1:
                    # Per-group: scale is [out, n_groups]
                    w_key = k.replace(".scale", ".weight_int8")
                    if w_key in state:
                        in_feat = state[w_key].shape[1]
                        group_size = in_feat // s.shape[1]
                        break

    if group_size:
        logger.info(f"Detected per-group INT8 with group_size={group_size}")
    else:
        logger.info("Detected per-channel INT8")

    # Discover INT8 layers
    if "quant_config.json" in files:
        quant_cfg = json.load(open(hf_hub_download(hf_repo, "quant_config.json")))
        int8_layers = quant_cfg.get("int8_layers", {})
    else:
        int8_layers = {}
        for k in state:
            if k.endswith(".weight_int8"):
                layer_name = k.rsplit(".weight_int8", 1)[0]
                w = state[k]
                has_bias = f"{layer_name}.bias" in state
                int8_layers[layer_name] = {
                    "in_features": w.shape[1],
                    "out_features": w.shape[0],
                    "has_bias": has_bias,
                }
        logger.info(f"Discovered {len(int8_layers)} INT8 layers")

    replaced = 0
    quant_prefixes = set()
    for layer_name, layer_info in int8_layers.items():
        parts = layer_name.split(".")
        quant_prefixes.add(layer_name)

        try:
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
        except AttributeError:
            continue

        int8_layer = Int8Linear(
            layer_info["in_features"], layer_info["out_features"],
            bias=layer_info.get("has_bias", False),
            group_size=group_size)

        wkey = f"{layer_name}.weight_int8"
        skey = f"{layer_name}.scale"
        bkey = f"{layer_name}.bias"
        if wkey in state:
            int8_layer.weight_int8.data = state[wkey]
        if skey in state:
            int8_layer.scale.data = state[skey]
        if bkey in state and int8_layer.bias is not None:
            int8_layer.bias.data = state[bkey]

        setattr(parent, parts[-1], int8_layer)
        replaced += 1

    logger.info(f"Replaced {replaced} layers with Int8Linear")

    # Load non-quantized weights (embeddings, norms, vision tower)
    non_quant_state = {}
    for k, v in state.items():
        prefix = k.rsplit('.', 1)[0] if '.' in k else k
        if prefix in quant_prefixes:
            continue
        non_quant_state[k] = v
    if non_quant_state:
        missing, unexpected = model.load_state_dict(non_quant_state, strict=False)
        logger.info(f"Loaded {len(non_quant_state)} non-quantized weights (missing={len(missing)}, unexpected={len(unexpected)})")

    model = model.to("cuda")
    model.eval()
    return model, processor


def load_pytorch_int4(hf_repo, base_model, family):
    """Load our custom PyTorch INT4 format."""
    from huggingface_hub import hf_hub_download, list_repo_files

    model, processor = _load_base_model(base_model, family)
    logger.info("Base model loaded on CPU")

    logger.info(f"Loading INT4 weights from {hf_repo}...")
    state, files = _load_quant_state(hf_repo, model)

    # Detect dequant mode from state dict keys (needed for both branches below)
    uses_add_mode = any(k.endswith(".zeros") for k in state)
    dequant_mode = "add" if uses_add_mode else "sub"

    # Try loading quant_config.json or quant_info.json
    group_size = 128
    if "quant_config.json" in files:
        quant_cfg = json.load(open(hf_hub_download(hf_repo, "quant_config.json")))
        group_size = quant_cfg.get("group_size", 128)
        int4_layers = quant_cfg.get("int4_layers", {})
    else:
        if "quant_info.json" in files:
            quant_cfg = json.load(open(hf_hub_download(hf_repo, "quant_info.json")))
            group_size = quant_cfg.get("group_size", 128)

        # Discover INT4 layers from keys
        # Handle both naming conventions: scale/zero_point AND scales/zeros
        # "scales"/"zeros" naming → dequant as w*s+z ("add" mode)
        # "scale"/"zero_point" naming → dequant as (w-z)*s ("sub" mode)
        int4_layers = {}
        uses_add_mode = any(k.endswith(".zeros") for k in state)
        for k in state:
            if k.endswith(".weight_packed"):
                layer_name = k.rsplit(".weight_packed", 1)[0]
                wp = state[k]
                # Try both scale naming conventions
                s = state.get(f"{layer_name}.scale") or state.get(f"{layer_name}.scales")
                if s is not None:
                    out_features = wp.shape[0]
                    in_features = wp.shape[1] * 2  # 2 values packed per byte
                    n_groups = s.shape[1] if len(s.shape) > 1 else s.shape[0]
                    if n_groups > 0:
                        group_size = in_features // n_groups
                    has_bias = f"{layer_name}.bias" in state
                    int4_layers[layer_name] = {
                        "in_features": in_features,
                        "out_features": out_features,
                        "has_bias": has_bias,
                    }
        dequant_mode = "add" if uses_add_mode else "sub"
        logger.info(f"Discovered {len(int4_layers)} INT4 layers from keys (dequant_mode={dequant_mode})")

    replaced = 0
    skipped_layers = []
    for layer_name, layer_info in int4_layers.items():
        parts = layer_name.split(".")

        # Navigate to the original layer and validate dimensions
        try:
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            orig_layer = getattr(parent, parts[-1])
        except AttributeError:
            skipped_layers.append(layer_name)
            continue

        # Check if original layer is nn.Linear and dimensions match
        if isinstance(orig_layer, nn.Linear):
            expected_in = orig_layer.in_features
            expected_out = orig_layer.out_features
            got_in = layer_info["in_features"]
            got_out = layer_info["out_features"]
            if got_in != expected_in or got_out != expected_out:
                logger.warning(f"Skipping {layer_name}: INT4 dims [{got_out},{got_in}] != model [{expected_out},{expected_in}]")
                skipped_layers.append(layer_name)
                continue
        elif isinstance(orig_layer, nn.Embedding):
            # Embeddings should not be quantized to INT4 — skip and keep original
            logger.warning(f"Skipping {layer_name}: is nn.Embedding, not nn.Linear")
            skipped_layers.append(layer_name)
            continue

        unpack_order = "low_high" if uses_add_mode else "high_low"
        int4_layer = Int4Linear(
            layer_info["in_features"], layer_info["out_features"],
            group_size=group_size,
            bias=layer_info.get("has_bias", False),
            dequant_mode=dequant_mode,
            unpack_order=unpack_order)

        wkey = f"{layer_name}.weight_packed"
        # Handle both naming conventions
        skey = f"{layer_name}.scale" if f"{layer_name}.scale" in state else f"{layer_name}.scales"
        zkey = f"{layer_name}.zero_point" if f"{layer_name}.zero_point" in state else f"{layer_name}.zeros"
        bkey = f"{layer_name}.bias"
        if wkey in state:
            int4_layer.weight_packed.data = state[wkey]
        if skey in state:
            int4_layer.scale.data = state[skey]
        if zkey in state:
            int4_layer.zero_point.data = state[zkey]
        if bkey in state and int4_layer.bias is not None:
            int4_layer.bias.data = state[bkey]

        setattr(parent, parts[-1], int4_layer)
        replaced += 1

    if skipped_layers:
        logger.warning(f"Skipped {len(skipped_layers)} layers with dimension mismatch (keeping FP16)")
    logger.info(f"Replaced {replaced} layers with Int4Linear")

    # Load non-quantized weights (embeddings, norms, vision tower, etc.)
    # Filter out quantized layer keys to avoid mismatches with renamed buffers
    quant_suffixes = ('.weight_packed', '.scale', '.zero_point', '.scales', '.zeros')
    quant_prefixes = set(int4_layers.keys())
    non_quant_state = {}
    for k, v in state.items():
        prefix = k.rsplit('.', 1)[0] if '.' in k else k
        if prefix in quant_prefixes:
            continue
        non_quant_state[k] = v
    if non_quant_state:
        missing, unexpected = model.load_state_dict(non_quant_state, strict=False)
        logger.info(f"Loaded {len(non_quant_state)} non-quantized weights (missing={len(missing)}, unexpected={len(unexpected)})")

    model = model.to("cuda")
    model.eval()
    return model, processor


def load_hqq_int4(hf_repo, base_model, family):
    """Load HQQ INT4 — keep weights packed as INT4, dequantize on-the-fly in forward()."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    model, processor = _load_base_model(base_model, family)
    logger.info("Base model loaded on CPU")

    logger.info(f"Loading HQQ weights from {hf_repo}...")
    state = load_file(hf_hub_download(hf_repo, "model.safetensors"), device="cpu")

    hqq_suffixes = ['.W_q', '.scale', '.zero', '.shape', '.group_size', '.nbits',
                    '.axis', '.channel_wise', '.compute_dtype', '.encoded_state_dict',
                    '.offload_meta', '.optimize', '.packing', '.quant_scale',
                    '.quant_zero', '.round_zero', '.stores_quant_config',
                    '.unpack_view_dtype', '.view_as_float']

    # Load non-HQQ tensors first, then free from state
    non_hqq_keys = [k for k in state if not any(k.endswith(s) for s in hqq_suffixes)]
    if non_hqq_keys:
        non_hqq = {k: state[k] for k in non_hqq_keys}
        model.load_state_dict(non_hqq, strict=False)
        logger.info(f"Loaded {len(non_hqq_keys)} non-HQQ tensors")
        del non_hqq
        for k in non_hqq_keys:
            del state[k]
        gc.collect()

    # Replace nn.Linear with HqqInt4Linear — keep weights packed
    wq_keys = [k for k in state if k.endswith('.W_q')]
    logger.info(f"Found {len(wq_keys)} HQQ layers, replacing with HqqInt4Linear...")

    for i, wq_key in enumerate(wq_keys):
        prefix = wq_key.rsplit('.W_q', 1)[0]
        shape = state[f"{prefix}.shape"].tolist()
        group_size = int(state[f"{prefix}.group_size"].item())
        out_feat, in_feat = int(shape[0]), int(shape[1])

        # Get bias from original linear layer
        parts = prefix.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        orig_linear = getattr(parent, parts[-1])
        bias_data = orig_linear.bias.data.clone() if orig_linear.bias is not None else None

        # Create HqqInt4Linear and load packed weights directly
        hqq_layer = HqqInt4Linear(out_feat, in_feat, group_size, bias_data)
        hqq_layer.W_q.copy_(state[f"{prefix}.W_q"].reshape(-1))
        hqq_layer.scale.copy_(state[f"{prefix}.scale"])
        hqq_layer.zero.copy_(state[f"{prefix}.zero"])
        setattr(parent, parts[-1], hqq_layer)

        # Free HQQ tensors for this layer
        for suffix in hqq_suffixes:
            key = f"{prefix}{suffix}"
            if key in state:
                del state[key]
        del orig_linear

        if (i + 1) % 50 == 0:
            gc.collect()
            logger.info(f"  Replaced {i+1}/{len(wq_keys)} layers")

    del state
    gc.collect()
    logger.info(f"Replaced all {len(wq_keys)} layers with HqqInt4Linear")

    logger.info("Moving model to GPU...")
    model = model.to("cuda")
    model.eval()
    return model, processor


def load_gptq_int4(hf_repo, base_model, family):
    """Load GPTQ INT4 format — pre-dequantize to FP16."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    model, processor = _load_base_model(base_model, family)
    logger.info("Base model loaded on CPU")

    logger.info(f"Loading GPTQ weights from {hf_repo}...")
    state = load_file(hf_hub_download(hf_repo, "model.safetensors"), device="cpu")

    # Read bits from config
    try:
        cfg = json.load(open(hf_hub_download(hf_repo, "config.json")))
        bits = cfg.get("quantization_config", {}).get("bits", 4)
    except Exception:
        bits = 4

    qw_keys = [k for k in state if k.endswith('.qweight')]
    logger.info(f"Found {len(qw_keys)} GPTQ layers, bits={bits}")

    for qw_key in qw_keys:
        prefix = qw_key.rsplit('.qweight', 1)[0]
        qweight = state[f"{prefix}.qweight"]
        qzeros = state[f"{prefix}.qzeros"]
        scales = state[f"{prefix}.scales"]
        g_idx = state[f"{prefix}.g_idx"]
        bias = state.get(f"{prefix}.bias", None)

        pack_factor = 32 // bits
        unpacked = [((qweight >> (bits * i)) & ((1 << bits) - 1)) for i in range(pack_factor)]
        w_int = torch.stack(unpacked, dim=1).reshape(-1, qweight.shape[1]).to(torch.int32)

        z_unpacked = [((qzeros >> (bits * i)) & ((1 << bits) - 1)) for i in range(pack_factor)]
        zeros = torch.stack(z_unpacked, dim=2).reshape(qzeros.shape[0], -1).to(torch.int32)

        g = g_idx.long()
        w_fp = (w_int.to(torch.float16) - zeros[g].to(torch.float16)) * scales[g]
        w_fp16 = w_fp.T  # [in, out] -> [out, in]

        parts = prefix.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        linear = getattr(parent, parts[-1])
        linear.weight.data = w_fp16.contiguous()
        if bias is not None:
            linear.bias = nn.Parameter(bias)

    non_gptq = {k: state[k] for k in state if not any(
        k.endswith(s) for s in ['.qweight', '.qzeros', '.scales', '.g_idx'])}
    if non_gptq:
        missing, unexpected = model.load_state_dict(non_gptq, strict=False)
        logger.info(f"Loaded {len(non_gptq)} non-GPTQ tensors")

    model = model.to("cuda")
    model.eval()
    return model, processor


# ── Benchmark ────────────────────────────────────────────────────────────────

def benchmark(hf_repo, base_model, family, n_samples=50, force=False, method_override=None):
    from huggingface_hub import hf_hub_download

    fmt, cfg = detect_format(hf_repo)
    logger.info(f"Detected format: {fmt}")

    method_name = method_override or fmt
    results_dir = Path(__file__).resolve().parents[1] / "results" / "hf_pipeline"
    results_dir.mkdir(parents=True, exist_ok=True)

    safe_name = f"{base_model.replace('/', '__')}__{method_name}"
    out_path = results_dir / f"{safe_name}.json"

    if out_path.exists() and not force:
        logger.info(f"Result exists: {out_path}. Use --force.")
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1e6

    t_load = time.perf_counter()
    loaders = {
        "pytorch_int8": load_pytorch_int8,
        "pytorch_int4": load_pytorch_int4,
        "hqq_int4": load_hqq_int4,
        "gptq_int4": load_gptq_int4,
    }
    model, processor = loaders[fmt](hf_repo, base_model, family)
    load_time = time.perf_counter() - t_load

    mem_after = torch.cuda.memory_allocated() / 1e6
    logger.info(f"Loaded in {load_time:.1f}s, GPU mem: {mem_before:.0f} -> {mem_after:.0f} MB")

    samples = load_vqav2(n_samples)

    device = "cuda"
    profiler = GPUProfiler(device_index=0)
    scores, multi_scores, latencies = [], [], []
    skipped = 0

    logger.info(f"Running VQAv2 ({n_samples} samples)...")
    with profiler:
        for sample in tqdm(samples, desc=fmt):
            t0 = time.perf_counter()
            try:
                pred = run_inference(model, processor, sample, family, device)
            except Exception as e:
                logger.warning(f"Skip: {e}")
                skipped += 1
                torch.cuda.empty_cache()
                continue
            latencies.append(time.perf_counter() - t0)
            m = _vqa_multi_metric(pred, sample["answers"])
            multi_scores.append(m)
            scores.append(m["exact_match"])

    n_eval = len(scores)
    stats = profiler.stats()
    avg_acc = sum(scores) / n_eval if n_eval else 0.0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    throughput = len(latencies) / stats.wall_time_s if stats.wall_time_s > 0 else 0.0

    metric_avgs = {}
    if multi_scores:
        metric_avgs = {
            name: round(sum(m[name] for m in multi_scores) / len(multi_scores), 4)
            for name in multi_scores[0].keys()
        }

    logger.info(f"Results: EM={avg_acc:.4f}, Lat={avg_lat:.2f}s, Mem={stats.peak_memory_mb:.0f}MB")
    logger.info(f"Multi-metric: {metric_avgs}")

    total_params = sum(p.numel() for p in model.parameters()) + sum(b.numel() for b in model.buffers())

    result = {
        "model_id": base_model,
        "hf_repo": hf_repo,
        "family": family,
        "method": method_name,
        "num_params_M": round(total_params / 1e6, 1),
        "gpu_mem_before_mb": round(mem_before, 1),
        "gpu_mem_after_mb": round(mem_after, 1),
        "gpu_mem_load_mb": round(mem_after - mem_before, 1),
        "load_time_s": round(load_time, 1),
        "benchmarks": {
            "vqav2": {
                "accuracy": round(avg_acc, 4),
                "avg_latency_s": round(avg_lat, 4),
                "peak_memory_mb": round(stats.peak_memory_mb, 1),
                "avg_memory_mb": round(stats.avg_memory_mb, 1),
                "throughput_sps": round(throughput, 3),
                "n_samples": n_samples,
                "n_evaluated": n_eval,
                "n_skipped": skipped,
                "all_failed": n_eval == 0,
                "zero_accuracy_warning": avg_acc == 0.0 and n_eval >= 5,
                "metrics": metric_avgs,
            }
        },
        "device": "jetson_orin_nano_8gb",
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved to {out_path}")

    del model, processor
    torch.cuda.empty_cache()
    gc.collect()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--method", default=None, help="Override method name for output file (e.g. gptq_int4)")
    args = parser.parse_args()
    benchmark(args.repo, args.base_model, args.family, args.n_samples, args.force, method_override=args.method)
