"""
models/model_loader.py
======================
Unified model + processor loader for all VLM families.

Supported families:
  smolvlm     — HuggingFace SmolVLM / SmolVLM2 (encoder-decoder)
  nanovlm     — HuggingFace nanoVLM (decoder-only)
  lfm2vl      — Liquid AI LFM2-VL (hybrid recurrent)
  moondream   — Moondream AI Moondream2 (decoder-only)
  fastvlm     — Apple FastVLM (decoder-only)
  qwen25vl    — Alibaba Qwen2.5-VL (decoder-only)
  internvl25  — OpenGVLab InternVL2.5 (encoder-decoder)
  gemma3      — Google Gemma 3 multimodal (decoder-only, 4B/12B VLM variants)
  ovis2       — AIDC-AI Ovis2 (decoder-only, 1B/2B/4B/8B)

Usage:
  from models.model_loader import load_model
  model, processor, meta = load_model("HuggingFaceTB/SmolVLM-256M-Instruct")
"""

import gc
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import pynvml
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)


# ── Compatibility shim for transformers 5.x ───────────────────────────────
# transformers 5.x renamed all_tied_weights_keys → _tied_weights_keys but
# remote model code (InternVL, Moondream) still references the old name.
# We need a descriptor that supports both get and set.

class _TiedWeightsDescriptor:
    """Descriptor that proxies all_tied_weights_keys to _tied_weights_keys."""
    ATTR = '_all_tied_weights_keys_storage'

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self  # class-level access
        val = obj.__dict__.get(self.ATTR)
        if val is not None:
            return val
        val = getattr(obj, '_tied_weights_keys', None)
        return val if val is not None else {}

    def __set__(self, obj, value):
        obj.__dict__[self.ATTR] = value


def _patch_all_tied_weights_keys(cls):
    """Add all_tied_weights_keys descriptor to a PreTrainedModel class."""
    cls.all_tied_weights_keys = _TiedWeightsDescriptor()


def _unpatch_all_tied_weights_keys(cls):
    """Remove the compatibility descriptor."""
    try:
        del cls.all_tied_weights_keys
    except AttributeError:
        pass


# ── Family detection ───────────────────────────────────────────────────────

FAMILY_MAP: dict[str, str] = {
    # smolvlm
    "huggingfacetb/smolvlm": "smolvlm",
    "huggingfacetb/smolvlm2": "smolvlm",
    # nanovlm
    "lusxvr/nanovlm": "nanovlm",
    # lfm2-vl
    "liquidai/lfm2-vl": "lfm2vl",
    "liquidai/lfm2.5-vl": "lfm2vl",
    # moondream
    "vikhyatk/moondream": "moondream",
    # fastvlm
    "apple/fastvlm": "fastvlm",
    # florence-2
    "microsoft/florence-2": "florence2",
    # qwen2.5-vl
    "qwen/qwen2.5-vl": "qwen25vl",
    "qwen/qwen2-vl": "qwen25vl",
    # internvl2.5
    "opengvlab/internvl2_5": "internvl25",
    "opengvlab/internvl2-": "internvl25",
    # gemma3 (only 4b+ are VLMs; 1b is text-only)
    "google/gemma-3-": "gemma3",
    "google/gemma-3n-": "gemma3",
    # ovis2
    "aidc-ai/ovis2-": "ovis2",
}


def detect_family(model_id: str) -> str:
    key = model_id.lower()
    for prefix, family in FAMILY_MAP.items():
        if key.startswith(prefix):
            return family
    raise ValueError(
        f"Cannot detect family for '{model_id}'. "
        f"Add it to FAMILY_MAP or pass family= explicitly."
    )


# ── GPU memory helpers ─────────────────────────────────────────────────────

def _gpu_mem_mb() -> float:
    """Return current GPU used memory in MB (respects CUDA_VISIBLE_DEVICES)."""
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info(torch.cuda.current_device())
    return (total - free) / 1024**2


def _log_gpu_mem(tag: str) -> float:
    used = _gpu_mem_mb()
    logger.info(f"[GPU mem] {tag}: {used:.1f} MB used")
    return used


# ── Quantization config builder ────────────────────────────────────────────

def build_bnb_config(quant: Optional[str]) -> Optional[BitsAndBytesConfig]:
    """
    quant: None | 'fp16' | 'int8' | 'int4'
    Returns BitsAndBytesConfig or None (fp16 uses torch dtype instead).
    """
    if quant is None or quant == "fp16":
        return None
    if quant == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quant == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    raise ValueError(f"Unknown quant mode '{quant}'. Use None, 'fp16', 'int8', or 'int4'.")


# ── Return type ────────────────────────────────────────────────────────────

@dataclass
class ModelMeta:
    model_id: str
    family: str
    quant: Optional[str]
    dtype: torch.dtype
    gpu_mem_before_mb: float
    gpu_mem_after_mb: float
    gpu_mem_delta_mb: float
    device: str
    extra: dict = field(default_factory=dict)


# ── Family-specific loaders ────────────────────────────────────────────────

def _load_florence2(model_id: str, bnb: Optional[BitsAndBytesConfig],
                    dtype: torch.dtype, device_map: str):
    from transformers import AutoModelForCausalLM, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    # attn_implementation="eager" avoids the _supports_sdpa AttributeError
    # present in Florence-2's custom modeling code.
    kwargs = dict(trust_remote_code=True, device_map=device_map,
                  attn_implementation="eager")
    if bnb:
        kwargs["quantization_config"] = bnb
    else:
        kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return model, processor


def _load_smolvlm(model_id: str, bnb: Optional[BitsAndBytesConfig],
                  dtype: torch.dtype, device_map: str):
    processor = AutoProcessor.from_pretrained(model_id)
    kwargs = dict(device_map=device_map)
    if bnb:
        kwargs["quantization_config"] = bnb
    else:
        kwargs["torch_dtype"] = dtype
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    return model, processor


def _load_nanovlm(model_id: str, bnb: Optional[BitsAndBytesConfig],
                  dtype: torch.dtype, device_map: str):
    # nanoVLM (lusxvr/nanoVLM-*) uses a custom 'nanovlm' library, not standard
    # transformers Auto classes.  We clone the repo, temporarily swap sys.path
    # and hide our own 'models' package so nanoVLM's intra-package imports work.
    import sys, os, json, dataclasses
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_model as st_load_model

    nanovlm_repo = "/tmp/nanoVLM"
    if not os.path.isdir(nanovlm_repo):
        import subprocess
        subprocess.check_call(
            ["git", "clone", "--depth", "1",
             "https://github.com/huggingface/nanoVLM.git", nanovlm_repo],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    # Download weights + config from HF first (before path swap)
    config_path = hf_hub_download(model_id, "config.json")
    weights_path = hf_hub_download(model_id, "model.safetensors")

    # Temporarily swap sys.path and hide our 'models' + 'data' modules
    # so nanoVLM's imports (from models.X, from data.X) resolve to its own code.
    saved_path = sys.path[:]
    saved_modules = {}
    for mod_name in list(sys.modules):
        if mod_name == "models" or mod_name.startswith("models.") or \
           mod_name == "data" or mod_name.startswith("data."):
            saved_modules[mod_name] = sys.modules.pop(mod_name)

    sys.path.insert(0, nanovlm_repo)
    try:
        from models.vision_language_model import VisionLanguageModel as NanoVLM
        from models.config import VLMConfig
        from data.processors import get_tokenizer, get_image_processor, get_image_string
        _proc_mod_funcs = (get_tokenizer, get_image_processor, get_image_string)

        # Load config, filtering out unknown fields for compatibility
        with open(config_path) as f:
            raw_cfg = json.load(f)
        valid_fields = {fld.name for fld in dataclasses.fields(VLMConfig)}
        filtered_cfg = {k: v for k, v in raw_cfg.items() if k in valid_fields}
        cfg = VLMConfig(**filtered_cfg)

        model = NanoVLM(cfg, load_backbone=False)
        st_load_model(model, weights_path)
    finally:
        # Restore our modules and path
        # Remove nanoVLM's modules from cache
        for mod_name in list(sys.modules):
            if mod_name == "models" or mod_name.startswith("models.") or \
               mod_name == "data" or mod_name.startswith("data."):
                sys.modules.pop(mod_name, None)
        sys.modules.update(saved_modules)
        sys.path = saved_path

    model = model.to(dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Build a processor wrapper compatible with our evaluation pipeline
    processor = _NanoVLMProcessor(model, _proc_mod_funcs)
    return model, processor


class _NanoVLMProcessor:
    """Wraps nanoVLM tokenizer + image processing for our eval interface."""

    def __init__(self, model, proc_mod_funcs):
        self.tokenizer = model.tokenizer
        self.cfg = model.cfg
        _get_tokenizer, _get_image_processor, self._get_image_string = proc_mod_funcs
        max_img = getattr(self.cfg, 'max_img_size', None) or self.cfg.vit_img_size
        # max_img_size must be divisible by vit_img_size (nanoVLM's
        # SplitImage splits into vit_img_size patches).  Round down.
        # Also cap to what the tokenizer supports (grid tokens r{i}c{j}).
        vis = self.cfg.vit_img_size
        max_img = (max_img // vis) * vis or vis
        # Find max supported grid by checking tokenizer grid tokens
        max_grid = 1
        for g in range(2, 20):
            if hasattr(self.tokenizer, f'r{g}c{g}'):
                max_grid = g
            else:
                break
        max_supported = max_grid * vis
        if max_img > max_supported:
            max_img = max_supported
        resize_max = getattr(self.cfg, 'resize_to_max_side_len', None) or False
        self._image_transform = _get_image_processor(
            max_img,
            self.cfg.vit_img_size,
            resize_max,
        )
        self._mp_image_token_length = getattr(
            self.cfg, 'mp_image_token_length',
            (self.cfg.vit_img_size // self.cfg.vit_patch_size) ** 2
            // (self.cfg.mp_pixel_shuffle_factor ** 2)
        )

    def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
        # nanoVLM uses its own chat template via the tokenizer
        # Extract text from messages for a simple prompt
        parts = []
        has_image = False
        for msg in messages:
            if isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if item.get("type") == "image":
                        has_image = True
                    elif item.get("type") == "text":
                        parts.append(item["text"])
            elif isinstance(msg.get("content"), str):
                parts.append(msg["content"])
        text = " ".join(parts)
        # Use nanoVLM chat template format
        chat_messages = [{"role": "user", "content": text}]
        prompt = self.tokenizer.apply_chat_template(
            chat_messages, add_generation_prompt=True, tokenize=False
        )
        return prompt

    def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
        import torchvision.transforms as T
        result = {}
        processed_images = None
        splitted_image_counts = []

        if images is not None:
            if not isinstance(images, (list, tuple)):
                images = [images]
            all_image_tensors = []
            for img in images:
                transformed = self._image_transform(img)
                # GlobalAndSplitImages returns (patches_tensor, (n_h, n_w))
                if isinstance(transformed, tuple) and len(transformed) == 2 \
                        and isinstance(transformed[1], tuple):
                    patches, grid = transformed
                    all_image_tensors.append(patches)
                    splitted_image_counts.append(grid)
                elif isinstance(transformed, (list, tuple)):
                    all_image_tensors.extend(transformed)
                    n_total = len(transformed)
                    splitted_image_counts.append((1, max(1, n_total - 1)))
                else:
                    all_image_tensors.append(transformed)
                    splitted_image_counts.append((1, 1))

            # Build image token string and prepend to text
            image_string = self._get_image_string(
                self.tokenizer, splitted_image_counts,
                self._mp_image_token_length,
            )
            if text:
                # Insert image tokens before the user's text in the prompt
                text = text.replace("<|im_start|>user\n",
                                   f"<|im_start|>user\n{image_string}")
            processed_images = [t.unsqueeze(0) if t.dim() == 3 else t
                               for t in all_image_tensors]

        if text is not None:
            tok = self.tokenizer(text, return_tensors=return_tensors,
                                padding=True, truncation=True)
            result["input_ids"] = tok["input_ids"]
            if "attention_mask" in tok:
                result["attention_mask"] = tok["attention_mask"]

        if processed_images is not None:
            result["images"] = processed_images

        return _BatchEncoding(result)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)


def _load_lfm2vl(model_id: str, bnb: Optional[BitsAndBytesConfig],
                 dtype: torch.dtype, device_map: str):
    processor = AutoProcessor.from_pretrained(model_id)
    kwargs = dict(device_map=device_map)
    if bnb:
        kwargs["quantization_config"] = bnb
    else:
        kwargs["torch_dtype"] = dtype
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    return model, processor


def _load_moondream(model_id: str, bnb: Optional[BitsAndBytesConfig],
                    dtype: torch.dtype, device_map: str):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    kwargs = dict(trust_remote_code=True, device_map=device_map)
    if bnb:
        kwargs["quantization_config"] = bnb
    else:
        kwargs["torch_dtype"] = dtype
    # transformers 5.x renamed all_tied_weights_keys → _tied_weights_keys;
    # Moondream's remote code still references the old name.
    from transformers import PreTrainedModel
    _patched = not hasattr(PreTrainedModel, 'all_tied_weights_keys')
    if _patched:
        _patch_all_tied_weights_keys(PreTrainedModel)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    finally:
        if _patched:
            _unpatch_all_tied_weights_keys(PreTrainedModel)
    return model, processor


def _load_fastvlm(model_id: str, bnb: Optional[BitsAndBytesConfig],
                  dtype: torch.dtype, device_map: str):
    # apple/FastVLM is a LLaVA-Qwen2 variant using MobileCLIP vision tower.
    # Load model first, then grab the image processor from the vision tower
    # (it knows the correct image size, e.g. 1024 for mobileclip_l_1024).
    kwargs = dict(trust_remote_code=True, device_map=device_map)
    if bnb:
        kwargs["quantization_config"] = bnb
    else:
        kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    # Get the image processor from the vision tower (correct size)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    vt = model.get_vision_tower()
    # Vision tower may lazily load in float32 — cast entire model to target dtype
    model.to(dtype=dtype)
    image_processor = vt.image_processor
    processor = _FastVLMProcessor(image_processor, tokenizer)

    return model, processor


class _FastVLMProcessor:
    """Minimal combined processor for FastVLM (LLaVA-Qwen2) models.

    Wraps CLIPImageProcessor + Qwen2Tokenizer to provide the same interface
    as a HuggingFace processor: __call__(text, images, return_tensors) and
    apply_chat_template / batch_decode.
    """

    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
        result = {}
        if images is not None:
            if not isinstance(images, (list, tuple)):
                images = [images]
            pixel_values = self.image_processor(
                images=images, return_tensors=return_tensors
            )["pixel_values"]
            result["pixel_values"] = pixel_values
        if text is not None:
            tok = self.tokenizer(
                text, return_tensors=return_tensors,
                padding=True, truncation=True, **kwargs
            )
            result["input_ids"] = tok["input_ids"]
            if "attention_mask" in tok:
                result["attention_mask"] = tok["attention_mask"]
        # Return a dict-like object that supports .to(device)
        return _BatchEncoding(result)

    def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
        # Qwen2's chat template expects content as a plain string, not a list
        # of {"type": "image"/"text"} dicts.  Flatten before passing through.
        flat_messages = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [item["text"] for item in content
                              if item.get("type") == "text"]
                content = " ".join(text_parts)
            flat_messages.append({"role": msg["role"], "content": content})
        # Always return a string (tokenize=False).  Drop 'tokenize' from
        # kwargs so callers can pass it without causing a duplicate-arg error.
        kwargs.pop("tokenize", None)
        return self.tokenizer.apply_chat_template(
            flat_messages, add_generation_prompt=add_generation_prompt,
            tokenize=False, **kwargs
        )

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)


class _BatchEncoding(dict):
    """Dict subclass with .to(device) support for processor output."""

    def to(self, device):
        import torch as _torch
        return _BatchEncoding({
            k: v.to(device) if isinstance(v, _torch.Tensor) else v
            for k, v in self.items()
        })


def _load_qwen25vl(model_id: str, bnb: Optional[BitsAndBytesConfig],
                   dtype: torch.dtype, device_map: str):
    # Qwen2.5-VL uses Qwen2_5_VLForConditionalGeneration; fall back to
    # AutoModelForVision2Seq for older Qwen2-VL model IDs.
    processor = AutoProcessor.from_pretrained(model_id)
    # Qwen2.5-VL requires bfloat16 — float16 causes logit overflow / CUDA assertions
    # on certain images (same issue as Gemma3).
    kwargs = dict(device_map=device_map, torch_dtype=torch.bfloat16)
    if bnb:
        if bnb.load_in_4bit:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=bnb.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=bnb.bnb_4bit_quant_type,
            )
        kwargs["quantization_config"] = bnb
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    except (ImportError, AttributeError):
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    return model, processor


def _load_internvl25(model_id: str, bnb: Optional[BitsAndBytesConfig],
                     dtype: torch.dtype, device_map: str):
    # InternVL2.5 uses AutoModel with trust_remote_code
    from transformers import AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    kwargs = dict(trust_remote_code=True, device_map=device_map)
    if bnb:
        kwargs["quantization_config"] = bnb
    else:
        kwargs["torch_dtype"] = dtype
    # InternVL's custom ViT code calls torch.linspace(...).item() inside __init__
    # which fails under transformers 5.x meta-device init.  Monkey-patch
    # torch.linspace to fall back to CPU when the default device is meta.
    _orig_linspace = torch.linspace
    def _safe_linspace(*args, **kw):
        try:
            return _orig_linspace(*args, **kw)
        except RuntimeError:
            # Force CPU creation, then move back to original device
            kw.pop("device", None)
            return _orig_linspace(*args, device="cpu", **kw)
    torch.linspace = _safe_linspace
    # transformers 5.x renamed all_tied_weights_keys → _tied_weights_keys.
    # InternVL's remote code still references the old name; patch PreTrainedModel.
    # Must support both get (returns dict or _tied_weights_keys) and set (post_init
    # assigns to it).
    from transformers import PreTrainedModel
    _patched_tied = not hasattr(PreTrainedModel, 'all_tied_weights_keys')
    if _patched_tied:
        _patch_all_tied_weights_keys(PreTrainedModel)

    try:
        model = AutoModel.from_pretrained(model_id, **kwargs)
    finally:
        torch.linspace = _orig_linspace
        if _patched_tied:
            _unpatch_all_tied_weights_keys(PreTrainedModel)
    # transformers >= 4.50 dropped GenerationMixin from PreTrainedModel base class.
    # InternLM2's remote code doesn't yet inherit from it explicitly, so patch it.
    from transformers import GenerationMixin, GenerationConfig
    lm = model.language_model
    if not isinstance(lm, GenerationMixin):
        lm.__class__ = type(lm.__class__.__name__, (lm.__class__, GenerationMixin), {})
    if getattr(lm, "generation_config", None) is None:
        lm.generation_config = GenerationConfig()
    # Disable DynamicCache — InternLM2's custom code expects legacy tuple format
    # for past_key_values; DynamicCache (transformers >= 4.44) causes NoneType errors.
    lm.__class__._supports_default_dynamic_cache = lambda *args: False
    # Wrap tokenizer as processor-like object for uniform interface
    model.tokenizer = tokenizer
    return model, tokenizer


def _load_gemma3(model_id: str, bnb: Optional[BitsAndBytesConfig],
                 dtype: torch.dtype, device_map: str):
    processor = AutoProcessor.from_pretrained(model_id)
    # Gemma 3 requires bfloat16 — float16/float32 causes logit overflow/NaN on GPU.
    # Apply bfloat16 always, including when quantized (non-quantized layers + compute dtype).
    kwargs = dict(device_map=device_map, torch_dtype=torch.bfloat16)
    if bnb:
        if bnb.load_in_4bit:
            # Override INT4 compute dtype to bfloat16 (float16 also overflows on Gemma3)
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=bnb.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=bnb.bnb_4bit_quant_type,
            )
        kwargs["quantization_config"] = bnb
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    return model, processor


def _patch_ovis2_remote_code(model_id: str):
    """Patch cached Ovis2 remote code for transformers 5.x compatibility.

    Fixes: tie_weights(**kwargs), Qwen2Tokenizer in supported list.
    """
    import re
    from pathlib import Path
    safe = model_id.replace("/", "_hyphen_").replace("-", "_hyphen_")
    # Actually, HF caches as org_hyphen_name/repo_hyphen_name
    cache_base = Path.home() / ".cache/huggingface/modules/transformers_modules"
    # Find the actual cache dir by glob
    for d in list(cache_base.glob("*Ovis2*")) + list(cache_base.glob("*/*Ovis2*")):
        if not d.is_dir():
            continue
        # d might be the revision dir itself or a parent; check both
        subdirs = [d] if (d / "modeling_ovis.py").exists() else [s for s in d.iterdir() if s.is_dir()]
        for sub in subdirs:
            # Patch modeling_ovis.py: tie_weights
            mod_file = sub / "modeling_ovis.py"
            if mod_file.exists():
                txt = mod_file.read_text()
                if "def tie_weights(self):" in txt:
                    txt = txt.replace("def tie_weights(self):", "def tie_weights(self, **kwargs):")
                    mod_file.write_text(txt)
            # Patch modeling_ovis.py: remove flash_attn assertion
            if mod_file.exists():
                txt = mod_file.read_text()
                if 'is_flash_attn_2_available()' in txt:
                    lines = txt.split('\n')
                    new_lines = []
                    skip = 0
                    for line in lines:
                        if skip > 0:
                            skip -= 1
                            continue
                        if 'self.config.llm_attn_implementation == "flash_attention_2"' in line:
                            new_lines.append('            # flash_attn assertion removed for transformers 5.x compat')
                            skip = 3
                        else:
                            new_lines.append(line)
                    mod_file.write_text('\n'.join(new_lines))
            # Patch modeling_ovis.py: use custom AIMv2Model instead of native one
            # In transformers 5.x, AutoModel resolves aimv2 config to the native
            # Aimv2Model which has different weight key names (vision_model.encoder.layers
            # vs trunk.blocks). Force Ovis2's custom AIMv2Model from modeling_aimv2.py.
            if mod_file.exists():
                txt = mod_file.read_text()
                old_backbone = 'self.backbone = AutoModel.from_config(self.config.backbone_config)'
                new_backbone = (
                    'from .modeling_aimv2 import AIMv2Model as _CustomAIMv2\n'
                    '        self.backbone = _CustomAIMv2(self.config.backbone_config)'
                )
                if old_backbone in txt:
                    txt = txt.replace(old_backbone, new_backbone)
                    mod_file.write_text(txt)
            # Patch modeling_ovis.py: fix aimv2 backbone call (positional → keyword)
            # In transformers 5.x, Aimv2Model.forward gained input_ids as first param,
            # so passing pixel_values positionally maps to input_ids, leaving pixel_values=None.
            if mod_file.exists():
                txt = mod_file.read_text()
                if 'self.backbone(pixel_values, output_hidden_states=True' in txt:
                    txt = txt.replace(
                        'self.backbone(pixel_values, output_hidden_states=True, return_dict=True)',
                        'self.backbone(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)',
                    )
                    mod_file.write_text(txt)
            # Patch modeling_ovis.py: force eager attention instead of config value
            if mod_file.exists():
                txt = mod_file.read_text()
                if 'attn_kwargs["attn_implementation"] = self.config.llm_attn_implementation' in txt:
                    txt = txt.replace(
                        'attn_kwargs["attn_implementation"] = self.config.llm_attn_implementation',
                        'attn_kwargs["attn_implementation"] = "eager"  # forced for Jetson compat',
                    )
                    mod_file.write_text(txt)
            # Clear __pycache__ so stale bytecode doesn't override patches
            pycache = sub / "__pycache__"
            if pycache.is_dir():
                import shutil
                shutil.rmtree(pycache)
            # Patch configuration_ovis.py: tokenizer types
            cfg_file = sub / "configuration_ovis.py"
            if cfg_file.exists():
                txt = cfg_file.read_text()
                if "'Qwen2Tokenizer'" not in txt and "support_tokenizer_types" in txt:
                    txt = txt.replace(
                        "['QWenTokenizer', 'Qwen2TokenizerFast']",
                        "['QWenTokenizer', 'Qwen2TokenizerFast', 'Qwen2Tokenizer']"
                    )
                    cfg_file.write_text(txt)


def _load_ovis2(model_id: str, bnb: Optional[BitsAndBytesConfig],
                dtype: torch.dtype, device_map: str):
    # Ovis2 uses a custom architecture with its own preprocess_inputs API.
    # Workaround: Ovis2's remote code re-registers 'aimv2' which is now native
    # in transformers >= 5.x, causing a ValueError collision.  Monkey-patch
    # AutoConfig.register to silently skip duplicates.
    import transformers
    _orig_register = transformers.AutoConfig.register
    # In transformers 5.x, register is a plain function, not a classmethod
    is_classmethod = hasattr(_orig_register, '__func__')
    raw_func = _orig_register.__func__ if is_classmethod else _orig_register
    def _safe_register(*args, exist_ok=False, **kw):
        try:
            return raw_func(*args, exist_ok=exist_ok, **kw)
        except ValueError as e:
            if "already used" in str(e):
                pass
            else:
                raise
    if is_classmethod:
        transformers.AutoConfig.register = classmethod(_safe_register)
    else:
        transformers.AutoConfig.register = _safe_register

    kwargs = dict(trust_remote_code=True, device_map=device_map,
                  attn_implementation="eager")
    if bnb:
        kwargs["quantization_config"] = bnb
    else:
        # Ovis2 visual tokenizer uses bfloat16 internally; float16 causes dtype mismatch
        kwargs["torch_dtype"] = torch.bfloat16

    # Auto-patch cached Ovis2 remote code for transformers 5.x compatibility:
    # 1. tie_weights() needs **kwargs  2. tokenizer type list needs Qwen2Tokenizer
    # 3. flash_attn assertion removed (we use eager attention)
    _patch_ovis2_remote_code(model_id)

    # Still need the fake flash_attn module in sys.modules because Ovis2's
    # __init__.py may do a top-level import check.
    import sys, types, importlib.metadata, importlib.machinery
    _had_flash = 'flash_attn' in sys.modules
    if not _had_flash:
        _fake = types.ModuleType('flash_attn')
        _fake.__version__ = '2.7.0'
        _fake.__spec__ = importlib.machinery.ModuleSpec('flash_attn', None)
        sys.modules['flash_attn'] = _fake
    _orig_version = importlib.metadata.version
    def _patched_version(name):
        if name == 'flash_attn':
            return '2.7.0'
        return _orig_version(name)
    importlib.metadata.version = _patched_version

    # Ovis2's __init__ accesses llm.is_parallelizable which was removed in
    # transformers 5.x. Monkey-patch it back onto PreTrainedModel.
    from transformers import PreTrainedModel as _PTM
    _had_is_par = hasattr(_PTM, 'is_parallelizable')
    if not _had_is_par:
        _PTM.is_parallelizable = False

    # transformers 5.x renamed all_tied_weights_keys; Ovis2 custom model
    # doesn't define it, causing AttributeError during loading.
    _patched_tied = not hasattr(_PTM, 'all_tied_weights_keys')
    if _patched_tied:
        _patch_all_tied_weights_keys(_PTM)

    # Ovis2's tie_weights() in cached modeling_ovis.py has been patched to accept
    # **kwargs (transformers 5.x passes missing_keys kwarg). If the cache is
    # refreshed, this may need re-patching — see setup notes.

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    finally:
        importlib.metadata.version = _orig_version
        if not _had_flash and 'flash_attn' in sys.modules:
            del sys.modules['flash_attn']
        if not _had_is_par and hasattr(_PTM, 'is_parallelizable'):
            del _PTM.is_parallelizable
        if _patched_tied:
            _unpatch_all_tied_weights_keys(_PTM)

    # Restore original register
    transformers.AutoConfig.register = _orig_register
    # Return the model's text tokenizer as processor for uniform interface;
    # run_inference uses the full Ovis2 API via model.preprocess_inputs().
    processor = model.get_text_tokenizer()
    return model, processor


_LOADERS = {
    "smolvlm":    _load_smolvlm,
    "nanovlm":    _load_nanovlm,
    "lfm2vl":     _load_lfm2vl,
    "moondream":  _load_moondream,
    "fastvlm":    _load_fastvlm,
    "florence2":  _load_florence2,
    "qwen25vl":   _load_qwen25vl,
    "internvl25": _load_internvl25,
    "gemma3":     _load_gemma3,
    "ovis2":      _load_ovis2,
}


# ── Public API ─────────────────────────────────────────────────────────────

def load_model(
    model_id: str,
    quant: Optional[str] = None,
    family: Optional[str] = None,
    device_map: str = "auto",
) -> tuple[Any, Any, ModelMeta]:
    """
    Load a VLM model and its processor.

    Parameters
    ----------
    model_id  : HuggingFace model ID, e.g. 'HuggingFaceTB/SmolVLM-256M-Instruct'
    quant     : None | 'fp16' | 'int8' | 'int4'
    family    : Override auto-detected family string (see FAMILY_MAP keys)
    device_map: passed to from_pretrained (default 'auto')

    Returns
    -------
    model, processor, ModelMeta
    """
    if family is None:
        family = detect_family(model_id)

    if family not in _LOADERS:
        raise ValueError(f"No loader for family '{family}'. Available: {list(_LOADERS)}")

    dtype = torch.float16 if (quant is None or quant == "fp16") else torch.float32
    bnb   = build_bnb_config(quant)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading {model_id}  family={family}  quant={quant}  dtype={dtype}")

    # Force all layers to GPU 0 to prevent cross-device tensor errors during
    # inference on multi-GPU machines. All target models (up to 16B fp16) fit
    # within the 48 GB A6000 VRAM budget.
    if device_map == "auto" and torch.cuda.is_available():
        device_map = {"": 0}

    mem_before = _log_gpu_mem("before load")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    loader = _LOADERS[family]
    model, processor = loader(model_id, bnb, dtype, device_map)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    mem_after = _log_gpu_mem("after load")

    meta = ModelMeta(
        model_id=model_id,
        family=family,
        quant=quant,
        dtype=dtype,
        gpu_mem_before_mb=mem_before,
        gpu_mem_after_mb=mem_after,
        gpu_mem_delta_mb=mem_after - mem_before,
        device=device,
    )

    logger.info(
        f"Loaded {model_id} | VRAM delta: {meta.gpu_mem_delta_mb:.1f} MB"
    )
    return model, processor, meta


def unload_model(model: Any) -> None:
    """Delete model and free GPU memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model unloaded and GPU cache cleared.")
