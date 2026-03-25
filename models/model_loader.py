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
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)

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
    "moondream/moondream": "moondream",
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
    model = AutoModelForVision2Seq.from_pretrained(model_id, **kwargs)
    return model, processor


def _load_nanovlm(model_id: str, bnb: Optional[BitsAndBytesConfig],
                  dtype: torch.dtype, device_map: str):
    # nanoVLM uses AutoModelForVision2Seq compatible interface.
    # Some community uploads (e.g. lusxvr/nanoVLM-*) lack model_type in
    # config.json, causing AutoProcessor to fail. Raise a clear error so the
    # evaluation script can log the skip and continue.
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except ValueError as e:
        raise RuntimeError(
            f"Cannot load processor for nanoVLM '{model_id}': {e}. "
            "The HuggingFace repo may be missing model_type in config.json."
        ) from e
    kwargs = dict(trust_remote_code=True, device_map=device_map)
    if bnb:
        kwargs["quantization_config"] = bnb
    else:
        kwargs["torch_dtype"] = dtype
    model = AutoModelForVision2Seq.from_pretrained(model_id, **kwargs)
    return model, processor


def _load_lfm2vl(model_id: str, bnb: Optional[BitsAndBytesConfig],
                 dtype: torch.dtype, device_map: str):
    from transformers import AutoModelForImageTextToText
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
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return model, processor


def _load_fastvlm(model_id: str, bnb: Optional[BitsAndBytesConfig],
                  dtype: torch.dtype, device_map: str):
    # apple/FastVLM uses a custom FastViT-HD image processor. Some HF uploads
    # lack preprocessor_config.json — raise a clear error if processor fails.
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except (OSError, ValueError) as e:
        raise RuntimeError(
            f"Cannot load processor for FastVLM '{model_id}': {e}. "
            "The HuggingFace repo may be missing preprocessor_config.json."
        ) from e
    kwargs = dict(trust_remote_code=True, device_map=device_map)
    if bnb:
        kwargs["quantization_config"] = bnb
    else:
        kwargs["torch_dtype"] = dtype
    model = AutoModelForVision2Seq.from_pretrained(model_id, **kwargs)
    return model, processor


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
    model = AutoModel.from_pretrained(model_id, **kwargs)
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
    from transformers import AutoModelForImageTextToText
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


def _load_ovis2(model_id: str, bnb: Optional[BitsAndBytesConfig],
                dtype: torch.dtype, device_map: str):
    # Ovis2 uses a custom architecture with its own preprocess_inputs API.
    kwargs = dict(trust_remote_code=True, device_map=device_map)
    if bnb:
        kwargs["quantization_config"] = bnb
    else:
        # Ovis2 visual tokenizer uses bfloat16 internally; float16 causes dtype mismatch
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
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
