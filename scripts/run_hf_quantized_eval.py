"""
scripts/run_hf_quantized_eval.py
=================================
Load pre-quantized SmolVLM-256M-Instruct models from HuggingFace (Azaz666)
and profile on VQAv2 (CPU-only, RPi5).

Mirrors the Jetson profiling pipeline (profiling/profile_all.py) with:
  - TokenTimingProcessor: per-token prefill/decode timing
  - CPUProfiler: RSS memory polling
  - Component hooks: per-module-category timing breakdown
  - Multi-metric accuracy (exact_match, contains, token_f1, bleu, rouge_l)

Supports 4 quantization methods:
  - PYTORCH-INT4: custom Int4Linear (weight_packed + scale + zero_point)
  - PYTORCH-INT8: custom Int8Linear (weight_int8 + scale)
  - HQQ-INT4:     HQQ library format (W_q + scale + zero + metadata)
  - GPTQ-INT4:    same format as PYTORCH-INT4 (GPTQ-calibrated, custom saved)

Usage:
  python scripts/run_hf_quantized_eval.py --method PYTORCH-INT4 --vqav2_n 50
  python scripts/run_hf_quantized_eval.py --method all --vqav2_n 50 --components
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
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.run_baseline import (
    load_vqav2,
    run_inference,
    _vqa_accuracy,
    _normalize,
    _token_f1,
    _bleu_single,
    _rouge_l,
    _best_gold,
)
from profiling.token_timer import TokenTimingProcessor
from profiling.cpu_profiler import CPUProfiler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "hf_quantized"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _vqa_multi_metric(pred: str, gold_answers: list) -> dict:
    """Compute multi-metric VQA scores (matches Jetson profiling format)."""
    pred_norm = _normalize(pred)
    gold = _best_gold(gold_answers)
    return {
        "exact_match": 1.0 if pred_norm == gold else 0.0,
        "contains": 1.0 if (gold in pred_norm or pred_norm in gold) else 0.0,
        "token_f1": _token_f1(pred, gold),
        "bleu": _bleu_single(pred, gold),
        "rouge_l": _rouge_l(pred, gold),
    }

HF_USER = "Azaz666"

# All sub-500M models we can profile on RPi5
MODELS = {
    "SmolVLM-256M": {
        "model_id": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "family": "smolvlm",
        "methods": {
            "FP32-BASELINE": "HuggingFaceTB/SmolVLM-256M-Instruct",
            "PYTORCH-INT4": f"{HF_USER}/SmolVLM-256M-Instruct-PYTORCH-INT4",
            "PYTORCH-INT8": f"{HF_USER}/SmolVLM-256M-Instruct-PYTORCH-INT8",
            "HQQ-INT4":     f"{HF_USER}/SmolVLM-256M-Instruct-HQQ-INT4",
            "GPTQ-INT4":    f"{HF_USER}/SmolVLM-256M-Instruct-GPTQ-INT4",
        },
    },
    "LFM2-VL-450M": {
        "model_id": "LiquidAI/LFM2-VL-450M",
        "family": "lfm2vl",
        "methods": {
            "FP32-BASELINE": "LiquidAI/LFM2-VL-450M",
        },
    },
    "SmolVLM-500M": {
        "model_id": "HuggingFaceTB/SmolVLM-500M-Instruct",
        "family": "smolvlm",
        "methods": {
            "FP32-BASELINE": "HuggingFaceTB/SmolVLM-500M-Instruct",
        },
    },
    "FastVLM-0.5B": {
        "model_id": "apple/FastVLM-0.5B",
        "family": "fastvlm",
        "methods": {
            "FP32-BASELINE": "apple/FastVLM-0.5B",
        },
    },
    "Florence-2-base-ft": {
        "model_id": "microsoft/Florence-2-base-ft",
        "family": "florence2",
        "methods": {
            "FP32-BASELINE": "microsoft/Florence-2-base-ft",
        },
    },
}

# Default for backward compat
BASE_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
FAMILY = "smolvlm"
METHODS = MODELS["SmolVLM-256M"]["methods"]


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


# ── Int4Linear (for PYTORCH-INT4 and GPTQ-INT4) ────────────────────────────

class Int4Linear(nn.Module):
    """INT4 linear layer: 2 int4 values packed per uint8 byte, per-group quant."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 group_size: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        self.in_features_padded = ((in_features + group_size - 1) // group_size) * group_size
        n_groups = self.in_features_padded // group_size
        packed_size = self.in_features_padded // 2

        self.register_buffer("weight_packed", torch.zeros(out_features, packed_size, dtype=torch.uint8))
        self.register_buffer("scale", torch.zeros(out_features, n_groups, dtype=torch.float32))
        self.register_buffer("zero_point", torch.zeros(out_features, n_groups, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None

    def _unpack_weights(self):
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
        w = w.reshape(self.out_features, -1)[:, :self.in_features]
        return nn.functional.linear(x, w, self.bias.to(dtype) if self.bias is not None else None)


# ── Int4Embedding (for quantized nn.Embedding layers) ──────────────────────

class Int4Embedding(nn.Module):
    """Embedding layer with INT4 weight storage. Dequantizes then indexes."""

    def __init__(self, num_embeddings: int, embedding_dim: int, group_size: int = 128):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.group_size = group_size

        self.embedding_dim_padded = ((embedding_dim + group_size - 1) // group_size) * group_size
        n_groups = self.embedding_dim_padded // group_size
        packed_size = self.embedding_dim_padded // 2

        self.register_buffer("weight_packed", torch.zeros(num_embeddings, packed_size, dtype=torch.uint8))
        self.register_buffer("scale", torch.zeros(num_embeddings, n_groups, dtype=torch.float32))
        self.register_buffer("zero_point", torch.zeros(num_embeddings, n_groups, dtype=torch.float32))

    def _dequantize_weight(self):
        low = self.weight_packed & 0x0F
        high = self.weight_packed >> 4
        unpacked = torch.stack([high, low], dim=-1).reshape(self.num_embeddings, -1)
        unpacked = unpacked[:, :self.embedding_dim_padded]
        unpacked = unpacked.reshape(self.num_embeddings, -1, self.group_size)
        scale = self.scale.unsqueeze(-1)
        zero = self.zero_point.unsqueeze(-1)
        w = (unpacked.float() - zero) * scale
        w = w.reshape(self.num_embeddings, -1)[:, :self.embedding_dim]
        return w

    def forward(self, input_ids):
        w = self._dequantize_weight()
        return nn.functional.embedding(input_ids, w)


# ── Int8Linear (for PYTORCH-INT8) ─────────���────────────────────────────────

class Int8Linear(nn.Module):
    """INT8 linear layer: per-channel symmetric quantization."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_int8", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scale", torch.zeros(out_features, 1, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None

    def forward(self, x):
        dtype = x.dtype
        w = self.weight_int8.to(dtype) * self.scale.to(dtype)
        return nn.functional.linear(x, w, self.bias.to(dtype) if self.bias is not None else None)


# ── HQQ dequantization (manual, no HQQ library needed at inference) ─────────

class HQQDequantLinear(nn.Module):
    """Dequantizes HQQ INT4 weights on-the-fly for CPU inference.

    HQQ stores: W_q (packed uint8), scale, zero, shape, group_size, nbits,
    plus several metadata tensors. We accept all of them as buffers.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 state_dict_subset: dict = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Register all tensors from the HQQ state dict as buffers
        if state_dict_subset:
            for key, tensor in state_dict_subset.items():
                # Convert fp16 to fp32 for CPU
                if tensor.dtype == torch.float16:
                    tensor = tensor.to(torch.float32)
                self.register_buffer(key, tensor)
        else:
            # Minimal defaults (will be overwritten by load_state_dict)
            self.register_buffer("W_q", torch.zeros(1, dtype=torch.uint8))
            self.register_buffer("scale", torch.zeros(1, dtype=torch.float32))
            self.register_buffer("zero", torch.zeros(1, dtype=torch.float32))
            self.register_buffer("shape", torch.tensor([out_features, in_features], dtype=torch.int64))
            self.register_buffer("nbits", torch.tensor(4, dtype=torch.int32))
            self.register_buffer("group_size", torch.tensor(64, dtype=torch.int32))

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None

    def _dequantize(self):
        nbits = self.nbits.item()
        group_size = self.group_size.item()
        out_f, in_f = self.shape[0].item(), self.shape[1].item()

        # Unpack int4 from uint8: each byte has 2 values
        wq = self.W_q.reshape(-1)
        if nbits == 4:
            low = (wq & 0x0F).to(torch.float32)
            high = (wq >> 4).to(torch.float32)
            unpacked = torch.stack([low, high], dim=-1).reshape(-1)
        elif nbits == 8:
            unpacked = wq.to(torch.float32)
        else:
            unpacked = wq.to(torch.float32)

        n_elements = out_f * in_f
        unpacked = unpacked[:n_elements]
        unpacked = unpacked.reshape(-1, group_size)

        scale = self.scale.to(torch.float32).reshape(-1, 1)
        zero = self.zero.to(torch.float32).reshape(-1, 1)

        w = (unpacked - zero) * scale
        return w.reshape(out_f, in_f)

    def forward(self, x):
        dtype = x.dtype
        w = self._dequantize().to(dtype)
        return nn.functional.linear(x, w, self.bias.to(dtype) if self.bias is not None else None)


# ── Model loading functions ─────────────────────────────────────────────────

def _load_base_model():
    """Load base SmolVLM-256M architecture and processor on CPU."""
    from transformers import AutoModelForVision2Seq, AutoProcessor

    logger.info(f"Loading base model architecture: {BASE_MODEL_ID}")
    t0 = time.time()

    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()
    logger.info(f"  Base model loaded in {time.time()-t0:.1f}s, RAM: {_rss_mb():.0f} MB")
    return model, processor


def _get_module_by_name(model, name):
    """Get a submodule by dot-separated name."""
    parts = name.split(".")
    mod = model
    for p in parts:
        mod = getattr(mod, p)
    return mod


def _set_module_by_name(model, name, new_module):
    """Set a submodule by dot-separated name."""
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def load_pytorch_int4(hf_repo: str):
    """Load PYTORCH-INT4 or GPTQ-INT4 (same format) from HuggingFace."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    logger.info(f"Downloading {hf_repo}...")
    repo_path = snapshot_download(hf_repo)

    # Read quant config
    quant_config_path = os.path.join(repo_path, "quant_config.json")
    if os.path.exists(quant_config_path):
        with open(quant_config_path) as f:
            quant_config = json.load(f)
        int4_layers = quant_config.get("int4_layers", {})
    else:
        int4_layers = {}

    model, processor = _load_base_model()

    # Load quantized state dict
    logger.info("Loading quantized safetensors...")
    state_dict = load_file(os.path.join(repo_path, "model.safetensors"))

    # Detect which layers are quantized from state dict keys
    if not int4_layers:
        # Infer from state dict: find all *.weight_packed keys
        for key in state_dict:
            if key.endswith(".weight_packed"):
                layer_name = key.rsplit(".weight_packed", 1)[0]
                wp = state_dict[key]
                scale_key = f"{layer_name}.scale"
                if scale_key in state_dict:
                    out_f = wp.shape[0]
                    n_groups = state_dict[scale_key].shape[1]
                    in_f_padded = wp.shape[1] * 2  # 2 int4 per byte
                    group_size = in_f_padded // n_groups

                    # Get actual in_features from original module (before padding)
                    in_f = in_f_padded
                    try:
                        orig = _get_module_by_name(model, layer_name)
                        if isinstance(orig, nn.Linear):
                            in_f = orig.in_features
                        elif isinstance(orig, nn.Embedding):
                            in_f = orig.embedding_dim
                    except AttributeError:
                        pass

                    int4_layers[layer_name] = {
                        "in_features": in_f,
                        "out_features": out_f,
                        "group_size": group_size,
                        "has_bias": f"{layer_name}.bias" in state_dict,
                    }

    n_linear = 0
    n_embed = 0
    for layer_name, cfg in int4_layers.items():
        # Check if the original module is an Embedding
        try:
            orig_mod = _get_module_by_name(model, layer_name)
        except AttributeError:
            orig_mod = None

        if isinstance(orig_mod, nn.Embedding):
            int4_mod = Int4Embedding(
                orig_mod.num_embeddings, orig_mod.embedding_dim,
                group_size=cfg.get("group_size", 128),
            )
            n_embed += 1
        else:
            int4_mod = Int4Linear(
                cfg["in_features"], cfg["out_features"],
                bias=cfg.get("has_bias", False),
                group_size=cfg.get("group_size", 128),
            )
            n_linear += 1
        _set_module_by_name(model, layer_name, int4_mod)

    logger.info(f"Replaced {n_linear} Linear + {n_embed} Embedding layers with Int4 variants")

    # Load state dict — convert fp16 to fp32 for CPU
    for key in state_dict:
        if state_dict[key].dtype == torch.float16:
            state_dict[key] = state_dict[key].to(torch.float32)

    model.load_state_dict(state_dict, strict=False)
    del state_dict
    gc.collect()

    model.eval()
    return model, processor


def load_pytorch_int8(hf_repo: str):
    """Load PYTORCH-INT8 from HuggingFace."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    logger.info(f"Downloading {hf_repo}...")
    repo_path = snapshot_download(hf_repo)

    # Read quant config
    quant_config_path = os.path.join(repo_path, "quant_config.json")
    if os.path.exists(quant_config_path):
        with open(quant_config_path) as f:
            quant_config = json.load(f)
        int8_layers = quant_config.get("int8_layers", {})
    else:
        int8_layers = {}

    model, processor = _load_base_model()

    logger.info("Loading quantized safetensors...")
    state_dict = load_file(os.path.join(repo_path, "model.safetensors"))

    # Detect quantized layers from state dict
    if not int8_layers:
        for key in state_dict:
            if key.endswith(".weight_int8"):
                layer_name = key.rsplit(".weight_int8", 1)[0]
                w = state_dict[key]
                int8_layers[layer_name] = {
                    "in_features": w.shape[1],
                    "out_features": w.shape[0],
                    "has_bias": f"{layer_name}.bias" in state_dict,
                }

    logger.info(f"Replacing {len(int8_layers)} layers with Int8Linear...")
    for layer_name, cfg in int8_layers.items():
        int8_mod = Int8Linear(
            cfg["in_features"], cfg["out_features"],
            bias=cfg.get("has_bias", False),
        )
        _set_module_by_name(model, layer_name, int8_mod)

    # Convert fp16 to fp32 for CPU
    for key in state_dict:
        if state_dict[key].dtype == torch.float16:
            state_dict[key] = state_dict[key].to(torch.float32)

    model.load_state_dict(state_dict, strict=False)
    del state_dict
    gc.collect()

    model.eval()
    return model, processor


def load_hqq_int4(hf_repo: str):
    """Load HQQ-INT4 from HuggingFace using manual dequantization."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    logger.info(f"Downloading {hf_repo}...")
    repo_path = snapshot_download(hf_repo)

    model, processor = _load_base_model()

    logger.info("Loading HQQ quantized safetensors...")
    state_dict = load_file(os.path.join(repo_path, "model.safetensors"))

    # Find all HQQ-quantized layers (have W_q key) and collect their tensors
    hqq_layers = {}
    hqq_prefixes = set()
    for key in state_dict:
        if key.endswith(".W_q"):
            layer_name = key.rsplit(".W_q", 1)[0]
            hqq_prefixes.add(layer_name)
            shape_key = f"{layer_name}.shape"
            if shape_key in state_dict:
                shape = state_dict[shape_key]
                out_f, in_f = shape[0].item(), shape[1].item()
            else:
                try:
                    orig = _get_module_by_name(model, layer_name)
                    out_f, in_f = orig.out_features, orig.in_features
                except Exception:
                    continue
            hqq_layers[layer_name] = {
                "in_features": in_f,
                "out_features": out_f,
                "has_bias": f"{layer_name}.bias" in state_dict,
            }

    # Collect per-layer tensor subsets
    layer_tensors = {name: {} for name in hqq_layers}
    for key in list(state_dict.keys()):
        for prefix in hqq_prefixes:
            if key.startswith(prefix + "."):
                short_key = key[len(prefix) + 1:]
                layer_tensors[prefix][short_key] = state_dict[key]
                break

    logger.info(f"Replacing {len(hqq_layers)} layers with HQQDequantLinear...")
    for layer_name, cfg in hqq_layers.items():
        hqq_mod = HQQDequantLinear(
            cfg["in_features"], cfg["out_features"],
            bias=cfg.get("has_bias", False),
            state_dict_subset=layer_tensors[layer_name],
        )
        _set_module_by_name(model, layer_name, hqq_mod)

    # Now load non-HQQ weights (vision encoder, layernorms, embeddings, connector)
    non_hqq_keys = {}
    for key in state_dict:
        is_hqq = any(key.startswith(p + ".") for p in hqq_prefixes)
        if not is_hqq:
            t = state_dict[key]
            if t.dtype == torch.float16:
                t = t.to(torch.float32)
            non_hqq_keys[key] = t

    model.load_state_dict(non_hqq_keys, strict=False)
    del state_dict, non_hqq_keys, layer_tensors
    gc.collect()

    model.eval()
    return model, processor


# ── Per-sample profiling (mirrors profiling/profile_all.py) ─────────────────

def profile_single_sample(
    model, processor, sample: dict, family: str, device: str,
    max_new_tokens: int = 30,
    use_components: bool = False,
    hook_manager=None,
) -> dict:
    """Profile a single VQA sample with granular timing."""
    image = sample["image"]
    # Florence-2 uses task tokens; others get short-answer nudge
    if family == "florence2":
        question = sample["question"]
    else:
        question = sample["question"] + " Answer with a single word or short phrase."

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    # 1. Preprocessing timing (family-aware)
    t_pre_start = time.perf_counter()
    fastvlm_extras = {}
    if family == "florence2":
        inputs = processor(
            text=f"<VQA> {question}",
            images=image,
            return_tensors="pt",
        ).to(device)
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float32)
    elif family == "fastvlm":
        # FastVLM (LLaVA-Qwen2): must inject IMAGE_TOKEN_INDEX=-200 into input_ids
        # and use model's vision tower image_processor directly
        IMAGE_TOKEN_INDEX = -200
        messages = [{"role": "user", "content": f"<image>\n{question}"}]
        rendered = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        pre, post = rendered.split("<image>", 1)
        pre_ids = processor.tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids
        post_ids = processor.tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids
        img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
        input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(device)
        attention_mask = torch.ones_like(input_ids, device=device)
        pixel_values = model.get_vision_tower().image_processor(
            images=image, return_tensors="pt"
        )["pixel_values"].to(device, dtype=next(model.parameters()).dtype)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        fastvlm_extras = {"images": pixel_values}
    elif family == "lfm2vl":
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)
    else:  # smolvlm
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)
    t_pre_end = time.perf_counter()
    preprocess_ms = (t_pre_end - t_pre_start) * 1000

    num_input_tokens = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
    mem_before = _rss_mb()

    # 2. Generation with TokenTimingProcessor
    token_timer = TokenTimingProcessor()

    if use_components and hook_manager:
        hook_manager.reset()

    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
    # Florence-2's custom generate() breaks with LogitsProcessor
    if family != "florence2":
        gen_kwargs["logits_processor"] = [token_timer]

    token_timer.record_start()
    with torch.no_grad():
        if family == "fastvlm":
            # FastVLM's custom generate() expects input_ids as first positional arg 'inputs'
            output_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                images=fastvlm_extras.get("images"),
                **gen_kwargs,
            )
        elif family == "florence2":
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                num_beams=1,
                **gen_kwargs,
            )
        else:
            output_ids = model.generate(**inputs, **gen_kwargs)
    token_timer.record_end()
    token_timer.finalize()

    # Decode output
    t_decode_start = time.perf_counter()
    if family == "florence2":
        # Florence-2: use post_process_generation for proper task-aware decoding
        raw_text = processor.batch_decode(output_ids, skip_special_tokens=False)[0]
        try:
            result = processor.post_process_generation(
                raw_text, task="<VQA>",
                image_size=(image.width, image.height),
            )
            pred = result.get("<VQA>", "")
        except Exception:
            pred = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    elif family == "fastvlm":
        # FastVLM: decode full output, take first line
        pred = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        pred = pred.split("\n")[0].strip()
    else:
        input_len = inputs["input_ids"].shape[1]
        pred = processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0]
    t_decode_end = time.perf_counter()
    decode_ms = (t_decode_end - t_decode_start) * 1000

    mem_after = _rss_mb()

    # 3. Component breakdown
    component_data = None
    if use_components and hook_manager:
        hook_manager.tracker.compute_timings()
        component_data = hook_manager.get_category_summary()

    # 4. Multi-metric accuracy
    gold_answers = sample.get("answers", [])
    if gold_answers and pred:
        metrics = _vqa_multi_metric(pred.strip(), gold_answers)
    else:
        metrics = {"exact_match": 0.0, "contains": 0.0, "token_f1": 0.0,
                   "bleu": 0.0, "rouge_l": 0.0}

    # 5. Assemble result
    result = {
        "preprocessing_ms": round(preprocess_ms, 4),
        "num_input_tokens": num_input_tokens,
        "token_timing": token_timer.to_dict(),
        "decode_ms": round(decode_ms, 4),
        "total_ms": round(preprocess_ms + (token_timer.total_ms or token_timer._wall_total_ms) + decode_ms, 4),
        "ram_before_mb": round(mem_before, 1),
        "ram_peak_mb": round(max(mem_before, mem_after), 1),
        "ram_after_mb": round(mem_after, 1),
        "generated_text": pred.strip() if pred else "",
        "accuracy": metrics,
    }

    if component_data:
        result["components"] = {
            cat: {
                "total_ms": round(stats["total_ms"], 2),
                "percentage": round(stats["percentage"], 1),
                "count": stats["count"],
            }
            for cat, stats in component_data.items()
        }

    return result


# ── Detailed vision sub-component categorizer ──────────────────────────────

def _make_detailed_categorizer():
    """Create a ModuleCategorizer with vision encoder broken into sub-components.

    The default ModuleCategorizer lumps everything under vision_model.* into
    a single 'vision_encoder' bucket.  This version inserts fine-grained
    vision sub-categories (vision_attention, vision_mlp, vision_norm,
    vision_embeddings) *before* the broad vision_encoder catch-all so they
    match first (categorize() returns the first match).
    """
    from profiling.hooks import ModuleCategorizer
    from collections import OrderedDict

    cat = ModuleCategorizer()

    # Build new ordered patterns: vision sub-categories first, then everything else
    new_patterns = OrderedDict()

    # Fine-grained vision sub-components (checked before broad vision_encoder)
    new_patterns["vision_attention"] = [
        r"vision_model\.encoder\.layers\.\d+\.self_attn\.",
        r"vision_model\..*\.self_attn\.",
    ]
    new_patterns["vision_mlp"] = [
        r"vision_model\.encoder\.layers\.\d+\.mlp\.",
        r"vision_model\..*\.mlp\.",
    ]
    new_patterns["vision_norm"] = [
        r"vision_model\.encoder\.layers\.\d+\.layer_norm",
        r"vision_model\..*layer_norm",
        r"vision_model\..*\.norm",
        r"vision_model\.post_layernorm",
        r"vision_model\.layernorm",
    ]
    new_patterns["vision_embeddings"] = [
        r"vision_model\.embeddings\.",
    ]

    # Now copy original patterns, keeping vision_encoder as catch-all for
    # anything inside vision_model that didn't match a sub-category above
    for key, pats in cat.patterns.items():
        if key not in new_patterns:
            new_patterns[key] = pats

    cat.patterns = new_patterns
    return cat


# ── Main profiling loop ─────────────────────────────────────────────────────

def profile_model(
    model, processor, method: str,
    samples: list,
    family: str = "smolvlm",
    num_warmup: int = 1,
    max_tokens: int = 30,
    use_components: bool = False,
) -> dict:
    """Profile a model on VQA samples with full instrumentation."""
    # Setup component hooks
    hook_manager = None
    if use_components:
        from profiling.hooks import HookManager, TimingTracker, ModuleCategorizer
        categorizer = _make_detailed_categorizer()
        tracker = TimingTracker()
        hook_manager = HookManager(categorizer, tracker)
        hook_manager.register_hooks(model)

    # Warmup
    if num_warmup > 0:
        logger.info(f"Warming up ({num_warmup} runs)...")
        if use_components and hook_manager:
            hook_manager.disable()
        for i in range(num_warmup):
            try:
                run_inference(model, processor, samples[i % len(samples)], family, "cpu", max_tokens)
            except Exception:
                pass
        if use_components and hook_manager:
            hook_manager.enable()

    # Profile each sample
    sample_results = []
    cpu_profiler = CPUProfiler(poll_interval_ms=100)

    logger.info(f"Profiling {len(samples)} samples...")
    with cpu_profiler:
        for sample in tqdm(samples, desc=f"{method}"):
            try:
                result = profile_single_sample(
                    model, processor, sample, family, "cpu",
                    max_new_tokens=max_tokens,
                    use_components=use_components,
                    hook_manager=hook_manager,
                )
                sample_results.append(result)
            except Exception as e:
                logger.warning(f"Sample failed: {e}")

    cpu_stats = cpu_profiler.stats()

    if hook_manager:
        hook_manager.remove_hooks()

    if not sample_results:
        logger.error("All samples failed!")
        return {"error": "all_samples_failed"}

    n = len(sample_results)

    # Per-token timing aggregation
    all_prefill = [r["token_timing"]["prefill_ms"] for r in sample_results if r["token_timing"].get("prefill_ms")]
    all_decode = [r["token_timing"]["decode_ms"] for r in sample_results if r["token_timing"].get("decode_ms")]
    all_total = [r["token_timing"]["total_ms"] for r in sample_results if r["token_timing"].get("total_ms")]
    all_tokens = [r["token_timing"]["num_tokens"] for r in sample_results if r["token_timing"].get("num_tokens")]

    avg_prefill = sum(all_prefill) / len(all_prefill) if all_prefill else 0.0
    avg_decode = sum(all_decode) / len(all_decode) if all_decode else 0.0
    avg_total_gen = sum(all_total) / len(all_total) if all_total else 0.0
    avg_tokens = sum(all_tokens) / len(all_tokens) if all_tokens else 0.0

    # Decode throughput
    decode_throughputs = []
    for r in sample_results:
        tt = r["token_timing"]
        n_decode = tt.get("num_tokens", 0) - 1
        decode_ms = tt.get("decode_ms", 0)
        if n_decode > 0 and decode_ms > 0:
            decode_throughputs.append(n_decode / (decode_ms / 1000.0))

    # Component aggregation
    component_agg = {}
    if use_components:
        for r in sample_results:
            for cat, stats in r.get("components", {}).items():
                if cat not in component_agg:
                    component_agg[cat] = {"total_ms": 0, "count": 0, "samples": 0}
                component_agg[cat]["total_ms"] += stats["total_ms"]
                component_agg[cat]["count"] += stats["count"]
                component_agg[cat]["samples"] += 1
        total_comp_ms = sum(c["total_ms"] for c in component_agg.values())
        for cat in component_agg:
            c = component_agg[cat]
            c["avg_ms"] = round(c["total_ms"] / c["samples"], 2)
            c["percentage"] = round((c["total_ms"] / total_comp_ms) * 100, 1) if total_comp_ms > 0 else 0.0
            c["total_ms"] = round(c["total_ms"], 2)

    # Accuracy aggregation
    metric_names = ["exact_match", "contains", "token_f1", "bleu", "rouge_l"]
    accuracy_agg = {}
    for name in metric_names:
        vals = [r["accuracy"][name] for r in sample_results if "accuracy" in r]
        accuracy_agg[name] = round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "num_samples": n,
        "num_warmup": num_warmup,
        "max_tokens": max_tokens,
        "accuracy": accuracy_agg,
        "timing": {
            "avg_preprocessing_ms": round(sum(r["preprocessing_ms"] for r in sample_results) / n, 2),
            "avg_prefill_ms": round(avg_prefill, 2),
            "avg_decode_ms": round(avg_decode, 2),
            "avg_total_generation_ms": round(avg_total_gen, 2),
            "avg_decode_detokenize_ms": round(sum(r["decode_ms"] for r in sample_results) / n, 2),
            "avg_total_ms": round(sum(r["total_ms"] for r in sample_results) / n, 2),
            "avg_tokens_generated": round(avg_tokens, 1),
            "avg_input_tokens": round(sum(r["num_input_tokens"] for r in sample_results) / n, 1),
        },
        "throughput": {
            "avg_tok_s": round(avg_tokens / (avg_total_gen / 1000.0), 2) if avg_total_gen > 0 else 0.0,
            "avg_decode_tok_s": round(sum(decode_throughputs) / len(decode_throughputs), 2) if decode_throughputs else 0.0,
            "samples_per_s": round(n / cpu_stats.wall_time_s, 3) if cpu_stats.wall_time_s > 0 else 0.0,
        },
        "memory": {
            "ram_peak_mb": round(cpu_stats.peak_memory_mb, 1),
            "ram_avg_mb": round(cpu_stats.avg_memory_mb, 1),
            "avg_inference_peak_mb": round(sum(r["ram_peak_mb"] for r in sample_results) / n, 1),
        },
        "components": component_agg if component_agg else None,
        "sample_details": sample_results[:3],
    }


# ── Main ────────────────────────────────────────────────────────────────────

def run_single_method(method: str, vqav2_n: int, force: bool = False,
                      num_warmup: int = 1, max_tokens: int = 30,
                      use_components: bool = False,
                      model_id: str = None, family: str = None):
    """Load a model (FP32 or quantized), profile it, and save results."""
    # Use provided model config or fall back to globals
    _model_id = model_id or BASE_MODEL_ID
    _family = family or FAMILY
    _methods = METHODS

    # For FP32-BASELINE, the hf_repo IS the model_id
    if method == "FP32-BASELINE":
        hf_repo = _model_id
    else:
        hf_repo = _methods[method]

    safe_name = hf_repo.replace("/", "__")
    out_path = RESULTS_DIR / f"{safe_name}.json"

    if out_path.exists() and not force:
        logger.info(f"Result exists: {out_path}. Use --force to overwrite.")
        return

    mem_before = _rss_mb()
    t0 = time.time()

    logger.info(f"\n{'='*60}")
    logger.info(f"Model: {_model_id} | Method: {method} | Family: {_family}")
    logger.info(f"HF repo: {hf_repo}")
    logger.info(f"RAM before: {mem_before:.0f} MB")
    logger.info(f"{'='*60}")

    # Load model
    if method == "FP32-BASELINE":
        from transformers import AutoModelForVision2Seq, AutoModelForImageTextToText, AutoModelForCausalLM, AutoProcessor
        logger.info(f"Loading FP32 baseline: {_model_id}")
        if _family == "fastvlm":
            # FastVLM is LLaVA-Qwen2 with MobileCLIP vision tower
            from transformers import AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(
                _model_id, torch_dtype=torch.float32, trust_remote_code=True
            ).eval()
            tokenizer = AutoTokenizer.from_pretrained(_model_id, trust_remote_code=True)
            vt = model.get_vision_tower()
            image_processor = vt.image_processor
            # Import FastVLM processor wrapper from model_loader
            from models.model_loader import _FastVLMProcessor
            processor = _FastVLMProcessor(image_processor, tokenizer)
        elif _family == "florence2":
            # Florence-2 uses AutoModelForCausalLM + trust_remote_code + eager attn
            model = AutoModelForCausalLM.from_pretrained(
                _model_id, torch_dtype=torch.float32,
                trust_remote_code=True, attn_implementation="eager"
            ).eval()
            processor = AutoProcessor.from_pretrained(_model_id, trust_remote_code=True)
        elif _family == "lfm2vl":
            model = AutoModelForImageTextToText.from_pretrained(
                _model_id, torch_dtype=torch.float32
            ).eval()
            processor = AutoProcessor.from_pretrained(_model_id)
        else:
            model = AutoModelForVision2Seq.from_pretrained(
                _model_id, torch_dtype=torch.float32
            ).eval()
            processor = AutoProcessor.from_pretrained(_model_id)
    elif method in ("PYTORCH-INT4", "GPTQ-INT4"):
        model, processor = load_pytorch_int4(hf_repo)
    elif method == "PYTORCH-INT8":
        model, processor = load_pytorch_int8(hf_repo)
    elif method == "HQQ-INT4":
        model, processor = load_hqq_int4(hf_repo)
    else:
        raise ValueError(f"Unknown method: {method}")

    load_time = time.time() - t0
    mem_after = _rss_mb()

    logger.info(f"Model loaded in {load_time:.1f}s")
    logger.info(f"RAM after load: {mem_after:.0f} MB (delta: {mem_after-mem_before:.0f} MB)")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    for name, buf in model.named_buffers():
        if "weight_packed" in name or "W_q" in name:
            n_params += buf.numel() * 2
        elif "weight_int8" in name:
            n_params += buf.numel()

    # Load VQAv2 samples
    logger.info(f"Loading VQAv2 ({vqav2_n} samples)...")
    samples = load_vqav2(vqav2_n)

    # Profile
    profiling_result = profile_model(
        model, processor, method, samples,
        family=_family,
        num_warmup=num_warmup,
        max_tokens=max_tokens,
        use_components=use_components,
    )

    # Assemble final result (same format as Jetson profiling/profile_all.py)
    result = {
        "model_id": _model_id,
        "method": method,
        "family": _family,
        "hf_repo": hf_repo,
        "device": "rpi5_cpu",
        "num_params_M": round(n_params / 1e6, 1),
        "load_time_s": round(load_time, 1),
        "ram_before_mb": round(mem_before, 1),
        "ram_after_mb": round(mem_after, 1),
        "ram_delta_mb": round(mem_after - mem_before, 1),
    }
    result.update(profiling_result)

    # Save
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    # Cleanup
    del model
    gc.collect()
    logger.info(f"RAM after cleanup: {_rss_mb():.0f} MB")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Profile VLMs on RPi5 (FP32 baseline or pre-quantized from HuggingFace)"
    )
    parser.add_argument(
        "--method", required=True,
        choices=list(METHODS.keys()) + ["all"],
        help="Quantization method to evaluate (or 'all')",
    )
    parser.add_argument(
        "--model", default=None,
        choices=list(MODELS.keys()) + ["all-fp32"],
        help="Model to profile. Use 'all-fp32' to run FP32 baseline on all sub-500M models.",
    )
    parser.add_argument("--vqav2_n", type=int, default=50)
    parser.add_argument("--num_warmup", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=30)
    parser.add_argument("--components", action="store_true",
                        help="Enable component-level hook profiling")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # Build list of (model_key, method) pairs to run
    runs = []
    if args.model == "all-fp32":
        for model_key in MODELS:
            runs.append((model_key, "FP32-BASELINE"))
    elif args.model and args.model in MODELS:
        methods = list(MODELS[args.model]["methods"].keys()) if args.method == "all" else [args.method]
        for m in methods:
            runs.append((args.model, m))
    else:
        # Legacy: no --model specified, use SmolVLM-256M
        methods = list(METHODS.keys()) if args.method == "all" else [args.method]
        for m in methods:
            runs.append(("SmolVLM-256M", m))

    all_results = {}
    for model_key, method in runs:
        mcfg = MODELS[model_key]
        try:
            result = run_single_method(
                method, args.vqav2_n, args.force,
                num_warmup=args.num_warmup,
                max_tokens=args.max_tokens,
                use_components=args.components,
                model_id=mcfg["model_id"],
                family=mcfg["family"],
            )
            if result:
                all_results[f"{model_key}/{method}"] = result
        except Exception as e:
            logger.error(f"FAILED: {model_key}/{method}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    if all_results:
        print(f"\n{'='*80}")
        print(f"{'Method':<15} {'ExactMatch':>10} {'Prefill':>10} {'Decode':>10} {'Total':>10} {'Tput':>10} {'RAM':>8}")
        print(f"{'='*80}")
        for method, r in all_results.items():
            acc = r.get("accuracy", {}).get("exact_match", 0)
            t = r.get("timing", {})
            tp = r.get("throughput", {})
            mem = r.get("memory", {})
            print(f"{method:<15} {acc:>10.4f} {t.get('avg_prefill_ms',0):>8.0f}ms "
                  f"{t.get('avg_decode_ms',0):>8.0f}ms {t.get('avg_total_ms',0):>8.0f}ms "
                  f"{tp.get('avg_tok_s',0):>7.2f}t/s {mem.get('ram_peak_mb',0):>6.0f}MB")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
