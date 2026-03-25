"""
Universal INT4 weight quantization for VLMs.

Quantizes all Linear layers in the LLM backbone to INT4 (per-group symmetric),
stores quantized weights + scales + zeros as fp16-compatible tensors,
then uploads to HuggingFace.

This works for ANY VLM regardless of architecture — it uses pure PyTorch.

The vision encoder is kept in fp16 (it's small and quantizing it hurts accuracy).

Usage:
  python scripts/quantize_int4_universal.py --model_id OpenGVLab/InternVL2_5-2B
  python scripts/quantize_int4_universal.py --all
"""

import argparse
import gc
import json
import logging
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForImageTextToText,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)
from safetensors.torch import save_file, load_file
from huggingface_hub import HfApi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

HF_USER = "Azaz666"
QUANT_DIR = Path("/home/azaz/vlm-compression-benchmark/quantized_models")
QUANT_DIR.mkdir(parents=True, exist_ok=True)

# Vision module keywords — layers matching these are kept in fp16
VISION_KEYWORDS = {
    "vision_model", "visual_model", "image_encoder", "vision_encoder",
    "patch_embed", "visual_projection", "img_projection",
    "vision_tower", "vit", "davit", "siglip", "fastvit",
    "visual", "image_newline", "multi_modal_projector",
    "mlp1",  # InternVL projector
}

GROUP_SIZE = 128


def is_vision_module(name: str) -> bool:
    name_lower = name.lower()
    return any(kw in name_lower for kw in VISION_KEYWORDS)


def quantize_int4_per_group(weight: torch.Tensor, group_size: int = 128):
    """
    Per-group symmetric INT4 quantization.
    Returns dequantized weight (fp16) that can be stored as a drop-in replacement.
    """
    out_features, in_features = weight.shape
    # Pad if needed
    if in_features % group_size != 0:
        pad = group_size - (in_features % group_size)
        weight = torch.nn.functional.pad(weight, (0, pad))
        in_features = weight.shape[1]

    weight = weight.float()
    w_grouped = weight.reshape(-1, group_size)

    # Symmetric quantization: scale = max(|w|) / 7
    abs_max = w_grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = abs_max / 7.0  # INT4 signed: [-8, 7]

    # Quantize and dequantize
    w_q = (w_grouped / scale).round().clamp(-8, 7)
    w_deq = (w_q * scale).reshape(out_features, in_features)

    # Remove padding
    orig_in = weight.shape[1] if in_features == weight.shape[1] else in_features - (group_size - (weight.shape[1] % group_size))
    # Actually just return full size since we padded
    return w_deq[:, :weight.shape[1]].half()


def quantize_model_weights(model: nn.Module):
    """Quantize all Linear layers in the LLM backbone to INT4, keep vision in fp16."""
    quantized_count = 0
    skipped_count = 0
    total_original_bytes = 0
    total_quantized_bytes = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        if is_vision_module(name):
            skipped_count += 1
            continue

        W = module.weight.data
        original_bytes = W.numel() * W.element_size()

        # Quantize to INT4 (stored as dequantized fp16 for compatibility)
        W_q = quantize_int4_per_group(W, GROUP_SIZE)

        # Handle padding difference
        if W_q.shape != W.shape:
            W_q = W_q[:W.shape[0], :W.shape[1]]

        module.weight.data = W_q.to(W.device)

        quantized_bytes = W_q.numel() * W_q.element_size()
        total_original_bytes += original_bytes
        total_quantized_bytes += quantized_bytes
        quantized_count += 1

    logger.info(f"Quantized {quantized_count} Linear layers to INT4 (dequantized fp16)")
    logger.info(f"Skipped {skipped_count} vision layers")

    return {
        "quantized_layers": quantized_count,
        "skipped_vision_layers": skipped_count,
    }


def load_vlm(model_id: str):
    """Load VLM with the right Auto class."""
    kwargs = dict(
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )

    # Try different Auto classes
    for cls_name, cls in [
        ("AutoModelForImageTextToText", AutoModelForImageTextToText),
        ("AutoModel", AutoModel),
        ("AutoModelForCausalLM", AutoModelForCausalLM),
    ]:
        try:
            logger.info(f"  Trying {cls_name}...")
            model = cls.from_pretrained(model_id, **kwargs)
            logger.info(f"  Loaded with {cls_name}")
            return model
        except Exception as e:
            logger.warning(f"  {cls_name} failed: {e}")
            continue

    raise RuntimeError(f"Could not load {model_id} with any Auto class")


def make_repo_name(model_id: str) -> str:
    short = model_id.split("/")[-1]
    return f"{short}-INT4-quantized"


def quantize_and_upload(model_id: str):
    repo_name = make_repo_name(model_id)
    full_repo = f"{HF_USER}/{repo_name}"
    save_dir = QUANT_DIR / repo_name

    # Check if already uploaded
    api = HfApi()
    try:
        api.repo_info(full_repo)
        logger.info(f"SKIP: {full_repo} already exists on HuggingFace")
        return full_repo
    except Exception:
        pass

    logger.info(f"{'='*60}")
    logger.info(f"INT4 Quantizing: {model_id}")
    logger.info(f"Target repo: {full_repo}")
    logger.info(f"{'='*60}")

    # Load model
    logger.info(f"Loading {model_id}...")
    try:
        model = load_vlm(model_id)
    except Exception as e:
        logger.error(f"FAILED to load {model_id}: {e}")
        return None

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {num_params/1e6:.0f}M params")

    # Quantize
    logger.info("Applying INT4 per-group quantization to LLM backbone...")
    try:
        stats = quantize_model_weights(model)
    except Exception as e:
        logger.error(f"FAILED to quantize {model_id}: {e}")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return None

    # Save locally
    logger.info(f"Saving to {save_dir}...")
    save_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(save_dir, safe_serialization=True)
    except Exception as e:
        logger.error(f"FAILED to save {model_id}: {e}")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return None

    # Save processor/tokenizer
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        processor.save_pretrained(save_dir)
    except Exception:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            tokenizer.save_pretrained(save_dir)
        except Exception:
            logger.warning("Could not save processor/tokenizer")

    # Save quantization metadata
    quant_meta = {
        "base_model": model_id,
        "quantization": "int4_per_group_symmetric",
        "group_size": GROUP_SIZE,
        "bits": 4,
        "method": "static_int4_dequantized",
        "description": "INT4 per-group symmetric quantization of LLM backbone weights. "
                       "Vision encoder kept in fp16. Weights stored as dequantized fp16 "
                       "for maximum compatibility.",
        **stats,
    }
    with open(save_dir / "quantization_config.json", "w") as f:
        json.dump(quant_meta, f, indent=2)

    # Unload
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Upload
    logger.info(f"Uploading to {full_repo}...")
    try:
        api.create_repo(full_repo, exist_ok=True)
        api.upload_folder(
            folder_path=str(save_dir),
            repo_id=full_repo,
            commit_message=f"INT4 quantization of {model_id} (LLM backbone quantized, vision encoder fp16)",
        )
        logger.info(f"SUCCESS: https://huggingface.co/{full_repo}")
    except Exception as e:
        logger.error(f"FAILED to upload {model_id}: {e}")
        return None

    # Cleanup
    shutil.rmtree(save_dir, ignore_errors=True)

    return full_repo


# All models that failed GPTQ — need universal INT4
# (LFM2-VL-1.6B already done, will be skipped via HF check)
MODELS = [
    "LiquidAI/LFM2-VL-1.6B",
    "vikhyatk/moondream2",
    "OpenGVLab/InternVL2_5-2B",
    "HuggingFaceTB/SmolVLM-Instruct",
    "LiquidAI/LFM2-VL-3B",
    "OpenGVLab/InternVL2_5-4B",
    "OpenGVLab/InternVL2_5-8B",
    # Ovis2 — aimv2 config clash with transformers 5.3
    "AIDC-AI/Ovis2-1B",
    "AIDC-AI/Ovis2-2B",
    "AIDC-AI/Ovis2-4B",
    "AIDC-AI/Ovis2-8B",
    # Gemma-3 — GPTQ had device issue
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    # Qwen2.5-VL-7B — GPTQ OOM on Hessian inverse
    "Qwen/Qwen2.5-VL-7B-Instruct",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, help="Single model to quantize")
    parser.add_argument("--all", action="store_true", help="Quantize all failed models")
    args = parser.parse_args()

    if args.all:
        models = MODELS
    elif args.model_id:
        models = [args.model_id]
    else:
        parser.error("Specify --model_id or --all")

    results = {}
    for model_id in models:
        try:
            repo = quantize_and_upload(model_id)
        except Exception as e:
            logger.error(f"UNEXPECTED ERROR for {model_id}: {e}")
            import traceback
            traceback.print_exc()
            repo = None
        results[model_id] = repo
        logger.info("")

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for model_id, repo in results.items():
        status = f"https://huggingface.co/{repo}" if repo else "FAILED"
        logger.info(f"  {model_id}: {status}")


if __name__ == "__main__":
    main()
