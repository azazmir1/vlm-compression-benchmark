"""
Quantize VLMs with GPTQ INT4 and upload to HuggingFace Hub.

Usage:
  python scripts/quantize_gptq_upload.py --model_id Qwen/Qwen2.5-VL-3B-Instruct
  python scripts/quantize_gptq_upload.py --all   # run all models >= 1B
"""

import argparse
import gc
import logging
import shutil
from pathlib import Path

import torch
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoProcessor, AutoTokenizer
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

# Models >= 1B params (sorted by size)
# Excluded: Qwen 3B & 7B (already quantized), InternVL (user skip)
MODELS = [
    "AIDC-AI/Ovis2-1B",
    "LiquidAI/LFM2-VL-1.6B",
    "vikhyatk/moondream2",
    "AIDC-AI/Ovis2-2B",
    "HuggingFaceTB/SmolVLM-Instruct",
    "LiquidAI/LFM2-VL-3B",
    "google/gemma-3-4b-it",
    "AIDC-AI/Ovis2-4B",
    "AIDC-AI/Ovis2-8B",
    "google/gemma-3-12b-it",
]


def make_calib_conversations(n_calib: int):
    """Create calibration data in conversation format for VLMs."""
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "A visual question answering model processes images and text.",
        "Machine learning models can be compressed using quantization.",
        "The image shows a red car parked on a street.",
        "What color is the sky in this photograph?",
        "Describe the objects visible in the image.",
        "How many people are in the picture?",
        "The model architecture consists of a vision encoder and language decoder.",
        "Natural language processing has advanced significantly.",
        "The neural network weights can be pruned for efficiency.",
        "Transfer learning allows reuse of pretrained models.",
        "Attention mechanisms capture long-range dependencies in sequences.",
        "Computer vision models detect objects in photographs.",
        "The transformer architecture uses self-attention layers.",
        "Quantization reduces model size with minimal accuracy loss.",
        "Deep learning requires large amounts of training data.",
    ]
    convos = []
    for i in range(n_calib):
        text = prompts[i % len(prompts)]
        convos.append([
            {"role": "user", "content": [{"type": "text", "text": text}]},
            {"role": "assistant", "content": [{"type": "text", "text": "OK."}]},
        ])
    return convos


def make_calib_tokenized(tokenizer, n_calib: int):
    """Tokenized calibration data using chat template when available."""
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "A visual question answering model processes images and text.",
        "Machine learning models can be compressed using quantization.",
        "What color is the sky in this photograph?",
        "Describe the objects visible in the image.",
        "The transformer architecture uses self-attention layers.",
        "Quantization reduces model size with minimal accuracy loss.",
        "Deep learning requires large amounts of training data.",
    ]
    texts = (prompts * (n_calib // len(prompts) + 1))[:n_calib]
    result = []
    for t in texts:
        try:
            # Try chat template first
            messages = [{"role": "user", "content": t}, {"role": "assistant", "content": "OK."}]
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
            result.append({"input_ids": ids})
        except Exception:
            # Plain tokenization fallback
            result.append({"input_ids": tokenizer(t, return_tensors="pt")["input_ids"].squeeze().tolist()})
    return result


def make_repo_name(model_id: str) -> str:
    short = model_id.split("/")[-1]
    return f"{short}-GPTQ-Int4"


def quantize_and_upload(model_id: str, n_calib: int = 256):
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
    logger.info(f"Quantizing: {model_id}")
    logger.info(f"Target repo: {full_repo}")
    logger.info(f"{'='*60}")

    quant_config = QuantizeConfig(bits=4, group_size=128)

    # Load model with tokenized calibration (more reliable across model families)
    logger.info(f"Loading {model_id} for GPTQ quantization...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    try:
        model = GPTQModel.load(
            model_id,
            quant_config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    except Exception as e:
        logger.error(f"FAILED to load {model_id}: {e}")
        return None

    logger.info(f"Preparing calibration data ({n_calib} samples)...")

    # Use conversation format for models that need it (e.g. Qwen2VL), tokenized otherwise
    calib_conv = make_calib_conversations(n_calib)
    calib_tok = make_calib_tokenized(tokenizer, n_calib)

    logger.info("Running GPTQ quantization...")
    try:
        model.quantize(calib_conv, batch_size=1)
    except Exception as e1:
        logger.warning(f"Conversation format failed ({e1}), trying tokenized format...")
        del model
        gc.collect()
        torch.cuda.empty_cache()

        try:
            model = GPTQModel.load(
                model_id, quant_config, torch_dtype=torch.float16, trust_remote_code=True,
            )
            model.quantize(calib_tok, batch_size=1, tokenizer=tokenizer)
        except Exception as e2:
            logger.error(f"FAILED to quantize {model_id}: {e2}")
            try:
                del model
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            return None

    # Save locally
    logger.info(f"Saving quantized model to {save_dir}...")
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save(save_dir)

    # Also save the processor/tokenizer
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        processor.save_pretrained(save_dir)
    except Exception:
        try:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            tok.save_pretrained(save_dir)
        except Exception:
            pass

    # Unload
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Upload to HuggingFace
    logger.info(f"Uploading to {full_repo}...")
    try:
        api.create_repo(full_repo, exist_ok=True)
        api.upload_folder(
            folder_path=str(save_dir),
            repo_id=full_repo,
            commit_message=f"GPTQ INT4 quantization of {model_id}",
        )
        logger.info(f"SUCCESS: https://huggingface.co/{full_repo}")
    except Exception as e:
        logger.error(f"FAILED to upload {model_id}: {e}")
        return None

    # Clean up local files to save disk space
    shutil.rmtree(save_dir, ignore_errors=True)

    return full_repo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, help="Single model to quantize")
    parser.add_argument("--all", action="store_true", help="Quantize all models >= 1B")
    parser.add_argument("--n_calib", type=int, default=256)
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
            repo = quantize_and_upload(model_id, args.n_calib)
        except Exception as e:
            logger.error(f"UNEXPECTED ERROR for {model_id}: {e}")
            repo = None
        results[model_id] = repo
        logger.info("")

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for model_id, repo in results.items():
        status = f"https://huggingface.co/{repo}" if repo else "FAILED"
        logger.info(f"  {model_id}: {status}")


if __name__ == "__main__":
    main()
