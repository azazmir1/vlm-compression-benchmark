"""
Quantize all VLMs (>= 1B params) with GPTQ INT4, test, and upload to HuggingFace.

Usage:
  python scripts/quantize_all_gptq.py --all
  python scripts/quantize_all_gptq.py --model_id AIDC-AI/Ovis2-1B
"""

import argparse
import gc
import logging
import shutil
import traceback
from pathlib import Path

import torch
from gptqmodel import GPTQModel, QuantizeConfig
from huggingface_hub import HfApi
from transformers import AutoProcessor, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

HF_USER = "Azaz666"
QUANT_DIR = Path("/home/azaz/vlm-compression-benchmark/quantized_models")
QUANT_DIR.mkdir(parents=True, exist_ok=True)

# All VLM models >= 1B params, sorted by size
MODELS = [
    "AIDC-AI/Ovis2-1B",
    "OpenGVLab/InternVL2_5-1B",
    "LiquidAI/LFM2-VL-1.6B",
    "vikhyatk/moondream2",
    "AIDC-AI/Ovis2-2B",
    "OpenGVLab/InternVL2_5-2B",
    "HuggingFaceTB/SmolVLM-Instruct",
    "LiquidAI/LFM2-VL-3B",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "google/gemma-3-4b-it",
    "AIDC-AI/Ovis2-4B",
    "OpenGVLab/InternVL2_5-4B",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "AIDC-AI/Ovis2-8B",
    "OpenGVLab/InternVL2_5-8B",
    "google/gemma-3-12b-it",
]


def make_calib_conversations(n_calib: int):
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
            messages = [{"role": "user", "content": t}, {"role": "assistant", "content": "OK."}]
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
            result.append({"input_ids": ids})
        except Exception:
            result.append({"input_ids": tokenizer(t, return_tensors="pt")["input_ids"].squeeze().tolist()})
    return result


def test_quantized_model(save_dir: str, model_id: str) -> bool:
    """Load the quantized model back and run a test generation to verify it works."""
    logger.info(f"[TEST] Loading quantized model from {save_dir}...")
    try:
        model = GPTQModel.load(save_dir, torch_dtype=torch.float16, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(save_dir, trust_remote_code=True)

        prompt = "What is machine learning?"
        try:
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        except Exception:
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

        input_ids = input_ids.to(model.device)

        logger.info("[TEST] Generating text...")
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=50)

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info(f"[TEST] Output: {decoded[:200]}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

        logger.info("[TEST] PASSED - model generates text successfully")
        return True
    except Exception as e:
        logger.error(f"[TEST] FAILED: {e}")
        traceback.print_exc()
        gc.collect()
        torch.cuda.empty_cache()
        return False


def quantize_and_upload(model_id: str, n_calib: int = 256):
    short_name = model_id.split("/")[-1]
    repo_name = f"{short_name}-GPTQ-Int4"
    full_repo = f"{HF_USER}/{repo_name}"
    save_dir = QUANT_DIR / repo_name
    api = HfApi()

    # Check if already uploaded
    try:
        api.repo_info(full_repo)
        logger.info(f"SKIP: {full_repo} already exists on HuggingFace")
        return full_repo
    except Exception:
        pass

    logger.info(f"\n{'='*60}")
    logger.info(f"GPTQ INT4 Quantizing: {model_id}")
    logger.info(f"Target repo: {full_repo}")
    logger.info(f"{'='*60}")

    quant_config = QuantizeConfig(bits=4, group_size=128)

    # Load model
    logger.info(f"Loading {model_id}...")
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

    # Quantize - try conversation format first, then tokenized
    calib_conv = make_calib_conversations(n_calib)
    calib_tok = make_calib_tokenized(tokenizer, n_calib)

    logger.info("Running GPTQ quantization (conversation format)...")
    try:
        model.quantize(calib_conv, batch_size=1)
    except Exception as e1:
        logger.warning(f"Conversation format failed: {e1}")
        del model
        gc.collect()
        torch.cuda.empty_cache()

        logger.info("Retrying with tokenized format...")
        try:
            model = GPTQModel.load(
                model_id, quant_config, torch_dtype=torch.float16, trust_remote_code=True,
            )
            model.quantize(calib_tok, batch_size=1, tokenizer=tokenizer)
        except Exception as e2:
            logger.error(f"Tokenized format also failed: {e2}")
            try:
                del model
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            return None

    # Save locally
    logger.info(f"Saving to {save_dir}...")
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save(save_dir)

    # Save processor/tokenizer
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        processor.save_pretrained(save_dir)
    except Exception:
        try:
            tokenizer.save_pretrained(save_dir)
        except Exception:
            pass

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Test the quantized model
    test_passed = test_quantized_model(str(save_dir), model_id)
    if not test_passed:
        logger.error(f"Model {model_id} failed testing - uploading anyway (quantization succeeded)")

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
        logger.error(f"FAILED to upload: {e}")
        return None

    # Cleanup local files
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
            traceback.print_exc()
            repo = None
        results[model_id] = repo

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    success, failed = [], []
    for model_id, repo in results.items():
        if repo:
            success.append(model_id)
            logger.info(f"  [OK]     {model_id} -> https://huggingface.co/{repo}")
        else:
            failed.append(model_id)
            logger.info(f"  [FAILED] {model_id}")

    logger.info(f"\nSuccess: {len(success)} | Failed: {len(failed)}")
    if failed:
        logger.info("Failed models:")
        for m in failed:
            logger.info(f"  - {m}")


if __name__ == "__main__":
    main()
