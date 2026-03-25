"""
compression/ptq/run_awq.py
===========================
AWQ (Activation-Aware Weight Quantization) pipeline for VLMs.

Based on: "AWQ: Activation-Aware Weight Quantization for LLM Compression
           and Acceleration" (Lin et al., MLSys 2024 — Best Paper)

Key idea: Protects salient weights by observing activation magnitudes.
Per-channel scaling reduces quantization error. INT4/INT3 with minimal
accuracy loss.

Requires: pip install autoawq

Usage:
  python compression/ptq/run_awq.py --model_id Qwen/Qwen2.5-VL-3B-Instruct
  python compression/ptq/run_awq.py --model_id OpenGVLab/InternVL2_5-1B --w_bit 4
"""

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.model_loader import load_model, unload_model, detect_family
from evaluation.run_baseline import (
    load_vqav2, run_inference,
    evaluate_dataset, _vqa_accuracy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "awq"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Families where AWQ can be applied via AutoAWQ's causal LM interface.
# We quantize the LLM backbone only — vision encoder stays in fp16.
AWQ_SUPPORTED_FAMILIES = {
    "qwen25vl", "internvl25", "gemma3", "smolvlm", "ovis2", "moondream",
}


def quantize_awq(model_id: str, family: str, save_dir: Path,
                 w_bit: int = 4, q_group_size: int = 128,
                 n_calib: int = 128) -> Path:
    """Quantize with AutoAWQ and save locally. Returns saved path."""
    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        raise ImportError(
            "AutoAWQ not installed. Install with: pip install autoawq"
        )
    from transformers import AutoTokenizer, AutoProcessor

    logger.info(f"[AWQ] Quantizing {model_id} (w_bit={w_bit}, group_size={q_group_size})...")

    # Load model for quantization
    model = AutoAWQForCausalLM.from_pretrained(
        model_id, safetensors=True, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    quant_config = {
        "zero_point": True,
        "q_group_size": q_group_size,
        "w_bit": w_bit,
        "version": "GEMM",
    }

    # Quantize — AWQ uses calibration data internally
    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    # Also save processor if available (for VLMs)
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        processor.save_pretrained(str(save_dir))
    except Exception:
        pass

    logger.info(f"[AWQ] Saved quantized model to {save_dir}")
    return save_dir


def load_awq_model(save_dir: Path, family: str):
    """Load a pre-quantized AWQ model for inference."""
    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        raise ImportError("AutoAWQ not installed. Install with: pip install autoawq")
    from transformers import AutoProcessor, AutoTokenizer

    logger.info(f"[AWQ] Loading quantized model from {save_dir}...")
    model = AutoAWQForCausalLM.from_quantized(
        str(save_dir), fuse_layers=False, trust_remote_code=True,
    )

    # Try processor first, fall back to tokenizer
    try:
        processor = AutoProcessor.from_pretrained(str(save_dir), trust_remote_code=True)
    except Exception:
        processor = AutoTokenizer.from_pretrained(str(save_dir), trust_remote_code=True)

    return model, processor


def _gpu_mem_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info(torch.cuda.current_device())
    return (total - free) / 1024**2


def main():
    parser = argparse.ArgumentParser(description="AWQ quantization pipeline")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--w_bit", type=int, default=4, choices=[3, 4],
                        help="Quantization bit width (default: 4)")
    parser.add_argument("--q_group_size", type=int, default=128,
                        help="Quantization group size (default: 128)")
    parser.add_argument("--n_calib", type=int, default=128,
                        help="Number of calibration samples")
    parser.add_argument("--vqav2_n", type=int, default=1000)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_id = args.model_id
    family = detect_family(model_id)
    safe_name = model_id.replace("/", "__")
    tag = f"{safe_name}__awq_w{args.w_bit}"
    out_path = RESULTS_DIR / f"{tag}.json"

    if out_path.exists() and not args.force:
        logger.info(f"Result already exists at {out_path}. Skipping.")
        return

    if family not in AWQ_SUPPORTED_FAMILIES:
        logger.warning(f"AWQ may not work for family '{family}'. Attempting anyway...")

    # Quantize
    awq_save_dir = RESULTS_DIR / f"{safe_name}__awq_w{args.w_bit}_weights"
    if not awq_save_dir.exists() or args.force:
        quantize_awq(model_id, family, awq_save_dir,
                     w_bit=args.w_bit, q_group_size=args.q_group_size,
                     n_calib=args.n_calib)

    # Load quantized model
    mem_before = _gpu_mem_mb()
    model, processor = load_awq_model(awq_save_dir, family)
    mem_after = _gpu_mem_mb()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_params = sum(p.numel() for p in model.parameters())

    results = {
        "model_id": model_id,
        "family": family,
        "method": "awq",
        "w_bit": args.w_bit,
        "q_group_size": args.q_group_size,
        "num_params_M": round(num_params / 1e6, 1),
        "gpu_mem_load_mb": round(mem_after - mem_before, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmarks": {},
    }

    # Evaluate
    if not args.skip_eval:
        samples = load_vqav2(n_samples=args.vqav2_n)
        results["benchmarks"]["vqav2"] = evaluate_dataset(
            model, processor, samples, family, device,
            "VQAv2", _vqa_accuracy,
        )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"AWQ results saved to {out_path}")

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
