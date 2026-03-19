"""
compression/ptq/run_ptq.py
===========================
Post-Training Quantization (PTQ) pipeline.

Supports:
  INT8  — via BitsAndBytes
  INT4  — via BitsAndBytes (NF4) or AutoAWQ

Usage:
  python compression/ptq/run_ptq.py --model_id HuggingFaceTB/SmolVLM-256M-Instruct --quant int8
  python compression/ptq/run_ptq.py --model_id microsoft/Florence-2-base --quant int4 --backend bnb
  python compression/ptq/run_ptq.py --model_id Qwen/Qwen2.5-VL-7B-Instruct --quant int4 --backend awq
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.model_loader import load_model, unload_model
from evaluation.run_baseline import (
    load_vqav2, load_textvqa, load_pope,
    evaluate_dataset, _vqa_accuracy, _pope_accuracy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "ptq"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── AWQ quantization helper ──────────────────────────────────────────────────

# AWQ only works reliably on pure decoder-only text LLM backbones.
# Vision-language families with custom architectures are not supported.
AWQ_SUPPORTED_FAMILIES = {"qwen25vl"}


def quantize_awq(model_id: str, family: str, save_dir: Path) -> Path:
    """Quantize with AutoAWQ and save locally. Returns saved path."""
    if family not in AWQ_SUPPORTED_FAMILIES:
        raise ValueError(
            f"AWQ backend is not supported for family '{family}'. "
            f"Supported: {AWQ_SUPPORTED_FAMILIES}. Use --backend bnb instead."
        )
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    logger.info(f"[AWQ] Quantizing {model_id} ...")
    model = AutoAWQForCausalLM.from_pretrained(model_id, safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }
    model.quantize(tokenizer, quant_config=quant_config)
    model.save_quantized(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    logger.info(f"[AWQ] Saved quantized model to {save_dir}")
    return save_dir


# ── Parameter counting ────────────────────────────────────────────────────────

def _count_logical_params(model, quant: str) -> int:
    """Count logical (unpacked) parameters.

    BnB INT4 stores Params4bit tensors with 2 weights packed per byte, so
    p.numel() returns logical_params/2 for those layers.  Multiplying by 2
    restores the true parameter count so INT4 and FP16 are comparable.
    """
    total = 0
    for p in model.parameters():
        n = p.numel()
        if quant == "int4":
            try:
                import bitsandbytes as bnb
                if isinstance(p, bnb.nn.Params4bit):
                    n *= 2
            except (ImportError, AttributeError):
                pass
        total += n
    return total


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PTQ compression pipeline")
    parser.add_argument("--model_id",    required=True)
    parser.add_argument("--quant",       required=True, choices=["int8", "int4"],
                        help="Quantization level")
    parser.add_argument("--backend",     default="bnb",
                        choices=["bnb", "awq"],
                        help="Backend: bnb (BitsAndBytes) or awq (AutoAWQ). Default: bnb")
    parser.add_argument("--vqav2_n",     type=int, default=1000)
    parser.add_argument("--skip_vqav2",  action="store_true")
    parser.add_argument("--skip_textvqa",action="store_true")
    parser.add_argument("--skip_pope",   action="store_true")
    parser.add_argument("--force",       action="store_true",
                        help="Overwrite existing result even if it already exists")
    args = parser.parse_args()

    model_id  = args.model_id
    safe_name = model_id.replace("/", "__")
    tag       = f"{safe_name}__{args.quant}__{args.backend}"
    out_path  = RESULTS_DIR / f"{tag}.json"

    if out_path.exists() and not args.force:
        logger.info(f"Result already exists at {out_path}. Skipping.")
        return

    # ── Quantize & load ──────────────────────────────────────────────────
    from models.model_loader import detect_family
    family = detect_family(model_id)

    if args.backend == "awq" and args.quant == "int4":
        awq_save = RESULTS_DIR / f"{safe_name}__awq_int4_weights"
        if not awq_save.exists():
            quantize_awq(model_id, family, awq_save)
        load_id = str(awq_save)
        quant   = None   # model is already quantized; load in fp16
    else:
        load_id = model_id
        quant   = args.quant  # 'int8' or 'int4' → BitsAndBytes

    logger.info(f"Loading {load_id} with quant={quant} backend={args.backend}")
    model, processor, meta = load_model(load_id, quant=quant)
    family = meta.family
    device = str(next(model.parameters()).device)

    fp16_size = meta.gpu_mem_delta_mb  # approximate; full precision would be ~2× INT8 or ~4× INT4
    num_params = _count_logical_params(model, args.quant)

    # Preserve any benchmark results that are being skipped this run
    existing_benchmarks: dict = {}
    if out_path.exists() and args.force:
        with open(out_path) as f:
            existing_benchmarks = json.load(f).get("benchmarks", {})

    results: dict = {
        "model_id":         model_id,
        "family":           family,
        "quant":            args.quant,
        "backend":          args.backend,
        "num_params_M":     round(num_params / 1e6, 1),
        "gpu_mem_load_mb":  meta.gpu_mem_delta_mb,
        "compression_ratio": None,   # filled below if baseline exists
        "benchmarks":       dict(existing_benchmarks),
    }

    # ── Compression ratio vs fp16 baseline ──────────────────────────────
    baseline_path = (
        Path(__file__).resolve().parents[2]
        / "results" / "baseline" / f"{safe_name}.json"
    )
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        baseline_mem = baseline.get("gpu_mem_load_mb", 0)
        if baseline_mem > 0 and meta.gpu_mem_delta_mb > 0:
            results["compression_ratio"] = round(baseline_mem / meta.gpu_mem_delta_mb, 2)

    # ── Evaluate ─────────────────────────────────────────────────────────
    if not args.skip_vqav2:
        samples = load_vqav2(n_samples=args.vqav2_n)
        results["benchmarks"]["vqav2"] = evaluate_dataset(
            model, processor, samples, family, device,
            "VQAv2", _vqa_accuracy,
        )

    if not args.skip_textvqa:
        samples = load_textvqa()
        results["benchmarks"]["textvqa"] = evaluate_dataset(
            model, processor, samples, family, device,
            "TextVQA", _vqa_accuracy,
        )

    if not args.skip_pope:
        samples = load_pope()
        results["benchmarks"]["pope"] = evaluate_dataset(
            model, processor, samples, family, device,
            "POPE", _pope_accuracy,
        )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"PTQ results saved to {out_path}")

    unload_model(model)


if __name__ == "__main__":
    main()
