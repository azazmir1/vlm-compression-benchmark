"""
compression/combined/run_awp.py
================================
AWP: Activation-Aware Weight Pruning + Quantization pipeline.

Based on: "AWP: Activation-Aware Weight Pruning and Quantization"
          (MERL, ICML 2025)

Key idea: Combines activation-aware pruning (Wanda) with quantization (BnB
INT4) for maximum compression. The paper shows that pruning-first
(Wanda → then quantize) outperforms quantization-first.

Pipeline:
  1. Load model in FP16
  2. Calibration pass to collect activation norms
  3. Apply Wanda pruning at target sparsity (e.g., 50% or 2:4 structured)
  4. Save pruned model
  5. Reload pruned model with INT4 quantization (BnB NF4)
  6. Evaluate combined compression

Usage:
  python compression/combined/run_awp.py \
      --model_id HuggingFaceTB/SmolVLM-256M-Instruct --sparsity 0.50

  python compression/combined/run_awp.py \
      --model_id Qwen/Qwen2.5-VL-3B-Instruct --sparsity 0.50 --quant int4
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
from models.model_loader import load_model, unload_model, detect_family
from compression.pruning.run_wanda import (
    ActivationCollector, apply_wanda_pruning, measure_sparsity,
)
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

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "awp"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _gpu_mem_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info(torch.cuda.current_device())
    return (total - free) / 1024**2


def main():
    parser = argparse.ArgumentParser(
        description="AWP: Combined Wanda pruning + quantization pipeline")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--sparsity", type=float, default=0.50,
                        help="Target sparsity for Wanda pruning (default: 0.50)")
    parser.add_argument("--quant", default="int4", choices=["int4", "int8"],
                        help="Post-pruning quantization (default: int4)")
    parser.add_argument("--n_calib", type=int, default=128,
                        help="Number of calibration samples")
    parser.add_argument("--vqav2_n", type=int, default=1000)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_id = args.model_id
    family = detect_family(model_id)
    safe_name = model_id.replace("/", "__")
    sp_tag = f"sp{int(args.sparsity * 100)}"
    tag = f"{safe_name}__awp_{sp_tag}_{args.quant}"
    out_path = RESULTS_DIR / f"{tag}.json"

    if out_path.exists() and not args.force:
        logger.info(f"Result already exists at {out_path}. Skipping.")
        return

    # ── Phase 1: Load FP16 + Wanda pruning ───────────────────────────────
    logger.info(f"Phase 1: Loading {model_id} (fp16) for Wanda pruning...")
    model, processor, meta = load_model(model_id, quant="fp16")
    device = str(next(model.parameters()).device)
    num_params = sum(p.numel() for p in model.parameters())

    # Calibration
    logger.info(f"Calibration pass ({args.n_calib} samples)...")
    calib_samples = load_vqav2(n_samples=args.n_calib)

    collector = ActivationCollector(model)
    skipped = 0
    for sample in calib_samples:
        try:
            with torch.no_grad():
                _ = run_inference(model, processor, sample, family, device)
        except Exception:
            skipped += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    if skipped:
        logger.warning(f"  Calibration: skipped {skipped}/{args.n_calib} samples")

    input_norms = collector.get_input_norms()
    collector.remove_hooks()

    # Apply Wanda pruning
    logger.info(f"Applying Wanda pruning at sparsity={args.sparsity}...")
    prune_stats = apply_wanda_pruning(model, input_norms, args.sparsity)
    sparsity_after_prune = measure_sparsity(model)
    logger.info(f"Post-pruning sparsity: {sparsity_after_prune:.4f}")

    mem_after_prune = _gpu_mem_mb()

    # ── Phase 2: Evaluate pruned-only (for comparison) ───────────────────
    pruned_only_metrics = {}
    if not args.skip_eval:
        logger.info("Evaluating pruned-only model (before quantization)...")
        eval_samples = load_vqav2(n_samples=min(args.vqav2_n, 100))
        pruned_only_metrics = evaluate_dataset(
            model, processor, eval_samples, family, device,
            "VQAv2 (pruned-only)", _vqa_accuracy,
        )

    # ── Phase 3: Save pruned model, reload with quantization ─────────────
    pruned_save_dir = RESULTS_DIR / f"{safe_name}__wanda_{sp_tag}_weights"
    pruned_save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving pruned model to {pruned_save_dir}...")
    model.save_pretrained(str(pruned_save_dir))
    try:
        processor.save_pretrained(str(pruned_save_dir))
    except Exception:
        pass

    # Unload FP16 model
    unload_model(model)
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(3)

    # Reload with quantization
    logger.info(f"Phase 2: Reloading pruned model with {args.quant} quantization...")
    mem_before_quant = _gpu_mem_mb()
    model_q, processor_q, meta_q = load_model(
        str(pruned_save_dir), quant=args.quant, family=family,
    )
    mem_after_quant = _gpu_mem_mb()
    device = str(next(model_q.parameters()).device)

    # ── Phase 4: Evaluate combined ───────────────────────────────────────
    results = {
        "model_id": model_id,
        "family": family,
        "method": "awp",
        "pipeline": f"wanda_{sp_tag}+{args.quant}",
        "pruning_method": "wanda",
        "target_sparsity": args.sparsity,
        "actual_sparsity": prune_stats["actual_sparsity"],
        "quantization": args.quant,
        "pruned_layers": prune_stats["pruned_layers"],
        "num_params_M": round(num_params / 1e6, 1),
        "gpu_mem_fp16_mb": round(meta.gpu_mem_delta_mb, 1),
        "gpu_mem_pruned_quant_mb": round(mem_after_quant - mem_before_quant, 1),
        "compression_ratio": round(
            meta.gpu_mem_delta_mb / max(mem_after_quant - mem_before_quant, 1), 2
        ),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmarks": {},
    }

    if pruned_only_metrics:
        results["benchmarks"]["vqav2_pruned_only"] = pruned_only_metrics

    if not args.skip_eval:
        logger.info("Evaluating combined pruned+quantized model...")
        eval_samples = load_vqav2(n_samples=args.vqav2_n)
        results["benchmarks"]["vqav2"] = evaluate_dataset(
            model_q, processor_q, eval_samples, family, device,
            "VQAv2", _vqa_accuracy,
        )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"AWP results saved to {out_path}")

    unload_model(model_q)


if __name__ == "__main__":
    main()
