"""
compression/quantized_pretrained/run_quantized_pretrained.py
============================================================
Load and evaluate pre-quantized models from HuggingFace.

These models are already quantized (GPTQ/AWQ) by their authors or the
community. We skip on-device quantization entirely — just download, load,
and evaluate. This is the primary path for getting Category 1 models
(OOM in FP16) running on Jetson Orin Nano.

The model's config.json contains quantization_config that transformers
auto-detects, so we load with our normal family loaders (quant="fp16",
no BnB config). The quantized weights are handled transparently.

Requirements:
  - GPTQ models: pip install optimum auto-gptq
  - AWQ models: pip install autoawq

Usage:
  # Run a specific pre-quantized model
  python compression/quantized_pretrained/run_quantized_pretrained.py \
      --quantized_model_id Qwen/Qwen2.5-VL-3B-Instruct-AWQ \
      --base_model_id Qwen/Qwen2.5-VL-3B-Instruct \
      --quant_method awq --quant_bits 4

  # Run all counterparts from the catalog
  python compression/quantized_pretrained/run_quantized_pretrained.py --run_all

  # Run all counterparts for a specific family
  python compression/quantized_pretrained/run_quantized_pretrained.py \
      --run_all --family qwen25vl
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
    load_vqav2,
    evaluate_dataset, _vqa_accuracy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "quantized_pretrained"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CATALOG_PATH = PROJECT_ROOT / "configs" / "quantized_counterparts.json"

# ── Jetson detection ─────────────────────────────────────────────────────────

def _is_jetson() -> bool:
    """Detect if running on Jetson platform."""
    try:
        with open("/proc/device-tree/model") as f:
            return "jetson" in f.read().lower()
    except (IOError, FileNotFoundError):
        pass
    import platform
    return platform.machine() == "aarch64"


def _available_memory_mb() -> float:
    """Available system memory in MB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except (IOError, ValueError):
        pass
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        return free / 1024**2
    return 0.0


# ── Catalog loading ──────────────────────────────────────────────────────────

def load_catalog() -> dict:
    """Load the pre-quantized counterparts catalog."""
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Catalog not found at {CATALOG_PATH}. "
            "Run the catalog builder first."
        )
    with open(CATALOG_PATH) as f:
        return json.load(f)


def get_runnable_counterparts(catalog: dict,
                              family_filter: str = None,
                              method_filter: str = None) -> list[dict]:
    """Extract (quantized_model_id, base_model_id, family, method, bits, est_gb)
    tuples from the catalog, filtering out GGUF (needs llama.cpp)."""
    counterparts = []
    for base_model_id, info in catalog["models"].items():
        family = info.get("family", "")
        if family_filter and family != family_filter:
            continue
        quantized = info.get("quantized", {})
        for quant_key, quant_info in quantized.items():
            # Skip GGUF — needs llama.cpp, not PyTorch
            if "gguf" in quant_key:
                continue
            # Determine method and bits from the key
            if "awq" in quant_key:
                method = "awq"
            elif "gptq" in quant_key:
                method = "gptq"
            else:
                continue
            if method_filter and method != method_filter:
                continue
            bits = 4 if "int4" in quant_key else 8
            counterparts.append({
                "quantized_model_id": quant_info["model_id"],
                "base_model_id": base_model_id,
                "family": family,
                "quant_method": method,
                "quant_bits": bits,
                "source": quant_info.get("source", "unknown"),
                "est_gb": quant_info.get("est_gb"),
            })
    return counterparts


# ── Single model evaluation ──────────────────────────────────────────────────

def evaluate_quantized_model(quantized_model_id: str,
                             base_model_id: str,
                             family: str,
                             quant_method: str,
                             quant_bits: int,
                             source: str = "unknown",
                             vqav2_n: int = 1000,
                             force: bool = False) -> dict:
    """Load a pre-quantized model, evaluate on VQAv2, save results."""

    safe_name = quantized_model_id.replace("/", "__")
    out_path = RESULTS_DIR / f"{safe_name}.json"

    if out_path.exists() and not force:
        logger.info(f"Result exists at {out_path}. Skipping.")
        with open(out_path) as f:
            return json.load(f)

    # Memory preflight on Jetson
    if _is_jetson():
        est_mb = {"awq": 500, "gptq": 500}.get(quant_method, 500)
        # Rough estimate: INT4 model ~0.5 bytes/param + overhead
        avail = _available_memory_mb()
        logger.info(f"Jetson memory: {avail:.0f} MB available")
        if avail < 1500:
            logger.warning(f"Only {avail:.0f} MB available — too low to attempt load")
            result = {
                "model_id": quantized_model_id,
                "base_model_id": base_model_id,
                "family": family,
                "quant_method": quant_method,
                "quant_bits": quant_bits,
                "source": source,
                "status": "MEM_CRITICAL",
                "error": f"Only {avail:.0f} MB available at preflight",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            return result

    # Load model using existing family loader
    # quant="fp16" means no BnB config — the model's own quantization_config
    # in config.json handles GPTQ/AWQ automatically via transformers
    logger.info(
        f"Loading pre-quantized model: {quantized_model_id} "
        f"(family={family}, method={quant_method}, bits={quant_bits})"
    )

    try:
        model, processor, meta = load_model(
            quantized_model_id, quant="fp16", family=family
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to load {quantized_model_id}: {error_msg}")
        status = "OOM_LOAD" if "out of memory" in error_msg.lower() else "ERROR"
        result = {
            "model_id": quantized_model_id,
            "base_model_id": base_model_id,
            "family": family,
            "quant_method": quant_method,
            "quant_bits": quant_bits,
            "source": source,
            "status": status,
            "error": error_msg[:500],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        return result

    device = str(next(model.parameters()).device)
    num_params = sum(p.numel() for p in model.parameters())

    results = {
        "model_id": quantized_model_id,
        "base_model_id": base_model_id,
        "family": family,
        "quant_method": quant_method,
        "quant_bits": quant_bits,
        "source": source,
        "num_params_M": round(num_params / 1e6, 1),
        "gpu_mem_load_mb": round(meta.gpu_mem_delta_mb, 1),
        "benchmarks": {},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Evaluate on VQAv2
    try:
        eval_samples = load_vqav2(n_samples=vqav2_n)
        results["benchmarks"]["vqav2"] = evaluate_dataset(
            model, processor, eval_samples, family, device,
            "VQAv2", _vqa_accuracy,
        )
        results["status"] = "PASS"
    except torch.cuda.OutOfMemoryError:
        results["status"] = "OOM_INFER"
        results["error"] = "CUDA OOM during inference"
    except Exception as e:
        results["status"] = "ERROR"
        results["error"] = str(e)[:500]

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    unload_model(model)
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pre-quantized HuggingFace models"
    )
    # Single model mode
    parser.add_argument("--quantized_model_id", type=str,
                        help="HuggingFace ID of pre-quantized model")
    parser.add_argument("--base_model_id", type=str,
                        help="HuggingFace ID of the original FP16 base model")
    parser.add_argument("--quant_method", type=str, choices=["awq", "gptq"],
                        help="Quantization method")
    parser.add_argument("--quant_bits", type=int, default=4,
                        help="Quantization bit width")
    # Batch mode
    parser.add_argument("--run_all", action="store_true",
                        help="Run all counterparts from catalog")
    parser.add_argument("--family", type=str,
                        help="Filter by family (e.g., qwen25vl, internvl25)")
    parser.add_argument("--method", type=str, choices=["awq", "gptq"],
                        help="Filter by quantization method")
    # Common
    parser.add_argument("--vqav2_n", type=int, default=1000)
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    if args.run_all:
        catalog = load_catalog()
        counterparts = get_runnable_counterparts(
            catalog,
            family_filter=args.family,
            method_filter=args.method,
        )
        logger.info(f"Found {len(counterparts)} runnable counterparts")

        # Sort by estimated size (smallest first — most likely to succeed on Jetson)
        counterparts.sort(key=lambda c: c.get("est_gb") or 99)

        for i, cp in enumerate(counterparts, 1):
            logger.info(
                f"\n{'='*60}\n"
                f"[{i}/{len(counterparts)}] {cp['quantized_model_id']}\n"
                f"  Base: {cp['base_model_id']} | "
                f"{cp['quant_method'].upper()} INT{cp['quant_bits']} | "
                f"~{cp.get('est_gb', '?')} GB\n"
                f"{'='*60}"
            )
            try:
                evaluate_quantized_model(
                    quantized_model_id=cp["quantized_model_id"],
                    base_model_id=cp["base_model_id"],
                    family=cp["family"],
                    quant_method=cp["quant_method"],
                    quant_bits=cp["quant_bits"],
                    source=cp["source"],
                    vqav2_n=args.vqav2_n,
                    force=args.force,
                )
            except Exception as e:
                logger.error(f"Unexpected error on {cp['quantized_model_id']}: {e}")
                continue

            # Clean up between models
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    elif args.quantized_model_id:
        if not args.base_model_id:
            parser.error("--base_model_id required in single-model mode")
        if not args.quant_method:
            parser.error("--quant_method required in single-model mode")

        family = detect_family(args.base_model_id)
        evaluate_quantized_model(
            quantized_model_id=args.quantized_model_id,
            base_model_id=args.base_model_id,
            family=family,
            quant_method=args.quant_method,
            quant_bits=args.quant_bits,
            vqav2_n=args.vqav2_n,
            force=args.force,
        )
    else:
        parser.error("Specify --quantized_model_id or --run_all")


if __name__ == "__main__":
    main()
