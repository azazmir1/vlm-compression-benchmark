"""
scripts/reorganize_results.py
==============================
Reorganize all benchmark results into a clean structure:
  results/a6000/{method}/{model}.json
  results/jetson/{method}/{model}.json

Scans both results/ and results_backup_20260324/results/ to collect everything.

Device classification:
  - device field == "jetson_orin_nano_8gb" -> Jetson
  - avg_power_w == 0.0 or missing + heuristics -> Jetson
  - avg_power_w >= 20 -> A6000
  - Special cases handled explicitly

Method normalization:
  - Unifies scattered naming (bnb int4/int8, awq, gptq, etc.)
  - Splits casp_slim into casp/ and slim/
  - Splits awq_gptq into awq/ and gptq/
"""

import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT / "results"
BACKUP_DIR = PROJECT / "results_backup_20260324" / "results"

# ── Collect all JSON result files ──────────────────────────────────────────

def load_result(path):
    """Load a JSON result file, return dict or None."""
    try:
        with open(path) as f:
            d = json.load(f)
        # Must be a dict with model_id to be a valid result
        if not isinstance(d, dict):
            return None
        if "model_id" not in d and "model" not in d:
            return None
        return d
    except Exception:
        return None


def classify_device(data, filepath):
    """Classify a result as 'a6000' or 'jetson' or None (skip)."""
    fp = str(filepath)

    # Explicit device field
    if data.get("device") == "jetson_orin_nano_8gb":
        return "jetson"

    # Jetson baseline directory
    if "/jetson/baseline/" in fp or "/jetson/cat2_" in fp:
        return "jetson"

    # Power-based detection
    bm = data.get("benchmarks", {}).get("vqav2", {})
    metrics_block = data.get("metrics", {})  # old jetson baseline format
    power = bm.get("avg_power_w", metrics_block.get("avg_power_w"))

    if power is not None and power >= 20:
        return "a6000"

    # power == 0.0 cases need heuristics
    method = get_method(data, filepath)

    # QLoRA only works on A6000 (needs BnB)
    if method in ("qlora_r16", "qlora_r64", "lora_r16", "lora_r64"):
        return "a6000"

    # Quantized pretrained - A6000 testing
    if "quantized_pretrained" in fp:
        return "a6000"

    # Methods with power=0.0 that ran on Jetson
    if power is not None and power == 0.0:
        # pytorch_int8 is the Jetson breakthrough method
        if "pytorch_int8" in fp or "pytorch_int4" in fp or "int4_casp" in fp:
            return "jetson"
        # Old backup files at power=0.0 with small n_samples
        n = bm.get("n_samples", metrics_block.get("n_evaluated"))
        if n is not None and n <= 30:
            return "jetson"

    # If power is None (no benchmarks), check context
    if power is None:
        if "jetson" in fp.lower():
            return "jetson"
        if "quantized_pretrained" in fp:
            return "a6000"
        # Old jetson baseline format
        if data.get("device") == "jetson_orin_nano_8gb":
            return "jetson"
        return None  # skip unclassifiable

    return "a6000"


def get_method(data, filepath):
    """Extract a normalized method name from the result."""
    fp = str(filepath)
    method = data.get("method", "")
    quant = data.get("quant", "")

    # Jetson baseline (old format)
    if "/jetson/baseline/" in fp:
        return "baseline_fp16"
    if method == "baseline_fp16":
        return "baseline_fp16"

    # Baseline
    if "/baseline/" in fp and not method:
        return "baseline_fp16"
    if method == "" and quant == "fp16" and "/baseline/" in fp:
        return "baseline_fp16"

    # BnB quantization
    if method in ("int4", "int8") or (quant in ("int4", "int8") and "bnb" in fp):
        return f"bnb_{method or quant}"
    # Old jetson format: precision field instead of method
    precision = data.get("precision", "")
    if precision in ("int4", "int8") and "bnb" in fp:
        return f"bnb_{precision}"
    if precision in ("int4", "int8") and "/ptq/" in fp:
        return f"bnb_{precision}"

    # Native pre-quantized
    if "quantized_pretrained" in fp:
        return "quantized_pretrained"

    # Moondream native int4
    if precision == "int4" and "native" in fp:
        return "native_int4"
    # Native int4 pre-quantized model (quant field but no method)
    if quant == "int4" and "native" in fp:
        return "native_int4"

    # PyTorch custom quantization
    if method == "pytorch_int8" or "pytorch_int8" in fp:
        return "pytorch_int8"
    if "pytorch_int4" in fp:
        return "pytorch_int4"
    if "int4_casp" in fp:
        return "int4_casp"

    # Pruning methods
    if "/pruning/" in fp and not method:
        # Extract from filename: *__sp20.json -> magnitude_sp20
        fname = Path(filepath).stem
        for tag in ("sp20", "sp40", "sp50"):
            if tag in fname:
                return f"magnitude_{tag}"
        return "magnitude_pruning"

    if method == "magnitude_sp20":
        return "magnitude_sp20"
    if method == "magnitude_sp40":
        return "magnitude_sp40"

    # Wanda
    if "/wanda/" in fp or "wanda" in method:
        fname = Path(filepath).stem
        for tag in ("sp20", "sp40", "sp50"):
            if tag in fname:
                return f"wanda_{tag}"
        return "wanda"

    # SparseGPT
    if "sparsegpt" in fp or "sparsegpt" in method:
        return "sparsegpt_sp50"

    # SVD-LLM
    if "svd_llm" in method or "svd" in fp:
        return "svd_llm"

    # PALU
    if method == "palu" or "palu" in fp:
        return "palu"

    # PACT
    if method == "pact" or "pact" in fp:
        return "pact"

    # CASP
    if method == "casp":
        return "casp"

    # SLIM
    if method == "slim":
        slim_tag = ""
        sp = data.get("sparsity", data.get("target_sparsity"))
        rr = data.get("rank_ratio")
        if sp is not None and rr is not None:
            slim_tag = f"_sp{int(sp*100)}_r{int(rr*100)}"
        elif "sp50_r30" in fp:
            slim_tag = "_sp50_r30"
        elif "sp30_r20" in fp:
            slim_tag = "_sp30_r20"
        elif "sp20_r10" in fp:
            slim_tag = "_sp20_r10"
        return f"slim{slim_tag}"

    # AWQ / GPTQ (simulated) — check method field first (awq_gptq dir has both)
    if method == "awq":
        return "awq"
    if method == "gptq":
        return "gptq"
    if "awq" in fp and "awp" not in fp and "gptq" not in fp:
        return "awq"

    # AWP
    if method == "awp" or "awp" in fp:
        return "awp"

    # QLoRA / LoRA
    if method == "lora" or "qlora" in fp or "lora" in fp:
        fname = Path(filepath).stem
        if "r16" in fname:
            return "qlora_r16"
        if "r64" in fname:
            return "qlora_r64"
        return "qlora"

    # Fallback
    if method:
        return method
    return "unknown"


def get_model_filename(data, filepath):
    """Generate a clean filename for the result."""
    model_id = data.get("model_id", "")
    safe_name = model_id.replace("/", "__")
    if not safe_name:
        return Path(filepath).name
    return safe_name + ".json"


def get_method_dir(method):
    """Map method name to directory name."""
    # Group related methods
    if method.startswith("magnitude_"):
        return "magnitude_pruning"
    if method.startswith("wanda_"):
        return "wanda"
    if method.startswith("slim"):
        return "slim"
    if method.startswith("qlora") or method.startswith("lora"):
        return "qlora"
    if method.startswith("bnb_"):
        return "bnb_quantization"
    return method


def scan_directory(base_dir, results_list):
    """Recursively scan for JSON result files."""
    skip_dirs = {"cat2_working_methods_logs", "untested_256m_logs"}
    skip_files = {"all_results_flat.json", "all_trials.json", "summary.json",
                  "gptq_comparison_qwen7b.json", "run_summary.json"}

    for root, dirs, files in os.walk(base_dir):
        # Skip non-result directories
        dirs[:] = [d for d in dirs if d not in skip_dirs]

        for fname in files:
            if not fname.endswith(".json"):
                continue
            if fname in skip_files:
                continue

            fpath = Path(root) / fname
            # Skip model weight files inside awp weights dirs
            if "weights" in str(fpath) or "wanda_sp50_weights" in str(fpath):
                continue

            data = load_result(fpath)
            if data is None:
                continue

            results_list.append((fpath, data))


def main():
    print("Scanning results directories...")

    all_results = []
    scan_directory(RESULTS_DIR, all_results)
    if BACKUP_DIR.exists():
        scan_directory(BACKUP_DIR, all_results)

    print(f"Found {len(all_results)} valid result files")

    # Classify and organize
    # Key: (device, method_dir, filename) -> (filepath, data, method, n_samples)
    organized = {}
    skipped = []

    for fpath, data in all_results:
        device = classify_device(data, fpath)
        if device is None:
            skipped.append(str(fpath))
            continue

        method = get_method(data, fpath)
        method_dir = get_method_dir(method)
        filename = get_model_filename(data, fpath)

        # For methods with variants (sp20/sp40), include in filename
        if method != method_dir:
            suffix = method.replace(method_dir, "").strip("_")
            if suffix:
                filename = filename.replace(".json", f"__{suffix}.json")

        key = (device, method_dir, filename)

        # Get sample count for dedup
        bm = data.get("benchmarks", {}).get("vqav2", {})
        metrics = data.get("metrics", {})
        n = bm.get("n_samples", metrics.get("n_evaluated", 0)) or 0

        if key in organized:
            existing_n = organized[key][3]
            if n > existing_n:
                organized[key] = (fpath, data, method, n)
            # else keep existing (more samples)
        else:
            organized[key] = (fpath, data, method, n)

    # Print plan
    by_device = defaultdict(lambda: defaultdict(list))
    for (device, method_dir, filename), (fpath, data, method, n) in organized.items():
        by_device[device][method_dir].append((filename, fpath, n))

    print(f"\n{'='*70}")
    print("REORGANIZATION PLAN")
    print(f"{'='*70}")

    total_files = 0
    for device in ["a6000", "jetson"]:
        methods = by_device[device]
        device_total = sum(len(v) for v in methods.values())
        total_files += device_total
        print(f"\n  results/{device}/  ({device_total} files)")
        for method_dir in sorted(methods.keys()):
            files = methods[method_dir]
            print(f"    {method_dir}/  ({len(files)} files)")
            for fname, src, n in sorted(files):
                print(f"      {fname}  (n={n}, from {src})")

    if skipped:
        print(f"\n  SKIPPED ({len(skipped)} files - unclassifiable):")
        for s in skipped:
            print(f"    {s}")

    print(f"\n  TOTAL: {total_files} files")
    print(f"{'='*70}")

    if "--dry-run" in sys.argv:
        print("\nDry run complete. Use --execute to apply changes.")
        return

    if "--execute" not in sys.argv:
        print("\nUse --dry-run to preview or --execute to apply changes.")
        return

    # Execute reorganization
    print("\nExecuting reorganization...")

    # Create new directories and copy files
    for (device, method_dir, filename), (fpath, data, method, n) in organized.items():
        dest_dir = RESULTS_DIR / device / method_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / filename

        # Add device field if missing
        if "device" not in data:
            data["device"] = "jetson_orin_nano_8gb" if device == "jetson" else "a6000_48gb"
        # Normalize method field
        if not data.get("method"):
            data["method"] = method

        with open(dest_path, "w") as f:
            json.dump(data, f, indent=2)

    print(f"Created {total_files} files in new structure.")

    # Clean up old scattered directories (move to _old/)
    old_dirs = [
        "awp", "awq_gptq", "baseline", "casp_slim", "int4_casp",
        "pact", "palu", "pruning", "ptq", "pytorch_int4",
        "pytorch_int8", "pytorch_int8_gpu", "sparsegpt", "svd_llm",
        "wanda", "onnx",
    ]
    archive = RESULTS_DIR / "_old_scattered"
    archive.mkdir(exist_ok=True)

    for d in old_dirs:
        src = RESULTS_DIR / d
        if src.exists() and src.is_dir():
            dest = archive / d
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(src), str(dest))
            print(f"  Moved results/{d}/ -> results/_old_scattered/{d}/")

    # Move stray files
    for stray in ["gptq_comparison_qwen7b.json"]:
        src = RESULTS_DIR / stray
        if src.exists():
            shutil.move(str(src), str(archive / stray))

    print("\nDone! Old directories moved to results/_old_scattered/")
    print("New structure: results/a6000/{method}/ and results/jetson/{method}/")


if __name__ == "__main__":
    main()
