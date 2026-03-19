"""
analysis/deployability_report.py
==================================
Generates the deployment feasibility report from all collected results.

Produces:
  1. Positive list  — models deployable on BOTH Jetson AND RPi5
  2. Negative list  — models that cannot run on either edge device (GPU-only)
  3. Jetson-only    — models that fit Jetson but not RPi5
  4. Parameter ceiling per architecture per device

Thresholds (from device_constraints.yaml):
  RPi5    : peak_memory_mb < 4096  AND  avg_latency_s < 10 s
  Jetson  : peak_memory_mb < 6144  AND  avg_latency_s < 3 s

Saves results/deployability_summary.json and prints a formatted report.

Usage:
  python analysis/deployability_report.py
  python analysis/deployability_report.py --verbose
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
OUT_PATH     = RESULTS_ROOT / "deployability_summary.json"

THRESHOLDS = {
    "rpi5":   {"memory_mb": 4096, "latency_s": 10.0},
    "jetson": {"memory_mb": 6144, "latency_s": 3.0},
}

BENCHMARKS = ["vqav2", "textvqa", "pope"]


def load_all_results() -> list[dict]:
    """Load all JSONs from results/ subdirectories."""
    rows = []
    for method_dir in sorted(RESULTS_ROOT.iterdir()):
        if not method_dir.is_dir() or method_dir.name.startswith("_"):
            continue
        method = method_dir.name
        for jf in sorted(method_dir.glob("*.json")):
            try:
                with open(jf) as f:
                    data = json.load(f)
                data["_method"] = method
                data["_file"]   = str(jf)
                rows.append(data)
            except Exception:
                continue
    return rows


def get_benchmark_kpi(result: dict, bench: str, kpi: str):
    return result.get("benchmarks", {}).get(bench, {}).get(kpi, None)


def check_deployable(result: dict, device: str, bench: str = "vqav2") -> bool:
    """Check if a result meets a device's memory + latency thresholds."""
    thresh = THRESHOLDS[device]
    mem = get_benchmark_kpi(result, bench, "peak_memory_mb")
    lat = get_benchmark_kpi(result, bench, "avg_latency_s")
    if mem is None or lat is None:
        return False
    return mem < thresh["memory_mb"] and lat < thresh["latency_s"]


def compute_param_ceiling(rows: list[dict], device: str) -> dict[str, dict]:
    """
    For each (family, compression_method) pair, find the maximum parameter count
    where the model is still deployable on the given device.
    Returns {family: {method: max_params_M}}
    """
    ceiling: dict[str, dict] = defaultdict(dict)

    for r in rows:
        if not r.get("benchmarks"):
            continue
        family  = r.get("family", "unknown")
        method  = r.get("_method", "unknown")
        params  = r.get("num_params_M", 0) or 0

        if check_deployable(r, device):
            current = ceiling[family].get(method, 0)
            ceiling[family][method] = max(current, params)

    return {f: dict(m) for f, m in ceiling.items()}


def print_separator(char="-", width=70):
    print(char * width)


def main():
    parser = argparse.ArgumentParser(description="Generate deployability report")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    rows = load_all_results()
    if not rows:
        print("No results found. Run experiments first.")
        return

    # ── Classify each result ──────────────────────────────────────────────
    both_devices = []
    rpi5_only    = []
    jetson_only  = []
    gpu_only     = []

    seen = set()  # deduplicate by (model_id, method, quant)
    for r in rows:
        model_id = r.get("model_id", "")
        method   = r.get("_method", "")
        quant    = r.get("quant", "")
        key      = (model_id, method, quant)
        if key in seen:
            continue
        seen.add(key)

        rpi5_ok   = check_deployable(r, "rpi5")
        jetson_ok = check_deployable(r, "jetson")

        entry = {
            "model_id":  model_id,
            "family":    r.get("family", ""),
            "params_M":  r.get("num_params_M", None),
            "method":    method,
            "quant":     quant,
            "vqav2_acc": get_benchmark_kpi(r, "vqav2", "accuracy"),
            "mem_mb":    get_benchmark_kpi(r, "vqav2", "peak_memory_mb"),
            "lat_s":     get_benchmark_kpi(r, "vqav2", "avg_latency_s"),
        }

        if rpi5_ok and jetson_ok:
            both_devices.append(entry)
        elif rpi5_ok:
            rpi5_only.append(entry)
        elif jetson_ok:
            jetson_only.append(entry)
        else:
            gpu_only.append(entry)

    # Sort by params (largest first for negative list; smallest first for positive)
    both_devices.sort(key=lambda x: x.get("params_M") or 0)
    gpu_only.sort(key=lambda x: -(x.get("params_M") or 0))

    # ── Parameter ceilings ────────────────────────────────────────────────
    rpi5_ceilings   = compute_param_ceiling(rows, "rpi5")
    jetson_ceilings = compute_param_ceiling(rows, "jetson")

    # ── Print report ──────────────────────────────────────────────────────
    print_separator("=")
    print("  DEPLOYABILITY REPORT — VLM Compression Benchmark")
    print_separator("=")

    print(f"\n✅  BOTH DEVICES (RPi5 + Jetson) — {len(both_devices)} combos")
    print_separator()
    for e in both_devices[:10]:
        acc = f"{e['vqav2_acc']:.3f}" if e['vqav2_acc'] else "N/A"
        mem = f"{e['mem_mb']:.0f}MB"  if e['mem_mb']    else "N/A"
        lat = f"{e['lat_s']:.2f}s"   if e['lat_s']     else "N/A"
        print(f"  {e['model_id'].split('/')[-1]:<35} {e['params_M']:>6}M  {e['method']:<10} {e['quant']:<6}  acc={acc}  mem={mem}  lat={lat}")

    print(f"\n⚠️   JETSON ONLY (too slow/large for RPi5) — {len(jetson_only)} combos")
    print_separator()
    for e in sorted(jetson_only, key=lambda x: x.get("params_M") or 0)[:8]:
        acc = f"{e['vqav2_acc']:.3f}" if e['vqav2_acc'] else "N/A"
        print(f"  {e['model_id'].split('/')[-1]:<35} {e['params_M']:>6}M  {e['method']:<10} {e['quant']:<6}  acc={acc}")

    print(f"\n❌  GPU-ONLY (cannot run on either edge device) — {len(gpu_only)} combos")
    print_separator()
    for e in gpu_only[:10]:
        acc = f"{e['vqav2_acc']:.3f}" if e['vqav2_acc'] else "N/A"
        mem = f"{e['mem_mb']:.0f}MB"  if e['mem_mb']    else "N/A"
        print(f"  {e['model_id'].split('/')[-1]:<35} {e['params_M']:>6}M  {e['method']:<10} {e['quant']:<6}  acc={acc}  mem={mem}")

    print("\n📊  PARAMETER CEILING PER ARCHITECTURE (RPi5)")
    print_separator()
    for family, methods in sorted(rpi5_ceilings.items()):
        for method, max_m in sorted(methods.items()):
            print(f"  {family:<12}  {method:<10}  max_deployable = {max_m:.0f}M")

    print("\n📊  PARAMETER CEILING PER ARCHITECTURE (Jetson)")
    print_separator()
    for family, methods in sorted(jetson_ceilings.items()):
        for method, max_m in sorted(methods.items()):
            print(f"  {family:<12}  {method:<10}  max_deployable = {max_m:.0f}M")

    # ── Save JSON ─────────────────────────────────────────────────────────
    summary = {
        "positive_both_devices": both_devices,
        "jetson_only":           jetson_only,
        "rpi5_only":             rpi5_only,
        "negative_gpu_only":     gpu_only,
        "param_ceiling_rpi5":    rpi5_ceilings,
        "param_ceiling_jetson":  jetson_ceilings,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Report saved to {OUT_PATH}")
    print_separator("=")


if __name__ == "__main__":
    main()
