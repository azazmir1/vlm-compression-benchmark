"""
scripts/show_results.py
========================
CLI tool to print a summary table of all collected results.

Usage:
  python scripts/show_results.py
  python scripts/show_results.py --method baseline
  python scripts/show_results.py --benchmark textvqa
  python scripts/show_results.py --family smolvlm
"""

import argparse
import json
from pathlib import Path

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
BENCHMARKS   = ["vqav2", "textvqa", "pope"]
COL_W        = 10


def load_all() -> list[dict]:
    rows = []
    for method_dir in sorted(RESULTS_ROOT.iterdir()):
        if not method_dir.is_dir() or method_dir.name == "__pycache__":
            continue
        method = method_dir.name
        for jf in sorted(method_dir.glob("*.json")):
            with open(jf) as f:
                data = json.load(f)
            base = {
                "method":    method,
                "model_id":  data.get("model_id", jf.stem),
                "family":    data.get("family", ""),
                "params_M":  data.get("num_params_M", ""),
                "quant":     data.get("quant", ""),
                "mem_mb":    data.get("gpu_mem_load_mb", ""),
                "sparsity":  data.get("actual_sparsity", ""),
                "lora_rank": data.get("lora_rank", ""),
            }
            for bname in BENCHMARKS:
                bdata = data.get("benchmarks", {}).get(bname, {})
                base[f"{bname}_acc"]     = bdata.get("accuracy", None)
                base[f"{bname}_lat_s"]   = bdata.get("avg_latency_s", None)
                base[f"{bname}_mem_mb"]  = bdata.get("peak_memory_mb", None)
                base[f"{bname}_tput"]    = bdata.get("throughput_sps", None)
            rows.append(base)
    return rows


def fmt(v, decimals=3) -> str:
    if v is None or v == "":
        return "-"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def print_table(rows: list[dict], benchmark: str = "vqav2"):
    if not rows:
        print("No results found.")
        return

    acc_k  = f"{benchmark}_acc"
    lat_k  = f"{benchmark}_lat_s"
    mem_k  = f"{benchmark}_mem_mb"
    tput_k = f"{benchmark}_tput"

    header = (
        f"{'MODEL':<45} {'FAMILY':<12} {'PARAMS':>7} {'METHOD':<10} "
        f"{'QUANT':<6} {benchmark.upper()+' ACC':>10} {'LAT(s)':>8} "
        f"{'MEM(MB)':>9} {'TPUT':>7}"
    )
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)
    for r in rows:
        short_id = r["model_id"].split("/")[-1][:44]
        print(
            f"{short_id:<45} {r['family']:<12} {fmt(r['params_M'],1):>7} "
            f"{r['method']:<10} {r['quant']:<6} "
            f"{fmt(r[acc_k]):>10} {fmt(r[lat_k]):>8} "
            f"{fmt(r[mem_k],0):>9} {fmt(r[tput_k],2):>7}"
        )
    print(sep)
    print(f"Total: {len(rows)} rows")


def main():
    parser = argparse.ArgumentParser(description="Show benchmark results summary")
    parser.add_argument("--method",    default="",       help="Filter by method (baseline/ptq/pruning/qlora)")
    parser.add_argument("--family",    default="",       help="Filter by family name")
    parser.add_argument("--benchmark", default="vqav2",  choices=BENCHMARKS,
                        help="Benchmark to display (default: vqav2)")
    args = parser.parse_args()

    rows = load_all()

    if args.method:
        rows = [r for r in rows if r["method"] == args.method]
    if args.family:
        rows = [r for r in rows if args.family.lower() in r["family"].lower()]

    print_table(rows, benchmark=args.benchmark)


if __name__ == "__main__":
    main()
