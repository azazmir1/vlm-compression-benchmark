"""
scripts/show_rpi5_results.py
==============================
Aggregate and display all RPi5 benchmark results in a comparison table.

Usage:
  python scripts/show_rpi5_results.py
  python scripts/show_rpi5_results.py --csv results/rpi5_comparison.csv
"""

import argparse
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

# Map result directories + method labels
RESULT_SOURCES = [
    (BASE / "results/baseline",          "Baseline (float32)",      None),
    (BASE / "results/ptq_cpu",           None,                      None),
    (BASE / "results/pruning",           None,                      None),
    (BASE / "results/wanda",             None,                      None),
    (BASE / "results/awp_cpu",           None,                      None),
    (BASE / "results/svd_cpu",           None,                      None),
    (BASE / "results/token_pruning_cpu", None,                      None),
]

METHOD_LABELS = {
    "int8__quanto":                  "INT8 (quanto)",
    "int4__quanto":                  "INT4 (quanto)",
    "sp20":                          "Magnitude Prune 20%",
    "sp40":                          "Magnitude Prune 40%",
    "wanda_sp20":                    "Wanda 20%",
    "wanda_sp40":                    "Wanda 40%",
    "awp_wandasp20_int8":            "AWP: Wanda-20% + INT8",
    "svd_r50":                       "SVD LowRank 50%",
    "svd_r30":                       "SVD LowRank 30%",
    "tokenpruning_k50":              "Visual Token 50%",
    "tokenpruning_k25":              "Visual Token 25%",
}


def _label_from_file(path: Path) -> str:
    stem = path.stem  # e.g. HuggingFaceTB__SmolVLM-256M-Instruct__int8__quanto
    parts = stem.split("__")
    # Drop model prefix (first 2 parts for org/model)
    suffix = "__".join(parts[2:]) if len(parts) > 2 else parts[-1]
    return METHOD_LABELS.get(suffix, suffix)


def load_results():
    rows = []
    for result_dir, fixed_label, _ in RESULT_SOURCES:
        if not result_dir.exists():
            continue
        for f in sorted(result_dir.glob("*.json")):
            # Only SmolVLM-256M
            # Only sub-1B RPi5 models
            if not any(m in f.name for m in ["SmolVLM-256M", "SmolVLM-500M", "LFM2-VL-450M"]):
                continue
            with open(f) as fp:
                data = json.load(fp)

            label = fixed_label or _label_from_file(f)
            model_id = data.get("model_id", "").split("/")[-1]
            vqav2 = data.get("benchmarks", {}).get("vqav2", {})
            if not vqav2:
                continue

            rows.append({
                "Model":         model_id,
                "Method":        label,
                "VQAv2 Acc":     vqav2.get("accuracy", 0),
                "Latency (s)":   vqav2.get("avg_latency_s", 0),
                "Peak RAM (MB)": vqav2.get("peak_memory_mb", 0),
                "Throughput":    vqav2.get("throughput_sps", 0),
                "N Samples":     vqav2.get("n_samples", 0),
            })
    return rows


def print_table(rows, baseline_acc=None):
    if not rows:
        print("No results found.")
        return

    col_w = [28, 26, 10, 12, 13, 10]
    headers = ["Model", "Method", "VQAv2 Acc", "Latency (s)", "Peak RAM (MB)", "N Samples"]
    sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_w) + " |"

    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        acc = row["VQAv2 Acc"]
        delta = f"({acc - baseline_acc:+.3f})" if baseline_acc and row["Method"] != "Baseline (float32)" else ""
        acc_str = f"{acc:.4f} {delta}".strip()
        print(fmt.format(
            row["Model"][:col_w[0]],
            row["Method"][:col_w[1]],
            acc_str[:col_w[2]],
            f"{row['Latency (s)']:.2f}",
            f"{row['Peak RAM (MB)']:.0f}",
            str(row["N Samples"]),
        ))
    print(sep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Save results to CSV")
    args = parser.parse_args()

    rows = load_results()
    # Put baseline first
    rows.sort(key=lambda r: (r["Model"], 0 if r["Method"] == "Baseline (float32)" else 1, r["Method"]))

    baseline_row = next((r for r in rows if r["Method"] == "Baseline (float32)"), None)
    baseline_acc = baseline_row["VQAv2 Acc"] if baseline_row else None

    print(f"\n{'='*80}")
    print("  RPi5 Baseline Viability — Sub-1B VLMs (float32, CPU)")
    print(f"{'='*80}\n")
    print_table(rows, baseline_acc)

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved to {args.csv}")


if __name__ == "__main__":
    main()
