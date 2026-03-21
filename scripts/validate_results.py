#!/usr/bin/env python3
"""
scripts/validate_results.py
============================
Scan all result JSON files and flag invalid/suspicious results.

Checks:
  - accuracy = 0.0 (broken inference or too-aggressive compression)
  - n_evaluated = 0 or all_failed (inference never produced output)
  - Missing benchmarks (evaluation was skipped)
  - BnB results on Jetson (known to crash)

Usage:
    python3 scripts/validate_results.py
    python3 scripts/validate_results.py --delete-invalid  # remove broken results
"""

import argparse
import json
import glob
import os
import sys


def validate_results(results_root="results", delete_invalid=False):
    issues = []
    valid = 0
    total = 0

    for f in sorted(glob.glob(f"{results_root}/**/*.json", recursive=True)):
        # Skip non-result files
        if "ceiling_report" in f or "device_constraints" in f:
            continue

        total += 1
        try:
            d = json.load(open(f))
        except json.JSONDecodeError:
            issues.append((f, "CORRUPT", "Invalid JSON"))
            continue

        short = f.replace(results_root + "/", "")

        # Check status
        status = d.get("status", d.get("metrics", {}).get("status"))
        if status in ("ERROR", "OOM_LOAD", "OOM_INFER", "TIMEOUT", "MEM_CRITICAL"):
            # These are valid failure records — not flagged
            valid += 1
            continue

        # Check benchmarks — Jetson results use "metrics" key instead of "benchmarks"
        benchmarks = d.get("benchmarks", {})
        metrics = d.get("metrics", {})
        if not benchmarks and metrics and "accuracy" in metrics:
            # Jetson format: convert metrics to benchmarks-like structure
            benchmarks = {"vqav2": metrics}
        if not benchmarks:
            issues.append((short, "NO_BENCHMARKS", "No benchmark results"))
            continue

        for bench_name, bench_data in benchmarks.items():
            acc = bench_data.get("accuracy")
            lat = bench_data.get("avg_latency_s", 0)
            n_eval = bench_data.get("n_evaluated", bench_data.get("n_samples", 0))
            all_failed = bench_data.get("all_failed", False)

            if all_failed or (acc == 0.0 and lat == 0.0):
                issues.append((
                    short, "ALL_FAILED",
                    f"{bench_name}: all samples failed (acc=0.0, lat=0.0) — "
                    f"inference is completely broken"
                ))
            elif acc is not None and acc == 0.0 and n_eval >= 5:
                issues.append((
                    short, "ZERO_ACCURACY",
                    f"{bench_name}: acc=0.0 on {n_eval} samples — "
                    f"model output is garbage (lat={lat:.2f}s)"
                ))
            else:
                valid += 1

    # Report
    print(f"\n{'='*70}")
    print(f"Result Validation: {total} files, {valid} valid, {len(issues)} issues")
    print(f"{'='*70}\n")

    if not issues:
        print("All results look valid.")
        return 0

    # Group by issue type
    by_type = {}
    for path, issue_type, msg in issues:
        by_type.setdefault(issue_type, []).append((path, msg))

    for issue_type, entries in sorted(by_type.items()):
        print(f"\n--- {issue_type} ({len(entries)} files) ---")
        for path, msg in entries:
            print(f"  {path}")
            print(f"    {msg}")

        if delete_invalid and issue_type in ("ALL_FAILED", "CORRUPT"):
            print(f"\n  Deleting {len(entries)} {issue_type} files...")
            for path, _ in entries:
                full_path = os.path.join(results_root, path) if not os.path.isabs(path) else path
                if os.path.exists(full_path):
                    os.remove(full_path)
                    print(f"    Deleted: {path}")

    print(f"\n{'='*70}")
    print("ZERO_ACCURACY files need investigation:")
    print("  - Too-aggressive compression (PALU rank_ratio, SVD-LLM rank_ratio)")
    print("  - Magnitude pruning above threshold (try Wanda instead)")
    print("ALL_FAILED files have broken inference:")
    print("  - BnB on Jetson (known crash, skip entirely)")
    print("  - Model/library incompatibility")
    print(f"{'='*70}")

    return len(issues)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete-invalid", action="store_true",
                        help="Delete ALL_FAILED and CORRUPT result files")
    args = parser.parse_args()
    sys.exit(0 if validate_results(delete_invalid=args.delete_invalid) == 0 else 1)
