#!/usr/bin/env python3
"""
Find the FP16 ceiling model for each VLM family on Jetson.

For each family, tries models smallest→largest in isolated subprocesses.
A model "passes" if it can: (1) load in FP16, and (2) complete 1 inference.

Usage:
    python3 scripts/find_ceilings.py
    python3 scripts/find_ceilings.py --families smolvlm,fastvlm
    python3 scripts/find_ceilings.py --timeout 300
"""
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ── Models per family, ordered smallest → largest ────────────────────────────

FAMILY_MODELS = {
    "smolvlm": [
        ("HuggingFaceTB/SmolVLM-256M-Instruct", 256),
        ("HuggingFaceTB/SmolVLM-500M-Instruct", 508),
        ("HuggingFaceTB/SmolVLM-Instruct", 2200),
    ],
    "nanovlm": [
        ("lusxvr/nanoVLM-222M", 222),
        ("lusxvr/nanoVLM-450M", 450),
    ],
    "lfm2vl": [
        ("LiquidAI/LFM2-VL-450M", 451),
        ("LiquidAI/LFM2-VL-1.6B", 1585),
        ("LiquidAI/LFM2-VL-3B", 3000),
    ],
    "moondream": [
        ("vikhyatk/moondream2", 1927),
    ],
    "fastvlm": [
        ("apple/FastVLM-0.5B", 759),
        ("apple/FastVLM-1.5B", 1500),
        ("apple/FastVLM-7B", 7000),
    ],
    "qwen25vl": [
        ("Qwen/Qwen2.5-VL-3B-Instruct", 3800),
        ("Qwen/Qwen2.5-VL-7B-Instruct", 8300),
    ],
    "internvl25": [
        ("OpenGVLab/InternVL2_5-1B", 938),
        ("OpenGVLab/InternVL2_5-2B", 2200),
        ("OpenGVLab/InternVL2_5-4B", 3700),
        ("OpenGVLab/InternVL2_5-8B", 8100),
    ],
    "gemma3": [
        ("google/gemma-3-4b-it", 4300),
        ("google/gemma-3-12b-it", 12200),
    ],
    "ovis2": [
        ("AIDC-AI/Ovis2-1B", 1258),
        ("AIDC-AI/Ovis2-2B", 2200),
        ("AIDC-AI/Ovis2-4B", 4300),
        ("AIDC-AI/Ovis2-8B", 8900),
    ],
}


def _get_available_memory_mb():
    """Available system memory from /proc/meminfo."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except (IOError, ValueError):
        pass
    return 0.0


def _try_load_and_infer(model_id, family, result_path):
    """Subprocess: load model in FP16, run 1 inference, write result to file."""
    import gc
    import traceback
    import torch
    sys.path.insert(0, str(PROJECT_ROOT))

    # Make ourselves OOM-killable (not sshd)
    try:
        with open(f"/proc/{os.getpid()}/oom_score_adj", "w") as f:
            f.write("900\n")
    except Exception:
        pass

    result = {
        "model_id": model_id,
        "family": family,
        "status": "ERROR",
        "gpu_mem_before_mb": 0,
        "gpu_mem_after_mb": 0,
        "gpu_mem_delta_mb": 0,
        "gpu_mem_peak_mb": 0,
        "num_params_M": 0,
        "latency_s": 0,
        "prediction": "",
        "error": None,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Step 1: Load model ──────────────────────────────────────────────
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            free_before, total = torch.cuda.mem_get_info(0)
            used_before = (total - free_before) / 1024**2
        else:
            used_before = 0
        result["gpu_mem_before_mb"] = round(used_before, 1)

        from models.model_loader import load_model, unload_model

        print(f"  Loading {model_id} ...", flush=True)
        model, processor, meta = load_model(model_id, quant=None, family=family)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            free_after, total = torch.cuda.mem_get_info(0)
            used_after = (total - free_after) / 1024**2
        else:
            used_after = 0

        result["gpu_mem_after_mb"] = round(used_after, 1)
        result["gpu_mem_delta_mb"] = round(used_after - used_before, 1)

        if hasattr(model, 'parameters'):
            n_params = sum(p.numel() for p in model.parameters()) / 1e6
            result["num_params_M"] = round(n_params, 1)

        print(f"  Loaded — mem_delta={result['gpu_mem_delta_mb']:.0f}MB, "
              f"total_used={used_after:.0f}MB, params={result['num_params_M']}M",
              flush=True)

        # Post-load check: is system memory critically low?
        avail = _get_available_memory_mb()
        if avail < 700:
            print(f"  MEM_CRITICAL — only {avail:.0f}MB free after load", flush=True)
            result["status"] = "MEM_CRITICAL"
            result["error"] = f"Only {avail:.0f} MB free after load"
            try:
                unload_model(model)
            except Exception:
                pass
            with open(result_path, "w") as f:
                json.dump(result, f)
            return

    except torch.cuda.OutOfMemoryError as e:
        result["status"] = "OOM_LOAD"
        result["error"] = str(e)[:300]
        print(f"  OOM_LOAD — {str(e)[:100]}", flush=True)
        with open(result_path, "w") as f:
            json.dump(result, f)
        return
    except MemoryError as e:
        result["status"] = "OOM_LOAD"
        result["error"] = str(e)[:300]
        print(f"  OOM_LOAD (MemoryError)", flush=True)
        with open(result_path, "w") as f:
            json.dump(result, f)
        return
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            result["status"] = "OOM_LOAD"
        else:
            result["status"] = "ERROR"
        result["error"] = str(e)[:500]
        print(f"  {result['status']} — {str(e)[:150]}", flush=True)
        with open(result_path, "w") as f:
            json.dump(result, f)
        return
    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = traceback.format_exc()[-500:]
        print(f"  ERROR during load — {str(e)[:150]}", flush=True)
        with open(result_path, "w") as f:
            json.dump(result, f)
        return

    # ── Step 2: Run 1 inference ─────────────────────────────────────────
    try:
        from evaluation.run_baseline import load_vqav2, run_inference

        print(f"  Loading 1 VQAv2 sample ...", flush=True)
        samples = load_vqav2(n_samples=1)
        sample = samples[0]

        print(f"  Running inference ...", flush=True)
        t0 = time.perf_counter()
        pred = run_inference(model, processor, sample, family, device,
                             max_new_tokens=30)
        latency = time.perf_counter() - t0

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            free_peak, total = torch.cuda.mem_get_info(0)
            peak_used = (total - free_peak) / 1024**2
            result["gpu_mem_peak_mb"] = round(peak_used, 1)

        result["latency_s"] = round(latency, 3)
        result["prediction"] = pred.strip()[:200]
        result["status"] = "PASS"

        print(f"  PASS — latency={latency:.2f}s, peak_mem={result['gpu_mem_peak_mb']}MB",
              flush=True)
        print(f"  Prediction: {pred.strip()[:80]!r}", flush=True)

    except torch.cuda.OutOfMemoryError as e:
        result["status"] = "OOM_INFER"
        result["error"] = str(e)[:300]
        print(f"  OOM_INFER — loaded but OOM during inference", flush=True)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            result["status"] = "OOM_INFER"
        else:
            result["status"] = "ERROR"
        result["error"] = str(e)[:500]
        print(f"  {result['status']} during inference — {str(e)[:150]}", flush=True)
    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = traceback.format_exc()[-500:]
        print(f"  ERROR during inference — {str(e)[:150]}", flush=True)

    # Cleanup
    try:
        unload_model(model)
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with open(result_path, "w") as f:
        json.dump(result, f)


def _verify_clean_state():
    """Check system memory is in a reasonable state."""
    avail = _get_available_memory_mb()
    print(f"  [cleanup] System memory available: {avail:.0f} MB")
    if avail < 3000:
        print(f"  [cleanup] Low memory — waiting 10s for reclamation...")
        time.sleep(10)
        avail = _get_available_memory_mb()
        print(f"  [cleanup] After wait: {avail:.0f} MB")


def run_ceiling_scan(families=None, timeout=300):
    print("=" * 60)
    print("  CEILING SCAN — FP16 on Jetson Orin Nano 8GB")
    print(f"  Timeout: {timeout}s per model")
    print("=" * 60)

    if families is None:
        families = list(FAMILY_MODELS.keys())

    print(f"\nFamilies to scan: {', '.join(families)}")
    print(f"\nChecking initial state...")
    _verify_clean_state()

    all_results = []
    ceilings = {}   # family -> best PASS result
    ctx = mp.get_context("spawn")

    for family in families:
        if family not in FAMILY_MODELS:
            print(f"\nWARNING: Unknown family '{family}' — skipping")
            continue

        models = FAMILY_MODELS[family]
        print(f"\n{'='*60}")
        print(f"FAMILY: {family} ({len(models)} models, smallest→largest)")
        print(f"{'='*60}")

        family_ceiling = None
        hit_failure = False

        for model_id, approx_params in models:
            if hit_failure:
                print(f"\n  [SKIP] {model_id} — larger model already failed")
                all_results.append({
                    "model_id": model_id, "family": family,
                    "approx_params_M": approx_params,
                    "status": "SKIPPED",
                })
                continue

            avail = _get_available_memory_mb()
            print(f"\n{'—'*50}")
            print(f"  Testing: {model_id} (~{approx_params}M params)")
            print(f"  Available memory: {avail:.0f} MB")
            print(f"{'—'*50}")

            # Run in isolated subprocess
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
                result_path = tf.name

            proc = ctx.Process(target=_try_load_and_infer,
                               args=(model_id, family, result_path))
            proc.start()
            proc.join(timeout=timeout)

            if proc.is_alive():
                proc.terminate()
                proc.join(10)
                if proc.is_alive():
                    proc.kill()
                    proc.join(5)
                result = {
                    "model_id": model_id, "family": family,
                    "approx_params_M": approx_params,
                    "status": "TIMEOUT",
                    "error": f"Process timed out after {timeout}s",
                }
                print(f"  TIMEOUT — killed after {timeout}s")
            elif os.path.exists(result_path) and os.path.getsize(result_path) > 0:
                with open(result_path) as f:
                    result = json.load(f)
                result["approx_params_M"] = approx_params
            else:
                exit_code = proc.exitcode
                status = "OOM_LOAD" if exit_code == -9 else "CRASH"
                result = {
                    "model_id": model_id, "family": family,
                    "approx_params_M": approx_params,
                    "status": status,
                    "error": f"Process exited with code {exit_code}"
                             + (" (SIGKILL — likely OOM killed)" if exit_code == -9 else ""),
                }
                print(f"  {status} — subprocess died (exit code {exit_code})")

            all_results.append(result)

            # Update ceiling if PASS
            if result["status"] == "PASS":
                family_ceiling = result
            else:
                # Any failure means bigger models won't fit either
                hit_failure = True

            # Cleanup
            try:
                os.unlink(result_path)
            except OSError:
                pass
            if proc.is_alive():
                proc.kill()
                proc.join(5)
            try:
                proc.close()
            except Exception:
                pass

            # Wait for memory reclamation
            time.sleep(5)
            _verify_clean_state()

        ceilings[family] = family_ceiling

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"CEILING SUMMARY (per family)")
    print(f"{'='*60}")

    for family in families:
        if family not in ceilings:
            continue
        c = ceilings[family]
        if c:
            params = c.get("num_params_M", c.get("approx_params_M", "?"))
            mem = c.get("gpu_mem_delta_mb", "?")
            lat = c.get("latency_s", "?")
            print(f"  {family:12s} | ceiling: {c['model_id']:40s} | {params}M | {mem}MB | {lat}s")
        else:
            print(f"  {family:12s} | NO CEILING (all models failed)")

    print(f"\nAll model results:")
    for r in all_results:
        status = r["status"]
        params = r.get("num_params_M", r.get("approx_params_M", "?"))
        mem = r.get("gpu_mem_delta_mb", "?")
        lat = r.get("latency_s", "?")
        err = (r.get("error") or "")[:80]
        print(f"  {r['family']:12s} {r['model_id']:45s} {status:15s} "
              f"params={params}M  mem={mem}MB  lat={lat}s  {err}")

    print(f"\nCategory 1 — Cannot Load (need compression to fit):")
    for r in all_results:
        if r["status"] in ("OOM_LOAD", "MEM_CRITICAL", "CRASH"):
            print(f"  {r['model_id']}")

    print(f"\nCategory 2 — Loads but Problematic:")
    for r in all_results:
        if r["status"] == "PASS" and r.get("latency_s", 0) > 3.0:
            print(f"  {r['model_id']} (slow: {r['latency_s']}s)")
        elif r["status"] in ("OOM_INFER", "TIMEOUT"):
            print(f"  {r['model_id']} ({r['status']})")

    # ── Save ────────────────────────────────────────────────────────────
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "available_memory_mb": round(_get_available_memory_mb()),
        "ceilings": {fam: (c if c else None) for fam, c in ceilings.items()},
        "all_results": all_results,
    }
    out_path = PROJECT_ROOT / "results" / "jetson" / "ceiling_scan_v2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", type=str, default=None,
                        help="Comma-separated family names (default: all)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout per model in seconds (default: 300)")
    args = parser.parse_args()

    families = None
    if args.families:
        families = [f.strip() for f in args.families.split(",")]

    mp.set_start_method("spawn", force=True)
    run_ceiling_scan(families=families, timeout=args.timeout)
