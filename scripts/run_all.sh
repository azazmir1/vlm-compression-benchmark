#!/usr/bin/env bash
# scripts/run_all.sh
# Master orchestration script — runs the full benchmark pipeline end-to-end.
#
# Order:
#   1. Smoke test (validates pipeline on tiny samples)
#   2. Baselines  (fp16 / int8 / int4 depending on model size)
#   3. PTQ        (int8 + int4 via BitsAndBytes)
#   4. Pruning    (20% + 40% sparsity)
#   5. QLoRA      (rank 16 + rank 64)
#
# Usage:
#   bash scripts/run_all.sh                    # full pipeline
#   bash scripts/run_all.sh --skip-smoke       # skip smoke test
#   bash scripts/run_all.sh --only baselines   # run one stage only
#
# Env overrides:
#   VQAV2_N=500 bash scripts/run_all.sh        # use 500 VQAv2 samples (faster)

set -e
cd "$(dirname "$0")/.."

export PATH="/home/azaz/miniconda3/envs/vlm-bench/bin:$PATH"

SKIP_SMOKE=false
ONLY=""

for arg in "$@"; do
    case $arg in
        --skip-smoke)  SKIP_SMOKE=true ;;
        --only)        shift; ONLY="$1" ;;
    esac
done

export VQAV2_N="${VQAV2_N:-1000}"

log() { echo ""; echo "##################################################"; echo "## $*"; echo "##################################################"; echo ""; }

# ── Stage 1: Smoke test ───────────────────────────────────────────────────
if [ "${SKIP_SMOKE}" = false ] && [ -z "${ONLY}" -o "${ONLY}" = "smoke" ]; then
    log "STAGE 1 — Smoke Test"
    bash scripts/smoke_test.sh
fi

# ── Stage 2: Baselines ───────────────────────────────────────────────────
if [ -z "${ONLY}" ] || [ "${ONLY}" = "baselines" ]; then
    log "STAGE 2 — Baselines (fp16 / int8 / int4)"
    bash scripts/run_all_baselines.sh
fi

# ── Stage 3: PTQ ────────────────────────────────────────────────────────
if [ -z "${ONLY}" ] || [ "${ONLY}" = "ptq" ]; then
    log "STAGE 3 — PTQ (INT8 + INT4)"
    bash scripts/run_all_ptq.sh
fi

# ── Stage 4: Pruning ─────────────────────────────────────────────────────
if [ -z "${ONLY}" ] || [ "${ONLY}" = "pruning" ]; then
    log "STAGE 4 — Pruning (20% + 40%)"
    bash scripts/run_all_pruning.sh
fi

# ── Stage 5: Deployability Report ────────────────────────────────────────
if [ -z "${ONLY}" ] || [ "${ONLY}" = "report" ]; then
    log "STAGE 6 — Deployability Report"
    python analysis/deployability_report.py
fi

log "ALL STAGES COMPLETE — open notebooks/results_analysis.ipynb to analyse results"
