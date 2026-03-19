#!/usr/bin/env bash
# scripts/run_1day.sh
# 1-day execution plan on RTX A6000 — all GPU experiments in ~21 hours.
#
# Constraints vs full plan:
#   VQAv2   : 500 samples  (not 5000)
#   TextVQA : SKIPPED
#   POPE    : kept (~500 adversarial samples)
#   PTQ     : INT4 only    (not INT8+INT4)
#   Pruning : 20% + 40%    (unchanged — cheap with reduced samples)
#   QLoRA   : sub-3B models, rank 16 only, 2K train samples
#   ONNX    : sub-500M models only
#
# Usage:
#   bash scripts/run_1day.sh
#   bash scripts/run_1day.sh --skip-smoke

set -e
cd "$(dirname "$0")/.."

SKIP_SMOKE=false
for arg in "$@"; do
    [ "$arg" = "--skip-smoke" ] && SKIP_SMOKE=true
done

export VQAV2_N=1000
export SKIP_TEXTVQA=1
export QUANT_LEVELS="int4"

log() { echo ""; echo "##################################################"; echo "## $*"; echo "##################################################"; echo ""; }

if [ "$SKIP_SMOKE" = false ]; then
    log "STAGE 1 — Smoke Test"
    bash scripts/smoke_test.sh
fi

log "STAGE 2 — Baselines (FP16)"
bash scripts/run_all_baselines.sh

log "STAGE 3 — PTQ (INT4 only)"
bash scripts/run_all_ptq.sh

log "STAGE 4 — Pruning (20% + 40%)"
bash scripts/run_all_pruning.sh

log "STAGE 5 — ONNX Export + INT8 quantization"
bash scripts/run_all_onnx.sh

log "STAGE 6 — Deployability Report"
python analysis/deployability_report.py

log "1-DAY RUN COMPLETE — open notebooks/results_analysis.ipynb to analyse results"
