#!/usr/bin/env bash
# =============================================================================
# run_all_baselines_rpi5.sh
# Run float32 baseline on all RPi5-candidate models.
# Records pass/fail/OOM for each model.
#
# Models tested:
#   PASS/FAIL expected:
#     SmolVLM-256M    (~1.0 GB) → should PASS
#     SmolVLM-500M    (~2.0 GB) → should PASS
#     LFM2-VL-450M    (~1.8 GB) → should PASS
#     moondream2      (~7.5 GB) → likely FAIL (OOM on 8GB RPi5)
#     SmolVLM-2.2B    (~8.8 GB) → likely FAIL (OOM)
#
#   Known broken (skip):
#     nanoVLM-*       → HF repo missing model_type in config.json
#     FastVLM-*       → HF repo missing preprocessor_config.json
#     Florence-2-*    → Not a conversational VLM, no VQA chat template
#
# Usage:
#   bash scripts/run_all_baselines_rpi5.sh
#   VQAV2_N=5 bash scripts/run_all_baselines_rpi5.sh   # faster with 5 samples
# =============================================================================

set -e
cd "$(dirname "$0")/.."

N="${VQAV2_N:-10}"
REPORT="results/baseline_rpi5_report.txt"
mkdir -p results

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$REPORT"; }

# Wipe old report
echo "RPi5 Baseline Model Viability Report" > "$REPORT"
echo "VQAv2 samples per model: $N" >> "$REPORT"
echo "Generated: $(date)" >> "$REPORT"
echo "======================================" >> "$REPORT"

declare -A RESULTS

try_baseline() {
    local model_id="$1"
    local safe_name="${model_id//\//__}"
    local out_path="results/baseline/${safe_name}.json"

    log "--- Testing: $model_id ---"

    if [ -f "$out_path" ]; then
        log "SKIP (already done): $model_id"
        RESULTS["$model_id"]="DONE (cached)"
        return
    fi

    # Run with a memory guard — if RAM exceeds ~7GB the Pi will OOM-kill it
    timeout 3600 python evaluation/run_baseline.py \
        --model_id "$model_id" \
        --vqav2_n "$N" \
        --skip_textvqa --skip_pope \
        2>&1 | tee -a "results/baseline_${safe_name}_run.log"

    local exit_code=${PIPESTATUS[0]}

    if [ "$exit_code" -eq 0 ]; then
        # Extract accuracy and RAM from saved JSON
        local acc ram lat
        acc=$(python -c "import json; d=json.load(open('$out_path')); print(d['benchmarks']['vqav2']['accuracy'])" 2>/dev/null || echo "N/A")
        ram=$(python -c "import json; d=json.load(open('$out_path')); print(d['benchmarks']['vqav2']['peak_memory_mb'])" 2>/dev/null || echo "N/A")
        lat=$(python -c "import json; d=json.load(open('$out_path')); print(d['benchmarks']['vqav2']['avg_latency_s'])" 2>/dev/null || echo "N/A")
        log "PASS: $model_id | acc=$acc | peak_ram=${ram}MB | lat=${lat}s"
        RESULTS["$model_id"]="PASS | acc=$acc | ram=${ram}MB | lat=${lat}s"
    elif [ "$exit_code" -eq 124 ]; then
        log "FAIL (timeout >1h): $model_id"
        RESULTS["$model_id"]="FAIL (timeout)"
    else
        # Check if OOM
        local oom_msg
        oom_msg=$(grep -i "killed\|memory\|oom\|cannot allocate\|RuntimeError" \
            "results/baseline_${safe_name}_run.log" 2>/dev/null | tail -2 || echo "")
        if [ -n "$oom_msg" ]; then
            log "FAIL (OOM): $model_id"
            RESULTS["$model_id"]="FAIL (OOM)"
        else
            log "FAIL (error, exit=$exit_code): $model_id"
            RESULTS["$model_id"]="FAIL (error $exit_code)"
        fi
    fi
    echo "" | tee -a "$REPORT"
}

# ── Models to test (<1B params only) ──────────────────────────────────────────

try_baseline "HuggingFaceTB/SmolVLM-256M-Instruct"   # 256M
try_baseline "HuggingFaceTB/SmolVLM-500M-Instruct"   # 507M
try_baseline "LiquidAI/LFM2-VL-450M"                  # 450M

# ── Summary ────────────────────────────────────────────────────────────────────
echo "" | tee -a "$REPORT"
log "========================================"
log "SUMMARY"
log "========================================"
for model in "${!RESULTS[@]}"; do
    log "  $(printf '%-45s' $model) ${RESULTS[$model]}"
done
log ""
log "Skipped (>1B params — too slow on CPU without compression):"
log "  vikhyatk/moondream2        → 1.9B params"
log "  HuggingFaceTB/SmolVLM-2.2B → 2.2B params"
log ""
log "Known skipped (broken HF repos):"
log "  nanoVLM-*  → missing model_type in config.json"
log "  FastVLM-*  → missing preprocessor_config.json"
log "  Florence-2 → not a conversational VLM"
log ""
log "Report saved to: $REPORT"
