#!/usr/bin/env bash
# scripts/run_all_compressions.sh
# ================================
# Run ALL compression methods on ALL models, smallest → largest.
# Each method runs as a separate subprocess for OOM crash isolation.
#
# Methods:
#   1. BnB INT8          (2x compression)
#   2. BnB INT4          (4x compression)
#   3. AWQ INT4          (4x, activation-aware)
#   4. GPTQ INT4         (4x, Hessian-based)
#   5. SVD-LLM 50%       (low-rank decomposition)
#   6. PALU 25%          (KV-cache compression)
#
# Usage:
#   tmux new -s bench
#   bash scripts/run_all_compressions.sh
#   # Detach: Ctrl+B, D     Reattach: tmux attach -t bench

# Do NOT use set -e: core dumps from BnB/CUDA crashes kill the whole script.
# Each method already has || error handling.
cd "$(dirname "$0")/.."

N_SAMPLES=1000
TIMEOUT=1800   # 30 min max per method per model
LOG=results/compression_run.log
mkdir -p results/{ptq,awq,gptq,svd_llm,palu} logs

# ── Jetson detection ──────────────────────────────────────────────────────
IS_JETSON=false
if [ -f /proc/device-tree/model ] && grep -qi "jetson\|orin" /proc/device-tree/model 2>/dev/null; then
    IS_JETSON=true
elif [ "$(uname -m)" = "aarch64" ] && [ -d /usr/local/cuda ]; then
    IS_JETSON=true
fi

if $IS_JETSON; then
    echo "=== Detected Jetson (aarch64) — BnB INT8/INT4 will be SKIPPED ===" | tee -a "$LOG"
fi

# ── Memory guardrail function ─────────────────────────────────────────────
check_memory() {
    local param_m=$1
    local method=$2
    local mem_avail
    mem_avail=$(grep MemAvailable /proc/meminfo | awk '{printf "%.0f", $2/1024}')
    # Estimate: FP16 needs ~2*params MB, INT8 ~1*params, INT4 ~0.5*params + overhead
    local est_mb
    case "$method" in
        int4) est_mb=$(echo "$param_m * 0.5 + 500" | bc -l | cut -d. -f1) ;;
        int8) est_mb=$(echo "$param_m * 1.0 + 500" | bc -l | cut -d. -f1) ;;
        *)    est_mb=$(echo "$param_m * 2.0 + 800" | bc -l | cut -d. -f1) ;;
    esac
    if [ "$est_mb" -gt "$((mem_avail * 85 / 100))" ]; then
        echo "  SKIP: estimated ${est_mb}MB > 85% of available ${mem_avail}MB" | tee -a "$LOG"
        return 1
    fi
    return 0
}

echo "============================================================" | tee "$LOG"
echo "=== Compression Benchmark Started: $(date) ===" | tee -a "$LOG"
echo "=== N_SAMPLES=$N_SAMPLES ===" | tee -a "$LOG"
echo "=== Memory: $(grep MemAvailable /proc/meminfo | awk '{printf "%.0f MB", $2/1024}') ===" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

# Models sorted smallest → largest (from baseline results)
MODELS=(
    "HuggingFaceTB/SmolVLM-256M-Instruct:smolvlm:257"
    "LiquidAI/LFM2-VL-450M:lfm2vl:451"
    "HuggingFaceTB/SmolVLM-500M-Instruct:smolvlm:508"
    "OpenGVLab/InternVL2_5-1B:internvl25:938"
    "AIDC-AI/Ovis2-1B:ovis2:1131"
    "LiquidAI/LFM2-VL-1.6B:lfm2vl:1585"
    "vikhyatk/moondream2:moondream:1927"
    "OpenGVLab/InternVL2_5-2B:internvl25:2206"
    "AIDC-AI/Ovis2-2B:ovis2:2223"
    "HuggingFaceTB/SmolVLM-Instruct:smolvlm:2246"
    "LiquidAI/LFM2-VL-3B:lfm2vl:2999"
    "OpenGVLab/InternVL2_5-4B:internvl25:3713"
    "Qwen/Qwen2.5-VL-3B-Instruct:qwen25vl:3755"
    "google/gemma-3-4b-it:gemma3:4300"
    "AIDC-AI/Ovis2-4B:ovis2:4305"
    "OpenGVLab/InternVL2_5-8B:internvl25:8075"
    "Qwen/Qwen2.5-VL-7B-Instruct:qwen25vl:8292"
    "AIDC-AI/Ovis2-8B:ovis2:8935"
    "google/gemma-3-12b-it:gemma3:12187"
)

TOTAL_MODELS=${#MODELS[@]}
MODEL_IDX=0

for entry in "${MODELS[@]}"; do
    IFS=':' read -r MODEL_ID FAMILY PARAM_M <<< "$entry"
    MODEL_IDX=$((MODEL_IDX + 1))
    SHORT=$(echo "$MODEL_ID" | sed 's|.*/||')

    echo "" | tee -a "$LOG"
    echo "╔════════════════════════════════════════════════════════════╗" | tee -a "$LOG"
    echo "║ [$MODEL_IDX/$TOTAL_MODELS] $SHORT (${PARAM_M}M) — $FAMILY" | tee -a "$LOG"
    echo "╚════════════════════════════════════════════════════════════╝" | tee -a "$LOG"

    MEM_AVAIL=$(grep MemAvailable /proc/meminfo | awk '{printf "%.0f", $2/1024}')
    echo "  Memory available: ${MEM_AVAIL} MB" | tee -a "$LOG"

    # ── Method 1: BnB INT8 ──────────────────────────────────────────
    echo "" | tee -a "$LOG"
    if $IS_JETSON; then
        echo "  ── [1/6] BnB INT8 — SKIPPED (BnB crashes on Jetson aarch64) ──" | tee -a "$LOG"
    elif ! check_memory "$PARAM_M" "int8"; then
        echo "  ── [1/6] BnB INT8 — SKIPPED (insufficient memory) ──" | tee -a "$LOG"
    else
        echo "  ── [1/6] BnB INT8 ──" | tee -a "$LOG"
        timeout $TIMEOUT python3 compression/ptq/run_ptq.py \
            --model_id "$MODEL_ID" \
            --quant int8 --backend bnb \
            --vqav2_n $N_SAMPLES \
            --skip_textvqa --skip_pope \
            --force \
            2>&1 | tee -a "$LOG" || echo "  ⚠ BnB INT8 FAILED (exit=$?)" | tee -a "$LOG"
    fi

    # GPU cleanup between methods
    sleep 3

    # ── Method 2: BnB INT4 ──────────────────────────────────────────
    echo "" | tee -a "$LOG"
    if $IS_JETSON; then
        echo "  ── [2/6] BnB INT4 — SKIPPED (BnB crashes on Jetson aarch64) ──" | tee -a "$LOG"
    elif ! check_memory "$PARAM_M" "int4"; then
        echo "  ── [2/6] BnB INT4 — SKIPPED (insufficient memory) ──" | tee -a "$LOG"
    else
        echo "  ── [2/6] BnB INT4 ──" | tee -a "$LOG"
        timeout $TIMEOUT python3 compression/ptq/run_ptq.py \
            --model_id "$MODEL_ID" \
            --quant int4 --backend bnb \
            --vqav2_n $N_SAMPLES \
            --skip_textvqa --skip_pope \
            --force \
            2>&1 | tee -a "$LOG" || echo "  ⚠ BnB INT4 FAILED (exit=$?)" | tee -a "$LOG"
    fi

    sleep 3

    # ── Method 3: AWQ INT4 ──────────────────────────────────────────
    echo "" | tee -a "$LOG"
    if ! check_memory "$PARAM_M" "int4"; then
        echo "  ── [3/6] AWQ INT4 — SKIPPED (insufficient memory) ──" | tee -a "$LOG"
    else
        echo "  ── [3/6] AWQ INT4 ──" | tee -a "$LOG"
        timeout $TIMEOUT python3 compression/ptq/run_awq.py \
            --model_id "$MODEL_ID" \
            --w_bit 4 \
            --vqav2_n $N_SAMPLES \
            --force \
            2>&1 | tee -a "$LOG" || echo "  ⚠ AWQ INT4 FAILED (exit=$?)" | tee -a "$LOG"
    fi

    sleep 3

    # ── Method 4: GPTQ INT4 ─────────────────────────────────────────
    echo "" | tee -a "$LOG"
    if ! check_memory "$PARAM_M" "int4"; then
        echo "  ── [4/6] GPTQ INT4 — SKIPPED (insufficient memory) ──" | tee -a "$LOG"
    else
        echo "  ── [4/6] GPTQ INT4 ──" | tee -a "$LOG"
        timeout $TIMEOUT python3 compression/ptq/run_gptq.py \
            --model_id "$MODEL_ID" \
            --bits 4 \
            --vqav2_n $N_SAMPLES \
            --force \
            2>&1 | tee -a "$LOG" || echo "  ⚠ GPTQ INT4 FAILED (exit=$?)" | tee -a "$LOG"
    fi

    sleep 3

    # ── Method 5: SVD-LLM (rank_ratio=0.50) ─────────────────────────
    echo "" | tee -a "$LOG"
    if ! check_memory "$PARAM_M" "fp16"; then
        echo "  ── [5/6] SVD-LLM 50% — SKIPPED (insufficient memory) ──" | tee -a "$LOG"
    else
        echo "  ── [5/6] SVD-LLM 50% ──" | tee -a "$LOG"
        timeout $TIMEOUT python3 compression/lowrank/run_svd_llm.py \
            --model_id "$MODEL_ID" \
            --rank_ratio 0.50 \
            --vqav2_n $N_SAMPLES \
            --force \
            2>&1 | tee -a "$LOG" || echo "  ⚠ SVD-LLM FAILED (exit=$?)" | tee -a "$LOG"
    fi

    sleep 3

    # ── Method 6: PALU (rank_ratio=0.25) ────────────────────────────
    echo "" | tee -a "$LOG"
    if ! check_memory "$PARAM_M" "fp16"; then
        echo "  ── [6/6] PALU 25% — SKIPPED (insufficient memory) ──" | tee -a "$LOG"
    else
        echo "  ── [6/6] PALU 25% ──" | tee -a "$LOG"
        timeout $TIMEOUT python3 compression/palu/run_palu.py \
            --model_id "$MODEL_ID" \
            --rank_ratio 0.25 \
            --vqav2_n $N_SAMPLES \
            --force \
            2>&1 | tee -a "$LOG" || echo "  ⚠ PALU FAILED (exit=$?)" | tee -a "$LOG"
    fi

    sleep 5

    # Post-model memory check
    MEM_AFTER=$(grep MemAvailable /proc/meminfo | awk '{printf "%.0f", $2/1024}')
    echo "" | tee -a "$LOG"
    echo "  Memory after $SHORT: ${MEM_AFTER} MB available" | tee -a "$LOG"

    if [ "$MEM_AFTER" -lt 2000 ]; then
        echo "  ⚠ Memory critically low — waiting 30s for OS reclaim..." | tee -a "$LOG"
        sleep 30
    fi
done

echo "" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "=== Compression Benchmark Complete: $(date) ===" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

# Print summary
echo "" | tee -a "$LOG"
echo "=== RESULTS SUMMARY ===" | tee -a "$LOG"
python3 -c "
import json, os

dirs = {
    'BnB INT8': 'results/ptq',
    'BnB INT4': 'results/ptq',
    'AWQ':      'results/awq',
    'GPTQ':     'results/gptq',
    'SVD-LLM':  'results/svd_llm',
    'PALU':     'results/palu',
}

print(f\"{'Model':<35} {'Method':<12} {'Acc':>8} {'Lat(s)':>8} {'Mem(MB)':>8} {'Ratio':>6}\")
print('─' * 82)

for d_name, d_path in sorted(set((v, v) for v in dirs.values())):
    if not os.path.isdir(d_path):
        continue
    for f in sorted(os.listdir(d_path)):
        if not f.endswith('.json'):
            continue
        with open(os.path.join(d_path, f)) as fh:
            d = json.load(fh)
        name = d.get('model_id', f).split('/')[-1][:33]
        method = d.get('quant', d.get('method', '?'))
        if 'backend' in d:
            method = f\"{d['quant']}_{d['backend']}\"
        bm = d.get('benchmarks', {}).get('vqav2', {})
        acc = f\"{bm['accuracy']:.4f}\" if 'accuracy' in bm else '-'
        lat = f\"{bm['avg_latency_s']:.2f}\" if 'avg_latency_s' in bm else '-'
        mem = f\"{d.get('gpu_mem_load_mb', 0):.0f}\" if d.get('gpu_mem_load_mb') else '-'
        ratio = f\"{d.get('compression_ratio', 0):.2f}\" if d.get('compression_ratio') else '-'
        print(f'{name:<35} {method:<12} {acc:>8} {lat:>8} {mem:>8} {ratio:>6}')
" 2>&1 | tee -a "$LOG"
