#!/bin/bash
# Run PyTorch INT4 quantization on all ceiling models (Jetson)
# Each model runs in a separate process to avoid GPU memory leaks
#
# Usage: bash scripts/run_int4_all_ceiling.sh
#        bash scripts/run_int4_all_ceiling.sh --force  # re-run existing

set -e
cd "$(dirname "$0")/.."

FORCE="${1:-}"
N_SAMPLES=30

MODELS=(
    "HuggingFaceTB/SmolVLM-256M-Instruct"
    "HuggingFaceTB/SmolVLM-500M-Instruct"
    "LiquidAI/LFM2-VL-450M"
    "LiquidAI/LFM2-VL-1.6B"
    "vikhyatk/moondream2"
    "OpenGVLab/InternVL2_5-1B"
    "OpenGVLab/InternVL2_5-2B"
    "Qwen/Qwen2.5-VL-3B-Instruct"
    "OpenGVLab/InternVL2_5-4B"
    "Qwen/Qwen2.5-VL-7B-Instruct"
)

TOTAL=${#MODELS[@]}
PASS=0
FAIL=0

echo "========================================"
echo "PyTorch INT4 — $TOTAL models, n=$N_SAMPLES"
echo "========================================"
echo ""

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    SHORT=$(echo "$MODEL" | cut -d'/' -f2)
    IDX=$((i + 1))

    echo "[$IDX/$TOTAL] $SHORT"

    CMD="python3 compression/ptq/run_pytorch_int4.py --model_id $MODEL --vqav2_n $N_SAMPLES"
    if [ "$FORCE" = "--force" ]; then
        CMD="$CMD --force"
    fi

    if $CMD 2>&1 | tee "results/jetson/pytorch_int4/${SHORT}.log"; then
        echo "  -> PASS"
        PASS=$((PASS + 1))
    else
        echo "  -> FAIL"
        FAIL=$((FAIL + 1))
    fi
    echo ""
done

echo "========================================"
echo "DONE: $PASS pass, $FAIL fail out of $TOTAL"
echo "========================================"
