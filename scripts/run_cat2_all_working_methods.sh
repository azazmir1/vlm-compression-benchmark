#!/bin/bash
# Run all 3 working methods (Wanda, Magnitude, SLIM) on Jetson-loadable models
# Date: 2026-03-24

BASE="/home/cselab/vlm-compression-benchmark"
LOGDIR="$BASE/results/cat2_working_methods_logs"
mkdir -p "$LOGDIR"

N=30  # eval samples

echo "============================================"
echo "Category 2: All working methods on Jetson"
echo "Methods: Wanda (sp20), Magnitude (sp20), SLIM (sp20_r10)"
echo "Started: $(date)"
echo "============================================"

# Models that load on Jetson (from ceiling scan)
MODELS=(
    "HuggingFaceTB/SmolVLM-256M-Instruct"
    "HuggingFaceTB/SmolVLM-500M-Instruct"
    "LiquidAI/LFM2-VL-450M"
    "OpenGVLab/InternVL2_5-1B"
)

RUN=1
TOTAL=12  # 4 models x 3 methods

for MODEL in "${MODELS[@]}"; do
    SAFE=$(echo "$MODEL" | sed 's|/|__|g')

    # ── Wanda sp20 ──
    echo ""
    echo ">>> [$RUN/$TOTAL] Wanda sp20 on $MODEL..."
    python3 "$BASE/compression/pruning/run_wanda.py" \
        --model_id "$MODEL" --sparsity 0.2 --vqav2_n $N --force \
        2>&1 | tee "$LOGDIR/${SAFE}__wanda_sp20.log"
    echo ">>> Exit code: $?"
    RUN=$((RUN+1))

    # ── Magnitude sp20 ──
    echo ""
    echo ">>> [$RUN/$TOTAL] Magnitude sp20 on $MODEL..."
    python3 "$BASE/compression/pruning/run_pruning.py" \
        --model_id "$MODEL" --sparsity 0.2 --vqav2_n $N --force \
        2>&1 | tee "$LOGDIR/${SAFE}__magnitude_sp20.log"
    echo ">>> Exit code: $?"
    RUN=$((RUN+1))

    # ── SLIM sp20 r10 ──
    echo ""
    echo ">>> [$RUN/$TOTAL] SLIM sp20_r10 on $MODEL..."
    python3 "$BASE/compression/casp_slim/run_casp_slim.py" \
        --model_id "$MODEL" --method slim --sparsity 0.20 --rank_ratio 0.10 \
        --vqav2_n $N --force \
        2>&1 | tee "$LOGDIR/${SAFE}__slim_sp20_r10.log"
    echo ">>> Exit code: $?"
    RUN=$((RUN+1))
done

echo ""
echo "============================================"
echo "All $TOTAL runs complete. Finished: $(date)"
echo "Logs: $LOGDIR/"
echo "============================================"
