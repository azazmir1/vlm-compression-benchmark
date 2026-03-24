#!/bin/bash
# Run all 5 untested methods against SmolVLM-256M-Instruct
# Date: 2026-03-23

MODEL="HuggingFaceTB/SmolVLM-256M-Instruct"
BASE="/home/cselab/vlm-compression-benchmark"
LOGDIR="$BASE/results/untested_256m_logs"
mkdir -p "$LOGDIR"

echo "============================================"
echo "Running 5 untested methods on $MODEL"
echo "Device: Jetson Orin Nano 8GB"
echo "Started: $(date)"
echo "============================================"

# 1. SparseGPT (pure PyTorch - best chance of working)
echo ""
echo ">>> [1/5] SparseGPT sp50..."
python3 "$BASE/compression/pruning/run_sparsegpt.py" \
    --model_id "$MODEL" --sparsity 0.50 --n_calib 32 --vqav2_n 30 \
    2>&1 | tee "$LOGDIR/sparsegpt_sp50.log"
echo ">>> SparseGPT exit code: $?"

# 2. SLIM (pure PyTorch)
echo ""
echo ">>> [2/5] SLIM..."
python3 "$BASE/compression/casp_slim/run_casp_slim.py" \
    --model_id "$MODEL" --method slim --vqav2_n 30 \
    2>&1 | tee "$LOGDIR/slim.log"
echo ">>> SLIM exit code: $?"

# 3. AWP (Wanda + INT4 - may fail due to BnB on Jetson)
echo ""
echo ">>> [3/5] AWP sp50..."
python3 "$BASE/compression/combined/run_awp.py" \
    --model_id "$MODEL" --sparsity 0.50 --vqav2_n 30 \
    2>&1 | tee "$LOGDIR/awp_sp50.log"
echo ">>> AWP exit code: $?"

# 4. AWQ (may fail - autoawq not installed)
echo ""
echo ">>> [4/5] AWQ..."
python3 "$BASE/compression/ptq/run_awq.py" \
    --model_id "$MODEL" --vqav2_n 30 \
    2>&1 | tee "$LOGDIR/awq.log"
echo ">>> AWQ exit code: $?"

# 5. GPTQ (may fail - auto-gptq not installed)
echo ""
echo ">>> [5/5] GPTQ..."
python3 "$BASE/compression/ptq/run_gptq.py" \
    --model_id "$MODEL" --vqav2_n 30 \
    2>&1 | tee "$LOGDIR/gptq.log"
echo ">>> GPTQ exit code: $?"

echo ""
echo "============================================"
echo "All 5 methods complete. Finished: $(date)"
echo "Logs saved to: $LOGDIR/"
echo "============================================"
