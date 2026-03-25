#!/bin/bash
# Task 1: Run GPTQ via pre-quantized SmolVLM-256M-GPTQ-Int4
# Task 2: Re-run SLIM with gentler settings + multi-metric eval
# Date: 2026-03-23

BASE="/home/cselab/vlm-compression-benchmark"
LOGDIR="$BASE/results/untested_256m_logs"
mkdir -p "$LOGDIR"

echo "============================================"
echo "Running GPTQ (pre-quantized) + SLIM (gentle)"
echo "Started: $(date)"
echo "============================================"

# ── Task 1: GPTQ via pre-quantized model ──
echo ""
echo ">>> [1/3] GPTQ: Pre-quantized SmolVLM-256M-GPTQ-Int4..."
python3 "$BASE/compression/quantized_pretrained/run_quantized_pretrained.py" \
    --quantized_model_id vasanth0475/SmolVLM-256M-Instruct-GPTQ-Int4 \
    --base_model_id HuggingFaceTB/SmolVLM-256M-Instruct \
    --quant_method gptq --quant_bits 4 \
    --vqav2_n 30 --force \
    2>&1 | tee "$LOGDIR/gptq_prequantized_256m.log"
echo ">>> GPTQ pre-quantized exit code: $?"

# ── Task 2: SLIM with gentle settings (sp=0.20, rank_ratio=0.10) ──
echo ""
echo ">>> [2/3] SLIM gentle (sp=0.20, rank_ratio=0.10)..."
python3 "$BASE/compression/casp_slim/run_casp_slim.py" \
    --model_id HuggingFaceTB/SmolVLM-256M-Instruct \
    --method slim --sparsity 0.20 --rank_ratio 0.10 \
    --vqav2_n 30 --force \
    2>&1 | tee "$LOGDIR/slim_gentle_sp20_r10.log"
echo ">>> SLIM gentle exit code: $?"

# ── Task 3: SLIM with moderate settings (sp=0.30, rank_ratio=0.20) ──
echo ""
echo ">>> [3/3] SLIM moderate (sp=0.30, rank_ratio=0.20)..."
python3 "$BASE/compression/casp_slim/run_casp_slim.py" \
    --model_id HuggingFaceTB/SmolVLM-256M-Instruct \
    --method slim --sparsity 0.30 --rank_ratio 0.20 \
    --vqav2_n 30 --force \
    2>&1 | tee "$LOGDIR/slim_moderate_sp30_r20.log"
echo ">>> SLIM moderate exit code: $?"

echo ""
echo "============================================"
echo "All done. Finished: $(date)"
echo "Logs: $LOGDIR/"
echo "============================================"
