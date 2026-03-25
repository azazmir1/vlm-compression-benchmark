#!/bin/bash
# ============================================================
# Full compression benchmark on Jetson Orin Nano 8GB
#
# Category 1: Models that CAN'T LOAD → memory compression
#   - Qwen2.5-VL-3B (OOM_LOAD) → pre-quantized AWQ/GPTQ
#   - Gemma3-4B (OOM_LOAD) → pre-quantized AWQ/GPTQ
#
# Category 2: Models that LOAD but are MEM_CRITICAL/slow → efficiency methods
#   - SmolVLM-2.2B (MEM_CRITICAL, 231MB free)
#   - LFM2-VL-3B (MEM_CRITICAL, 155MB free)
#   - FastVLM-1.5B (MEM_CRITICAL, 187MB free)
#   - InternVL2.5-2B (MEM_CRITICAL, 686MB free)
#   - FastVLM-0.5B (slow, 4.14s)
# ============================================================

set -e
cd /home/cselab/vlm-compression-benchmark

LOG="results/compression_run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "  COMPRESSION BENCHMARK — $(date)"
echo "  Log: $LOG"
echo "============================================================"

# Helper: wait for memory to stabilize between runs
wait_mem() {
    echo ""
    echo "--- Waiting 10s for memory reclamation ---"
    sleep 10
    free -m | head -2
    echo ""
}

# ============================================================
# CATEGORY 1: Pre-quantized models for OOM_LOAD targets
# ============================================================
echo ""
echo "============================================================"
echo "  CATEGORY 1: MEMORY COMPRESSION (make unloadable models load)"
echo "============================================================"

# ── Qwen2.5-VL-3B — AWQ (official, est ~2GB) ──
echo ""
echo ">>> [Cat1] Qwen2.5-VL-3B — AWQ pre-quantized"
python3 compression/quantized_pretrained/run_quantized_pretrained.py \
    --quantized_model_id "Qwen/Qwen2.5-VL-3B-Instruct-AWQ" \
    --base_model_id "Qwen/Qwen2.5-VL-3B-Instruct" \
    --quant_method awq --quant_bits 4 \
    --vqav2_n 10 \
    || echo "FAILED: Qwen2.5-VL-3B AWQ"
wait_mem

# ── Qwen2.5-VL-3B — GPTQ (community, est ~2GB) ──
echo ""
echo ">>> [Cat1] Qwen2.5-VL-3B — GPTQ pre-quantized"
python3 compression/quantized_pretrained/run_quantized_pretrained.py \
    --quantized_model_id "hfl/Qwen2.5-VL-3B-Instruct-GPTQ-Int4" \
    --base_model_id "Qwen/Qwen2.5-VL-3B-Instruct" \
    --quant_method gptq --quant_bits 4 \
    --vqav2_n 10 \
    || echo "FAILED: Qwen2.5-VL-3B GPTQ"
wait_mem

# ── Gemma3-4B — AWQ (community, est ~2.5GB) ──
echo ""
echo ">>> [Cat1] Gemma3-4B — AWQ pre-quantized"
python3 compression/quantized_pretrained/run_quantized_pretrained.py \
    --quantized_model_id "gaunernst/gemma-3-4b-it-int4-awq" \
    --base_model_id "google/gemma-3-4b-it" \
    --quant_method awq --quant_bits 4 \
    --vqav2_n 10 \
    || echo "FAILED: Gemma3-4B AWQ"
wait_mem

# ── Gemma3-4B — GPTQ (community, est ~2.5GB) ──
echo ""
echo ">>> [Cat1] Gemma3-4B — GPTQ pre-quantized"
python3 compression/quantized_pretrained/run_quantized_pretrained.py \
    --quantized_model_id "ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g" \
    --base_model_id "google/gemma-3-4b-it" \
    --quant_method gptq --quant_bits 4 \
    --vqav2_n 10 \
    || echo "FAILED: Gemma3-4B GPTQ"
wait_mem

# ============================================================
# CATEGORY 2: Efficiency methods for MEM_CRITICAL/slow models
# ============================================================
echo ""
echo "============================================================"
echo "  CATEGORY 2: EFFICIENCY METHODS (make loaded models usable)"
echo "============================================================"

# Models to apply Category 2 methods to:
CAT2_MODELS=(
    "HuggingFaceTB/SmolVLM-Instruct"
    "LiquidAI/LFM2-VL-3B"
    "apple/FastVLM-1.5B"
    "OpenGVLab/InternVL2_5-2B"
    "apple/FastVLM-0.5B"
)

for MODEL in "${CAT2_MODELS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  MODEL: $MODEL"
    echo "============================================================"

    # ── Wanda pruning (20% sparsity) ──
    echo ""
    echo ">>> [Cat2] $MODEL — Wanda 20%"
    python3 compression/pruning/run_wanda.py \
        --model_id "$MODEL" \
        --sparsity 0.20 \
        --n_calib 32 \
        --vqav2_n 10 \
        --skip_textvqa --skip_pope \
        || echo "FAILED: $MODEL Wanda 20%"
    wait_mem

    # ── Wanda pruning (50% sparsity) ──
    echo ""
    echo ">>> [Cat2] $MODEL — Wanda 50%"
    python3 compression/pruning/run_wanda.py \
        --model_id "$MODEL" \
        --sparsity 0.50 \
        --n_calib 32 \
        --vqav2_n 10 \
        --skip_textvqa --skip_pope \
        || echo "FAILED: $MODEL Wanda 50%"
    wait_mem

    # ── PALU (KV-cache compression, rank_ratio=0.25) ──
    echo ""
    echo ">>> [Cat2] $MODEL — PALU 25%"
    python3 compression/palu/run_palu.py \
        --model_id "$MODEL" \
        --rank_ratio 0.25 \
        --vqav2_n 10 \
        --skip_textvqa --skip_pope \
        || echo "FAILED: $MODEL PALU"
    wait_mem

    # ── PACT (token pruning + merging) ──
    echo ""
    echo ">>> [Cat2] $MODEL — PACT (prune=0.3, merge=0.2)"
    python3 compression/pact/run_pact.py \
        --model_id "$MODEL" \
        --prune_ratio 0.30 \
        --merge_ratio 0.20 \
        --vqav2_n 10 \
        --skip_textvqa --skip_pope \
        || echo "FAILED: $MODEL PACT"
    wait_mem

    # ── SparseGPT (50% sparsity) ──
    echo ""
    echo ">>> [Cat2] $MODEL — SparseGPT 50%"
    python3 compression/pruning/run_sparsegpt.py \
        --model_id "$MODEL" \
        --sparsity 0.50 \
        --n_calib 32 \
        --vqav2_n 10 \
        --skip_eval \
        || echo "FAILED: $MODEL SparseGPT"
    wait_mem

done

echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
echo "============================================================"
echo "Results in: results/"
