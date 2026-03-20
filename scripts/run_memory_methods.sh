#!/usr/bin/env bash
# Run real memory-saving compression methods: SVD-LLM, PALU, AWQ/GPTQ
# 10 VQAv2 samples per model. VQAv2 only.

set -uo pipefail  # no -e: don't abort on single model failure

export PATH="/home/azaz/miniconda3/envs/vlm-bench/bin:/usr/bin:/bin:$PATH"
cd /home/azaz/vlm-compression-benchmark

VQAV2_N=10
GPU=0

MODELS=(
  "HuggingFaceTB/SmolVLM-256M-Instruct"
  "HuggingFaceTB/SmolVLM-500M-Instruct"
  "HuggingFaceTB/SmolVLM-Instruct"
  "LiquidAI/LFM2-VL-450M"
  "LiquidAI/LFM2-VL-1.6B"
  "LiquidAI/LFM2-VL-3B"
  "vikhyatk/moondream2"
  "Qwen/Qwen2.5-VL-3B-Instruct"
  "Qwen/Qwen2.5-VL-7B-Instruct"
  "OpenGVLab/InternVL2_5-1B"
  "OpenGVLab/InternVL2_5-2B"
  "OpenGVLab/InternVL2_5-4B"
  "OpenGVLab/InternVL2_5-8B"
  "google/gemma-3-4b-it"
  "google/gemma-3-12b-it"
  "AIDC-AI/Ovis2-1B"
  "AIDC-AI/Ovis2-2B"
  "AIDC-AI/Ovis2-4B"
  "AIDC-AI/Ovis2-8B"
)

echo "================================================================"
echo " MEMORY OPTIMIZATION METHODS — 3 methods × 19 models"
echo " Samples: $VQAV2_N per model (VQAv2 only)"
echo "================================================================"

# ─────────────────────────────────────────────────────────────────────
# Method 1: SVD-LLM (50% rank reduction → ~1.5-2x memory saving)
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "============= SVD-LLM (50% rank reduction) ===================="
for model in "${MODELS[@]}"; do
  safe="${model//\//__}"
  mkdir -p logs/svd_llm
  echo "$(date '+%H:%M:%S') | svd_llm | START | $model"
  CUDA_VISIBLE_DEVICES=$GPU python compression/svd_llm/run_svd_llm.py \
    --model_id "$model" \
    --rank_ratio 0.50 \
    --min_rank 8 \
    --vqav2_n $VQAV2_N \
    --skip_textvqa --skip_pope --force \
    > "logs/svd_llm/${safe}.log" 2>&1
  if [ $? -eq 0 ]; then
    echo "$(date '+%H:%M:%S') | svd_llm | DONE  | $model"
  else
    echo "$(date '+%H:%M:%S') | svd_llm | FAIL  | $model"
  fi
done

# ─────────────────────────────────────────────────────────────────────
# Method 2: PALU (KV-cache compression, 25% rank → modest savings)
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "============= PALU (KV-cache 25% rank) ========================"
for model in "${MODELS[@]}"; do
  safe="${model//\//__}"
  mkdir -p logs/palu
  echo "$(date '+%H:%M:%S') | palu | START | $model"
  CUDA_VISIBLE_DEVICES=$GPU python compression/palu/run_palu.py \
    --model_id "$model" \
    --rank_ratio 0.25 \
    --min_rank 8 \
    --vqav2_n $VQAV2_N \
    --skip_textvqa --skip_pope --force \
    > "logs/palu/${safe}.log" 2>&1
  if [ $? -eq 0 ]; then
    echo "$(date '+%H:%M:%S') | palu | DONE  | $model"
  else
    echo "$(date '+%H:%M:%S') | palu | FAIL  | $model"
  fi
done

# ─────────────────────────────────────────────────────────────────────
# Method 3: AWQ INT4 (activation-aware ~4x compression)
# Note: AWQ requires autoawq library. Falls back gracefully if missing.
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "============= AWQ INT4 (activation-aware quantization) ========"
for model in "${MODELS[@]}"; do
  safe="${model//\//__}"
  mkdir -p logs/awq_gptq
  echo "$(date '+%H:%M:%S') | awq | START | $model"
  CUDA_VISIBLE_DEVICES=$GPU python compression/awq_gptq/run_awq_gptq.py \
    --model_id "$model" \
    --method awq \
    --n_calib 16 \
    --vqav2_n $VQAV2_N \
    --skip_textvqa --skip_pope --force \
    > "logs/awq_gptq/${safe}__awq.log" 2>&1
  if [ $? -eq 0 ]; then
    echo "$(date '+%H:%M:%S') | awq | DONE  | $model"
  else
    echo "$(date '+%H:%M:%S') | awq | FAIL  | $model"
  fi
done

# ─────────────────────────────────────────────────────────────────────
# Method 4: GPTQ INT4 (Hessian-based ~4x compression)
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "============= GPTQ INT4 (Hessian-based quantization) =========="
for model in "${MODELS[@]}"; do
  safe="${model//\//__}"
  mkdir -p logs/awq_gptq
  echo "$(date '+%H:%M:%S') | gptq | START | $model"
  CUDA_VISIBLE_DEVICES=$GPU python compression/awq_gptq/run_awq_gptq.py \
    --model_id "$model" \
    --method gptq \
    --n_calib 16 \
    --vqav2_n $VQAV2_N \
    --skip_textvqa --skip_pope --force \
    > "logs/awq_gptq/${safe}__gptq.log" 2>&1
  if [ $? -eq 0 ]; then
    echo "$(date '+%H:%M:%S') | gptq | DONE  | $model"
  else
    echo "$(date '+%H:%M:%S') | gptq | FAIL  | $model"
  fi
done

echo ""
echo "================================================================"
echo "ALL MEMORY METHODS COMPLETE"
echo "================================================================"
echo "Results:"
for dir in svd_llm palu awq_gptq; do
  count=$(ls results/${dir}/*.json 2>/dev/null | wc -l)
  echo "  results/${dir}/: ${count} files"
done
