#!/usr/bin/env bash
# Run real memory-saving compression methods: SVD-LLM, PALU, AWQ/GPTQ
# 1000 VQAv2 samples per model. Smallest models first.
# VQAv2 only (skip TextVQA, POPE).

set -uo pipefail

export PATH="/home/azaz/miniconda3/envs/vlm-bench/bin:/usr/bin:/bin:$PATH"
cd /home/azaz/vlm-compression-benchmark

VQAV2_N=50
GPU=0

# Models sorted by size (smallest first)
MODELS=(
  "HuggingFaceTB/SmolVLM-256M-Instruct"
  "LiquidAI/LFM2-VL-450M"
  "HuggingFaceTB/SmolVLM-500M-Instruct"
  "OpenGVLab/InternVL2_5-1B"
  "AIDC-AI/Ovis2-1B"
  "LiquidAI/LFM2-VL-1.6B"
  "OpenGVLab/InternVL2_5-2B"
  "AIDC-AI/Ovis2-2B"
  "HuggingFaceTB/SmolVLM-Instruct"
  "LiquidAI/LFM2-VL-3B"
  "Qwen/Qwen2.5-VL-3B-Instruct"
  "OpenGVLab/InternVL2_5-4B"
  "AIDC-AI/Ovis2-4B"
  "google/gemma-3-4b-it"
  "vikhyatk/moondream2"
  "Qwen/Qwen2.5-VL-7B-Instruct"
  "OpenGVLab/InternVL2_5-8B"
  "AIDC-AI/Ovis2-8B"
  "google/gemma-3-12b-it"
)

echo "================================================================"
echo " MEMORY OPTIMIZATION: 3 methods × 19 models (smallest first)"
echo " Methods: SVD-LLM, PALU, AWQ/GPTQ"
echo " Samples: $VQAV2_N per model (VQAv2 only)"
echo "================================================================"

for model in "${MODELS[@]}"; do
  safe="${model//\//__}"
  echo ""
  echo "================================================================"
  echo " MODEL: $model"
  echo "================================================================"

  # ── SVD-LLM (energy=0.95) ──────────────────────────────────────────
  mkdir -p logs/svd_llm
  echo "$(date '+%H:%M:%S') | svd_llm | START | $model"
  CUDA_VISIBLE_DEVICES=$GPU python compression/svd_llm/run_svd_llm.py \
    --model_id "$model" \
    --energy 0.95 \
    --min_rank 32 \
    --vqav2_n $VQAV2_N \
    --skip_textvqa --skip_pope --force \
    > "logs/svd_llm/${safe}.log" 2>&1
  if [ $? -eq 0 ]; then
    echo "$(date '+%H:%M:%S') | svd_llm | DONE  | $model"
  else
    echo "$(date '+%H:%M:%S') | svd_llm | FAIL  | $model (see logs/svd_llm/${safe}.log)"
  fi

  # ── PALU (energy=0.95) ─────────────────────────────────────────────
  mkdir -p logs/palu
  echo "$(date '+%H:%M:%S') | palu    | START | $model"
  CUDA_VISIBLE_DEVICES=$GPU python compression/palu/run_palu.py \
    --model_id "$model" \
    --energy 0.95 \
    --min_rank 8 \
    --vqav2_n $VQAV2_N \
    --skip_textvqa --skip_pope --force \
    > "logs/palu/${safe}.log" 2>&1
  if [ $? -eq 0 ]; then
    echo "$(date '+%H:%M:%S') | palu    | DONE  | $model"
  else
    echo "$(date '+%H:%M:%S') | palu    | FAIL  | $model (see logs/palu/${safe}.log)"
  fi

  # ── AWQ INT4 ────────────────────────────────────────────────────────
  mkdir -p logs/awq_gptq
  echo "$(date '+%H:%M:%S') | awq     | START | $model"
  CUDA_VISIBLE_DEVICES=$GPU python compression/awq_gptq/run_awq_gptq.py \
    --model_id "$model" \
    --method awq \
    --n_calib 128 \
    --vqav2_n $VQAV2_N \
    --skip_textvqa --skip_pope --force \
    > "logs/awq_gptq/${safe}__awq.log" 2>&1
  if [ $? -eq 0 ]; then
    echo "$(date '+%H:%M:%S') | awq     | DONE  | $model"
  else
    echo "$(date '+%H:%M:%S') | awq     | FAIL  | $model (see logs/awq_gptq/${safe}__awq.log)"
  fi

  # ── GPTQ INT4 ───────────────────────────────────────────────────────
  echo "$(date '+%H:%M:%S') | gptq    | START | $model"
  CUDA_VISIBLE_DEVICES=$GPU python compression/awq_gptq/run_awq_gptq.py \
    --model_id "$model" \
    --method gptq \
    --n_calib 128 \
    --vqav2_n $VQAV2_N \
    --skip_textvqa --skip_pope --force \
    > "logs/awq_gptq/${safe}__gptq.log" 2>&1
  if [ $? -eq 0 ]; then
    echo "$(date '+%H:%M:%S') | gptq    | DONE  | $model"
  else
    echo "$(date '+%H:%M:%S') | gptq    | FAIL  | $model (see logs/awq_gptq/${safe}__gptq.log)"
  fi

  echo "$(date '+%H:%M:%S') | ALL 4 methods done for $model"
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
