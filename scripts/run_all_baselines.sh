#!/usr/bin/env bash
# Run all 19 baselines across 7 GPUs in waves.
# Each model runs as a separate process — GPU memory is fully freed when process exits.
# Between waves, verify all GPUs are free before proceeding.

set -euo pipefail

export PATH="/home/azaz/miniconda3/envs/vlm-bench/bin:/usr/bin:/bin:$PATH"
cd /home/azaz/vlm-compression-benchmark

LOG_DIR="logs/baseline"
mkdir -p "$LOG_DIR"

# All 19 models
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

wait_for_gpus_free() {
  echo "$(date '+%H:%M:%S') | Verifying all GPUs are free..."
  for i in $(seq 1 30); do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    if [ "$used" -lt 500 ]; then
      echo "$(date '+%H:%M:%S') | All GPUs free (total used: ${used} MiB)"
      return 0
    fi
    echo "$(date '+%H:%M:%S') | GPUs still in use (${used} MiB). Waiting 10s... ($i/30)"
    sleep 10
  done
  echo "$(date '+%H:%M:%S') | WARNING: GPUs not fully free after 5 min. Proceeding anyway."
}

run_model() {
  local gpu=$1
  local model_id=$2
  local safe_name="${model_id//\//__}"
  local log_file="${LOG_DIR}/${safe_name}.log"

  echo "$(date '+%H:%M:%S') | GPU $gpu | START | $model_id"
  CUDA_VISIBLE_DEVICES=$gpu python evaluation/run_baseline.py \
    --model_id "$model_id" \
    --vqav2_n 1000 \
    --skip_textvqa \
    --skip_pope \
    --force \
    > "$log_file" 2>&1
  local status=$?
  if [ $status -eq 0 ]; then
    echo "$(date '+%H:%M:%S') | GPU $gpu | DONE  | $model_id"
  else
    echo "$(date '+%H:%M:%S') | GPU $gpu | FAIL  | $model_id (exit=$status)"
  fi
  return $status
}

total=${#MODELS[@]}
idx=0
wave=1

while [ $idx -lt $total ]; do
  # Determine how many to run in this wave (max 7)
  remaining=$((total - idx))
  batch_size=$((remaining < 7 ? remaining : 7))

  echo ""
  echo "================================================================"
  echo "WAVE $wave: models $((idx+1))-$((idx+batch_size)) of $total"
  echo "================================================================"

  wait_for_gpus_free

  # Launch up to 7 models in parallel
  pids=()
  for gpu in $(seq 0 $((batch_size - 1))); do
    run_model $gpu "${MODELS[$((idx + gpu))]}" &
    pids+=($!)
  done

  # Wait for all processes in this wave
  failed=0
  for pid in "${pids[@]}"; do
    wait $pid || ((failed++))
  done

  echo "$(date '+%H:%M:%S') | Wave $wave complete. Failed: $failed/$batch_size"

  idx=$((idx + batch_size))
  wave=$((wave + 1))
done

echo ""
echo "================================================================"
echo "ALL BASELINES COMPLETE"
echo "================================================================"
echo "Results in: results/baseline/"
ls -la results/baseline/*.json 2>/dev/null | wc -l
echo " result files generated."
