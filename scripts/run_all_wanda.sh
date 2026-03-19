#!/usr/bin/env bash
# Run all Wanda pruning experiments (20% + 40% sparsity) across 7 GPUs in waves.
# Each model+sparsity runs as a separate process — GPU freed on exit.
# Only evaluates on VQAv2 (1000 samples).

set -euo pipefail

export PATH="/home/azaz/miniconda3/envs/vlm-bench/bin:/usr/bin:/bin:$PATH"
cd /home/azaz/vlm-compression-benchmark

LOG_DIR="logs/wanda"
mkdir -p "$LOG_DIR"

# All 19 models × 2 sparsity levels = 38 jobs
# Format: "model_id|sparsity"
JOBS=(
  # ── 20% sparsity ──
  "HuggingFaceTB/SmolVLM-256M-Instruct|0.20"
  "HuggingFaceTB/SmolVLM-500M-Instruct|0.20"
  "HuggingFaceTB/SmolVLM-Instruct|0.20"
  "LiquidAI/LFM2-VL-450M|0.20"
  "LiquidAI/LFM2-VL-1.6B|0.20"
  "LiquidAI/LFM2-VL-3B|0.20"
  "vikhyatk/moondream2|0.20"
  "Qwen/Qwen2.5-VL-3B-Instruct|0.20"
  "Qwen/Qwen2.5-VL-7B-Instruct|0.20"
  "OpenGVLab/InternVL2_5-1B|0.20"
  "OpenGVLab/InternVL2_5-2B|0.20"
  "OpenGVLab/InternVL2_5-4B|0.20"
  "OpenGVLab/InternVL2_5-8B|0.20"
  "google/gemma-3-4b-it|0.20"
  "google/gemma-3-12b-it|0.20"
  "AIDC-AI/Ovis2-1B|0.20"
  "AIDC-AI/Ovis2-2B|0.20"
  "AIDC-AI/Ovis2-4B|0.20"
  "AIDC-AI/Ovis2-8B|0.20"
  # ── 40% sparsity ──
  "HuggingFaceTB/SmolVLM-256M-Instruct|0.40"
  "HuggingFaceTB/SmolVLM-500M-Instruct|0.40"
  "HuggingFaceTB/SmolVLM-Instruct|0.40"
  "LiquidAI/LFM2-VL-450M|0.40"
  "LiquidAI/LFM2-VL-1.6B|0.40"
  "LiquidAI/LFM2-VL-3B|0.40"
  "vikhyatk/moondream2|0.40"
  "Qwen/Qwen2.5-VL-3B-Instruct|0.40"
  "Qwen/Qwen2.5-VL-7B-Instruct|0.40"
  "OpenGVLab/InternVL2_5-1B|0.40"
  "OpenGVLab/InternVL2_5-2B|0.40"
  "OpenGVLab/InternVL2_5-4B|0.40"
  "OpenGVLab/InternVL2_5-8B|0.40"
  "google/gemma-3-4b-it|0.40"
  "google/gemma-3-12b-it|0.40"
  "AIDC-AI/Ovis2-1B|0.40"
  "AIDC-AI/Ovis2-2B|0.40"
  "AIDC-AI/Ovis2-4B|0.40"
  "AIDC-AI/Ovis2-8B|0.40"
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

run_wanda_job() {
  local gpu=$1
  local model_id=$2
  local sparsity=$3
  local safe_name="${model_id//\//__}"
  local sp_tag="sp$(echo "$sparsity" | sed 's/0\.//' | sed 's/\.//g')"
  local log_file="${LOG_DIR}/${safe_name}__wanda_${sp_tag}.log"

  echo "$(date '+%H:%M:%S') | GPU $gpu | START | $model_id (wanda sp=$sparsity)"
  CUDA_VISIBLE_DEVICES=$gpu python compression/pruning/run_wanda.py \
    --model_id "$model_id" \
    --sparsity "$sparsity" \
    --n_calib 128 \
    --vqav2_n 1000 \
    --skip_textvqa \
    --skip_pope \
    --force \
    > "$log_file" 2>&1
  local status=$?
  if [ $status -eq 0 ]; then
    echo "$(date '+%H:%M:%S') | GPU $gpu | DONE  | $model_id (wanda sp=$sparsity)"
  else
    echo "$(date '+%H:%M:%S') | GPU $gpu | FAIL  | $model_id (wanda sp=$sparsity) (exit=$status)"
  fi
  return $status
}

total=${#JOBS[@]}
idx=0
wave=1

echo "================================================================"
echo " Wanda Pruning Experiments: $total jobs across 7 GPUs"
echo " Sparsity levels: 20%, 40%"
echo " Calibration: 128 samples"
echo " Evaluation: VQAv2 (1000 samples) only"
echo "================================================================"

while [ $idx -lt $total ]; do
  remaining=$((total - idx))
  batch_size=$((remaining < 7 ? remaining : 7))

  echo ""
  echo "================================================================"
  echo "WAVE $wave: jobs $((idx+1))-$((idx+batch_size)) of $total"
  echo "================================================================"

  wait_for_gpus_free

  pids=()
  for gpu in $(seq 0 $((batch_size - 1))); do
    job="${JOBS[$((idx + gpu))]}"
    model_id="${job%%|*}"
    sparsity="${job##*|}"
    run_wanda_job $gpu "$model_id" "$sparsity" &
    pids+=($!)
  done

  failed=0
  for pid in "${pids[@]}"; do
    wait $pid || failed=$((failed + 1))
  done

  echo "$(date '+%H:%M:%S') | Wave $wave complete. Failed: $failed/$batch_size"

  idx=$((idx + batch_size))
  wave=$((wave + 1))
done

echo ""
echo "================================================================"
echo "ALL WANDA PRUNING EXPERIMENTS COMPLETE"
echo "================================================================"
echo "Results in: results/wanda/"
ls results/wanda/*.json 2>/dev/null | wc -l
echo " result files generated."
