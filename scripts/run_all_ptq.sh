#!/usr/bin/env bash
# Run all PTQ experiments (INT8 + INT4) across 7 GPUs in waves.
# Each model+quant runs as a separate process — GPU freed on exit.
# Between waves, verify all GPUs are free before proceeding.
# Only evaluates on VQAv2 (1000 samples).

set -euo pipefail

export PATH="/home/azaz/miniconda3/envs/vlm-bench/bin:/usr/bin:/bin:$PATH"
cd /home/azaz/vlm-compression-benchmark

LOG_DIR="logs/ptq"
mkdir -p "$LOG_DIR"

# All 19 models × 2 quant levels = 38 jobs
# Format: "model_id|quant"
JOBS=(
  # ── INT8 runs ──
  "HuggingFaceTB/SmolVLM-256M-Instruct|int8"
  "HuggingFaceTB/SmolVLM-500M-Instruct|int8"
  "HuggingFaceTB/SmolVLM-Instruct|int8"
  "LiquidAI/LFM2-VL-450M|int8"
  "LiquidAI/LFM2-VL-1.6B|int8"
  "LiquidAI/LFM2-VL-3B|int8"
  "vikhyatk/moondream2|int8"
  "Qwen/Qwen2.5-VL-3B-Instruct|int8"
  "Qwen/Qwen2.5-VL-7B-Instruct|int8"
  "OpenGVLab/InternVL2_5-1B|int8"
  "OpenGVLab/InternVL2_5-2B|int8"
  "OpenGVLab/InternVL2_5-4B|int8"
  "OpenGVLab/InternVL2_5-8B|int8"
  "google/gemma-3-4b-it|int8"
  "google/gemma-3-12b-it|int8"
  "AIDC-AI/Ovis2-1B|int8"
  "AIDC-AI/Ovis2-2B|int8"
  "AIDC-AI/Ovis2-4B|int8"
  "AIDC-AI/Ovis2-8B|int8"
  # ── INT4 runs ──
  "HuggingFaceTB/SmolVLM-256M-Instruct|int4"
  "HuggingFaceTB/SmolVLM-500M-Instruct|int4"
  "HuggingFaceTB/SmolVLM-Instruct|int4"
  "LiquidAI/LFM2-VL-450M|int4"
  "LiquidAI/LFM2-VL-1.6B|int4"
  "LiquidAI/LFM2-VL-3B|int4"
  "vikhyatk/moondream2|int4"
  "Qwen/Qwen2.5-VL-3B-Instruct|int4"
  "Qwen/Qwen2.5-VL-7B-Instruct|int4"
  "OpenGVLab/InternVL2_5-1B|int4"
  "OpenGVLab/InternVL2_5-2B|int4"
  "OpenGVLab/InternVL2_5-4B|int4"
  "OpenGVLab/InternVL2_5-8B|int4"
  "google/gemma-3-4b-it|int4"
  "google/gemma-3-12b-it|int4"
  "AIDC-AI/Ovis2-1B|int4"
  "AIDC-AI/Ovis2-2B|int4"
  "AIDC-AI/Ovis2-4B|int4"
  "AIDC-AI/Ovis2-8B|int4"
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

run_ptq_job() {
  local gpu=$1
  local model_id=$2
  local quant=$3
  local safe_name="${model_id//\//__}"
  local log_file="${LOG_DIR}/${safe_name}__${quant}__bnb.log"

  echo "$(date '+%H:%M:%S') | GPU $gpu | START | $model_id ($quant)"
  CUDA_VISIBLE_DEVICES=$gpu python compression/ptq/run_ptq.py \
    --model_id "$model_id" \
    --quant "$quant" \
    --backend bnb \
    --vqav2_n 1000 \
    --skip_textvqa \
    --skip_pope \
    --force \
    > "$log_file" 2>&1
  local status=$?
  if [ $status -eq 0 ]; then
    echo "$(date '+%H:%M:%S') | GPU $gpu | DONE  | $model_id ($quant)"
  else
    echo "$(date '+%H:%M:%S') | GPU $gpu | FAIL  | $model_id ($quant) (exit=$status)"
  fi
  return $status
}

total=${#JOBS[@]}
idx=0
wave=1

echo "================================================================"
echo " PTQ Experiments: $total jobs across 7 GPUs"
echo " Dataset: VQAv2 (1000 samples) only"
echo "================================================================"

while [ $idx -lt $total ]; do
  remaining=$((total - idx))
  batch_size=$((remaining < 7 ? remaining : 7))

  echo ""
  echo "================================================================"
  echo "WAVE $wave: jobs $((idx+1))-$((idx+batch_size)) of $total"
  echo "================================================================"

  wait_for_gpus_free

  # Launch up to 7 jobs in parallel
  pids=()
  for gpu in $(seq 0 $((batch_size - 1))); do
    job="${JOBS[$((idx + gpu))]}"
    model_id="${job%%|*}"
    quant="${job##*|}"
    run_ptq_job $gpu "$model_id" "$quant" &
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
echo "ALL PTQ EXPERIMENTS COMPLETE"
echo "================================================================"
echo "Results in: results/ptq/"
ls results/ptq/*.json 2>/dev/null | wc -l
echo " result files generated."
