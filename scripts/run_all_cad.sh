#!/usr/bin/env bash
# ================================================================
# CAD SERVER: Run ALL compression methods on ALL 19 models
# 20 VQAv2 samples per run (quick reference baseline for Jetson comparison)
# 7 GPUs in waves — each job gets its own GPU.
# VQAv2 only (skip TextVQA, POPE).
# ================================================================

set -uo pipefail

export PATH="/home/azaz/miniconda3/envs/vlm-bench/bin:/usr/bin:/bin:$PATH"
cd /home/azaz/vlm-compression-benchmark

VQAV2_N=20
N_CALIB=16
NUM_GPUS=7

# All 19 models (sorted by size)
MODELS=(
  "HuggingFaceTB/SmolVLM-256M-Instruct"
  "LiquidAI/LFM2-VL-450M"
  "HuggingFaceTB/SmolVLM-500M-Instruct"
  "OpenGVLab/InternVL2_5-1B"
  "AIDC-AI/Ovis2-1B"
  "LiquidAI/LFM2-VL-1.6B"
  "OpenGVLab/InternVL2_5-2B"
  "AIDC-AI/Ovis2-2B"
  "vikhyatk/moondream2"
  "HuggingFaceTB/SmolVLM-Instruct"
  "LiquidAI/LFM2-VL-3B"
  "Qwen/Qwen2.5-VL-3B-Instruct"
  "OpenGVLab/InternVL2_5-4B"
  "AIDC-AI/Ovis2-4B"
  "google/gemma-3-4b-it"
  "Qwen/Qwen2.5-VL-7B-Instruct"
  "OpenGVLab/InternVL2_5-8B"
  "AIDC-AI/Ovis2-8B"
  "google/gemma-3-12b-it"
)

# ── Helpers ──────────────────────────────────────────────────────

wait_for_gpus_free() {
  for i in $(seq 1 30); do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    if [ "$used" -lt 1000 ]; then
      return 0
    fi
    echo "$(date '+%H:%M:%S') | GPUs in use (${used} MiB). Waiting 10s... ($i/30)"
    sleep 10
  done
  echo "$(date '+%H:%M:%S') | WARNING: GPUs not fully free. Proceeding anyway."
}

# run_method METHOD_NAME SCRIPT [EXTRA_ARGS...]
# Runs the given script on all 19 models in waves of $NUM_GPUS
run_method() {
  local method_name="$1"
  local script="$2"
  shift 2
  local extra_args=("$@")
  local total=${#MODELS[@]}
  local idx=0
  local wave=1

  echo ""
  echo "================================================================"
  echo " METHOD: $method_name ($total jobs)"
  echo "================================================================"

  while [ $idx -lt $total ]; do
    local remaining=$((total - idx))
    local batch=$((remaining < NUM_GPUS ? remaining : NUM_GPUS))

    wait_for_gpus_free

    echo "$(date '+%H:%M:%S') | Wave $wave: jobs $((idx+1))-$((idx+batch)) of $total"

    local pids=()
    for g in $(seq 0 $((batch - 1))); do
      local model_id="${MODELS[$((idx + g))]}"
      local safe="${model_id//\//__}"
      local log_dir="logs/${method_name}"
      mkdir -p "$log_dir"
      local log_file="${log_dir}/${safe}.log"

      echo "$(date '+%H:%M:%S') | GPU $g | START | $model_id"
      CUDA_VISIBLE_DEVICES=$g python "$script" \
        --model_id "$model_id" \
        --vqav2_n $VQAV2_N \
        --skip_textvqa --skip_pope \
        --force \
        "${extra_args[@]}" \
        > "$log_file" 2>&1 &
      pids+=($!)
    done

    local failed=0
    for pid in "${pids[@]}"; do
      wait $pid || ((failed++))
    done

    if [ $failed -gt 0 ]; then
      echo "$(date '+%H:%M:%S') | Wave $wave done. Failed: $failed/$batch"
    else
      echo "$(date '+%H:%M:%S') | Wave $wave done. All $batch OK"
    fi

    idx=$((idx + batch))
    wave=$((wave + 1))
  done
}

# ── Start ────────────────────────────────────────────────────────

START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "================================================================"
echo " CAD SERVER — FULL BENCHMARK RUN"
echo " 16 method variants × 19 models = 304 total jobs"
echo " Samples: $VQAV2_N VQAv2 per run"
echo " GPUs: $NUM_GPUS × NVIDIA RTX A6000"
echo " Started: $START_TIME"
echo "================================================================"

# ── 1. Baseline (FP16) ──────────────────────────────────────────
run_method "baseline" "evaluation/run_baseline.py" \
  --quant fp16

# ── 2. PTQ INT8 (BitsAndBytes) ──────────────────────────────────
run_method "ptq_int8" "compression/ptq/run_ptq.py" \
  --quant int8 --backend bnb

# ── 3. PTQ INT4 (BitsAndBytes) ──────────────────────────────────
run_method "ptq_int4" "compression/ptq/run_ptq.py" \
  --quant int4 --backend bnb

# ── 4. Magnitude Pruning 20% ────────────────────────────────────
run_method "pruning_sp20" "compression/pruning/run_pruning.py" \
  --sparsity 0.20

# ── 5. Magnitude Pruning 40% ────────────────────────────────────
run_method "pruning_sp40" "compression/pruning/run_pruning.py" \
  --sparsity 0.40

# ── 6. Wanda 20% ────────────────────────────────────────────────
run_method "wanda_sp20" "compression/pruning/run_wanda.py" \
  --sparsity 0.20 --n_calib 128

# ── 7. Wanda 40% ────────────────────────────────────────────────
run_method "wanda_sp40" "compression/pruning/run_wanda.py" \
  --sparsity 0.40 --n_calib 128

# ── 8. AWQ INT4 ──────────────────────────────────────────────────
run_method "awq" "compression/awq_gptq/run_awq_gptq.py" \
  --method awq --n_calib 128

# ── 9. GPTQ INT4 ────────────────────────────────────────────────
run_method "gptq" "compression/awq_gptq/run_awq_gptq.py" \
  --method gptq --n_calib 128

# ── 10. SparseGPT 50% ───────────────────────────────────────────
run_method "sparsegpt" "compression/sparsegpt/run_sparsegpt.py" \
  --sparsity 0.50 --n_calib $N_CALIB --blocksize 128

# ── 11. AWP (Wanda 50% + INT4) ──────────────────────────────────
run_method "awp" "compression/awp/run_awp.py" \
  --sparsity 0.50 --n_calib $N_CALIB

# ── 12. PACT (visual token prune 30% + merge 20%) ───────────────
run_method "pact" "compression/pact/run_pact.py" \
  --prune_ratio 0.30 --merge_ratio 0.20 --target_layer 1

# ── 13. SVD-LLM (energy 0.95) ───────────────────────────────────
run_method "svd_llm" "compression/svd_llm/run_svd_llm.py" \
  --energy 0.95 --min_rank 32

# ── 14. PALU (energy 0.95) ──────────────────────────────────────
run_method "palu" "compression/palu/run_palu.py" \
  --energy 0.95 --min_rank 8

# ── 15. CASP ─────────────────────────────────────────────────────
run_method "casp" "compression/casp_slim/run_casp_slim.py" \
  --method casp --n_calib $N_CALIB

# ── 16. SLIM ─────────────────────────────────────────────────────
run_method "slim" "compression/casp_slim/run_casp_slim.py" \
  --method slim --sparsity 0.50 --rank_ratio 0.30 --n_calib $N_CALIB

# ── Summary ──────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " ALL METHODS COMPLETE"
echo " Started:  $START_TIME"
echo " Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
echo ""
echo "Results summary:"
for dir in baseline ptq pruning wanda awq_gptq sparsegpt awp pact svd_llm palu casp_slim; do
  count=$(ls results/${dir}/*.json 2>/dev/null | wc -l)
  echo "  results/${dir}/: ${count} files"
done
