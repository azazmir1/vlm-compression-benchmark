#!/usr/bin/env bash
# Run all 7 new compression methods across all 19 models.
# Uses 10 VQAv2 samples per model (quick evaluation).
# Methods: AWQ, GPTQ, SparseGPT, AWP, PACT, SVD-LLM, PALU, CASP, SLIM
#
# Sequential execution (one model at a time) to avoid OOM.
# Skips TextVQA and POPE — VQAv2 only.

set -euo pipefail

export PATH="/home/azaz/miniconda3/envs/vlm-bench/bin:/usr/bin:/bin:$PATH"
cd /home/azaz/vlm-compression-benchmark

VQAV2_N=10
N_CALIB=16   # Small calibration set for speed
GPU=0        # Default GPU

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

run_method() {
  local method_name=$1
  local script=$2
  shift 2
  local extra_args=("$@")
  local model_id=$MODEL
  local safe_name="${model_id//\//__}"
  local log_dir="logs/${method_name}"
  mkdir -p "$log_dir"
  local log_file="${log_dir}/${safe_name}.log"

  echo "$(date '+%H:%M:%S') | ${method_name} | START | ${model_id}"
  CUDA_VISIBLE_DEVICES=$GPU python "$script" \
    --model_id "$model_id" \
    --vqav2_n $VQAV2_N \
    --skip_textvqa \
    --skip_pope \
    --force \
    "${extra_args[@]}" \
    > "$log_file" 2>&1
  local status=$?
  if [ $status -eq 0 ]; then
    echo "$(date '+%H:%M:%S') | ${method_name} | DONE  | ${model_id}"
  else
    echo "$(date '+%H:%M:%S') | ${method_name} | FAIL  | ${model_id} (exit=$status)"
  fi
  return 0  # Don't abort on single model failure
}

echo "================================================================"
echo " NEW COMPRESSION METHODS — ALL 19 MODELS"
echo " Samples: $VQAV2_N per model (VQAv2 only)"
echo " Calibration: $N_CALIB samples"
echo "================================================================"

total_models=${#MODELS[@]}

# ─────────────────────────────────────────────────────────────────────────
# Method 1: SparseGPT (50% sparsity)
# ─────────────────────────────────────────────────────────────────────────
echo ""
echo "============= METHOD: SparseGPT (50% sparsity) ================"
for MODEL in "${MODELS[@]}"; do
  run_method "sparsegpt" "compression/sparsegpt/run_sparsegpt.py" \
    --sparsity 0.50 --n_calib $N_CALIB --blocksize 128
done

# ─────────────────────────────────────────────────────────────────────────
# Method 2: AWP (Wanda 50% + simulated INT4)
# ─────────────────────────────────────────────────────────────────────────
echo ""
echo "============= METHOD: AWP (Wanda 50% + INT4) =================="
for MODEL in "${MODELS[@]}"; do
  run_method "awp" "compression/awp/run_awp.py" \
    --sparsity 0.50 --n_calib $N_CALIB
done

# ─────────────────────────────────────────────────────────────────────────
# Method 3: PACT (visual token compression)
# ─────────────────────────────────────────────────────────────────────────
echo ""
echo "============= METHOD: PACT (token prune 30% + merge 20%) ======"
for MODEL in "${MODELS[@]}"; do
  run_method "pact" "compression/pact/run_pact.py" \
    --prune_ratio 0.30 --merge_ratio 0.20 --target_layer 1
done

# ─────────────────────────────────────────────────────────────────────────
# Method 4: SVD-LLM (50% rank reduction)
# ─────────────────────────────────────────────────────────────────────────
echo ""
echo "============= METHOD: SVD-LLM (50% rank reduction) ============"
for MODEL in "${MODELS[@]}"; do
  run_method "svd_llm" "compression/svd_llm/run_svd_llm.py" \
    --rank_ratio 0.50 --min_rank 8
done

# ─────────────────────────────────────────────────────────────────────────
# Method 5: PALU (KV-cache compression, 25% rank)
# ─────────────────────────────────────────────────────────────────────────
echo ""
echo "============= METHOD: PALU (KV-cache 25% rank) ================"
for MODEL in "${MODELS[@]}"; do
  run_method "palu" "compression/palu/run_palu.py" \
    --rank_ratio 0.25 --min_rank 8
done

# ─────────────────────────────────────────────────────────────────────────
# Method 6: CASP (attention-sparsity mixed precision)
# ─────────────────────────────────────────────────────────────────────────
echo ""
echo "============= METHOD: CASP (mixed precision + QK low-rank) ===="
for MODEL in "${MODELS[@]}"; do
  run_method "casp" "compression/casp_slim/run_casp_slim.py" \
    --method casp --n_calib $N_CALIB
done

# ─────────────────────────────────────────────────────────────────────────
# Method 7: SLIM (triple: SVD + pruning + INT4)
# ─────────────────────────────────────────────────────────────────────────
echo ""
echo "============= METHOD: SLIM (SVD + prune + INT4) ==============="
for MODEL in "${MODELS[@]}"; do
  run_method "slim" "compression/casp_slim/run_casp_slim.py" \
    --method slim --sparsity 0.50 --rank_ratio 0.30 --n_calib $N_CALIB
done

echo ""
echo "================================================================"
echo "ALL NEW METHODS COMPLETE"
echo "================================================================"
echo "Results:"
for dir in sparsegpt awp pact svd_llm palu casp_slim; do
  count=$(ls results/${dir}/*.json 2>/dev/null | wc -l)
  echo "  results/${dir}/: ${count} files"
done
