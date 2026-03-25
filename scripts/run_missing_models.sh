#!/usr/bin/env bash
# =============================================================================
# Run missing models: 3 new families + missing variants
# Same 15 methods, VQAv2 50 samples, multi-metric evaluation
# =============================================================================

set -uo pipefail

export PATH="/home/azaz/miniconda3/envs/vlm-bench/bin:/usr/bin:/bin:$PATH"
cd /home/azaz/vlm-compression-benchmark

VQAV2_N=50
GPU=4
LOG_DIR="logs/full_benchmark"
mkdir -p "$LOG_DIR"

# 3 missing models: FastVLM family
MODELS=(
  "apple/FastVLM-0.5B"
  "apple/FastVLM-1.5B"
  "apple/FastVLM-7B"
)

# ── Helper: run a method and log status ─────────────────────────────────────
run_method() {
  local method_name="$1"
  local safe="$2"
  local log_file="$3"
  shift 3

  echo "$(date '+%H:%M:%S') | ${method_name} | START | ${safe}"
  if CUDA_VISIBLE_DEVICES=$GPU "$@" > "$log_file" 2>&1; then
    echo "$(date '+%H:%M:%S') | ${method_name} | DONE  | ${safe}"
  else
    echo "$(date '+%H:%M:%S') | ${method_name} | FAIL  | ${safe} (see ${log_file})"
  fi
}

echo "================================================================"
echo " MISSING MODELS BENCHMARK"
echo " Models: ${#MODELS[@]}"
echo " Samples: $VQAV2_N (VQAv2 only)"
echo " Methods: baseline, PTQ(int4), pruning(sp20,sp40),"
echo "          wanda(sp20,sp40), AWQ, GPTQ, SparseGPT(sp50),"
echo "          AWP(sp50), PACT, SVD-LLM, PALU, CASP, SLIM"
echo " Started: $(date)"
echo "================================================================"

TOTAL_MODELS=${#MODELS[@]}
MODEL_IDX=0

for model in "${MODELS[@]}"; do
  MODEL_IDX=$((MODEL_IDX + 1))
  safe="${model//\//__}"
  mkdir -p "$LOG_DIR/$safe"

  echo ""
  echo "================================================================"
  echo " MODEL $MODEL_IDX/$TOTAL_MODELS: $model"
  echo " $(date)"
  echo "================================================================"

  # ── 1. Baseline ──────────────────────────────────────────────────────
  run_method "baseline" "$safe" "$LOG_DIR/$safe/baseline.log" \
    python evaluation/run_baseline.py \
      --model_id "$model" \
      --vqav2_n $VQAV2_N \
      --skip_textvqa --skip_pope --force

  # ── 2. PTQ INT4 ─────────────────────────────────────────────────────
  run_method "ptq_int4" "$safe" "$LOG_DIR/$safe/ptq_int4.log" \
    python compression/ptq/run_ptq.py \
      --model_id "$model" --quant int4 \
      --vqav2_n $VQAV2_N \
      --skip_textvqa --skip_pope --force

  # ── 3. Magnitude Pruning 20% ────────────────────────────────────────
  run_method "pruning_sp20" "$safe" "$LOG_DIR/$safe/pruning_sp20.log" \
    python compression/pruning/run_pruning.py \
      --model_id "$model" --sparsity 0.20 \
      --vqav2_n $VQAV2_N \
      --skip_textvqa --skip_pope --force

  # ── 4. Magnitude Pruning 40% ────────────────────────────────────────
  run_method "pruning_sp40" "$safe" "$LOG_DIR/$safe/pruning_sp40.log" \
    python compression/pruning/run_pruning.py \
      --model_id "$model" --sparsity 0.40 \
      --vqav2_n $VQAV2_N \
      --skip_textvqa --skip_pope --force

  # ── 5. Wanda 20% ───────────────────────────────────────────────────
  run_method "wanda_sp20" "$safe" "$LOG_DIR/$safe/wanda_sp20.log" \
    python compression/pruning/run_wanda.py \
      --model_id "$model" --sparsity 0.20 \
      --vqav2_n $VQAV2_N \
      --skip_textvqa --skip_pope --force

  # ── 6. Wanda 40% ───────────────────────────────────────────────────
  run_method "wanda_sp40" "$safe" "$LOG_DIR/$safe/wanda_sp40.log" \
    python compression/pruning/run_wanda.py \
      --model_id "$model" --sparsity 0.40 \
      --vqav2_n $VQAV2_N \
      --skip_textvqa --skip_pope --force

  # ── 7. AWQ INT4 ────────────────────────────────────────────────────
  run_method "awq_int4" "$safe" "$LOG_DIR/$safe/awq_int4.log" \
    python compression/awq_gptq/run_awq_gptq.py \
      --model_id "$model" --method awq \
      --n_calib 128 --vqav2_n $VQAV2_N \
      --skip_textvqa --skip_pope --force

  # ── 8. GPTQ INT4 ──────────────────────────────────────────────────
  run_method "gptq_int4" "$safe" "$LOG_DIR/$safe/gptq_int4.log" \
    python compression/awq_gptq/run_awq_gptq.py \
      --model_id "$model" --method gptq \
      --n_calib 128 --vqav2_n $VQAV2_N \
      --skip_textvqa --skip_pope --force

  # ── 9. SparseGPT 50% ─────────────────────────────────────────────
  run_method "sparsegpt_sp50" "$safe" "$LOG_DIR/$safe/sparsegpt_sp50.log" \
    python compression/sparsegpt/run_sparsegpt.py \
      --model_id "$model" --sparsity 0.50 \
      --n_calib 128 --vqav2_n $VQAV2_N \
      --skip_textvqa --skip_pope --force

  # ── 10. AWP (Wanda sp50 + INT4) ───────────────────────────────────
  run_method "awp_sp50" "$safe" "$LOG_DIR/$safe/awp_sp50.log" \
    python compression/awp/run_awp.py \
      --model_id "$model" --sparsity 0.50 \
      --n_calib 128 --vqav2_n $VQAV2_N \
      --skip_textvqa --skip_pope --force

  # ── 11. PACT (token pruning + merging) ────────────────────────────
  run_method "pact" "$safe" "$LOG_DIR/$safe/pact.log" \
    python compression/pact/run_pact.py \
      --model_id "$model" \
      --prune_ratio 0.30 --merge_ratio 0.20 \
      --vqav2_n $VQAV2_N \
      --skip_textvqa --skip_pope --force

  # ── 12. SVD-LLM (energy=0.95) ────────────────────────────────────
  run_method "svd_llm" "$safe" "$LOG_DIR/$safe/svd_llm.log" \
    python compression/svd_llm/run_svd_llm.py \
      --model_id "$model" --energy 0.95 --min_rank 32 \
      --vqav2_n $VQAV2_N \
      --skip_textvqa --skip_pope --force

  # ── 13. PALU (energy=0.95) ───────────────────────────────────────
  run_method "palu" "$safe" "$LOG_DIR/$safe/palu.log" \
    python compression/palu/run_palu.py \
      --model_id "$model" --energy 0.95 --min_rank 8 \
      --vqav2_n $VQAV2_N \
      --skip_textvqa --skip_pope --force

  # ── 14. CASP ─────────────────────────────────────────────────────
  run_method "casp" "$safe" "$LOG_DIR/$safe/casp.log" \
    python compression/casp_slim/run_casp_slim.py \
      --model_id "$model" --method casp \
      --n_calib 32 --vqav2_n $VQAV2_N \
      --skip_textvqa --skip_pope --force

  # ── 15. SLIM ─────────────────────────────────────────────────────
  run_method "slim" "$safe" "$LOG_DIR/$safe/slim.log" \
    python compression/casp_slim/run_casp_slim.py \
      --model_id "$model" --method slim \
      --sparsity 0.50 --rank_ratio 0.30 \
      --n_calib 32 --vqav2_n $VQAV2_N \
      --skip_textvqa --skip_pope --force

  # ── Clear GPU cache ─────────────────────────────────────────────
  echo "$(date '+%H:%M:%S') | cache_clear | Clearing GPU cache for $model"
  python -c "
import torch, gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info(0)
    print(f'  GPU memory after clear: {(total-free)/1024**2:.0f} MB used / {total/1024**2:.0f} MB total')
"
  echo "$(date '+%H:%M:%S') | ALL 15 methods done for $model"
  echo ""
done

echo "================================================================"
echo " MISSING MODELS BENCHMARK COMPLETE — $(date)"
echo "================================================================"
