#!/usr/bin/env bash
# =============================================================================
# run_rpi5_benchmark.sh
# Full memory optimization benchmark for SmolVLM-256M on Raspberry Pi 5
#
# Methods from papers in VLM_Memory_Optimization_Papers.xlsx:
#   1. Baseline (float32)             — already done
#   2. INT8 quantization (quanto)     — BnB/NF4 paper
#   3. INT4 quantization (quanto)     — BnB/NF4 paper
#   4. Magnitude pruning 20%          — classic L1
#   5. Magnitude pruning 40%          — classic L1
#   6. Wanda pruning 20%              — Sun et al. ICLR 2024
#   7. Wanda pruning 40%              — Sun et al. ICLR 2024
#   8. AWP: Wanda-20% + INT8          — MERL ICML 2025
#   9. SVD low-rank 50% rank          — SVD-LLM ICLR 2025
#  10. SVD low-rank 30% rank          — SVD-LLM ICLR 2025
#  11. Visual token reduction 50%     — PACT CVPR 2025
#  12. Visual token reduction 25%     — PACT CVPR 2025
#
# Usage:
#   bash scripts/run_rpi5_benchmark.sh                  # default 10 samples
#   VQAV2_N=50 bash scripts/run_rpi5_benchmark.sh       # more samples
#   bash scripts/run_rpi5_benchmark.sh --only svd        # single method
# =============================================================================

set -e
cd "$(dirname "$0")/.."

MODEL_ID="HuggingFaceTB/SmolVLM-256M-Instruct"
N="${VQAV2_N:-10}"
ONLY=""

for arg in "$@"; do
  case $arg in
    --only) ONLY_NEXT=1 ;;
    *) [ "${ONLY_NEXT:-0}" = "1" ] && ONLY="$arg" && ONLY_NEXT=0 ;;
  esac
done

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_step() {
  local name="$1"; shift
  if [ -n "$ONLY" ] && [ "$ONLY" != "$name" ]; then return; fi
  log "=== Starting: $name ==="
  python "$@"
  log "=== Done: $name ==="
  echo ""
}

# ── 1. Baseline (skip if already done) ────────────────────────────────────────
run_step "baseline" evaluation/run_baseline.py \
  --model_id "$MODEL_ID" \
  --vqav2_n "$N" --skip_textvqa --skip_pope

# ── 2. INT8 quantization (quanto) ─────────────────────────────────────────────
run_step "int8" compression/ptq/run_ptq_cpu.py \
  --model_id "$MODEL_ID" --quant int8 \
  --vqav2_n "$N" --skip_textvqa --skip_pope

# ── 3. INT4 quantization (quanto) ─────────────────────────────────────────────
run_step "int4" compression/ptq/run_ptq_cpu.py \
  --model_id "$MODEL_ID" --quant int4 \
  --vqav2_n "$N" --skip_textvqa --skip_pope

# ── 4. Magnitude pruning 20% ──────────────────────────────────────────────────
run_step "mag20" compression/pruning/run_pruning.py \
  --model_id "$MODEL_ID" --sparsity 0.20 \
  --vqav2_n "$N" --skip_textvqa --skip_pope

# ── 5. Magnitude pruning 40% ──────────────────────────────────────────────────
run_step "mag40" compression/pruning/run_pruning.py \
  --model_id "$MODEL_ID" --sparsity 0.40 \
  --vqav2_n "$N" --skip_textvqa --skip_pope

# ── 6. Wanda pruning 20% ──────────────────────────────────────────────────────
run_step "wanda20" compression/pruning/run_wanda.py \
  --model_id "$MODEL_ID" --sparsity 0.20 \
  --n_calib 16 \
  --vqav2_n "$N" --skip_textvqa --skip_pope

# ── 7. Wanda pruning 40% ──────────────────────────────────────────────────────
run_step "wanda40" compression/pruning/run_wanda.py \
  --model_id "$MODEL_ID" --sparsity 0.40 \
  --n_calib 16 \
  --vqav2_n "$N" --skip_textvqa --skip_pope

# ── 8. AWP: Wanda-20% + INT8 ──────────────────────────────────────────────────
run_step "awp" compression/combined/run_awp_cpu.py \
  --model_id "$MODEL_ID" --sparsity 0.20 \
  --n_calib 16 \
  --vqav2_n "$N" --skip_vqav2  # skip eval during combined — run separately

# ── 9. SVD 50% rank ───────────────────────────────────────────────────────────
run_step "svd50" compression/svd/run_svd_cpu.py \
  --model_id "$MODEL_ID" --rank_ratio 0.5 \
  --vqav2_n "$N"

# ── 10. SVD 30% rank ──────────────────────────────────────────────────────────
run_step "svd30" compression/svd/run_svd_cpu.py \
  --model_id "$MODEL_ID" --rank_ratio 0.3 \
  --vqav2_n "$N"

# ── 11. Visual token reduction 50% ────────────────────────────────────────────
run_step "token50" compression/token_pruning/run_visual_token_cpu.py \
  --model_id "$MODEL_ID" --keep_ratio 0.5 \
  --vqav2_n "$N"

# ── 12. Visual token reduction 25% ────────────────────────────────────────────
run_step "token25" compression/token_pruning/run_visual_token_cpu.py \
  --model_id "$MODEL_ID" --keep_ratio 0.25 \
  --vqav2_n "$N"

log "=== All methods complete. Run python scripts/show_rpi5_results.py to compare ==="
