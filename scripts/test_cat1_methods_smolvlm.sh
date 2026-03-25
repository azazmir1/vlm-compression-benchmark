#!/bin/bash
# ============================================================
# Test ALL Category 1 methods on SmolVLM-2.2B (the ceiling model)
# Target: HuggingFaceTB/SmolVLM-Instruct (2.2B params, OOMs in FP16)
# ============================================================
set -o pipefail

MODEL="HuggingFaceTB/SmolVLM-Instruct"
LOG="results/cat1_smolvlm_test_$(date +%Y%m%d_%H%M%S).log"
N_EVAL=10  # small sample for quick test

echo "============================================================" | tee "$LOG"
echo "  CAT1 METHOD TEST: $MODEL" | tee -a "$LOG"
echo "  Date: $(date)" | tee -a "$LOG"
echo "  Device: $(uname -m) / $(cat /etc/nv_tegra_release 2>/dev/null | head -1)" | tee -a "$LOG"
echo "  Memory: $(free -m | grep Mem | awk '{print $7}') MB available" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

run_method() {
    local method_name="$1"
    shift
    echo "" | tee -a "$LOG"
    echo ">>> [$method_name] Starting at $(date +%H:%M:%S)" | tee -a "$LOG"
    echo "    Command: $@" | tee -a "$LOG"
    free -m | grep Mem | tee -a "$LOG"

    timeout 300 "$@" 2>&1 | tee -a "$LOG"
    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        echo ">>> [$method_name] SUCCESS (exit=$exit_code)" | tee -a "$LOG"
    elif [ $exit_code -eq 124 ]; then
        echo ">>> [$method_name] TIMEOUT (5 min limit)" | tee -a "$LOG"
    elif [ $exit_code -eq 137 ]; then
        echo ">>> [$method_name] KILLED (OOM)" | tee -a "$LOG"
    else
        echo ">>> [$method_name] FAILED (exit=$exit_code)" | tee -a "$LOG"
    fi

    # Cleanup memory
    sleep 5
    sync
    sleep 3
    echo "    Memory after cleanup: $(free -m | grep Mem | awk '{print $7}') MB" | tee -a "$LOG"
    return $exit_code
}

echo ""
echo "============================================================" | tee -a "$LOG"
echo "  METHOD 1: GPTQ (own quantization)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
run_method "GPTQ" python3 compression/ptq/run_gptq.py \
    --model_id "$MODEL" --bits 4 --n_calib 32 --vqav2_n $N_EVAL --force

echo ""
echo "============================================================" | tee -a "$LOG"
echo "  METHOD 2: AWQ (own quantization)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
run_method "AWQ" python3 compression/ptq/run_awq.py \
    --model_id "$MODEL" --w_bit 4 --n_calib 32 --vqav2_n $N_EVAL --force

echo ""
echo "============================================================" | tee -a "$LOG"
echo "  METHOD 3: SVD-LLM (rank_ratio=0.30)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
run_method "SVD-LLM" python3 compression/lowrank/run_svd_llm.py \
    --model_id "$MODEL" --rank_ratio 0.30 --vqav2_n $N_EVAL --force

echo ""
echo "============================================================" | tee -a "$LOG"
echo "  METHOD 4: AWP (Wanda + INT4 combined)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
run_method "AWP" python3 compression/combined/run_awp.py \
    --model_id "$MODEL" --sparsity 0.50 --vqav2_n $N_EVAL --force

echo ""
echo "============================================================" | tee -a "$LOG"
echo "  METHOD 5: CASP (VLM-specific)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
run_method "CASP" python3 compression/casp_slim/run_casp_slim.py \
    --model_id "$MODEL" --method casp --vqav2_n $N_EVAL --force

echo ""
echo "============================================================" | tee -a "$LOG"
echo "  METHOD 6: SLIM (triple compression)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
run_method "SLIM" python3 compression/casp_slim/run_casp_slim.py \
    --model_id "$MODEL" --method slim --vqav2_n $N_EVAL --force

echo ""
echo "============================================================" | tee -a "$LOG"
echo "  METHOD 7: QLoRA (INT4 base + LoRA)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
run_method "QLoRA" python3 compression/qlora/run_qlora.py \
    --model_id "$MODEL" --lora_rank 16 --vqav2_n $N_EVAL --force

echo ""
echo "============================================================" | tee -a "$LOG"
echo "  ALL METHODS COMPLETE" | tee -a "$LOG"
echo "  Log: $LOG" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

# Summary
echo ""
echo "=== SUMMARY ===" | tee -a "$LOG"
grep -E ">>> \[.*\] (SUCCESS|FAILED|KILLED|TIMEOUT)" "$LOG" | tee -a "$LOG"
