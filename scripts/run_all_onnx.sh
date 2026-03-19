#!/usr/bin/env bash
# scripts/run_all_onnx.sh
# ONNX export + INT8 quantization for eligible models on GPU.
# Model cap: ≤ 16B parameters. ONNX export targets sub-500M models by default
# (larger models can be added but export takes significant time).

set -e
cd "$(dirname "$0")/.."

export PATH="/home/azaz/miniconda3/envs/vlm-bench/bin:$PATH"

# Sub-500M models — best candidates for ONNX edge deployment
MODELS=(
    # Florence-2 (microsoft/*): removed — not a conversational VLM, no VQA support
    "HuggingFaceTB/SmolVLM-256M-Instruct"
    "HuggingFaceTB/SmolVLM-500M-Instruct"
    # nanoVLM (lusxvr/*): HF repo broken — omitted
    # FastVLM  (apple/*): HF repo broken — omitted
    "LiquidAI/LFM2-VL-450M"
    "vikhyatk/moondream2"
    # ── Ovis2 sub-500M-class models for ONNX ──────────────────────────────
    "AIDC-AI/Ovis2-1B"
    # Gemma3 excluded from ONNX — no optimum export support yet
)

PROVIDER="${ONNX_PROVIDER:-CUDAExecutionProvider}"
VQAV2_N="${VQAV2_N:-1000}"

echo "=========================================="
echo " ONNX Export + INT8 Quantization"
echo " Provider : ${PROVIDER}"
echo " Models   : ${#MODELS[@]}"
echo "=========================================="

for MODEL_ID in "${MODELS[@]}"; do
    echo ""
    echo "[ONNX] ${MODEL_ID}"
    python compression/onnx/run_onnx.py \
        --model_id  "${MODEL_ID}" \
        --quantize \
        --provider  "${PROVIDER}" \
        --vqav2_n   "${VQAV2_N}" || echo "[FAIL] ${MODEL_ID}"
done

echo "All ONNX runs complete."
