#!/usr/bin/env bash
# scripts/smoke_test.sh
# Quick end-to-end smoke test — runs 10 VQAv2 samples on one small model
# from each family to verify the full pipeline works before committing GPU time.
#
# Usage:
#   bash scripts/smoke_test.sh              # test all 8 families
#   bash scripts/smoke_test.sh florence2   # test one family only

set -e
cd "$(dirname "$0")/.."

export PATH="/home/azaz/miniconda3/envs/vlm-bench/bin:$PATH"

N=10   # samples per benchmark — just enough to validate the pipeline

# One representative (smallest) model per family
declare -A FAMILY_MODEL=(
    [smolvlm]="HuggingFaceTB/SmolVLM-256M-Instruct"
    [lfm2vl]="LiquidAI/LFM2-VL-450M"
    [moondream]="vikhyatk/moondream2"
    [qwen25vl]="Qwen/Qwen2.5-VL-3B-Instruct"
    [internvl25]="OpenGVLab/InternVL2_5-1B"
    [gemma3]="google/gemma-3-4b-it"
    [ovis2]="AIDC-AI/Ovis2-1B"
    # nanovlm (lusxvr/nanoVLM-222M): removed — HF repo missing model_type in config.json
    # fastvlm  (apple/FastVLM-*):    removed — HF repo missing preprocessor_config.json
    # florence2 (microsoft/*):       removed — not a conversational VLM, no VQA support
)

FILTER="${1:-}"   # optional: test only one family

PASS=0; FAIL=0

echo "=========================================="
echo " Smoke Test  (n=${N} samples per benchmark)"
echo "=========================================="

for FAMILY in "${!FAMILY_MODEL[@]}"; do
    if [ -n "${FILTER}" ] && [ "${FILTER}" != "${FAMILY}" ]; then
        continue
    fi

    MODEL_ID="${FAMILY_MODEL[$FAMILY]}"
    echo ""
    echo "[TEST] ${FAMILY}  →  ${MODEL_ID}"

    # Use a temp output dir so smoke test results don't pollute real results
    TMPDIR=$(mktemp -d)
    trap "rm -rf ${TMPDIR}" EXIT

    python evaluation/run_baseline.py \
            --model_id      "${MODEL_ID}" \
            --vqav2_n       "${N}" \
            --skip_textvqa \
            --skip_pope \
            --force 2>&1 | tee "${TMPDIR}/${FAMILY}.log"
    EXIT_CODE=${PIPESTATUS[0]}
    tail -5 "${TMPDIR}/${FAMILY}.log"
    if [ "${EXIT_CODE}" -eq 0 ]; then
        echo "[PASS] ${FAMILY}"
        ((PASS++)) || true
    else
        echo "[FAIL] ${FAMILY}  (exit ${EXIT_CODE}) — log: ${TMPDIR}/${FAMILY}.log"
        ((FAIL++)) || true
    fi
done

echo ""
echo "=========================================="
echo " Smoke test done.  PASS=${PASS}  FAIL=${FAIL}"
if [ "${FAIL}" -gt 0 ]; then
    echo " Fix failures before running full experiments."
    exit 1
fi
echo "=========================================="
