#!/usr/bin/env bash
# scripts/run_all_on_model.sh
# ============================
# Run ALL compression methods on a SINGLE model for side-by-side comparison.
#
# Methods (10 total):
#   1. Baseline FP16     (no compression)
#   2. BnB INT8          (2x compression)
#   3. BnB INT4          (4x compression)
#   4. AWQ INT4          (4x, activation-aware)
#   5. GPTQ INT4         (4x, Hessian-based)
#   6. SVD-LLM 50%       (low-rank decomposition)
#   7. PALU 25%          (KV-cache compression)
#   8. SparseGPT 50%     (unstructured pruning)
#   9. Wanda 50%         (pruning without backprop)
#  10. QLoRA r16         (INT4 + LoRA adapters)
#  11. QLoRA r64         (INT4 + LoRA adapters, higher rank)
#
# Default model: Qwen/Qwen2.5-VL-3B-Instruct (3.7B params)
#
# Usage:
#   bash scripts/run_all_on_model.sh
#   bash scripts/run_all_on_model.sh "Qwen/Qwen2.5-VL-7B-Instruct"
#   MODEL_ID="LiquidAI/LFM2-VL-3B" N_SAMPLES=500 bash scripts/run_all_on_model.sh

set -e
cd "$(dirname "$0")/.."

MODEL_ID="${1:-${MODEL_ID:-Qwen/Qwen2.5-VL-3B-Instruct}}"
N_SAMPLES="${N_SAMPLES:-500}"
TIMEOUT="${TIMEOUT:-2400}"  # 40 min max per method
SHORT=$(echo "$MODEL_ID" | sed 's|.*/||')
LOG="results/single_model_${SHORT}.log"

mkdir -p results/{baseline,ptq,awq,gptq,svd_llm,palu,sparsegpt,wanda,qlora}

TOTAL_METHODS=11
METHOD_IDX=0

echo "╔══════════════════════════════════════════════════════════════════╗" | tee "$LOG"
echo "║  All-Methods Benchmark: $SHORT" | tee -a "$LOG"
echo "║  Model: $MODEL_ID" | tee -a "$LOG"
echo "║  Eval samples: $N_SAMPLES (VQAv2)" | tee -a "$LOG"
echo "║  Started: $(date)" | tee -a "$LOG"
echo "╚══════════════════════════════════════════════════════════════════╝" | tee -a "$LOG"

run_method() {
    local idx=$1
    local name=$2
    shift 2
    echo "" | tee -a "$LOG"
    echo "  ── [$idx/$TOTAL_METHODS] $name ──" | tee -a "$LOG"
    timeout $TIMEOUT "$@" \
        2>&1 | tee -a "$LOG" || echo "  ⚠ $name FAILED (exit=$?)" | tee -a "$LOG"
    sleep 5
}

# ── 1. Baseline FP16 ───────────────────────────────────────────────
METHOD_IDX=$((METHOD_IDX + 1))
run_method $METHOD_IDX "Baseline FP16" \
    python3 evaluation/run_baseline.py \
        --model_id "$MODEL_ID" \
        --quant fp16 \
        --vqav2_n $N_SAMPLES \
        --skip_textvqa --skip_pope \
        --force

# ── 2. BnB INT8 ────────────────────────────────────────────────────
METHOD_IDX=$((METHOD_IDX + 1))
run_method $METHOD_IDX "BnB INT8" \
    python3 compression/ptq/run_ptq.py \
        --model_id "$MODEL_ID" \
        --quant int8 --backend bnb \
        --vqav2_n $N_SAMPLES \
        --skip_textvqa --skip_pope \
        --force

# ── 3. BnB INT4 ────────────────────────────────────────────────────
METHOD_IDX=$((METHOD_IDX + 1))
run_method $METHOD_IDX "BnB INT4" \
    python3 compression/ptq/run_ptq.py \
        --model_id "$MODEL_ID" \
        --quant int4 --backend bnb \
        --vqav2_n $N_SAMPLES \
        --skip_textvqa --skip_pope \
        --force

# ── 4. AWQ INT4 ────────────────────────────────────────────────────
METHOD_IDX=$((METHOD_IDX + 1))
run_method $METHOD_IDX "AWQ INT4" \
    python3 compression/ptq/run_awq.py \
        --model_id "$MODEL_ID" \
        --w_bit 4 \
        --vqav2_n $N_SAMPLES \
        --force

# ── 5. GPTQ INT4 ───────────────────────────────────────────────────
METHOD_IDX=$((METHOD_IDX + 1))
run_method $METHOD_IDX "GPTQ INT4" \
    python3 compression/ptq/run_gptq.py \
        --model_id "$MODEL_ID" \
        --bits 4 \
        --vqav2_n $N_SAMPLES \
        --force

# ── 6. SVD-LLM 50% ─────────────────────────────────────────────────
METHOD_IDX=$((METHOD_IDX + 1))
run_method $METHOD_IDX "SVD-LLM 50%" \
    python3 compression/lowrank/run_svd_llm.py \
        --model_id "$MODEL_ID" \
        --rank_ratio 0.50 \
        --vqav2_n $N_SAMPLES \
        --force

# ── 7. PALU 25% ────────────────────────────────────────────────────
METHOD_IDX=$((METHOD_IDX + 1))
run_method $METHOD_IDX "PALU 25%" \
    python3 compression/palu/run_palu.py \
        --model_id "$MODEL_ID" \
        --rank_ratio 0.25 \
        --vqav2_n $N_SAMPLES \
        --force

# ── 8. SparseGPT 50% ───────────────────────────────────────────────
METHOD_IDX=$((METHOD_IDX + 1))
run_method $METHOD_IDX "SparseGPT 50%" \
    python3 compression/pruning/run_sparsegpt.py \
        --model_id "$MODEL_ID" \
        --sparsity 0.50 \
        --vqav2_n $N_SAMPLES \
        --force

# ── 9. Wanda 50% ───────────────────────────────────────────────────
METHOD_IDX=$((METHOD_IDX + 1))
run_method $METHOD_IDX "Wanda 50%" \
    python3 compression/pruning/run_wanda.py \
        --model_id "$MODEL_ID" \
        --sparsity 0.50 \
        --vqav2_n $N_SAMPLES \
        --force

# ── 10. QLoRA rank 16 ──────────────────────────────────────────────
METHOD_IDX=$((METHOD_IDX + 1))
run_method $METHOD_IDX "QLoRA r16" \
    python3 compression/qlora/run_qlora.py \
        --model_id "$MODEL_ID" \
        --lora_rank 16 \
        --train_samples 200 \
        --vqav2_n $N_SAMPLES \
        --force

# ── 11. QLoRA rank 64 ──────────────────────────────────────────────
METHOD_IDX=$((METHOD_IDX + 1))
run_method $METHOD_IDX "QLoRA r64" \
    python3 compression/qlora/run_qlora.py \
        --model_id "$MODEL_ID" \
        --lora_rank 64 \
        --train_samples 200 \
        --vqav2_n $N_SAMPLES \
        --force

# ── Summary ─────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "╔══════════════════════════════════════════════════════════════════╗" | tee -a "$LOG"
echo "║  Benchmark Complete: $(date)" | tee -a "$LOG"
echo "╚══════════════════════════════════════════════════════════════════╝" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== RESULTS COMPARISON: $SHORT ===" | tee -a "$LOG"

python3 -c "
import json, os, glob

model_id = '$MODEL_ID'
short = model_id.split('/')[-1]
safe = model_id.replace('/', '__')

# Collect all results for this model
results = []

# Baseline
for f in glob.glob('results/baseline/*.json'):
    with open(f) as fh:
        d = json.load(fh)
    if d.get('model_id') == model_id:
        bm = d.get('benchmarks', {}).get('vqav2', {})
        results.append({
            'method': f\"Baseline ({d.get('quant', 'fp16')})\",
            'acc': bm.get('accuracy', 0),
            'lat': bm.get('avg_latency_s', 0),
            'mem': d.get('gpu_mem_load_mb', 0),
            'peak_mem': bm.get('peak_memory_mb', 0),
        })

# Compression methods
search_dirs = {
    'results/ptq': lambda d: f\"{d.get('quant','?')}_{d.get('backend','bnb')}\",
    'results/awq': lambda d: f\"AWQ w{d.get('w_bit', 4)}\",
    'results/gptq': lambda d: f\"GPTQ w{d.get('bits', 4)}\",
    'results/svd_llm': lambda d: f\"SVD-LLM {d.get('rank_ratio', 0.5):.0%}\",
    'results/palu': lambda d: f\"PALU {d.get('rank_ratio', 0.25):.0%}\",
    'results/sparsegpt': lambda d: f\"SparseGPT {d.get('sparsity', 0.5):.0%}\",
    'results/wanda': lambda d: f\"Wanda {d.get('sparsity', 0.5):.0%}\",
    'results/qlora': lambda d: f\"QLoRA r{d.get('lora_rank', 16)}\",
}

for dir_path, method_fn in search_dirs.items():
    if not os.path.isdir(dir_path):
        continue
    for f in sorted(os.listdir(dir_path)):
        if not f.endswith('.json'):
            continue
        fp = os.path.join(dir_path, f)
        with open(fp) as fh:
            d = json.load(fh)
        if d.get('model_id') != model_id:
            continue
        bm = d.get('benchmarks', {}).get('vqav2', {})
        if not bm:
            continue
        results.append({
            'method': method_fn(d),
            'acc': bm.get('accuracy', 0),
            'lat': bm.get('avg_latency_s', 0),
            'mem': d.get('gpu_mem_load_mb', 0),
            'peak_mem': bm.get('peak_memory_mb', 0),
        })

if not results:
    print('No results found.')
else:
    # Sort by accuracy descending
    results.sort(key=lambda x: x['acc'], reverse=True)

    baseline_acc = max((r['acc'] for r in results if 'Baseline' in r['method']), default=0)

    print(f\"{'Method':<22} {'Accuracy':>8} {'vs Base':>8} {'Latency':>8} {'Load MB':>8} {'Peak MB':>8}\")
    print('─' * 76)
    for r in results:
        delta = r['acc'] - baseline_acc if baseline_acc else 0
        delta_s = f\"{delta:+.4f}\" if 'Baseline' not in r['method'] else '   ref'
        print(f\"{r['method']:<22} {r['acc']:>8.4f} {delta_s:>8} {r['lat']:>7.2f}s {r['mem']:>7.0f} {r['peak_mem']:>7.0f}\")

    print()
    print(f'Total methods evaluated: {len(results)}')
    if baseline_acc:
        best = max((r for r in results if 'Baseline' not in r['method']), key=lambda x: x['acc'], default=None)
        if best:
            print(f\"Best compression: {best['method']} (acc={best['acc']:.4f}, {best['acc']-baseline_acc:+.4f} vs baseline)\")
        most_compact = min((r for r in results), key=lambda x: x['mem'] if x['mem'] > 0 else 1e9, default=None)
        if most_compact:
            print(f\"Smallest memory: {most_compact['method']} ({most_compact['mem']:.0f} MB load)\")
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Full log: $LOG" | tee -a "$LOG"
