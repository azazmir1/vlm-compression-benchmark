#!/usr/bin/env bash
# scripts/run_baseline_careful.sh
# Run FP16 baseline for each model ONE AT A TIME with GPU cleanup.
# Usage: bash scripts/run_baseline_careful.sh [N_SAMPLES]
#   N_SAMPLES defaults to 1

set -e
cd "$(dirname "$0")/.."

N=${1:-1}
LOG="results/jetson/baseline_n${N}.log"
mkdir -p results/jetson/baseline

echo "=== Baseline Run (N=$N) started: $(date) ===" | tee "$LOG"
echo "Available memory: $(grep MemAvailable /proc/meminfo | awk '{printf "%.0f MB", $2/1024}')" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# All models sorted by size
MODELS=(
    # nanoVLM-222M/450M dropped: 450M repo doesn't exist, 222M has no tokenizer/processor
    "HuggingFaceTB/SmolVLM-256M-Instruct:smolvlm:256"
    "LiquidAI/LFM2-VL-450M:lfm2vl:450"
    "HuggingFaceTB/SmolVLM-500M-Instruct:smolvlm:500"
    "apple/FastVLM-0.5B:fastvlm:500"
    "OpenGVLab/InternVL2_5-1B:internvl25:1000"
    "AIDC-AI/Ovis2-1B:ovis2:1000"
    "apple/FastVLM-1.5B:fastvlm:1500"
    "LiquidAI/LFM2-VL-1.6B:lfm2vl:1600"
    "vikhyatk/moondream2:moondream:2000"
    "OpenGVLab/InternVL2_5-2B:internvl25:2000"
    "AIDC-AI/Ovis2-2B:ovis2:2000"
    "HuggingFaceTB/SmolVLM-Instruct:smolvlm:2200"
    "LiquidAI/LFM2-VL-3B:lfm2vl:3000"
    "Qwen/Qwen2.5-VL-3B-Instruct:qwen25vl:3000"
    "OpenGVLab/InternVL2_5-4B:internvl25:4000"
    "google/gemma-3-4b-it:gemma3:4000"
    "AIDC-AI/Ovis2-4B:ovis2:4000"
)

TOTAL=${#MODELS[@]}
IDX=0

for entry in "${MODELS[@]}"; do
    IFS=':' read -r MODEL_ID FAMILY PARAM_M <<< "$entry"
    IDX=$((IDX + 1))
    SHORT=$(echo "$MODEL_ID" | sed 's|.*/||')

    echo "" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
    echo "[$IDX/$TOTAL] $SHORT (${PARAM_M}M) — family: $FAMILY" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"

    # Pre-check: GPU/memory state
    MEM_AVAIL=$(grep MemAvailable /proc/meminfo | awk '{printf "%.0f", $2/1024}')
    echo "  Memory before: ${MEM_AVAIL} MB available" | tee -a "$LOG"

    # Run single model in isolated subprocess via python
    # --force to overwrite old results, --scan_only to only do FP16
    # --max_param_M skips models that are clearly too big (>= 5000M)
    python3 -c "
import sys, json, time, gc, os, signal
sys.path.insert(0, '.')

SAMPLE_TIMEOUT = 60  # seconds per sample — skip if exceeded

class SampleTimeout(Exception):
    pass

def _timeout_handler(signum, frame):
    raise SampleTimeout('Sample exceeded ${SAMPLE_TIMEOUT:-60}s timeout')

# Force garbage collection before starting
gc.collect()

import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

from models.model_loader import load_model, unload_model
from evaluation.run_baseline import load_vqav2, run_inference, _vqa_accuracy

model_id = '$MODEL_ID'
family = '$FAMILY'
param_M = $PARAM_M
n_samples = $N

print(f'  Loading {model_id} @ fp16 ...')
t0 = time.time()

try:
    model, processor, meta = load_model(model_id, quant='fp16', family=family)
except Exception as e:
    print(f'  LOAD FAILED: {e}')
    result = {
        'model_id': model_id, 'family': family, 'param_M': param_M,
        'precision': 'fp16', 'status': 'ERROR',
        'metrics': {'status': 'ERROR', 'error_msg': str(e)},
        'device': 'jetson_orin_nano_8gb',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    safe = model_id.replace('/', '__')
    with open(f'results/jetson/baseline/{safe}.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    sys.exit(0)

load_time = time.time() - t0
device = str(next(model.parameters()).device)
mem_used = meta.gpu_mem_delta_mb
print(f'  Loaded in {load_time:.1f}s | mem delta: {mem_used:.0f} MB')

# Check memory state
mem_avail = 0
try:
    with open('/proc/meminfo') as f:
        for line in f:
            if 'MemAvailable' in line:
                mem_avail = int(line.split()[1]) / 1024
                break
except:
    pass
print(f'  Memory available after load: {mem_avail:.0f} MB')

if mem_avail < 300:
    print(f'  WARNING: Memory critically low ({mem_avail:.0f} MB) — skipping eval')
    result = {
        'model_id': model_id, 'family': family, 'param_M': param_M,
        'precision': 'fp16', 'status': 'MEM_CRITICAL',
        'metrics': {'status': 'MEM_CRITICAL', 'error_msg': f'Only {mem_avail:.0f} MB free after load'},
        'gpu_mem_load_mb': round(mem_used, 1),
        'device': 'jetson_orin_nano_8gb',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    safe = model_id.replace('/', '__')
    with open(f'results/jetson/baseline/{safe}.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    unload_model(model)
    del model, processor, meta
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.exit(0)

# Evaluate
print(f'  Evaluating on {n_samples} VQAv2 samples (timeout={SAMPLE_TIMEOUT}s/sample) ...')
samples = load_vqav2(n_samples=n_samples)

scores = []
latencies = []
skipped = 0
oom_count = 0
error_count = 0

for idx, sample in enumerate(samples):
    # Set per-sample alarm
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(SAMPLE_TIMEOUT)
    try:
        t1 = time.time()
        with torch.no_grad():
            pred = run_inference(model, processor, sample, family, device)
        lat = time.time() - t1
        signal.alarm(0)  # cancel alarm on success

        acc = _vqa_accuracy(pred, sample['answers'])
        scores.append(acc)
        latencies.append(lat)
        print(f'    Sample {idx}: acc={acc:.2f} lat={lat:.2f}s pred=\"{str(pred)[:50]}\"')

    except SampleTimeout:
        signal.alarm(0)
        print(f'    Sample {idx}: TIMEOUT (>{SAMPLE_TIMEOUT}s) — skipping')
        skipped += 1
    except torch.cuda.OutOfMemoryError as e:
        signal.alarm(0)
        print(f'    Sample {idx}: OOM — {e}')
        oom_count += 1
        torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        signal.alarm(0)
        print(f'    Sample {idx}: ERROR — {e}')
        error_count += 1
    finally:
        signal.alarm(0)
        # Aggressively clear KV cache / CUDA context after EVERY sample
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

signal.alarm(0)  # ensure no pending alarm

avg_acc = sum(scores) / len(scores) if scores else 0.0
avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

# Peak memory
peak_mem = 0
if torch.cuda.is_available():
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2

# Determine status
if len(scores) == 0:
    if oom_count > 0:
        status = 'OOM_INFER'
    elif skipped == n_samples:
        status = 'TOO_SLOW'
    else:
        status = 'ERROR'
elif oom_count > 0 or skipped > 0:
    status = 'PARTIAL'  # some succeeded, some failed
else:
    status = 'PASS'

error_parts = []
if skipped > 0:
    error_parts.append(f'{skipped} timed out')
if oom_count > 0:
    error_parts.append(f'{oom_count} OOM')
if error_count > 0:
    error_parts.append(f'{error_count} errors')
error_msg = '; '.join(error_parts) if error_parts else None

print(f'  Result: {status} | acc={avg_acc:.4f} | lat={avg_lat:.2f}s | peak_mem={peak_mem:.0f}MB | {len(scores)}/{n_samples} ok, {skipped} timeout, {oom_count} oom')

result = {
    'model_id': model_id, 'family': family, 'param_M': param_M,
    'precision': 'fp16', 'status': status,
    'num_params_M': round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
    'gpu_mem_load_mb': round(mem_used, 1),
    'metrics': {
        'status': status,
        'accuracy': round(avg_acc, 4),
        'avg_latency_s': round(avg_lat, 4),
        'peak_memory_mb': round(peak_mem, 1),
        'n_evaluated': len(scores),
        'n_skipped_timeout': skipped,
        'n_oom': oom_count,
        'n_errors': error_count,
        'n_total': n_samples,
        'error_msg': error_msg,
    },
    'device': 'jetson_orin_nano_8gb',
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
}

safe = model_id.replace('/', '__')
with open(f'results/jetson/baseline/{safe}.json', 'w') as f:
    json.dump(result, f, indent=2, default=str)
print(f'  Saved → results/jetson/baseline/{safe}.json')

# Cleanup
unload_model(model)
del model, processor, meta
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print('  Model unloaded, GPU cleaned.')
" 2>&1 | tee -a "$LOG"

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "  ⚠ Process exited with code $EXIT_CODE (timeout or crash)" | tee -a "$LOG"
    fi

    # Post-cleanup: verify GPU is free
    sleep 5
    MEM_AFTER=$(grep MemAvailable /proc/meminfo | awk '{printf "%.0f", $2/1024}')
    echo "  Memory after cleanup: ${MEM_AFTER} MB available" | tee -a "$LOG"

    # If memory is low after cleanup, wait longer
    if [ "$MEM_AFTER" -lt 3000 ]; then
        echo "  Memory low — waiting 15s for OS reclaim..." | tee -a "$LOG"
        sleep 15
        MEM_AFTER2=$(grep MemAvailable /proc/meminfo | awk '{printf "%.0f", $2/1024}')
        echo "  Memory after extra wait: ${MEM_AFTER2} MB available" | tee -a "$LOG"
    fi
done

echo "" | tee -a "$LOG"
echo "=== Baseline Run Complete: $(date) ===" | tee -a "$LOG"

# Print summary table
echo "" | tee -a "$LOG"
echo "=== SUMMARY ===" | tee -a "$LOG"
python3 -c "
import json, os
base = 'results/jetson/baseline'
print(f\"{'Model':<42} {'Status':<15} {'Acc':>6} {'Lat':>7} {'MemLoad':>8} {'Params':>8}\")
print(f\"{'─'*42} {'─'*15} {'─'*6} {'─'*7} {'─'*8} {'─'*8}\")
for f in sorted(os.listdir(base)):
    if not f.endswith('.json'): continue
    with open(os.path.join(base, f)) as fh:
        d = json.load(fh)
    st = d.get('status', '?')
    m = d.get('metrics', {})
    name = f.replace('.json','').replace('__','/')
    short = name.split('/')[-1][:40]
    acc = f\"{m.get('accuracy', 0):.4f}\" if st == 'PASS' else '  -'
    lat = f\"{m.get('avg_latency_s', 0):.2f}s\" if st == 'PASS' else '   -'
    mem = f\"{d.get('gpu_mem_load_mb', 0):.0f}MB\" if d.get('gpu_mem_load_mb') else '   -'
    params = f\"{d.get('num_params_M', 0):.0f}M\" if d.get('num_params_M') else '   -'
    print(f'{short:<42} {st:<15} {acc:>6} {lat:>7} {mem:>8} {params:>8}')
" 2>&1 | tee -a "$LOG"
