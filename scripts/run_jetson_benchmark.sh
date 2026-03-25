#!/usr/bin/env bash
# scripts/run_jetson_benchmark.sh
# Full Jetson benchmark — model-by-model, smallest to largest
#
# Each model gets the full treatment: FP16 → INT8 → INT4 → Pruning
# No preflight rejection — let timeout/OOM be the natural signal.
#
# Usage:
#   1. Start tmux:    tmux new -s jetson
#   2. Run this:      bash scripts/run_jetson_benchmark.sh
#   3. Exit Claude and VS Code to free ~4GB
#   4. Reconnect:     tmux attach -t jetson
#   5. Check log:     tail -f results/jetson/benchmark.log

set -e
cd "$(dirname "$0")/.."

LOG=results/jetson/benchmark.log
mkdir -p results/jetson logs/jetson

echo "=== Jetson Benchmark Started: $(date) ===" | tee "$LOG"
echo "Available memory: $(grep MemAvailable /proc/meminfo | awk '{printf "%.0f MB", $2/1024}')" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Full benchmark: all families, all methods, model-by-model smallest→largest
# N=25 samples for reasonable speed while still getting accuracy signal
# --fp16_max_param_M 3000: skip FP16 for 4B+ models (OOM-killer risk)
#   but still run compression methods on them
# Advanced methods: SparseGPT, Wanda 50%, SVD-LLM
echo "=== FULL BENCHMARK (N=25, all families, all methods) ===" | tee -a "$LOG"
python3 jetson/run_jetson.py \
    --n_samples 25 \
    --fp16_max_param_M 3000 \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Benchmark Complete: $(date) ===" | tee -a "$LOG"
echo "Results in: results/jetson/" | tee -a "$LOG"
