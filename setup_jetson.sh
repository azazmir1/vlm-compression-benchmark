#!/usr/bin/env bash
# setup_jetson.sh — Jetson Orin Nano 8 GB (JetPack 6.x, CUDA 12.6, aarch64)
# Assumes PyTorch + CUDA are already installed by JetPack / NVIDIA pip wheels.
# Uses system pip3 (no conda).
set -e

echo "=========================================="
echo " VLM Compression Benchmark — Jetson Setup"
echo " JetPack 6.x | CUDA 12.6 | aarch64"
echo "=========================================="

# ── 0. Verify PyTorch + CUDA already present ──────────────────────────────
echo ""
echo "[0/5] Checking existing PyTorch + CUDA installation..."
python3 - <<'PYEOF'
import sys, torch
print(f"  PyTorch  : {torch.__version__}")
print(f"  CUDA ok  : {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("  ERROR: CUDA not available. Install JetPack PyTorch wheels first.")
    sys.exit(1)
print(f"  CUDA ver : {torch.version.cuda}")
print(f"  Device   : {torch.cuda.get_device_name(0)}")
PYEOF

# ── 1. Core dependencies already satisfied on this Jetson ─────────────────
# torch, torchvision, transformers, accelerate, autoawq, datasets are present.
echo ""
echo "[1/5] Upgrading pip/setuptools/wheel..."
pip3 install --upgrade pip wheel setuptools

# ── 2. Compression libraries ──────────────────────────────────────────────
echo ""
echo "[2/5] Installing compression libraries..."

# bitsandbytes ≥ 0.43 ships multi-backend support including Jetson/aarch64
pip3 install "bitsandbytes>=0.43.0"

# auto-gptq — try pre-built wheel; falls back to source build automatically
pip3 install "auto-gptq>=0.7.1" --no-build-isolation 2>/dev/null \
    && echo "  auto-gptq installed (pre-built)" \
    || { echo "  auto-gptq pre-built failed, trying source build...";
         pip3 install "auto-gptq>=0.7.1" --no-build-isolation; }

# llmcompressor
pip3 install "llmcompressor>=0.4.0"

echo "  Compression libraries done."

# ── 3. Evaluation & utility packages ──────────────────────────────────────
echo ""
echo "[3/5] Installing evaluation & utility packages..."

pip3 install \
    "lm-eval>=0.4.3" \
    "evaluate>=0.4.1" \
    "tokenizers>=0.19.0" \
    "pynvml>=11.5.0" \
    "py3nvml>=0.2.7" \
    "seaborn>=0.13.0" \
    "plotly>=5.21.0" \
    "jupyter>=1.0.0" \
    "ipywidgets>=8.1.0" \
    "opencv-python-headless>=4.9.0" \
    "sentencepiece>=0.2.0" \
    "loguru>=0.7.2" \
    "jsonlines>=4.0.0"

echo "  Evaluation & utility packages done."

# ── 4. Final verification ─────────────────────────────────────────────────
echo ""
echo "[4/5] Verifying key packages..."
python3 - <<'PYEOF'
import torch, sys

checks = {
    "torch"          : lambda: torch.__version__,
    "transformers"   : lambda: __import__("transformers").__version__,
    "accelerate"     : lambda: __import__("accelerate").__version__,
    "bitsandbytes"   : lambda: __import__("bitsandbytes").__version__,
    "autoawq"        : lambda: __import__("awq").__version__,
    "datasets"       : lambda: __import__("datasets").__version__,
    "lm_eval"        : lambda: __import__("lm_eval").__version__,
    "pynvml"         : lambda: __import__("pynvml").__version__,
}

all_ok = True
for name, fn in checks.items():
    try:
        ver = fn()
        print(f"  {name:<20} {ver}")
    except Exception as e:
        print(f"  {name:<20} MISSING ({e})")
        all_ok = False

print()
print(f"  CUDA available : {torch.cuda.is_available()}")
print(f"  CUDA version   : {torch.version.cuda}")
print(f"  GPU            : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

try:
    import pynvml
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
    print(f"  Unified memory : {mem.free/1024**3:.1f} free / {mem.total/1024**3:.1f} GB total")
    pynvml.nvmlShutdown()
except Exception as e:
    print(f"  pynvml check   : {e}")

if all_ok:
    print("\n  Setup PASSED.")
else:
    print("\n  Setup completed with some missing packages (see above).")
PYEOF

echo ""
echo "=========================================="
echo " Jetson setup complete."
echo " Run experiments with:  python3 evaluation/run_baseline.py ..."
echo "=========================================="
