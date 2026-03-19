#!/usr/bin/env bash
set -e

ENV_NAME="vlm-bench"
PYTHON_VERSION="3.10"
CUDA_VERSION="12.4"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

echo "=========================================="
echo " VLM Compression Benchmark — Environment Setup"
echo " CUDA target : ${CUDA_VERSION}"
echo "=========================================="

# ── 1. Create conda environment ────────────────────────────────────────────
echo ""
echo "[1/6] Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
echo "      Done."

# ── 2. Activate ────────────────────────────────────────────────────────────
echo ""
echo "[2/6] Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
echo "      Active: $(which python)  ($(python --version))"

# ── 3. Install PyTorch with CUDA 12.4 ─────────────────────────────────────
echo ""
echo "[3/6] Installing PyTorch (CUDA ${CUDA_VERSION})..."
pip install --upgrade pip wheel setuptools
pip install torch torchvision torchaudio --index-url "${TORCH_INDEX}"
echo "      Done."

# ── 4. Install remaining requirements (excluding torch) ───────────────────
echo ""
echo "[4/6] Installing project requirements..."
# Install everything except torch/torchvision (already installed with correct CUDA)
grep -v "^torch" requirements.txt | pip install -r /dev/stdin
echo "      Done."

# ── 5. Install flash-attention (optional, skip if build fails) ─────────────
echo ""
echo "[5/6] Attempting flash-attention install (optional, may skip)..."
pip install flash-attn --no-build-isolation 2>/dev/null && echo "      flash-attn installed." \
    || echo "      flash-attn skipped (not critical)."

# ── 6. Verify GPU + CUDA version ──────────────────────────────────────────
echo ""
echo "[6/6] Verifying GPU and CUDA ${CUDA_VERSION}..."
python - <<'PYEOF'
import torch, pynvml, sys

print(f"  PyTorch version : {torch.__version__}")
print(f"  CUDA available  : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("  WARNING: No CUDA GPU detected.")
    sys.exit(1)

cuda_ver = torch.version.cuda
print(f"  CUDA version    : {cuda_ver}")

if not cuda_ver.startswith("12.4"):
    print(f"  WARNING: Expected CUDA 12.4, got {cuda_ver}.")
    print(f"           Ensure your driver supports CUDA 12.4 (driver >= 550.x).")

print(f"  GPU count       : {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}           : {props.name}  ({props.total_memory / 1024**3:.1f} GB)")

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
info   = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(f"  VRAM free/total : {info.free/1024**3:.1f} / {info.total/1024**3:.1f} GB")
pynvml.nvmlShutdown()
print("  GPU check PASSED.")
PYEOF

echo ""
echo "=========================================="
echo " Setup complete. Activate with:"
echo "   conda activate ${ENV_NAME}"
echo "=========================================="
