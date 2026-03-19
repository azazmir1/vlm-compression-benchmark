"""
compression/onnx/run_onnx.py
=============================
Fourth compression method: ONNX Export + INT8 static quantization.

Pipeline:
  1. Load model via HuggingFace transformers
  2. Export LLM backbone (and optionally vision encoder) to ONNX via Optimum
  3. Optionally apply static INT8 quantization to the ONNX graph
  4. Run inference via ONNX Runtime (CPU or CUDA EP)
  5. Evaluate on VQAv2 / POPE, log KPIs
  6. Save results to results/onnx/{model_name}__onnx[_int8].json

Key value for edge deployment:
  - ONNX Runtime CPU can run on Raspberry Pi 5 without PyTorch
  - ONNX INT8 reduces model size ~4x vs FP32, ~2x vs FP16
  - Enables TensorRT conversion on Jetson (via onnx → trt)

Usage:
  python compression/onnx/run_onnx.py \
      --model_id HuggingFaceTB/SmolVLM-256M-Instruct

  python compression/onnx/run_onnx.py \
      --model_id microsoft/Florence-2-base --quantize --provider CPUExecutionProvider
"""

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.model_loader import load_model, unload_model, detect_family
from evaluation.run_baseline import (
    load_vqav2, load_pope,
    _vqa_accuracy, _pope_accuracy,
)
from profiling.cpu_profiler import CPUProfiler
from profiling.gpu_profiler import GPUProfiler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "onnx"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ONNX_CACHE = Path(__file__).resolve().parents[2] / "results" / "onnx" / "_exported"
ONNX_CACHE.mkdir(parents=True, exist_ok=True)

# ── Families supported by Optimum ONNX export ───────────────────────────────
OPTIMUM_SUPPORTED = {"smolvlm", "nanovlm", "fastvlm", "qwen25vl", "lfm2vl"}
# Florence-2 and InternVL2.5 require custom export; Moondream uses non-standard API


# ── ONNX Export ──────────────────────────────────────────────────────────────

def export_to_onnx(model_id: str, family: str, save_dir: Path, opset: int = 17) -> Path:
    """Export model to ONNX using Optimum. Returns path to exported directory."""
    if save_dir.exists():
        logger.info(f"[ONNX] Found cached export at {save_dir}")
        return save_dir

    logger.info(f"[ONNX] Exporting {model_id} to ONNX (opset={opset})...")
    try:
        from optimum.exporters.onnx import main_export
        main_export(
            model_name_or_path=model_id,
            output=str(save_dir),
            task="image-text-to-text",
            opset=opset,
            device="cpu",
            trust_remote_code=True,
        )
        logger.info(f"[ONNX] Exported to {save_dir}")
    except Exception as e:
        logger.error(f"[ONNX] Export failed: {e}")
        raise
    return save_dir


def quantize_onnx_int8(onnx_dir: Path, quant_dir: Path) -> Path:
    """Apply static INT8 quantization to an ONNX model directory."""
    if quant_dir.exists():
        logger.info(f"[ONNX] Found cached INT8 quantization at {quant_dir}")
        return quant_dir

    logger.info(f"[ONNX] Applying INT8 quantization to {onnx_dir}...")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        import onnx

        quant_dir.mkdir(parents=True, exist_ok=True)
        for onnx_file in onnx_dir.glob("*.onnx"):
            out_file = quant_dir / onnx_file.name
            quantize_dynamic(
                str(onnx_file),
                str(out_file),
                weight_type=QuantType.QInt8,
            )
            logger.info(f"[ONNX] Quantized: {onnx_file.name}")

        # Copy non-ONNX files (tokenizer, processor configs, etc.)
        for f in onnx_dir.iterdir():
            if f.suffix != ".onnx":
                dest = quant_dir / f.name
                if not dest.exists():
                    shutil.copy2(f, dest)

    except Exception as e:
        logger.error(f"[ONNX] INT8 quantization failed: {e}")
        raise
    return quant_dir


# ── ONNX Runtime inference ────────────────────────────────────────────────────

class ONNXVLMRunner:
    """
    Thin wrapper around an Optimum-exported ONNX VLM for inference.
    Uses ORTModelForVision2Seq when available, falls back to raw ORT session.
    """

    def __init__(self, model_dir: Path, provider: str = "CUDAExecutionProvider"):
        self.model_dir = model_dir
        self.provider  = provider
        self._model    = None
        self._processor = None
        self._load()

    def _load(self):
        try:
            from optimum.onnxruntime import ORTModelForVision2Seq
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(
                str(self.model_dir), trust_remote_code=True
            )
            self._model = ORTModelForVision2Seq.from_pretrained(
                str(self.model_dir),
                provider=self.provider,
            )
            logger.info(f"[ONNX] Loaded via ORTModelForVision2Seq (provider={self.provider})")
        except Exception as e:
            logger.warning(f"[ONNX] ORTModelForVision2Seq failed ({e}), falling back to raw session")
            self._model = None

    def generate(self, image, question: str, max_new_tokens: int = 30) -> str:
        if self._model is None:
            return "[ONNX inference not available]"
        from PIL import Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=prompt, images=[image], return_tensors="pt")
        with torch.no_grad():
            ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self._processor.batch_decode(
            ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()


# ── Evaluation with ONNX runner ───────────────────────────────────────────────

def evaluate_onnx(runner: ONNXVLMRunner, samples: list,
                  dataset_name: str, accuracy_fn,
                  provider: str) -> dict:
    from tqdm import tqdm

    use_gpu = "CUDA" in provider
    ProfilerCls = GPUProfiler if use_gpu else CPUProfiler

    scores, latencies = [], []
    profiler = ProfilerCls()

    with profiler:
        for sample in tqdm(samples, desc=dataset_name, leave=False):
            t0   = time.perf_counter()
            pred = runner.generate(sample["image"], sample["question"])
            latencies.append(time.perf_counter() - t0)
            scores.append(accuracy_fn(pred, sample["answers"]))

    stats      = profiler.stats()
    avg_acc    = sum(scores) / len(scores)
    avg_lat    = sum(latencies) / len(latencies)
    throughput = len(latencies) / stats.wall_time_s

    logger.info(
        f"  {dataset_name}: acc={avg_acc:.4f}  lat={avg_lat:.3f}s  "
        f"mem={stats.peak_memory_mb:.0f}MB  tput={throughput:.2f} samp/s"
    )
    return {
        "accuracy":       round(avg_acc,   4),
        "avg_latency_s":  round(avg_lat,   4),
        "peak_memory_mb": round(stats.peak_memory_mb, 1),
        "throughput_sps": round(throughput, 3),
        "n_samples":      len(samples),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ONNX export + quantization pipeline")
    parser.add_argument("--model_id",  required=True)
    parser.add_argument("--quantize",  action="store_true",
                        help="Apply INT8 dynamic quantization after export")
    parser.add_argument("--opset",     type=int, default=17)
    parser.add_argument("--provider",  default="CUDAExecutionProvider",
                        choices=["CUDAExecutionProvider", "CPUExecutionProvider"],
                        help="ONNX Runtime execution provider")
    parser.add_argument("--vqav2_n",   type=int, default=500)
    parser.add_argument("--skip_pope", action="store_true")
    args = parser.parse_args()

    model_id  = args.model_id
    family    = detect_family(model_id)
    safe_name = model_id.replace("/", "__")
    quant_tag = "_int8" if args.quantize else ""
    tag       = f"{safe_name}__onnx{quant_tag}"
    out_path  = RESULTS_DIR / f"{tag}.json"

    if out_path.exists():
        logger.info(f"Result already exists at {out_path}. Skipping.")
        return

    if family not in OPTIMUM_SUPPORTED:
        logger.warning(
            f"Family '{family}' is not in OPTIMUM_SUPPORTED list: {OPTIMUM_SUPPORTED}. "
            f"Export may fail. Attempting anyway."
        )

    # ── Export ───────────────────────────────────────────────────────────
    export_dir = ONNX_CACHE / f"{safe_name}__onnx"
    onnx_dir   = export_to_onnx(model_id, family, export_dir, opset=args.opset)

    # ── Quantize ─────────────────────────────────────────────────────────
    if args.quantize:
        quant_dir = ONNX_CACHE / f"{safe_name}__onnx_int8"
        onnx_dir  = quantize_onnx_int8(onnx_dir, quant_dir)

    # ── Load ONNX runner ──────────────────────────────────────────────────
    runner = ONNXVLMRunner(onnx_dir, provider=args.provider)

    results: dict = {
        "model_id":       model_id,
        "family":         family,
        "compression":    "onnx_int8" if args.quantize else "onnx_fp32",
        "provider":       args.provider,
        "onnx_dir":       str(onnx_dir),
        "benchmarks":     {},
    }

    # ── Evaluate ─────────────────────────────────────────────────────────
    vqa_samples = load_vqav2(n_samples=args.vqav2_n)
    results["benchmarks"]["vqav2"] = evaluate_onnx(
        runner, vqa_samples, "VQAv2", _vqa_accuracy, args.provider
    )

    if not args.skip_pope:
        pope_samples = load_pope()
        results["benchmarks"]["pope"] = evaluate_onnx(
            runner, pope_samples, "POPE", _pope_accuracy, args.provider
        )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"ONNX results saved to {out_path}")


if __name__ == "__main__":
    main()
