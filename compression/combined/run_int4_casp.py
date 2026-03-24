"""
compression/combined/run_int4_casp.py
======================================
Combined Category 1 + Category 2: INT4 quantization + CASP compression.

Pipeline (all on CPU where SVD works):
  1. Load model on CPU in FP16
  2. Apply CASP: QK low-rank factorization + mixed-precision simulated quantization
     - SVD works on CPU (avoids Jetson cuSOLVER crash)
     - QK low-rank reduces weight memory for Q/K projections
  3. Quantize remaining nn.Linear layers to INT4
     - PALULinear (from CASP QK low-rank) stays as-is (already compressed)
  4. Move to GPU
  5. Run inference with memory-safe settings

Usage:
  python compression/combined/run_int4_casp.py \
      --model_id Qwen/Qwen2.5-VL-7B-Instruct --vqav2_n 10
"""

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.model_loader import detect_family, _gpu_mem_mb
from evaluation.run_baseline import (
    load_vqav2, evaluate_dataset, _vqa_accuracy, run_inference,
)
from compression.ptq.run_pytorch_int4 import Int4Linear
from compression.palu.run_palu import PALULinear

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "int4_casp"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Vision module detection ──────────────────────────────────────────────────

VISION_KEYWORDS = [
    "vision_model", "vision_tower", "visual", "vit", "image_encoder",
    "img_encoder", "vision_encoder", "patch_embed", "pixel_shuffle",
    "connector", "multi_modal_projector", "mlp1",
]

QK_PROJ_PATTERNS = {"q_proj", "k_proj", "query_proj", "key_proj"}


def _is_vision(name: str) -> bool:
    return any(kw in name.lower() for kw in VISION_KEYWORDS)


def _is_qk_proj(name: str) -> bool:
    return any(p in name.lower().split(".")[-1] for p in QK_PROJ_PATTERNS)


# ── Step 2: CASP QK low-rank on CPU ─────────────────────────────────────────

def apply_qk_lowrank_cpu(model: nn.Module, rank_fraction: float = 0.25):
    """Apply low-rank SVD to Q/K projections on CPU (where SVD works).

    This reduces the parameter count of Q/K projections and produces
    smaller intermediate computations during attention.

    Returns: dict with compression stats.
    """
    module_dict = dict(model.named_modules())
    replacements = []
    n_lowrank = 0
    params_saved = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if _is_vision(name):
            continue
        if not _is_qk_proj(name):
            continue

        W = module.weight.data.float().cpu()
        out_features, in_features = W.shape
        full_rank = min(out_features, in_features)

        if full_rank < 16:
            continue

        rank = max(int(full_rank * rank_fraction), 8)

        try:
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            A = (U[:, :rank] * S[:rank].unsqueeze(0)).to(torch.float16)  # (out, rank)
            B = Vt[:rank, :].to(torch.float16)                           # (rank, in)

            bias = module.bias.data.clone().to(torch.float16) if module.bias is not None else None
            new_module = PALULinear(A, B, bias)

            # Find parent and replace
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = module_dict.get(parts[0])
                attr_name = parts[1]
            else:
                parent = model
                attr_name = name

            if parent is not None:
                replacements.append((parent, attr_name, new_module))
                orig_params = out_features * in_features
                new_params = out_features * rank + rank * in_features
                params_saved += orig_params - new_params
                n_lowrank += 1

        except RuntimeError as e:
            logger.warning(f"  SVD failed for {name}: {e}")
            continue

    # Apply replacements
    for parent, attr_name, new_module in replacements:
        setattr(parent, attr_name, new_module)

    # Free memory
    del replacements
    gc.collect()

    logger.info(f"  QK low-rank: {n_lowrank} layers factorized, "
                f"{params_saved / 1e6:.1f}M params saved")
    return {"qk_lowrank_layers": n_lowrank, "params_saved_M": round(params_saved / 1e6, 1)}


# ── Step 3: INT4 quantization (skip PALULinear and vision) ───────────────────

def quantize_to_int4(model: nn.Module, skip_vision: bool = True,
                     group_size: int = 128):
    """Replace nn.Linear with Int4Linear. Skips PALULinear (already compressed)."""
    n_replaced = 0
    n_skipped_vision = 0
    n_skipped_palu = 0

    for mod_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                if isinstance(child, PALULinear):
                    n_skipped_palu += 1
                continue

            full_name = f"{mod_name}.{child_name}" if mod_name else child_name

            if skip_vision and _is_vision(full_name):
                n_skipped_vision += 1
                continue

            int4_layer = Int4Linear(
                child.in_features, child.out_features,
                bias=child.bias is not None,
                group_size=group_size,
            )
            int4_layer.set_weight_from_fp16(child.weight.data)
            if child.bias is not None:
                int4_layer.bias = child.bias.data.to(torch.float16)
            setattr(parent, child_name, int4_layer)
            n_replaced += 1
            del child

            if n_replaced % 50 == 0:
                gc.collect()
                logger.info(f"    ... quantized {n_replaced} layers")

    logger.info(f"  INT4: {n_replaced} layers quantized, "
                f"{n_skipped_vision} vision skipped, "
                f"{n_skipped_palu} PALULinear skipped")
    return {
        "int4_layers": n_replaced,
        "vision_skipped": n_skipped_vision,
        "palu_skipped": n_skipped_palu,
    }


# ── Combined loader ─────────────────────────────────────────────────────────

def _find_vision_submodule(model):
    """Find the vision encoder submodule name for split-device placement."""
    for name, child in model.named_children():
        child_name_lower = name.lower()
        if any(kw in child_name_lower for kw in ["visual", "vision", "vit", "image_encoder"]):
            return name
        # Check one level deeper (e.g., model.model.visual)
        for subname, subchild in child.named_children():
            sub_lower = subname.lower()
            if any(kw in sub_lower for kw in ["visual", "vision", "vit", "image_encoder"]):
                return f"{name}.{subname}"
    return None


def _add_device_transfer_hook(model, vision_path: str):
    """Add a forward hook to the vision encoder that moves its output to GPU.

    This allows the vision encoder to run on CPU while the LLM runs on GPU.
    The hook transparently moves vision outputs to GPU before they reach the LLM.
    """
    # Navigate to the vision module
    parts = vision_path.split(".")
    module = model
    for p in parts:
        module = getattr(module, p)

    def move_output_to_cuda(mod, inp, output):
        if isinstance(output, torch.Tensor):
            return output.to("cuda")
        elif isinstance(output, (tuple, list)):
            return type(output)(
                t.to("cuda") if isinstance(t, torch.Tensor) else t
                for t in output
            )
        return output

    module.register_forward_hook(move_output_to_cuda)
    logger.info(f"  Added device-transfer hook on {vision_path}")


def load_int4_casp(model_id: str, family: str = None,
                   skip_vision: bool = True, group_size: int = 128,
                   qk_rank_fraction: float = 0.25,
                   split_device: bool = False):
    """Load model with CASP + INT4 combined compression.

    1. Load on CPU (FP16)
    2. Apply QK low-rank (SVD on CPU)
    3. Quantize to INT4
    4. Move to GPU (or split: vision on CPU, LLM on GPU)

    If split_device=True, keeps the vision encoder on CPU to save ~1.3 GB GPU memory.
    A forward hook transparently moves vision outputs to GPU.
    """
    from models.model_loader import load_model as _load_model_standard

    if family is None:
        family = detect_family(model_id)

    mem_before = _gpu_mem_mb()
    t0 = time.time()

    # Step 1: Load on CPU
    logger.info(f"Step 1: Loading {model_id} on CPU...")
    model, processor, load_meta = _load_model_standard(model_id, device_map="cpu")
    logger.info(f"  CPU load done. GPU mem: {_gpu_mem_mb():.0f} MB")

    # Step 2: CASP QK low-rank on CPU
    logger.info(f"Step 2: Applying QK low-rank (rank={qk_rank_fraction:.0%})...")
    casp_stats = apply_qk_lowrank_cpu(model, rank_fraction=qk_rank_fraction)
    gc.collect()

    # Step 3: INT4 quantization on CPU (always skip vision — it stays FP16)
    logger.info("Step 3: Quantizing to INT4...")
    int4_stats = quantize_to_int4(model, skip_vision=True, group_size=group_size)
    gc.collect()

    # Step 4: Move to GPU
    vision_path = None
    if split_device:
        vision_path = _find_vision_submodule(model)
        if vision_path:
            logger.info(f"Step 4: Split-device mode — vision ({vision_path}) stays CPU, LLM → GPU")
            # Move everything to GPU first, then move vision back to CPU
            model = model.to("cuda").eval()
            gc.collect()
            torch.cuda.empty_cache()

            # Move vision encoder back to CPU
            parts = vision_path.split(".")
            vision_module = model
            for p in parts:
                vision_module = getattr(vision_module, p)
            vision_module.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

            # Add hook to move vision outputs to GPU
            _add_device_transfer_hook(model, vision_path)
            logger.info(f"  Vision encoder moved to CPU, hook installed")
        else:
            logger.warning("  Could not find vision submodule, loading all to GPU")
            model = model.to("cuda").eval()
    else:
        logger.info("Step 4: Moving entire model to GPU...")
        model = model.to("cuda").eval()

    gc.collect()
    torch.cuda.empty_cache()

    load_time = time.time() - t0
    mem_after = _gpu_mem_mb()

    logger.info(f"  GPU memory: {mem_after:.0f} MB (delta: {mem_after - mem_before:.0f} MB)")
    logger.info(f"  Total load time: {load_time:.1f}s")

    # Count params
    n_params = sum(p.numel() for p in model.parameters())
    for name, buf in model.named_buffers():
        if "weight_packed" in name:
            n_params += buf.numel() * 2

    meta = {
        "model_id": model_id,
        "family": family,
        "method": "int4_casp",
        "quant_bits": 4,
        "group_size": group_size,
        "qk_rank_fraction": qk_rank_fraction,
        "skip_vision_quant": skip_vision,
        "split_device": split_device,
        "vision_on_cpu": vision_path if split_device else None,
        **casp_stats,
        **int4_stats,
        "num_params_M": round(n_params / 1e6, 1),
        "gpu_mem_before_mb": round(mem_before, 1),
        "gpu_mem_after_mb": round(mem_after, 1),
        "gpu_mem_load_mb": round(mem_after - mem_before, 1),
        "load_time_s": round(load_time, 1),
    }

    return model, processor, meta


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Combined INT4 + CASP compression for maximum memory reduction"
    )
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--vqav2_n", type=int, default=10)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--quantize_vision", action="store_true")
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--qk_rank", type=float, default=0.25,
                        help="QK low-rank fraction (default: 0.25 = keep 25%% of rank)")
    parser.add_argument("--split_device", action="store_true",
                        help="Keep vision on CPU, LLM on GPU (saves ~1.3GB GPU)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_id = args.model_id
    family = detect_family(model_id)
    safe_name = model_id.replace("/", "__")

    out_path = RESULTS_DIR / f"{safe_name}__int4_casp.json"
    if out_path.exists() and not args.force:
        logger.info(f"Result exists: {out_path}")
        return

    # Load with combined compression
    model, processor, meta = load_int4_casp(
        model_id, family,
        skip_vision=not args.quantize_vision,
        group_size=args.group_size,
        qk_rank_fraction=args.qk_rank,
        split_device=args.split_device,
    )

    if not args.skip_eval:
        samples = load_vqav2(args.vqav2_n)
        vqa_result = evaluate_dataset(
            model, processor, samples, family, "cuda",
            "vqav2", _vqa_accuracy,
        )
        meta["benchmarks"] = {"vqav2": vqa_result}
    else:
        meta["benchmarks"] = {}

    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
