"""
compression/awq_gptq/run_awq_gptq.py
======================================
Simulated AWQ and GPTQ INT4 quantization for VLMs.

AutoAWQ/AutoGPTQ only support text-only CausalLM architectures.
For VLMs, we implement the core algorithms directly:

AWQ: Activation-Aware Weight Quantization (Lin et al., MLSys 2024)
  - Collect activation magnitudes via calibration pass
  - Scale weight channels by activation importance before quantizing
  - Per-channel INT4 quantize-then-dequantize (simulated)

GPTQ: Accurate Post-Training Quantization (Frantar et al., ICLR 2023)
  - Collect Hessian (X^T X) per layer via calibration
  - Quantize weights column-by-column with Hessian-based error compensation
  - Simulated: quantize + dequantize in fp16

Both methods target Linear layers in the LLM backbone (skip vision encoder).

Usage:
  python compression/awq_gptq/run_awq_gptq.py \
      --model_id HuggingFaceTB/SmolVLM-256M-Instruct --method awq
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.model_loader import load_model, unload_model
from evaluation.run_baseline import (
    load_vqav2, run_inference,
    evaluate_dataset, _vqa_accuracy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "awq_gptq"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Vision module detection ──────────────────────────────────────────────────

VISION_MODULE_KEYWORDS = {
    "vision_model", "visual_model", "image_encoder", "vision_encoder",
    "patch_embed", "visual_projection", "img_projection",
    "vision_tower", "vit", "davit", "siglip", "fastvit",
}


def _is_vision_module(name: str) -> bool:
    return any(kw in name.lower() for kw in VISION_MODULE_KEYWORDS)


# ── Activation collection ────────────────────────────────────────────────────

class ActivationCollector:
    """Collects per-channel activation statistics for AWQ and Hessians for GPTQ."""

    def __init__(self, model: nn.Module, collect_hessian: bool = False):
        self.act_norms: dict[str, torch.Tensor] = {}
        self.hessians: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = {}
        self.hooks = []
        self.collect_hessian = collect_hessian
        self._register(model)

    def _register(self, model: nn.Module):
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if _is_vision_module(name):
                continue
            hook = module.register_forward_hook(self._make_hook(name))
            self.hooks.append(hook)

    def _make_hook(self, name: str):
        def hook_fn(module, inp, out):
            with torch.no_grad():
                x = inp[0]
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                # Flatten to (tokens, features)
                x_flat = x.reshape(-1, x.shape[-1]).float()
                n_tokens = x_flat.shape[0]

                # Per-channel L2 norm for AWQ
                sq_sum = x_flat.pow(2).sum(dim=0).cpu()
                if name in self.act_norms:
                    self.act_norms[name] += sq_sum
                    self.counts[name] += n_tokens
                else:
                    self.act_norms[name] = sq_sum
                    self.counts[name] = n_tokens

                # Hessian (X^T X) for GPTQ — on CPU to save GPU memory
                if self.collect_hessian:
                    xtx = (x_flat.t() @ x_flat).cpu()
                    if name in self.hessians:
                        self.hessians[name] += xtx
                    else:
                        self.hessians[name] = xtx

        return hook_fn

    def get_act_norms(self) -> dict[str, torch.Tensor]:
        result = {}
        for name in self.act_norms:
            result[name] = (self.act_norms[name] / self.counts[name]).sqrt()
        return result

    def get_hessians(self) -> dict[str, torch.Tensor]:
        result = {}
        for name in self.hessians:
            result[name] = self.hessians[name] / self.counts[name]
        return result

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ── AWQ: Activation-Aware INT4 Quantization ──────────────────────────────────

def _quantize_int4_per_channel(W: torch.Tensor) -> torch.Tensor:
    """Per-channel (per-row) symmetric INT4 quantize + dequantize."""
    abs_max = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = abs_max / 7.0  # INT4 signed: [-8, 7]
    W_q = (W / scale).round().clamp(-8, 7)
    return W_q * scale


def apply_awq_quantization(model: nn.Module,
                           act_norms: dict[str, torch.Tensor]) -> dict:
    """
    AWQ: Scale salient channels before INT4 quantization.

    For each Linear layer:
    1. Compute importance = |W| * activation_norm (per input channel)
    2. Find scaling factors s_j that minimize quantization error
       (simplified: s_j = sqrt(act_norm_j / mean(act_norm)))
    3. Scale weights: W[:, j] *= s_j, then quantize, then unscale
    """
    quantized_layers = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if _is_vision_module(name):
            continue
        if name not in act_norms:
            continue

        W = module.weight.data.float()
        x_norm = act_norms[name].to(W.device)

        if x_norm.shape[0] != W.shape[1]:
            continue

        # AWQ scaling: protect important channels
        mean_norm = x_norm.mean().clamp(min=1e-8)
        scale = (x_norm / mean_norm).sqrt().clamp(min=0.1, max=10.0)

        # Scale weights, quantize, unscale
        W_scaled = W * scale.unsqueeze(0)
        W_q = _quantize_int4_per_channel(W_scaled)
        W_final = W_q / scale.unsqueeze(0)

        module.weight.data = W_final.to(module.weight.dtype)
        quantized_layers += 1

    logger.info(f"AWQ: quantized {quantized_layers} layers to simulated INT4")
    return {"quantized_layers": quantized_layers}


# ── GPTQ: Hessian-based INT4 Quantization ────────────────────────────────────

def apply_gptq_quantization(model: nn.Module,
                            hessians: dict[str, torch.Tensor]) -> dict:
    """
    Simplified GPTQ: quantize with Hessian-based error compensation.

    For each Linear layer:
    1. Use H = X^T X (Hessian approximation)
    2. Quantize columns sequentially, compensating remaining columns
       using H to minimize overall output error
    3. Block-wise for efficiency (block_size=128)
    """
    quantized_layers = 0
    block_size = 128

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if _is_vision_module(name):
            continue
        if name not in hessians:
            continue

        W = module.weight.data.float()
        H = hessians[name].to(W.device).float()

        if H.shape[0] != W.shape[1]:
            continue

        n_cols = W.shape[1]

        # Damping for numerical stability
        damp = 0.01 * H.diagonal().mean()
        H.diagonal().add_(damp)

        # Cholesky for efficient inverse
        try:
            L = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(L)
        except RuntimeError:
            # Fallback: just do naive INT4 quantization
            module.weight.data = _quantize_int4_per_channel(W).to(module.weight.dtype)
            quantized_layers += 1
            continue

        # Block-wise GPTQ
        W_q = W.clone()
        for start in range(0, n_cols, block_size):
            end = min(start + block_size, n_cols)
            block = W_q[:, start:end].clone()
            H_inv_block = H_inv[start:end, start:end]

            # Per-channel quantize each column in the block
            abs_max = W_q.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
            scale = abs_max / 7.0

            block_q = (block / scale).round().clamp(-8, 7) * scale
            err = block - block_q
            W_q[:, start:end] = block_q

            # Compensate remaining columns
            if end < n_cols:
                H_comp = H_inv[start:end, end:]
                diag = H_inv_block.diagonal().clamp(min=1e-8)
                scaled_err = err / diag.unsqueeze(0)
                W_q[:, end:] -= scaled_err @ H_comp

        module.weight.data = W_q.to(module.weight.dtype)
        quantized_layers += 1

        # Free Hessian memory
        del H, H_inv

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"GPTQ: quantized {quantized_layers} layers to simulated INT4")
    return {"quantized_layers": quantized_layers}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AWQ/GPTQ quantization pipeline")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--method", required=True, choices=["awq", "gptq"],
                        help="Quantization method: awq or gptq")
    parser.add_argument("--n_calib", type=int, default=128,
                        help="Number of calibration samples")
    parser.add_argument("--vqav2_n", type=int, default=1000)
    parser.add_argument("--skip_vqav2", action="store_true")
    parser.add_argument("--skip_textvqa", action="store_true")
    parser.add_argument("--skip_pope", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_id = args.model_id
    safe_name = model_id.replace("/", "__")
    tag = f"{safe_name}__{args.method}_int4"
    out_path = RESULTS_DIR / f"{tag}.json"

    if out_path.exists() and not args.force:
        logger.info(f"Result exists at {out_path}. Skipping.")
        return

    # ── Load model ────────────────────────────────────────────────────────
    logger.info(f"Loading {model_id} (fp16) for {args.method.upper()}...")
    model, processor, meta = load_model(model_id, quant="fp16")
    family = meta.family
    device = str(next(model.parameters()).device)
    num_params = sum(p.numel() for p in model.parameters())

    # ── Calibration pass ──────────────────────────────────────────────────
    collect_hessian = (args.method == "gptq")
    logger.info(f"Calibration pass ({args.n_calib} samples, hessian={collect_hessian})...")
    calib_samples = load_vqav2(n_samples=args.n_calib)

    collector = ActivationCollector(model, collect_hessian=collect_hessian)
    skipped = 0
    for sample in calib_samples:
        try:
            _ = run_inference(model, processor, sample, family, device)
        except Exception:
            skipped += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    if skipped:
        logger.warning(f"  Calibration: skipped {skipped}/{args.n_calib}")

    act_norms = collector.get_act_norms()
    hessians = collector.get_hessians() if collect_hessian else {}
    collector.remove_hooks()

    # ── Apply quantization ────────────────────────────────────────────────
    if args.method == "awq":
        logger.info("Applying AWQ (activation-aware INT4 quantization)...")
        quant_stats = apply_awq_quantization(model, act_norms)
    else:
        logger.info("Applying GPTQ (Hessian-based INT4 quantization)...")
        quant_stats = apply_gptq_quantization(model, hessians)

    # Free calibration data
    del act_norms, hessians
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Measure memory ────────────────────────────────────────────────────
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(torch.cuda.current_device())
        gpu_mem_after = (total - free) / 1024**2
    else:
        gpu_mem_after = 0.0

    # ── Compression ratio vs baseline ─────────────────────────────────────
    baseline_path = (
        Path(__file__).resolve().parents[2]
        / "results" / "baseline" / f"{safe_name}.json"
    )
    compression_ratio = None
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        baseline_mem = baseline.get("gpu_mem_load_mb", 0)
        if baseline_mem > 0 and meta.gpu_mem_delta_mb > 0:
            compression_ratio = round(baseline_mem / meta.gpu_mem_delta_mb, 2)

    results = {
        "model_id": model_id,
        "family": family,
        "method": args.method,
        "quant": "simulated_int4",
        "bits": 4,
        "group_size": "per_channel",
        "n_calib_samples": args.n_calib,
        **quant_stats,
        "num_params_M": round(num_params / 1e6, 1),
        "gpu_mem_load_mb": meta.gpu_mem_delta_mb,
        "gpu_mem_after_quant_mb": round(gpu_mem_after, 1),
        "compression_ratio": compression_ratio,
        "benchmarks": {},
    }

    # ── Evaluate ──────────────────────────────────────────────────────────
    if not args.skip_vqav2:
        eval_samples = load_vqav2(n_samples=args.vqav2_n)
        results["benchmarks"]["vqav2"] = evaluate_dataset(
            model, processor, eval_samples, family, device,
            "VQAv2", _vqa_accuracy,
        )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    unload_model(model)


if __name__ == "__main__":
    main()
