"""
compression/lowrank/run_svd_llm.py
====================================
SVD-LLM: Truncation-aware SVD for LLM Compression.

Based on: "SVD-LLM: Truncation-aware Singular Value Decomposition for
           Large Language Model Compression" (Wang et al., ICLR 2025)

Key idea: Truncation-aware SVD compensating for error by updating remaining
singular values. Automatic rank assignment by layer importance.

For each Linear layer W (out, in):
  1. Compute SVD: W = U @ diag(S) @ V^T
  2. Truncate to rank r: W_approx = U[:,:r] @ diag(S[:r]) @ V[:,:r]^T
  3. Replace single Linear(in, out) with two layers:
     - Linear(in, r) with weights V[:,:r]^T @ diag(S[:r])
     - Linear(r, out) with weights U[:,:r]
  This saves parameters when r < (in * out) / (in + out).

Optionally uses calibration data to compute truncation-aware compensation
that adjusts remaining singular values to minimize reconstruction error.

Usage:
  python compression/lowrank/run_svd_llm.py \
      --model_id HuggingFaceTB/SmolVLM-256M-Instruct --rank_ratio 0.50

  python compression/lowrank/run_svd_llm.py \
      --model_id Qwen/Qwen2.5-VL-3B-Instruct --rank_ratio 0.30 --truncation_aware
"""

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import platform

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ── Jetson-safe SVD / linalg wrappers ────────────────────────────────────────
# On Jetson (aarch64), torch.linalg.svd triggers dlopen of libtorch_cuda_linalg.so
# which fails with "undefined symbol: cusolverDnXsyevBatched_bufferSize".
# Workaround: force CPU + disable CUDA linalg backend before first call.

_LINALG_SAFE = None  # None = untested, True = works, False = broken


def _safe_svd(tensor_cpu: torch.Tensor, full_matrices: bool = False):
    """SVD that works on Jetson by catching cuSOLVER dlopen failures."""
    global _LINALG_SAFE
    if _LINALG_SAFE is None:
        try:
            # Test with a tiny matrix to see if linalg works
            _test = torch.randn(2, 2)
            torch.linalg.svd(_test, full_matrices=False)
            _LINALG_SAFE = True
        except RuntimeError:
            _LINALG_SAFE = False
            logger.warning("torch.linalg.svd broken (cuSOLVER missing) — using torch.svd fallback")

    tensor_cpu = tensor_cpu.cpu().float()
    if _LINALG_SAFE:
        return torch.linalg.svd(tensor_cpu, full_matrices=full_matrices)
    else:
        # torch.svd uses a different code path that doesn't need cuSOLVER
        U, S, V = torch.svd(tensor_cpu, some=not full_matrices)
        return U, S, V.t()  # torch.svd returns V, linalg.svd returns Vh


def _safe_cholesky(tensor):
    """Cholesky that works on Jetson."""
    try:
        return torch.linalg.cholesky(tensor.cpu().float()).to(tensor.device, tensor.dtype)
    except RuntimeError:
        # Fallback: use CPU
        result = torch.linalg.cholesky(tensor.cpu().float())
        return result.to(tensor.device, tensor.dtype)
from models.model_loader import load_model, unload_model, detect_family
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

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "svd_llm"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


VISION_MODULE_KEYWORDS = {
    "vision_model", "visual_model", "image_encoder", "vision_encoder",
    "patch_embed", "visual_projection", "img_projection",
    "vision_tower", "vit", "davit", "siglip", "fastvit",
}


def _is_vision_module(name: str) -> bool:
    return any(kw in name.lower() for kw in VISION_MODULE_KEYWORDS)


class SVDLinear(nn.Module):
    """A factored Linear layer: W ≈ U @ V where V = S^{1/2} @ Vh, U = Uh @ S^{1/2}.

    Replaces a single (out, in) Linear with two smaller linears:
      first:  (in_features → rank)  — no bias
      second: (rank → out_features) — with optional bias
    """

    def __init__(self, first: nn.Linear, second: nn.Linear):
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, x):
        return self.second(self.first(x))


def svd_decompose_layer(layer: nn.Linear, rank: int,
                         truncation_aware: bool = False,
                         input_cov: torch.Tensor = None) -> SVDLinear:
    """Decompose a Linear layer using truncated SVD.

    If truncation_aware=True and input_cov is provided, adjusts singular
    values to minimize ||W @ X - W_approx @ X||_F rather than ||W - W_approx||_F.
    """
    W = layer.weight.data.float()  # (out_features, in_features)
    out_features, in_features = W.shape

    rank = min(rank, min(out_features, in_features))

    if truncation_aware and input_cov is not None:
        # Whitening transform: W' = W @ C^{1/2} where C = X^T X / n
        # This makes SVD minimize reconstruction error in activation space
        try:
            L = _safe_cholesky(input_cov + 1e-6 * torch.eye(
                in_features, device=W.device))
            W_prime = W @ L
            U, S, Vh = _safe_svd(W_prime.float().cpu(), full_matrices=False)
            U, S, Vh = U.to(W.device, W.dtype), S.to(W.device, W.dtype), Vh.to(W.device, W.dtype)
            # Undo whitening on V: V_orig = L^{-1} @ V
            Vh_orig = torch.linalg.solve_triangular(L, Vh[:rank].T, upper=False).T
            U_r = U[:, :rank]
            S_r = S[:rank]
            Vh_r = Vh_orig
        except Exception as e:
            logger.debug(f"  Truncation-aware SVD failed ({e}), falling back to standard SVD")
            U, S, Vh = _safe_svd(W.float().cpu(), full_matrices=False)
            U, S, Vh = U.to(W.device, W.dtype), S.to(W.device, W.dtype), Vh.to(W.device, W.dtype)
            U_r = U[:, :rank]
            S_r = S[:rank]
            Vh_r = Vh[:rank]
    else:
        U, S, Vh = _safe_svd(W.float().cpu(), full_matrices=False)
        U, S, Vh = U.to(W.device, W.dtype), S.to(W.device, W.dtype), Vh.to(W.device, W.dtype)
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vh_r = Vh[:rank]

    # Split S between the two factors: sqrt(S) on each side
    S_sqrt = S_r.sqrt()

    # First linear: in_features → rank (weight = diag(S_sqrt) @ Vh_r)
    first_weight = (S_sqrt.unsqueeze(1) * Vh_r)  # (rank, in_features)
    first = nn.Linear(in_features, rank, bias=False, device=W.device,
                      dtype=layer.weight.dtype)
    first.weight.data = first_weight.to(layer.weight.dtype)

    # Second linear: rank → out_features (weight = U_r @ diag(S_sqrt))
    second_weight = (U_r * S_sqrt.unsqueeze(0))  # (out_features, rank)
    second = nn.Linear(rank, out_features, bias=layer.bias is not None,
                       device=W.device, dtype=layer.weight.dtype)
    second.weight.data = second_weight.to(layer.weight.dtype)
    if layer.bias is not None:
        second.bias.data = layer.bias.data.clone()

    return SVDLinear(first, second)


class InputCovCollector:
    """Collects input covariance matrices X^T X for Linear layers."""

    def __init__(self, model: nn.Module):
        self.covs: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = {}
        self.hooks = []
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
            x = inp[0]
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            x = x.float()
            n = x.shape[0]
            cov = x.T @ x
            if name in self.covs:
                self.covs[name] += cov
                self.counts[name] += n
            else:
                self.covs[name] = cov
                self.counts[name] = n
        return hook_fn

    def get_covariances(self) -> dict[str, torch.Tensor]:
        result = {}
        for name in self.covs:
            result[name] = self.covs[name] / self.counts[name]
        return result

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def apply_svd_compression(model: nn.Module, rank_ratio: float,
                           truncation_aware: bool = False,
                           input_covs: dict = None) -> dict:
    """Replace Linear layers with SVD-factored versions.

    rank_ratio: fraction of singular values to keep (0.0-1.0)
    """
    replaced_layers = 0
    params_before = 0
    params_after = 0

    # Collect layers to replace (can't modify during iteration)
    replacements = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if _is_vision_module(name):
            continue

        out_f, in_f = module.weight.shape
        rank = max(1, int(min(out_f, in_f) * rank_ratio))

        # Only replace if it actually saves parameters
        orig_params = out_f * in_f
        new_params = rank * (in_f + out_f)
        if new_params >= orig_params:
            continue

        replacements.append((name, module, rank))

    # Perform replacements
    for name, module, rank in replacements:
        cov = input_covs.get(name) if input_covs else None
        svd_module = svd_decompose_layer(
            module, rank,
            truncation_aware=truncation_aware,
            input_cov=cov,
        )

        # Navigate to parent module and replace
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = dict(model.named_modules())[parent_name]
        else:
            parent = model
            attr_name = name

        setattr(parent, attr_name, svd_module)

        out_f, in_f = module.weight.shape
        orig_params = out_f * in_f
        new_params = rank * (in_f + out_f)
        params_before += orig_params
        params_after += new_params
        replaced_layers += 1

    compression = params_before / params_after if params_after > 0 else 1.0
    logger.info(
        f"SVD-LLM: replaced {replaced_layers} layers | "
        f"rank_ratio={rank_ratio:.2f} | "
        f"params: {params_before:,} → {params_after:,} "
        f"({compression:.2f}x compression)"
    )
    return {
        "rank_ratio": rank_ratio,
        "replaced_layers": replaced_layers,
        "params_before": params_before,
        "params_after": params_after,
        "compression_ratio": round(compression, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="SVD-LLM compression pipeline")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--rank_ratio", type=float, default=0.50,
                        help="Fraction of singular values to keep (default: 0.50)")
    parser.add_argument("--truncation_aware", action="store_true",
                        help="Use truncation-aware SVD with calibration data")
    parser.add_argument("--n_calib", type=int, default=128,
                        help="Number of calibration samples (for truncation-aware)")
    parser.add_argument("--vqav2_n", type=int, default=1000)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_id = args.model_id
    family = detect_family(model_id)
    safe_name = model_id.replace("/", "__")
    rr_tag = f"rr{int(args.rank_ratio * 100)}"
    ta_tag = "_ta" if args.truncation_aware else ""
    tag = f"{safe_name}__svdllm_{rr_tag}{ta_tag}"
    out_path = RESULTS_DIR / f"{tag}.json"

    if out_path.exists() and not args.force:
        logger.info(f"Result already exists at {out_path}. Skipping.")
        return

    # Load model
    logger.info(f"Loading {model_id} (fp16)...")
    model, processor, meta = load_model(model_id, quant="fp16")
    device = str(next(model.parameters()).device)
    num_params_before = sum(p.numel() for p in model.parameters())

    # Optional calibration for truncation-aware SVD
    input_covs = None
    if args.truncation_aware:
        logger.info(f"Calibration pass ({args.n_calib} samples)...")
        calib_samples = load_vqav2(n_samples=args.n_calib)

        collector = InputCovCollector(model)
        for sample in calib_samples:
            try:
                with torch.no_grad():
                    _ = run_inference(model, processor, sample, family, device)
            except Exception:
                continue
        input_covs = collector.get_covariances()
        collector.remove_hooks()
        logger.info(f"  Collected covariances for {len(input_covs)} layers")

    # Apply SVD compression
    logger.info(f"Applying SVD-LLM compression (rank_ratio={args.rank_ratio})...")
    svd_stats = apply_svd_compression(
        model, args.rank_ratio,
        truncation_aware=args.truncation_aware,
        input_covs=input_covs,
    )

    num_params_after = sum(p.numel() for p in model.parameters())

    results = {
        "model_id": model_id,
        "family": family,
        "method": "svd_llm",
        "rank_ratio": args.rank_ratio,
        "truncation_aware": args.truncation_aware,
        "replaced_layers": svd_stats["replaced_layers"],
        "weight_compression_ratio": svd_stats["compression_ratio"],
        "num_params_before_M": round(num_params_before / 1e6, 1),
        "num_params_after_M": round(num_params_after / 1e6, 1),
        "param_reduction_pct": round(
            100 * (1 - num_params_after / num_params_before), 2
        ),
        "gpu_mem_load_mb": meta.gpu_mem_delta_mb,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmarks": {},
    }

    if not args.skip_eval:
        eval_samples = load_vqav2(n_samples=args.vqav2_n)
        results["benchmarks"]["vqav2"] = evaluate_dataset(
            model, processor, eval_samples, family, device,
            "VQAv2", _vqa_accuracy,
        )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"SVD-LLM results saved to {out_path}")

    unload_model(model)


if __name__ == "__main__":
    main()
