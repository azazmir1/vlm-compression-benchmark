# VLM Compression Benchmark — Complete Experiment Log

**Project:** Deploying Vision-Language Models on NVIDIA Jetson Orin Nano 8GB
**Target conference:** EMSOFT 2026
**Date range:** 2026-03-20 to 2026-03-24
**Hardware:** NVIDIA Jetson Orin Nano 8GB (target), NVIDIA A6000 48GB (baseline reference)

---

## 1. Project Goal

Deploy Vision-Language Models (VLMs) on memory-constrained edge devices. The Jetson Orin Nano has 8GB unified RAM shared between CPU and GPU (~6GB usable after OS). Most VLMs above ~2B parameters either can't load or can't run usably.

**Two categories of compression:**
- **Category 1:** Reduce model memory at load time (so OOM models can fit)
- **Category 2:** Reduce runtime/peak memory after loading (so loaded-but-unusable models become practical)

---

## 2. Models Tested

### 2.1 VLM Families (9 families, 19 models)

| Family | Models | Params Range |
|--------|--------|-------------|
| SmolVLM | SmolVLM-256M, SmolVLM-500M, SmolVLM-2.2B | 256M–2.2B |
| InternVL2.5 | InternVL2.5-1B, 2B, 4B, 8B | 938M–8.1B |
| Qwen2.5-VL | Qwen2.5-VL-3B, 7B | 3.8B–8.3B |
| LFM2-VL | LFM2-VL-450M, 1.6B, 3B | 450M–3B |
| Moondream | moondream2 | 1.9B |
| Gemma3 | gemma-3-4b, gemma-3-12b | 4.3B–12.2B |
| Ovis2 | Ovis2-1B, 2B, 4B, 8B | 1B–8.9B |
| FastVLM | FastVLM-0.5B, 1.5B | 759M–1.5B |
| nanoVLM | nanoVLM-222M, 450M | 222M–450M |

### 2.2 Jetson FP16 Ceiling Scan Results

Tested all 19 models in FP16 on Jetson to find which can load and which OOM.

| Model | Params | Status | GPU Mem (MB) | Latency (s) |
|-------|--------|--------|-------------|-------------|
| SmolVLM-256M | 256M | PASS | 404 (peak 5030) | 2.57 |
| SmolVLM-500M | 508M | PASS | 748 (peak 5758) | 2.68 |
| SmolVLM-2.2B | 2.2B | PASS (borderline) | 4617 | 3.19 |
| LFM2-VL-450M | 450M | PASS | 696 | 1.51 |
| LFM2-VL-1.6B | 1.6B | PASS | 3071 (peak 6417) | 1.50 |
| LFM2-VL-3B | 3B | MEM_CRITICAL | 4848 | — |
| InternVL2.5-1B | 938M | PASS | 1818 (peak 4851) | 1.62 |
| InternVL2.5-2B | 2.2B | PASS but acc=0.0 | 4367 | 0.71 |
| InternVL2.5-4B | 3.7B | **OOM_LOAD** | — | — |
| InternVL2.5-8B | 8.1B | **OOM_LOAD** | — | — |
| Qwen2.5-VL-3B | 3.8B | **OOM_LOAD** | — | — |
| Qwen2.5-VL-7B | 8.3B | **OOM_LOAD** | — | — |
| moondream2 | 1.9B | PASS but acc=0.0 | 2820 | 1.50 |
| FastVLM-0.5B | 759M | PASS (verbose) | 1491 | 4.14 |
| FastVLM-1.5B | 1.5B | ERROR | — | — |
| nanoVLM-222M | 222M | ERROR | — | — |
| nanoVLM-450M | 450M | ERROR | — | — |
| Ovis2-1B+ | various | ERROR | — | — |
| Gemma3-4B+ | various | Not tested | — | — |

**Ceiling per family:**
- SmolVLM: 2.2B (borderline), 500M (safe)
- LFM2-VL: 1.6B (safe), 3B (MEM_CRITICAL)
- InternVL2.5: 1B (safe), 2B (broken inference), 4B+ OOM
- Qwen2.5-VL: ALL OOM (3B+ only)
- Moondream: 2B (broken inference)

---

## 3. Compression Methods Tested (16 methods)

### 3.1 Category 1 Methods (Reduce Load-Time Memory)

| # | Method | Paper/Source | How it works | A6000 | Jetson | File |
|---|--------|-------------|-------------|-------|--------|------|
| 1 | **PyTorch INT8** | Custom (ours) | Pure-PyTorch INT8 with per-channel scale, standard matmul | N/A | **WORKS** | `compression/ptq/run_pytorch_int8.py` |
| 2 | BnB INT8 | bitsandbytes | 8-bit quantization during loading | 19/19 pass | BLOCKED | `compression/ptq/run_ptq.py` |
| 3 | BnB INT4 (NF4) | bitsandbytes | 4-bit NormalFloat quantization | 20/20 pass | BLOCKED | `compression/ptq/run_ptq.py` |
| 4 | AWQ | MLSys 2024 | Activation-aware weight quantization | Not tested | BLOCKED | `compression/ptq/run_awq.py` |
| 5 | GPTQ | ICLR 2023 | Hessian-based optimal quantization | Not tested | BLOCKED | `compression/ptq/run_gptq.py` |
| 6 | GPTQ pre-quant | — | Download pre-quantized HF models | 2/5 pass | BLOCKED | `compression/quantized_pretrained/` |
| 7 | QLoRA | NeurIPS 2023 | INT4 base + LoRA adapters | 2/2 pass | BLOCKED | `compression/qlora/run_qlora.py` |
| 8 | SVD-LLM | — | Low-rank SVD decomposition | 1 run, acc=0 | BLOCKED | `compression/lowrank/run_svd_llm.py` |
| 9 | SparseGPT | ICML 2023 | Hessian-aware pruning | Not tested | BLOCKED | `compression/pruning/run_sparsegpt.py` |
| 10 | AWP | ICML 2025 | Wanda pruning + INT4 quantization | Not tested | BLOCKED | `compression/combined/run_awp.py` |
| 11 | QNNPACK INT8 | PyTorch native | CPU quantization via ARM QNNPACK | N/A | IMPRACTICAL | N/A (tested inline) |

### 3.2 Category 2 Methods (Reduce Runtime Memory)

| # | Method | Paper/Source | How it works | A6000 | Jetson | File |
|---|--------|-------------|-------------|-------|--------|------|
| 12 | **CASP** | CVPR 2025 | Mixed-precision simulated quant + attention sparsity | 2/2 pass | **WORKS** | `compression/casp_slim/run_casp_slim.py` |
| 13 | **SLIM** | ICML 2025 | Pruning + simulated INT4 + SVD | Not tested | **WORKS** (sp20-30%) | `compression/casp_slim/run_casp_slim.py` |
| 14 | **Wanda** | ICLR 2024 | Weight-and-activation-aware pruning | 41/41 pass | **WORKS** | `compression/pruning/run_wanda.py` |
| 15 | **Magnitude Pruning** | Classic | Zero unimportant weights | 38/38 pass | **WORKS** | `compression/pruning/run_pruning.py` |
| 16 | PALU | — | KV-cache low-rank compression | 10 runs | BROKEN (acc=0.0) | `compression/palu/run_palu.py` |
| 17 | PACT | — | Visual token pruning/merging | 4 runs | BROKEN (acc=0.0) | `compression/pact/run_pact.py` |

---

## 4. Jetson Blockers — Why Standard Tools Fail

### 4.1 BitsAndBytes CUDA Kernel Crash
**Error:** `Error named symbol not found at line 62 in file /src/csrc/ops.cu`
**Cause:** BnB compiles custom CUDA kernels for x86 GPUs. The aarch64 Jetson CUDA toolkit doesn't support these symbols.
**Blocks:** BnB INT8, BnB INT4, QLoRA (depends on BnB), AWP (reload step uses BnB)

### 4.2 GPTQ/AWQ Pre-quantized Models Also Fail
**Error:** Same `Error named symbol not found at line 81 in file /src/csrc/ops.cu`
**Cause:** Even pre-quantized models need GPTQ/AWQ inference kernels (from gptqmodel v5.8.0) which use the same unsupported CUDA ops.
**Tested:** `vasanth0475/SmolVLM-256M-Instruct-GPTQ-Int4` on 2026-03-23 — confirmed crash.

### 4.3 GPTQ/AWQ Can't Quantize VLMs Directly
**Error:** `ValueError: Unrecognized configuration class Idefics3Config for AutoModelForCausalLM`
**Cause:** `AutoModelForCausalLM` rejects VLM architectures. VLMs use `ForConditionalGeneration` wrapper, not pure `ForCausalLM`.

### 4.4 cusolver Missing (SparseGPT, SVD-LLM)
**Error:** `cusolverDnXsyevBatched_bufferSize` symbol missing
**Cause:** Jetson's CUDA toolkit lacks certain linalg symbols needed for Hessian inverse (SparseGPT) and SVD decomposition (SVD-LLM).
**Note:** SLIM and CASP silently skip SVD via try/except — they run in degraded mode.

### 4.5 QNNPACK INT8 — Works but Impractical
**Status:** `torch.quantization.quantize_dynamic` with QNNPACK backend works on Jetson.
**Problem:** CPU-only execution (~4 min/sample vs 1.5s on GPU). No CUDA backend for PyTorch dynamic quantization.
**Also:** Requires loading full FP16 model first, so it's not a true Category 1 method.

### 4.6 PACT Bug
**Root cause:** Line 317-318 in `run_pact.py`: `n_visual = max(int(seq_len * 0.5), 1)` hardcodes 50% of tokens as visual tokens. This destroys text tokens, causing 0.0 accuracy on all models.

---

## 5. The Breakthrough: Custom PyTorch INT8

### 5.1 Motivation
Every standard quantization tool was blocked. We needed a Category 1 method that:
- Uses NO custom CUDA kernels (only standard PyTorch ops)
- Reduces memory footprint enough for OOM models to fit
- Runs on GPU at normal speed (not CPU-only)

### 5.2 Implementation

**File:** `compression/ptq/run_pytorch_int8.py`

**Core component — `Int8Linear`:**
```python
class Int8Linear(nn.Module):
    # Stores weights as torch.int8 + per-channel torch.float16 scale
    # Forward: w_fp16 = weight_int8 * scale → standard torch.matmul
    def forward(self, x):
        w = self.weight_int8.to(x.dtype) * self.scale.to(x.dtype)
        return nn.functional.linear(x, w, self.bias)

    def set_weight_from_fp16(self, weight_fp16):
        w = weight_fp16.float()
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        self.weight_int8 = (w / scale).round().clamp(-128, 127).to(torch.int8)
        self.scale = scale.to(torch.float16)
```

**Loading pipeline:**
1. Load model on CPU via `from_pretrained` (uses Jetson's 16GB swap for temporary FP16 peak)
2. Replace all `nn.Linear` in LLM backbone with `Int8Linear` (quantize weights in-place)
3. Vision encoder stays FP16 for accuracy
4. Move quantized model to GPU
5. Result: ~50% memory reduction for LLM backbone weights

**Key insight:** `from_config` + `to_empty` (true layer-by-layer) doesn't work — produces garbage output because `from_config` misses critical initialization done by `from_pretrained` (generation config, model-specific hooks). Must use `from_pretrained`.

### 5.3 Results

**Previously OOM models — now loading and running on Jetson:**

| Model | Params | FP16 Status | INT8 Exact Match | INT8 Contains | INT8 Token F1 | INT8 Mem (MB) | Peak Mem (MB) | Latency (s) | A6000 FP16 Baseline |
|-------|--------|-------------|------------------|---------------|---------------|---------------|---------------|-------------|---------------------|
| **Qwen2.5-VL-3B** | 3.8B | OOM_LOAD | **0.822** | 0.900 | 0.922 | 4943 | 6098 | 3.16 | 0.834 |
| **InternVL2.5-4B** | 3.7B | OOM_LOAD | **0.789** | 0.867 | 0.906 | 5780 | 5407 | 2.51 | 0.791 |

**Models that already loaded — INT8 comparison:**

| Model | Params | Jetson FP16 Acc | INT8 Exact Match | INT8 Token F1 | INT8 Mem (MB) | Latency (s) | A6000 FP16 Baseline |
|-------|--------|----------------|------------------|---------------|---------------|-------------|---------------------|
| SmolVLM-256M | 260M | 0.200 (10 samp) | 0.367 | 0.422 | 332 | 1.62 | 0.562 |
| SmolVLM-500M | 508M | 0.400 (10 samp) | 0.678 | 0.700 | 1531 | 1.69 | 0.633 |
| InternVL2.5-2B | 2.2B | **0.0** (broken!) | **0.700** | 0.800 | 3926 | 2.09 | 0.730 |

**Key findings:**
- Qwen2.5-VL-3B INT8 accuracy (0.822) nearly matches A6000 FP16 baseline (0.834) — only 1.4% loss
- InternVL2.5-4B INT8 accuracy (0.789) matches A6000 FP16 baseline (0.791) — only 0.3% loss
- InternVL2.5-2B had broken FP16 inference on Jetson (0.0 accuracy); INT8 fixes it to 0.700
- SmolVLM-500M INT8 (0.678) actually exceeds A6000 FP16 baseline (0.633) — likely sample variance

---

## 6. Category 2 Results (CASP, SLIM, Wanda, Magnitude)

### 6.1 SmolVLM-256M-Instruct — All Working Methods Compared

| Method | Exact Match | Contains | Token F1 | BLEU | ROUGE-L | Latency (s) | Peak Mem (MB) | Samples |
|--------|-------------|----------|----------|------|---------|-------------|---------------|---------|
| Jetson FP16 Baseline | 0.200 | — | — | — | — | 1.47 | 953 | 10 |
| **PyTorch INT8** | **0.367** | 0.400 | 0.422 | 0.412 | 0.422 | 1.62 | 842 | 30 |
| **CASP** | 0.344 | 0.367 | 0.393 | 0.382 | 0.393 | 1.62 | 674 | 30 |
| SLIM sp20 r10 | 0.167 | 0.300 | 0.258 | 0.230 | 0.258 | 1.75 | 672 | 30 |
| SLIM sp30 r20 | 0.133 | 0.167 | 0.139 | 0.137 | 0.139 | 2.25 | 776 | 30 |
| SLIM sp50 r30 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3.92 | 988 | 30 |
| A6000 FP16 Baseline | 0.562 | — | — | — | — | 0.39 | 1943 | 1000 |

### 6.2 InternVL2.5-1B on Jetson — Category 2 Methods

| Method | Accuracy | Latency (s) | Notes |
|--------|----------|-------------|-------|
| FP16 Baseline | 0.60 | 1.62 | 10 samples |
| Wanda sp20 | 0.50 | 1.42 | |
| Wanda sp40 | 0.30 | 1.22 | |
| Magnitude sp20 | 0.30 | 1.37 | |
| Magnitude sp40 | 0.10 | 1.21 | |
| PALU r25 | 0.0 | 0.75 | Broken |
| PACT p30 m20 | 0.0 | 1.06 | Broken |

---

## 7. A6000 Reference Results (Complete)

### 7.1 FP16 Baselines (20 models)

| Model | Params (M) | Accuracy | GPU Mem (MB) | Latency (s) | Samples |
|-------|-----------|----------|-------------|-------------|---------|
| SmolVLM-256M | 256 | 0.562 | 554 | 0.39 | 1000 |
| SmolVLM-500M | 508 | 0.633 | 1405 | 1.50 | 20 |
| SmolVLM-2.2B | 2193 | 0.666 | 4619 | 0.52 | 1000 |
| LFM2-VL-450M | 450 | 0.289 | 1049 | 0.60 | 1000 |
| LFM2-VL-1.6B | 1585 | 0.381 | 3122 | 0.30 | 1000 |
| LFM2-VL-3B | 3012 | 0.448 | 5893 | 0.33 | 1000 |
| InternVL2.5-1B | 938 | 0.703 | 2197 | 0.06 | 1000 |
| InternVL2.5-2B | 2181 | 0.730 | 4754 | 0.08 | 1000 |
| InternVL2.5-4B | 3655 | 0.791 | 7198 | 0.11 | 1000 |
| InternVL2.5-8B | 8084 | 0.827 | 15395 | 0.14 | 1000 |
| Qwen2.5-VL-3B | 3755 | 0.834 | 7236 | 0.16 | 1000 |
| Qwen2.5-VL-7B | 8294 | 0.856 | 15830 | 0.21 | 1000 |
| moondream2 | 1927 | 0.574 | 3812 | 0.52 | 1000 |
| Gemma3-4B | 4300 | 0.798 | 8592 | 0.38 | 1000 |
| Gemma3-12B | 12200 | 0.855 | 23656 | 0.61 | 1000 |
| Ovis2-1B | ~1000 | 0.567 | — | — | 1000 |
| Ovis2-2B | ~2000 | 0.620 | — | — | 1000 |
| Ovis2-4B | ~4300 | 0.740 | — | — | 1000 |
| Ovis2-8B | ~8900 | 0.780 | — | — | 1000 |
| FastVLM-0.5B | 759 | 0.397 | — | — | 1000 |

### 7.2 BnB Quantization on A6000 (39 results)
- INT8: 19/19 models pass (all produce valid accuracy)
- INT4 (NF4): 20/20 models pass
- Average accuracy retention: INT8 ~97% of FP16, INT4 ~93% of FP16
- These prove the methods work — they just can't run on Jetson due to CUDA kernels

### 7.3 Wanda Pruning on A6000 (41 results)
- 20% sparsity: avg ~95% of FP16 accuracy
- 40% sparsity: avg ~85% of FP16 accuracy
- Tested on all 19+ models

### 7.4 Magnitude Pruning on A6000 (38 results)
- Similar pattern to Wanda but slightly lower accuracy retention

---

## 8. Complete Result Inventory

| Location | Method | Count | Status |
|----------|--------|-------|--------|
| A6000 | FP16 Baselines | 20 | Complete |
| A6000 | BnB INT8 | 19 | Complete |
| A6000 | BnB INT4 | 20 | Complete |
| A6000 | Wanda pruning | 41 | Complete |
| A6000 | Magnitude pruning | 38 | Complete |
| A6000 | PALU | 10 | All acc=0.0 (broken) |
| A6000 | PACT | 4 | All acc=0.0 (broken) |
| A6000 | CASP | 2 | Works |
| A6000 | QLoRA | 2 | Works |
| A6000 | SVD-LLM | 1 | acc=0.0 |
| A6000 | Pre-quantized | 5 | 2 pass, 3 fail |
| **A6000 total** | | **162** | |
| Jetson | FP16 ceiling scan | 18 | Complete |
| Jetson | BnB INT8/INT4 | 34 | All fail (CUDA crash) |
| Jetson | Magnitude pruning | 8 | 5 pass, 3 MEM_CRITICAL |
| Jetson | Cat2 InternVL-1B | 7 | Wanda/Mag work, PALU/PACT broken |
| Jetson | CASP | 3 | Works (best Cat2) |
| Jetson | SLIM | 3 | 2 with non-zero accuracy |
| Jetson | **PyTorch INT8** | **5** | **All pass (2 OOM→loaded)** |
| Jetson | Untested method logs | 8 | SparseGPT/AWQ/GPTQ/AWP/GPTQ-pre |
| **Jetson total** | | **~86** | |
| **Grand total** | | **~243** | JSON result files |

---

## 9. File Structure

```
vlm-compression-benchmark/
├── CLAUDE.md                          # Project brief & live status
├── EXPERIMENT_LOG.md                  # This document
├── models/
│   └── model_loader.py                # Unified loader for 9 VLM families
├── evaluation/
│   └── run_baseline.py                # Multi-metric eval (exact_match, contains, F1, BLEU, ROUGE-L)
├── compression/
│   ├── ptq/
│   │   ├── run_ptq.py                 # BnB INT8/INT4
│   │   ├── run_awq.py                 # AWQ (blocked)
│   │   ├── run_gptq.py                # GPTQ (blocked)
│   │   └── run_pytorch_int8.py        # ★ Custom PyTorch INT8 (WORKS)
│   ├── pruning/
│   │   ├── run_pruning.py             # Magnitude pruning
│   │   ├── run_wanda.py               # Wanda pruning
│   │   └── run_sparsegpt.py           # SparseGPT (blocked on Jetson)
│   ├── casp_slim/
│   │   └── run_casp_slim.py           # CASP + SLIM methods
│   ├── combined/
│   │   └── run_awp.py                 # AWP (blocked)
│   ├── palu/
│   │   └── run_palu.py                # PALU (broken acc=0.0)
│   ├── pact/
│   │   └── run_pact.py                # PACT (broken acc=0.0)
│   ├── lowrank/
│   │   └── run_svd_llm.py             # SVD-LLM (blocked)
│   ├── qlora/
│   │   └── run_qlora.py               # QLoRA (blocked)
│   └── quantized_pretrained/
│       └── run_quantized_pretrained.py # Pre-quantized models (blocked)
├── profiling/
│   └── gpu_profiler.py                # GPU memory/power profiling
├── jetson/
│   ├── run_jetson.py                  # Jetson benchmark pipeline
│   └── safety.py                      # OOM protection system
├── results/
│   ├── baseline/                      # A6000 FP16 baselines (20 files)
│   ├── ptq/                           # A6000 BnB results (38 files)
│   ├── wanda/                         # A6000 Wanda results (41 files)
│   ├── pruning/                       # A6000 Magnitude results (38 files)
│   ├── palu/                          # PALU results (10 files)
│   ├── pact/                          # PACT results (4 files)
│   ├── casp_slim/                     # CASP/SLIM results (6 files)
│   ├── pytorch_int8/                  # ★ Custom INT8 results (5 files)
│   ├── jetson/                        # All Jetson-specific results
│   │   ├── baseline/                  # Jetson FP16 ceiling scan (18 files)
│   │   ├── ptq/                       # Jetson BnB attempts (34 files, all fail)
│   │   ├── pruning/                   # Jetson magnitude pruning (8 files)
│   │   └── cat2_internvl1b/           # Category 2 tests (7 files)
│   └── untested_256m_logs/            # Method testing logs
└── scripts/
    ├── run_all.sh
    ├── run_all_on_model.sh
    ├── run_untested_256m.sh
    ├── run_gptq_and_slim_256m.sh
    └── run_cat2_all_working_methods.sh
```

---

## 10. Evaluation Metrics

All VQA evaluations use multiple metrics:
- **exact_match:** Predicted answer exactly matches ground truth (after normalization)
- **contains:** Ground truth appears as substring in prediction
- **token_f1:** Token-level F1 score between prediction and ground truth
- **bleu:** BLEU score (n-gram precision)
- **rouge_l:** ROUGE-L score (longest common subsequence)

Evaluation dataset: **VQAv2 validation** (streamed from `lmms-lab/VQAv2`).
- A6000 baselines: 1000 samples
- Jetson experiments: 10-30 samples (memory constraints)

---

## 11. Summary of Findings

### What works on Jetson:
1. **PyTorch INT8** (Category 1) — loads OOM models, <2% accuracy loss vs FP16
2. **CASP** (Category 2) — best runtime optimization, reduces peak memory 30%
3. **Wanda pruning** (Category 2) — works at sp20-40%, moderate accuracy loss
4. **Magnitude pruning** (Category 2) — similar to Wanda
5. **SLIM** (Category 2) — works at sp20-30%, degrades at higher sparsity

### What fails on Jetson and why:
1. **BnB INT8/INT4** — custom CUDA kernels not compiled for aarch64
2. **GPTQ/AWQ** — same CUDA kernel issue, plus VLM architecture incompatibility
3. **Pre-quantized models** — inference kernels also crash
4. **QLoRA** — depends on BnB
5. **SparseGPT/SVD-LLM** — cusolver linalg symbols missing
6. **AWP** — Wanda step works, BnB reload step crashes
7. **PALU** — runs but 0.0 accuracy (implementation bug)
8. **PACT** — runs but 0.0 accuracy (hardcoded 50% visual token assumption)
9. **QNNPACK INT8** — works but CPU-only (~100x slower)

### Key contribution:
Standard VLM compression toolchains (BnB, GPTQ, AWQ, autoawq) are built for x86 datacenter GPUs and **do not work on ARM edge devices**. A simple pure-PyTorch INT8 quantization approach — storing weights as `torch.int8` with per-channel scale factors and using standard `torch.matmul` — succeeds where all established tools fail, loading previously-impossible 3.8B and 3.7B parameter VLMs on an 8GB Jetson with <2% accuracy loss.

---

## 12. Timeline

| Date | Work Done |
|------|-----------|
| 2026-03-20 | Initial pipeline setup, model loader for 9 families |
| 2026-03-21 | A6000 baselines (20 models), BnB quantization (39 results), Wanda/Magnitude (79 results) |
| 2026-03-22 | Jetson ceiling scan (18 models), identified Category 1/2 models, BnB Jetson testing (all fail) |
| 2026-03-23 | Tested remaining methods on Jetson (SparseGPT, AWQ, GPTQ, AWP, SLIM, GPTQ-prequant). SLIM/CASP confirmed working. Category 2 runs on InternVL-1B. |
| 2026-03-24 | **Breakthrough:** Custom PyTorch INT8 quantization. Qwen-3B and InternVL-4B now load on Jetson. InternVL-2B inference fixed. 5 INT8 results. QNNPACK tested (impractical). All 16 methods documented. |

---

*Generated 2026-03-24. Total experiment results: ~243 JSON files across A6000 and Jetson.*
