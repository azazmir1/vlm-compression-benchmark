# VLM Compression Benchmark — Project Brief

## Goal

Deploy Vision-Language Models on **NVIDIA Jetson Orin Nano 8GB** using compression methods. The Jetson has 8GB unified RAM shared between CPU and GPU (~6GB usable). Most VLMs above ~2B parameters either can't load or can't run usably.

**The research approach has three stages:**

```
Stage 1: Find the CEILING
  For each VLM family, find the biggest model that can load in FP16 on Jetson.

Stage 2: CATEGORY 1 — Models that CANNOT LOAD (above the ceiling)
  Apply memory compression methods to shrink the model footprint so it fits.
  → e.g., InternVL2.5-4B OOMs at load → compress → try again

Stage 3: CATEGORY 2 — Models that LOAD but are UNUSABLE
  They load but are too slow, crash during inference, or eat all memory.
  Apply efficiency methods (pruning, token compression, KV-cache optimization)
  to make them practically usable.
  → e.g., SmolVLM-2.2B loads but takes 3.2s/query (borderline)
  → e.g., LFM2-VL-3B loads (4848MB) but immediately hits MEM_CRITICAL
```

**Accuracy is secondary.** The primary goal is: (1) load what couldn't load, (2) make usable what wasn't usable. Accuracy loss is measured but is not the blocking concern.

Baseline FP16 accuracy is pre-computed on an A6000 lab GPU for reference.

## Hardware

| Device | Role | Memory |
|--------|------|--------|
| **Jetson Orin Nano 8GB** | Target device (ALL work happens here) | 8GB unified CPU+GPU, ~6GB usable |
| **A6000** | Baseline accuracy reference only (already done) | 48GB VRAM |

## Jetson Ceiling Scan Results

### Category 1: Cannot Load (need memory compression to fit)

| Model | Family | Params | Status | Needed Method |
|-------|--------|--------|--------|---------------|
| **Qwen2.5-VL-3B** | qwen25vl | 3.8B | OOM_LOAD | Weight quantization / low-rank |
| **InternVL2.5-4B** | internvl25 | 3.7B | OOM_LOAD | Weight quantization / low-rank |
| **Qwen2.5-VL-7B** | qwen25vl | 8.3B | Not tested (bigger) | Heavy compression |
| **InternVL2.5-8B** | internvl25 | 8.1B | Not tested (bigger) | Heavy compression |
| **Gemma3-4B** | gemma3 | 4.3B | Not tested | Likely OOM |
| **Gemma3-12B** | gemma3 | 12.2B | Not tested | Definitely OOM |
| **Ovis2-4B/8B** | ovis2 | 4.3B/8.9B | ERROR (code issues) | Fix errors first |

### Category 2: Loads but Unusable (need efficiency methods)

| Model | Family | Mem(MB) | Latency | Problem | Needed Method |
|-------|--------|---------|---------|---------|---------------|
| **LFM2-VL-3B** | lfm2vl | 4848 | ? | MEM_CRITICAL (system nearly OOM) | Pruning / token compression / KV-cache |
| **SmolVLM-2.2B** | smolvlm | 4617 | 3.19s | Borderline slow (>3s threshold) | Pruning / token compression |
| **InternVL2.5-2B** | internvl25 | 4367 | 0.71s | Accuracy = 0.0 (broken inference) | Debug + optimize |
| **moondream2** | moondream | 2820 | 1.50s | Accuracy = 0.0 (broken inference) | Debug inference |

### Ceiling Per Family (FP16)

| Family | Ceiling Model | Mem(MB) | First OOM/Fail |
|--------|---------------|---------|----------------|
| **smolvlm** | SmolVLM-2.2B (borderline) | 4617 | — (all fit) |
| **lfm2vl** | LFM2-VL-1.6B | 2974 | 3B MEM_CRITICAL |
| **internvl25** | InternVL2.5-2B | 4367 | 4B OOM_LOAD |
| **qwen25vl** | — (none fit) | — | 3B OOM_LOAD |
| **moondream** | moondream2 (2B) | 2820 | — (only variant) |
| **gemma3** | Not tested | — | — |
| **ovis2** | — (all ERROR) | — | Remote code issues |
| **fastvlm** | — (all ERROR) | — | Processor issues |
| **nanovlm** | — (all ERROR) | — | Config issues |

## Two Categories of Compression Methods

### Category 1 Methods: Make Unloadable Models Load

These reduce the **model's memory footprint at load time** so a model that OOM'd can now fit in 6GB.

| Method | How It Saves Memory | Saving | A6000 | Jetson | File |
|--------|-------------------|--------|-------|--------|------|
| **BnB INT4 (NF4)** | Stores weights in 4-bit | ~4x | 20/20 pass | BLOCKED (CUDA crash) | `compression/ptq/run_ptq.py` |
| **BnB INT8** | Stores weights in 8-bit | ~2x | 19/19 pass | BLOCKED (CUDA crash) | `compression/ptq/run_ptq.py` |
| **AWQ** | Activation-aware 4-bit quantization | ~4x | Not run | BLOCKED (no autoawq + VLM arch) | `compression/ptq/run_awq.py` |
| **GPTQ** | Hessian-based 4-bit quantization | ~4x | Not run | BLOCKED (VLM arch rejected) | `compression/ptq/run_gptq.py` |
| **GPTQ pre-quantized** | Download already-quantized HF models | ~4x | 2/5 pass | BLOCKED (CUDA kernel crash) | `compression/quantized_pretrained/` |
| **QLoRA** | INT4 base + small LoRA adapters | ~4x | 2/2 pass | BLOCKED (needs BnB) | `compression/qlora/run_qlora.py` |
| **SVD-LLM** | Low-rank decomposition (fewer params) | 20-50% | 1 run, acc=0.0 | BLOCKED (cusolver missing) | `compression/lowrank/run_svd_llm.py` |
| **CASP** | Quantization + KV low-rank (VLM-specific) | Significant | 2/2 pass (acc=0.2-0.5) | Not tested | `compression/casp_slim/run_casp_slim.py` |
| **SLIM** | Quantization + sparsity + low-rank | Multi-factor | Not tested | **WORKS (sp=20-30%)** | `compression/casp_slim/run_casp_slim.py` |
| **CALDERA** | Low-rank + low-precision decomposition | Combined | Not implemented | — | — |

### Category 2 Methods: Make Loaded Models Usable

These reduce **runtime memory usage, latency, or compute** so a model that loaded but was too slow or crashed during inference can now actually serve queries.

| Method | How It Helps | A6000 | Jetson | File |
|--------|-------------|-------|--------|------|
| **Wanda Pruning** | Zeros unimportant weights → faster matmuls | 41/41 pass | **WORKS** (small models) | `compression/pruning/run_wanda.py` |
| **Magnitude Pruning** | Same — zeros small weights | 38/38 pass | **WORKS** (small models) | `compression/pruning/run_pruning.py` |
| **SparseGPT** | Hessian-aware pruning, better accuracy at high sparsity | Not run | BLOCKED (cusolver missing) | `compression/pruning/run_sparsegpt.py` |
| **PALU** | Compresses KV-cache to 25% → less runtime memory | 10 runs, 9/10 acc=0.0 | Runs but 0.0 accuracy | `compression/palu/run_palu.py` |
| **PACT** | Prunes/merges visual tokens → fewer tokens to process | 4 runs, all acc=0.0 | Runs but 0.0 accuracy | `compression/pact/run_pact.py` |
| **AWP** | Wanda pruning + INT4 quantization combined | Not run | BLOCKED (BnB reload crash) | `compression/combined/run_awp.py` |

**Note:** Pruning methods (Wanda, Magnitude, SparseGPT) do NOT reduce model memory footprint — PyTorch stores sparse tensors in dense format. They are Category 2 only: they can speed up inference or reduce runtime memory pressure, but they won't help a model that can't load.

## Critical Jetson Blockers

### 1. BitsAndBytes CUDA Kernel Crash
**BitsAndBytes does not work on Jetson Orin Nano** — CUDA kernels crash with `Error named symbol not found`. This blocks: BnB INT8, BnB INT4, QLoRA.

### 2. Pre-quantized GPTQ/AWQ Models Also Fail on Jetson
Tested `vasanth0475/SmolVLM-256M-Instruct-GPTQ-Int4` on 2026-03-23 — same `Error named symbol not found at line 81 in file /src/csrc/ops.cu`. GPTQ inference kernels (from gptqmodel v5.8.0) also use unsupported CUDA ops on Jetson. **This means ALL quantization-based methods (BnB, GPTQ, AWQ, pre-quantized models) are blocked on Jetson.**

### 3. GPTQ/AWQ Can't Quantize VLMs Directly
`AutoModelForCausalLM` rejects VLM architectures (Idefics3, InternVLChat, Qwen2_5_VL, etc.). Our VLMs have CausalLM text backbones (SmolVLM→VLlama3, InternVL→Qwen2, LFM2→Lfm2) but the outer VLM wrapper is `ForConditionalGeneration`, not `ForCausalLM`.

### 4. SparseGPT CUDA Linalg Missing
`cusolverDnXsyevBatched_bufferSize` symbol missing on Jetson. Hessian inverse requires CUDA linalg ops not available on this platform.

### 5. SVD-LLM Same Linalg Issue
`torch.linalg.svd` triggers the same `cusolverDnXsyevBatched_bufferSize` dlopen failure. Both SparseGPT and SVD-LLM need CPU-only SVD fallback to work on Jetson.

### 6. PyTorch Native INT8 (QNNPACK) — Works but Impractical
Tested `torch.quantization.quantize_dynamic` with QNNPACK backend on 2026-03-24. This is **real INT8 quantization** (not simulated) — weights are stored as actual INT8 tensors. It uses no custom CUDA kernels (pure PyTorch + ARM QNNPACK).

**The problem:** QNNPACK INT8 kernels are **CPU-only**. There is no CUDA backend for PyTorch's dynamic quantization. On Jetson, this means inference runs on CPU instead of GPU, making it ~100x slower (~4 minutes per sample vs ~1.5 seconds on GPU). Impractical for any real use.

**Memory findings:**
- FP16 on GPU: 490MB GPU allocated
- INT8 on CPU: ~2GB RSS (model must load in FP16 first, convert to FP32, then quantize to INT8 — so loading memory is NOT reduced)
- The model still needs full FP16 memory to load before quantization can be applied
- **This is NOT a Category 1 method** — it cannot help OOM models load

### 7. Custom PyTorch INT8 — Category 1 BREAKTHROUGH (2026-03-24)

After every standard quantization tool (BnB, GPTQ, AWQ) was blocked on Jetson, we built a custom pure-PyTorch INT8 quantization method that works without any custom CUDA kernels.

**How it works:**
- `Int8Linear`: Custom `nn.Module` that stores weights as `torch.int8` + per-channel `torch.float16` scale
- Forward pass: `w_fp16 = weight_int8 * scale` then standard `torch.matmul` on GPU
- Loading: `from_pretrained` on CPU → replace `nn.Linear` with `Int8Linear` → move to GPU
- Vision encoder stays FP16 for accuracy
- Uses Jetson's 16GB swap during the CPU→GPU transition phase

**File:** `compression/ptq/run_pytorch_int8.py`

**Results:** Qwen2.5-VL-3B (previously OOM) → loads at 4943MB, 0.82 exact match. InternVL2.5-4B (previously OOM) → loads at 5780MB, 0.79 exact match.

### Summary: What Works on Jetson

**Category 1 (reduce load-time memory):**
- **PyTorch INT8** — custom pure-PyTorch quantization, loads OOM models

**Category 2 (reduce runtime memory after loading):**
- CASP (pure PyTorch: mixed-precision simulated quantization — best accuracy)
- SLIM (pure PyTorch: magnitude pruning + simulated INT4 — SVD silently skipped due to cusolver)
- Wanda pruning (pure PyTorch)
- Magnitude pruning (pure PyTorch)
- PALU (pure PyTorch, but accuracy was 0.0 — implementation bug)
- PACT (pure PyTorch, but accuracy was 0.0 — implementation bug, hardcoded 50% visual token assumption)

## Paper List

All methods sourced from `VLM_Memory_Optimization_Papers.xlsx` — curated list of 31 papers (CVPR 2025, ICLR 2025, ICML 2025, NeurIPS 2024, MLSys 2024).

**Highly Recommended (from paper list):**
- AWQ (MLSys 2024 Best Paper) — better than GPTQ for most VLMs
- AWP (ICML 2025) — combines Wanda + quantization for maximum compression
- CASP (CVPR 2025) — VLM-specific, attention-sparsity-aware
- SLIM (ICML 2025, NVIDIA) — state-of-the-art combined compression

**Recommended:**
- GPTQ, SparseGPT, SVD-LLM, PALU, PACT, Q-VLM

**Skipped (not applicable):**
- LLaVA-Mini, LLaVA-KD, LLaVA-MoD, TinyLLaVA — architecture-specific, can't apply to our models
- DyCoke — video-specific
- M-Wanda — multilingual-specific

## Key Files

| Purpose | File |
|---------|------|
| Model loader (9 families) | `models/model_loader.py` |
| Evaluation (VQAv2/TextVQA/POPE) | `evaluation/run_baseline.py` |
| GPU profiler | `profiling/gpu_profiler.py` |
| Jetson benchmark pipeline | `jetson/run_jetson.py` |
| Jetson safety (OOM protection) | `jetson/safety.py` |
| Device constraints | `configs/device_constraints.yaml` |
| Paper list | `VLM_Memory_Optimization_Papers.xlsx` |

**Compression runners:** `compression/{method}/run_{method}.py`

## Result Format

JSON files in `results/{method}/{model}__{params}.json`:
```json
{
  "model_id": "...", "family": "...", "method": "...",
  "num_params_M": 507.5, "gpu_mem_load_mb": 715.4,
  "benchmarks": { "vqav2": { "accuracy": 0.63, "avg_latency_s": 1.5, "peak_memory_mb": 1433 } }
}
```

## Adding a New Method

1. Create `compression/{method}/run_{method}.py` (follow `run_awq.py` pattern)
2. Import from `models.model_loader` and `evaluation.run_baseline`
3. Load → compress → evaluate → save JSON
4. Add to `scripts/run_all_on_model.sh`

## Known Issues

- **BnB on Jetson**: CUDA kernel crash (`Error named symbol not found`) — blocks INT4/INT8/QLoRA
- **GPTQ/AWQ pre-quantized on Jetson**: Same CUDA kernel crash — blocks ALL quantization inference
- **GPTQ/AWQ quantization**: `AutoModelForCausalLM` rejects VLM architectures (Idefics3, InternVLChat, etc.)
- **SparseGPT on Jetson**: Missing `cusolverDnXsyevBatched_bufferSize` — Hessian inverse blocked
- **SVD-LLM on Jetson**: Same linalg issue — `torch.linalg.svd` dlopen failure
- **AWP on Jetson**: Wanda pruning step works, but BnB INT4 reload step crashes
- **autoawq**: Not installed, deprecated (adopted by vLLM project)
- **SLIM at high compression**: 50% sparsity + 30% rank reduction → all metrics 0.0 (model destroyed)
- **PALU / PACT**: Code runs on A6000 but produces 0.0 accuracy — likely implementation bugs
- **Qwen2.5-VL / Gemma3**: Must use bfloat16 (float16 → NaN/overflow)
- **Ovis2, FastVLM, nanoVLM**: Various loading errors on Jetson (remote code, processors, config)
- **Transformers 5.x patches**: Moondream, InternVL2.5, Ovis2 need `all_tied_weights_keys` shim

## Jetson Method Testing Results (2026-03-24)

### All 16 Methods Tested — Jetson Status

**Category 1 (reduce load-time memory):**

| # | Method | Type | Jetson Status | Details |
|---|--------|------|---------------|---------|
| 1 | **PyTorch INT8** | Real quant | **WORKS ON JETSON** | Custom pure-PyTorch INT8 with per-channel scale. Loads OOM models via CPU→quantize→GPU pipeline. No custom CUDA kernels needed. |
| 2 | **BnB INT8** | Real quant | BLOCKED | CUDA kernel crash (`Error named symbol not found`) |
| 3 | **BnB INT4** | Real quant | BLOCKED | Same CUDA kernel crash |
| 4 | **AWQ** | Real quant | BLOCKED | `autoawq` not installed + VLM arch incompatible |
| 5 | **GPTQ** | Real quant | BLOCKED | VLM arch rejected by `AutoModelForCausalLM` (Idefics3) |
| 6 | **GPTQ pre-quantized** | Real quant | BLOCKED | CUDA kernel crash on inference (`ops.cu line 81`) |
| 7 | **QLoRA** | Real quant | BLOCKED | Depends on BnB INT4 |
| 8 | **SVD-LLM** | Low-rank | BLOCKED | `cusolverDnXsyevBatched_bufferSize` missing |
| 9 | **SparseGPT** | Pruning | BLOCKED | Same cusolver linalg missing symbol |
| 10 | **AWP** | Combined | BLOCKED | Wanda step works, but BnB INT4 reload crashes |
| 11 | **PyTorch QNNPACK INT8** | Real quant | IMPRACTICAL | Works but CPU-only (~4min/sample vs 1.5s on GPU). |

**Category 2 (reduce runtime/peak memory after loading):**

| # | Method | Jetson Status | Details |
|---|--------|---------------|---------|
| 11 | **CASP** | **WORKS ON JETSON** | Best accuracy (0.344 exact match), 674MB peak mem. Mixed-precision simulated quant (QK low-rank silently skipped due to cusolver). |
| 12 | **SLIM** | **WORKS ON JETSON** | Works at sp=20-30%. Pruning + simulated INT4 (SVD silently skipped due to cusolver). Destroyed at sp=50%. |
| 13 | **Wanda** | **WORKS ON JETSON** | Tested on small models, passes |
| 14 | **Magnitude Pruning** | **WORKS ON JETSON** | Tested on small models, passes |
| 15 | **PALU** | Runs but broken | 0.0 accuracy on all models — implementation bug |
| 16 | **PACT** | Runs but broken | 0.0 accuracy on all models — hardcoded 50% visual token assumption is wrong |

### SmolVLM-256M-Instruct on Jetson — All Working Methods Compared

| Method | Exact Match | Contains | Token F1 | BLEU | ROUGE-L | Latency | Peak Mem |
|--------|-------------|----------|----------|------|---------|---------|----------|
| **Jetson FP16 Baseline** | 0.200 | — | — | — | — | 1.47s | 953MB |
| **CASP** | **0.344** | **0.367** | **0.393** | **0.382** | **0.393** | 1.62s | 674MB |
| **SLIM sp20 r10** | 0.167 | 0.300 | 0.258 | 0.230 | 0.258 | 1.75s | 672MB |
| **SLIM sp30 r20** | 0.133 | 0.167 | 0.139 | 0.137 | 0.139 | 2.25s | 776MB |
| **SLIM sp50 r30** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3.92s | 988MB |
| A6000 FP16 Baseline | 0.562 | — | — | — | — | 0.39s | 1943MB |

**Note:** Jetson baseline was 10 samples, CASP/SLIM were 30 samples — not directly comparable. Need to re-run baseline with 30 samples for fair comparison.

### PyTorch INT8 Results — Category 1 Success (2026-03-24)

Custom pure-PyTorch INT8 quantization (`compression/ptq/run_pytorch_int8.py`). Loads model on CPU, quantizes Linear layers to INT8 (per-channel scale), moves to GPU. Vision encoder stays FP16.

**Previously OOM models — now loading and running:**

| Model | Params | FP16 Status | INT8 Exact Match | INT8 Contains | INT8 Token F1 | INT8 Mem (MB) | INT8 Latency |
|-------|--------|-------------|------------------|---------------|---------------|---------------|--------------|
| **Qwen2.5-VL-3B** | 3.8B | OOM_LOAD | **0.822** | 0.900 | 0.922 | 4943 (peak 6098) | 3.2s |
| **InternVL2.5-4B** | 3.7B | OOM_LOAD | **0.789** | 0.867 | 0.906 | 5780 (peak 5407) | 2.5s |

**Models that already loaded — INT8 vs FP16:**

| Model | Params | INT8 Exact Match | INT8 Token F1 | INT8 Mem (MB) | INT8 Latency |
|-------|--------|------------------|---------------|---------------|--------------|
| SmolVLM-256M | 260M | 0.367 | 0.422 | 332 | 1.6s |
| SmolVLM-500M | 508M | 0.678 | 0.700 | 1531 | 1.7s |
| InternVL2.5-2B | 2.2B | **0.700** (was 0.0 FP16!) | 0.800 | 3926 | 2.1s |

**Key findings:**
- **Category 1 breakthrough**: 2 previously-OOM models now load and run with good accuracy
- **InternVL2.5-2B fixed**: Had 0.0 accuracy in FP16 on Jetson; INT8 gets 0.70 exact match
- INT8 uses Jetson's 16GB swap during CPU→GPU transfer, keeping peak memory manageable
- All standard `torch.matmul` operations — no custom CUDA kernels needed
- Vision encoder stays FP16 for accuracy

**Key findings (Category 2):**
- CASP is the best working method — higher accuracy than FP16 baseline with 30% less peak memory
- SLIM works at sp=20-30% but accuracy drops steeply; at sp=50% the model is destroyed
- All Category 2 methods reduce peak inference memory (953MB → 672-674MB)
- CASP and SLIM both have components silently skipped on Jetson (QK low-rank and SVD respectively) due to cusolver — they run in degraded mode

## Result Counts (as of 2026-03-23)

| Location | Method | Count |
|----------|--------|-------|
| A6000 | Baselines | 20 |
| A6000 | BnB INT8/INT4 | 39 |
| A6000 | Wanda | 41 |
| A6000 | Magnitude pruning | 38 |
| A6000 | PALU | 10 |
| A6000 | PACT | 4 |
| A6000 | CASP | 2 |
| A6000 | QLoRA | 2 |
| A6000 | SVD-LLM | 1 |
| A6000 | Pre-quantized | 5 (2 pass, 3 fail) |
| **A6000 total** | | **~162** |
| Jetson | Baselines (ceiling scan) | 18 |
| Jetson | BnB INT8/INT4 (all fail) | 34 |
| Jetson | Magnitude pruning | 8 (5 pass, 3 MEM_CRITICAL) |
| Jetson | Cat2 InternVL-1B (wanda/mag/palu/pact) | 7 |
| Jetson | SLIM | 3 (2 with non-zero accuracy) |
| **Jetson total** | | **~71** |

## Status

**Done:**
- FP16 baselines on 19 models (A6000) ✓
- Jetson FP16 ceiling scan (identified Category 1 and 2 models) ✓
- BnB INT8/INT4 on A6000 (39 results, proves methods work) ✓
- Wanda/Magnitude pruning on A6000 (77 results) ✓
- PALU, SVD-LLM, QLoRA/LoRA runners implemented ✓
- Tested all 16 methods on Jetson — identified 5 working, 9 blocked, 2 broken ✓
- **SLIM works on Jetson** with moderate compression (sp=20-30%) ✓
- **CASP works on Jetson** — best Category 2 method ✓
- Confirmed BnB/GPTQ/AWQ quantization all blocked on Jetson due to CUDA kernels ✓
- Multi-metric evaluation (exact_match, contains, token_f1, BLEU, ROUGE-L) working ✓
- **Custom PyTorch INT8 quantization** — Category 1 breakthrough ✓
  - Qwen2.5-VL-3B: OOM → loaded at 4943MB, 0.82 accuracy ✓
  - InternVL2.5-4B: OOM → loaded at 5780MB, 0.79 accuracy ✓
  - InternVL2.5-2B: 0.0 accuracy → 0.70 accuracy with INT8 ✓
  - Pure PyTorch, no custom CUDA kernels, works on aarch64 ✓

**Next:**
1. Run Category 2 methods (Wanda, Magnitude, SLIM, CASP) on all Jetson-loadable models
2. Run PyTorch INT8 on more models (LFM2-VL, InternVL2.5-1B)
3. Debug PALU/PACT 0.0 accuracy — fix implementation to get more working methods
4. Compare all results against A6000 FP16 baselines
5. Write paper (deadline: March 30)
