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

## Directory Structure

```
vlm-compression-benchmark/
├── models/                     # Model loading (9 VLM families)
│   └── model_loader.py
├── evaluation/                 # VQA evaluation (VQAv2/TextVQA/POPE)
│   └── run_baseline.py
├── compression/                # Compression method runners
│   ├── ptq/                    #   Post-training quantization (BnB, GPTQ, AWQ, PyTorch INT8/INT4)
│   ├── pruning/                #   Pruning (Wanda, Magnitude, SparseGPT)
│   ├── casp_slim/              #   CASP + SLIM (VLM-specific)
│   ├── palu/                   #   KV-cache compression
│   ├── pact/                   #   Visual token pruning
│   ├── lowrank/                #   SVD-LLM low-rank decomposition
│   ├── combined/               #   Combined methods (AWP, INT4+CASP)
│   ├── qlora/                  #   QLoRA fine-tuning
│   └── quantized_pretrained/   #   Pre-quantized HF models
├── profiling/                  # ★ Profiling framework (self-contained)
│   ├── profile_all.py          #   Main profiling pipeline (run this)
│   ├── benchmark_compressed_hf.py  # Quantized model loader (HQQ/INT8/INT4/GPTQ)
│   ├── token_timer.py          #   Per-token timing via LogitsProcessor
│   ├── tegrastats_monitor.py   #   Jetson power/thermal/RAM monitoring
│   ├── gpu_profiler.py         #   GPU memory + utilization polling
│   ├── cpu_profiler.py         #   CPU profiling utilities
│   ├── detailed_metrics.py     #   Dataclasses for profiling results
│   └── hooks/                  #   Component-level forward hook profiling
│       ├── hook_manager.py     #     Hook registration + category aggregation
│       ├── timing_tracker.py   #     CUDA event-based per-module timing
│       └── module_categorizer.py   # Regex-based module classification
├── jetson/                     # Jetson-specific benchmark pipeline
│   ├── run_jetson.py
│   └── safety.py               #   OOM protection
├── scripts/                    # Utility scripts (quantization, analysis, etc.)
├── results/                    # All benchmark results (JSON)
│   ├── a6000/                  #   A6000 FP16 baselines + compression results
│   ├── jetson/                 #   Jetson results by method
│   └── profiling/              #   Detailed profiling results (from profile_all.py)
├── research_paper/             # Paper drafts and figures
├── configs/                    # Device constraints YAML
├── analysis/                   # Deployability analysis
└── notebooks/                  # Jupyter analysis notebooks
```

## Profiling Framework (`profiling/`)

Self-contained profiling pipeline. Can be fetched independently for use on other devices (e.g., Raspberry Pi).

### Running

```bash
# Single model (FP16)
python3 profiling/profile_all.py --model_id HuggingFaceTB/SmolVLM-256M-Instruct \
    --num_samples 50 --num_warmup 3 --max_tokens 30 --components

# All models from Excel (best quantization method per OOM model)
python3 profiling/profile_all.py --all --best-only \
    --num_samples 50 --num_warmup 3 --max_tokens 30 --components

# Specific family
python3 profiling/profile_all.py --family SmolVLM --num_samples 10 --components
```

### What It Measures

**Per-sample granular profiling:**
- `token_timing`: CUDA-event per-token latency via LogitsProcessor (prefill_ms, per_token_ms[], decode_ms, decode_throughput_tok_s)
- `components`: Forward hook timing per module category (vision_encoder, projection, text_embeddings, attention, feedforward, normalization, output, other)
- `memory`: GPU memory before/peak/after each sample
- `accuracy`: Multi-metric (exact_match, contains, token_f1, bleu, rouge_l) per sample

**Aggregated across all samples:**
- `timing`: avg_preprocessing_ms, avg_prefill_ms, avg_decode_ms, avg_total_ms, avg_tokens_generated, avg_input_tokens
- `throughput`: avg_tok_s, avg_decode_tok_s, samples_per_s
- `memory`: gpu_peak_mb, gpu_avg_mb, avg_inference_peak_mb
- `tegrastats` (Jetson-only): avg/peak power (W), avg/peak GPU temp (°C), avg/peak CPU temp (°C), avg/peak RAM (MB), avg/peak GPU utilization (%)
- `components`: Aggregated category breakdown with total_ms, avg_ms, percentage, hook call count

### Architecture

- **TokenTimingProcessor** (`token_timer.py`): LogitsProcessor that records CUDA events at each `.generate()` step. Gives exact prefill vs decode separation without streaming.
- **HookManager** (`hooks/hook_manager.py`): Registers pre/post forward hooks on all leaf modules. Tracks `current_token_idx` for per-generation-step attribution.
- **TimingTracker** (`hooks/timing_tracker.py`): CUDA event pairs with deferred synchronization — single `torch.cuda.synchronize()` after all hooks fire.
- **ModuleCategorizer** (`hooks/module_categorizer.py`): Regex patterns for 8 component categories across all 9 VLM families.
- **TegraStatsMonitor** (`tegrastats_monitor.py`): Background thread parsing `tegrastats --interval 200` for power (VDD_IN mW), temperature (gpu@XC, cpu@XC), RAM, GPU utilization (GR3D_FREQ %).
- **GPUProfiler** (`gpu_profiler.py`): pynvml-based GPU memory and utilization polling.

### Profiling Result Format

JSON files in `results/profiling/{ModelVariant}__{method}.json`:
```json
{
  "model_id": "HuggingFaceTB/SmolVLM-256M-Instruct",
  "method": "fp16",
  "family": "smolvlm",
  "num_samples": 50,
  "accuracy": { "exact_match": 0.47, "contains": 0.52, "token_f1": 0.52, "bleu": 0.51, "rouge_l": 0.52 },
  "timing": { "avg_prefill_ms": 851.4, "avg_decode_ms": 356.6, "avg_total_ms": 1623.9, "avg_tokens_generated": 3.7, "avg_input_tokens": 918.2 },
  "throughput": { "avg_tok_s": 3.03, "avg_decode_tok_s": 7.47 },
  "memory": { "gpu_peak_mb": 886.7 },
  "tegrastats": { "avg_power_w": 12.5, "peak_power_w": 18.5, "avg_gpu_temp_c": 55.3, "avg_gpu_util_pct": 45.7 },
  "components": {
    "vision_encoder":   { "avg_ms": 393.9, "percentage": 57.6 },
    "normalization":    { "avg_ms": 118.8, "percentage": 17.4 },
    "feedforward":      { "avg_ms": 81.8,  "percentage": 12.0 },
    "attention":        { "avg_ms": 79.0,  "percentage": 11.5 },
    "projection":       { "avg_ms": 1.6,   "percentage": 0.2 },
    "text_embeddings":  { "avg_ms": 0.8,   "percentage": 0.1 },
    "output":           { "avg_ms": 3.4,   "percentage": 0.5 }
  },
  "sample_details": [ { "per_token_ms": [...], "components": {...}, "accuracy": {...} } ]
}
```

### Profiling Results (2026-03-27, 50 samples each)

| Model | Method | EM | Prefill | Decode | Total | Throughput | Memory | Power |
|-------|--------|-----|---------|--------|-------|-----------|--------|-------|
| LFM2-VL-450M | fp16 | 0.807 | 149ms | 102ms | 268ms | 14.1 t/s | 902MB | 11.8W |
| SmolVLM-256M | fp16 | 0.473 | 851ms | 357ms | 1624ms | 7.5 t/s | 887MB | 12.5W |
| SmolVLM-500M | fp16 | 0.613 | 870ms | 367ms | 1630ms | 7.0 t/s | 1473MB | 13.7W |
| InternVL2.5-1B | fp16 | 0.647 | — | — | 614ms | — | 1884MB | 13.7W |
| FastVLM-0.5B | fp16 | 0.193 | 509ms | 3414ms | 3987ms | 8.5 t/s | 1346MB | 8.6W |
| LFM2-VL-1.6B | HQQ-INT4 | 0.820 | 775ms | 676ms | 1476ms | 1.9 t/s | 2600MB | 16.5W |
| LFM2-VL-3B | HQQ-INT4 | 0.900 | 1524ms | 1586ms | 3136ms | 0.9 t/s | 3381MB | 16.5W |
| SmolVLM-2.2B | HQQ-INT4 | 0.647 | 3133ms | 1819ms | 5151ms | 1.4 t/s | 2957MB | 18.7W |
| Qwen2.5-VL-3B | PT-INT4 | 0.800 | 2282ms | 1645ms | 3960ms | 0.8 t/s | 4969MB | 17.2W |
| InternVL2.5-4B | PT-INT8 | 0.807 | — | — | 2125ms | — | 5408MB | 18.0W |

**Note:** InternVL2.5 shows Prefill=0 because it uses `model.chat()` (not `.generate()`), so TokenTimingProcessor can't instrument per-token timing. Total latency is measured via CUDA events.
**gemma-3-4b-it (INT4)** was OOM-killed — too large even at INT4 for 8GB Jetson.

## Quantized Model Repos (HuggingFace)

Pre-quantized models uploaded to `Azaz666/` on HuggingFace:
- `Azaz666/SmolVLM-Instruct-{HQQ-INT4,PYTORCH-INT8,PYTORCH-INT4,GPTQ-INT4}`
- `Azaz666/LFM2-VL-{1.6B,3B}-{HQQ-INT4,PYTORCH-INT8,PYTORCH-INT4,GPTQ-INT4}`
- `Azaz666/InternVL2_5-4B-{PYTORCH-INT8,PYTORCH-INT4}`
- `Azaz666/Qwen2.5-VL-3B-Instruct-{HQQ-INT4,PYTORCH-INT8,PYTORCH-INT4,GPTQ-Int4}`
- `Azaz666/gemma-3-{4b,12b}-it-{PYTORCH-INT8,PYTORCH-INT4}`

## Key Files

| Purpose | File |
|---------|------|
| Model loader (9 families) | `models/model_loader.py` |
| Evaluation (VQAv2) | `evaluation/run_baseline.py` |
| **Profiling pipeline** | `profiling/profile_all.py` |
| **Quantized model loader** | `profiling/benchmark_compressed_hf.py` |
| **Per-token timer** | `profiling/token_timer.py` |
| **Jetson power monitor** | `profiling/tegrastats_monitor.py` |
| **Component hooks** | `profiling/hooks/` |
| GPU profiler | `profiling/gpu_profiler.py` |
| Jetson benchmark pipeline | `jetson/run_jetson.py` |
| Jetson safety (OOM protection) | `jetson/safety.py` |
| Device constraints | `configs/device_constraints.yaml` |
| Model tracking | `VLM_Model_Families_Jetson_Status.xlsx` |
| Paper list | `VLM_Memory_Optimization_Papers.xlsx` |

**Compression runners:** `compression/{method}/run_{method}.py`

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

### Category 2 Methods: Make Loaded Models Usable

These reduce **runtime memory usage, latency, or compute** so a model that loaded but was too slow or crashed during inference can now actually serve queries.

| Method | How It Helps | A6000 | Jetson | File |
|--------|-------------|-------|--------|------|
| **Wanda Pruning** | Zeros unimportant weights | 41/41 pass | **WORKS** | `compression/pruning/run_wanda.py` |
| **Magnitude Pruning** | Zeros small weights | 38/38 pass | **WORKS** | `compression/pruning/run_pruning.py` |
| **SparseGPT** | Hessian-aware pruning | Not run | BLOCKED (cusolver) | `compression/pruning/run_sparsegpt.py` |
| **PALU** | KV-cache compression | 10 runs, 9/10 acc=0.0 | Runs but 0.0 accuracy | `compression/palu/run_palu.py` |
| **PACT** | Visual token pruning | 4 runs, all acc=0.0 | Runs but 0.0 accuracy | `compression/pact/run_pact.py` |
| **AWP** | Wanda + INT4 combined | Not run | BLOCKED (BnB crash) | `compression/combined/run_awp.py` |

## Critical Jetson Blockers

1. **BitsAndBytes CUDA Kernel Crash** — `Error named symbol not found`. Blocks BnB INT8, BnB INT4, QLoRA.
2. **Pre-quantized GPTQ/AWQ Also Fail** — Same CUDA kernel crash on inference (`ops.cu line 81`).
3. **GPTQ/AWQ Can't Quantize VLMs** — `AutoModelForCausalLM` rejects VLM architectures.
4. **SparseGPT/SVD-LLM CUDA Linalg Missing** — `cusolverDnXsyevBatched_bufferSize` symbol missing on Jetson.
5. **PyTorch QNNPACK INT8** — Works but CPU-only (~4min/sample vs 1.5s GPU). Impractical.

### What Works on Jetson

**Category 1:** Custom PyTorch INT8/INT4, HQQ-INT4 (pure-PyTorch dequantize-on-forward)
**Category 2:** CASP, SLIM (sp=20-30%), Wanda pruning, Magnitude pruning

## Known Issues

- **BnB on Jetson**: CUDA kernel crash — blocks INT4/INT8/QLoRA
- **GPTQ/AWQ on Jetson**: Same CUDA crash — blocks ALL quantization inference
- **SparseGPT/SVD-LLM on Jetson**: Missing cusolver symbols
- **SLIM at high compression**: 50% sparsity + 30% rank → all metrics 0.0
- **PALU/PACT**: 0.0 accuracy — implementation bugs
- **Qwen2.5-VL / Gemma3**: Must use bfloat16 (float16 → NaN/overflow)
- **Ovis2, FastVLM, nanoVLM**: Various loading errors on Jetson
- **InternVL2.5 profiling**: Uses `model.chat()` not `.generate()` — no per-token timing, only total latency
- **gemma-3-4b-it**: OOM even at INT4 on 8GB Jetson

## Status (2026-03-27)

**Done:**
- FP16 baselines on 19 models (A6000) ✓
- Jetson FP16 ceiling scan ✓
- Tested all 16 compression methods on Jetson ✓
- Custom PyTorch INT8/INT4 quantization — loads OOM models ✓
- HQQ-INT4 pure-PyTorch quantization working ✓
- Multi-metric evaluation (exact_match, contains, token_f1, BLEU, ROUGE-L) ✓
- **Full profiling pipeline** with component-level timing, power monitoring, per-token latency ✓
- **Profiled 10 models** on Jetson (5 FP16 + 5 quantized, 50 samples each) ✓
- Quantized models uploaded to HuggingFace (Azaz666/) ✓

**Next:**
1. More granular profiling (sub-component, per-layer, prefill vs decode breakdown)
2. Run profiling on Raspberry Pi (fetch `profiling/` folder independently)
3. Debug PALU/PACT 0.0 accuracy
4. Write paper
