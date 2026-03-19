# VLM Compression Benchmark

Benchmarking Vision-Language Models (VLMs) under different compression methods
(Post-Training Quantization, Structured Pruning, QLoRA) on lab GPUs,
with results informing edge deployment on **NVIDIA Jetson Orin Nano** and **Raspberry Pi 5**.

---

## Model Families

### RPi5-Compatible Track — sub-500M variants available

| Family | Org | Variants (params) | Architecture | HF IDs |
|---|---|---|---|---|
| Florence-2 | Microsoft | 232M, 771M | Encoder-Decoder (DaViT+BART) | `microsoft/Florence-2-base`, `microsoft/Florence-2-large` |
| SmolVLM | HuggingFace | 256M, 500M, 2.2B | Encoder-Decoder (SigLIP+SmolLM2) | `HuggingFaceTB/SmolVLM-256M-Instruct`, `SmolVLM-500M-Instruct`, `SmolVLM-Instruct` |
| nanoVLM | HuggingFace | 222M, 450M | Decoder-only (SigLIP+SmolLM2) | `lusxvr/nanoVLM-222M`, `lusxvr/nanoVLM-450M` |
| LFM2-VL | Liquid AI | 450M, 1.6B, 3B | Hybrid Recurrent (SigLIP2+LFM2) | `LiquidAI/LFM2-VL-450M`, `LFM2-VL-1.6B`, `LFM2-VL-3B` |
| Moondream | Moondream AI | 500M⚠, 2B | Decoder-only (SigLIP+custom) | `vikhyatk/moondream2` (revision-based) |
| FastVLM | Apple | ~500M⚠, 1.5B, 7B | Decoder-only (FastViT-HD+Qwen2) | `apple/FastVLM-0.5B`, `apple/FastVLM-1.5B`, `apple/FastVLM-7B` |

⚠ = borderline at exactly 500M

### GPU / Jetson-Only Track — popular, no sub-500M variant

| Family | Org | Variants | Architecture | HF IDs |
|---|---|---|---|---|
| Qwen2.5-VL | Alibaba | 3B, 7B, 32B, 72B | Decoder-only | `Qwen/Qwen2.5-VL-3B-Instruct`, `7B`, `32B`, `72B` |
| InternVL2.5 | OpenGVLab | 1B, 2B, 4B, 8B, 26B+ | Encoder-Decoder | `OpenGVLab/InternVL2_5-1B`, `2B`, `4B`, `8B` |

---

## Project Structure

```
vlm-compression-benchmark/
├── models/
│   └── model_loader.py          # Unified loader for all 8 families
├── compression/
│   ├── ptq/run_ptq.py           # INT8/INT4 via BitsAndBytes or AutoAWQ
│   ├── pruning/run_pruning.py   # Magnitude pruning at 20%/40% sparsity
│   └── qlora/run_qlora.py       # QLoRA fine-tuning at rank 16/64
├── evaluation/
│   └── run_baseline.py          # FP16 baseline eval on VQAv2/TextVQA/POPE
├── profiling/
│   └── gpu_profiler.py          # pynvml context-manager profiler
├── results/
│   ├── baseline/                # {model_name}.json per model
│   ├── ptq/
│   ├── pruning/
│   └── qlora/
├── scripts/
│   ├── run_all_baselines.sh
│   ├── run_all_ptq.sh
│   ├── run_all_pruning.sh
│   └── run_all_qlora.sh
├── configs/                     # YAML config per family
│   ├── florence2.yaml
│   ├── smolvlm.yaml
│   ├── nanovlm.yaml
│   ├── lfm2vl.yaml
│   ├── moondream.yaml
│   ├── fastvlm.yaml
│   ├── qwen25vl.yaml
│   └── internvl25.yaml
├── notebooks/
│   └── results_analysis.ipynb   # Aggregation, plots, deployability flags
├── requirements.txt
├── setup.sh
└── README.md
```

---

## Setup

```bash
# Create conda environment and install all dependencies
bash setup.sh

# Activate
conda activate vlm-bench
```

---

## Evaluation Benchmarks

| Benchmark | Split | Samples | Metric |
|---|---|---|---|
| VQAv2 | validation | 5 000 | VQA soft accuracy |
| TextVQA | validation | full | VQA soft accuracy |
| POPE | adversarial | full | Binary yes/no accuracy |

---

## Running Experiments

### Baselines (FP16)

```bash
# Single model
python evaluation/run_baseline.py --model_id HuggingFaceTB/SmolVLM-256M-Instruct

# All models (resumable — skips if result already exists)
bash scripts/run_all_baselines.sh

# Reduce VQAv2 subset for quick testing
VQAV2_N=500 bash scripts/run_all_baselines.sh
```

### PTQ (INT8 / INT4)

```bash
# Single model — INT8 via BitsAndBytes
python compression/ptq/run_ptq.py \
    --model_id HuggingFaceTB/SmolVLM-256M-Instruct \
    --quant int8 --backend bnb

# Single model — INT4 via AutoAWQ
python compression/ptq/run_ptq.py \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --quant int4 --backend awq

# All models
bash scripts/run_all_ptq.sh
```

### Structured Pruning (20% / 40% sparsity)

```bash
# Single model
python compression/pruning/run_pruning.py \
    --model_id microsoft/Florence-2-base \
    --sparsity 0.20

# All models
bash scripts/run_all_pruning.sh
```

### QLoRA Fine-tuning (rank 16 / 64)

```bash
# Single model — rank 16, 1 epoch
python compression/qlora/run_qlora.py \
    --model_id HuggingFaceTB/SmolVLM-256M-Instruct \
    --rank 16 --epochs 1

# rank 64, 2 epochs
python compression/qlora/run_qlora.py \
    --model_id Qwen/Qwen2.5-VL-3B-Instruct \
    --rank 64 --epochs 2

# All models
bash scripts/run_all_qlora.sh
```

---

## Results & Analysis

All results are saved as JSON files under `results/{method}/{model_name}.json`.

Open the notebook for aggregation and visualisation:

```bash
jupyter notebook notebooks/results_analysis.ipynb
```

The notebook produces:
- **`results/master_table.csv`** — full comparison table
- **`results/accuracy_vs_method.png`** — per-family accuracy bar charts
- **`results/memory_vs_params.png`** — memory vs param count scatter
- **`results/latency_vs_accuracy.png`** — latency-accuracy tradeoff curves
- **`results/accuracy_drop.png`** — accuracy drop vs FP16 baseline

### Deployability Threshold (Raspberry Pi 5)

A model/compression combo is flagged **RPi5-deployable** if:
- Peak memory < **4 GB**
- Avg latency per sample < **10 s**

---

## KPIs Tracked

| KPI | Unit | Notes |
|---|---|---|
| Accuracy | 0–1 | VQA soft / binary for POPE |
| Avg latency | s/sample | Wall-clock per inference |
| Peak GPU memory | MB | Via pynvml |
| Throughput | samples/s | |
| Avg power draw | W | Via pynvml |
| GPU utilisation | % | |
| Compression ratio | × | vs FP16 memory footprint |
| Sparsity | 0–1 | Pruning only |
| LoRA rank | — | QLoRA only |

---

## Hardware Targets

| Device | GPU | RAM | Deployment Track |
|---|---|---|---|
| Lab GPU | A100 / RTX-class | 40–80 GB VRAM | All models, all methods |
| Jetson Orin Nano | 1024-core Ampere GPU | 8 GB unified | GPU/Jetson track |
| Raspberry Pi 5 | None (CPU only) | ~8 GB RAM | Sub-500M models, INT4 only |
