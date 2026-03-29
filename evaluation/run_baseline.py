"""
evaluation/run_baseline.py
===========================
Baseline evaluation pipeline — full precision (fp16) inference.

Benchmarks:
  • VQAv2   — validation subset (5 000 samples)
  • TextVQA — validation set
  • POPE    — adversarial split

KPIs logged per model:
  accuracy, avg latency (s/sample), peak GPU memory (MB),
  throughput (samples/s), num_params

Usage:
  python evaluation/run_baseline.py --model_id HuggingFaceTB/SmolVLM-256M-Instruct
  python evaluation/run_baseline.py --model_id microsoft/Florence-2-base --batch_size 4
"""

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# ── project imports ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.model_loader import load_model, unload_model, detect_family
from profiling.gpu_profiler import GPUProfiler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "baseline"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset loaders ─────────────────────────────────────────────────────────

def load_vqav2(n_samples: int = 5000):
    from datasets import load_dataset
    logger.info(f"Loading VQAv2 validation (n={n_samples})...")
    ds = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
    samples = []
    for item in ds:
        samples.append({
            "image":    item["image"],
            "question": item["question"],
            "answers":  [a["answer"] for a in item["answers"]],
        })
        if len(samples) >= n_samples:
            break
    logger.info(f"  Loaded {len(samples)} VQAv2 samples.")
    return samples


def load_textvqa(n_samples: int = 1000):
    from datasets import load_dataset
    logger.info("Loading TextVQA validation...")
    ds = load_dataset("lmms-lab/textvqa", split="validation")
    samples = [
        {
            "image":    item["image"],
            "question": item["question"],
            "answers":  item["answers"],
        }
        for item in ds
    ]
    if n_samples and n_samples < len(samples):
        samples = samples[:n_samples]
    logger.info(f"  Loaded {len(samples)} TextVQA samples.")
    return samples


def load_pope(n_samples: int = 1000):
    from datasets import load_dataset
    logger.info("Loading POPE adversarial...")
    ds = load_dataset("lmms-lab/POPE", split="test")
    samples = []
    for item in ds:
        if item.get("category", "") == "adversarial":
            samples.append({
                "image":    item["image"],
                "question": item["question"],
                "answers":  [item["answer"]],
            })
    if not samples:                  # fallback: use all
        samples = [
            {"image": item["image"], "question": item["question"], "answers": [item["answer"]]}
            for item in ds
        ]
    if n_samples and n_samples < len(samples):
        samples = samples[:n_samples]
    logger.info(f"  Loaded {len(samples)} POPE samples.")
    return samples


# ── Inference helpers ────────────────────────────────────────────────────────

def _vqa_accuracy(pred: str, gold_answers: list[str]) -> float:
    """VQA soft accuracy: min(count/3, 1)."""
    pred_clean = pred.strip().lower().rstrip(".")
    count = sum(1 for a in gold_answers if a.strip().lower() == pred_clean)
    return min(count / 3.0, 1.0)


def _pope_accuracy(pred: str, gold_answers: list[str]) -> float:
    # Use first-word exact match (POPE standard): substring "yes" in pred_clean
    # is too permissive — verbose outputs like "I cannot say yes" would be
    # mis-scored, causing inconsistency across quantization levels.
    tokens = pred.strip().lower().split()
    first_word = tokens[0].rstrip(".,!?") if tokens else ""
    pred_yes = first_word == "yes"
    gold_yes = gold_answers[0].strip().lower() == "yes"
    return 1.0 if pred_yes == gold_yes else 0.0


def run_inference(model, processor, sample: dict, family: str,
                  device: str, max_new_tokens: int = 30) -> str:
    """Run single-sample inference and return generated text."""
    image    = sample["image"]
    # Florence-2 uses dedicated task tokens (<VQA>) and ignores free-text
    # instructions; all other families get a short-answer nudge to avoid
    # verbose sentences that fail exact-match VQA scoring.
    if family == "florence2":
        question = sample["question"]
    else:
        question = sample["question"] + " Answer with a single word or short phrase."

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    if family == "florence2":
        inputs = processor(
            text=f"<VQA> {question}",
            images=image,
            return_tensors="pt",
        ).to(device)
        # Florence-2 vision tower prefers fp16 on GPU; use model dtype on CPU
        target_dtype = torch.float16 if torch.cuda.is_available() else next(model.parameters()).dtype
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=target_dtype)
        with torch.no_grad():
            ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                num_beams=1,
            )
        pred = processor.batch_decode(ids, skip_special_tokens=True)[0]

    elif family in ("smolvlm", "nanovlm", "fastvlm"):
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": question}],
        }]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        pred = processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True)[0]

    elif family == "lfm2vl":
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image},
                        {"type": "text",  "text": question}],
        }]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text, images=[image], return_tensors="pt").to(device)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        pred = processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True)[0]

    elif family == "moondream":
        enc_image = model.encode_image(image)
        pred = model.answer_question(enc_image, question, processor)

    elif family == "qwen25vl":
        from transformers import AutoProcessor as _AP
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": question},
            ],
        }]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image],
                           return_tensors="pt").to(device)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        pred = processor.batch_decode(
            ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]

    elif family == "gemma3":
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": question},
            ],
        }]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
        ).to(device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        pred = processor.decode(ids[0][input_len:], skip_special_tokens=True)

    elif family == "ovis2":
        text_tokenizer = processor  # loader stores text_tokenizer as processor
        # Ovis2 requires <image>\n prefix so the visual tokenizer processes the image
        ovis_query = f"<image>\n{question}"
        prompt, input_ids, pixel_values = model.preprocess_inputs(ovis_query, [image])
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
        input_ids     = input_ids.unsqueeze(0).to(device=device)
        attention_mask = attention_mask.unsqueeze(0).to(device=device)
        pixel_values  = [pixel_values.to(dtype=next(model.parameters()).dtype, device=device)]
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=text_tokenizer.pad_token_id,
        )
        with torch.no_grad():
            ids = model.generate(input_ids, pixel_values=pixel_values,
                                 attention_mask=attention_mask, **gen_kwargs)
        pred = text_tokenizer.decode(ids[0], skip_special_tokens=True)

    elif family == "internvl25":
        tokenizer = processor  # internvl25 returns tokenizer as processor
        pixel_values = _internvl_preprocess(image).to(device=device, dtype=torch.float16)
        question_fmt = f"<image>\n{question}"
        pred = model.chat(tokenizer, pixel_values, question_fmt,
                          dict(max_new_tokens=max_new_tokens, do_sample=False))

    else:
        raise ValueError(f"Unknown family '{family}'")

    return pred.strip()


def _internvl_preprocess(image: Image.Image):
    """Simple preprocessing for InternVL2.5 — resize + normalize."""
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((448, 448)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image.convert("RGB")).unsqueeze(0)


# ── Evaluation loop ──────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation/whitespace for metric computation."""
    return text.strip().lower().rstrip(".")


def _token_f1(pred: str, gold: str) -> float:
    """Token-level F1 between prediction and gold answer."""
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _bleu_single(pred: str, gold: str) -> float:
    """Smoothed BLEU-1/2/3/4 geometric mean for a single pair.

    Uses add-1 (Laplace) smoothing for n-gram orders where the prediction
    is too short to have any n-grams, so single-word VQA answers don't
    collapse to 0.
    """
    import math
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0
    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(gold_tokens) / len(pred_tokens))) if pred_tokens else 0.0
    # N-gram precisions (1-4) with add-1 smoothing
    log_prec = 0.0
    for n in range(1, 5):
        pred_ngrams = [tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens) - n + 1)]
        gold_ngrams = [tuple(gold_tokens[i:i+n]) for i in range(len(gold_tokens) - n + 1)]
        if not pred_ngrams:
            # Smoothing: treat as 1/(1+1) = 0.5 for missing n-gram orders
            log_prec += math.log(1.0 / (1.0 + 1.0)) / 4.0
            continue
        gold_counts: dict = {}
        for ng in gold_ngrams:
            gold_counts[ng] = gold_counts.get(ng, 0) + 1
        clipped = 0
        for ng in pred_ngrams:
            if gold_counts.get(ng, 0) > 0:
                clipped += 1
                gold_counts[ng] -= 1
        # Add-1 smoothing: (clipped + 1) / (len + 1) when clipped == 0
        if clipped == 0:
            prec = 1.0 / (len(pred_ngrams) + 1.0)
        else:
            prec = clipped / len(pred_ngrams)
        log_prec += math.log(prec) / 4.0
    return bp * math.exp(log_prec)


def _rouge_l(pred: str, gold: str) -> float:
    """ROUGE-L F1 based on longest common subsequence."""
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0
    m, n = len(pred_tokens), len(gold_tokens)
    # LCS via DP
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if pred_tokens[i-1] == gold_tokens[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(curr[j-1], prev[j])
        prev = curr
    lcs_len = prev[n]
    if lcs_len == 0:
        return 0.0
    precision = lcs_len / m
    recall = lcs_len / n
    return 2 * precision * recall / (precision + recall)


def _best_gold(gold_answers: list[str]) -> str:
    """Pick the most frequent gold answer (majority vote)."""
    counts: dict = {}
    for a in gold_answers:
        key = _normalize(a)
        counts[key] = counts.get(key, 0) + 1
    return max(counts, key=counts.get) if counts else ""


def _compute_metrics(predictions: list[tuple[str, list[str]]]) -> dict:
    """Compute multi-metric scores over all (pred, gold_answers) pairs."""
    exact_matches, contains_scores = [], []
    f1_scores, bleu_scores, rouge_scores = [], [], []

    for pred, gold_answers in predictions:
        pred_norm = _normalize(pred)
        gold = _best_gold(gold_answers)

        # Exact match
        exact_matches.append(1.0 if pred_norm == gold else 0.0)

        # Contains: pred contains gold OR gold contains pred
        contains_scores.append(
            1.0 if (gold in pred_norm or pred_norm in gold) else 0.0
        )

        # Token F1
        f1_scores.append(_token_f1(pred, gold))

        # BLEU
        bleu_scores.append(_bleu_single(pred, gold))

        # ROUGE-L
        rouge_scores.append(_rouge_l(pred, gold))

    n = len(predictions) or 1
    return {
        "exact_match": round(sum(exact_matches) / n, 4),
        "contains":    round(sum(contains_scores) / n, 4),
        "token_f1":    round(sum(f1_scores) / n, 4),
        "bleu":        round(sum(bleu_scores) / n, 4),
        "rouge_l":     round(sum(rouge_scores) / n, 4),
    }


def evaluate_dataset(model, processor, samples: list, family: str,
                     device: str, dataset_name: str,
                     accuracy_fn, batch_size: int = 1) -> dict:
    logger.info(f"  Evaluating on {dataset_name} ({len(samples)} samples)...")
    scores, latencies = [], []
    predictions = []  # (pred_text, gold_answers) for multi-metric
    profiler = GPUProfiler(device_index=0)

    skipped = 0
    with profiler:
        for sample in tqdm(samples, desc=dataset_name, leave=False):
            t0 = time.perf_counter()
            try:
                pred = run_inference(model, processor, sample, family, device)
            except Exception as e:
                logger.warning(f"  Skipping sample due to error: {traceback.format_exc()}")
                skipped += 1
                # Clear GPU cache to prevent memory buildup from failed samples
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            latencies.append(time.perf_counter() - t0)
            scores.append(accuracy_fn(pred, sample["answers"]))
            predictions.append((pred, sample["answers"]))
    if skipped:
        logger.warning(f"  {dataset_name}: skipped {skipped} samples due to errors")

    stats = profiler.stats()
    n_evaluated = len(scores)
    avg_acc   = sum(scores) / n_evaluated if n_evaluated else 0.0
    avg_lat   = sum(latencies) / n_evaluated if n_evaluated else 0.0
    throughput = n_evaluated / stats.wall_time_s if stats.wall_time_s > 0 else 0.0

    # Compute multi-metric scores
    metrics = _compute_metrics(predictions) if predictions else {
        "exact_match": 0.0, "contains": 0.0, "token_f1": 0.0,
        "bleu": 0.0, "rouge_l": 0.0,
    }

    logger.info(
        f"  {dataset_name}: acc={avg_acc:.4f}  "
        f"lat={avg_lat:.3f}s  mem={stats.peak_memory_mb:.0f}MB  "
        f"tput={throughput:.2f} samp/s"
    )
    logger.info(
        f"  {dataset_name} metrics: exact_match={metrics['exact_match']:.4f}  "
        f"contains={metrics['contains']:.4f}  f1={metrics['token_f1']:.4f}  "
        f"bleu={metrics['bleu']:.4f}  rouge_l={metrics['rouge_l']:.4f}"
    )
    return {
        "accuracy":        round(avg_acc,   4),
        "avg_latency_s":   round(avg_lat,   4),
        "peak_memory_mb":  round(stats.peak_memory_mb, 1),
        "avg_memory_mb":   round(stats.avg_memory_mb,  1),
        "throughput_sps":  round(throughput, 3),
        "avg_power_w":     round(stats.avg_power_w,    1),
        "avg_gpu_util_pct":round(stats.avg_gpu_util_pct, 1),
        "n_samples":       len(samples),
        "n_evaluated":     n_evaluated,
        "n_skipped":       skipped,
        "all_failed":      n_evaluated == 0,
        "zero_accuracy_warning": n_evaluated > 0 and avg_acc == 0.0,
        "metrics":         metrics,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Baseline VLM evaluation")
    parser.add_argument("--model_id",    required=True,  help="HuggingFace model ID")
    parser.add_argument("--quant",       default="fp16",
                        choices=["fp16", "int8", "int4"],
                        help="Precision for loading (default: fp16). Use int8/int4 for large models.")
    parser.add_argument("--batch_size",  type=int, default=1)
    parser.add_argument("--vqav2_n",     type=int, default=1000,
                        help="Number of VQAv2 samples (default 1000)")
    parser.add_argument("--textvqa_n",   type=int, default=1000,
                        help="Number of TextVQA samples (default 1000)")
    parser.add_argument("--pope_n",      type=int, default=1000,
                        help="Number of POPE samples (default 1000)")
    parser.add_argument("--skip_vqav2",  action="store_true")
    parser.add_argument("--skip_textvqa",action="store_true")
    parser.add_argument("--skip_pope",   action="store_true")
    parser.add_argument("--force",       action="store_true",
                        help="Overwrite existing result even if it already exists")
    args = parser.parse_args()

    model_id  = args.model_id
    safe_name = model_id.replace("/", "__")
    out_path  = RESULTS_DIR / f"{safe_name}.json"

    # ── Resumability ─────────────────────────────────────────────────────
    if out_path.exists() and not args.force:
        logger.info(f"Result already exists at {out_path}. Skipping.")
        return

    # ── Load model ───────────────────────────────────────────────────────
    logger.info(f"Loading model: {model_id}  quant={args.quant}")
    model, processor, meta = load_model(model_id, quant=args.quant)
    family = meta.family

    # device detection: use full device string (e.g. "cuda:0") so that
    # inputs.to(device) lands on exactly the same device as model weights.
    try:
        device = str(next(model.parameters()).device)
    except StopIteration:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Params: {num_params/1e6:.1f}M  |  VRAM delta: {meta.gpu_mem_delta_mb:.0f} MB")

    # Preserve any benchmark results that are being skipped this run
    existing_benchmarks: dict = {}
    if out_path.exists() and args.force:
        with open(out_path) as f:
            existing_benchmarks = json.load(f).get("benchmarks", {})

    results: dict = {
        "model_id":         model_id,
        "family":           family,
        "quant":            args.quant,
        "num_params_M":     round(num_params / 1e6, 1),
        "gpu_mem_load_mb":  meta.gpu_mem_delta_mb,
        "benchmarks":       dict(existing_benchmarks),
    }

    # ── VQAv2 ────────────────────────────────────────────────────────────
    if not args.skip_vqav2:
        samples = load_vqav2(n_samples=args.vqav2_n)
        results["benchmarks"]["vqav2"] = evaluate_dataset(
            model, processor, samples, family, device,
            "VQAv2", _vqa_accuracy, args.batch_size,
        )

    # ── TextVQA ──────────────────────────────────────────────────────────
    if not args.skip_textvqa:
        samples = load_textvqa(n_samples=args.textvqa_n)
        results["benchmarks"]["textvqa"] = evaluate_dataset(
            model, processor, samples, family, device,
            "TextVQA", _vqa_accuracy, args.batch_size,
        )

    # ── POPE ─────────────────────────────────────────────────────────────
    if not args.skip_pope:
        samples = load_pope(n_samples=args.pope_n)
        results["benchmarks"]["pope"] = evaluate_dataset(
            model, processor, samples, family, device,
            "POPE", _pope_accuracy, args.batch_size,
        )

    # ── Save ─────────────────────────────────────────────────────────────
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    unload_model(model)


if __name__ == "__main__":
    main()
