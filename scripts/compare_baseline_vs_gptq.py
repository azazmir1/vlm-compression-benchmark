"""
Quick comparison: baseline (FP16) vs GPTQ-INT4 on 10 VQAv2 samples.
"""

import sys
import re
import string
import json
import time
import gc
from math import exp, log
from collections import Counter

import torch
from pathlib import Path
from PIL import Image
from datasets import load_dataset

# ── Metrics (inlined to avoid broken import chain) ──

def _normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def _vqa_accuracy(pred, gold_answers):
    pred_clean = pred.strip().lower().rstrip(".")
    count = sum(1 for a in gold_answers if a.strip().lower() == pred_clean)
    return min(count / 3.0, 1.0)

def _contains_accuracy(pred, gold_answers):
    pred_norm = _normalize_text(pred)
    if not pred_norm: return 0.0
    for gold in gold_answers:
        gold_norm = _normalize_text(gold)
        if gold_norm and gold_norm in pred_norm: return 1.0
    return 0.0

def _token_f1(pred, gold_answers):
    pred_tokens = _normalize_text(pred).split()
    if not pred_tokens: return 0.0
    best_f1 = 0.0
    for gold in gold_answers:
        gold_tokens = _normalize_text(gold).split()
        if not gold_tokens: continue
        common = set(pred_tokens) & set(gold_tokens)
        if not common: continue
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        best_f1 = max(best_f1, 2 * precision * recall / (precision + recall))
    return best_f1

def _bleu_score(pred, gold_answers, max_n=4):
    pred_tokens = _normalize_text(pred).split()
    if not pred_tokens: return 0.0
    best_bleu = 0.0
    for gold in gold_answers:
        gold_tokens = _normalize_text(gold).split()
        if not gold_tokens: continue
        log_precisions = []
        for n in range(1, min(max_n, len(pred_tokens), len(gold_tokens)) + 1):
            pred_ng = Counter(tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens)-n+1))
            gold_ng = Counter(tuple(gold_tokens[i:i+n]) for i in range(len(gold_tokens)-n+1))
            clipped = sum(min(pred_ng[ng], gold_ng.get(ng, 0)) for ng in pred_ng)
            total = max(sum(pred_ng.values()), 1)
            if n == 1:
                if clipped == 0: break
                precision = clipped / total
            else:
                precision = (clipped + 1) / (total + 1)
            log_precisions.append(log(precision))
        if not log_precisions: continue
        avg_log_prec = sum(log_precisions) / len(log_precisions)
        bp = min(1.0, exp(1 - len(gold_tokens)/len(pred_tokens))) if len(pred_tokens) < len(gold_tokens) else 1.0
        best_bleu = max(best_bleu, bp * exp(avg_log_prec))
    return best_bleu

def _lcs_length(x, y):
    m, n = len(x), len(y)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            curr[j] = prev[j-1] + 1 if x[i-1] == y[j-1] else max(prev[j], curr[j-1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]

def _rouge_l(pred, gold_answers):
    pred_tokens = _normalize_text(pred).split()
    if not pred_tokens: return 0.0
    best_f1 = 0.0
    for gold in gold_answers:
        gold_tokens = _normalize_text(gold).split()
        if not gold_tokens: continue
        lcs = _lcs_length(pred_tokens, gold_tokens)
        if lcs == 0: continue
        precision = lcs / len(pred_tokens)
        recall = lcs / len(gold_tokens)
        best_f1 = max(best_f1, 2 * precision * recall / (precision + recall))
    return best_f1

def _vqa_multi_metric(pred, gold_answers):
    return {
        "exact_match": _vqa_accuracy(pred, gold_answers),
        "contains": _contains_accuracy(pred, gold_answers),
        "token_f1": _token_f1(pred, gold_answers),
        "bleu": _bleu_score(pred, gold_answers),
        "rouge_l": _rouge_l(pred, gold_answers),
    }

# ── Qwen2.5-VL inference ──

def qwen_inference(model, processor, sample, device, max_new_tokens=30):
    image = sample["image"]
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    question = sample["question"] + " Answer with a single word or short phrase."
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    pred = processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return pred.strip()

# ── Config ──
ORIGINAL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
GPTQ_ID = "Azaz666/Qwen2.5-VL-7B-Instruct-GPTQ-Int4"
N_SAMPLES = 10
DEVICE = "cuda:0"

def load_vqav2_samples(n):
    print(f"Loading {n} VQAv2 samples...")
    ds = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
    samples = []
    for item in ds:
        samples.append({
            "image": item["image"],
            "question": item["question"],
            "answers": [a["answer"] for a in item["answers"]],
        })
        if len(samples) >= n:
            break
    return samples


def load_original_model():
    print(f"\n{'='*60}")
    print(f"Loading ORIGINAL model: {ORIGINAL_ID}")
    print(f"{'='*60}")
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(ORIGINAL_ID, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        ORIGINAL_ID,
        torch_dtype=torch.float16,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def load_gptq_model():
    print(f"\n{'='*60}")
    print(f"Loading GPTQ model: {GPTQ_ID}")
    print(f"{'='*60}")
    from gptqmodel import GPTQModel
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(GPTQ_ID, trust_remote_code=True)
    model = GPTQModel.load(GPTQ_ID, device_map="auto", trust_remote_code=True)
    model.eval()
    return model, processor


def run_eval(model, processor, samples):
    results = []
    total_time = 0
    for i, s in enumerate(samples):
        t0 = time.time()
        try:
            pred = qwen_inference(model, processor, s, DEVICE, max_new_tokens=30)
        except Exception as e:
            pred = f"ERROR: {e}"
        elapsed = time.time() - t0
        total_time += elapsed

        metrics = _vqa_multi_metric(pred, s["answers"])
        results.append({
            "idx": i,
            "question": s["question"],
            "gold": s["answers"][:3],
            "pred": pred,
            "metrics": metrics,
            "latency_s": round(elapsed, 3),
        })
        print(f"  [{i+1}/{len(samples)}] Q: {s['question'][:50]}...")
        print(f"           Gold: {s['answers'][:3]}  |  Pred: {pred}")
        print(f"           Metrics: {metrics}  |  Time: {elapsed:.2f}s")

    # Aggregate
    n = len(results)
    avg = {}
    for key in ["exact_match", "contains", "token_f1", "bleu", "rouge_l"]:
        avg[key] = round(sum(r["metrics"][key] for r in results) / n, 4)
    avg["avg_latency_s"] = round(total_time / n, 3)
    return results, avg


def main():
    samples = load_vqav2_samples(N_SAMPLES)

    # ── Baseline (FP16) ──
    model_orig, proc_orig = load_original_model()
    print(f"\n--- Running Baseline (FP16) on {N_SAMPLES} samples ---")
    results_orig, avg_orig = run_eval(model_orig, proc_orig, samples)

    # Free memory
    del model_orig, proc_orig
    gc.collect(); torch.cuda.empty_cache()

    # ── GPTQ INT4 ──
    model_gptq, proc_gptq = load_gptq_model()
    print(f"\n--- Running GPTQ INT4 on {N_SAMPLES} samples ---")
    results_gptq, avg_gptq = run_eval(model_gptq, proc_gptq, samples)

    del model_gptq, proc_gptq
    gc.collect(); torch.cuda.empty_cache()

    # ── Print comparison ──
    print(f"\n{'='*70}")
    print(f"COMPARISON: {ORIGINAL_ID}")
    print(f"{'='*70}")
    print(f"{'Metric':<20} {'Baseline (FP16)':>18} {'GPTQ INT4':>18} {'Delta':>12}")
    print(f"{'-'*70}")
    for key in ["exact_match", "contains", "token_f1", "bleu", "rouge_l", "avg_latency_s"]:
        v_orig = avg_orig[key]
        v_gptq = avg_gptq[key]
        delta = v_gptq - v_orig
        sign = "+" if delta >= 0 else ""
        print(f"{key:<20} {v_orig:>18.4f} {v_gptq:>18.4f} {sign}{delta:>11.4f}")

    # Save results
    out = {
        "model": ORIGINAL_ID,
        "gptq_model": GPTQ_ID,
        "n_samples": N_SAMPLES,
        "baseline_avg": avg_orig,
        "gptq_avg": avg_gptq,
        "baseline_details": results_orig,
        "gptq_details": results_gptq,
    }
    out_path = Path("results/gptq_comparison_qwen7b.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
