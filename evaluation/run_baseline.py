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


# ── Accuracy metrics ─────────────────────────────────────────────────────────
#
# We compute multiple metrics per sample so that compression quality is
# visible even when exact-match drops to zero:
#
#   exact_match   — standard VQAv2 soft accuracy (strict)
#   contains      — does any gold answer appear inside the prediction?
#   token_f1      — token-level F1 between prediction and best gold answer
#   bleu          — BLEU score (n-gram precision with brevity penalty)
#   rouge_l       — ROUGE-L (longest common subsequence F1)

import re
import string
from collections import Counter
from math import exp, log


def _normalize_text(s: str) -> str:
    """Lower-case, strip punctuation/articles, collapse whitespace."""
    s = s.lower().strip()
    # remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


# ── 1. Exact Match (standard VQAv2) ─────────────────────────────────────────

def _vqa_accuracy(pred: str, gold_answers: list[str]) -> float:
    """VQA soft accuracy: min(count/3, 1)."""
    pred_clean = pred.strip().lower().rstrip(".")
    count = sum(1 for a in gold_answers if a.strip().lower() == pred_clean)
    return min(count / 3.0, 1.0)


# ── 2. Contains Match ────────────────────────────────────────────────────────

def _contains_accuracy(pred: str, gold_answers: list[str]) -> float:
    """1.0 if any gold answer appears as a substring in the prediction.

    Catches verbose predictions like "The answer is red" when gold is "red".
    Uses normalized text to ignore punctuation/articles.
    """
    pred_norm = _normalize_text(pred)
    if not pred_norm:
        return 0.0
    for gold in gold_answers:
        gold_norm = _normalize_text(gold)
        if gold_norm and gold_norm in pred_norm:
            return 1.0
    return 0.0


# ── 3. Token F1 ──────────────────────────────────────────────────────────────

def _token_f1(pred: str, gold_answers: list[str]) -> float:
    """Token-level F1 score (best match against any gold answer).

    Standard SQuAD-style: treats prediction and gold as bags of tokens,
    computes precision (how many pred tokens are in gold) and recall
    (how many gold tokens are in pred), returns their harmonic mean.
    """
    pred_tokens = _normalize_text(pred).split()
    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for gold in gold_answers:
        gold_tokens = _normalize_text(gold).split()
        if not gold_tokens:
            continue
        common = set(pred_tokens) & set(gold_tokens)
        if not common:
            continue
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1


# ── 4. BLEU Score ────────────────────────────────────────────────────────────

def _get_ngrams(tokens: list[str], n: int) -> Counter:
    """Extract n-gram counts from a token list."""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _bleu_score(pred: str, gold_answers: list[str],
                max_n: int = 4) -> float:
    """Sentence-level BLEU (up to max_n-grams) against best gold answer.

    Pure Python implementation — no dependencies. For unigram-only
    matches (common in VQA), uses exact precision without smoothing
    so that zero overlap = zero BLEU. Smoothing is only applied for
    higher-order n-grams to avoid log(0).
    """
    pred_tokens = _normalize_text(pred).split()
    if not pred_tokens:
        return 0.0

    best_bleu = 0.0
    for gold in gold_answers:
        gold_tokens = _normalize_text(gold).split()
        if not gold_tokens:
            continue

        # Clipped n-gram precisions
        log_precisions = []
        for n in range(1, min(max_n, len(pred_tokens), len(gold_tokens)) + 1):
            pred_ngrams = _get_ngrams(pred_tokens, n)
            gold_ngrams = _get_ngrams(gold_tokens, n)
            clipped = sum(min(pred_ngrams[ng], gold_ngrams.get(ng, 0))
                          for ng in pred_ngrams)
            total = max(sum(pred_ngrams.values()), 1)
            if n == 1:
                # No smoothing for unigrams: zero overlap = zero BLEU
                if clipped == 0:
                    break
                precision = clipped / total
            else:
                # Add-1 smoothing for higher n-grams (Chen & Cherry, 2014)
                precision = (clipped + 1) / (total + 1)
            log_precisions.append(log(precision))

        if not log_precisions:
            continue

        # Geometric mean of precisions
        avg_log_prec = sum(log_precisions) / len(log_precisions)

        # Brevity penalty
        bp = min(1.0, exp(1 - len(gold_tokens) / len(pred_tokens))) \
            if len(pred_tokens) < len(gold_tokens) else 1.0

        bleu = bp * exp(avg_log_prec)
        best_bleu = max(best_bleu, bleu)

    return best_bleu


# ── 5. ROUGE-L ───────────────────────────────────────────────────────────────

def _lcs_length(x: list[str], y: list[str]) -> int:
    """Length of the longest common subsequence between two token lists."""
    m, n = len(x), len(y)
    # Space-optimized: only keep two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def _rouge_l(pred: str, gold_answers: list[str]) -> float:
    """ROUGE-L F1 score (longest common subsequence) against best gold answer.

    Captures word-order similarity — a prediction that contains the gold
    answer's words in roughly the right order scores high even if it has
    extra words.
    """
    pred_tokens = _normalize_text(pred).split()
    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for gold in gold_answers:
        gold_tokens = _normalize_text(gold).split()
        if not gold_tokens:
            continue

        lcs = _lcs_length(pred_tokens, gold_tokens)
        if lcs == 0:
            continue

        precision = lcs / len(pred_tokens)
        recall = lcs / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1


# ── Multi-metric scorer ──────────────────────────────────────────────────────

def _vqa_multi_metric(pred: str, gold_answers: list[str]) -> dict:
    """Compute all accuracy metrics for one VQA sample."""
    return {
        "exact_match": _vqa_accuracy(pred, gold_answers),
        "contains":    _contains_accuracy(pred, gold_answers),
        "token_f1":    _token_f1(pred, gold_answers),
        "bleu":        _bleu_score(pred, gold_answers),
        "rouge_l":     _rouge_l(pred, gold_answers),
    }


# ── POPE (unchanged — binary yes/no, exact match is appropriate) ─────────────

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
        # Florence-2 custom vision tower expects fp16; cast pixel_values explicitly
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)
        with torch.no_grad():
            ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                num_beams=1,
            )
        pred = processor.batch_decode(ids, skip_special_tokens=True)[0]

    elif family == "nanovlm":
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": question}],
        }]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)
        # Cast image tensors to model dtype (ToTensor produces float32)
        model_dtype = next(model.parameters()).dtype
        images_list = inputs.get("images")
        if images_list is not None:
            images_list = [t.to(dtype=model_dtype) for t in images_list]
        with torch.no_grad():
            ids = model.generate(
                input_ids=inputs["input_ids"],
                images=images_list,
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                greedy=True,
            )
        pred = processor.batch_decode(ids, skip_special_tokens=True)[0]

    elif family == "smolvlm":
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

    elif family == "fastvlm":
        # FastVLM (LlavaQwen2) needs the special IMAGE_TOKEN_INDEX = -200
        # inserted into input_ids so the model knows where to inject vision
        # features.  Follow the same pattern as the working profiling framework.
        IMAGE_TOKEN_INDEX = -200
        messages = [{"role": "user", "content": f"<image>\n{question}"}]
        rendered = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        # Split at <image> placeholder and tokenize each segment
        pre, post = rendered.split("<image>", 1)
        pre_ids = processor.tokenizer(
            pre, return_tensors="pt", add_special_tokens=False,
        ).input_ids
        post_ids = processor.tokenizer(
            post, return_tensors="pt", add_special_tokens=False,
        ).input_ids
        img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
        input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(device)
        attention_mask = torch.ones_like(input_ids, device=device)
        # Get pixel values from the vision tower's image processor
        image_rgb = image.convert("RGB")
        pixel_values = model.get_vision_tower().image_processor(
            images=image_rgb, return_tensors="pt",
        )["pixel_values"]
        pixel_values = pixel_values.to(device, dtype=next(model.parameters()).dtype)
        with torch.no_grad():
            ids = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                images=pixel_values,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        # FastVLM's custom generate() returns only new tokens (not
        # input_ids prefix), so decode directly without slicing.
        pred = processor.batch_decode(ids, skip_special_tokens=True)[0]
        # Take only the first line — FastVLM tends to repeat/ramble after it.
        pred = pred.split("\n")[0].strip()

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

def evaluate_dataset(model, processor, samples: list, family: str,
                     device: str, dataset_name: str,
                     accuracy_fn, batch_size: int = 1) -> dict:
    """Evaluate model on a dataset with multi-metric scoring.

    For VQA datasets (accuracy_fn == _vqa_accuracy), computes all metrics:
      exact_match, contains, token_f1, bleu, rouge_l.
    For POPE, only exact match is used (binary yes/no task).
    """
    logger.info(f"  Evaluating on {dataset_name} ({len(samples)} samples)...")

    # Determine if we should compute multi-metric (VQA-style) or single (POPE)
    use_multi = (accuracy_fn is _vqa_accuracy)

    scores, latencies = [], []
    multi_scores = []  # list of dicts when use_multi=True
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

            if use_multi:
                m = _vqa_multi_metric(pred, sample["answers"])
                multi_scores.append(m)
                scores.append(m["exact_match"])
            else:
                scores.append(accuracy_fn(pred, sample["answers"]))

    if skipped:
        logger.warning(f"  {dataset_name}: skipped {skipped}/{len(samples)} samples due to errors")

    n_evaluated = len(scores)
    stats = profiler.stats()
    avg_acc   = sum(scores) / n_evaluated if n_evaluated else 0.0
    avg_lat   = sum(latencies) / len(latencies) if latencies else 0.0
    throughput = len(latencies) / stats.wall_time_s if stats.wall_time_s > 0 else 0.0

    # Flag suspicious results
    if n_evaluated == 0:
        logger.error(
            f"  {dataset_name}: ALL {len(samples)} samples failed! "
            f"Inference is broken — accuracy=0.0 is NOT valid."
        )
    elif avg_acc == 0.0 and n_evaluated >= 5:
        logger.error(
            f"  {dataset_name}: acc=0.0 on {n_evaluated} samples — "
            f"model output is likely broken (empty/garbage predictions)."
        )

    # Aggregate multi-metric scores
    if use_multi and multi_scores:
        metric_names = list(multi_scores[0].keys())
        metric_avgs = {
            name: round(sum(m[name] for m in multi_scores) / len(multi_scores), 4)
            for name in metric_names
        }
        logger.info(
            f"  {dataset_name}: "
            + "  ".join(f"{k}={v:.4f}" for k, v in metric_avgs.items())
            + f"  lat={avg_lat:.3f}s  mem={stats.peak_memory_mb:.0f}MB  "
            + f"tput={throughput:.2f} samp/s  "
            + f"evaluated={n_evaluated}/{len(samples)}"
        )
    else:
        metric_avgs = {}
        logger.info(
            f"  {dataset_name}: acc={avg_acc:.4f}  "
            f"lat={avg_lat:.3f}s  mem={stats.peak_memory_mb:.0f}MB  "
            f"tput={throughput:.2f} samp/s  "
            f"evaluated={n_evaluated}/{len(samples)}"
        )

    result = {
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
        "zero_accuracy_warning": avg_acc == 0.0 and n_evaluated >= 5,
    }

    # Add multi-metric averages (only for VQA-style datasets)
    if metric_avgs:
        result["metrics"] = metric_avgs

    return result


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
