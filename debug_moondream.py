#!/usr/bin/env python3
"""Quick debug: see what moondream2 actually produces on VQAv2."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from models.model_loader import load_model
from evaluation.run_baseline import load_vqav2, _vqa_accuracy

print("=== Loading moondream2 ===")
model, processor, meta = load_model("vikhyatk/moondream2")
print(f"Device: {meta.device}, Dtype: {meta.dtype}, Mem: {meta.gpu_mem_delta_mb:.0f} MB")

print("\n=== Loading 5 VQAv2 samples ===")
samples = load_vqav2(n_samples=5)

for i, sample in enumerate(samples):
    question = sample["question"] + " Answer with a single word or short phrase."
    image = sample["image"].convert("RGB")
    gold = sample["answers"]

    print(f"\n--- Sample {i} ---")
    print(f"Q: {question}")
    print(f"Gold answers: {gold[:5]}")

    # Run inference the same way as run_baseline.py
    enc_image = model.encode_image(image)
    pred = model.answer_question(enc_image, question, processor)
    pred_stripped = pred.strip()

    score = _vqa_accuracy(pred_stripped, gold)
    print(f"Prediction: '{pred_stripped}'")
    print(f"Accuracy: {score}")
    print(f"Pred clean: '{pred_stripped.lower().rstrip('.')}'")

# Cleanup
del model, processor
import gc; gc.collect()
torch.cuda.empty_cache()
print("\n=== Done ===")
