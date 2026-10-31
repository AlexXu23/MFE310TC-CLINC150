"""Run frozen BERT inference and show three rejection scores and decisions.

Uses the original BERT (default seed 42), its train-fitted Mahalanobis detector
and frozen validation thresholds. Inference only: no fitting or threshold tuning.
Output is printed and saved to outputs/demo/.

  python BERT_Rejector_Selection_20261030/demo_rejector.py --text "what is my account balance"
  python BERT_Rejector_Selection_20261030/demo_rejector.py --interactive
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RS = ROOT / "BERT_Rejection_Study_20261019"
sys.dont_write_bytecode = True
sys.path[:0] = [str(RS), str(ROOT / "CLINC150_Team_20261019" / "src")]
for key, value in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "USE_TF": "0", "USE_FLAX": "0"}.items():
    os.environ[key] = value

import numpy as np
import torch
from transformers import AutoTokenizer, BertForSequenceClassification

from distance_model import SharedMahalanobisDetector
from evaluate_oos import anomaly_scores

SCORES = ("msp", "entropy", "mahalanobis")


class Rejector:
    def __init__(self, seed, policy):
        model_dir = ROOT / f"CLINC150_Team_20261019/outputs/bert/seed{seed}/best_model"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
        self.model = BertForSequenceClassification.from_pretrained(model_dir, local_files_only=True,
                                                                   attn_implementation="eager")
        self.model.to(device=self.device, dtype=torch.float32).eval()
        self.detector = SharedMahalanobisDetector.load(RS / f"outputs/seed{seed}/distance_model.npz")
        frozen = json.loads((RS / f"outputs/seed{seed}/frozen_thresholds.json").read_text(encoding="utf-8"))
        self.thresholds = {score: frozen[f"{score}/{policy}"]["threshold"] for score in SCORES}
        mapping = json.loads((ROOT / f"CLINC150_Team_20261019/outputs/bert/seed{seed}/label_mapping.json").read_text(encoding="utf-8"))
        self.id_to_label = mapping["id_to_label"]

    @torch.inference_mode()
    def __call__(self, texts):
        batch = self.tokenizer(texts, truncation=True, max_length=64, padding=True, pad_to_multiple_of=8, return_tensors="pt")
        result = self.model(**{k: v.to(self.device) for k, v in batch.items()}, output_hidden_states=True, return_dict=True)
        logits = result.logits.float().cpu().numpy()
        features = result.hidden_states[-1][:, 0, :].float().cpu().numpy()
        scores = anomaly_scores(logits)
        scores["mahalanobis"] = self.detector.score_samples(features)
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        rows = []
        for i, text in enumerate(texts):
            row = {"text": text, "predicted_intent": self.id_to_label[str(int(logits[i].argmax()))],
                   "top_probability": float(probabilities[i].max())}
            for score in SCORES:
                value = float(scores[score][i])
                row[f"{score}_score"] = value
                row[f"{score}_threshold"] = self.thresholds[score]
                row[f"{score}_decision"] = "REJECT" if value >= self.thresholds[score] else "accept"
            rows.append(row)
        return rows


def show(rows):
    for row in rows:
        print(f"\n\"{row['text']}\"\n  predicted intent: {row['predicted_intent']} (p = {row['top_probability']:.2f})")
        for score, name in zip(SCORES, ("MSP", "Entropy", "Mahalanobis")):
            print(f"  {name:<12} score {row[f'{score}_score']:>10.3f}   threshold {row[f'{score}_threshold']:>10.3f}"
                  f"   -> {row[f'{score}_decision']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", action="append", default=[])
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--seed", type=int, default=42, choices=(42, 43, 44))
    parser.add_argument("--policy", default="is_acceptance_95", choices=("is_acceptance_95", "max_oos_f1"))
    args = parser.parse_args()
    if not args.text and not args.interactive:
        parser.error("give --text at least once or use --interactive")
    rejector = Rejector(args.seed, args.policy)
    print(f"Loaded seed {args.seed} on {rejector.device}; policy {args.policy}; reject iff score >= threshold")
    rows = rejector(args.text) if args.text else []
    show(rows)
    while args.interactive:
        text = input("\nsentence (empty to stop)> ").strip()
        if not text:
            break
        new = rejector([text])
        show(new)
        rows += new
    if rows:
        out = Path(__file__).resolve().parent / "outputs/demo"
        out.mkdir(parents=True, exist_ok=True)
        target = out / f"demo_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S}.csv"
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nsaved {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
