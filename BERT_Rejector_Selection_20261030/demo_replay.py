"""Replay saved model outputs for real CLINC sentences without model weights.

This demo does NOT run BERT. It looks up saved predictions and frozen validation
thresholds. Always label the display "replay of saved outputs, no live inference".

  python BERT_Rejector_Selection_20261030/demo_replay.py --text "how do i make my android phone more secure"
  python BERT_Rejector_Selection_20261030/demo_replay.py --row-id 4508 --split test
  python BERT_Rejector_Selection_20261030/demo_replay.py --cases
  python BERT_Rejector_Selection_20261030/demo_replay.py --interactive
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RS = ROOT / "BERT_Rejection_Study_20261019/outputs"
SCORES = (("msp", "MSP"), ("entropy", "Entropy"), ("mahalanobis", "Mahalanobis"))


def load(seed):
    frames = []
    for split in ("validation", "test"):
        frame = pd.read_csv(RS / f"seed{seed}/{split}_predictions.csv", keep_default_na=False, encoding="utf-8-sig")
        frame["split"] = split
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def find(rows, text, limit=5):
    """Exact (case-insensitive) match first, otherwise rows containing the text."""
    query = " ".join(text.lower().split())
    normalised = rows.text.str.lower().str.split().str.join(" ")
    exact = rows[normalised == query]
    return exact if len(exact) else rows[normalised.str.contains(query, regex=False)].head(limit)


def decide(row, thresholds):
    result = {"split": row.split, "row_id": int(row.row_id), "text": row.text, "true_label": row.true_label,
              "predicted_intent": row.predicted_label}
    for score, _ in SCORES:
        value = float(row[f"{score}_score"])
        result[f"{score}_score"] = value
        result[f"{score}_threshold"] = thresholds[score]
        result[f"{score}_decision"] = "REJECT" if value >= thresholds[score] else "accept"
    return result


def show(result):
    truth = "out of scope" if result["true_label"] == "oos" else result["true_label"]
    print(f"\n\"{result['text']}\"   [{result['split']} row {result['row_id']}; ground truth: {truth}]")
    print(f"  predicted intent: {result['predicted_intent']}")
    for score, name in SCORES:
        print(f"  {name:<12} score {result[f'{score}_score']:>10.3f}   threshold {result[f'{score}_threshold']:>10.3f}"
              f"   -> {result[f'{score}_decision']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", action="append", default=[], help="sentence (or part of one) to look up")
    parser.add_argument("--row-id", type=int, action="append", default=[], help="row_id within --split")
    parser.add_argument("--split", default="test", choices=("validation", "test"), help="split for --row-id")
    parser.add_argument("--cases", action="store_true", help="replay outputs/disagreement_cases.csv (seed 42)")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--seed", type=int, default=42, choices=(42, 43, 44))
    parser.add_argument("--policy", default="is_acceptance_95", choices=("is_acceptance_95", "max_oos_f1"))
    args = parser.parse_args()
    if not (args.text or args.row_id or args.cases or args.interactive):
        parser.error("give --text, --row-id, --cases or --interactive")
    rows = load(args.seed)
    frozen = json.loads((RS / f"seed{args.seed}/frozen_thresholds.json").read_text(encoding="utf-8"))
    thresholds = {score: frozen[f"{score}/{args.policy}"]["threshold"] for score, _ in SCORES}
    print(f"Replay of saved outputs (no live inference): original BERT seed {args.seed}, policy {args.policy}, "
          f"reject iff score >= threshold")
    results = []

    def run(matches, label):
        if matches.empty:
            print(f"\n{label}: not in the saved validation/test rows")
        for _, row in matches.iterrows():
            results.append(decide(row, thresholds))
            show(results[-1])

    for text in args.text:
        run(find(rows, text), f"\"{text}\"")
    for row_id in args.row_id:
        run(rows[(rows.split == args.split) & (rows.row_id == row_id)], f"{args.split} row {row_id}")
    if args.cases:
        cases = pd.read_csv(Path(__file__).resolve().parent / "outputs/disagreement_cases.csv", keep_default_na=False)
        for category, group in cases.groupby("category", sort=False):
            print(f"\n=== {category}")
            run(rows[(rows.split == "test") & rows.row_id.isin(group.row_id)], category)
    while args.interactive:
        text = input("\nsentence (empty to stop)> ").strip()
        if not text:
            break
        run(find(rows, text), f"\"{text}\"")
    if results:
        out = Path(__file__).resolve().parent / "outputs/demo"
        out.mkdir(parents=True, exist_ok=True)
        target = out / f"replay_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S}.csv"
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nsaved {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
