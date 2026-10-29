"""Reproducible exploratory analysis for the official CLINC150 full split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import EXPECTED_SPLITS, load_dataset, validate_full_dataset


def token_count(text: str) -> int:
    return len(text.split())


def char_count(text: str) -> int:
    return len(text)


def build_frame(dataset: dict) -> pd.DataFrame:
    rows = []
    for split, pairs in dataset.items():
        base_split = split.replace("oos_", "")
        for text, label in pairs:
            rows.append(
                {
                    "source_split": split,
                    "split": base_split,
                    "scope": "oos" if split.startswith("oos_") else "in_scope",
                    "label": label,
                    "text": text,
                    "word_count": token_count(text),
                    "char_count": char_count(text),
                }
            )
    return pd.DataFrame(rows)


def save_length_plot(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"in_scope": "#31688e", "oos": "#d1495b"}
    bins_words = np.arange(0, frame["word_count"].max() + 2) - 0.5
    bins_chars = 30
    for scope in ("in_scope", "oos"):
        subset = frame.loc[frame["scope"] == scope]
        axes[0].hist(
            subset["word_count"], bins=bins_words, density=True, alpha=0.55,
            label=scope, color=colors[scope]
        )
        axes[1].hist(
            subset["char_count"], bins=bins_chars, density=True, alpha=0.55,
            label=scope, color=colors[scope]
        )
    axes[0].set(xlabel="Whitespace-delimited words", ylabel="Density", title="Text length by scope")
    axes[1].set(xlabel="Characters", ylabel="Density", title="Character length by scope")
    for axis in axes:
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_oos_plot(frame: pd.DataFrame, output: Path) -> None:
    counts = frame.groupby(["split", "scope"]).size().unstack(fill_value=0)
    counts = counts.reindex(["train", "val", "test"])
    ax = counts.plot.bar(stacked=True, color=["#31688e", "#d1495b"], figsize=(7.5, 4.5))
    ax.set(xlabel="Official split", ylabel="Examples", title="In-scope and OOS composition")
    ax.tick_params(axis="x", rotation=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="Scope", frameon=False)
    ax.figure.tight_layout()
    ax.figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(ax.figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.data)
    validation = validate_full_dataset(dataset)
    frame = build_frame(dataset)

    split_summary = (
        frame.groupby(["source_split", "scope"], as_index=False)
        .agg(examples=("text", "size"), unique_texts=("text", "nunique"), intents=("label", "nunique"))
        .sort_values("source_split")
    )
    length_summary = (
        frame.groupby(["source_split", "scope"], as_index=False)
        .agg(
            examples=("text", "size"),
            word_mean=("word_count", "mean"),
            word_median=("word_count", "median"),
            word_p95=("word_count", lambda s: s.quantile(0.95)),
            word_max=("word_count", "max"),
            char_mean=("char_count", "mean"),
            char_median=("char_count", "median"),
            char_p95=("char_count", lambda s: s.quantile(0.95)),
            char_max=("char_count", "max"),
        )
        .sort_values("source_split")
    )

    intent_rows = []
    for split in ("train", "val", "test"):
        counts = Counter(label for _, label in dataset[split])
        for label, count in sorted(counts.items()):
            intent_rows.append({"split": split, "intent": label, "count": count})
    intent_counts = pd.DataFrame(intent_rows)

    duplicate_texts = int(frame.duplicated(subset=["text"], keep=False).sum())
    cross_split_duplicates = int(
        frame.groupby("text")["source_split"].nunique().gt(1).sum()
    )
    blank_texts = int(frame["text"].str.strip().eq("").sum())

    oos_shares = {}
    for split in ("train", "val", "test"):
        subset = frame[frame["split"] == split]
        oos_shares[split] = float((subset["scope"] == "oos").mean())

    summary = {
        **validation,
        "total_examples": int(len(frame)),
        "blank_texts": blank_texts,
        "duplicate_rows_involved_by_exact_text": duplicate_texts,
        "exact_texts_appearing_across_source_splits": cross_split_duplicates,
        "oos_share_by_combined_split": oos_shares,
        "max_word_count_all": int(frame["word_count"].max()),
        "p95_word_count_all": float(frame["word_count"].quantile(0.95)),
    }

    split_summary.to_csv(output / "split_summary.csv", index=False)
    length_summary.to_csv(output / "text_length_summary.csv", index=False, float_format="%.3f")
    intent_counts.to_csv(output / "intent_counts.csv", index=False)
    frame.to_csv(output / "examples_with_lengths.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_length_plot(frame, output / "text_length_distribution.png")
    save_oos_plot(frame, output / "scope_distribution.png")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nText-length summary:\n", length_summary.to_string(index=False))


if __name__ == "__main__":
    main()

