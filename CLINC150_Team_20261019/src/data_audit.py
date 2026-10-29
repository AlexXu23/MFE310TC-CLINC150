"""Extended, non-destructive CLINC150 audit; never changes official splits.

Network access is opt-in and only pins public provenance. Normal runs use the
cached commit-pinned files, the local tokenizer, and the original dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer

from common import EXPECTED_SPLITS, load_dataset, validate_full_dataset
from evaluate_oos import file_hash, write_json


def normalize_text(text: str) -> str:
    """Conservative audit key: NFKC, case folding and whitespace only."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def pin_provenance(project: Path) -> dict:
    """Cache immutable upstream metadata and verify byte identity locally."""
    target = project / "data/provenance"
    target.mkdir(parents=True, exist_ok=True)
    def fetch(url: str) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": "CLINC150-coursework-audit"})
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read()
    commits = json.loads(fetch("https://api.github.com/repos/clinc/oos-eval/commits?per_page=1"))
    commit = commits[0]["sha"]
    base = f"https://raw.githubusercontent.com/clinc/oos-eval/{commit}"
    remote_data = fetch(f"{base}/data/data_full.json")
    local_hash = file_hash(project / "data/raw/data_full.json")
    if hashlib.sha256(remote_data).hexdigest() != local_hash:
        raise AssertionError("Local dataset differs from the pinned upstream version")
    for name, relative in (("domains.json", "data/domains.json"), ("LICENSE", "LICENSE"), ("README.md", "README.md")):
        (target / name).write_bytes(fetch(f"{base}/{relative}"))
    manifest = {
        "repository": "https://github.com/clinc/oos-eval", "commit": commit,
        "data_url": f"{base}/data/data_full.json", "data_sha256": local_hash,
        "local_matches_pinned_remote_bytes": True,
        "license_identifier": "CC-BY-3.0", "license_url": f"{base}/LICENSE",
        "paper": "https://aclanthology.org/D19-1131/",
        "note": "Identifies the repository license; preserve attribution and identify derived typo versions.",
        "files_sha256": {name: file_hash(target / name) for name in ("domains.json", "LICENSE", "README.md")},
    }
    write_json(target / "manifest.json", manifest)
    return manifest


def duplicate_groups(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    """List every member of a repeated text group, with actual split conflicts."""
    repeated = frame[frame.duplicated(key, keep=False)].copy()
    if repeated.empty:
        repeated["group_id"] = pd.Series(dtype=int)
        repeated["cross_official_split"] = pd.Series(dtype=bool)
        repeated["label_conflict"] = pd.Series(dtype=bool)
        return repeated
    groups = repeated.groupby(key, sort=True)
    repeated["group_id"] = groups.ngroup()
    repeated["cross_official_split"] = groups["split"].transform("nunique") > 1
    repeated["label_conflict"] = groups["label"].transform("nunique") > 1
    return repeated.sort_values(["group_id", "source_split", "source_row"])


def audit(project: Path, output: Path) -> dict:
    """Export structural, linguistic, leakage and vocabulary audit tables."""
    data_path = project / "data/raw/data_full.json"
    dataset = load_dataset(data_path)
    verified = validate_full_dataset(dataset)
    raw = json.loads(data_path.read_text(encoding="utf-8"))
    if set(raw) != set(EXPECTED_SPLITS):
        raise AssertionError("Unexpected raw dataset keys")
    tokenizer_path = project / "outputs/bert/seed42/best_model"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, use_fast=True)
    rows = []
    for source, pairs in dataset.items():
        for index, (text, label) in enumerate(pairs):
            rows.append({"source_split": source, "source_row": index,
                         "split": source.removeprefix("oos_"), "scope": "oos" if source.startswith("oos_") else "in_scope",
                         "text": text, "label": label, "normalized_text": normalize_text(text),
                         "words": len(text.split()), "characters": len(text),
                         "has_digit": bool(re.search(r"\d", text)), "has_non_ascii": not text.isascii(),
                         "has_question_mark": "?" in text,
                         "email_like": bool(re.search(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b", text))})
    frame = pd.DataFrame(rows)
    encoded = tokenizer(frame["text"].tolist(), add_special_tokens=True, truncation=False)["input_ids"]
    frame["wordpieces_with_special_tokens"] = [len(value) for value in encoded]
    frame["unk_tokens"] = [value.count(tokenizer.unk_token_id) for value in encoded]
    frame["truncated_at_64"] = frame["wordpieces_with_special_tokens"] > 64
    domains = json.loads((project / "data/provenance/domains.json").read_text(encoding="utf-8"))
    label_domain = {intent: domain for domain, intents in domains.items() for intent in intents}
    if len(label_domain) != 150 or set(label_domain) != set(verified["in_scope_intents"]):
        raise AssertionError("Upstream domain mapping does not cover exactly 150 intents")
    frame["domain"] = frame["label"].map(label_domain).fillna("oos")
    output.mkdir(parents=True, exist_ok=True)
    duplicates = {}
    for key, name in (("text", "exact"), ("normalized_text", "normalized")):
        table = duplicate_groups(frame, key)
        table.to_csv(output / f"duplicates_{name}.csv", index=False)
        duplicates[name] = {
            "groups": int(table[key].nunique()), "rows_involved": len(table),
            "cross_official_split_groups": int(table.loc[table["cross_official_split"], key].nunique()),
            "cross_split_conflicting_label_groups": int(table.loc[table["cross_official_split"] & table["label_conflict"], key].nunique()),
        }
    lengths = frame.groupby(["source_split", "scope"], sort=True).agg(
        n=("text", "size"), word_mean=("words", "mean"), word_median=("words", "median"),
        word_p95=("words", lambda s: s.quantile(.95)), word_max=("words", "max"),
        wordpiece_mean=("wordpieces_with_special_tokens", "mean"),
        wordpiece_p95=("wordpieces_with_special_tokens", lambda s: s.quantile(.95)),
        wordpiece_max=("wordpieces_with_special_tokens", "max"), truncations_at_64=("truncated_at_64", "sum"),
        unknown_tokens=("unk_tokens", "sum"), digit_share=("has_digit", "mean"),
        non_ascii_share=("has_non_ascii", "mean"), question_mark_share=("has_question_mark", "mean"))
    lengths.to_csv(output / "linguistic_summary.csv")
    frame.groupby(["split", "scope", "domain"]).agg(n=("text", "size"), intents=("label", "nunique")).to_csv(output / "domain_distribution.csv")
    balance = []
    for split, expected in (("train", 100), ("val", 20), ("test", 30)):
        counts = Counter(label for _, label in dataset[split])
        assert set(counts.values()) == {expected}
        balance.append({"split": split, "intents": len(counts), "min_per_intent": min(counts.values()), "max_per_intent": max(counts.values())})
    pd.DataFrame(balance).to_csv(output / "intent_balance.csv", index=False)
    scope = frame.groupby(["split", "scope"]).size().unstack().reindex(["train", "val", "test"])
    scope["total"] = scope.sum(axis=1)
    scope["oos_share"] = scope["oos"] / scope["total"]
    scope.to_csv(output / "scope_prevalence.csv")
    # Train-only lexical features: audit candidates, not automatic relabeling.
    train = frame[frame["source_split"] == "train"].reset_index(drop=True)
    other = frame[frame["split"].isin(["val", "test"])].copy().reset_index(drop=True)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    train_x = vectorizer.fit_transform(train["text"])
    other_x = vectorizer.transform(other["text"])
    neighbors = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=2).fit(train_x)
    distances, indices = neighbors.kneighbors(other_x)
    other["nearest_train_row"] = indices[:, 0]
    other["cosine_similarity"] = np.clip(1 - distances[:, 0], 0, 1)
    other["nearest_train_text"] = train.iloc[indices[:, 0]]["text"].to_numpy()
    other["nearest_train_label"] = train.iloc[indices[:, 0]]["label"].to_numpy()
    candidates = other[other["cosine_similarity"] >= .95 - 1e-12]
    candidates.to_csv(output / "near_duplicate_candidates_cosine_ge_095.csv", index=False)
    analyzer = vectorizer.build_analyzer()
    vocabulary = vectorizer.vocabulary_
    oov = []
    for source, subset in frame.groupby("source_split"):
        tokens = [token for text in subset["text"] for token in analyzer(text) if " " not in token]
        oov.append({"source_split": source, "unigram_tokens": len(tokens),
                    "oov_tokens": sum(token not in vocabulary for token in tokens),
                    "oov_rate": sum(token not in vocabulary for token in tokens) / len(tokens)})
    pd.DataFrame(oov).to_csv(output / "train_vocabulary_oov.csv", index=False)
    frame.to_csv(output / "row_audit.csv", index=False)
    summary = {
        "data_sha256": file_hash(data_path), "rows": len(frame), "schema_and_split_inventory_valid": True,
        "blank_texts": int(frame["text"].str.strip().eq("").sum()), "blank_labels": int(frame["label"].str.strip().eq("").sum()),
        "missing_values": int(frame.isna().sum().sum()), "duplicates": duplicates,
        "domains": len(domains), "domain_mapping_covers_all_intents": True,
        "truncated_at_64": int(frame["truncated_at_64"].sum()), "maximum_wordpieces": int(frame["wordpieces_with_special_tokens"].max()),
        "email_like_rows_heuristic_only": int(frame["email_like"].sum()), "near_duplicate_candidates_ge_095": len(candidates),
        "normalization": "NFKC + casefold + whitespace collapse; punctuation preserved",
        "near_duplicate_limit": "Train-fit word TF-IDF cosine >=0.95 flags review candidates, not proven leakage; misses semantic paraphrases.",
        "data_policy": "No source rows changed; sensitivity masks exclude only evaluation overlap with earlier official splits.",
        "bias_limitations": ["Balanced intent counts do not demonstrate demographic fairness.", "English crowdsourced assistant requests do not validate industrial or multilingual deployment.", "No demographic annotations: protected-group fairness is not measured.", "OOS label merges heterogeneous unsupported requests and is not a well-sampled 151st intent.", "Simple email regex is not a complete privacy audit."],
    }
    write_json(output / "summary.json", summary)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for axis, field, label in zip(axes[:2], ("words", "wordpieces_with_special_tokens"), ("Whitespace words", "BERT WordPieces incl. special tokens")):
        for scope_name, color in (("in_scope", "#286694"), ("oos", "#ce6238")):
            axis.hist(frame.loc[frame["scope"] == scope_name, field], bins=np.arange(0, frame[field].max() + 2) - .5,
                      density=True, histtype="step", label=scope_name, color=color)
        axis.set(xlabel=label, ylabel="Density")
        axis.legend(frameon=False, fontsize=8)
    axes[2].bar(scope.index, scope["oos_share"] * 100, color="#ce6238")
    axes[2].set(xlabel="Official split", ylabel="OOS share (%)")
    for i, value in enumerate(scope["oos_share"]):
        axes[2].text(i, value * 100 + .4, f"{100 * value:.2f}%", ha="center")
    axes[2].set_ylim(0, 22)
    fig.tight_layout()
    fig.savefig(output / "data_audit_overview.png", dpi=180)
    plt.close(fig)
    return summary


def main() -> None:
    """Run from the project root, optionally pinning upstream metadata once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("outputs/data_audit"))
    parser.add_argument("--fetch-provenance", action="store_true")
    args = parser.parse_args()
    if args.fetch_provenance:
        pin_provenance(args.project)
    print(json.dumps(audit(args.project, args.output), indent=2))


if __name__ == "__main__":
    main()
