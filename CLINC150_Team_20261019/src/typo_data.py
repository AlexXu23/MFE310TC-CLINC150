"""Model-independent, nested character-typo stress tests for official test rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import pandas as pd

from common import load_dataset, validate_full_dataset


TOKEN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{5,}(?![A-Za-z0-9])")
CONDITIONS = ("clean", "typo_light", "typo_medium")


def mutate_token(token: str, rng: random.Random) -> tuple[str, str, int]:
    swaps = [i for i in range(1, len(token) - 2) if token[i] != token[i + 1]]
    if swaps and rng.randrange(2) == 0:
        index = rng.choice(swaps)
        changed = token[:index] + token[index + 1] + token[index] + token[index + 2:]
        return changed, "swap", index
    index = rng.randrange(1, len(token) - 1)
    return token[:index] + token[index + 1:], "delete", index


def make_variants(text: str, row_id: int, seed: int = 2026) -> dict:
    digest = hashlib.sha256(f"{seed}\0{row_id}\0{text}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    tokens = list(TOKEN.finditer(text))
    rng.shuffle(tokens)
    mutations = []
    for match in tokens[:2]:
        replacement, operation, index = mutate_token(match.group(), rng)
        mutations.append({
            "start": match.start(), "end": match.end(), "original": match.group(),
            "replacement": replacement, "operation": operation, "character_index": index,
        })
    result = {}
    for condition, limit in zip(CONDITIONS, (0, 1, 2)):
        selected = mutations[:limit]
        changed = text
        for mutation in sorted(selected, key=lambda item: item["start"], reverse=True):
            changed = changed[:mutation["start"]] + mutation["replacement"] + changed[mutation["end"]:]
        result[condition] = {
            "text": changed, "changed": changed != text,
            "eligible_tokens": len(tokens), "modified_tokens": len(selected),
            "mutations": selected,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    output = Path(args.output)
    if (output / "summary.json").exists():
        raise FileExistsError("Typo data already exist; choose a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    data = load_dataset(args.data)
    validation = validate_full_dataset(data)
    mapping = {label: index for index, label in enumerate(validation["in_scope_intents"])}
    all_rows = data["test"] + data["oos_test"]
    frames = {condition: [] for condition in CONDITIONS}
    for row_id, (text, label) in enumerate(all_rows):
        variants = make_variants(text, row_id, args.seed)
        for condition, variant in variants.items():
            frames[condition].append({
                "row_id": row_id, "source_split": "oos_test" if label == "oos" else "test",
                "true_label": label, "label_id": mapping.get(label, -1),
                "is_oos": label == "oos", "original_text": text,
                "condition": condition, **{key: value for key, value in variant.items() if key != "mutations"},
                "mutations_json": json.dumps(variant["mutations"], ensure_ascii=False),
            })
    coverage = []
    hashes = {}
    for condition, rows in frames.items():
        frame = pd.DataFrame(rows)
        path = output / f"{condition}.csv"
        frame.to_csv(path, index=False)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for is_oos, group in frame.groupby("is_oos"):
            coverage.append({
                "condition": condition, "scope": "oos" if is_oos else "in_scope",
                "n": len(group), "changed": int(group["changed"].sum()),
                "changed_fraction": float(group["changed"].mean()),
                "mean_modified_tokens": float(group["modified_tokens"].mean()),
                "two_tokens_changed": int((group["modified_tokens"] == 2).sum()),
            })
    pd.DataFrame(coverage).to_csv(output / "coverage.csv", index=False)
    summary = {
        "seed": args.seed, "examples_per_condition": len(all_rows), "conditions": list(CONDITIONS),
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "files_sha256": hashes, "coverage": coverage,
        "medium_contains_light_mutations": True,
        "label_preservation": "Inherited labels; automatic character stress test, not human-verified semantic equivalence.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(pd.DataFrame(coverage).to_string(index=False))


if __name__ == "__main__":
    main()
