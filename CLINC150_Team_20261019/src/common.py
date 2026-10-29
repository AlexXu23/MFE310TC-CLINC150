"""Shared loading and validation helpers for the official CLINC150 JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


Pair = Tuple[str, str]

EXPECTED_SPLITS = {
    "train": 15_000,
    "val": 3_000,
    "test": 4_500,
    "oos_train": 100,
    "oos_val": 100,
    "oos_test": 1_000,
}


def load_dataset(path: str | Path) -> Dict[str, List[Pair]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    missing = sorted(set(EXPECTED_SPLITS) - set(raw))
    if missing:
        raise ValueError(f"Missing required split(s): {missing}")

    dataset: Dict[str, List[Pair]] = {}
    for split in EXPECTED_SPLITS:
        rows = raw[split]
        if not isinstance(rows, list):
            raise TypeError(f"Split {split!r} is not a list")
        parsed: List[Pair] = []
        for index, row in enumerate(rows):
            if not (isinstance(row, list) and len(row) == 2):
                raise ValueError(f"Invalid row at {split}[{index}]: {row!r}")
            text, label = row
            if not isinstance(text, str) or not isinstance(label, str):
                raise TypeError(f"Non-string value at {split}[{index}]")
            parsed.append((text, label))
        dataset[split] = parsed
    return dataset


def validate_full_dataset(dataset: Dict[str, List[Pair]]) -> dict:
    actual_sizes = {key: len(dataset[key]) for key in EXPECTED_SPLITS}
    if actual_sizes != EXPECTED_SPLITS:
        raise ValueError(
            f"Unexpected full-dataset sizes: {actual_sizes}; "
            f"expected {EXPECTED_SPLITS}"
        )

    in_scope_labels = sorted({label for _, label in dataset["train"]})
    if len(in_scope_labels) != 150:
        raise ValueError(f"Expected 150 train intents, found {len(in_scope_labels)}")

    for split in ("train", "val", "test"):
        labels = {label for _, label in dataset[split]}
        if labels != set(in_scope_labels):
            raise ValueError(f"Intent inventory mismatch in {split}")

    for split in ("oos_train", "oos_val", "oos_test"):
        labels = {label for _, label in dataset[split]}
        if labels != {"oos"}:
            raise ValueError(f"Unexpected labels in {split}: {sorted(labels)}")

    return {
        "split_sizes": actual_sizes,
        "n_in_scope_intents": len(in_scope_labels),
        "in_scope_intents": in_scope_labels,
    }

