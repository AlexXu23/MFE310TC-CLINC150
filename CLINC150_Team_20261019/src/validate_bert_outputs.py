"""Validate BERT logit exports and independently recompute in-scope metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


EXPECTED = {
    "validation": {"rows": 3_100, "in_scope": 3_000, "oos": 100},
    "test": {"rows": 5_500, "in_scope": 4_500, "oos": 1_000},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="BERT run output directory")
    args = parser.parse_args()
    run = Path(args.run)

    mapping = json.loads((run / "label_mapping.json").read_text(encoding="utf-8"))
    n_labels = len(mapping["label_to_id"])
    if n_labels != 150:
        raise ValueError(f"Expected 150 labels, found {n_labels}")

    results = {}
    for split, expected in EXPECTED.items():
        logits = np.load(run / f"{split}_logits.npy", allow_pickle=False)
        targets = np.load(run / f"{split}_targets.npz", allow_pickle=False)
        metadata = pd.read_csv(run / f"{split}_metadata.csv")
        label_ids = targets["label_ids"]
        is_oos = targets["is_oos"]

        expected_shape = (expected["rows"], n_labels)
        if logits.shape != expected_shape:
            raise ValueError(f"{split}: expected {expected_shape}, found {logits.shape}")
        if len(metadata) != expected["rows"] or len(label_ids) != expected["rows"]:
            raise ValueError(f"{split}: row count mismatch")
        if not np.isfinite(logits).all():
            raise ValueError(f"{split}: logits contain NaN or infinity")
        if int(is_oos.sum()) != expected["oos"]:
            raise ValueError(f"{split}: unexpected OOS count")
        if not np.array_equal(metadata["is_oos"].astype(bool).to_numpy(), is_oos):
            raise ValueError(f"{split}: metadata and target OOS flags differ")
        if not np.array_equal(metadata["label_id"].to_numpy(dtype=np.int64), label_ids):
            raise ValueError(f"{split}: metadata and numeric label ids differ")
        if not np.all(label_ids[is_oos] == -1):
            raise ValueError(f"{split}: OOS label ids must equal -1")

        in_scope = ~is_oos
        predictions = logits[in_scope].argmax(axis=1)
        results[split] = {
            "shape": list(logits.shape),
            "dtype": str(logits.dtype),
            "in_scope_rows": int(in_scope.sum()),
            "oos_rows": int(is_oos.sum()),
            "in_scope_accuracy": float(accuracy_score(label_ids[in_scope], predictions)),
            "in_scope_macro_f1": float(
                f1_score(label_ids[in_scope], predictions, average="macro")
            ),
            "logit_min": float(logits.min()),
            "logit_max": float(logits.max()),
        }

    (run / "validation_report.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
