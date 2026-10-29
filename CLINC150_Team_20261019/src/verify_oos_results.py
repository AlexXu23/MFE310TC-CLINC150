"""Independent checks of stored results using sklearn and source JSON rows.

This verifier deliberately does not import evaluate_oos so the metric formulas
and row alignment are checked via a separate implementation.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import (
    average_precision_score, log_loss, precision_recall_fscore_support, roc_auc_score,
)


def require_close(actual, expected, label):
    if not np.isclose(actual, expected, atol=1e-10, rtol=1e-8):
        raise AssertionError(f"{label}: {actual} != {expected}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    run, output = Path(args.run), Path(args.output)
    original = json.loads(Path(args.data).read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    for name, expected_hash in summary["input_sha256"].items():
        if hashlib.sha256((run / name).read_bytes()).hexdigest() != expected_hash:
            raise AssertionError(f"Input modified: {name}")

    logits, metadata, scores, is_oos = {}, {}, {}, {}
    for split in ("validation", "test"):
        metadata[split] = pd.read_csv(run / f"{split}_metadata.csv", keep_default_na=False)
        source_split = "val" if split == "validation" else "test"
        source = original[source_split] + original[f"oos_{source_split}"]
        if metadata[split][["text", "true_label"]].values.tolist() != source:
            raise AssertionError(f"{split} metadata does not match official JSON order")
        logits[split] = np.load(run / f"{split}_logits.npy", allow_pickle=False).astype(float)
        scores[split] = pd.read_csv(output / f"{split}_scores.csv", keep_default_na=False)
        is_oos[split] = metadata[split]["is_oos"].to_numpy()

    results = pd.read_csv(output / "oos_results.csv")
    checked_rows = 0
    for row in results.to_dict("records"):
        column = f"{row['calibration']}_{row['score']}_anomaly"
        for split, prefix in (("validation", "validation_"), ("test", "")):
            y = is_oos[split]
            score = scores[split][column].to_numpy()
            # Read the threshold at full JSON precision, avoiding CSV parser
            # round-off when a cutoff equals one of the validation scores.
            decisions = json.loads((output / "frozen_validation_decisions.json").read_text(encoding="utf-8"))["decisions"]
            frozen = next(d for d in decisions if all(d[key] == row[key] for key in ("calibration", "score", "policy")))
            p = softmax(logits[split] / row["temperature"], axis=1)
            if row["score"] == "msp":
                recomputed = 1 - p.max(axis=1)
            elif row["score"] == "entropy":
                recomputed = -(p * np.log(p)).sum(axis=1)
            else:
                top = np.sort(p, axis=1)[:, -2:]
                recomputed = 1 - top[:, 1] + top[:, 0]
            np.testing.assert_allclose(score, recomputed, atol=1e-13, rtol=1e-12)
            # Exact exported numbers are authoritative for equality at a cutoff.
            # Load score CSV with round_trip to preserve float64 boundary values.
            exact_score = pd.read_csv(output / f"{split}_scores.csv", float_precision="round_trip")[column].to_numpy()
            rejected = exact_score >= frozen["threshold"]
            precision, recall, f1, _ = precision_recall_fscore_support(y, rejected, average="binary", zero_division=0)
            for name, value in (("oos_precision", precision), ("oos_recall", recall), ("oos_f1", f1), ("in_scope_acceptance", (~rejected[~y]).mean())):
                require_close(value, row[prefix + name], f"{split}:{column}:{name}")
            if split == "test":
                require_close(roc_auc_score(y, score), row["oos_auroc"], "AUROC")
                require_close(average_precision_score(y, score), row["oos_aupr_average_precision"], "AUPR")
            elif row["policy"] == "is_acceptance_95" and (~rejected[~y]).mean() < .95 - 1e-12:
                raise AssertionError("95% validation acceptance constraint violated")
        checked_rows += 1

    calibration = pd.read_csv(output / "calibration_metrics.csv")
    for row in calibration.to_dict("records"):
        split = row["split"]
        mask = ~is_oos[split]
        labels = metadata[split].loc[mask, "label_id"].to_numpy()
        p = softmax(logits[split][mask] / row["temperature"], axis=1)
        require_close(log_loss(labels, p, labels=np.arange(150)), row["nll"], "Calibration NLL")
        correct = p.argmax(axis=1) == labels
        confidence = p.max(axis=1)
        bins = np.digitize(confidence, np.linspace(0, 1, 16)[1:-1], right=False)
        ece = sum(abs(correct[bins == b].mean() - confidence[bins == b].mean()) * (bins == b).mean() for b in np.unique(bins))
        require_close(ece, row["ece_15_bins"], "Calibration ECE")

    # NLL derivative w.r.t. inverse temperature: E_p[z] - z_y.
    t = summary["temperature_fit"]["temperature"]
    mask = ~is_oos["validation"]
    z = logits["validation"][mask]
    y = metadata["validation"].loc[mask, "label_id"].to_numpy()
    gradient = float(((softmax(z / t, axis=1) * z).sum(axis=1) - z[np.arange(len(y)), y]).mean())
    if abs(gradient) > 1e-5:
        raise AssertionError(f"Temperature is not a stationary point: {gradient}")
    result = {"status": "passed", "verified_operating_points": checked_rows, "source_json_row_alignment": True, "input_hashes_unchanged": True, "independent_sklearn_metrics": True, "temperature_inverse_gradient": gradient}
    (output / "independent_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
