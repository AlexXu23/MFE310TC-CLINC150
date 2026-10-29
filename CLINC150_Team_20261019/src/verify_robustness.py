"""Independently verify typo edits, per-seed metrics, and sample-SD aggregates."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_recall_fscore_support, roc_auc_score


def close(actual, expected, description):
    if not np.isclose(actual, expected, rtol=1e-8, atol=1e-10):
        raise AssertionError(f"{description}: {actual} != {expected}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=".")
    parser.add_argument("--robustness", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--typo-data", required=True)
    args = parser.parse_args()
    project, root, aggregate_path, typo_path = Path(args.project), Path(args.robustness), Path(args.summary), Path(args.typo_data)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    for relative, expected_hash in summary["input_sha256"].items():
        if hashlib.sha256((project / relative).read_bytes()).hexdigest() != expected_hash:
            raise AssertionError(f"Changed frozen input: {relative}")
    original = json.loads((project / "data/raw/data_full.json").read_text(encoding="utf-8"))
    original_rows = original["test"] + original["oos_test"]
    typo_frames = {}
    for condition in ("clean", "typo_light", "typo_medium"):
        frame = pd.read_csv(typo_path / f"{condition}.csv", keep_default_na=False)
        if len(frame) != 5500 or frame[["original_text", "true_label"]].values.tolist() != original_rows:
            raise AssertionError("Typo labels or row order differ from source JSON")
        for row in frame.to_dict("records"):
            edits = json.loads(row["mutations_json"])
            changed = row["original_text"]
            if len(edits) != row["modified_tokens"]:
                raise AssertionError("Invalid mutation count")
            for edit in sorted(edits, key=lambda item: item["start"], reverse=True):
                token, replacement, index = edit["original"], edit["replacement"], edit["character_index"]
                if len(token) < 5 or not token.isalpha() or token[0] != replacement[0] or token[-1] != replacement[-1]:
                    raise AssertionError("Invalid eligible token or endpoint modification")
                if row["original_text"][edit["start"]:edit["end"]] != token:
                    raise AssertionError("Edit span does not match original text")
                if edit["operation"] == "delete":
                    expected = token[:index] + token[index + 1:]
                    assert 1 <= index < len(token) - 1
                elif edit["operation"] == "swap":
                    expected = token[:index] + token[index + 1] + token[index] + token[index + 2:]
                    assert 1 <= index < len(token) - 2
                else:
                    raise AssertionError("Unknown character operation")
                if replacement != expected or replacement == token:
                    raise AssertionError("Invalid character replacement")
                changed = changed[:edit["start"]] + replacement + changed[edit["end"]:]
            if changed != row["text"] or (changed != row["original_text"]) != row["changed"]:
                raise AssertionError("Stored mutated text differs from replayed edits")
        typo_frames[condition] = frame
    for row_id in range(5500):
        light = json.loads(typo_frames["typo_light"].iloc[row_id]["mutations_json"])
        medium = json.loads(typo_frames["typo_medium"].iloc[row_id]["mutations_json"])
        if light != medium[:len(light)]:
            raise AssertionError("Medium edits do not contain light edits")

    cls = pd.read_csv(root / "classification_by_seed.csv", float_precision="round_trip")
    reject = pd.read_csv(root / "rejection_by_seed.csv", float_precision="round_trip")
    cache = {}
    checked_logits = 0
    for row in cls.to_dict("records"):
        model, seed, condition = row["model"], row["seed"], row["condition"]
        key = (model, seed, condition)
        if key not in cache:
            model_path = root / (f"seed{seed}" if model == "bert" else "tfidf_lr")
            logits = np.load(model_path / f"{condition}_logits.npy", allow_pickle=False).astype(float)
            predictions = pd.read_csv(model_path / f"{condition}_predictions.csv", keep_default_na=False, float_precision="round_trip")
            if logits.shape != (5500, 150) or not np.isfinite(logits).all():
                raise AssertionError("Invalid robustness logits")
            if predictions[["row_id", "true_label", "text"]].values.tolist() != typo_frames[condition][["row_id", "true_label", "text"]].values.tolist():
                raise AssertionError("Prediction metadata not aligned with typo source")
            np.testing.assert_array_equal(predictions["predicted_label_id"], logits.argmax(axis=1))
            cache[key] = (logits, predictions)
            checked_logits += 1
        logits, predictions = cache[key]
        mask = ~predictions["is_oos"].to_numpy(dtype=bool)
        y = predictions["label_id"].to_numpy(dtype=int)
        if row["calibration"] == "raw":
            temperature = 1.0
        else:
            temperature = json.loads((project / f"outputs/oos/seed{seed}/temperature_fit.json").read_text(encoding="utf-8"))["temperature"]
        p = softmax(logits[mask] / temperature, axis=1)
        close(accuracy_score(y[mask], p.argmax(axis=1)), row["accuracy"], "Accuracy")
        close(f1_score(y[mask], p.argmax(axis=1), average="macro"), row["macro_f1"], "Macro-F1")
        close(log_loss(y[mask], p, labels=np.arange(150)), row["nll"], "NLL")
        close(p.max(axis=1).mean(), row["mean_confidence"], "Mean confidence")
        eligible = typo_frames["typo_light"]["changed"].to_numpy() & mask
        close((logits.argmax(axis=1)[eligible] == y[eligible]).mean(), row["eligible_only_accuracy"], "Changed-only accuracy")

    for row in reject.to_dict("records"):
        model, seed, condition = row["model"], row["seed"], row["condition"]
        logits, predictions = cache[(model, seed, condition)]
        decisions_file = project / f"outputs/oos/seed{seed}/frozen_validation_decisions.json" if model == "bert" else root / "tfidf_lr/baseline_validation_decisions.json"
        decisions = json.loads(decisions_file.read_text(encoding="utf-8"))["decisions"]
        decision = next(value for value in decisions if all(value[key] == row[key] for key in ("calibration", "score", "policy")))
        close(row["threshold"], decision["threshold"], "Frozen threshold")
        close(row["temperature"], decision["temperature"], "Frozen temperature")
        score = predictions[f"{row['calibration']}_{row['score']}_anomaly"].to_numpy()
        rejected = score >= decision["threshold"]
        np.testing.assert_array_equal(rejected, predictions[f"{row['calibration']}_{row['score']}_{row['policy']}_reject"])
        y = predictions["is_oos"].to_numpy(dtype=bool)
        labels = predictions["label_id"].to_numpy(dtype=int)
        predicted_ids = logits.argmax(axis=1)
        correct_accepted = (~y) & (~rejected) & (predicted_ids == labels)
        close(correct_accepted.sum() / (~y).sum(), row["in_scope_correct_and_accepted_rate"], "Correct and accepted rate")
        precision, recall, f1, _ = precision_recall_fscore_support(y, rejected, average="binary", zero_division=0)
        for name, value in (("oos_precision", precision), ("oos_recall", recall), ("oos_f1", f1), ("in_scope_false_rejection_rate", rejected[~y].mean()), ("oos_auroc", roc_auc_score(y, score))):
            close(value, row[name], name)

    aggregate_cls = pd.read_csv(aggregate_path / "typo_classification_mean_sd.csv")
    for row in aggregate_cls.to_dict("records"):
        group = cls[(cls["model"] == row["model"]) & (cls["calibration"] == row["calibration"]) & (cls["condition"] == row["condition"])]
        for metric in ("accuracy", "macro_f1", "macro_f1_change_pp"):
            close(np.mean(group[metric]), row[f"{metric}_mean"], "Aggregate mean")
            if len(group) > 1:
                close(np.std(group[metric], ddof=1), row[f"{metric}_sd"], "Sample SD")
            elif not pd.isna(row[f"{metric}_sd"]):
                raise AssertionError("A one-fit baseline must not claim across-seed variability")
    result = {"status": "passed", "typo_rows_replayed": 16500, "nested_light_medium": True, "logit_arrays_verified": checked_logits, "classification_rows_verified": len(cls), "rejection_rows_verified": len(reject), "fixed_clean_validation_thresholds": True, "sample_sd_ddof": 1}
    (aggregate_path / "independent_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
