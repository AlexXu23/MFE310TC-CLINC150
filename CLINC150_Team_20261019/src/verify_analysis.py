"""Check new analysis against original JSON/logits using independent arithmetic.

Does not call the analysis metric implementations or rewrite experimental
inputs. This verifies outputs, not linguistic interpretations or human review.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path: Path) -> str:
    """Hash bytes for the immutable-input and derived-artifact audit trail."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probabilities(z: np.ndarray, t: float = 1.) -> np.ndarray:
    """Stable softmax independent of the evaluation helper."""
    shifted = z.astype(float) / t
    shifted -= shifted.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    """150-class macro-F1 from integer count arrays, without sklearn."""
    tp = np.bincount(y[y == pred], minlength=150)
    support = np.bincount(y, minlength=150)
    predicted = np.bincount(pred, minlength=150)
    return float(np.divide(2 * tp, support + predicted, out=np.zeros(150), where=support + predicted > 0).mean())


def main() -> None:
    """Verify all current extended analysis products from the project root."""
    project = Path(".")
    raw = json.loads((project / "data/raw/data_full.json").read_text(encoding="utf-8"))
    provenance = json.loads((project / "data/provenance/manifest.json").read_text(encoding="utf-8"))
    assert digest(project / "data/raw/data_full.json") == provenance["data_sha256"]
    for name, value in provenance["files_sha256"].items():
        assert digest(project / "data/provenance" / name) == value
    expected = raw["test"] + raw["oos_test"]
    test_text = [row[0] for row in expected]
    test_labels = [row[1] for row in expected]
    earlier = raw["train"] + raw["oos_train"] + raw["val"] + raw["oos_val"]
    normalized = lambda s: " ".join(unicodedata.normalize("NFKC", s).casefold().split())
    earlier_exact = {row[0] for row in earlier}
    earlier_normal = {normalized(row[0]) for row in earlier}
    masks = {"official": np.ones(5500, dtype=bool),
             "exclude_exact_earlier_overlap": np.array([s not in earlier_exact for s in test_text]),
             "exclude_normalized_earlier_overlap": np.array([normalized(s) not in earlier_normal for s in test_text])}
    output = project / "outputs/error_analysis"
    for name, mask in masks.items():
        assert np.array_equal(mask, pd.read_csv(output / f"sensitivity_mask_{name}.csv")["keep"])
    per_intent = pd.read_csv(output / "per_intent_by_seed_condition.csv")
    sensitivity = pd.read_csv(output / "duplicate_sensitivity.csv")
    system = pd.read_csv(output / "system_outcomes.csv")
    prediction_files, rejection_arrays, intent_rows, sensitivity_rows = 0, 0, 0, 0
    mapping = json.loads((project / "outputs/bert/seed42/label_mapping.json").read_text(encoding="utf-8"))["label_to_id"]
    target = np.array([mapping.get(label, -1) for label in test_labels])
    ins = target >= 0
    frame_lookup = {}
    for model, seed, folder in [("bert", seed, f"seed{seed}") for seed in (42, 43, 44)] + [("tfidf_lr", 42, "tfidf_lr")]:
        base = project / "outputs/robustness/multiseed" / folder
        decisions_path = project / f"outputs/oos/seed{seed}/frozen_validation_decisions.json" if model == "bert" else base / "baseline_validation_decisions.json"
        decisions = json.loads(decisions_path.read_text(encoding="utf-8"))["decisions"]
        for condition in ("clean", "typo_light", "typo_medium"):
            frame = pd.read_csv(base / f"{condition}_predictions.csv", keep_default_na=False)
            frame_lookup[(model, seed, condition)] = frame
            z = np.load(base / f"{condition}_logits.npy", allow_pickle=False)
            pred = z.argmax(axis=1)
            assert frame["original_text"].tolist() == test_text
            assert frame["true_label"].tolist() == test_labels
            assert np.array_equal(frame["label_id"], target)
            assert np.array_equal(frame["predicted_label_id"], pred)
            assert np.array_equal(frame["is_oos"], ~ins)
            assert np.array_equal(frame["intent_prediction_correct"], ins & (target == pred))
            tp = np.bincount(target[ins & (target == pred)], minlength=150)
            predicted_counts = np.bincount(pred[ins], minlength=150)
            for label, index in mapping.items():
                row = per_intent[(per_intent.model == model) & (per_intent.seed == seed) & (per_intent.condition == condition) & (per_intent.intent == label)].iloc[0]
                assert np.isclose(row["f1-score"], 2 * tp[index] / (30 + predicted_counts[index]), atol=1e-12)
                assert row["support"] == 30
                intent_rows += 1
            for d in decisions:
                p = probabilities(z, d["temperature"])
                top = np.sort(p, axis=1)[:, -2:]
                values = {"msp": 1 - p.max(axis=1), "entropy": -(p * np.log(p)).sum(axis=1), "margin": 1 - (top[:, 1] - top[:, 0])}[d["score"]]
                reject = values >= d["threshold"]
                col = f"{d['calibration']}_{d['score']}_{d['policy']}_reject"
                assert np.array_equal(reject, frame[col]), (folder, condition, col)
                rejection_arrays += 1
            for col in ("raw_msp_max_oos_f1_reject", "raw_entropy_is_acceptance_95_reject"):
                reject = frame[col].to_numpy(dtype=bool)
                counts = {"missed_oos": int((~ins & ~reject).sum()), "false_oos": int((ins & reject).sum()),
                          "accepted_intent_error": int((ins & ~reject & (target != pred)).sum()),
                          "correct": int(((~ins & reject) | (ins & ~reject & (target == pred))).sum())}
                actual = system[(system.model == model) & (system.seed == seed) & (system.condition == condition) & (system.policy_column == col)]
                assert dict(zip(actual.outcome, actual.n)) == counts
                assert sum(counts.values()) == 5500
            if condition == "clean":
                for view, mask in masks.items():
                    for calibration in (["raw", "temperature_scaled"] if model == "bert" else ["raw"]):
                        row = sensitivity[(sensitivity.model == model) & (sensitivity.seed == seed) & (sensitivity.view == view) & (sensitivity.calibration == calibration)].iloc[0]
                        assert row["removed_n"] == int((~mask).sum())
                        assert np.isclose(row["macro_f1"], macro_f1(target[ins & mask], pred[ins & mask]))
                        reject = frame[f"{calibration}_msp_max_oos_f1_reject"].to_numpy(dtype=bool)
                        tp_oos = int((~ins & mask & reject).sum())
                        fp_oos = int((ins & mask & reject).sum())
                        fn_oos = int((~ins & mask & ~reject).sum())
                        assert np.isclose(row["oos_f1"], 2 * tp_oos / (2 * tp_oos + fp_oos + fn_oos))
                        sensitivity_rows += 1
            prediction_files += 1
    review = pd.read_csv(output / "sampled_review_cases.csv", keep_default_na=False)
    calibration = pd.read_csv(project / "outputs/controlled_audit/calibration_paired_effects.csv")
    for row in calibration.to_dict("records"):
        seed = int(row["seed"])
        z = np.load(project / f"outputs/bert/seed{seed}/test_logits.npy", allow_pickle=False)[ins]
        y = target[ins]
        for arm, temperature in (("raw", 1.), ("scaled", row["temperature"])):
            p = probabilities(z, temperature)
            confidence = p.max(axis=1)
            correct = p.argmax(axis=1) == y
            ids = np.minimum((confidence * 15).astype(int), 14)
            bin_error = np.bincount(ids, weights=correct.astype(float) - confidence, minlength=15)
            expected_metrics = {"accuracy": correct.mean(), "macro_f1": macro_f1(y, p.argmax(axis=1)),
                                "nll": -np.log(p[np.arange(len(y)), y]).mean(),
                                "ece_15_bins": np.abs(bin_error).sum() / len(y),
                                "brier_sum_over_classes": ((p * p).sum(axis=1) - 2*p[np.arange(len(y)), y]+1).mean(),
                                "mean_confidence": confidence.mean()}
            for metric, value in expected_metrics.items():
                assert np.isclose(row[f"{arm}_{metric}"], value, atol=1e-12, rtol=0), (seed, arm, metric)
        for loss in ("nll", "brier", "ece"):
            assert row[f"{loss}_delta_ci_low"] <= row[f"{loss}_delta_ci_high"] < 0
    annotations = json.loads((project / "analysis/qualitative_annotations.json").read_text(encoding="utf-8"))
    annotation_ids = [row["case_id"] for row in annotations["annotations"]]
    assert len(annotation_ids) == len(set(annotation_ids)) == 72
    assert set(annotation_ids) == set(review["case_id"])
    for annotation in annotations["annotations"]:
        assert annotation["category"] in annotations["categories"]
        assert len(annotation["note"]) > 30
    for row in review.to_dict("records"):
        condition = "typo_medium" if row["review_pool"] == "typo_induced" else "clean"
        frame = frame_lookup[("bert", 42, condition)]
        original = frame.iloc[int(row["row_id"])]
        assert row["text"] == original["text"]
        assert row["true_label"] == original["true_label"]
        assert row["predicted_label"] == original["predicted_label"]
    # Also ensure all immutable inputs recorded in both new protocols still match.
    for path in (project / "outputs/controlled_audit/verification.json", output / "protocol.json"):
        protocol = json.loads(path.read_text(encoding="utf-8"))
        for name, value in protocol["input_sha256"].items():
            assert digest(project / name) == value
    result = {"status": "PASS", "dataset_and_provenance_hashes": "PASS", "original_prediction_files_checked": prediction_files,
              "independent_rejection_arrays_checked": rejection_arrays, "independent_per_intent_rows_checked": intent_rows,
              "independent_sensitivity_rows_checked": sensitivity_rows, "review_cases_traceable_to_source": len(review),
              "independent_calibration_rows_checked": 6,
              "qualitative_notes": "Coverage and provenance verified; linguistic judgments remain AI-assisted and require human confirmation.",
              "limitations": "Does not prove semantic correctness, fairness or causal model explanations; calibration bootstrap is conditional, not a repeated-training significance test."}
    target_path = project / "outputs/analysis_verification.json"
    target_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
