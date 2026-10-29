"""TF-IDF + Logistic Regression baseline with validation-tuned OOS rejection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from common import load_dataset, validate_full_dataset


def unpack(rows):
    return [row[0] for row in rows], [row[1] for row in rows]


def oos_metrics(y_is_oos: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = scores < threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_is_oos, predicted, average="binary", zero_division=0
    )
    return {
        "threshold": float(threshold),
        "oos_precision": float(precision),
        "oos_recall": float(recall),
        "oos_f1": float(f1),
        "oos_auroc": float(roc_auc_score(y_is_oos, -scores)),
        "in_scope_acceptance": float((~predicted[~y_is_oos]).mean()),
    }


def threshold_curve(y_is_oos: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    candidates = np.unique(np.r_[0.0, scores, 1.0])
    rows = [oos_metrics(y_is_oos, scores, value) for value in candidates]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.data)
    validation = validate_full_dataset(dataset)

    train_x, train_y = unpack(dataset["train"])
    val_x, val_y = unpack(dataset["val"])
    test_x, test_y = unpack(dataset["test"])
    oos_val_x, _ = unpack(dataset["oos_val"])
    oos_test_x, _ = unpack(dataset["oos_test"])

    candidate_cs = [0.5, 1.0, 2.0, 4.0]
    tuning_rows = []
    for c_value in candidate_cs:
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
                (
                    "classifier",
                    LogisticRegression(
                        C=c_value,
                        max_iter=1_000,
                        solver="lbfgs",
                        random_state=args.seed,
                    ),
                ),
            ]
        )
        pipeline.fit(train_x, train_y)
        val_prediction = pipeline.predict(val_x)
        report = classification_report(val_y, val_prediction, output_dict=True, zero_division=0)
        tuning_rows.append(
            {
                "C": c_value,
                "val_accuracy": accuracy_score(val_y, val_prediction),
                "val_macro_f1": report["macro avg"]["f1-score"],
            }
        )

    tuning = pd.DataFrame(tuning_rows).sort_values(
        ["val_macro_f1", "val_accuracy", "C"], ascending=[False, False, True]
    )
    best_c = float(tuning.iloc[0]["C"])
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
            (
                "classifier",
                LogisticRegression(
                    C=best_c,
                    max_iter=1_000,
                    solver="lbfgs",
                    random_state=args.seed,
                ),
            ),
        ]
    )
    model.fit(train_x, train_y)

    test_prediction = model.predict(test_x)
    test_probability = model.predict_proba(test_x)
    test_report = classification_report(test_y, test_prediction, output_dict=True, zero_division=0)

    val_all_x = val_x + oos_val_x
    val_is_oos = np.array([False] * len(val_x) + [True] * len(oos_val_x))
    val_msp = model.predict_proba(val_all_x).max(axis=1)
    curve = threshold_curve(val_is_oos, val_msp)
    best_threshold_row = curve.sort_values(
        ["oos_f1", "in_scope_acceptance", "threshold"], ascending=[False, False, True]
    ).iloc[0]
    threshold = float(best_threshold_row["threshold"])

    test_all_x = test_x + oos_test_x
    test_is_oos = np.array([False] * len(test_x) + [True] * len(oos_test_x))
    test_all_probability = model.predict_proba(test_all_x)
    test_msp = test_all_probability.max(axis=1)
    final_oos = oos_metrics(test_is_oos, test_msp, threshold)

    accepted_known = test_msp[: len(test_x)] >= threshold
    predicted_all = model.classes_[np.argmax(test_all_probability, axis=1)].astype(object)
    predicted_all[test_msp < threshold] = "oos"
    true_all = np.array(test_y + ["oos"] * len(oos_test_x), dtype=object)
    all_report = classification_report(true_all, predicted_all, output_dict=True, zero_division=0)

    metrics = {
        "dataset": validation,
        "seed": args.seed,
        "selected_C": best_c,
        "selected_threshold_from_validation": threshold,
        "in_scope_test": {
            "accuracy_before_rejection": float(accuracy_score(test_y, test_prediction)),
            "macro_f1_before_rejection": float(test_report["macro avg"]["f1-score"]),
            "acceptance_after_rejection": float(accepted_known.mean()),
            "accepted_accuracy": float(
                accuracy_score(np.array(test_y)[accepted_known], test_prediction[accepted_known])
            ) if accepted_known.any() else None,
        },
        "oos_test": final_oos,
        "combined_151_label_test": {
            "accuracy": float(accuracy_score(true_all, predicted_all)),
            "macro_f1": float(all_report["macro avg"]["f1-score"]),
        },
    }

    tuning.to_csv(output / "c_tuning.csv", index=False, float_format="%.6f")
    curve.to_csv(output / "validation_threshold_curve.csv", index=False, float_format="%.8f")
    pd.DataFrame(test_report).T.to_csv(output / "in_scope_classification_report.csv")
    pd.DataFrame(all_report).T.to_csv(output / "combined_classification_report.csv")

    labels = validation["in_scope_intents"]
    matrix = confusion_matrix(test_y, test_prediction, labels=labels)
    confusions = []
    for i, true_label in enumerate(labels):
        for j, predicted_label in enumerate(labels):
            if i != j and matrix[i, j] > 0:
                confusions.append(
                    {"true_intent": true_label, "predicted_intent": predicted_label, "count": int(matrix[i, j])}
                )
    pd.DataFrame(confusions).sort_values("count", ascending=False).to_csv(
        output / "intent_confusions.csv", index=False
    )
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
