"""Systematic error tables, fixed-checkpoint sensitivity, and sampled review cases.

This is post-hoc diagnosis, never model selection. Qualitative interpretations
live in a separate reviewed annotation file and are explicitly AI-assisted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from data_audit import normalize_text
from evaluate_oos import evaluate, file_hash, write_json


PRIMARY = "raw_msp_max_oos_f1_reject"
SECONDARY = "raw_entropy_is_acceptance_95_reject"


def system_errors(frame: pd.DataFrame, rejection_column: str = PRIMARY) -> np.ndarray:
    """Partition all rows into mutually exclusive system outcomes."""
    oos = frame["is_oos"].to_numpy(dtype=bool)
    reject = frame[rejection_column].to_numpy(dtype=bool)
    correct = frame["predicted_label_id"].to_numpy() == frame["label_id"].to_numpy()
    return np.select([oos & ~reject, ~oos & reject, ~oos & ~reject & ~correct],
                     ["missed_oos", "false_oos", "accepted_intent_error"], default="correct")


def choose_cases(frame: pd.DataFrame, pool: str, n: int) -> pd.DataFrame:
    """Stable uniform-like hash ordering within each declared diagnostic stratum."""
    frame = frame.copy()
    frame["sampling_key"] = frame["row_id"].map(lambda row: hashlib.sha256(f"20261019:{pool}:{row}".encode()).hexdigest())
    result = frame.sort_values("sampling_key").head(n).copy()
    if len(result) != n:
        raise AssertionError(f"Not enough cases in {pool}")
    result["review_pool"] = pool
    return result


def add_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach descriptive bins; bins are diagnostic, not tuning parameters."""
    frame = frame.copy()
    frame["word_count"] = frame["original_text"].str.split().str.len()
    frame["length_bin"] = pd.cut(frame["word_count"], [0, 4, 8, 12, np.inf], labels=["1-4", "5-8", "9-12", "13+"]).astype(str)
    frame["confidence_bin"] = pd.cut(frame["raw_confidence"], [-np.inf, .2, .4, .6, .8, np.inf], labels=["0-.2", ".2-.4", ".4-.6", ".6-.8", ".8-1"]).astype(str)
    frame["mutation_type"] = frame["mutations_json"].map(lambda value: "+".join(sorted(item["operation"] for item in json.loads(value))) or "unchanged")
    frame["closed_error"] = ~frame["intent_prediction_correct"].astype(bool) & ~frame["is_oos"].astype(bool)
    frame["system_error_type"] = system_errors(frame)
    return frame


def run(project: Path, output: Path) -> dict:
    """Process stored predictions, retaining all model/seed/condition identifiers."""
    output.mkdir(parents=True, exist_ok=True)
    base = project / "outputs/robustness/multiseed"
    audit_rows = pd.read_csv(project / "outputs/data_audit/row_audit.csv", keep_default_na=False)
    previous = audit_rows[audit_rows["split"].isin(["train", "val"])]
    test = audit_rows[audit_rows["split"] == "test"]
    # Canonical order is in-scope test, followed by OOS test, not alphabetical.
    test = pd.concat([test[test["source_split"] == "test"], test[test["source_split"] == "oos_test"]], ignore_index=True)
    masks = {"official": np.ones(5500, dtype=bool),
             "exclude_exact_earlier_overlap": ~test["text"].isin(set(previous["text"])).to_numpy(),
             "exclude_normalized_earlier_overlap": ~test["normalized_text"].isin(set(previous["normalized_text"])).to_numpy()}
    for name, mask in masks.items():
        pd.DataFrame({"row_id": np.arange(5500), "keep": mask}).to_csv(output / f"sensitivity_mask_{name}.csv", index=False)
    per_intent, confusions, breakdowns, system_counts, sensitivity, transitions, inputs = [], [], [], [], [], [], {}
    frames = {}
    mapping = json.loads((project / "outputs/bert/seed42/label_mapping.json").read_text(encoding="utf-8"))["label_to_id"]
    names = sorted(mapping, key=mapping.get)
    for model, seed, directory in [("bert", seed, f"seed{seed}") for seed in (42, 43, 44)] + [("tfidf_lr", 42, "tfidf_lr")]:
        clean = None
        for condition in ("clean", "typo_light", "typo_medium"):
            path = base / directory / f"{condition}_predictions.csv"
            frame = add_groups(pd.read_csv(path, keep_default_na=False))
            assert frame["row_id"].tolist() == list(range(5500))
            assert frame["original_text"].tolist() == test["text"].tolist()
            assert frame["true_label"].tolist() == test["label"].tolist()
            frames[(model, seed, condition)] = frame
            inputs[str(path.relative_to(project))] = file_hash(path)
            identity = {"model": model, "seed": seed, "condition": condition}
            ins = frame[~frame["is_oos"]]
            report = classification_report(ins["true_label"], ins["predicted_label"], labels=names, output_dict=True, zero_division=0)
            per_intent.extend({**identity, "intent": label, **report[label]} for label in names)
            pairs = ins[ins["closed_error"]].groupby(["true_label", "predicted_label"]).size()
            confusions.extend({**identity, "true_label": true, "predicted_label": predicted, "count": int(count), "true_class_n": 30, "rate": count / 30} for (true, predicted), count in pairs.items())
            for policy_column in (PRIMARY, SECONDARY):
                error = system_errors(frame, policy_column)
                system_counts.extend({**identity, "policy_column": policy_column, "outcome": outcome, "n": int((error == outcome).sum()), "total_n": len(frame)} for outcome in ("correct", "false_oos", "missed_oos", "accepted_intent_error"))
            for grouping in ("length_bin", "confidence_bin", "mutation_type", "modified_tokens"):
                for (scope, group), subset in frame.groupby(["is_oos", grouping], sort=True):
                    correct = subset["intent_prediction_correct"].to_numpy(dtype=bool)
                    reject = subset[PRIMARY].to_numpy(dtype=bool)
                    breakdowns.append({**identity, "scope": "oos" if scope else "in_scope", "grouping": grouping, "group": str(group), "n": len(subset),
                                       "closed_error_rate": float((~correct).mean()) if not scope else None,
                                       "rejection_rate": float(reject.mean()), "system_error_rate": float((subset["system_error_type"] != "correct").mean()),
                                       "mean_raw_confidence": float(subset["raw_confidence"].mean())})
            if condition == "clean":
                clean = frame
                decisions_path = project / (f"outputs/oos/seed{seed}/frozen_validation_decisions.json" if model == "bert" else "outputs/robustness/multiseed/tfidf_lr/baseline_validation_decisions.json")
                decisions = json.loads(decisions_path.read_text(encoding="utf-8"))["decisions"]
                for view, keep in masks.items():
                    scope_flags = frame["is_oos"].to_numpy(dtype=bool)[keep]
                    y = frame["label_id"].to_numpy()[keep]
                    pred = frame["predicted_label_id"].to_numpy()[keep]
                    classification = {"accuracy": accuracy_score(y[~scope_flags], pred[~scope_flags]),
                                      "macro_f1": f1_score(y[~scope_flags], pred[~scope_flags], labels=np.arange(150), average="macro", zero_division=0)}
                    for decision in decisions:
                        if decision["score"] != "msp" or decision["policy"] != "max_oos_f1":
                            continue
                        cal = decision["calibration"]
                        values = frame[f"{cal}_msp_anomaly"].to_numpy()[keep]
                        metrics = evaluate(values, scope_flags, y, pred, decision["threshold"])
                        sensitivity.append({**identity, "view": view, "calibration": cal, "removed_n": int((~keep).sum()), "in_scope_n": int((~scope_flags).sum()), "oos_n": int(scope_flags.sum()), **classification, **metrics})
            else:
                ins_mask = ~frame["is_oos"].to_numpy(dtype=bool)
                was_correct = clean["intent_prediction_correct"].to_numpy(dtype=bool)
                now_correct = frame["intent_prediction_correct"].to_numpy(dtype=bool)
                transitions.append({**identity, "in_scope_n": 4500,
                                    "clean_correct_to_wrong": int((ins_mask & was_correct & ~now_correct).sum()),
                                    "clean_wrong_to_correct": int((ins_mask & ~was_correct & now_correct).sum()),
                                    "unchanged_wrong": int((ins_mask & ~was_correct & ~now_correct).sum()),
                                    "unchanged_correct": int((ins_mask & was_correct & now_correct).sum())})
    intent_table = pd.DataFrame(per_intent)
    intent_table.to_csv(output / "per_intent_by_seed_condition.csv", index=False)
    intent_mean = intent_table.groupby(["model", "condition", "intent"]).agg(f1_mean=("f1-score", "mean"), f1_sd=("f1-score", "std"), recall_mean=("recall", "mean"), precision_mean=("precision", "mean"), n_runs=("seed", "size")).reset_index()
    intent_mean.to_csv(output / "per_intent_mean_sd.csv", index=False)
    confusion = pd.DataFrame(confusions)
    confusion.to_csv(output / "confusions_by_seed_condition.csv", index=False)
    grouped = confusion.groupby(["model", "condition", "true_label", "predicted_label"])["count"].sum().reset_index()
    grouped["true_support_across_runs"] = np.where(grouped["model"] == "bert", 90, 30)
    grouped["rate_per_true_class"] = grouped["count"] / grouped["true_support_across_runs"]
    grouped.sort_values("count", ascending=False).to_csv(output / "confusions_pooled.csv", index=False)
    pd.DataFrame(breakdowns).to_csv(output / "group_breakdowns.csv", index=False)
    pd.DataFrame(system_counts).to_csv(output / "system_outcomes.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(output / "duplicate_sensitivity.csv", index=False)
    pd.DataFrame(transitions).to_csv(output / "typo_transitions.csv", index=False)
    reference = frames[("bert", 42, "clean")].copy()
    for seed in (42, 43, 44):
        other = frames[("bert", seed, "clean")]
        reference[f"seed{seed}_prediction"] = other["predicted_label"]
        reference[f"seed{seed}_system_error_type"] = other["system_error_type"]
    reference["bert_clean_closed_error_seed_count"] = sum(frames[("bert", seed, "clean")]["closed_error"].astype(int) for seed in (42, 43, 44))
    reference["baseline_prediction"] = frames[("tfidf_lr", 42, "clean")]["predicted_label"]
    reference["baseline_correct"] = frames[("tfidf_lr", 42, "clean")]["intent_prediction_correct"]
    reference.to_csv(output / "clean_cross_seed_diagnostics.csv", index=False)
    samples = [choose_cases(reference[reference["system_error_type"] == pool], pool, 20) for pool in ("accepted_intent_error", "false_oos", "missed_oos")]
    noisy = frames[("bert", 42, "typo_medium")].copy()
    noisy["clean_prediction"] = reference["predicted_label"]
    noisy["baseline_prediction"] = frames[("tfidf_lr", 42, "typo_medium")]["predicted_label"]
    noisy["baseline_correct"] = frames[("tfidf_lr", 42, "typo_medium")]["intent_prediction_correct"]
    eligible = ~noisy["is_oos"] & reference["intent_prediction_correct"] & ~noisy["intent_prediction_correct"]
    samples.append(choose_cases(noisy[eligible], "typo_induced", 12))
    review = pd.concat(samples, ignore_index=True)
    review["case_id"] = [f"{row.review_pool}:{row.row_id}" for row in review.itertuples()]
    review["reviewer"] = "AI-assisted analysis; human confirmation pending"
    review.to_csv(output / "sampled_review_cases.csv", index=False)
    display_columns = ["case_id", "original_text", "text", "true_label", "predicted_label", "baseline_prediction", "raw_confidence", "temperature_scaled_confidence"]
    review[display_columns].to_json(output / "review_cases_readable.json", orient="records", indent=2, force_ascii=False)
    error_seed_counts = reference.loc[~reference["is_oos"], "bert_clean_closed_error_seed_count"].value_counts().reindex(range(4), fill_value=0)
    write_json(output / "protocol.json", {
        "primary_diagnostic_policy": PRIMARY, "secondary_diagnostic_policy": SECONDARY,
        "case_sampling": "Seed 42 fixed before linguistic review; SHA256 ordering with key 20261019 within 3 disjoint system-error pools, 20 cases each; 12 additional clean-correct/medium-wrong cases.",
        "scope": "All 12 model/condition prediction files analyzed; qualitative sample is not a prevalence estimate.",
        "human_review_status": "Pending; do not claim independent human annotation, agreement statistics or student authorship of AI draft interpretations.",
        "sensitivity": "Evaluation-only exclusions of test text already present in train OR validation (including OOS); no source edits, retraining, recalibration or threshold reselection.",
        "interpretation_limits": "Associations and plausible linguistic explanations are not causal proof or attribution. No data/model changes may be selected using these test errors.",
        "input_sha256": inputs})
    summary = {"prediction_files": len(inputs), "per_intent_rows": len(intent_table), "review_cases": len(review),
               "clean_closed_error_seed_count_distribution": {str(k): int(v) for k, v in error_seed_counts.items()},
               "sensitivity_removed": {name: int((~mask).sum()) for name, mask in masks.items()}}
    # Preserve both historical threshold conventions instead of silently mixing
    # early baseline numbers with the later unified anomaly-score evaluation.
    import joblib
    from common import load_dataset
    historical = json.loads((project / "outputs/baseline/metrics.json").read_text(encoding="utf-8"))
    baseline_decisions = json.loads((base / "tfidf_lr/baseline_validation_decisions.json").read_text(encoding="utf-8"))["decisions"]
    current = next(d for d in baseline_decisions if d["score"] == "msp" and d["policy"] == "max_oos_f1")
    baseline_model = joblib.load(base / "tfidf_lr/baseline_model.joblib")
    raw_data = load_dataset(project / "data/raw/data_full.json")
    val_probability = baseline_model.predict_proba([text for text, _ in raw_data["val"] + raw_data["oos_val"]])
    old_val = val_probability.max(axis=1) < historical["selected_threshold_from_validation"]
    log_p = np.log(val_probability)
    from evaluate_oos import anomaly_scores
    new_val = anomaly_scores(log_p)["msp"] >= current["threshold"]
    baseline_frame = frames[("tfidf_lr", 42, "clean")]
    old_test = baseline_frame["raw_confidence"].to_numpy() < historical["selected_threshold_from_validation"]
    new_test = baseline_frame[PRIMARY].to_numpy(dtype=bool)
    write_json(output / "baseline_threshold_convention.json", {
        "historical_reject_rule": "MSP < tau at an observed validation probability",
        "historical_tau": historical["selected_threshold_from_validation"],
        "historical_test_oos_f1": historical["oos_test"]["oos_f1"],
        "unified_reject_rule": "1-MSP >= anomaly threshold at an observed validation anomaly",
        "unified_anomaly_threshold": current["threshold"],
        "validation_decision_disagreements": int((old_val != new_val).sum()),
        "test_decision_disagreements": int((old_test != new_test).sum()),
        "changed_test_row_ids": baseline_frame.loc[old_test != new_test, "row_id"].tolist(),
        "explanation": "Different endpoints of an equivalent validation decision interval; a test score can fall inside the interval. Neither threshold uses test labels. Current error/sensitivity tables use the frozen unified robustness convention; historical baseline artifacts remain unchanged."})
    write_json(output / "summary.json", summary)
    top = grouped[(grouped["model"] == "bert") & (grouped["condition"] == "clean")].nlargest(10, "count").sort_values("count")
    bottom = intent_mean[(intent_mean["model"] == "bert") & (intent_mean["condition"] == "clean")].nsmallest(10, "f1_mean").sort_values("f1_mean")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].barh(top["true_label"] + " → " + top["predicted_label"], top["count"], color="#286694")
    axes[0].set(xlabel="Error events across three seeds (not unique texts)", title="Top clean BERT intent confusions")
    axes[1].barh(bottom["intent"], bottom["f1_mean"] * 100, xerr=bottom["f1_sd"] * 100, color="#ce6238")
    axes[1].set(xlabel="In-scope F1 (%), mean ± sample SD", title="Bottom ten clean BERT intents", xlim=(0, 100))
    fig.tight_layout()
    fig.savefig(output / "intent_error_overview.png", dpi=180)
    plt.close(fig)
    return summary


def main() -> None:
    """Generate diagnosis artifacts without touching original predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("outputs/error_analysis"))
    args = parser.parse_args()
    print(json.dumps(run(args.project, args.output), indent=2))


if __name__ == "__main__":
    main()
