"""Evaluation-only duplicate sensitivity for the fixed Mahalanobis 95% policy.

Run only after all new-model test results have been frozen. The same original
test row IDs are excluded for every arm, seed and spelling condition. No model,
training data, distance statistic or threshold is changed. Uses only stdlib.
"""
from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import unicodedata

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent / "CLINC150_Team_20261019"
PRIOR = ROOT.parent / "BERT_Rejection_Study_20261019"
OUTPUT = ROOT / "outputs"
SEEDS = (42, 43, 44)
ARMS = ("original", "clean_continue", "typo_mix")
CONDITIONS = ("clean", "typo_light", "typo_medium")
VIEWS = ("official", "exclude_original_train_overlap")
POLICY = "is_acceptance_95"
SCORE = "mahalanobis"
METRICS = (
    "in_scope_false_rejection_rate", "in_scope_correct_and_accepted_rate",
    "closed_set_accuracy", "oos_f1", "oos_recall", "in_scope_acceptance",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def boolean(value):
    if value not in ("True", "False"):
        raise ValueError(f"Invalid stored Boolean: {value!r}")
    return value == "True"


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def overlap_rows(data, transform):
    lookup = defaultdict(list)
    for row_id, (text, label) in enumerate(data["train"]):
        lookup[transform(text)].append(dict(train_row_id=row_id, text=text, label=label))
    overlap = []
    for row_id, (text, label) in enumerate(data["test"] + data["oos_test"]):
        if transform(text) in lookup:
            overlap.append(dict(row_id=row_id, text=text, test_label=label,
                                original_train_matches=lookup[transform(text)]))
    return overlap


def evaluate_rows(rows, excluded):
    counts = dict(in_scope_n=0, oos_n=0, in_scope_false_rejections=0,
                  in_scope_correct_and_accepted=0, in_scope_wrong_and_accepted=0,
                  in_scope_correct_and_rejected=0, in_scope_wrong_and_rejected=0,
                  in_scope_closed_set_correct=0, oos_true_positive=0, oos_missed=0)
    decision = f"{SCORE}_{POLICY}_reject"
    for row in rows:
        if int(row["row_id"]) in excluded:
            continue
        reject = boolean(row[decision])
        if boolean(row["is_oos"]):
            counts["oos_n"] += 1
            counts["oos_true_positive" if reject else "oos_missed"] += 1
        else:
            counts["in_scope_n"] += 1
            correct = row["predicted_label"] == row["true_label"]
            counts["in_scope_closed_set_correct"] += int(correct)
            counts["in_scope_false_rejections"] += int(reject)
            key = "in_scope_" + ("correct" if correct else "wrong") + "_and_" + ("rejected" if reject else "accepted")
            counts[key] += 1
    n_is, n_oos = counts["in_scope_n"], counts["oos_n"]
    tp, fp = counts["oos_true_positive"], counts["in_scope_false_rejections"]
    assert n_is in (4500, 4498) and n_oos == 1000
    assert (counts["in_scope_correct_and_accepted"] + counts["in_scope_wrong_and_accepted"]
            + counts["in_scope_correct_and_rejected"] + counts["in_scope_wrong_and_rejected"]) == n_is
    return dict(**counts,
                in_scope_false_rejection_rate=fp / n_is,
                in_scope_correct_and_accepted_rate=counts["in_scope_correct_and_accepted"] / n_is,
                closed_set_accuracy=counts["in_scope_closed_set_correct"] / n_is,
                oos_f1=2 * tp / (n_oos + tp + fp),
                oos_recall=tp / n_oos,
                in_scope_acceptance=1 - fp / n_is)


def main():
    new_completion_path = OUTPUT / "TEST_COMPLETE.json"
    if not new_completion_path.exists():
        raise FileNotFoundError("New-model test results must be complete and frozen before this audit")
    csv_output = OUTPUT / "duplicate_sensitivity.csv"
    json_output = OUTPUT / "duplicate_sensitivity.json"
    if csv_output.exists() or json_output.exists():
        raise FileExistsError("Duplicate sensitivity already exists; refusing to overwrite")
    prior_output = PRIOR / "typo_stress_test/outputs"
    prior_completion_path = prior_output / "TEST_COMPLETE.json"
    new_completion = read_json(new_completion_path)
    prior_completion = read_json(prior_completion_path)
    new_freeze_path = ROOT / "INPUTS_FROZEN.json"
    new_freeze = read_json(new_freeze_path)
    assert digest(new_freeze_path) == new_completion["inputs_frozen_sha256"]
    development_path = OUTPUT / "DEVELOPMENT_FROZEN.json"
    development = read_json(development_path)
    assert digest(development_path) == new_completion["development_frozen_sha256"]
    raw_path = PROJECT / "data/raw/data_full.json"
    mapping_path = PROJECT / "outputs/bert/seed42/label_mapping.json"
    raw_key = raw_path.resolve().as_posix()
    assert digest(raw_path) == new_freeze["input_hashes"][raw_key]
    data = read_json(raw_path)
    official = data["test"] + data["oos_test"]
    mapping = read_json(mapping_path)["label_to_id"]
    id_to_label = {value: key for key, value in mapping.items()}
    exact = overlap_rows(data, lambda text: text)
    normalized_overlap = overlap_rows(data, normalized)
    excluded = {row["row_id"] for row in exact}
    assert excluded == {599, 938}, f"Unexpected original training/test overlap: {sorted(excluded)}"
    assert {row["row_id"] for row in normalized_overlap} == excluded
    inputs = {p.resolve().as_posix(): digest(p) for p in (
        Path(__file__), new_completion_path, prior_completion_path, new_freeze_path,
        development_path, raw_path, mapping_path)}
    records = []
    for arm in ARMS:
        for seed in SEEDS:
            folder = prior_output / f"seed{seed}" if arm == "original" else OUTPUT / arm / f"seed{seed}"
            threshold_folder = PRIOR / f"outputs/seed{seed}" if arm == "original" else folder
            threshold_path = threshold_folder / "frozen_thresholds.json"
            threshold_key = threshold_path.resolve().as_posix()
            threshold_hash = digest(threshold_path)
            expected_threshold_hash = (new_freeze["input_hashes"][threshold_key] if arm == "original"
                                       else development["development_hashes"][threshold_key])
            assert threshold_hash == expected_threshold_hash, f"Frozen threshold file changed: {threshold_path}"
            inputs[threshold_key] = threshold_hash
            threshold = float(read_json(threshold_path)[f"{SCORE}/{POLICY}"]["threshold"])
            for condition in CONDITIONS:
                path = folder / f"{condition}_predictions.csv"
                actual_hash = digest(path)
                expected_hash = (prior_completion["output_hashes"][path.relative_to(prior_output).as_posix()]
                                 if arm == "original" else new_completion["output_hashes"][path.resolve().as_posix()])
                assert actual_hash == expected_hash, f"Prediction file differs from completion manifest: {path}"
                inputs[path.resolve().as_posix()] = actual_hash
                rows = read_csv(path)
                assert len(rows) == 5500
                for i, row in enumerate(rows):
                    assert int(row["row_id"]) == i
                    assert [row["original_text"], row["true_label"]] == official[i]
                    assert row["condition"] == condition
                    assert boolean(row["is_oos"]) == (official[i][1] == "oos")
                    assert int(row["label_id"]) == mapping.get(official[i][1], -1)
                    assert id_to_label[int(row["predicted_label_id"])] == row["predicted_label"]
                    assert boolean(row[f"{SCORE}_{POLICY}_reject"]) == (float(row[f"{SCORE}_score"]) >= threshold)
                for view in VIEWS:
                    mask = set() if view == "official" else excluded
                    records.append(dict(arm=arm, seed=seed, condition=condition, view=view,
                                        score=SCORE, policy=POLICY, threshold=threshold,
                                        removed_n=len(mask), **evaluate_rows(rows, mask)))
    assert len(records) == 54
    groups = []
    for arm in ARMS:
        for condition in CONDITIONS:
            for view in VIEWS:
                selected = [r for r in records if (r["arm"], r["condition"], r["view"]) == (arm, condition, view)]
                assert {r["seed"] for r in selected} == set(SEEDS)
                result = dict(arm=arm, condition=condition, view=view, n_seeds=3)
                for metric in METRICS:
                    result[metric + "_mean"] = statistics.mean(r[metric] for r in selected)
                    result[metric + "_sd"] = statistics.stdev(r[metric] for r in selected)
                groups.append(result)
    lookup = {(r["arm"], r["condition"], r["view"]): r for r in groups}
    differences = []
    for reference in ("clean_continue", "original"):
        for condition in CONDITIONS:
            for metric in METRICS:
                deltas = {}
                for view in VIEWS:
                    value = lookup["typo_mix", condition, view][metric + "_mean"] - lookup[reference, condition, view][metric + "_mean"]
                    deltas[view] = value
                differences.append(dict(comparison="typo_mix_minus_" + reference, condition=condition,
                                        metric=metric, official_delta=deltas["official"],
                                        excluded_delta=deltas["exclude_original_train_overlap"],
                                        official_delta_pp=100 * deltas["official"],
                                        excluded_delta_pp=100 * deltas["exclude_original_train_overlap"],
                                        delta_change_pp_after_exclusion=100 * (deltas["exclude_original_train_overlap"] - deltas["official"])))
    for path, expected in inputs.items():
        assert digest(Path(path)) == expected, f"Input changed during sensitivity evaluation: {path}"
    with csv_output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = dict(completed_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                  kind="fixed_model_evaluation_only_original_training_overlap_sensitivity",
                  score=SCORE, policy=POLICY, seeds=list(SEEDS), arms=list(ARMS), conditions=list(CONDITIONS),
                  excluded_original_row_ids=sorted(excluded), exact_overlap_evidence=exact,
                  normalized_overlap_same_mask=True, evaluated_rows=len(records),
                  input_sha256=inputs, output_csv_sha256=digest(csv_output),
                  group_mean_and_sample_sd=groups, mean_method_differences=differences,
                  max_absolute_change_in_method_difference_pp=max(abs(r["delta_change_pp_after_exclusion"]) for r in differences),
                  limitations=[
                      "The identical original row-ID mask is applied to clean, light and medium for every model arm and seed.",
                      "Mixed continuation text has one rather than two exact clean-test overlaps, but initialization and clean-train distance statistics still used the original training data.",
                      "This is an evaluation-only exclusion; no retraining, covariance refitting, recalibration or threshold reselection occurred.",
                      "Three original train-validation overlaps remain; removing two test rows does not remove their model/threshold-selection effects.",
                      "Exact and normalized overlap exclusion does not eliminate semantic duplicates, pretraining contamination or prior test exposure.",
                      "Means are across the three fixed seeds; no new significance test or bootstrap is claimed by this sensitivity report.",
                  ])
    with json_output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Wrote {csv_output} and {json_output}; 54 fixed-system evaluations")
    print("Maximum absolute change to a method difference (percentage points):",
          report["max_absolute_change_in_method_difference_pp"])
    print(json.dumps(differences, indent=2))


if __name__ == "__main__":
    main()
