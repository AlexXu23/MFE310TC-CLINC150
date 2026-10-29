"""Read-only fixed-prediction sensitivity and rejection-transition diagnostics.

No experiment modules are imported, and no feature/model inference is performed.
The original-source duplicate exclusion is identical across arms/conditions.
Run --check-only to recompute and compare existing diagnostic files unchanged.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
AUG = ROOT.parent / "BERT_Typo_Augmentation_20261019"
PROJECT = ROOT.parent / "CLINC150_Team_20261019"
SEEDS = (42, 43, 44)
ARMS = ("clean_only", "clean_duplicate", "clean_plus_typo")
CONDITIONS = ("clean", "typo_light", "typo_medium")
POLICIES = ("max_oos_f1", "is_acceptance_95")
PRIMARY = "is_acceptance_95"
METRICS = ("in_scope_false_rejection_rate", "in_scope_correct_and_accepted_rate",
           "closed_set_accuracy", "oos_f1", "oos_recall", "in_scope_acceptance")
TRANSITIONS = ("accept_accept", "accept_reject", "reject_accept", "reject_reject")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def boolean(value):
    require(value in ("True", "False"), f"Invalid Boolean: {value!r}")
    return value == "True"


def rejection(row, policy=PRIMARY):
    return boolean(row[f"mahalanobis_{policy}_reject"])


def correctness(row):
    return row["predicted_label"] == row["true_label"]


def evaluate(rows, excluded, policy):
    counts = dict(in_scope_n=0, oos_n=0, in_scope_correct_and_accepted=0,
        in_scope_wrong_and_accepted=0, in_scope_correct_and_rejected=0,
        in_scope_wrong_and_rejected=0, oos_true_positive=0, oos_missed=0)
    for row in rows:
        if int(row["row_id"]) in excluded:
            continue
        reject = rejection(row, policy)
        if boolean(row["is_oos"]):
            counts["oos_n"] += 1
            counts["oos_true_positive" if reject else "oos_missed"] += 1
        else:
            counts["in_scope_n"] += 1
            key = "in_scope_" + ("correct" if correctness(row) else "wrong") + "_and_" + ("rejected" if reject else "accepted")
            counts[key] += 1
    n_is, n_oos = counts["in_scope_n"], counts["oos_n"]
    fp = counts["in_scope_correct_and_rejected"] + counts["in_scope_wrong_and_rejected"]
    tp = counts["oos_true_positive"]
    counts["in_scope_false_rejections"] = fp
    counts["in_scope_closed_set_correct"] = counts["in_scope_correct_and_accepted"] + counts["in_scope_correct_and_rejected"]
    require(n_is > 0 and n_oos > 0, "Empty scope after exclusion")
    require(sum(counts[k] for k in ("in_scope_correct_and_accepted", "in_scope_wrong_and_accepted",
        "in_scope_correct_and_rejected", "in_scope_wrong_and_rejected")) == n_is, "State sum mismatch")
    return dict(**counts, in_scope_false_rejection_rate=fp / n_is,
        in_scope_correct_and_accepted_rate=counts["in_scope_correct_and_accepted"] / n_is,
        closed_set_accuracy=counts["in_scope_closed_set_correct"] / n_is,
        oos_f1=2 * tp / (n_oos + tp + fp), oos_recall=tp / n_oos, in_scope_acceptance=1 - fp / n_is)


def summarize(records, group_keys, excluded_fields=()):
    groups = defaultdict(list)
    for row in records:
        groups[tuple(row[key] for key in group_keys)].append(row)
    summaries = []
    for key, rows in groups.items():
        require(len(rows) == 3 and {r["seed"] for r in rows} == set(SEEDS), "Missing seed in summary")
        group = dict(zip(group_keys, key))
        group["n_seeds"] = len(rows)
        for field in rows[0]:
            if field in (*group_keys, "seed", *excluded_fields):
                continue
            values = [r[field] for r in rows]
            if all(isinstance(value, (int, float)) for value in values):
                group[field + "_mean"] = statistics.mean(values)
                group[field + "_sd"] = statistics.stdev(values)
        summaries.append(group)
    return summaries


def compute():
    inputs = {}
    def remember(path):
        path = Path(path)
        inputs[path.resolve().as_posix()] = digest(path)
        return path
    completion = read_json(remember(OUTPUT / "TEST_COMPLETE.json"))
    freeze_path = remember(ROOT / "INPUTS_FROZEN.json")
    development_path = remember(OUTPUT / "DEVELOPMENT_FROZEN.json")
    require(digest(freeze_path) == completion["inputs_frozen_sha256"], "Input gate hash mismatch")
    require(digest(development_path) == completion["development_frozen_sha256"], "Development gate hash mismatch")
    development = read_json(development_path)
    audit = read_json(remember(ROOT / "data_audit.json"))
    raw_path = remember(PROJECT / "data/raw/data_full.json")
    raw = read_json(raw_path)
    require(digest(raw_path) == read_json(freeze_path)["input_hashes"][raw_path.as_posix()], "Official data hash mismatch")
    train_texts = {text for text, _ in raw["train"]}
    official = raw["test"] + raw["oos_test"]
    original_mask = {i for i, (text, _) in enumerate(official) if text in train_texts}
    require(original_mask == {599, 938}, "Original duplicate mask changed")
    audit_masks = audit["sensitivity_masks"]
    require(audit_masks["original_clean_train_exact_test_row_ids"] == sorted(original_mask), "Audit overlap mask mismatch")
    all_collision_mask = set(audit_masks["exact"]["union_with_original_overlap_row_ids"])
    views = {"official": set(), "exclude_original_train_overlap": original_mask}
    if all_collision_mask != original_mask:
        views["exclude_union_of_original_and_new_exact_collisions"] = all_collision_mask
    cached, thresholds, records = {}, {}, []
    for arm in ARMS:
        for seed in SEEDS:
            folder = OUTPUT / arm / f"seed{seed}"
            threshold_path = remember(folder / "frozen_thresholds.json")
            require(digest(threshold_path) == development["development_hashes"][threshold_path.as_posix()], "Threshold hash mismatch")
            thresholds[arm, seed] = read_json(threshold_path)
            for condition in CONDITIONS:
                path = remember(folder / f"{condition}_predictions.csv")
                require(digest(path) == completion["output_hashes"][path.as_posix()], "Frozen prediction hash mismatch")
                rows = read_csv(path)
                require(len(rows) == 5500, "Unexpected prediction row count")
                for i, row in enumerate(rows):
                    require(int(row["row_id"]) == i and [row["original_text"], row["true_label"]] == official[i], "Official row identity mismatch")
                    require(row["condition"] == condition and boolean(row["is_oos"]) == (official[i][1] == "oos"), "Condition/scope mismatch")
                    for policy in POLICIES:
                        cutoff = thresholds[arm, seed][f"mahalanobis/{policy}"]["threshold"]
                        require(rejection(row, policy) == (float(row["mahalanobis_score"]) >= cutoff), "Stored decision mismatch")
                cached[arm, seed, condition] = rows
                for policy in POLICIES:
                    for view, excluded in views.items():
                        records.append(dict(arm=arm, seed=seed, condition=condition, policy=policy,
                            view=view, score="mahalanobis", threshold=thresholds[arm, seed][f"mahalanobis/{policy}"]["threshold"],
                            removed_n=len(excluded), **evaluate(rows, excluded, policy)))
    for seed in SEEDS:
        for condition in CONDITIONS:
            reference = cached["clean_only", seed, condition]
            for arm in ARMS[1:]:
                for left, right in zip(reference, cached[arm, seed, condition]):
                    for field in ("row_id", "text", "true_label", "predicted_label", "predicted_label_id", "is_oos"):
                        require(left[field] == right[field], f"Classifier/input changed across fit arms: {field}")
    published_path = remember(OUTPUT / "metrics_by_seed.csv")
    require(digest(published_path) == completion["output_hashes"][published_path.as_posix()], "Published metrics hash mismatch")
    published = {(r["arm"], int(r["seed"]), r["condition"], r["policy"]): r for r in read_csv(published_path)}
    for row in records:
        if row["view"] == "official":
            saved = published[row["arm"], row["seed"], row["condition"], row["policy"]]
            for metric in METRICS:
                require(abs(row[metric] - float(saved[metric])) < 1e-12, f"Published metric mismatch: {metric}")
    groups = summarize(records, ("arm", "condition", "policy", "view"), ("score",))
    lookup = {(r["arm"], r["condition"], r["policy"], r["view"]): r for r in groups}
    differences = []
    for reference in ("clean_only", "clean_duplicate"):
        for condition in CONDITIONS:
            for policy in POLICIES:
                for metric in METRICS:
                    deltas = {view: lookup["clean_plus_typo", condition, policy, view][metric + "_mean"]
                              - lookup[reference, condition, policy, view][metric + "_mean"] for view in views}
                    differences.append(dict(comparison="clean_plus_typo_minus_" + reference,
                        condition=condition, policy=policy, metric=metric,
                        **{view + "_delta_pp": 100 * value for view, value in deltas.items()},
                        exclusion_change_pp=100 * (deltas["exclude_original_train_overlap"] - deltas["official"])))
    old_completion = read_json(remember(AUG / "outputs/TEST_COMPLETE.json"))
    old_cached = {}
    for seed in SEEDS:
        for condition in CONDITIONS[1:]:
            for arm in ("clean_continue", "typo_mix"):
                path = remember(AUG / f"outputs/{arm}/seed{seed}/{condition}_predictions.csv")
                require(digest(path) == old_completion["output_hashes"][path.as_posix()], "Prior augmentation prediction hash mismatch")
                old_cached[arm, seed, condition] = read_csv(path)
            for i, current in enumerate(cached["clean_only", seed, condition]):
                old_mix = old_cached["typo_mix", seed, condition][i]
                old_control = old_cached["clean_continue", seed, condition][i]
                for field in ("row_id", "text", "true_label", "is_oos"):
                    require(current[field] == old_mix[field] == old_control[field], "Prior/current row alignment mismatch")
                require(current["predicted_label"] == old_mix["predicted_label"], "Frozen augmented classifier differs from prior run")
                require(all(rejection(current, p) == rejection(old_mix, p) for p in POLICIES), "Copied clean-only rejection differs from prior run")
    transitions = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            target = cached["clean_plus_typo", seed, condition]
            subsets = {
                "all_in_scope": [i for i, row in enumerate(target) if not boolean(row["is_oos"])],
                "fixed_head_correct": [i for i, row in enumerate(target) if not boolean(row["is_oos"]) and correctness(row)],
                "fixed_head_wrong": [i for i, row in enumerate(target) if not boolean(row["is_oos"]) and not correctness(row)]}
            if condition != "clean":
                subsets["previously_newly_correct_from_augmentation"] = [i for i in subsets["fixed_head_correct"]
                    if not correctness(old_cached["clean_continue", seed, condition][i])]
            for reference in ("clean_only", "clean_duplicate"):
                source = cached[reference, seed, condition]
                for subset, indices in subsets.items():
                    counts = dict.fromkeys(TRANSITIONS, 0)
                    for i in indices:
                        state = ("reject" if rejection(source[i]) else "accept") + "_" + ("reject" if rejection(target[i]) else "accept")
                        counts[state] += 1
                    n = len(indices)
                    before_rejected = counts["reject_accept"] + counts["reject_reject"]
                    after_rejected = counts["accept_reject"] + counts["reject_reject"]
                    require(sum(counts.values()) == n, "Transition partition mismatch")
                    transitions.append(dict(seed=seed, condition=condition, policy=PRIMARY,
                        comparison="clean_plus_typo_minus_" + reference, subset=subset, n=n, **counts,
                        reference_rejected=before_rejected, mixed_rejected=after_rejected,
                        reference_accepted=n-before_rejected, mixed_accepted=n-after_rejected,
                        net_additional_accepts=before_rejected-after_rejected,
                        reference_rejection_rate=before_rejected/n, mixed_rejection_rate=after_rejected/n,
                        rejection_delta_pp=100*(after_rejected-before_rejected)/n,
                        fraction_of_previously_rejected_now_accepted=counts["reject_accept"]/before_rejected if before_rejected else None))
    transition_groups = summarize(transitions, ("condition", "policy", "comparison", "subset"))
    illustration = []
    for seed in SEEDS:
        for arm in ARMS:
            light, medium = cached[arm, seed, "typo_light"][82], cached[arm, seed, "typo_medium"][82]
            require(light["text"] == medium["text"], "Row82 light/medium are no longer identical")
            require(light["mahalanobis_score"] == medium["mahalanobis_score"], "Identical row82 texts scored differently")
            illustration.append(dict(seed=seed, arm=arm, row_id=82, original_text=light["original_text"],
                text=light["text"], true_label=light["true_label"], predicted_label=light["predicted_label"],
                score=float(light["mahalanobis_score"]), threshold=thresholds[arm, seed][f"mahalanobis/{PRIMARY}"]["threshold"],
                rejected=rejection(light), note="Light and medium are the same text, one illustration only"))
    for path, expected in inputs.items():
        require(digest(Path(path)) == expected, f"Input changed during calculation: {path}")
    metadata = dict(completed_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        input_sha256=inputs, script_path=Path(__file__).resolve().as_posix(), script_sha256=digest(Path(__file__)))
    sensitivity_report = dict(**metadata, kind="fixed_detector_evaluation_duplicate_sensitivity",
        original_train_test_excluded_row_ids=sorted(original_mask), new_exact_collision_union_row_ids=sorted(all_collision_mask),
        additional_collision_sensitivity_needed=all_collision_mask != original_mask,
        extra_sensitivity_note="No additional new exact collision: union mask equals original 599/938; no duplicate result table is needed" if all_collision_mask == original_mask else "Additional union mask evaluated in parallel across every arm and condition",
        records=records, group_mean_and_sample_sd=groups, mean_method_differences=differences,
        max_absolute_change_in_method_difference_pp=max(abs(r["exclusion_change_pp"]) for r in differences),
        checks=["All 54 official setting metrics independently reproduced", "All arms retain identical classifier predictions",
            "All source prediction/threshold/gate hashes checked", "Common original-row exclusion masks applied to every arm/condition", "Inputs unchanged"],
        limitations=["Evaluation-only exclusions do not undo prior training, covariance fit or validation threshold selection.",
            "Existing train-validation overlaps remain; no claim of eliminating all contamination.",
            "No new confidence interval or significance test is computed here."])
    mechanism_report = dict(**metadata, kind="fixed_classifier_rejection_transitions",
        policy=PRIMARY, transition_order="reference then clean_plus_typo", records=transitions,
        mean_across_fixed_seeds=transition_groups, row82_illustration=illustration,
        subgroup_definition="previously_newly_correct_from_augmentation: old clean_continue classification wrong, frozen typo_mix classification correct on identical perturbed IS row",
        interpretation="reject_accept is beneficial for fixed_head_correct but an additional wrong-class acceptance for fixed_head_wrong; both must be reported",
        limitations=["All subsets and transitions use the same fixed classifier within this experiment; labels/predictions do not change across detector arms.",
            "The prior newly-correct subset is an exploratory post hoc group defined by earlier classifier outcomes.",
            "Both means and covariance and validation thresholds differ across arms; transitions do not isolate their causal components.",
            "All diagnostic counts use the official 4500 IS rows; the separate sensitivity quantifies duplicate exclusion.",
            "The timer example was selected before this experiment; light and medium are one identical string, not independent examples.",
            "Raw distances from different fitted covariance matrices should not be compared as direct mechanistic evidence."])
    return records, sensitivity_report, transitions, mechanism_report


def process_files(csv_path, json_path, records, report, check_only):
    if check_only:
        before = {path: digest(path) for path in (csv_path, json_path)}
        expected_csv = [{k: "" if v is None else str(v) for k, v in row.items()} for row in records]
        require(read_csv(csv_path) == expected_csv, f"Existing CSV differs: {csv_path}")
        saved = read_json(json_path)
        saved.pop("completed_at")
        report = dict(report)
        report.pop("completed_at")
        require(saved == report, f"Existing JSON differs: {json_path}")
        require(all(digest(path) == value for path, value in before.items()), "Diagnostic changed during verification")
    else:
        if csv_path.exists() or json_path.exists():
            raise FileExistsError("Diagnostics exist; use --check-only to verify unchanged")
        with csv_path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        with json_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    records, sensitivity, transitions, mechanism = compute()
    process_files(OUTPUT / "duplicate_sensitivity.csv", OUTPUT / "duplicate_sensitivity.json", records, sensitivity, args.check_only)
    process_files(OUTPUT / "fixed_head_rejection_diagnostic.csv", OUTPUT / "fixed_head_rejection_diagnostic.json", transitions, mechanism, args.check_only)
    print("PASS:", len(records), "sensitivity evaluations and", len(transitions), "transition groups", "verified unchanged" if args.check_only else "saved")
    print("Maximum duplicate-exclusion change to a method difference (pp):", sensitivity["max_absolute_change_in_method_difference_pp"])
    selected = [r for r in mechanism["mean_across_fixed_seeds"] if r["comparison"] == "clean_plus_typo_minus_clean_only"]
    print(json.dumps(selected, indent=2))
    print("Row82:", json.dumps(mechanism["row82_illustration"], indent=2))


if __name__ == "__main__":
    main()
