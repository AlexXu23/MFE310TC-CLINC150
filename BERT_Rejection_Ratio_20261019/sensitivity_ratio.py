"""Independently recompute fixed-ratio metrics with uniform duplicate exclusion.

Only frozen prediction CSVs are used. No runner/report is imported and there is
no training, model inference, covariance fitting or threshold selection. Run
after TEST_COMPLETE, or --check-only to verify existing outputs unchanged.
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
PROJECT = ROOT.parent / "CLINC150_Team_20261019"
ARMS = ("clean_reference", "ratio_00", "ratio_10", "ratio_25", "ratio_50")
SEEDS = (42, 43, 44)
CONDITIONS = ("clean", "typo_light", "typo_medium")
POLICIES = ("max_oos_f1", "is_acceptance_95")
METRICS = ("in_scope_false_rejection_rate", "in_scope_correct_and_accepted_rate",
    "in_scope_wrong_and_accepted_rate", "closed_set_accuracy", "oos_f1", "oos_recall",
    "oos_false_acceptance_rate", "in_scope_acceptance")
COMPARISONS = tuple((candidate, reference) for candidate in ("ratio_10", "ratio_25")
                   for reference in ("ratio_00", "ratio_50", "clean_reference")) + (("ratio_50", "ratio_00"),)


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
    require(value in ("True", "False"), f"Unexpected stored Boolean: {value!r}")
    return value == "True"


def evaluate(rows, excluded, policy):
    counts = dict(in_scope_n=0, oos_n=0, in_scope_correct_and_accepted=0,
        in_scope_wrong_and_accepted=0, in_scope_correct_and_rejected=0,
        in_scope_wrong_and_rejected=0, oos_true_positive=0, oos_missed=0)
    for row in rows:
        if int(row["row_id"]) in excluded:
            continue
        reject = boolean(row[f"mahalanobis_{policy}_reject"])
        if boolean(row["is_oos"]):
            counts["oos_n"] += 1
            counts["oos_true_positive" if reject else "oos_missed"] += 1
        else:
            counts["in_scope_n"] += 1
            correct = row["predicted_label"] == row["true_label"]
            counts["in_scope_" + ("correct" if correct else "wrong") + "_and_" + ("rejected" if reject else "accepted")] += 1
    n_is, n_oos = counts["in_scope_n"], counts["oos_n"]
    require(n_is > 0 and n_oos > 0, "Empty scope after exclusions")
    fp = counts["in_scope_correct_and_rejected"] + counts["in_scope_wrong_and_rejected"]
    tp = counts["oos_true_positive"]
    counts["in_scope_false_rejections"] = fp
    correct = counts["in_scope_correct_and_accepted"] + counts["in_scope_correct_and_rejected"]
    require(fp + counts["in_scope_correct_and_accepted"] + counts["in_scope_wrong_and_accepted"] == n_is, "IS state sum mismatch")
    return dict(**counts, in_scope_false_rejection_rate=fp/n_is,
        in_scope_correct_and_accepted_rate=counts["in_scope_correct_and_accepted"]/n_is,
        in_scope_wrong_and_accepted_rate=counts["in_scope_wrong_and_accepted"]/n_is,
        closed_set_accuracy=correct/n_is, oos_f1=2*tp/(n_oos+tp+fp),
        oos_recall=tp/n_oos, oos_false_acceptance_rate=counts["oos_missed"]/n_oos,
        in_scope_acceptance=1-fp/n_is)


def compute():
    inputs = {}
    def remember(path):
        inputs[path.resolve().as_posix()] = digest(path)
        return path
    completion = read_json(remember(OUTPUT / "TEST_COMPLETE.json"))
    freeze_path = remember(ROOT / "INPUTS_FROZEN.json")
    development_path = remember(OUTPUT / "DEVELOPMENT_FROZEN.json")
    require(digest(freeze_path) == completion["inputs_frozen_sha256"], "Input gate hash mismatch")
    require(digest(development_path) == completion["development_frozen_sha256"], "Development gate hash mismatch")
    development, frozen = read_json(development_path), read_json(freeze_path)
    audit = read_json(remember(ROOT / "data_audit.json"))
    raw_path = remember(PROJECT / "data/raw/data_full.json")
    require(digest(raw_path) == frozen["input_hashes"][raw_path.as_posix()], "Official data hash mismatch")
    raw = read_json(raw_path)
    official = raw["test"] + raw["oos_test"]
    train_texts = {text for text, _ in raw["train"]}
    original_mask = {i for i, (text, _) in enumerate(official) if text in train_texts}
    require(original_mask == {599, 938}, "Unexpected inherited duplicate row IDs")
    require(audit["sensitivity_masks"]["inherited_original_train_test_row_ids"] == sorted(original_mask), "Audit original mask mismatch")
    union_mask = set(audit["sensitivity_masks"]["exact"]["union_with_inherited_overlap_row_ids"])
    views = {"official": set(), "exclude_original_train_overlap": original_mask}
    if union_mask != original_mask:
        views["exclude_union_of_original_and_new_exact_collisions"] = union_mask
    records, cached = [], {}
    for arm in ARMS:
        for seed in SEEDS:
            folder = OUTPUT / arm / f"seed{seed}"
            threshold_path = remember(folder / "frozen_thresholds.json")
            require(digest(threshold_path) == development["development_hashes"][threshold_path.as_posix()], "Frozen threshold hash mismatch")
            thresholds = read_json(threshold_path)
            for condition in CONDITIONS:
                path = remember(folder / f"{condition}_predictions.csv")
                require(digest(path) == completion["output_hashes"][path.as_posix()], "Frozen prediction hash mismatch")
                rows = read_csv(path)
                require(len(rows) == 5500, "Unexpected prediction count")
                for i, row in enumerate(rows):
                    require(int(row["row_id"]) == i and [row["original_text"], row["true_label"]] == official[i], "Original test identity mismatch")
                    require(row["condition"] == condition and boolean(row["is_oos"]) == (official[i][1] == "oos"), "Condition/scope mismatch")
                    for policy in POLICIES:
                        cutoff = thresholds[f"mahalanobis/{policy}"]["threshold"]
                        require(boolean(row[f"mahalanobis_{policy}_reject"]) == (float(row["mahalanobis_score"]) >= cutoff), "Stored rejection mismatch")
                cached[arm, seed, condition] = rows
                for policy in POLICIES:
                    for view, excluded in views.items():
                        records.append(dict(arm=arm, seed=seed, condition=condition, policy=policy,
                            score="mahalanobis", view=view, removed_n=len(excluded),
                            threshold=thresholds[f"mahalanobis/{policy}"]["threshold"], **evaluate(rows, excluded, policy)))
    for seed in SEEDS:
        for condition in CONDITIONS:
            reference = cached["clean_reference", seed, condition]
            for arm in ARMS[1:]:
                for left, right in zip(reference, cached[arm, seed, condition]):
                    for field in ("row_id", "text", "true_label", "predicted_label_id", "predicted_label", "is_oos"):
                        require(left[field] == right[field], f"Input/classifier varies across ratios: {field}")
    published_path = remember(OUTPUT / "metrics_by_seed.csv")
    require(digest(published_path) == completion["output_hashes"][published_path.as_posix()], "Published metrics hash mismatch")
    published = {(r["arm"], int(r["seed"]), r["condition"], r["policy"]): r for r in read_csv(published_path)}
    require(len(published) == 90, "Expected all 90 official settings")
    for row in records:
        if row["view"] == "official":
            saved = published[row["arm"], row["seed"], row["condition"], row["policy"]]
            for metric in METRICS:
                if metric in saved:
                    require(abs(row[metric]-float(saved[metric])) < 1e-12, f"Published metric mismatch: {metric}")
    grouped = defaultdict(list)
    for row in records:
        grouped[row["arm"], row["condition"], row["policy"], row["view"]].append(row)
    groups = []
    for key, rows in grouped.items():
        require(len(rows) == 3 and {r["seed"] for r in rows} == set(SEEDS), "Missing seed in summary")
        group = dict(zip(("arm", "condition", "policy", "view"), key))
        group["n_seeds"] = 3
        for field in rows[0]:
            if field in ("arm", "condition", "policy", "view", "seed", "score"):
                continue
            group[field+"_mean"] = statistics.mean(r[field] for r in rows)
            group[field+"_sd"] = statistics.stdev(r[field] for r in rows)
        groups.append(group)
    lookup = {(r["arm"], r["condition"], r["policy"], r["view"]): r for r in groups}
    differences = []
    for candidate, reference in COMPARISONS:
        for condition in CONDITIONS:
            for policy in POLICIES:
                for metric in METRICS:
                    deltas = {view: lookup[candidate, condition, policy, view][metric+"_mean"]
                              - lookup[reference, condition, policy, view][metric+"_mean"] for view in views}
                    differences.append(dict(candidate=candidate, reference=reference,
                        comparison=candidate+"_minus_"+reference, condition=condition, policy=policy, metric=metric,
                        **{view+"_delta_pp":100*value for view,value in deltas.items()},
                        exclusion_change_pp=100*(deltas["exclude_original_train_overlap"]-deltas["official"])))
    guardrails = []
    for candidate in ("ratio_10", "ratio_25", "ratio_50"):
        for reference in ("ratio_00", "clean_reference"):
            for view in views:
                delta = {metric:100*(lookup[candidate,"clean","is_acceptance_95",view][metric+"_mean"]
                         - lookup[reference,"clean","is_acceptance_95",view][metric+"_mean"])
                         for metric in ("oos_f1", "oos_recall", "in_scope_acceptance")}
                guardrails.append(dict(candidate=candidate, reference=reference, view=view,
                    clean_delta_pp=delta, strict_all_three_means_nondecreasing=all(value>=0 for value in delta.values())))
    for path, expected in inputs.items():
        require(digest(Path(path)) == expected, f"Input changed during evaluation: {path}")
    report = dict(completed_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        kind="independent_fixed_ratio_duplicate_sensitivity", input_sha256=inputs,
        script_path=Path(__file__).resolve().as_posix(), script_sha256=digest(Path(__file__)),
        excluded_original_row_ids=sorted(original_mask), additional_collision_union_row_ids=sorted(union_mask),
        no_additional_exact_collision_mask_needed=union_mask==original_mask,
        settings=len(records), group_mean_and_sample_sd=groups, mean_method_differences=differences,
        strict_descriptive_clean_guardrails=guardrails,
        max_absolute_change_in_method_difference_pp=max(abs(r["exclusion_change_pp"]) for r in differences),
        checks=["All 90 official settings independently recomputed", "All five arms retain identical classifier labels and test rows",
            "Every stored reject decision matches its frozen threshold", "All read prediction, threshold, metric and gate hashes match",
            "Same original-row mask is used for every arm, policy and condition", "Inputs unchanged during computation"],
        limitations=["Removing test rows does not undo inherited training/validation exposure or prior test inspection.",
            "No covariance refit, threshold fitting or additional model experiment was performed.",
            "Sample means describe the three fixed seeds; this sensitivity does not add confidence intervals or significance claims.",
            "The 15000-row clean_reference is historical; ratio_00 is the equal-30000-row zero-noise comparator.",
            "All predeclared candidates are retained; this script does not select a winning ratio on test data."])
    return records, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    records, report = compute()
    csv_path, json_path = OUTPUT / "duplicate_sensitivity.csv", OUTPUT / "duplicate_sensitivity.json"
    if args.check_only:
        before = {path:digest(path) for path in (csv_path,json_path)}
        require(read_csv(csv_path) == [{k:str(v) for k,v in r.items()} for r in records], "Existing sensitivity CSV differs")
        previous, comparable = read_json(json_path), dict(report)
        previous.pop("completed_at"); comparable.pop("completed_at")
        require(previous == comparable, "Existing sensitivity JSON differs")
        require(all(digest(path)==value for path,value in before.items()), "Existing outputs changed during read-only check")
        print("PASS: complete sensitivity CSV/JSON reproduced unchanged")
    else:
        if csv_path.exists() or json_path.exists():
            raise FileExistsError("Sensitivity already exists; use --check-only")
        with csv_path.open("x", encoding="utf-8", newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
        with json_path.open("x",encoding="utf-8") as stream:
            json.dump(report,stream,indent=2,ensure_ascii=False,allow_nan=False)
        print(f"Saved {len(records)} sensitivity evaluations to {csv_path} and {json_path}")
    print("Maximum exclusion change to a method difference (pp):",report["max_absolute_change_in_method_difference_pp"])
    print(json.dumps([r for r in report["strict_descriptive_clean_guardrails"] if r["view"]=="official"],indent=2))


if __name__ == "__main__":
    main()
