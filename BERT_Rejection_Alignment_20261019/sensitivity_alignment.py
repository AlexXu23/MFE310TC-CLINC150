"""Independent validation-IS threshold audit and fixed-score sensitivity.

Only existing CSVs and JSON manifests are read. This script does not train,
infer, refit distributions or use OOS scores to select aligned thresholds.
"""
from __future__ import annotations

import argparse
import bisect
from collections import defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "BERT_Rejection_Ratio_20261019"
PROJECT = ROOT.parent / "CLINC150_Team_20261019"
OUTPUT = ROOT / "outputs"
ARMS = ("clean_reference", "ratio_00", "ratio_10", "ratio_25", "ratio_50")
SEEDS = (42, 43, 44)
CONDITIONS = ("clean", "typo_light", "typo_medium")
TARGET_COUNT = 2850
OLD_POLICY = "is_acceptance_95"
NEW_POLICY = "aligned_is_acceptance_95"
POLICIES = (OLD_POLICY, NEW_POLICY)
METRICS = ("in_scope_false_rejection_rate", "in_scope_correct_and_accepted_rate",
    "in_scope_wrong_and_accepted_rate", "closed_set_accuracy", "oos_f1", "oos_recall",
    "oos_false_acceptance_rate", "in_scope_acceptance", "combined_151_accuracy")


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
    require(value in ("True", "False"), f"Invalid stored Boolean: {value!r}")
    return value == "True"


def independent_is_thresholds():
    """Recreate feasible counts with strict score<threshold, using IS only."""
    gate_path = PRIOR / "outputs/DEVELOPMENT_FROZEN.json"
    gate = read_json(gate_path)
    inputs = {gate_path.as_posix(): digest(gate_path)}
    models = {}
    common = set(range(3001))
    for arm in ARMS:
        for seed in SEEDS:
            folder = PRIOR / "outputs" / arm / f"seed{seed}"
            path = folder / "validation_predictions.csv"
            actual_hash = digest(path)
            require(actual_hash == gate["development_hashes"][path.as_posix()], "Source validation hash mismatch")
            inputs[path.as_posix()] = actual_hash
            rows = read_csv(path)
            scores = sorted(float(row["mahalanobis_score"]) for row in rows if not boolean(row["is_oos"]))
            require(len(scores) == 3000 and all(math.isfinite(value) for value in scores), "Expected 3000 finite IS scores")
            candidates = sorted(set(scores)) + [math.nextafter(max(scores), math.inf)]
            require(all(math.isfinite(value) for value in candidates), "Non-finite candidate threshold")
            count_to_threshold = {}
            for threshold in candidates:
                count = bisect.bisect_left(scores, threshold)
                count_to_threshold.setdefault(count, threshold)
            common.intersection_update(count_to_threshold)
            threshold_path = folder / "frozen_thresholds.json"
            require(digest(threshold_path) == gate["development_hashes"][threshold_path.as_posix()], "Source threshold hash mismatch")
            inputs[threshold_path.as_posix()] = digest(threshold_path)
            old_threshold = read_json(threshold_path)["mahalanobis/is_acceptance_95"]["threshold"]
            models[arm,seed] = dict(scores=scores, count_to_threshold=count_to_threshold,
                distinct_is_scores=len(set(scores)), old_threshold=old_threshold)
    feasible = sorted(count for count in common if count >= TARGET_COUNT)
    require(bool(feasible), "No common feasible count meets the target")
    actual_count = feasible[0]
    records = []
    for (arm, seed), model in models.items():
        threshold = model["count_to_threshold"][actual_count]
        actual = bisect.bisect_left(model["scores"],threshold)
        require(actual == actual_count, "Selected threshold does not realize common accepted count")
        old_count = bisect.bisect_left(model["scores"],model["old_threshold"])
        records.append(dict(arm=arm, seed=seed, validation_is_n=3000,
            target_accepted_count=TARGET_COUNT, actual_accepted_count=actual_count,
            target_acceptance=TARGET_COUNT/3000, actual_acceptance=actual_count/3000,
            distinct_is_scores=model["distinct_is_scores"], feasible_count_n=len(model["count_to_threshold"]),
            old_threshold=model["old_threshold"], aligned_threshold=threshold,
            old_accepted_count=old_count, old_acceptance=old_count/3000,
            selected_threshold_is_minimum_candidate=True))
    for path, expected in inputs.items():
        require(digest(Path(path)) == expected, f"Source changed during threshold audit: {path}")
    return records, dict(target_accepted_count=TARGET_COUNT, actual_common_accepted_count=actual_count,
        actual_common_acceptance=actual_count/3000, common_feasible_count_n=len(common),
        common_feasible_accepted_counts=sorted(common), input_sha256=inputs,
        selection_uses_validation_is_scores_only=True)


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
            counts["in_scope_"+("correct" if correct else "wrong")+"_and_"+("rejected" if reject else "accepted")] += 1
    n_is, n_oos = counts["in_scope_n"], counts["oos_n"]
    tp = counts["oos_true_positive"]
    fp = counts["in_scope_correct_and_rejected"] + counts["in_scope_wrong_and_rejected"]
    counts["in_scope_false_rejections"] = fp
    correct = counts["in_scope_correct_and_accepted"] + counts["in_scope_correct_and_rejected"]
    require(n_is in (4500,4498) and n_oos==1000, "Unexpected denominators")
    return dict(**counts, in_scope_false_rejection_rate=fp/n_is,
        in_scope_correct_and_accepted_rate=counts["in_scope_correct_and_accepted"]/n_is,
        in_scope_wrong_and_accepted_rate=counts["in_scope_wrong_and_accepted"]/n_is,
        closed_set_accuracy=correct/n_is, oos_f1=2*tp/(n_oos+tp+fp),
        oos_recall=tp/n_oos, oos_false_acceptance_rate=counts["oos_missed"]/n_oos,
        in_scope_acceptance=1-fp/n_is,
        combined_151_accuracy=(counts["in_scope_correct_and_accepted"]+tp)/(n_is+n_oos))


def compute():
    threshold_records, threshold_summary = independent_is_thresholds()
    inputs = dict(threshold_summary["input_sha256"])
    def remember(path):
        inputs[path.resolve().as_posix()] = digest(path)
        return path
    completion = read_json(remember(OUTPUT / "TEST_COMPLETE.json"))
    freeze_path = remember(ROOT / "INPUTS_FROZEN.json")
    development_path = remember(OUTPUT / "DEVELOPMENT_FROZEN.json")
    require(digest(freeze_path)==completion["inputs_frozen_sha256"], "Input gate mismatch")
    require(digest(development_path)==completion["development_frozen_sha256"], "Development gate mismatch")
    development, frozen = read_json(development_path), read_json(freeze_path)
    require(development["actual_common_accepted"]==threshold_summary["actual_common_accepted_count"], "Global accepted count mismatch")
    require(development["selection_used_oos"] is False, "New thresholds claim OOS-dependent selection")
    require(datetime.fromisoformat(development["frozen_at"]) <= datetime.fromisoformat(completion["started_at"]), "Test began before threshold freeze")
    summary_path = remember(OUTPUT / "alignment_summary.json")
    require(digest(summary_path)==development["development_hashes"][summary_path.as_posix()], "Alignment summary hash mismatch")
    summary = read_json(summary_path)
    require(summary["target_accepted"]==TARGET_COUNT and summary["actual_common_accepted"]==threshold_summary["actual_common_accepted_count"], "Alignment count summary differs")
    require(summary["common_feasible_counts"]==threshold_summary["common_feasible_accepted_counts"]
            and summary["selection_used_oos"] is False, "Alignment feasible set differs")
    summary_lookup = {(r["arm"],r["seed"]):r for r in summary["models"]}
    threshold_lookup, candidate_rows_checked = {}, 0
    for expected in threshold_records:
        arm, seed = expected["arm"], expected["seed"]
        folder = OUTPUT / arm / f"seed{seed}"
        path = remember(folder / "frozen_thresholds.json")
        require(digest(path)==development["development_hashes"][path.as_posix()], "Frozen threshold hash mismatch")
        thresholds = read_json(path)
        old, new = thresholds[f"mahalanobis/{OLD_POLICY}"], thresholds[f"mahalanobis/{NEW_POLICY}"]
        require(old["threshold"]==expected["old_threshold"] and new["threshold"]==expected["aligned_threshold"], "Independent threshold differs from frozen cutoff")
        require(new["target_accepted"]==TARGET_COUNT and new["actual_accepted"]==expected["actual_accepted_count"]
                and new["validation_is_rows"]==3000 and new["in_scope_acceptance"]==expected["actual_acceptance"], "New cutoff metadata mismatch")
        old_source = PRIOR / "outputs" / arm / f"seed{seed}" / "validation_predictions.csv"
        require(new["validation_predictions_sha256"]==digest(old_source), "Threshold validation binding mismatch")
        saved = summary_lookup[arm,seed]
        require(saved["original_accepted"]==expected["old_accepted_count"] and saved["original_threshold"]==expected["old_threshold"]
                and saved["aligned_threshold"]==expected["aligned_threshold"] and saved["unique_is_scores"]==expected["distinct_is_scores"], "Operating point summary differs")
        candidate_path = remember(folder / "validation_is_threshold_candidates.csv")
        require(digest(candidate_path)==development["development_hashes"][candidate_path.as_posix()], "Candidate file hash mismatch")
        scores = sorted(float(r["mahalanobis_score"]) for r in read_csv(old_source) if not boolean(r["is_oos"]))
        candidates = sorted(set(scores))+[math.nextafter(max(scores),math.inf)]
        published_candidates = read_csv(candidate_path)
        require(len(candidates)==len(published_candidates), "Candidate count differs")
        for threshold, candidate in zip(candidates,published_candidates):
            require(float(candidate["threshold"])==threshold and int(candidate["accepted_is"])==bisect.bisect_left(scores,threshold)
                    and int(candidate["validation_is_rows"])==3000, "Candidate accepted count differs")
            candidate_rows_checked += 1
        threshold_lookup[arm,seed] = thresholds
    raw_path = remember(PROJECT / "data/raw/data_full.json")
    require(digest(raw_path)==frozen["input_hashes"][raw_path.as_posix()], "Original data hash mismatch")
    raw = read_json(raw_path)
    official = raw["test"]+raw["oos_test"]
    train_texts = {text for text,_ in raw["train"]}
    mask = {i for i,(text,_) in enumerate(official) if text in train_texts}
    require(mask=={599,938}, "Unexpected duplicate mask")
    views = {"official":set(),"exclude_original_train_overlap":mask}
    old_completion = read_json(remember(PRIOR / "outputs/TEST_COMPLETE.json"))
    cached, records, transitions = {}, [], []
    for arm in ARMS:
        for seed in SEEDS:
            thresholds = threshold_lookup[arm,seed]
            old_cut = thresholds[f"mahalanobis/{OLD_POLICY}"]["threshold"]
            new_cut = thresholds[f"mahalanobis/{NEW_POLICY}"]["threshold"]
            for condition in CONDITIONS:
                path = remember(OUTPUT / arm / f"seed{seed}" / f"{condition}_predictions.csv")
                old_path = remember(PRIOR / "outputs" / arm / f"seed{seed}" / f"{condition}_predictions.csv")
                require(digest(path)==completion["output_hashes"][path.as_posix()], "New prediction hash mismatch")
                require(digest(old_path)==old_completion["output_hashes"][old_path.as_posix()], "Source prediction hash mismatch")
                rows, old_rows = read_csv(path), read_csv(old_path)
                require(len(rows)==len(old_rows)==5500, "Unexpected prediction row count")
                changes = dict(normal_correct_added_rejects=0,normal_wrong_added_rejects=0,oos_added_rejects=0,
                    normal_correct_removed_rejects=0,normal_wrong_removed_rejects=0,oos_removed_rejects=0)
                for i,(row,old_row) in enumerate(zip(rows,old_rows)):
                    require(int(row["row_id"])==i and [row["original_text"],row["true_label"]]==official[i], "Original identity mismatch")
                    for field in ("row_id","text","original_text","true_label","is_oos","predicted_label","predicted_label_id"):
                        require(row[field]==old_row[field], f"Frozen text/classifier changed: {field}")
                    require(float(row["mahalanobis_score"])==float(old_row["mahalanobis_score"]), "Detector score changed")
                    require(row[f"mahalanobis_{OLD_POLICY}_reject"]==old_row[f"mahalanobis_{OLD_POLICY}_reject"], "Old policy decision changed")
                    decisions = {}
                    for policy in POLICIES:
                        decisions[policy] = boolean(row[f"mahalanobis_{policy}_reject"])
                        require(decisions[policy]==(float(row["mahalanobis_score"])>=thresholds[f"mahalanobis/{policy}"]["threshold"]), "Stored decision differs from threshold")
                    before, after = decisions[OLD_POLICY], decisions[NEW_POLICY]
                    if new_cut<=old_cut:
                        require(not(before and not after), "Lower threshold removed a rejection")
                    if new_cut>=old_cut:
                        require(not(after and not before), "Higher threshold added a rejection")
                    if before!=after:
                        prefix = "oos" if boolean(row["is_oos"]) else "normal_correct" if row["predicted_label"]==row["true_label"] else "normal_wrong"
                        changes[prefix+("_added_rejects" if after else "_removed_rejects")] += 1
                transitions.append(dict(arm=arm,seed=seed,condition=condition,threshold_delta=new_cut-old_cut,**changes))
                cached[arm,seed,condition] = rows
                for policy in POLICIES:
                    for view,excluded in views.items():
                        records.append(dict(arm=arm,seed=seed,condition=condition,policy=policy,view=view,
                            score="mahalanobis",removed_n=len(excluded),threshold=thresholds[f"mahalanobis/{policy}"]["threshold"],
                            **evaluate(rows,excluded,policy)))
    for seed in SEEDS:
        for condition in CONDITIONS:
            reference = cached["clean_reference",seed,condition]
            for arm in ARMS[1:]:
                for left,right in zip(reference,cached[arm,seed,condition]):
                    require(left["predicted_label"]==right["predicted_label"] and left["text"]==right["text"], "Classifier/input changed across detector arms")
    metric_path = remember(OUTPUT / "metrics_by_seed.csv")
    require(digest(metric_path)==completion["output_hashes"][metric_path.as_posix()], "Published metrics hash mismatch")
    published = {(r["arm"],int(r["seed"]),r["condition"],r["policy"]):r for r in read_csv(metric_path)}
    require(len(published)==90, "Expected all 90 test settings")
    for row in records:
        if row["view"]=="official":
            saved = published[row["arm"],row["seed"],row["condition"],row["policy"]]
            for metric in METRICS:
                if metric in saved:
                    require(abs(row[metric]-float(saved[metric]))<1e-12, f"Published metric differs: {metric}")
    grouped = defaultdict(list)
    for row in records:
        grouped[row["arm"],row["condition"],row["policy"],row["view"]].append(row)
    groups=[]
    for key,rows in grouped.items():
        require(len(rows)==3 and {r["seed"] for r in rows}==set(SEEDS), "Missing seed")
        group=dict(zip(("arm","condition","policy","view"),key)); group["n_seeds"]=3
        for field in rows[0]:
            if field not in ("arm","seed","condition","policy","view","score"):
                group[field+"_mean"]=statistics.mean(r[field] for r in rows)
                group[field+"_sd"]=statistics.stdev(r[field] for r in rows)
        groups.append(group)
    lookup={(r["arm"],r["condition"],r["policy"],r["view"]):r for r in groups}
    decomposition=[]
    for candidate in ("ratio_10","ratio_25"):
        for reference in ("ratio_00","ratio_50"):
            for condition in CONDITIONS:
                for metric in METRICS:
                    by_view={}
                    for view in views:
                        old_gap=100*(lookup[candidate,condition,OLD_POLICY,view][metric+"_mean"]-lookup[reference,condition,OLD_POLICY,view][metric+"_mean"])
                        new_gap=100*(lookup[candidate,condition,NEW_POLICY,view][metric+"_mean"]-lookup[reference,condition,NEW_POLICY,view][metric+"_mean"])
                        by_view[view]=dict(old_gap_pp=old_gap,aligned_gap_pp=new_gap,old_minus_aligned_gap_pp=old_gap-new_gap)
                    decomposition.append(dict(candidate=candidate,reference=reference,condition=condition,metric=metric,
                        **by_view,exclusion_change_in_old_gap_pp=by_view["exclude_original_train_overlap"]["old_gap_pp"]-by_view["official"]["old_gap_pp"],
                        exclusion_change_in_aligned_gap_pp=by_view["exclude_original_train_overlap"]["aligned_gap_pp"]-by_view["official"]["aligned_gap_pp"],
                        exclusion_change_in_decomposition_pp=by_view["exclude_original_train_overlap"]["old_minus_aligned_gap_pp"]-by_view["official"]["old_minus_aligned_gap_pp"]))
    for path,expected in inputs.items():
        require(digest(Path(path))==expected, f"Input changed during calculation: {path}")
    metadata=dict(completed_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),input_sha256=inputs,
        script_path=Path(__file__).resolve().as_posix(),script_sha256=digest(Path(__file__)))
    audit_report=dict(**metadata,kind="independent_IS_only_common_operating_point_threshold_audit",
        **{k:v for k,v in threshold_summary.items() if k!="input_sha256"},per_detector=threshold_records,
        candidate_rows_checked=candidate_rows_checked,
        checks=["All feasible counts recomputed from validation IS scores only", "All 15 chosen thresholds equal independent smallest-candidate solutions",
            "Every published IS threshold candidate and acceptance count reproduced", "Old thresholds and validation counts preserved",
            "All new thresholds frozen before test rethresholding", "No OOS score used by this independent selector"])
    report=dict(**metadata,kind="fixed_score_alignment_duplicate_sensitivity_and_gap_decomposition",
        excluded_original_row_ids=sorted(mask),evaluations=len(records),group_mean_and_sample_sd=groups,
        gap_decomposition=decomposition,within_detector_threshold_transitions=transitions,
        max_absolute_exclusion_change_in_method_gap_pp=max(max(abs(r["exclusion_change_in_old_gap_pp"]),abs(r["exclusion_change_in_aligned_gap_pp"])) for r in decomposition),
        max_absolute_exclusion_change_in_gap_decomposition_pp=max(abs(r["exclusion_change_in_decomposition_pp"]) for r in decomposition),
        checks=["All 90 official settings independently recomputed", "All scores and classifier predictions equal their frozen source CSVs",
            "Every threshold change obeys rejection monotonicity", "Same duplicate mask for every arm/policy/condition", "Inputs unchanged"],
        decomposition_definition="Gap is candidate minus reference, in percentage points. Decomposition is old gap minus aligned gap, not a causal percentage.",
        limitations=["Aligning validation IS acceptance does not guarantee equal test acceptance or false-rejection rate.",
            "The intervention replaces the previous OOS-aware cutoff rule with an IS-only fixed-acceptance rule; both differ only in thresholds on the same detector scores.",
            "Gap changes are descriptive for these fixed scores, the selected 95% target and reused datasets; they do not identify a universal causal contribution or optimal ratio.",
            "Duplicate exclusion does not undo training/validation exposure; previous tests motivated this exploratory follow-up.",
            "No confidence interval or significance test is added by this sensitivity calculation."])
    return records,report,audit_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only",action="store_true")
    args=parser.parse_args()
    records,report,audit=compute()
    csv_path=OUTPUT/"duplicate_sensitivity.csv"
    report_path=OUTPUT/"duplicate_sensitivity.json"
    audit_path=OUTPUT/"threshold_audit.json"
    if args.check_only:
        before={p:digest(p) for p in (csv_path,report_path,audit_path)}
        require(read_csv(csv_path)==[{k:str(v) for k,v in r.items()} for r in records],"Existing sensitivity CSV differs")
        for path,current in ((report_path,report),(audit_path,audit)):
            previous,comparable=read_json(path),dict(current)
            previous.pop("completed_at");comparable.pop("completed_at")
            require(previous==comparable,f"Existing JSON differs: {path}")
        require(all(digest(p)==value for p,value in before.items()),"Audit outputs changed during verification")
        print("PASS: threshold audit and complete sensitivity CSV/JSON reproduced unchanged")
    else:
        require(not any(p.exists() for p in (csv_path,report_path,audit_path)),"Outputs exist; use --check-only")
        with csv_path.open("x",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        for path,result in ((report_path,report),(audit_path,audit)):
            with path.open("x",encoding="utf-8") as stream:
                json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False)
        print("Saved independent threshold audit and",len(records),"sensitivity evaluations")
    print("Common validation IS accepted:",audit["actual_common_accepted_count"],"/ 3000; candidates checked:",audit["candidate_rows_checked"])
    print("Maximum exclusion effect on a method gap (pp):",report["max_absolute_exclusion_change_in_method_gap_pp"])
    print("Maximum exclusion effect on old-minus-aligned gap (pp):",report["max_absolute_exclusion_change_in_gap_decomposition_pp"])


if __name__=="__main__":
    main()
