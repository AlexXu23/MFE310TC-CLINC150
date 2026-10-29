"""Fixed-score operating-point sensitivity with uniform source-row exclusion.

No production runner/report is imported. Only frozen CSV predictions are read;
there is no training, inference, distribution refit or cutoff selection.
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
ALIGNMENT = ROOT.parent / "BERT_Rejection_Alignment_20261019"
PROJECT = ROOT.parent / "CLINC150_Team_20261019"
ARMS = ("clean_reference", "ratio_00", "ratio_10", "ratio_25", "ratio_50")
SEEDS = (42,43,44)
CONDITIONS = ("clean","typo_light","typo_medium")
TARGETS = ((.9,2700,"aligned_is_acceptance_90"),(.925,2775,"aligned_is_acceptance_92p5"),
    (.95,2850,"aligned_is_acceptance_95"),(.975,2925,"aligned_is_acceptance_97p5"),
    (.99,2970,"aligned_is_acceptance_99"))
METRICS = ("in_scope_false_rejection_rate","in_scope_correct_and_accepted_rate",
    "in_scope_wrong_and_accepted_rate","closed_set_accuracy","oos_f1","oos_recall",
    "oos_false_acceptance_rate","in_scope_acceptance","combined_151_accuracy")


def require(condition,message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8",newline="") as stream:
        return list(csv.DictReader(stream))


def boolean(value):
    require(value in ("True","False"),f"Invalid stored Boolean: {value!r}")
    return value=="True"


def evaluate(rows,excluded,policy):
    counts=dict(in_scope_n=0,oos_n=0,in_scope_correct_and_accepted=0,
        in_scope_wrong_and_accepted=0,in_scope_correct_and_rejected=0,
        in_scope_wrong_and_rejected=0,oos_true_positive=0,oos_missed=0)
    for row in rows:
        if int(row["row_id"]) in excluded:
            continue
        reject=boolean(row[f"mahalanobis_{policy}_reject"])
        if boolean(row["is_oos"]):
            counts["oos_n"]+=1
            counts["oos_true_positive" if reject else "oos_missed"]+=1
        else:
            counts["in_scope_n"]+=1
            correct=row["predicted_label"]==row["true_label"]
            counts["in_scope_"+("correct" if correct else "wrong")+"_and_"+("rejected" if reject else "accepted")]+=1
    n_is,n_oos=counts["in_scope_n"],counts["oos_n"]
    require(n_is in (4500,4498) and n_oos==1000,"Unexpected sensitivity denominators")
    tp=counts["oos_true_positive"]
    fp=counts["in_scope_correct_and_rejected"]+counts["in_scope_wrong_and_rejected"]
    counts["in_scope_false_rejections"]=fp
    correct=counts["in_scope_correct_and_accepted"]+counts["in_scope_correct_and_rejected"]
    require(fp+counts["in_scope_correct_and_accepted"]+counts["in_scope_wrong_and_accepted"]==n_is,"IS state partition mismatch")
    return dict(**counts,in_scope_false_rejection_rate=fp/n_is,
        in_scope_correct_and_accepted_rate=counts["in_scope_correct_and_accepted"]/n_is,
        in_scope_wrong_and_accepted_rate=counts["in_scope_wrong_and_accepted"]/n_is,
        closed_set_accuracy=correct/n_is,oos_f1=2*tp/(n_oos+tp+fp),oos_recall=tp/n_oos,
        oos_false_acceptance_rate=counts["oos_missed"]/n_oos,in_scope_acceptance=1-fp/n_is,
        combined_151_accuracy=(counts["in_scope_correct_and_accepted"]+tp)/(n_is+n_oos))


def compute():
    inputs={}
    def remember(path):
        inputs[path.resolve().as_posix()]=digest(path)
        return path
    completion=read_json(remember(OUTPUT/"TEST_COMPLETE.json"))
    freeze_path=remember(ROOT/"INPUTS_FROZEN.json")
    dev_path=remember(OUTPUT/"DEVELOPMENT_FROZEN.json")
    require(digest(freeze_path)==completion["inputs_frozen_sha256"],"Input gate hash mismatch")
    require(digest(dev_path)==completion["development_frozen_sha256"],"Development gate hash mismatch")
    frozen,development=read_json(freeze_path),read_json(dev_path)
    require(development["total_thresholds"]==75 and development["selection_used_oos"] is False,"Unexpected cutoff protocol")
    require(datetime.fromisoformat(development["frozen_at"])<=datetime.fromisoformat(completion["started_at"]),"Test began before cutoff freeze")
    old_completion=read_json(remember(ALIGNMENT/"outputs/TEST_COMPLETE.json"))
    old_development=read_json(remember(ALIGNMENT/"outputs/DEVELOPMENT_FROZEN.json"))
    raw_path=remember(PROJECT/"data/raw/data_full.json")
    require(digest(raw_path)==frozen["input_hashes"][raw_path.as_posix()],"Original dataset hash mismatch")
    raw=read_json(raw_path)
    official=raw["test"]+raw["oos_test"]
    train_texts={text for text,_ in raw["train"]}
    mask={i for i,(text,_) in enumerate(official) if text in train_texts}
    require(mask=={599,938},"Unexpected inherited overlap mask")
    views={"official":set(),"exclude_original_train_overlap":mask}
    cached,records={},[]
    for arm in ARMS:
        for seed in SEEDS:
            folder=OUTPUT/arm/f"seed{seed}"
            threshold_path=remember(folder/"frozen_thresholds.json")
            require(digest(threshold_path)==development["development_hashes"][threshold_path.as_posix()],"Cutoff hash mismatch")
            thresholds=read_json(threshold_path)
            old_threshold_path=remember(ALIGNMENT/"outputs"/arm/f"seed{seed}"/"frozen_thresholds.json")
            require(digest(old_threshold_path)==old_development["development_hashes"][old_threshold_path.as_posix()],"95 anchor cutoff hash mismatch")
            anchor=read_json(old_threshold_path)["mahalanobis/aligned_is_acceptance_95"]["threshold"]
            previous_cutoff=None
            for target,count,policy in TARGETS:
                item=thresholds[f"mahalanobis/{policy}"]
                require(item["target_acceptance"]==target and item["target_accepted"]==count,"Target definition mismatch")
                require(item["actual_accepted"]>=count and item["in_scope_acceptance"]==item["actual_accepted"]/3000,"Attained validation acceptance mismatch")
                require(previous_cutoff is None or item["threshold"]>=previous_cutoff,"Cutoffs are not monotone with acceptance target")
                if target==.95:
                    require(item["threshold"]==anchor,"95 cutoff differs from Alignment")
                previous_cutoff=item["threshold"]
            for condition in CONDITIONS:
                path=remember(folder/f"{condition}_predictions.csv")
                old_path=remember(ALIGNMENT/"outputs"/arm/f"seed{seed}"/f"{condition}_predictions.csv")
                require(digest(path)==completion["output_hashes"][path.as_posix()],"Prediction hash mismatch")
                require(digest(old_path)==old_completion["output_hashes"][old_path.as_posix()],"Anchor source prediction hash mismatch")
                rows,old_rows=read_csv(path),read_csv(old_path)
                require(len(rows)==len(old_rows)==5500,"Unexpected test row count")
                for i,(row,old_row) in enumerate(zip(rows,old_rows)):
                    require(int(row["row_id"])==i and [row["original_text"],row["true_label"]]==official[i],"Official row identity mismatch")
                    for field in ("row_id","text","original_text","true_label","is_oos","predicted_label_id","predicted_label"):
                        require(row[field]==old_row[field],f"Frozen row/classifier changed: {field}")
                    require(float(row["mahalanobis_score"])==float(old_row["mahalanobis_score"]),"Frozen detector score changed")
                    require(row["condition"]==condition,"Condition mismatch")
                    previous_accept=False
                    for target,count,policy in TARGETS:
                        reject=boolean(row[f"mahalanobis_{policy}_reject"])
                        require(reject==(float(row["mahalanobis_score"])>=thresholds[f"mahalanobis/{policy}"]["threshold"]),"Decision differs from frozen cutoff")
                        require(not(previous_accept and reject),"Larger target rejected a previously accepted row")
                        previous_accept=not reject
                        if target==.95:
                            require(row[f"mahalanobis_{policy}_reject"]==old_row[f"mahalanobis_{policy}_reject"],"95 anchor decision differs")
                cached[arm,seed,condition]=rows
                for target,count,policy in TARGETS:
                    item=thresholds[f"mahalanobis/{policy}"]
                    for view,excluded in views.items():
                        records.append(dict(arm=arm,seed=seed,condition=condition,policy=policy,score="mahalanobis",view=view,
                            target_acceptance=target,target_accepted=count,actual_validation_accepted=item["actual_accepted"],
                            actual_validation_acceptance=item["in_scope_acceptance"],threshold=item["threshold"],removed_n=len(excluded),
                            **evaluate(rows,excluded,policy)))
    for seed in SEEDS:
        for condition in CONDITIONS:
            reference=cached["clean_reference",seed,condition]
            for arm in ARMS[1:]:
                for left,right in zip(reference,cached[arm,seed,condition]):
                    require(left["predicted_label"]==right["predicted_label"] and left["text"]==right["text"],"Classifier/text differs across detector arms")
    metric_path=remember(OUTPUT/"metrics_by_seed.csv")
    require(digest(metric_path)==completion["output_hashes"][metric_path.as_posix()],"Metrics hash mismatch")
    published={(r["arm"],int(r["seed"]),r["condition"],r["policy"]):r for r in read_csv(metric_path)}
    require(len(published)==225,"Expected all 225 settings")
    old_metrics_path=remember(ALIGNMENT/"outputs/metrics_by_seed.csv")
    require(digest(old_metrics_path)==old_completion["output_hashes"][old_metrics_path.as_posix()],"95 source metrics hash mismatch")
    old_metrics={(r["arm"],int(r["seed"]),r["condition"],r["policy"]):r for r in read_csv(old_metrics_path)}
    for row in records:
        if row["view"]=="official":
            key=row["arm"],row["seed"],row["condition"],row["policy"]
            saved=published[key]
            for field in (*METRICS,"target_acceptance","target_accepted","actual_validation_accepted","actual_validation_acceptance"):
                require(abs(row[field]-float(saved[field]))<1e-12,f"Published field differs: {field}")
            if row["target_acceptance"]==.95:
                for metric in METRICS:
                    if metric in old_metrics[key]:
                        require(abs(row[metric]-float(old_metrics[key][metric]))<1e-12,f"95 metric differs from Alignment: {metric}")
    grouped=defaultdict(list)
    for row in records:
        grouped[row["arm"],row["condition"],row["policy"],row["view"]].append(row)
    groups=[]
    for key,rows in grouped.items():
        require(len(rows)==3 and {r["seed"] for r in rows}==set(SEEDS),"Missing seed in summary")
        group=dict(zip(("arm","condition","policy","view"),key))
        group.update(n_seeds=3,target_acceptance=rows[0]["target_acceptance"],target_accepted=rows[0]["target_accepted"])
        for field in rows[0]:
            if field not in ("arm","seed","condition","policy","view","score","target_acceptance","target_accepted"):
                group[field+"_mean"]=statistics.mean(r[field] for r in rows)
                group[field+"_sd"]=statistics.stdev(r[field] for r in rows)
        groups.append(group)
    lookup={(r["arm"],r["condition"],r["target_acceptance"],r["view"]):r for r in groups}
    gaps=[]
    for reference in ("ratio_00","ratio_50"):
        for condition in CONDITIONS:
            for target,count,policy in TARGETS:
                for metric in METRICS:
                    by_view={}
                    for view in views:
                        gap=100*(lookup["ratio_25",condition,target,view][metric+"_mean"]-lookup[reference,condition,target,view][metric+"_mean"])
                        anchor_gap=100*(lookup["ratio_25",condition,.95,view][metric+"_mean"]-lookup[reference,condition,.95,view][metric+"_mean"])
                        by_view[view]=dict(gap_pp=gap,anchor_95_gap_pp=anchor_gap,gap_minus_95_gap_pp=gap-anchor_gap)
                    gaps.append(dict(candidate="ratio_25",reference=reference,condition=condition,policy=policy,
                        target_acceptance=target,target_accepted=count,metric=metric,**by_view,
                        exclusion_change_in_gap_pp=by_view["exclude_original_train_overlap"]["gap_pp"]-by_view["official"]["gap_pp"],
                        exclusion_change_in_gap_minus_95_gap_pp=by_view["exclude_original_train_overlap"]["gap_minus_95_gap_pp"]-by_view["official"]["gap_minus_95_gap_pp"]))
    directions=[]
    for reference in ("ratio_00","ratio_50"):
        for condition in CONDITIONS:
            for metric in METRICS:
                selected=[r for r in gaps if r["reference"]==reference and r["condition"]==condition and r["metric"]==metric]
                values=[r["official"]["gap_pp"] for r in selected]
                directions.append(dict(candidate="ratio_25",reference=reference,condition=condition,metric=metric,
                    targets=[r["target_acceptance"] for r in selected],signed_gaps_pp=values,
                    all_nonpositive=all(v<=0 for v in values),all_nonnegative=all(v>=0 for v in values),
                    strict_sign_changes=(min(values)<0<max(values)),min_gap_pp=min(values),max_gap_pp=max(values)))
    for path,expected in inputs.items():
        require(digest(Path(path))==expected,f"Input changed during sensitivity calculation: {path}")
    report=dict(completed_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),kind="fixed_operating_point_grid_duplicate_sensitivity",
        script_path=Path(__file__).resolve().as_posix(),script_sha256=digest(Path(__file__)),input_sha256=inputs,
        excluded_original_row_ids=sorted(mask),evaluations=len(records),group_mean_and_sample_sd=groups,
        primary_ratio25_gaps_and_changes_from_95=gaps,descriptive_mean_gap_direction_across_targets=directions,
        max_absolute_exclusion_change_in_gap_pp=max(abs(r["exclusion_change_in_gap_pp"]) for r in gaps),
        max_absolute_exclusion_change_in_gap_minus_95_gap_pp=max(abs(r["exclusion_change_in_gap_minus_95_gap_pp"]) for r in gaps),
        checks=["All 225 official metric settings independently reproduced",
            "All 75 frozen cutoff decisions and monotonicity across targets checked",
            "95% thresholds, row decisions and metrics reproduce Alignment exactly",
            "All frozen detector scores and classifier predictions unchanged",
            "Same original row-ID exclusion for all arms, conditions and targets", "Inputs unchanged"],
        gap_definition="ratio_25 minus reference at each target, in percentage points; change from 95 is gap(target)-gap(95)",
        limitations=["Direction summaries describe five prespecified points on the same reused test set, not five independent replications.",
            "This script adds no inference or confidence interval; pointwise intervals elsewhere are not simultaneous bands.",
            "A stable sign does not imply a constant effect size, universal utility optimum or guaranteed behavior outside the five points.",
            "Validation IS acceptance alignment is not a test acceptance guarantee.",
            "Uniform evaluation exclusions do not undo training or validation exposure; the analysis remains exploratory.",
            "No detector, ratio, threshold target or source subset was selected or changed using these results."])
    return records,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only",action="store_true")
    args=parser.parse_args()
    records,report=compute()
    csv_path,json_path=OUTPUT/"duplicate_sensitivity.csv",OUTPUT/"duplicate_sensitivity.json"
    if args.check_only:
        before={p:digest(p) for p in (csv_path,json_path)}
        require(read_csv(csv_path)==[{k:str(v) for k,v in r.items()} for r in records],"Existing CSV differs from recomputation")
        previous,comparable=read_json(json_path),dict(report)
        previous.pop("completed_at");comparable.pop("completed_at")
        require(previous==comparable,"Existing JSON differs from recomputation")
        require(all(digest(p)==value for p,value in before.items()),"Sensitivity output changed during verification")
        print("PASS: complete operating-point sensitivity CSV/JSON reproduced unchanged")
    else:
        require(not csv_path.exists() and not json_path.exists(),"Sensitivity outputs exist; use --check-only")
        with csv_path.open("x",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        with json_path.open("x",encoding="utf-8") as stream:
            json.dump(report,stream,indent=2,ensure_ascii=False,allow_nan=False)
        print("Saved",len(records),"uniform-exclusion evaluations")
    print("Maximum exclusion effect on primary gap (pp):",report["max_absolute_exclusion_change_in_gap_pp"])
    print("Maximum exclusion effect on gap(target)-gap(95) (pp):",report["max_absolute_exclusion_change_in_gap_minus_95_gap_pp"])


if __name__=="__main__":
    main()
