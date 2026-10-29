"""IS-only clean-validation acceptance alignment on fixed detector scores.

No inference, feature extraction or detector fitting. Fifteen IS-only cutoffs
are jointly frozen before test decisions are recomputed from cached scores.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/"BERT_Rejection_Ratio_20261019"
sys.path.insert(0,str(PRIOR))
import run_ratio as ratio
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

base=ratio.base
read=ratio.read
write=ratio.write
digest=ratio.digest
hashes=ratio.hashes
verify=ratio.verify
csv=ratio.csv
OUTPUT=ROOT/"outputs"
ARMS=ratio.ARMS
SEEDS=ratio.SEEDS
CONDITIONS=ratio.CONDITIONS
OLD_POLICY="is_acceptance_95"
NEW_POLICY="aligned_is_acceptance_95"
POLICIES=(OLD_POLICY,NEW_POLICY)
TARGET_ACCEPTED=2850
VALIDATION_IS=3000


def source(arm,seed):
    return PRIOR/"outputs"/arm/f"seed{seed}"


def protocol():
    return dict(date="2026-10-19",kind="exploratory_fixed_score_operating_point_alignment",
        arms=list(ARMS),seeds=list(SEEDS),conditions=list(CONDITIONS),policies=list(POLICIES),
        primary_policy=NEW_POLICY,reference_policy=OLD_POLICY,
        frozen="All BERT weights, classifier predictions, feature arrays,15 detector means/covariances/precisions and cached scores remain identical to the ratio experiment. No model inference or detector refitting.",
        alignment=dict(target_acceptance=.95,target_accepted=TARGET_ACCEPTED,normal_validation_rows=VALIDATION_IS,
            selection_data="Only3000 in-scope rows from each saved clean validation_predictions.csv; OOS labels/scores never supplied to the selection function",
            candidates="Sorted unique IS scores plus nextafter(max_IS_score,+inf); accept iff score<threshold, reject iff score>=threshold",
            joint_target="Intersect attainable integer accepted-count sets of all15 detectors; choose smallest common count>=2850",
            threshold="For each detector choose smallest listed candidate yielding that common count; record requested and achieved acceptance explicitly",
            ties="No randomized tie-breaking, no test-dependent tolerance; if ties prevent95%, align at nearest higher common attainable count and disclose it"),
        original_policy="Original constraint ISaccept>=95%, then maximize validation OOS recall/F1 with original tie rules; retain exact saved threshold for within-detector comparison",
        validation_oos="Evaluate100 OOS only after IS-only cutoffs are determined; never tune new cutoffs using those rows",
        gate="All15 new thresholds frozen with scores/source hashes before any test rethresholding",
        comparisons=dict(within_detector="Aligned minus original policy for each arm,seed,condition",
            across_ratios="Each of10%/25% versus0%/50%, under both policies",
            gap_decomposition="old_gap=M(candidate,old)-M(reference,old); aligned_gap=M(candidate,new)-M(reference,new); gap_change=old_gap-aligned_gap. Exact fixed-score counterfactual accounting at this selected common validation point, not universal causal attribution or an optimality claim.",
            historical_reference="clean_reference15000 remains supplementary; ratio arms all use30000 views"),
        metrics=["normal false rejection","normal correct-and-accepted","normal wrong-head-and-accepted","OOS false acceptance","OOS F1","OOS recall","normal acceptance","combined accuracy"],
        invariants=["Scores and classifier labels identical within each detector before/after alignment",
            "All arms share identical classifier predictions for same seed/input",
            "Old-policy decisions and metrics reproduce prior results",
            "Within a detector, moving a threshold downward can only add rejections; upward can only remove them"],
        bootstrap="2000 paired stratified draws,150 IS intents x30 plus1000 OOS, shared row weights across arms/policies/seeds/conditions; fixed scores and both sets of thresholds",
        limitations=["Previously inspected tests motivated alignment; exploratory repeated-test followup, not untouched confirmation",
            "Aligning validation acceptance does not guarantee equal clean-test or noisy-test acceptance",
            "Changing old OOS-aware threshold selection to IS-only selection is part of the intervention; residual differences can include validation-to-test transfer and distribution/score ordering effects",
            "Gap change is conditional on the95% target, fixed datasets/models/subsets/thresholds, not an invariant percentage of causation",
            "Inherited train-val/test duplicates remain; exclusion599/938 does not cure historical exposure",
            "Intervals omit validation threshold selection,detector/model refitting,new-noise and subset uncertainty; no multiplicity correction"])


def prepare():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    write(ROOT/"PROTOCOL.json",protocol(),immutable=True)
    target=ROOT/"INPUTS_FROZEN.json"
    if target.exists():result=read(target);verify(result["input_hashes"]);return result
    oldinputs=read(PRIOR/"INPUTS_FROZEN.json")
    olddev=read(PRIOR/"outputs/DEVELOPMENT_FROZEN.json")
    oldtest=read(PRIOR/"outputs/TEST_COMPLETE.json")
    assert oldtest["inputs_frozen_sha256"]==digest(PRIOR/"INPUTS_FROZEN.json")
    assert oldtest["development_frozen_sha256"]==digest(PRIOR/"outputs/DEVELOPMENT_FROZEN.json")
    paths=[Path(__file__),ROOT/"PROTOCOL.json",PRIOR/"INPUTS_FROZEN.json",PRIOR/"outputs/DEVELOPMENT_FROZEN.json",PRIOR/"outputs/TEST_COMPLETE.json"]
    for values in (oldinputs["input_hashes"],olddev["development_hashes"],oldtest["output_hashes"]):
        verify(values);paths += [Path(p) for p in values]
    saved=dict(frozen_at=base.now(),execution="Re-threshold saved scores only; no fitting or inference",input_hashes=hashes(paths))
    write(target,saved,immutable=True)
    print("INPUTS_FROZEN: all original scores,detectors,models and threshold policies bound",flush=True)
    return saved


def candidates(is_scores):
    """Inputs contain IS scores only: deliberately no OOS or text/label argument."""
    values=np.asarray(is_scores,dtype=np.float64)
    assert values.shape==(VALIDATION_IS,) and np.isfinite(values).all()
    ordered=np.sort(values)
    thresholds=np.r_[np.unique(ordered),np.nextafter(ordered[-1],np.inf)]
    accepted=np.searchsorted(ordered,thresholds,side="left")
    return thresholds,accepted


def evaluate(arm,seed,condition,frame,thresholds,folder):
    # Classifier and scores are carried through directly, without recomputation.
    result=frame.copy()
    normal=~result.is_oos.to_numpy();targets=result.label_id.to_numpy()
    prediction=result.predicted_label_id.to_numpy();scores=result.mahalanobis_score.to_numpy()
    rows=[]
    for policy in POLICIES:
        cutoff=thresholds[f"mahalanobis/{policy}"]["threshold"]
        reject=scores>=cutoff
        if policy==OLD_POLICY:
            assert np.array_equal(reject,result[f"mahalanobis_{OLD_POLICY}_reject"])
        result[f"mahalanobis_{policy}_reject"]=reject
        rows.append(dict(arm=arm,seed=seed,condition=condition,score="mahalanobis",policy=policy,threshold=cutoff,
            **base.evaluate(scores,result.is_oos.to_numpy(),targets,prediction,cutoff),
            closed_set_accuracy=float(np.mean(prediction[normal]==targets[normal])),
            closed_set_macro_f1=float(f1_score(targets[normal],prediction[normal],average="macro"))))
    result.to_csv(folder/f"{condition}_predictions.csv",index=False)
    pd.DataFrame(rows).to_csv(folder/f"{condition}_metrics.csv",index=False)
    return rows


def development():
    inputs=prepare();gate=OUTPUT/"DEVELOPMENT_FROZEN.json"
    if gate.exists():verify(read(gate)["development_hashes"]);print("Thresholds already frozen",flush=True);return
    frames={};options={};common=None
    for arm in ARMS:
        for seed in SEEDS:
            frame=csv(source(arm,seed)/"validation_predictions.csv")
            assert int((~frame.is_oos).sum())==VALIDATION_IS and int(frame.is_oos.sum())==100
            frames[(arm,seed)]=frame
            # No OOS scores reach the selector.
            thresholds,accepted=candidates(frame.loc[~frame.is_oos,"mahalanobis_score"].to_numpy())
            options[(arm,seed)]=(thresholds,accepted)
            counts=set(map(int,accepted))
            common=counts if common is None else common & counts
    feasible=sorted(n for n in common if n>=TARGET_ACCEPTED)
    assert feasible
    joint_accepted=feasible[0]
    summaries=[];all_thresholds={};paths=[]
    # Determine every cutoff before evaluating any validation OOS outcome.
    for (arm,seed),(thresholds,accepted) in options.items():
        chosen=float(np.min(thresholds[accepted==joint_accepted]))
        old=read(source(arm,seed)/"frozen_thresholds.json")[f"mahalanobis/{OLD_POLICY}"]
        all_thresholds[(arm,seed)]={f"mahalanobis/{OLD_POLICY}":old,
            f"mahalanobis/{NEW_POLICY}":dict(threshold=chosen,target_accepted=TARGET_ACCEPTED,
                actual_accepted=joint_accepted,validation_is_rows=VALIDATION_IS,
                in_scope_acceptance=joint_accepted/VALIDATION_IS,selection="IS-only common attainable count; smallest candidate at count",
                validation_predictions_sha256=digest(source(arm,seed)/"validation_predictions.csv"))}
        frame=frames[(arm,seed)];is_scores=frame.loc[~frame.is_oos,"mahalanobis_score"].to_numpy()
        summaries.append(dict(arm=arm,seed=seed,requested_accepted=TARGET_ACCEPTED,actual_accepted=joint_accepted,
            requested_acceptance=.95,actual_acceptance=joint_accepted/VALIDATION_IS,
            original_accepted=int((is_scores<old["threshold"]).sum()),original_threshold=old["threshold"],aligned_threshold=chosen,
            unique_is_scores=len(np.unique(is_scores)),candidate_count=len(thresholds),threshold_delta=chosen-old["threshold"]))
    write(OUTPUT/"alignment_summary.json",dict(target_accepted=TARGET_ACCEPTED,actual_common_accepted=joint_accepted,
        validation_is_rows=VALIDATION_IS,target_acceptance=.95,actual_common_acceptance=joint_accepted/VALIDATION_IS,
        common_feasible_count_count=len(common),common_feasible_counts=sorted(common),models=summaries,
        selection_used_oos=False,selected_at=base.now()))
    pd.DataFrame(summaries).to_csv(OUTPUT/"operating_points.csv",index=False)
    paths += [OUTPUT/"alignment_summary.json",OUTPUT/"operating_points.csv"]
    rows=[]
    for arm in ARMS:
        for seed in SEEDS:
            folder=OUTPUT/arm/f"seed{seed}";folder.mkdir(parents=True,exist_ok=True)
            thresholds=all_thresholds[(arm,seed)]
            write(folder/"frozen_thresholds.json",thresholds)
            values,counts=options[(arm,seed)]
            pd.DataFrame(dict(threshold=values,accepted_is=counts,validation_is_rows=VALIDATION_IS)).to_csv(folder/"validation_is_threshold_candidates.csv",index=False)
            rows+=evaluate(arm,seed,"validation",frames[(arm,seed)],thresholds,folder)
            local=[p for p in folder.glob("*") if p.is_file()]
            done=folder/"DEVELOPMENT_COMPLETE.json"
            write(done,dict(completed_at=base.now(),arm=arm,seed=seed,inputs_frozen_sha256=digest(ROOT/"INPUTS_FROZEN.json"),
                detector_path=(source(arm,seed)/"distance_model.npz").resolve().as_posix(),
                detector_sha256=digest(source(arm,seed)/"distance_model.npz"),output_hashes=hashes(local)))
            paths+=local+[done]
            print(f"{arm}/seed{seed}:exact IS acceptance {joint_accepted}/{VALIDATION_IS}; original scores unchanged",flush=True)
    pd.DataFrame(rows).to_csv(OUTPUT/"validation_metrics_by_seed.csv",index=False)
    paths.append(OUTPUT/"validation_metrics_by_seed.csv")
    verify(inputs["input_hashes"])
    write(gate,dict(frozen_at=base.now(),inputs_frozen_sha256=digest(ROOT/"INPUTS_FROZEN.json"),
        existing_detectors=15,new_thresholds=15,total_thresholds_including_original=30,
        target_accepted=TARGET_ACCEPTED,actual_common_accepted=joint_accepted,selection_used_oos=False,
        development_hashes=hashes(paths),status="All IS-only thresholds frozen before test rethresholding"),immutable=True)
    print("DEVELOPMENT_FROZEN:15 aligned cutoffs, no OOS-selected or test-selected new cutoff",flush=True)


def test():
    if (OUTPUT/"TEST_COMPLETE.json").exists():raise FileExistsError("Completed experiment cannot be overwritten")
    inputs=read(ROOT/"INPUTS_FROZEN.json");verify(inputs["input_hashes"])
    gate=read(OUTPUT/"DEVELOPMENT_FROZEN.json")
    assert gate["inputs_frozen_sha256"]==digest(ROOT/"INPUTS_FROZEN.json")
    verify(gate["development_hashes"])
    rows=[];paths=[];started=base.now()
    for arm in ARMS:
        for seed in SEEDS:
            folder=OUTPUT/arm/f"seed{seed}"
            thresholds=read(folder/"frozen_thresholds.json")
            for condition in CONDITIONS:
                frame=csv(source(arm,seed)/f"{condition}_predictions.csv")
                rows+=evaluate(arm,seed,condition,frame,thresholds,folder)
                paths += [folder/f"{condition}_predictions.csv",folder/f"{condition}_metrics.csv"]
            print(f"{arm}/seed{seed}:all test conditions rethresholded; no scores changed",flush=True)
    pd.DataFrame(rows).to_csv(OUTPUT/"metrics_by_seed.csv",index=False)
    paths.append(OUTPUT/"metrics_by_seed.csv")
    verify(inputs["input_hashes"]);verify(gate["development_hashes"])
    write(OUTPUT/"TEST_COMPLETE.json",dict(started_at=started,completed_at=base.now(),test_settings=len(rows),
        inputs_frozen_sha256=digest(ROOT/"INPUTS_FROZEN.json"),development_frozen_sha256=digest(OUTPUT/"DEVELOPMENT_FROZEN.json"),
        output_hashes=hashes(paths),original_models_detectors_features_scores_unchanged=True))
    print("TEST_COMPLETE:90 original/aligned settings; only threshold decisions changed",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=("prepare","development","test"),required=True)
    args=parser.parse_args();{"prepare":prepare,"development":development,"test":test}[args.stage]()
