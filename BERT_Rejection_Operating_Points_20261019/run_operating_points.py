"""Prespecified clean-IS acceptance sweep on frozen BERT/detector scores."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/"BERT_Rejection_Alignment_20261019"
sys.path.insert(0,str(PRIOR))
import run_alignment as alignment
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

base=alignment.base
read=alignment.read
write=alignment.write
digest=alignment.digest
hashes=alignment.hashes
verify=alignment.verify
csv=alignment.csv
OUTPUT=ROOT/"outputs"
ARMS=alignment.ARMS
SEEDS=alignment.SEEDS
CONDITIONS=alignment.CONDITIONS
TARGETS=(("aligned_is_acceptance_90",.90,2700),
         ("aligned_is_acceptance_92p5",.925,2775),
         ("aligned_is_acceptance_95",.95,2850),
         ("aligned_is_acceptance_97p5",.975,2925),
         ("aligned_is_acceptance_99",.99,2970))
POLICIES=tuple(t[0] for t in TARGETS)
ANCHOR_POLICY="aligned_is_acceptance_95"
VALIDATION_IS=3000


def source(arm,seed):
    return PRIOR/"outputs"/arm/f"seed{seed}"


def protocol():
    return dict(date="2026-10-19",kind="exploratory_fixed_score_common_operating_point_sweep",
        arms=list(ARMS),seeds=list(SEEDS),conditions=list(CONDITIONS),
        targets=[dict(policy=p,target_acceptance=a,target_accepted=n) for p,a,n in TARGETS],
        frozen="All3 BERT checkpoints, all15 detector distributions, training/evaluation features, scores and classifier predictions remain fixed. No inference or fitting.",
        primary_comparisons="ratio_25 minus ratio_00 and ratio_50 at every target; ratio_10 comparisons supplementary; show entire grid without test-selected winner",
        historical_reference="clean_reference uses15000 training views; ratio_00/10/25/50 all30000 views and unchanged nested noise subsets",
        selection=dict(data="Only3000 clean validation IS scores per detector;100 OOS validation rows only evaluated after all cutoffs selected",
            candidates="Unique IS scores plus nextafter(max_IS,+inf); accept score<threshold, reject score>=threshold",
            common_count="Intersect attainable counts of all15 detectors, independently choose smallest common count >= each requested target count",
            cutoff="Smallest candidate threshold achieving the chosen common count for each detector",
            ties="No randomized tie splitting; disclose requested and attained counts if any target requires moving upward",
            anchor="95% cutoffs and decisions must exactly reproduce BERT_Rejection_Alignment_20261019",
            freeze="All75 cutoffs frozen before any new test rethresholding"),
        metrics=["normal false rejection","normal correct and accepted","normal wrong-head and accepted","OOS false acceptance","OOS F1","OOS recall","normal acceptance","combined accuracy"],
        stability=dict(direction="For each noisy condition disclose whether25% lies between0% and50% in FR, known wrong-head acceptance and OOS false acceptance at all5 targets; also show per-seed signs",
            magnitude="Report candidate-reference gaps at every target and gap(target)-gap(95%); shared paired sampling across targets",
            uncertainty="2000 paired text-stratified draws for primary25% contrasts, pointwise95% conditional intervals. CI crossing zero is not equivalence; direction stability is not constant magnitude.",
            clean="Evaluate clean test IS acceptance and OOS F1/recall versus both0%/50% at every target; no posthoc noninferiority tolerance",
            no_utility="No costs or utility weights specified; no universally best ratio or target claimed"),
        bootstrap=dict(repeats=2000,seed=20261023,strata="150 IS intents x30 samples with replacement plus1000 OOS; same source-text draws across ratios,targets,seeds,conditions",
            seeds="3 fixed model scores averaged; do not treat them as independent test texts",
            changes="Compute paired target-minus95 gap changes within each draw; not by subtracting CI endpoints"),
        sensitivity="Exclude inherited train-test original row_ids599/938 identically from all225 settings; report gap and gap-change sensitivity, not a cure for historical exposure",
        limitations=["95% results and previous tests motivated this exploratory sweep; targets fixed for this turn but test set is reused",
            "Acceptance alignment holds on validation, not as a clean/noisy test guarantee",
            "Pointwise intervals condition on fitted models,distributions,noise realization and selected thresholds; omit refitting/validation calibration uncertainty and multiplicity correction",
            "99% cutoff uses only30 rejected validation IS per model and can be sensitive to the validation tail; no guarantee outside these5 points",
            "One nested training-noise realization and matched synthetic spelling family; natural noise and semantic preservation not independently validated",
            "3 inherited train-validation and2 train-test overlaps remain; exclusion sensitivity does not undo exposure"])


def prepare():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    write(ROOT/"PROTOCOL.json",protocol(),immutable=True)
    gate=ROOT/"INPUTS_FROZEN.json"
    if gate.exists():result=read(gate);verify(result["input_hashes"]);return result
    oldinputs=read(PRIOR/"INPUTS_FROZEN.json")
    olddev=read(PRIOR/"outputs/DEVELOPMENT_FROZEN.json")
    oldtest=read(PRIOR/"outputs/TEST_COMPLETE.json")
    assert oldtest["inputs_frozen_sha256"]==digest(PRIOR/"INPUTS_FROZEN.json")
    assert oldtest["development_frozen_sha256"]==digest(PRIOR/"outputs/DEVELOPMENT_FROZEN.json")
    paths=[Path(__file__),ROOT/"PROTOCOL.json",PRIOR/"INPUTS_FROZEN.json",PRIOR/"outputs/DEVELOPMENT_FROZEN.json",PRIOR/"outputs/TEST_COMPLETE.json"]
    for values in (oldinputs["input_hashes"],olddev["development_hashes"],oldtest["output_hashes"]):
        verify(values);paths += [Path(p) for p in values]
    result=dict(frozen_at=base.now(),execution="Only select validation-IS cutoffs and threshold cached scores",input_hashes=hashes(paths))
    write(gate,result,immutable=True)
    print("INPUTS_FROZEN:5 targets declared; all previous models/distributions/scores bound",flush=True)
    return result


def evaluate(arm,seed,condition,frame,thresholds,folder):
    result=frame.copy();normal=~result.is_oos.to_numpy()
    targets=result.label_id.to_numpy();predicted=result.predicted_label_id.to_numpy();scores=result.mahalanobis_score.to_numpy()
    closed_accuracy=float(np.mean(predicted[normal]==targets[normal]))
    closed_f1=float(f1_score(targets[normal],predicted[normal],average="macro"))
    rows=[];previous_accept=None;previous_threshold=None
    for policy,requested_rate,requested_count in TARGETS:
        item=thresholds[f"mahalanobis/{policy}"];cutoff=item["threshold"];reject=scores>=cutoff
        if policy==ANCHOR_POLICY:assert np.array_equal(reject,result[f"mahalanobis_{policy}_reject"])
        if previous_accept is not None:
            assert cutoff>=previous_threshold and not np.any(previous_accept&reject)
        previous_accept=~reject;previous_threshold=cutoff
        result[f"mahalanobis_{policy}_reject"]=reject
        row=dict(arm=arm,seed=seed,condition=condition,score="mahalanobis",policy=policy,
            target_acceptance=requested_rate,target_accepted=requested_count,
            actual_validation_accepted=item["actual_accepted"],actual_validation_acceptance=item["in_scope_acceptance"],threshold=cutoff,
            **base.evaluate(scores,result.is_oos.to_numpy(),targets,predicted,cutoff),
            closed_set_accuracy=closed_accuracy,closed_set_macro_f1=closed_f1)
        row["in_scope_wrong_and_accepted_rate"]=float(np.mean((~reject&(predicted!=targets))[normal]))
        row["oos_false_acceptance_rate"]=float(np.mean((~reject)[~normal]))
        rows.append(row)
    result.to_csv(folder/f"{condition}_predictions.csv",index=False)
    pd.DataFrame(rows).to_csv(folder/f"{condition}_metrics.csv",index=False)
    return rows


def development():
    inputs=prepare();gate=OUTPUT/"DEVELOPMENT_FROZEN.json"
    if gate.exists():verify(read(gate)["development_hashes"]);print("All75 cutoffs already frozen",flush=True);return
    frames={};options={};common=None
    for arm in ARMS:
        for seed in SEEDS:
            frame=csv(source(arm,seed)/"validation_predictions.csv")
            assert (~frame.is_oos).sum()==3000 and frame.is_oos.sum()==100
            frames[(arm,seed)]=frame
            values,counts=alignment.candidates(frame.loc[~frame.is_oos,"mahalanobis_score"].to_numpy())
            options[(arm,seed)]=(values,counts)
            feasible=set(map(int,counts));common=feasible if common is None else common&feasible
    joint={p:min(n for n in common if n>=requested) for p,_,requested in TARGETS}
    records=[];all_thresholds={}
    # Determine every cutoff using IS alone before any validation OOS evaluation.
    for (arm,seed),(values,counts) in options.items():
        all_thresholds[(arm,seed)]={}
        anchor=read(source(arm,seed)/"frozen_thresholds.json")[f"mahalanobis/{ANCHOR_POLICY}"]["threshold"]
        last=None
        for policy,rate,requested in TARGETS:
            threshold=float(np.min(values[counts==joint[policy]]))
            assert last is None or threshold>=last
            if policy==ANCHOR_POLICY:assert threshold==anchor
            last=threshold
            item=dict(threshold=threshold,target_acceptance=rate,target_accepted=requested,actual_accepted=joint[policy],
                validation_is_rows=3000,in_scope_acceptance=joint[policy]/3000,
                selection="IS-only common attainable count,smallest candidate",validation_predictions_sha256=digest(source(arm,seed)/"validation_predictions.csv"))
            all_thresholds[(arm,seed)][f"mahalanobis/{policy}"]=item
            records.append(dict(arm=arm,seed=seed,policy=policy,target_acceptance=rate,target_accepted=requested,
                actual_accepted=joint[policy],actual_acceptance=joint[policy]/3000,threshold=threshold,
                unique_is_scores=len(values)-1,candidate_count=len(values),anchor_95_threshold=anchor,
                threshold_minus_95=threshold-anchor))
    summary=dict(selected_at=base.now(),selection_used_oos=False,validation_is_rows=3000,
        targets=[dict(policy=p,target_acceptance=a,target_accepted=n,actual_common_accepted=joint[p],actual_common_acceptance=joint[p]/3000) for p,a,n in TARGETS],
        common_feasible_count_count=len(common),common_feasible_counts=sorted(common),models=records)
    write(OUTPUT/"alignment_summary.json",summary)
    pd.DataFrame(records).to_csv(OUTPUT/"operating_points.csv",index=False)
    paths=[OUTPUT/"alignment_summary.json",OUTPUT/"operating_points.csv"];rows=[]
    for arm in ARMS:
        for seed in SEEDS:
            folder=OUTPUT/arm/f"seed{seed}";folder.mkdir(parents=True,exist_ok=True)
            thresholds=all_thresholds[(arm,seed)];write(folder/"frozen_thresholds.json",thresholds)
            values,counts=options[(arm,seed)]
            pd.DataFrame(dict(threshold=values,accepted_is=counts,validation_is_rows=3000)).to_csv(folder/"validation_is_threshold_candidates.csv",index=False)
            rows+=evaluate(arm,seed,"validation",frames[(arm,seed)],thresholds,folder)
            local=[p for p in folder.glob("*") if p.is_file()]
            detector=alignment.source(arm,seed)/"distance_model.npz"
            done=folder/"DEVELOPMENT_COMPLETE.json"
            write(done,dict(completed_at=base.now(),arm=arm,seed=seed,inputs_frozen_sha256=digest(ROOT/"INPUTS_FROZEN.json"),
                detector_path=detector.resolve().as_posix(),detector_sha256=digest(detector),output_hashes=hashes(local)))
            paths+=local+[done]
            print(f"{arm}/seed{seed}:5 IS-only cutoffs selected;95% anchor identical",flush=True)
    pd.DataFrame(rows).to_csv(OUTPUT/"validation_metrics_by_seed.csv",index=False);paths.append(OUTPUT/"validation_metrics_by_seed.csv")
    verify(inputs["input_hashes"])
    write(gate,dict(frozen_at=base.now(),inputs_frozen_sha256=digest(ROOT/"INPUTS_FROZEN.json"),existing_detectors=15,
        total_thresholds=75,targets=summary["targets"],selection_used_oos=False,development_hashes=hashes(paths),
        status="All75 cutoffs frozen before test rethresholding"),immutable=True)
    print("DEVELOPMENT_FROZEN:75 cutoffs across5 common IS-only operating points",flush=True)


def test():
    if (OUTPUT/"TEST_COMPLETE.json").exists():raise FileExistsError("Completed test cannot be overwritten")
    inputs=read(ROOT/"INPUTS_FROZEN.json");verify(inputs["input_hashes"])
    gate=read(OUTPUT/"DEVELOPMENT_FROZEN.json")
    assert gate["inputs_frozen_sha256"]==digest(ROOT/"INPUTS_FROZEN.json")
    verify(gate["development_hashes"]);started=base.now();rows=[];paths=[]
    for arm in ARMS:
        for seed in SEEDS:
            folder=OUTPUT/arm/f"seed{seed}";thresholds=read(folder/"frozen_thresholds.json")
            for condition in CONDITIONS:
                frame=csv(source(arm,seed)/f"{condition}_predictions.csv")
                rows+=evaluate(arm,seed,condition,frame,thresholds,folder)
                paths += [folder/f"{condition}_predictions.csv",folder/f"{condition}_metrics.csv"]
            print(f"{arm}/seed{seed}:15 test settings from unchanged scores",flush=True)
    pd.DataFrame(rows).to_csv(OUTPUT/"metrics_by_seed.csv",index=False);paths.append(OUTPUT/"metrics_by_seed.csv")
    verify(inputs["input_hashes"]);verify(gate["development_hashes"])
    write(OUTPUT/"TEST_COMPLETE.json",dict(started_at=started,completed_at=base.now(),test_settings=len(rows),
        inputs_frozen_sha256=digest(ROOT/"INPUTS_FROZEN.json"),development_frozen_sha256=digest(OUTPUT/"DEVELOPMENT_FROZEN.json"),
        output_hashes=hashes(paths),original_models_detectors_features_scores_unchanged=True,anchor_95_decisions_identical=True))
    print("TEST_COMPLETE:225 settings, fixed scores,95% regression passed",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=("prepare","development","test"),required=True)
    args=parser.parse_args();{"prepare":prepare,"development":development,"test":test}[args.stage]()
