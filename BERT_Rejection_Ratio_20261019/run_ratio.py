"""Predeclared detector noise ratios on frozen augmented-BERT cached features.

No training or model inference. All ratio choices precede development/test.
All fixed-count arms use 30000 views of the same 15000 source requests.
"""
from __future__ import annotations
import argparse
import gc
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/"BERT_Mixed_Rejection_20261019"
sys.path.insert(0,str(PREVIOUS))
import run_mixed_rejection as previous
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from distance_model import SharedMahalanobisDetector

base=previous.base
AUG=previous.AUG
OUTPUT=ROOT/"outputs"
DATA=ROOT/"data"
SEEDS=(42,43,44)
ARMS=("clean_reference","ratio_00","ratio_10","ratio_25","ratio_50")
CONDITIONS=previous.CONDITIONS
POLICIES=previous.POLICIES
RATIO={"clean_reference":0.,"ratio_00":0.,"ratio_10":.10,"ratio_25":.25,"ratio_50":.50}
REFERENCES={"clean_reference":"clean_only","ratio_00":"clean_duplicate","ratio_50":"clean_plus_typo"}
ALLOCATION_SEED=31020261021
read=previous.read
write=previous.write
digest=previous.digest
verify=previous.verify
hashes=previous.hashes
csv=previous.csv
source=previous.source


def protocol():
    return dict(date="2026-10-19",kind="exploratory_prespecified_detector_noise_ratio_comparison",
        seeds=list(SEEDS),arms=list(ARMS),conditions=list(CONDITIONS),
        planned_noisy_pool_ratios=RATIO,reference_map=REFERENCES,
        frozen_model="Same augmented BERT42/43/44 and all cached clean/noisy training, validation and test CLS/logits. No model forward or optimization.",
        fit_views="All ratio_ arms:30000 rows; every source exactly twice. First15000 clean, second15000 clean except selected source positions replaced by the existing noisy CLS. clean_reference15000 is historical, not fixed-count comparator.",
        allocation=dict(seed=ALLOCATION_SEED,policy_indices={"typo_light":0,"typo_medium":1},
            method="For each (label_id,policy), ascending source rows permuted once by default_rng(seed+label_id*2+policy_index); select rank<10/25/50 for ratio_10/25/50",
            nested=True,per_class_noisy_counts={"ratio_00":0,"ratio_10":20,"ratio_25":50,"ratio_50":100},
            unchanged_fallbacks="Retained; no filtering on changed/eligible tokens or evaluation collisions",
            shared_across_model_seeds=True),
        estimator=previous.protocol()["estimator"],
        feature_origin="Clean train and all eval: BERT_Typo_Augmentation_20261019/outputs/typo_mix/seedN; noisy train: BERT_Mixed_Rejection_20261019/features/seedN",
        thresholds="Original clean3000 IS+100 OOS validation, original two policies/tie rules, reject score>=threshold. Three reference arms reuse exact saved detectors/thresholds. Six new detectors selected on validation. All15 detectors/30 thresholds freeze before new test scoring.",
        primary_policy="is_acceptance_95",secondary_policy="max_oos_f1",
        primary_comparisons="Each of ratio_10 and ratio_25 versus ratio_00 and ratio_50; full grid disclosed, no test-picked winning ratio",
        metrics=dict(noisy=["in_scope_false_rejection_rate","in_scope_correct_and_accepted_rate","in_scope_wrong_and_accepted_rate","oos_recall","oos_f1"],
            clean=["oos_f1","oos_recall","in_scope_acceptance"],invariant="All classifier predictions, accuracy and Macro-F1 exactly equal across arms"),
        interpretation="Descriptive per-metric comparison, not statistical noninferiority. Report whether each lower ratio retains noisy FR benefit versus ratio_00, reduces wrong-head acceptance and OOS false acceptance versus ratio_50, and improves each clean guardrail versus ratio_50. Strict clean preservation separately requires all3 clean means>=ratio_00 and separately>=clean_reference; no posthoc tolerance.",
        benefit_retention="100*(mean_FR_ratio00-mean_FR_candidate)/(mean_FR_ratio00-mean_FR_ratio50), each condition; only defined when denominator positive; not percent accuracy",
        bootstrap="2000 paired stratified original-text draws,150 IS intents+OOS, same resampled rows across arms/seeds/conditions; fixed features/detectors/thresholds",
        limitations=["Ratio candidates are predeclared for this turn, but previous test results motivated it; exploratory repeated-test followup, not a new untouched holdout",
            "One nested source subset realization and matched synthetic typo family; no claim about all subsets/natural noise",
            "30000 views contain only15000 independent source requests; fixed n removes pure fit-count changes, not data-dependent shrinkage changes",
            "Means,covariance,LedoitWolf shrinkage and validation thresholds change together",
            "Inherited3 train-val and2 train-test exact overlaps persist; exclude599/938 sensitivity does not remove historical exposure",
            "Paired intervals omit validation/refitting,subset-selection,new-noise uncertainty; comparisons not multiplicity-adjusted"])


def make_data():
    noise=csv(PREVIOUS/"data/noisy_train.csv")
    assert np.array_equal(noise.source_row,np.arange(15000))
    selection=noise[["source_row","label_id","policy","changed"]].rename(columns={"changed":"changed_existing"}).copy()
    selection["within_group_rank"]=-1
    for label in range(150):
        for policy_index,policy in enumerate(("typo_light","typo_medium")):
            rows=selection.index[(selection.label_id==label)&(selection.policy==policy)].to_numpy()
            assert len(rows)==50
            ordered=np.random.default_rng(ALLOCATION_SEED+label*2+policy_index).permutation(rows)
            selection.loc[ordered,"within_group_rank"]=np.arange(50)
    for arm,k in (("ratio_10",10),("ratio_25",25),("ratio_50",50)):
        selection[f"selected_{arm}"]=selection.within_group_rank<k
    selection.to_csv(DATA/"selection_manifest.csv",index=False)
    manifests=[];summaries={}
    for arm in ARMS:
        n=15000 if arm=="clean_reference" else 30000
        mask=selection[f"selected_{arm}"].to_numpy() if arm in ("ratio_10","ratio_25","ratio_50") else np.zeros(15000,dtype=bool)
        first=pd.DataFrame(dict(arm=arm,fit_row=np.arange(15000),source_row=noise.source_row,view="clean",label_id=noise.label_id))
        blocks=[first]
        if n==30000:
            second=first.copy();second["fit_row"]+=15000;second["view"]=np.where(mask,"noisy","clean");blocks.append(second)
        manifests+=blocks
        changed=mask & noise.changed.to_numpy()
        summaries[arm]=dict(fit_rows=n,independent_source_rows=15000,planned_noisy_ratio=RATIO[arm],
            selected_noisy_views=int(mask.sum()),selected_light=int((mask & noise.policy.eq("typo_light")).sum()),
            selected_medium=int((mask & noise.policy.eq("typo_medium")).sum()),actual_changed_rows=int(changed.sum()),
            actual_changed_pool_fraction=float(changed.sum()/n),unchanged_pool_rows=n-int(changed.sum()),
            selected_modified_token_counts={str(k):int(v) for k,v in noise.loc[mask,"modified_tokens"].value_counts().sort_index().items()})
    pd.concat(manifests,ignore_index=True).to_csv(DATA/"fit_manifest.csv",index=False)
    write(DATA/"data_summary.json",dict(allocation_seed=ALLOCATION_SEED,arms=summaries,
        nested_10_in_25_in_50=True,one_shared_selection_across_model_seeds=True,
        source_noise_seed=31020261020,source_noise_file_sha256=digest(PREVIOUS/"data/noisy_train.csv")))


def prepare():
    DATA.mkdir(parents=True,exist_ok=True);OUTPUT.mkdir(exist_ok=True)
    write(ROOT/"PROTOCOL.json",protocol(),immutable=True)
    gate=ROOT/"INPUTS_FROZEN.json"
    if gate.exists():result=read(gate);verify(result["input_hashes"]);return result
    oldinputs=read(PREVIOUS/"INPUTS_FROZEN.json")
    olddev=read(PREVIOUS/"outputs/DEVELOPMENT_FROZEN.json")
    oldtest=read(PREVIOUS/"outputs/TEST_COMPLETE.json")
    assert digest(PREVIOUS/"INPUTS_FROZEN.json")==oldtest["inputs_frozen_sha256"]
    assert digest(PREVIOUS/"outputs/DEVELOPMENT_FROZEN.json")==oldtest["development_frozen_sha256"]
    paths=[Path(__file__),ROOT/"PROTOCOL.json",PREVIOUS/"INPUTS_FROZEN.json",
        PREVIOUS/"outputs/DEVELOPMENT_FROZEN.json",PREVIOUS/"outputs/TEST_COMPLETE.json"]
    for values in (oldinputs["input_hashes"],olddev["development_hashes"],oldtest["output_hashes"]):
        verify(values);paths += [Path(p) for p in values]
    make_data();paths+=list(DATA.glob("*"))
    result=dict(frozen_at=base.now(),execution="CPU cached-feature fitting/scoring only; no new BERT inference",
        prior_runtime=oldinputs["runtime"],input_hashes=hashes(paths))
    write(gate,result,immutable=True)
    print("INPUTS_FROZEN: nested10/25% allocation fixed; old models and cached arrays hash-bound",flush=True)
    return result


def development():
    inputs=prepare();gate=OUTPUT/"DEVELOPMENT_FROZEN.json"
    if gate.exists():verify(read(gate)["development_hashes"]);print("Development already frozen",flush=True);return
    selection=csv(DATA/"selection_manifest.csv");allrows=[];paths=[]
    for seed in SEEDS:
        original=source(seed)
        clean=np.load(original/"train_cls.npy",allow_pickle=False)
        noise=np.load(PREVIOUS/f"features/seed{seed}/noisy_train_cls.npy",allow_pickle=False)
        train=csv(original/"train_metadata.csv")
        assert np.array_equal(train.row_id,selection.source_row) and np.array_equal(train.label_id,selection.label_id)
        vf=np.load(original/"validation_cls.npy",allow_pickle=False)
        vl=np.load(original/"validation_logits_fp32.npy",allow_pickle=False)
        vm=csv(original/"validation_metadata.csv")
        for arm in ARMS:
            folder=OUTPUT/arm/f"seed{seed}";folder.mkdir(parents=True,exist_ok=True)
            done=folder/"DEVELOPMENT_COMPLETE.json"
            if done.exists():
                verify(read(done)["output_hashes"]);allrows+=csv(folder/"validation_metrics.csv").to_dict("records")
                paths+=[p for p in folder.glob("*") if p.is_file()];continue
            if arm in REFERENCES:
                ref=PREVIOUS/"outputs"/REFERENCES[arm]/f"seed{seed}"
                shutil.copyfile(ref/"distance_model.npz",folder/"distance_model.npz")
                thresholds=read(ref/"frozen_thresholds.json")
                detector=SharedMahalanobisDetector.load(folder/"distance_model.npz")
            else:
                mask=selection[f"selected_{arm}"].to_numpy()
                second=clean.copy();second[mask]=noise[mask]
                fit=np.concatenate([clean,second]);targets=np.tile(train.label_id.to_numpy(),2)
                with threadpool_limits(limits=4):detector=SharedMahalanobisDetector().fit(fit,targets)
                detector.save(folder/"distance_model.npz")
                thresholds=None;del fit,second,targets
            with threadpool_limits(limits=4):scores=detector.score_samples(vf)
            curve=base.threshold_curve(scores,vm.is_oos.to_numpy())
            curve.to_csv(folder/"validation_mahalanobis_threshold_curve.csv",index=False)
            chosen={f"mahalanobis/{p}":base.choose_threshold(curve,p) for p in POLICIES}
            if thresholds is not None:assert thresholds==chosen,"Reference threshold differs"
            else:thresholds=chosen
            write(folder/"frozen_thresholds.json",thresholds)
            allrows+=previous.evaluate(arm,seed,"validation",folder,vm,vf,vl,detector,thresholds)
            localpaths=[p for p in folder.glob("*") if p.is_file()]
            write(done,dict(arm=arm,seed=seed,completed_at=base.now(),inputs_frozen_sha256=digest(ROOT/"INPUTS_FROZEN.json"),
                fit_rows=detector.n_samples_seen_,planned_noisy_ratio=RATIO[arm],shrinkage=detector.shrinkage_,
                class_counts=detector.class_counts_.tolist(),output_hashes=hashes(localpaths)))
            paths+=localpaths+[done]
            print(f"{arm}/seed{seed}:fit{detector.n_samples_seen_};shrinkage{detector.shrinkage_:.8g};clean-validation thresholds ready",flush=True)
        del clean,noise,vf,vl,detector;gc.collect()
    pd.DataFrame(allrows).to_csv(OUTPUT/"validation_metrics_by_seed.csv",index=False)
    paths.append(OUTPUT/"validation_metrics_by_seed.csv")
    verify(inputs["input_hashes"])
    write(gate,dict(frozen_at=base.now(),inputs_frozen_sha256=digest(ROOT/"INPUTS_FROZEN.json"),
        detectors=15,validation_thresholds=30,new_detectors=6,new_validation_thresholds=12,
        development_hashes=hashes(paths),status="Full ratio grid frozen before new test scoring"),immutable=True)
    print("DEVELOPMENT_FROZEN:15 detectors/30 thresholds; no test-driven selection",flush=True)


def test():
    if (OUTPUT/"TEST_COMPLETE.json").exists():raise FileExistsError("Completed experiment cannot be overwritten")
    inputs=read(ROOT/"INPUTS_FROZEN.json");verify(inputs["input_hashes"])
    gate=read(OUTPUT/"DEVELOPMENT_FROZEN.json")
    assert gate["inputs_frozen_sha256"]==digest(ROOT/"INPUTS_FROZEN.json")
    verify(gate["development_hashes"])
    rows=[];paths=[];started=base.now()
    for seed in SEEDS:
        for condition in CONDITIONS:
            original=source(seed)
            frame=csv(original/f"{condition}_metadata.csv")
            features=np.load(original/f"{condition}_cls.npy",allow_pickle=False)
            logits=np.load(original/f"{condition}_logits_fp32.npy",allow_pickle=False)
            for arm in ARMS:
                folder=OUTPUT/arm/f"seed{seed}"
                detector=SharedMahalanobisDetector.load(folder/"distance_model.npz")
                rows+=previous.evaluate(arm,seed,condition,folder,frame,features,logits,detector,read(folder/"frozen_thresholds.json"))
                paths += [folder/f"{condition}_predictions.csv",folder/f"{condition}_metrics.csv"]
            print(f"seed{seed}/{condition}:all5 arms evaluated with identical classifier predictions",flush=True)
    pd.DataFrame(rows).to_csv(OUTPUT/"metrics_by_seed.csv",index=False)
    paths.append(OUTPUT/"metrics_by_seed.csv")
    verify(inputs["input_hashes"]);verify(gate["development_hashes"])
    write(OUTPUT/"TEST_COMPLETE.json",dict(started_at=started,completed_at=base.now(),test_settings=len(rows),
        inputs_frozen_sha256=digest(ROOT/"INPUTS_FROZEN.json"),development_frozen_sha256=digest(OUTPUT/"DEVELOPMENT_FROZEN.json"),
        output_hashes=hashes(paths),original_models_inputs_and_reference_results_unchanged=True))
    print("TEST_COMPLETE:90 settings; no new BERT inference; no ratio selection based on test",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",required=True,choices=("prepare","development","test"))
    args=parser.parse_args();{"prepare":prepare,"development":development,"test":test}[args.stage]()
