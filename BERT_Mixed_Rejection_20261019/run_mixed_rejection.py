"""Frozen augmented BERT: change only the training distribution used by rejection.

No optimizer or training step exists. All detector/threshold development finishes
before any new test scoring. Prior artifacts are hash-bound, read-only inputs.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "BERT_Rejection_Study_20261019"
AUG = ROOT.parent / "BERT_Typo_Augmentation_20261019"
sys.path.insert(0, str(PRIOR))
import run_rejection_study as base
from distance_model import SharedMahalanobisDetector
from typo_data import make_variants
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer, BertForSequenceClassification

OUTPUT = ROOT / "outputs"
DATA = ROOT / "data"
SEEDS = (42, 43, 44)
ARMS = ("clean_only", "clean_duplicate", "clean_plus_typo")
CONDITIONS = ("clean", "typo_light", "typo_medium")
POLICIES = base.POLICIES
NOISE_SEED = 31020261020
read = base.read_json
digest = base.digest
write = base.write_json


def csv(path):
    return pd.read_csv(path, float_precision="round_trip", keep_default_na=False)


def hashes(paths):
    return {p.resolve().as_posix(): digest(p) for p in sorted(set(paths))}


def verify(values):
    for path, expected in values.items():
        assert digest(path) == expected, f"Frozen file changed: {path}"


def source(seed):
    return AUG / "outputs" / "typo_mix" / f"seed{seed}"


def protocol():
    return dict(date="2026-10-19", kind="exploratory_frozen_encoder_detector_distribution",
        seeds=list(SEEDS), arms=list(ARMS), conditions=list(CONDITIONS),
        model="Fixed augmented BERT final_model for seeds42/43/44; eval+requires_grad(False)+inference_mode; no optimizer",
        arms_definition={
            "clean_only":"Copy the existing 15000-clean-row detector and exact Mahalanobis thresholds of the same augmented BERT",
            "clean_duplicate":"Fit on original clean training CLS repeated twice (30000 rows); controls LedoitWolf sample-count sensitivity",
            "clean_plus_typo":"Fit on original clean15000 plus one train-only noisy view per source row15000; planned pool50%clean25%light25%medium"},
        augmentation=dict(seed=NOISE_SEED, allocation="Within each label_id: default_rng(seed+label_id).permutation(source indices); first50 light, last50 medium",
            operation="Original internal-character deletion/swap generator; new seed; all unchanged fallbacks retained",
            source="Official 15000 in-scope train rows only; every source contributes one clean and one allocated noisy view; no OOS train",
            policy_counts={"typo_light":7500,"typo_medium":7500}, independent_of_model_seed=True),
        inference=base.protocol()["inference"],
        features="Reuse original augmented-model clean train, validation and test CLS/logits; only noisy train views require new FP32 inference",
        estimator="Same 768D final CLS; per-class means; LedoitWolf(assume_centered=True) of class-centered residuals; no L2/PCA",
        threshold="For new detectors use original clean3000 IS +100 OOS validation only; same two original policies and tie rules; reject score>=threshold; freeze all18 thresholds before test scoring",
        classifier="Identical cached original classifier head argmax in all arms; nearest centroid is diagnostic only",
        primary_comparison="clean_plus_typo minus clean_only, mahalanobis/is_acceptance_95",
        count_control="clean_plus_typo minus clean_duplicate; equal fit-row counts do not create independent source examples",
        clean_guardrails=["oos_f1","oos_recall","in_scope_acceptance"],
        interpretation="Strict descriptive retention requires no negative mean change in any clean guardrail versus clean_only; closed-set classification must be exactly identical. No posthoc tolerance or statistical noninferiority claim.",
        bootstrap="2000 paired stratified original-text resamples, 150 IS intents plus OOS, fixed three model/detector/threshold pairs",
        limitations=["Existing tests informed this experiment; exploratory, not untouched confirmation",
            "Same synthetic noise family, separate training sources and random seed; not established natural/unseen typo robustness",
            "Inherited synthetic labels are not human-verified",
            "Existing3 train-val and2 train-test text overlaps persist; sensitivity excludes test599,938 in every arm without curing historical leakage",
            "Both class means and covariance change, plus validation recalibration; this design does not isolate those components",
            "Paired bootstrap excludes retraining, validation selection, new noise realization and detector-refitting uncertainty"])


def make_data():
    original = csv(source(42) / "train_metadata.csv")
    assert len(original) == 15000 and not original.is_oos.any()
    frame = pd.DataFrame(dict(source_row=original.row_id, original_text=original.text,
        text=original.text, true_label=original.true_label, label_id=original.label_id,
        policy="typo_medium", changed=False, eligible_tokens=0, modified_tokens=0, mutations_json="[]"))
    for label, group in frame.groupby("label_id", sort=True):
        indices = np.random.default_rng(NOISE_SEED+int(label)).permutation(group.index.to_numpy())
        assert len(indices) == 100
        frame.loc[indices[:50], "policy"] = "typo_light"
    for i, row in frame.iterrows():
        variant = make_variants(row.original_text, int(row.source_row), NOISE_SEED)[row.policy]
        for field in ("text", "changed", "eligible_tokens", "modified_tokens"):
            frame.at[i,field] = variant[field]
        frame.at[i,"mutations_json"] = json.dumps(variant["mutations"], ensure_ascii=False)
    frame.to_csv(DATA/"noisy_train.csv", index=False)
    manifest = []
    for arm in ARMS:
        views = ["clean"] if arm == "clean_only" else ["clean", "noisy" if arm == "clean_plus_typo" else "clean"]
        for block,view in enumerate(views):
            manifest.append(pd.DataFrame(dict(arm=arm,fit_row=np.arange(15000)+block*15000,
                source_row=original.row_id,view=view,label_id=original.label_id)))
    pd.concat(manifest, ignore_index=True).to_csv(DATA/"fit_manifest.csv", index=False)
    write(DATA/"data_summary.json",dict(source_rows=15000,noise_seed=NOISE_SEED,
        policy_counts=frame.policy.value_counts().to_dict(), actual_changed_rows=int(frame.changed.sum()),
        noisy_view_changed_fraction=float(frame.changed.mean()),
        mixed_pool_changed_fraction=float(frame.changed.sum()/30000),
        modified_token_counts={str(k):int(v) for k,v in frame.modified_tokens.value_counts().sort_index().items()},
        planned_mixed_pool_counts={"clean":15000,"typo_light":7500,"typo_medium":7500},
        actual_mixed_pool_unchanged_rows=15000+int((~frame.changed).sum()),
        fit_rows={"clean_only":15000,"clean_duplicate":30000,"clean_plus_typo":30000}))


def prepare():
    DATA.mkdir(parents=True,exist_ok=True);OUTPUT.mkdir(exist_ok=True)
    write(ROOT/"PROTOCOL.json",protocol(),immutable=True)
    env=base.prepare_environment()
    assert env == read(PRIOR/"outputs/environment.json")
    frozen=ROOT/"INPUTS_FROZEN.json"
    if frozen.exists():
        result=read(frozen);verify(result["input_hashes"]);return result
    # Verify the selected models and cached arrays against prior immutable gates.
    olddev=read(AUG/"outputs/DEVELOPMENT_FROZEN.json")
    oldtest=read(AUG/"outputs/TEST_COMPLETE.json")
    assert digest(AUG/"outputs/DEVELOPMENT_FROZEN.json")==oldtest["development_frozen_sha256"]
    paths=[Path(__file__),ROOT/"PROTOCOL.json",PRIOR/"distance_model.py",PRIOR/"run_rejection_study.py",
        base.PROJECT/"src/evaluate_oos.py",base.PROJECT/"src/typo_data.py",base.PROJECT/"src/common.py",
        base.PROJECT/"data/raw/data_full.json",base.PROJECT/"outputs/data_audit/duplicates_normalized.csv",
        AUG/"outputs/DEVELOPMENT_FROZEN.json",AUG/"outputs/TEST_COMPLETE.json",AUG/"PROTOCOL.json",
        PRIOR/"outputs/environment.json"]
    for seed in SEEDS:
        folder=source(seed)
        for path in folder.rglob("*"):
            if not path.is_file(): continue
            key=path.resolve().as_posix()
            if key in olddev["development_hashes"]:
                assert digest(path)==olddev["development_hashes"][key]
            elif key in oldtest["output_hashes"]:
                assert digest(path)==oldtest["output_hashes"][key]
            else:
                raise AssertionError(f"Unbound source artifact: {path}")
            paths.append(path)
        paths.append(base.PROJECT/f"outputs/bert/seed{seed}/label_mapping.json")
    make_data()
    paths += list(DATA.glob("*"))
    result=dict(frozen_at=base.now(),runtime=env,input_hashes=hashes(paths))
    write(frozen,result,immutable=True)
    print("INPUTS_FROZEN: three augmented models and all reused arrays bound; training views fixed",flush=True)
    return result


def extract_noisy(seed,frame):
    folder=ROOT/"features"/f"seed{seed}";folder.mkdir(parents=True,exist_ok=True)
    name="noisy_train";fp=folder/f"{name}_cls.npy";lp=folder/f"{name}_logits_fp32.npy"
    record=folder/f"{name}_extraction.json"
    binding=dict(inputs_frozen_sha256=digest(ROOT/"INPUTS_FROZEN.json"),
        model_weights_sha256=digest(source(seed)/"final_model/model.safetensors"),
        text_sha256=hashlib.sha256(json.dumps(frame.text.tolist(),ensure_ascii=False).encode()).hexdigest())
    if record.exists():
        saved=read(record);assert all(saved[k]==v for k,v in binding.items())
        assert digest(fp)==saved["features_sha256"] and digest(lp)==saved["logits_sha256"]
        return np.load(fp,allow_pickle=False)
    modelpath=source(seed)/"final_model"
    tokenizer=AutoTokenizer.from_pretrained(modelpath,local_files_only=True,use_fast=True)
    model=BertForSequenceClassification.from_pretrained(modelpath,local_files_only=True,attn_implementation="eager").to("cuda").float().eval()
    model.requires_grad_(False)
    features=np.empty((len(frame),768),np.float32);logits=np.empty((len(frame),150),np.float32)
    started=time.monotonic();max_tokens=0;truncated=0;unknown=0
    token_ids=tokenizer(frame.text.tolist(),truncation=False,padding=False)["input_ids"]
    lengths=np.array([len(ids) for ids in token_ids])
    max_tokens=int(lengths.max());truncated=int((lengths>64).sum())
    unknown=sum(ids.count(tokenizer.unk_token_id) for ids in token_ids)
    with torch.inference_mode():
        for start in range(0,len(frame),32):
            end=min(start+32,len(frame))
            batch=tokenizer(frame.text.iloc[start:end].tolist(),truncation=True,max_length=64,padding=True,pad_to_multiple_of=8,return_tensors="pt")
            result=model(**{k:v.to("cuda") for k,v in batch.items()},output_hidden_states=True,return_dict=True)
            features[start:end]=result.hidden_states[-1][:,0,:].cpu().numpy()
            logits[start:end]=result.logits.cpu().numpy()
            if end%3200==0 or end==len(frame):print(f"seed{seed}: frozen noisy-train extraction {end}/{len(frame)}",flush=True)
    assert np.isfinite(features).all() and np.isfinite(logits).all()
    np.save(fp,features,allow_pickle=False);np.save(lp,logits,allow_pickle=False)
    frame.to_csv(folder/f"{name}_metadata.csv",index=False)
    write(record,dict(**binding,rows=len(frame),features_sha256=digest(fp),logits_sha256=digest(lp),
        inference=base.protocol()["inference"],max_token_length=max_tokens,truncated_rows=truncated,unknown_tokens=unknown,
        elapsed_seconds=time.monotonic()-started,completed_at=base.now()))
    del model,tokenizer,logits;gc.collect();torch.cuda.empty_cache()
    return features


def evaluate(arm,seed,condition,folder,frame,features,logits,detector,thresholds):
    result=frame.copy();pred=logits.argmax(1);result["predicted_label_id"]=pred
    labels=read(base.PROJECT/f"outputs/bert/seed{seed}/label_mapping.json")["id_to_label"]
    result["predicted_label"]=[labels[str(k)] for k in pred]
    with threadpool_limits(limits=4):distance,nearest=detector.score_with_nearest(features)
    result["mahalanobis_score"]=distance
    result["nearest_centroid_label_id_diagnostic_only"]=nearest
    normal=~frame.is_oos.to_numpy();y=frame.label_id.to_numpy();rows=[]
    for policy in POLICIES:
        threshold=thresholds[f"mahalanobis/{policy}"]["threshold"]
        result[f"mahalanobis_{policy}_reject"]=distance>=threshold
        rows.append(dict(arm=arm,seed=seed,condition=condition,score="mahalanobis",policy=policy,threshold=threshold,
            **base.evaluate(distance,frame.is_oos.to_numpy(),y,pred,threshold),
            closed_set_accuracy=float(np.mean(pred[normal]==y[normal])),
            closed_set_macro_f1=float(f1_score(y[normal],pred[normal],average="macro"))))
    result.to_csv(folder/f"{condition}_predictions.csv",index=False)
    pd.DataFrame(rows).to_csv(folder/f"{condition}_metrics.csv",index=False)
    return rows


def development():
    inputs=prepare()
    gate=OUTPUT/"DEVELOPMENT_FROZEN.json"
    if gate.exists():verify(read(gate)["development_hashes"]);print("Development already frozen",flush=True);return
    noise=csv(DATA/"noisy_train.csv");allrows=[];paths=[]
    for seed in SEEDS:
        original=source(seed)
        clean=np.load(original/"train_cls.npy",allow_pickle=False)
        trainmeta=csv(original/"train_metadata.csv")
        assert np.array_equal(trainmeta.row_id,noise.source_row) and trainmeta.text.equals(noise.original_text)
        assert np.array_equal(trainmeta.label_id,noise.label_id)
        noisy=extract_noisy(seed,noise)
        paths += [p for p in (ROOT/"features"/f"seed{seed}").glob("*") if p.is_file()]
        vf=np.load(original/"validation_cls.npy",allow_pickle=False)
        vl=np.load(original/"validation_logits_fp32.npy",allow_pickle=False)
        vm=csv(original/"validation_metadata.csv")
        for arm in ARMS:
            folder=OUTPUT/arm/f"seed{seed}";folder.mkdir(parents=True,exist_ok=True)
            done=folder/"DEVELOPMENT_COMPLETE.json"
            if done.exists():
                verify(read(done)["output_hashes"])
                allrows+=csv(folder/"validation_metrics.csv").to_dict("records")
                paths += [p for p in folder.glob("*") if p.is_file()];continue
            if arm=="clean_only":
                shutil.copyfile(original/"distance_model.npz",folder/"distance_model.npz")
                detector=SharedMahalanobisDetector.load(folder/"distance_model.npz")
                thresholds={k:v for k,v in read(original/"frozen_thresholds.json").items() if k.startswith("mahalanobis/")}
            else:
                features=np.concatenate([clean, clean if arm=="clean_duplicate" else noisy])
                targets=np.tile(trainmeta.label_id.to_numpy(),2)
                with threadpool_limits(limits=4):detector=SharedMahalanobisDetector().fit(features,targets)
                detector.save(folder/"distance_model.npz")
                thresholds=None
                del features,targets
            with threadpool_limits(limits=4):valscore=detector.score_samples(vf)
            curve=base.threshold_curve(valscore,vm.is_oos.to_numpy())
            curve.to_csv(folder/"validation_mahalanobis_threshold_curve.csv",index=False)
            calculated={f"mahalanobis/{p}":base.choose_threshold(curve,p) for p in POLICIES}
            if thresholds is not None:assert calculated==thresholds,"Copied clean reference differs"
            else:thresholds=calculated
            write(folder/"frozen_thresholds.json",thresholds)
            allrows+=evaluate(arm,seed,"validation",folder,vm,vf,vl,detector,thresholds)
            localpaths=[p for p in folder.glob("*") if p.is_file()]
            write(done,dict(completed_at=base.now(),inputs_frozen_sha256=digest(ROOT/"INPUTS_FROZEN.json"),
                arm=arm,seed=seed,fit_rows=detector.n_samples_seen_,shrinkage=detector.shrinkage_,
                class_counts=detector.class_counts_.tolist(),output_hashes=hashes(localpaths)))
            paths+=localpaths+[done]
            print(f"{arm}/seed{seed}: fit{detector.n_samples_seen_}, shrinkage={detector.shrinkage_:.8g}, clean-validation thresholds ready",flush=True)
        del clean,noisy,vf,vl,detector;gc.collect()
    verify(inputs["input_hashes"])
    pd.DataFrame(allrows).to_csv(OUTPUT/"validation_metrics_by_seed.csv",index=False)
    paths.append(OUTPUT/"validation_metrics_by_seed.csv")
    write(gate,dict(frozen_at=base.now(),inputs_frozen_sha256=digest(ROOT/"INPUTS_FROZEN.json"),
        detectors=9,validation_thresholds=18,development_hashes=hashes(paths),
        status="All detector statistics and clean-validation thresholds frozen before new test scoring"),immutable=True)
    print("DEVELOPMENT_FROZEN: nine detectors, eighteen thresholds; BERT unchanged",flush=True)


def test():
    if (OUTPUT/"TEST_COMPLETE.json").exists():raise FileExistsError("Test complete: choose a new experiment, do not overwrite")
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
                rows+=evaluate(arm,seed,condition,folder,frame,features,logits,detector,read(folder/"frozen_thresholds.json"))
                paths += [folder/f"{condition}_predictions.csv",folder/f"{condition}_metrics.csv"]
            print(f"seed{seed}/{condition}: three fixed-detector decisions from identical cached BERT predictions",flush=True)
    pd.DataFrame(rows).to_csv(OUTPUT/"metrics_by_seed.csv",index=False)
    paths.append(OUTPUT/"metrics_by_seed.csv")
    verify(inputs["input_hashes"]);verify(gate["development_hashes"])
    write(OUTPUT/"TEST_COMPLETE.json",dict(started_at=started,completed_at=base.now(),test_settings=len(rows),
        inputs_frozen_sha256=digest(ROOT/"INPUTS_FROZEN.json"),
        development_frozen_sha256=digest(OUTPUT/"DEVELOPMENT_FROZEN.json"),
        output_hashes=hashes(paths),original_inputs_and_models_unchanged=True))
    print("TEST_COMPLETE:54 settings; all source models and predictions unchanged",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=("prepare","development","test"),required=True)
    args=parser.parse_args();{"prepare":prepare,"development":development,"test":test}[args.stage]()
