"""Matched continuation training with train-only spelling augmentation.

Development fits all six models and clean-validation rejection rules before
the separate test stage evaluates any newly trained model on test examples.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "BERT_Rejection_Study_20261019"
sys.path.insert(0, str(PRIOR))
import run_rejection_study as base
from distance_model import SharedMahalanobisDetector
from typo_data import make_variants
from train_bert import encode_dataset
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

PROJECT = base.PROJECT
OUTPUT = ROOT / "outputs"
DATA = ROOT / "data"
SEEDS = (42, 43, 44)
ARMS = ("clean_continue", "typo_mix")
CONDITIONS = ("clean", "typo_light", "typo_medium")
SCORES, POLICIES = base.SCORES, base.POLICIES
AUG_SEED = 31020261019
TRAIN_BATCH = 16
EVAL_BATCH = 32
LR = 1e-5
STEPS = 938


def read(path):
    return base.read_json(path)


def csv(path):
    return pd.read_csv(path, float_precision="round_trip", keep_default_na=False)


def digest(path):
    return base.digest(path)


def hashes(paths):
    return {p.resolve().as_posix(): digest(p) for p in sorted(set(paths))}


def verify(values):
    for path, expected in values.items():
        assert digest(path) == expected, f"Frozen input changed: {path}"


def protocol():
    return dict(date="2026-10-19", kind="exploratory_matched_continuation_training",
        seeds=list(SEEDS), arms=list(ARMS), conditions=list(CONDITIONS),
        initialization="Each arm starts from its seed's identical original fine-tuned BERT checkpoint; fresh identical AdamW and LR schedule.",
        train=dict(epochs=1, examples=15000, batch_size=TRAIN_BATCH, optimizer_updates=STEPS, last_batch_size=8,
                   learning_rate=LR, optimizer="AdamW", weight_decay=.01, warmup_ratio=.1,
                   gradient_clip_norm=1., precision="CUDA autocast bfloat16 with FP32 parameters; no GradScaler skips",
                   row_order="numpy.default_rng(model_seed).permutation(15000), same order across paired arms",
                   padding="dynamic to multiple of 8; equal presentations/updates, not equal FLOPs",
                   checkpoint="Fixed end of epoch 1; no checkpoint selection"),
        augmentation=dict(seed=AUG_SEED, source="Official 15000 IS train rows only; no OOS train",
                          per_class_policy_counts={"clean":50,"typo_light":25,"typo_medium":25},
                          allocation="numpy.default_rng(augmentation_seed+label_id).permutation(class source rows)",
                          operation="Reuse internal character deletion/swap generator; every source row once per arm; some allocated noisy rows cannot change",
                          same_frozen_data_across_model_seeds=True),
        inference=base.protocol()["inference"],
        detector="Each new model: final hidden CLS; original clean 15000 train rows only; per-class means and class-residual LedoitWolf covariance; no L2/PCA",
        thresholds="Re-estimate for each new model using the same original clean validation rows and original two policies, exact >= rule. Old numeric cutoffs are not transferred to a new representation.",
        causal_comparison="typo_mix minus clean_continue; augmentation plus identical post-training reconstruction of detector and thresholds",
        historical_comparison="typo_mix minus original; includes additional optimization, so not isolated augmentation effect",
        focus=dict(score="mahalanobis",policy="is_acceptance_95",noise_metrics=["IS false rejection rate", "rejected despite correct classifier", "IS correct and accepted rate"],
                   clean_metrics=["oos_f1","oos_recall","in_scope_acceptance","closed_set_accuracy"]),
        interpretation="Report signed changes for both noisy levels and all four clean guardrails. Strict descriptive retention means no mean decrease in any clean guardrail vs original; any decrease is a tradeoff, not hidden by noisy improvement. This is not a statistical non-inferiority claim.",
        gate="All six models, clean-train distance statistics and 36 validation thresholds frozen before new-model test inference",
        bootstrap="2000 paired stratified draws, same original texts across arms/seeds/conditions; 150 IS intents plus OOS; fixed trained models and thresholds",
        limitations=["Existing test outcomes informed this follow-up; exploratory, not a new untouched test",
                     "Train/test noise shares a generator family but uses separate source rows and seeds; does not establish robustness to natural/unseen noise",
                     "Original train-val/test exact duplicates are inherited; report exclusion sensitivity without changing official training data",
                     "Labels inherited for synthetic spelling variants; semantic equivalence not human-verified"],
        method_context="https://aclanthology.org/2020.acl-main.138/ supports clean/noisy mixture as a training idea in sequence labeling; this intent/OOS experiment is not an exact replication")


def runtime():
    environment = base.prepare_environment()
    assert environment == read(PRIOR / "outputs/environment.json"), "Inference runtime differs from prior clean study"
    assert torch.cuda.is_bf16_supported()
    torch.backends.cudnn.deterministic = True
    return environment


def make_training_data():
    raw = read(PROJECT / "data/raw/data_full.json")
    mapping = read(PROJECT / "outputs/bert/seed42/label_mapping.json")["label_to_id"]
    frame = pd.DataFrame([dict(source_row=i, original_text=t, text=t, true_label=y, label_id=mapping[y],
                              policy="clean", changed=False, eligible_tokens=0, modified_tokens=0, mutations_json="[]")
                          for i,(t,y) in enumerate(raw["train"])])
    mixed = frame.copy()
    for label_id, group in frame.groupby("label_id", sort=True):
        indices = np.random.default_rng(AUG_SEED+int(label_id)).permutation(group.index.to_numpy())
        assert len(indices) == 100
        mixed.loc[indices[50:75], "policy"] = "typo_light"
        mixed.loc[indices[75:], "policy"] = "typo_medium"
    for i, row in mixed.iterrows():
        variant = make_variants(row.original_text, int(row.source_row), AUG_SEED)[row.policy]
        for col in ("text", "changed", "eligible_tokens", "modified_tokens"):
            mixed.at[i,col] = variant[col]
        mixed.at[i,"mutations_json"] = json.dumps(variant["mutations"], ensure_ascii=False)
        frame.at[i,"eligible_tokens"] = variant["eligible_tokens"]
    frame.to_csv(DATA / "train_clean.csv",index=False)
    mixed.to_csv(DATA / "train_mix.csv",index=False)
    coverage = mixed.groupby("policy").agg(rows=("source_row","size"), changed=("changed","sum"),
                       modified_tokens=("modified_tokens","sum"), two_tokens_changed=("modified_tokens",lambda v:int((v==2).sum())))
    coverage.to_csv(DATA / "augmentation_coverage.csv")
    base.write_json(DATA / "data_summary.json",dict(train_rows=len(frame),augmentation_seed=AUG_SEED,
                    policy_counts=mixed.policy.value_counts().to_dict(), changed_rows=int(mixed.changed.sum()),
                    modified_token_counts={str(k):int(v) for k,v in mixed.modified_tokens.value_counts().sort_index().items()},
                    train_mix_sha256=digest(DATA/"train_mix.csv")))


def prepare():
    ROOT.mkdir(exist_ok=True); OUTPUT.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)
    base.write_json(ROOT/"PROTOCOL.json",protocol(),immutable=True)
    env = runtime()
    frozen_path = ROOT/"INPUTS_FROZEN.json"
    if frozen_path.exists():
        saved = read(frozen_path)
        assert saved["runtime"] == env
        verify(saved["input_hashes"])
        return saved
    make_training_data()
    original = read(PRIOR/"outputs/DEVELOPMENT_FROZEN.json")
    paths = [Path(__file__),ROOT/"PROTOCOL.json",PRIOR/"outputs/DEVELOPMENT_FROZEN.json",PRIOR/"outputs/environment.json",
             PROJECT/"src/typo_data.py",PROJECT/"src/train_bert.py",PROJECT/"src/common.py"]
    for key,parent in (("source_hashes",PROJECT),("code_hashes",PRIOR),("development_hashes",PRIOR/"outputs")):
        for rel,expected in original[key].items():
            p=parent/rel
            assert digest(p)==expected
            paths.append(p)
    paths += list(DATA.glob("*"))
    typo = PROJECT/"outputs/robustness/typo_data"
    paths += [typo/f"{c}.csv" for c in CONDITIONS]
    paths += [typo/"summary.json",PROJECT/"outputs/data_audit/duplicates_normalized.csv"]
    previous = PRIOR/"typo_stress_test/outputs"
    completion=read(previous/"TEST_COMPLETE.json")
    assert digest(PRIOR/"typo_stress_test/INPUTS_FROZEN.json")==completion["inputs_frozen_sha256"]
    for rel,expected in completion["output_hashes"].items():
        assert digest(previous/rel)==expected
    paths += [previous/"TEST_COMPLETE.json",previous/"metrics_by_seed.csv"]
    paths += [previous/f"seed{s}/{c}_predictions.csv" for s in SEEDS for c in CONDITIONS]
    saved=dict(frozen_at=base.now(),runtime=env,input_hashes=hashes(paths))
    base.write_json(frozen_path,saved,immutable=True)
    return saved


def train(arm,seed,folder):
    done=folder/"TRAINING_COMPLETE.json"
    if done.exists():
        result=read(done)
        assert result["inputs_frozen_sha256"]==digest(ROOT/"INPUTS_FROZEN.json")
        verify(result["output_hashes"])
        return
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    checkpoint=PROJECT/f"outputs/bert/seed{seed}/best_model"
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,use_fast=True)
    model=BertForSequenceClassification.from_pretrained(checkpoint,local_files_only=True,attn_implementation="eager").to("cuda")
    model.train().requires_grad_(True)
    frame=csv(DATA/("train_mix.csv" if arm=="typo_mix" else "train_clean.csv"))
    order=np.random.default_rng(seed).permutation(len(frame))
    pd.DataFrame(dict(presentation_index=np.arange(len(frame)),source_row=frame.source_row.to_numpy()[order])).to_csv(folder/"training_order.csv",index=False)
    ordered=frame.iloc[order]
    dataset=encode_dataset(tokenizer,ordered.text.tolist(),ordered.label_id.tolist(),64)
    loader=DataLoader(dataset,batch_size=TRAIN_BATCH,shuffle=False,num_workers=0,pin_memory=True,
                      collate_fn=DataCollatorWithPadding(tokenizer,pad_to_multiple_of=8))
    assert len(loader)==STEPS
    optimizer=AdamW(model.parameters(),lr=LR,weight_decay=.01)
    scheduler=get_linear_schedule_with_warmup(optimizer,int(round(STEPS*.1)),STEPS)
    records=[];seen=0;started=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for step,batch in enumerate(loader,1):
        batch={k:v.to("cuda",non_blocking=True) for k,v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            loss=model(**batch).loss
        assert torch.isfinite(loss)
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        lr=optimizer.param_groups[0]["lr"]
        optimizer.step();scheduler.step()
        n=int(batch["labels"].shape[0]);seen+=n
        records.append(dict(step=step,batch_size=n,samples_seen=seen,learning_rate=lr,loss=float(loss.detach()),grad_norm=float(grad)))
        if step%100==0 or step==STEPS:
            print(f"{arm} seed={seed} train {step}/{STEPS}; mean_loss={np.mean([r['loss'] for r in records]):.5f}; elapsed={time.monotonic()-started:.1f}s",flush=True)
    assert seen==15000 and len(records)==STEPS
    pd.DataFrame(records).to_csv(folder/"training_steps.csv",index=False)
    model.save_pretrained(folder/"final_model",safe_serialization=True)
    tokenizer.save_pretrained(folder/"final_model")
    paths=[folder/"training_order.csv",folder/"training_steps.csv"]+list((folder/"final_model").glob("*"))
    base.write_json(done,dict(arm=arm,seed=seed,epochs=1,presentations=seen,optimizer_updates=len(records),
         train_precision="bfloat16 autocast, float32 parameters",inputs_frozen_sha256=digest(ROOT/"INPUTS_FROZEN.json"),
         mean_batch_loss=float(np.mean([r["loss"] for r in records])),elapsed_seconds=time.monotonic()-started,
         peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),completed_at=base.now(),output_hashes=hashes(paths)))
    del model,optimizer,scheduler,tokenizer,loader,dataset
    gc.collect();torch.cuda.empty_cache()


def load_final(folder):
    tokenizer=AutoTokenizer.from_pretrained(folder/"final_model",local_files_only=True,use_fast=True)
    model=BertForSequenceClassification.from_pretrained(folder/"final_model",local_files_only=True,attn_implementation="eager").to("cuda").float().eval()
    model.requires_grad_(False)
    return model,tokenizer


def extract(folder,name,frame,model,tokenizer):
    record_path=folder/f"{name}_extraction.json"
    fp=folder/f"{name}_cls.npy";lp=folder/f"{name}_logits_fp32.npy"
    binding=dict(training_complete_sha256=digest(folder/"TRAINING_COMPLETE.json"),
                 inputs_frozen_sha256=digest(ROOT/"INPUTS_FROZEN.json"),
                 text_sha256=hashlib.sha256(json.dumps(frame.text.tolist(),ensure_ascii=False).encode()).hexdigest())
    if record_path.exists():
        rec=read(record_path)
        assert all(rec[k]==v for k,v in binding.items())
        assert digest(fp)==rec["features_sha256"] and digest(lp)==rec["logits_sha256"]
        return np.load(fp,allow_pickle=False),np.load(lp,allow_pickle=False)
    features=np.empty((len(frame),768),np.float32);logits=np.empty((len(frame),150),np.float32)
    started=time.monotonic()
    with torch.inference_mode():
        for start in range(0,len(frame),EVAL_BATCH):
            end=min(start+EVAL_BATCH,len(frame))
            batch=tokenizer(frame.text.iloc[start:end].tolist(),truncation=True,max_length=64,padding=True,pad_to_multiple_of=8,return_tensors="pt")
            result=model(**{k:v.to("cuda") for k,v in batch.items()},output_hidden_states=True,return_dict=True)
            features[start:end]=result.hidden_states[-1][:,0,:].cpu().numpy()
            logits[start:end]=result.logits.cpu().numpy()
    assert np.isfinite(features).all() and np.isfinite(logits).all()
    np.save(fp,features,allow_pickle=False);np.save(lp,logits,allow_pickle=False)
    frame.to_csv(folder/f"{name}_metadata.csv",index=False)
    base.write_json(record_path,dict(**binding,rows=len(frame),features_sha256=digest(fp),logits_sha256=digest(lp),
                                   elapsed_seconds=time.monotonic()-started,completed_at=base.now()))
    print(f"{folder.parent.name}/{folder.name} {name} extracted {len(frame)} rows",flush=True)
    return features,logits


def evaluate(arm,seed,condition,folder,frame,features,logits,detector,thresholds):
    result=frame.copy();pred=logits.argmax(1);result["predicted_label_id"]=pred
    label_map=read(PROJECT/f"outputs/bert/seed{seed}/label_mapping.json")["id_to_label"]
    result["predicted_label"]=[label_map[str(k)] for k in pred]
    with threadpool_limits(limits=4):
        distance,nearest=detector.score_with_nearest(features)
    result["nearest_centroid_label_id_diagnostic_only"]=nearest
    arrays=base.anomaly_scores(logits);arrays["mahalanobis"]=distance
    normal=~frame.is_oos.to_numpy();y=frame.label_id.to_numpy()
    rows=[]
    for score in SCORES:
        result[f"{score}_score"]=arrays[score]
        for policy in POLICIES:
            threshold=thresholds[f"{score}/{policy}"]["threshold"]
            result[f"{score}_{policy}_reject"]=arrays[score]>=threshold
            rows.append(dict(arm=arm,seed=seed,condition=condition,score=score,policy=policy,threshold=threshold,
              **base.evaluate(arrays[score],frame.is_oos.to_numpy(),y,pred,threshold),
              closed_set_accuracy=float(np.mean(pred[normal]==y[normal])),
              closed_set_macro_f1=float(f1_score(y[normal],pred[normal],average="macro"))))
    result.to_csv(folder/f"{condition}_predictions.csv",index=False)
    pd.DataFrame(rows).to_csv(folder/f"{condition}_metrics.csv",index=False)
    return rows


def development():
    frozen=prepare()
    if (OUTPUT/"DEVELOPMENT_FROZEN.json").exists():
        verify(read(OUTPUT/"DEVELOPMENT_FROZEN.json")["development_hashes"])
        print("All development already frozen; refusing to refit",flush=True);return
    raw=read(PROJECT/"data/raw/data_full.json")
    mapping=read(PROJECT/"outputs/bert/seed42/label_mapping.json")["label_to_id"]
    trainframe=base.metadata(raw["train"],["train"]*15000,mapping)
    valframe=base.metadata(raw["val"]+raw["oos_val"],["val"]*3000+["oos_val"]*100,mapping)
    allrows=[];devpaths=[]
    for seed in SEEDS:
        for arm in ARMS:
            folder=OUTPUT/arm/f"seed{seed}";folder.mkdir(parents=True,exist_ok=True)
            train(arm,seed,folder)
            done=folder/"DEVELOPMENT_COMPLETE.json"
            if done.exists():
                verify(read(done)["output_hashes"])
                allrows += csv(folder/"validation_metrics.csv").to_dict("records")
                devpaths += [p for p in folder.rglob("*") if p.is_file()]
                continue
            model,tokenizer=load_final(folder)
            features,_=extract(folder,"train",trainframe,model,tokenizer)
            valfeatures,vallogits=extract(folder,"validation",valframe,model,tokenizer)
            del model,tokenizer;gc.collect();torch.cuda.empty_cache()
            with threadpool_limits(limits=4):
                detector=SharedMahalanobisDetector().fit(features,trainframe.label_id.to_numpy())
                distances=detector.score_samples(valfeatures)
            detector.save(folder/"distance_model.npz",overwrite=True)
            arrays=base.anomaly_scores(vallogits);arrays["mahalanobis"]=distances
            thresholds={}
            for score in SCORES:
                curve=base.threshold_curve(arrays[score],valframe.is_oos.to_numpy())
                curve.to_csv(folder/f"validation_{score}_threshold_curve.csv",index=False)
                for policy in POLICIES:
                    thresholds[f"{score}/{policy}"]=base.choose_threshold(curve,policy)
            base.write_json(folder/"frozen_thresholds.json",thresholds)
            rows=evaluate(arm,seed,"validation",folder,valframe,valfeatures,vallogits,detector,thresholds)
            allrows+=rows
            paths=[p for p in folder.rglob("*") if p.is_file()]
            base.write_json(done,dict(completed_at=base.now(),inputs_frozen_sha256=digest(ROOT/"INPUTS_FROZEN.json"),
                         shrinkage=detector.shrinkage_,class_counts=detector.class_counts_.tolist(),output_hashes=hashes(paths)))
            devpaths+=paths+[done]
            del features,valfeatures,vallogits,detector;gc.collect()
            print(f"{arm} seed={seed}: trained, clean-train statistics fit, clean-validation thresholds frozen",flush=True)
    verify(frozen["input_hashes"])
    pd.DataFrame(allrows).to_csv(OUTPUT/"validation_metrics_by_seed.csv",index=False)
    devpaths.append(OUTPUT/"validation_metrics_by_seed.csv")
    base.write_json(OUTPUT/"DEVELOPMENT_FROZEN.json",dict(frozen_at=base.now(),
             inputs_frozen_sha256=digest(ROOT/"INPUTS_FROZEN.json"),development_hashes=hashes(devpaths),
             models=6,validation_thresholds=36,status="All new-model development frozen before new-model test inference"))
    print("DEVELOPMENT_FROZEN: six models ready for test",flush=True)


def test():
    if (OUTPUT/"TEST_COMPLETE.json").exists():
        raise FileExistsError("Test complete; do not overwrite frozen evaluation")
    frozen=read(ROOT/"INPUTS_FROZEN.json");verify(frozen["input_hashes"])
    assert runtime()==frozen["runtime"]
    development=read(OUTPUT/"DEVELOPMENT_FROZEN.json")
    assert development["inputs_frozen_sha256"]==digest(ROOT/"INPUTS_FROZEN.json")
    verify(development["development_hashes"])
    rows=[];result_paths=[]
    frames={c:csv(PROJECT/f"outputs/robustness/typo_data/{c}.csv") for c in CONDITIONS}
    for seed in SEEDS:
        for arm in ARMS:
            folder=OUTPUT/arm/f"seed{seed}"
            model,tokenizer=load_final(folder)
            detector=SharedMahalanobisDetector.load(folder/"distance_model.npz")
            thresholds=read(folder/"frozen_thresholds.json")
            for condition,frame in frames.items():
                features,logits=extract(folder,condition,frame,model,tokenizer)
                rows+=evaluate(arm,seed,condition,folder,frame,features,logits,detector,thresholds)
                result_paths += [p for p in folder.glob(f"{condition}_*") if p.is_file()]
            del model,tokenizer,detector,features,logits;gc.collect();torch.cuda.empty_cache()
    original=csv(PRIOR/"typo_stress_test/outputs/metrics_by_seed.csv");original.insert(0,"arm","original")
    pd.concat([original,pd.DataFrame(rows)],ignore_index=True).to_csv(OUTPUT/"metrics_by_seed.csv",index=False)
    result_paths.append(OUTPUT/"metrics_by_seed.csv")
    verify(frozen["input_hashes"]);verify(development["development_hashes"])
    base.write_json(OUTPUT/"TEST_COMPLETE.json",dict(completed_at=base.now(),evaluated_new_settings=len(rows),
               total_test_settings=len(rows)+len(original),inputs_frozen_sha256=digest(ROOT/"INPUTS_FROZEN.json"),
               development_frozen_sha256=digest(OUTPUT/"DEVELOPMENT_FROZEN.json"),output_hashes=hashes(result_paths),
               original_inputs_and_new_development_unchanged=True))
    print("TEST_COMPLETE: 162 test settings, no test-driven changes",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=("prepare","development","test"),required=True)
    args=parser.parse_args()
    {"prepare":prepare,"development":development,"test":test}[args.stage]()
