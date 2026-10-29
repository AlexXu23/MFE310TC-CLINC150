"""Frozen-checkpoint OOS study with development and test stages kept separate.

No optimizer is created. Train rows fit representation statistics; validation
rows select all thresholds. The test stage requires a hash-verified development
freeze. The existing shared experiment snapshot is read-only throughout.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent / "CLINC150_Team_20261019"
OUTPUT = ROOT / "outputs"
for key, value in {"USE_TF":"0", "USE_FLAX":"0", "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1",
                   "HF_HOME":str(ROOT / ".hf-cache"), "TOKENIZERS_PARALLELISM":"false"}.items():
    os.environ[key] = value
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer, BertForSequenceClassification

from distance_model import SharedMahalanobisDetector
sys.path.insert(0, str(PROJECT / "src"))
from evaluate_oos import anomaly_scores, choose_threshold, evaluate, threshold_curve

SEEDS = (42, 43, 44)
SCORES = ("msp", "entropy", "mahalanobis")
POLICIES = ("max_oos_f1", "is_acceptance_95")
BATCH_SIZE = 32
MAX_LENGTH = 64


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value, *, immutable=False):
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    if immutable and path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise FileExistsError(f"Immutable file differs: {path}")
        return
    path.write_text(text, encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def protocol():
    return {
        "date":"2026-10-19", "kind":"post_hoc_followup_experiment", "seeds":list(SEEDS),
        "question":"Does train-fitted representation distance improve OOS detection over MSP and entropy for the same frozen BERT?",
        "feature":"outputs.hidden_states[-1][:,0,:], 768-dimensional final CLS, no L2 normalization or PCA",
        "statistics":"Per-seed class means and shared LedoitWolf(assume_centered=True) on float64 train class-centered residuals",
        "score":"minimum squared Mahalanobis distance over all 150 training class means; larger means OOS",
        "training_rows":15000, "oos_train_used":False,
        "classification":"Original classifier head argmax from current forward; nearest centroid never substitutes the classifier",
        "inference":{"precision":"float32 for features AND logits in all new comparison arms", "autocast":False,
                     "tf32":False,"attention_implementation":"eager","batch_size":BATCH_SIZE,"max_length":MAX_LENGTH,
                     "padding":"dynamic to multiple of 8", "device":"cuda", "eval_mode":True},
        "primary":{"comparison":"mahalanobis minus msp", "threshold_policy":"max_oos_f1", "metric":"oos_f1"},
        "secondary":{"comparisons":["mahalanobis minus entropy"], "policies":list(POLICIES),
                     "metrics":["oos_auroc","oos_recall","in_scope_acceptance","combined_151_accuracy"]},
        "threshold":"reject score>=threshold; exact unique validation scores plus nextafter(max,+inf)",
        "tie_breaks":{"max_oos_f1":["oos_f1","in_scope_acceptance","threshold"],
                      "is_acceptance_95":["filter validation IS acceptance>=.95-1e-12","oos_recall","oos_f1","in_scope_acceptance","threshold"]},
        "test_gate":"All three seeds' statistics and validation thresholds frozen and hashed before test-stage inference",
        "model_search":"One feature, one covariance estimator, one distance; no test-guided retuning",
        "comparability":"All three new arms share same current FP32 forward. Old FP16 logits are historical references; report drift separately.",
        "limits":"Original test has been inspected previously; exploratory followup, not a new untouched confirmation set. 95% acceptance only constrains validation.",
        "bootstrap":"2000 paired stratified draws by 150 IS intents plus OOS; fixed three model/threshold pairs; excludes training and validation-fitting uncertainty",
        "reference":"https://arxiv.org/abs/2101.03778 (distance formulation); Ledoit-Wolf is a pre-specified estimator choice, not a claim of exact paper reproduction",
    }


def prepare_environment():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this recorded experiment")
    torch.set_num_threads(6)
    torch.manual_seed(20261019)
    np.random.seed(20261019)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                transformers=transformers.__version__, numpy=np.__version__, sklearn=sklearn.__version__,
                protocol_inference=protocol()["inference"])


def metadata(rows, sources, mapping):
    return pd.DataFrame({"row_id":np.arange(len(rows)), "source_split":sources,
                         "text":[r[0] for r in rows],"true_label":[r[1] for r in rows],
                         "label_id":[mapping.get(r[1], -1) for r in rows],
                         "is_oos":[r[1] not in mapping for r in rows]})


def load_model(seed):
    path = PROJECT / f"outputs/bert/seed{seed}/best_model"
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    model = BertForSequenceClassification.from_pretrained(path, local_files_only=True, attn_implementation="eager")
    model.to(device="cuda", dtype=torch.float32).eval()
    model.requires_grad_(False)
    return model, tokenizer


def extract(seed, split, frame, model, tokenizer):
    folder = OUTPUT / f"seed{seed}"
    feature_path = folder / f"{split}_cls.npy"
    logit_path = folder / f"{split}_logits_fp32.npy"
    record_path = folder / f"{split}_extraction.json"
    input_hash = hashlib.sha256(json.dumps(frame.text.tolist(), ensure_ascii=False).encode()).hexdigest()
    if record_path.exists():
        record = read_json(record_path)
        assert record["text_sha256"] == input_hash and record["protocol_sha256"] == digest(ROOT / "PROTOCOL.json")
        assert record["feature_sha256"] == digest(feature_path) and record["logit_sha256"] == digest(logit_path)
        return np.load(feature_path, allow_pickle=False), np.load(logit_path, allow_pickle=False)
    started = time.monotonic()
    features = np.empty((len(frame), 768), dtype=np.float32)
    logits = np.empty((len(frame), 150), dtype=np.float32)
    torch.cuda.reset_peak_memory_stats()
    texts = frame.text.tolist()
    with torch.inference_mode():
        for start in range(0, len(texts), BATCH_SIZE):
            end = min(start+BATCH_SIZE, len(texts))
            batch = tokenizer(texts[start:end], truncation=True, max_length=MAX_LENGTH,
                              padding=True, pad_to_multiple_of=8, return_tensors="pt")
            batch = {k:v.to("cuda") for k,v in batch.items()}
            result = model(**batch, output_hidden_states=True, return_dict=True)
            features[start:end] = result.hidden_states[-1][:, 0, :].cpu().numpy()
            logits[start:end] = result.logits.cpu().numpy()
            if start % (50*BATCH_SIZE) == 0 or end == len(texts):
                print(f"seed={seed} {split} {end}/{len(texts)} rows; elapsed={time.monotonic()-started:.1f}s", flush=True)
    assert np.isfinite(features).all() and np.isfinite(logits).all()
    np.save(feature_path, features, allow_pickle=False)
    np.save(logit_path, logits, allow_pickle=False)
    frame.to_csv(folder / f"{split}_metadata.csv", index=False)
    write_json(record_path,dict(seed=seed, split=split,rows=len(frame),feature_shape=list(features.shape),
                               text_sha256=input_hash,protocol_sha256=digest(ROOT / "PROTOCOL.json"),
                               feature_sha256=digest(feature_path),logit_sha256=digest(logit_path),
                               elapsed_seconds=time.monotonic()-started,peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
                               completed_at=now()))
    return features, logits


def old_forward_drift(seed, split, logits):
    old = np.load(PROJECT / f"outputs/bert/seed{seed}/{split}_logits.npy", allow_pickle=False).astype(float)
    changed = np.flatnonzero(old.argmax(axis=1) != logits.argmax(axis=1))
    return dict(reference="Archived original FP16 forward",current="Unified FP32 eager forward",rows=len(old),
                changed_top1_count=len(changed),changed_top1_row_ids=changed.tolist(),
                max_abs_logit_difference=float(np.max(np.abs(old-logits))),
                mean_abs_logit_difference=float(np.mean(np.abs(old-logits))))


def evaluate_frame(seed, split, frame, logits, distance, nearest, decisions):
    score_arrays = anomaly_scores(logits)
    score_arrays["mahalanobis"] = distance
    frame = frame.copy()
    prediction = logits.argmax(axis=1)
    mapping = read_json(PROJECT / f"outputs/bert/seed{seed}/label_mapping.json")["id_to_label"]
    frame["predicted_label_id"] = prediction
    frame["predicted_label"] = [mapping[str(k)] for k in prediction]
    frame["nearest_centroid_label_id_diagnostic_only"] = nearest
    rows = []
    for score in SCORES:
        values = score_arrays[score]
        frame[f"{score}_score"] = values
        for policy in POLICIES:
            decision = decisions[f"{score}/{policy}"]
            threshold = decision["threshold"]
            frame[f"{score}_{policy}_reject"] = values >= threshold
            rows.append(dict(seed=seed,split=split,score=score,policy=policy,threshold=threshold,
                             **evaluate(values,frame.is_oos.to_numpy(),frame.label_id.to_numpy(),prediction,threshold)))
    folder = OUTPUT / f"seed{seed}"
    frame.to_csv(folder / f"{split}_predictions.csv",index=False)
    pd.DataFrame(rows).to_csv(folder / f"{split}_metrics.csv",index=False)
    return rows


def development():
    if (OUTPUT / "DEVELOPMENT_FROZEN.json").exists():
        print("Development already frozen; refusing to refit.")
        return
    OUTPUT.mkdir(exist_ok=True)
    write_json(ROOT / "PROTOCOL.json",protocol(),immutable=True)
    write_json(OUTPUT / "environment.json",prepare_environment())
    data_path = PROJECT / "data/raw/data_full.json"
    data = read_json(data_path)
    source_paths = [data_path, PROJECT / "src/evaluate_oos.py"]
    source_paths += [p for seed in SEEDS for p in (PROJECT / f"outputs/bert/seed{seed}/best_model").iterdir() if p.is_file()]
    source_paths += [PROJECT / f"outputs/bert/seed{seed}/label_mapping.json" for seed in SEEDS]
    source_hashes = {p.relative_to(PROJECT).as_posix():digest(p) for p in source_paths}
    code_hashes = {name:digest(ROOT/name) for name in ("run_rejection_study.py","distance_model.py","PROTOCOL.json")}
    all_rows = []
    frozen_paths = []
    for seed in SEEDS:
        folder = OUTPUT / f"seed{seed}"
        folder.mkdir(exist_ok=True)
        mapping = read_json(PROJECT / f"outputs/bert/seed{seed}/label_mapping.json")["label_to_id"]
        train = metadata(data["train"],["train"]*15000,mapping)
        val = metadata(data["val"]+data["oos_val"],["val"]*3000+["oos_val"]*100,mapping)
        assert len(train)==15000 and train.label_id.value_counts().eq(100).all()
        assert len(val)==3100 and val.is_oos.sum()==100
        model, tokenizer = load_model(seed)
        train_features, _ = extract(seed,"train",train,model,tokenizer)
        val_features, val_logits = extract(seed,"validation",val,model,tokenizer)
        del model, tokenizer
        gc.collect(); torch.cuda.empty_cache()
        started = time.monotonic()
        with threadpool_limits(limits=4):
            detector = SharedMahalanobisDetector().fit(train_features,train.label_id.to_numpy())
            val_distance, nearest = detector.score_with_nearest(val_features)
        detector_path = folder / "distance_model.npz"
        detector.save(detector_path, overwrite=True)
        print(f"seed={seed} train-only covariance fitted; shrinkage={detector.shrinkage_:.8g}; seconds={time.monotonic()-started:.1f}",flush=True)
        scores = anomaly_scores(val_logits)
        scores["mahalanobis"] = val_distance
        decisions = {}
        for score in SCORES:
            curve = threshold_curve(scores[score],val.is_oos.to_numpy())
            curve.to_csv(folder / f"validation_{score}_threshold_curve.csv",index=False)
            for policy in POLICIES:
                decisions[f"{score}/{policy}"] = choose_threshold(curve,policy)
        write_json(folder / "frozen_thresholds.json", decisions)
        write_json(folder / "development_summary.json",dict(seed=seed,feature_dim=768,training_rows=15000,
                   class_counts=detector.class_counts_.tolist(),shrinkage=detector.shrinkage_,
                   fit_and_score_seconds=time.monotonic()-started,validation_drift=old_forward_drift(seed,"validation",val_logits),
                   frozen_at=now()))
        all_rows += evaluate_frame(seed,"validation",val,val_logits,val_distance,nearest,decisions)
        frozen_paths += [detector_path,folder / "frozen_thresholds.json",folder / "development_summary.json",
                         folder / "train_extraction.json",folder / "validation_extraction.json"]
        del train_features,val_features,val_logits,detector
        gc.collect()
    for relative, expected in source_hashes.items():
        assert digest(PROJECT/relative)==expected, f"Source changed: {relative}"
    for relative,expected in code_hashes.items():
        assert digest(ROOT/relative)==expected, f"Protocol/code changed during development: {relative}"
    pd.DataFrame(all_rows).to_csv(OUTPUT / "validation_metrics_by_seed.csv",index=False)
    write_json(OUTPUT / "DEVELOPMENT_FROZEN.json",dict(frozen_at=now(),source_hashes=source_hashes,code_hashes=code_hashes,
                development_hashes={p.relative_to(OUTPUT).as_posix():digest(p) for p in frozen_paths},
                status="All 18 validation thresholds and three train-only distance models frozen before test-stage inference"))
    print("DEVELOPMENT_FROZEN: all three seeds ready for test",flush=True)


def test():
    freeze = read_json(OUTPUT / "DEVELOPMENT_FROZEN.json")
    for relative, expected in freeze["code_hashes"].items():
        assert digest(ROOT/relative)==expected, f"Frozen code/protocol changed: {relative}"
    for relative, expected in freeze["development_hashes"].items():
        assert digest(OUTPUT/relative)==expected, f"Development artifact changed: {relative}"
    for relative, expected in freeze["source_hashes"].items():
        assert digest(PROJECT/relative)==expected, f"Checkpoint/input changed: {relative}"
    environment = prepare_environment()
    assert environment == read_json(OUTPUT / "environment.json"), "Environment changed between stages"
    if (OUTPUT / "TEST_COMPLETE.json").exists():
        raise FileExistsError("Test already evaluated; results are frozen")
    data = read_json(PROJECT / "data/raw/data_full.json")
    all_rows=[]
    for seed in SEEDS:
        folder = OUTPUT / f"seed{seed}"
        mapping = read_json(PROJECT / f"outputs/bert/seed{seed}/label_mapping.json")["label_to_id"]
        frame = metadata(data["test"]+data["oos_test"],["test"]*4500+["oos_test"]*1000,mapping)
        model,tokenizer=load_model(seed)
        features,logits=extract(seed,"test",frame,model,tokenizer)
        del model,tokenizer
        gc.collect();torch.cuda.empty_cache()
        detector=SharedMahalanobisDetector.load(folder/"distance_model.npz")
        with threadpool_limits(limits=4):
            distances,nearest=detector.score_with_nearest(features)
        decisions=read_json(folder/"frozen_thresholds.json")
        all_rows+=evaluate_frame(seed,"test",frame,logits,distances,nearest,decisions)
        write_json(folder/"test_drift.json",old_forward_drift(seed,"test",logits))
        print(f"seed={seed} test evaluated with frozen decisions",flush=True)
    validation=pd.read_csv(OUTPUT/"validation_metrics_by_seed.csv",float_precision="round_trip")
    pd.concat([validation,pd.DataFrame(all_rows)],ignore_index=True).to_csv(OUTPUT/"metrics_by_seed.csv",index=False)
    for relative,expected in freeze["source_hashes"].items():
        assert digest(PROJECT/relative)==expected
    write_json(OUTPUT/"TEST_COMPLETE.json",dict(completed_at=now(),development_freeze_sha256=digest(OUTPUT/"DEVELOPMENT_FROZEN.json"),
               metrics_sha256=digest(OUTPUT/"metrics_by_seed.csv"),source_inputs_unchanged=True))
    print("TEST_COMPLETE: no training or test-driven method changes",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=("development","test"),required=True)
    args=parser.parse_args()
    (development if args.stage=="development" else test)()
