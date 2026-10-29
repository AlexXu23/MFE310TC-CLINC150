"""Apply identical deterministic typo tests to fixed BERT seeds and TF-IDF.

Temperatures and all BERT thresholds are loaded from clean validation choices.
TF-IDF is reconstructed using its already selected C and verified against the
historical clean metrics. No selection uses clean or perturbed test metrics.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from common import load_dataset, validate_full_dataset
from evaluate_oos import (
    POLICIES, SCORES, anomaly_scores, calibration_metrics, choose_threshold,
    evaluate, file_hash, load_split, log_probabilities, threshold_curve, write_json,
)
from train_bert import collect_logits, encode_dataset, make_loader, set_seed
from typo_data import CONDITIONS, make_variants


def load_typos(path: Path, data: dict) -> dict[str, pd.DataFrame]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    original = data["test"] + data["oos_test"]
    frames = {}
    for condition in CONDITIONS:
        if file_hash(path / f"{condition}.csv") != summary["files_sha256"][f"{condition}.csv"]:
            raise AssertionError("Typo input hash differs from saved manifest")
        frame = pd.read_csv(path / f"{condition}.csv", keep_default_na=False)
        if frame[["original_text", "true_label"]].values.tolist() != [list(row) for row in original]:
            raise AssertionError("Perturbation rows differ from original test JSON")
        if not np.array_equal(frame["row_id"], np.arange(5500)):
            raise AssertionError("Typo row ids are not consecutive")
        for row in frame.to_dict("records"):
            expected = make_variants(row["original_text"], row["row_id"], summary["seed"])[condition]
            if row["text"] != expected["text"] or row["modified_tokens"] != expected["modified_tokens"]:
                raise AssertionError("Perturbation cannot be reproduced from its seed")
        frames[condition] = frame
    return frames


def fit_baseline(project: Path, data: dict, mapping: dict, output: Path) -> tuple[Pipeline, list[dict]]:
    old = json.loads((project / "outputs/baseline/metrics.json").read_text(encoding="utf-8"))
    c_value = float(old["selected_C"])
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ("classifier", LogisticRegression(C=c_value, max_iter=1000, solver="lbfgs", random_state=42)),
    ])
    pipeline.fit([text for text, _ in data["train"]], [label for _, label in data["train"]])
    if pipeline.classes_.tolist() != sorted(mapping, key=mapping.get):
        raise AssertionError("TF-IDF class columns do not match BERT labels")
    val_rows = data["val"] + data["oos_val"]
    val_logits = np.log(pipeline.predict_proba([text for text, _ in val_rows]))
    val_oos = np.array([label == "oos" for _, label in val_rows])
    choices = []
    for score, values in anomaly_scores(val_logits).items():
        for policy in POLICIES:
            choices.append({"calibration": "raw", "score": score, "policy": policy, "temperature": 1.0, **choose_threshold(threshold_curve(values, val_oos), policy)})
    write_json(output / "baseline_validation_decisions.json", {"selected_C": c_value, "decisions": choices})
    joblib.dump(pipeline, output / "baseline_model.joblib", compress=3)
    return pipeline, choices


def score_conditions(model_name: str, seed: int, logits_by_condition: dict, decisions: list[dict], frames: dict, model_output: Path, mapping: dict) -> tuple[list[dict], list[dict]]:
    classification_rows, rejection_rows = [], []
    model_output.mkdir(parents=True, exist_ok=True)
    labels = frames["clean"]["label_id"].to_numpy(dtype=int)
    is_oos = frames["clean"]["is_oos"].to_numpy(dtype=bool)
    in_scope = ~is_oos
    eligible = frames["typo_light"]["changed"].to_numpy(dtype=bool) & in_scope
    clean_correct = logits_by_condition["clean"].argmax(axis=1) == labels
    id_to_label = {value: key for key, value in mapping.items()}
    calibrations = {decision["calibration"]: decision["temperature"] for decision in decisions}
    for condition in CONDITIONS:
        logits = logits_by_condition[condition]
        np.save(model_output / f"{condition}_logits.npy", logits, allow_pickle=False)
        prediction = logits.argmax(axis=1)
        correct = prediction == labels
        frame = frames[condition].copy()
        frame["predicted_label_id"] = prediction
        frame["predicted_label"] = [id_to_label[index] for index in prediction]
        frame["intent_prediction_correct"] = correct & in_scope
        scores = {}
        for calibration, temperature in calibrations.items():
            scores[calibration] = anomaly_scores(logits, temperature)
            metrics, _ = calibration_metrics(logits[in_scope], labels[in_scope], temperature)
            classification_rows.append({
                "model": model_name, "seed": seed, "calibration": calibration, "condition": condition,
                "in_scope_n": int(in_scope.sum()), "changed_in_scope_n": int((frame["changed"].to_numpy() & in_scope).sum()),
                "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                "mean_confidence": metrics["mean_confidence"], "nll": metrics["nll"], "ece": metrics["ece_15_bins"],
                "eligible_only_n": int(eligible.sum()), "eligible_only_accuracy": float(correct[eligible].mean()),
                "clean_correct_to_wrong": int((in_scope & clean_correct & ~correct).sum()),
                "clean_wrong_to_correct": int((in_scope & ~clean_correct & correct).sum()),
            })
            frame[f"{calibration}_confidence"] = np.exp(log_probabilities(logits, temperature)).max(axis=1)
            for score_name in SCORES:
                frame[f"{calibration}_{score_name}_anomaly"] = scores[calibration][score_name]
        for decision in decisions:
            calibration, score_name, policy = decision["calibration"], decision["score"], decision["policy"]
            values = scores[calibration][score_name]
            frame[f"{calibration}_{score_name}_{policy}_reject"] = values >= decision["threshold"]
            rejection_rows.append({
                "model": model_name, "seed": seed, "condition": condition,
                "calibration": calibration, "score": score_name, "policy": policy,
                "threshold": decision["threshold"], "temperature": decision["temperature"],
                **evaluate(values, is_oos, labels, prediction, decision["threshold"]),
            })
        frame.to_csv(model_output / f"{condition}_predictions.csv", index=False)
    return classification_rows, rejection_rows


def add_deltas(frame: pd.DataFrame, keys: list[str], columns: list[str]) -> pd.DataFrame:
    clean = frame[frame["condition"] == "clean"].set_index(keys)
    for column in columns:
        frame[f"{column}_change_pp"] = [100 * (row[column] - clean.loc[tuple(row[key] for key in keys), column]) for row in frame.to_dict("records")]
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=".")
    parser.add_argument("--typo-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    project, output = Path(args.project).resolve(), Path(args.output)
    if (output / "summary.json").exists():
        raise FileExistsError("Robustness results already exist; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this configuration")
    started = time.time()
    data = load_dataset(project / "data/raw/data_full.json")
    validate_full_dataset(data)
    typo_path = Path(args.typo_data)
    frames = load_typos(typo_path, data)
    mapping = json.loads((project / "outputs/bert/seed42/label_mapping.json").read_text(encoding="utf-8"))["label_to_id"]
    protocol = {
        "seeds": args.seeds, "conditions": list(CONDITIONS), "same_perturbations_for_all_models": True,
        "thresholds": "Frozen clean validation decisions per model seed; no noisy validation or test tuning",
        "baseline": "Historical C=4.0, one deterministic fit, not replicated as independent seeds",
        "aggregate_sd": "Sample SD across training seeds, ddof=1; not confidence intervals",
        "scope": "Both in-scope and OOS test text perturbed; original scope labels inherited",
        "typo_manifest_sha256": file_hash(typo_path / "summary.json"),
    }
    write_json(output / "protocol.json", protocol)
    classification, rejection, consistency = [], [], []
    input_hashes = {}

    for seed in args.seeds:
        set_seed(seed)
        run = project / f"outputs/bert/seed{seed}"
        summary = json.loads((run / "run_summary.json").read_text(encoding="utf-8"))
        if summary["config"]["seed"] != seed:
            raise AssertionError("Seed and run directory differ")
        if summary["data_sha256"] != file_hash(project / "data/raw/data_full.json"):
            raise AssertionError("Training dataset hashes differ across runs")
        reference = json.loads((project / "outputs/bert/seed42/run_summary.json").read_text(encoding="utf-8"))["config"]
        if {k: v for k, v in summary["config"].items() if k != "seed"} != {k: v for k, v in reference.items() if k != "seed"}:
            raise AssertionError("Across-seed training hyperparameters differ")
        current_mapping = json.loads((run / "label_mapping.json").read_text(encoding="utf-8"))["label_to_id"]
        if current_mapping != mapping:
            raise AssertionError("Label mapping differs across seeds")
        choices_file = project / f"outputs/oos/seed{seed}/frozen_validation_decisions.json"
        decisions = json.loads(choices_file.read_text(encoding="utf-8"))["decisions"]
        input_hashes[str(choices_file.relative_to(project))] = file_hash(choices_file)
        model = AutoModelForSequenceClassification.from_pretrained(run / "best_model").to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(run / "best_model", use_fast=True)
        collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
        logits_by_condition = {}
        for condition in CONDITIONS:
            dataset = encode_dataset(tokenizer, frames[condition]["text"].tolist(), frames[condition]["label_id"].tolist(), summary["config"]["max_length"])
            loader = make_loader(dataset, collator, summary["config"]["eval_batch_size"], False)
            logits_by_condition[condition] = collect_logits(model, loader, torch.device("cuda"), summary["config"]["fp16"])
            print(f"BERT seed={seed}: inferred {condition}, rows={len(dataset)}", flush=True)
        historical = np.load(run / "test_logits.npy", allow_pickle=False)
        current = logits_by_condition["clean"]
        agreement = float((historical.argmax(axis=1) == current.argmax(axis=1)).mean())
        consistency.append({"seed": seed, "clean_max_abs_logit_delta": float(np.abs(current - historical).max()), "clean_prediction_agreement": agreement})
        if agreement != 1.0:
            raise AssertionError("Clean predictions differ from historical export")
        cls_rows, reject_rows = score_conditions("bert", seed, logits_by_condition, decisions, frames, output / f"seed{seed}", mapping)
        classification.extend(cls_rows)
        rejection.extend(reject_rows)
        del model, logits_by_condition
        torch.cuda.empty_cache()
        input_hashes[f"outputs/bert/seed{seed}/test_logits.npy"] = file_hash(run / "test_logits.npy")

    baseline_output = output / "tfidf_lr"
    baseline_output.mkdir(exist_ok=True)
    baseline, baseline_decisions = fit_baseline(project, data, mapping, baseline_output)
    baseline_logits = {condition: np.log(baseline.predict_proba(frames[condition]["text"].tolist())) for condition in CONDITIONS}
    cls_rows, reject_rows = score_conditions("tfidf_lr", 42, baseline_logits, baseline_decisions, frames, baseline_output, mapping)
    historical_baseline = json.loads((project / "outputs/baseline/metrics.json").read_text(encoding="utf-8"))["in_scope_test"]
    clean_baseline = next(row for row in cls_rows if row["condition"] == "clean")
    for key in ("accuracy", "macro_f1"):
        if not np.isclose(clean_baseline[key], historical_baseline[f"{key}_before_rejection"], atol=1e-12):
            raise AssertionError("Reconstructed TF-IDF clean metric differs from historical baseline")
    classification.extend(cls_rows)
    rejection.extend(reject_rows)
    classification_frame = add_deltas(pd.DataFrame(classification), ["model", "seed", "calibration"], ["accuracy", "macro_f1", "eligible_only_accuracy", "mean_confidence"])
    rejection_frame = add_deltas(pd.DataFrame(rejection), ["model", "seed", "calibration", "score", "policy"], ["oos_f1", "oos_recall", "in_scope_false_rejection_rate", "in_scope_correct_and_accepted_rate"])
    classification_frame.to_csv(output / "classification_by_seed.csv", index=False)
    rejection_frame.to_csv(output / "rejection_by_seed.csv", index=False)
    write_json(output / "clean_inference_consistency.json", {"checks": consistency, "baseline_clean_reproduced": True})
    write_json(output / "summary.json", {
        "seeds": args.seeds, "classification_rows": len(classification_frame), "rejection_rows": len(rejection_frame),
        "runtime_seconds": time.time() - started, "input_sha256": input_hashes,
        "typo_data_sha256": {condition: file_hash(typo_path / f"{condition}.csv") for condition in CONDITIONS},
        "clean_consistency": consistency,
        "classification": classification_frame.to_dict("records"),
    })
    print(classification_frame[["model", "seed", "calibration", "condition", "accuracy", "macro_f1", "macro_f1_change_pp"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
