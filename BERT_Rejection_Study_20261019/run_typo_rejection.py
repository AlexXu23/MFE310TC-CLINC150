"""Stress-test frozen rejection rules on existing, paired spelling perturbations.

No fitting, threshold search, model selection, or change to the clean experiment.
All original checkpoints, train statistics, thresholds and clean predictions are
hash-bound before the two perturbed conditions are inferred.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import run_rejection_study as base
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from distance_model import SharedMahalanobisDetector

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / "typo_stress_test"
OUTPUT = STUDY / "outputs"
SOURCE = base.PROJECT / "outputs/robustness/typo_data"
CONDITIONS = ("clean", "typo_light", "typo_medium")
SEEDS, SCORES, POLICIES = base.SEEDS, base.SCORES, base.POLICIES


def read_json(path):
    return base.read_json(path)


def read_frame(path):
    frame = pd.read_csv(path, float_precision="round_trip", keep_default_na=False)
    for col in ("is_oos", "changed"):
        if col in frame:
            text = frame[col].astype(str).str.lower()
            assert text.isin(["true", "false"]).all(), (path, col)
            frame[col] = text.eq("true")
    return frame


def protocol():
    return {
        "date": "2026-10-19", "kind": "frozen_system_synthetic_typo_stress_test",
        "seeds": list(SEEDS), "conditions": list(CONDITIONS),
        "scores": list(SCORES), "policies": list(POLICIES),
        "question": "Do normal requests become falsely rejected under existing light/medium spelling noise?",
        "frozen": "All BERT weights, class means, shared covariance, inference settings and all 18 clean-validation thresholds remain unchanged.",
        "no_fit_or_selection": True,
        "inference": base.protocol()["inference"],
        "clean_reference": "Reuse the already frozen FP32 clean features/logits. Verify reconstructed decisions against original clean predictions.",
        "perturbations": "Reuse seed2026 CSVs exactly. ASCII alphabetic tokens length>=5; internal deletion or adjacent internal swap; first/last characters preserved; light<=1 token; medium<=2 and contains light mutation.",
        "labels": "Inherited original labels; this is synthetic character stress, not human-verified semantic equivalence.",
        "focus": {"policy": "is_acceptance_95", "outcomes": ["IS false rejection rate", "new rejection among clean-accepted IS", "loss of clean correct-and-accepted requests"]},
        "comparisons": "Report both noise levels and all methods/policies; preserve original clean experiment primary endpoint. No winner or new operating point selected from noisy test.",
        "normal_denominator": "4500 IS requests primary; also report same fixed changed subset and conditional denominators explicitly.",
        "classification": "Each condition uses original classifier-head argmax, shared by all three rejection methods. Decompose accepted_correct, accepted_wrong, rejected_correct, rejected_wrong.",
        "transition": "Pair by original row_id. New rejection and recovered acceptance are distinct; net false-rejection change=new rejection minus recovered acceptance.",
        "bootstrap": "2000 paired draws of 30 rows per IS intent, same weights across conditions/seeds/methods. Fixed-model/threshold uncertainty only, not new spelling variants, training or validation uncertainty.",
        "limits": "95% acceptance was constrained only on clean validation. Original clean and typo classification results were inspected earlier; exploratory robustness evaluation, not fresh confirmatory data.",
    }


def check_hashes(hashes):
    for path, expected in hashes.items():
        assert base.digest(path) == expected, f"Frozen input changed: {path}"


def freeze_inputs(environment):
    original = read_json(ROOT / "outputs/DEVELOPMENT_FROZEN.json")
    complete = read_json(ROOT / "outputs/TEST_COMPLETE.json")
    assert base.digest(ROOT / "outputs/DEVELOPMENT_FROZEN.json") == complete["development_freeze_sha256"]
    assert base.digest(ROOT / "outputs/metrics_by_seed.csv") == complete["metrics_sha256"]
    paths = [ROOT / "outputs/DEVELOPMENT_FROZEN.json", ROOT / "outputs/TEST_COMPLETE.json",
             ROOT / "outputs/environment.json", ROOT / "outputs/metrics_by_seed.csv"]
    for key, parent in (("source_hashes", base.PROJECT), ("code_hashes", ROOT), ("development_hashes", ROOT / "outputs")):
        for relative, expected in original[key].items():
            path = parent / relative
            assert base.digest(path) == expected, f"Original frozen input changed: {path}"
            paths.append(path)
    for seed in SEEDS:
        folder = ROOT / f"outputs/seed{seed}"
        for name in ("test_cls.npy", "test_logits_fp32.npy", "test_metadata.csv", "test_predictions.csv", "test_extraction.json"):
            paths.append(folder / name)
        rec = read_json(folder / "test_extraction.json")
        assert base.digest(folder / "test_cls.npy") == rec["feature_sha256"]
        assert base.digest(folder / "test_logits_fp32.npy") == rec["logit_sha256"]
    paths += [SOURCE / f"{condition}.csv" for condition in CONDITIONS]
    paths += [SOURCE / "summary.json", SOURCE / "coverage.csv", base.PROJECT / "src/typo_data.py"]
    paths += [Path(__file__), STUDY / "PROTOCOL.json"]
    hashes = {p.resolve().as_posix(): base.digest(p) for p in sorted(set(paths))}
    record = {"input_hashes": hashes, "runtime": environment,
              "protocol_sha256": base.digest(STUDY / "PROTOCOL.json"),
              "runner_sha256": base.digest(__file__)}
    path = STUDY / "INPUTS_FROZEN.json"
    if path.exists():
        previous = read_json(path)
        assert all(previous[k] == v for k, v in record.items()), "Resume refused: input, code or runtime changed"
        return previous
    record["frozen_at"] = base.now()
    base.write_json(path, record, immutable=True)
    return record


def input_frames():
    summary = read_json(SOURCE / "summary.json")
    frames = {c: read_frame(SOURCE / f"{c}.csv") for c in CONDITIONS}
    for condition, frame in frames.items():
        assert base.digest(SOURCE / f"{condition}.csv") == summary["files_sha256"][f"{condition}.csv"]
        assert len(frame) == 5500 and frame.is_oos.sum() == 1000
        assert np.array_equal(frame.row_id, np.arange(5500))
        assert frame.condition.eq(condition).all()
        assert frame.changed.eq(frame.text.ne(frame.original_text)).all()
        for col in ("row_id", "original_text", "true_label", "label_id", "is_oos", "source_split"):
            assert frame[col].equals(frames["clean"][col]), (condition, col)
        assert frame.loc[~frame.is_oos, "label_id"].value_counts().eq(30).all()
    assert frames["clean"].text.equals(frames["clean"].original_text)
    assert not frames["clean"].changed.any()
    assert frames["typo_light"].changed.equals(frames["typo_medium"].changed)
    return frames


def extract(seed, condition, frame, model, tokenizer, frozen):
    folder = OUTPUT / f"seed{seed}"
    features_path = folder / f"{condition}_cls.npy"
    logits_path = folder / f"{condition}_logits_fp32.npy"
    record_path = folder / f"{condition}_extraction.json"
    binding = {"inputs_frozen_sha256": base.digest(STUDY / "INPUTS_FROZEN.json"),
               "input_csv_sha256": base.digest(SOURCE / f"{condition}.csv"),
               "runtime": frozen["runtime"], "seed": seed, "condition": condition}
    if record_path.exists():
        record = read_json(record_path)
        assert all(record[k] == value for k, value in binding.items())
        assert base.digest(features_path) == record["feature_sha256"]
        assert base.digest(logits_path) == record["logit_sha256"]
        return np.load(features_path, allow_pickle=False), np.load(logits_path, allow_pickle=False)
    features = np.empty((len(frame), 768), dtype=np.float32)
    logits = np.empty((len(frame), 150), dtype=np.float32)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for start in range(0, len(frame), base.BATCH_SIZE):
            end = min(start + base.BATCH_SIZE, len(frame))
            batch = tokenizer(frame.text.iloc[start:end].tolist(), truncation=True, max_length=base.MAX_LENGTH,
                              padding=True, pad_to_multiple_of=8, return_tensors="pt")
            result = model(**{k: v.to("cuda") for k, v in batch.items()}, output_hidden_states=True, return_dict=True)
            features[start:end] = result.hidden_states[-1][:, 0, :].cpu().numpy()
            logits[start:end] = result.logits.cpu().numpy()
            if start % (50 * base.BATCH_SIZE) == 0 or end == len(frame):
                print(f"seed={seed} {condition} {end}/{len(frame)}; elapsed={time.monotonic()-started:.1f}s", flush=True)
    assert np.isfinite(features).all() and np.isfinite(logits).all()
    np.save(features_path, features, allow_pickle=False)
    np.save(logits_path, logits, allow_pickle=False)
    base.write_json(record_path, dict(**binding, feature_sha256=base.digest(features_path),
                    logit_sha256=base.digest(logits_path), elapsed_seconds=time.monotonic()-started,
                    peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(), completed_at=base.now()))
    return features, logits


def evaluate_condition(seed, condition, frame, features, logits, detector, thresholds):
    frame = frame.copy()
    prediction = logits.argmax(axis=1)
    frame["predicted_label_id"] = prediction
    mapping = read_json(base.PROJECT / f"outputs/bert/seed{seed}/label_mapping.json")["id_to_label"]
    frame["predicted_label"] = [mapping[str(k)] for k in prediction]
    with threadpool_limits(limits=4):
        distances, nearest = detector.score_with_nearest(features)
    frame["nearest_centroid_label_id_diagnostic_only"] = nearest
    scores = base.anomaly_scores(logits)
    scores["mahalanobis"] = distances
    rows = []
    normal = ~frame.is_oos.to_numpy()
    for score in SCORES:
        frame[f"{score}_score"] = scores[score]
        for policy in POLICIES:
            threshold = thresholds[f"{score}/{policy}"]["threshold"]
            frame[f"{score}_{policy}_reject"] = scores[score] >= threshold
            rows.append(dict(seed=seed, condition=condition, score=score, policy=policy, threshold=threshold,
                        **base.evaluate(scores[score], frame.is_oos.to_numpy(), frame.label_id.to_numpy(), prediction, threshold),
                        closed_set_accuracy=float(np.mean(prediction[normal] == frame.label_id.to_numpy()[normal])),
                        closed_set_macro_f1=float(f1_score(frame.label_id.to_numpy()[normal], prediction[normal], average="macro"))))
    frame.to_csv(OUTPUT / f"seed{seed}/{condition}_predictions.csv", index=False)
    return frame, rows


def main():
    if (OUTPUT / "TEST_COMPLETE.json").exists():
        raise FileExistsError("Completed stress test is frozen; use the existing reports/verification")
    STUDY.mkdir(exist_ok=True)
    OUTPUT.mkdir(exist_ok=True)
    base.write_json(STUDY / "PROTOCOL.json", protocol(), immutable=True)
    environment = base.prepare_environment()
    assert environment == read_json(ROOT / "outputs/environment.json"), "Runtime differs from clean study"
    frozen = freeze_inputs(environment)
    frames = input_frames()
    print("Frozen input hashes verified; no fitting or threshold selection", flush=True)
    rows, unchanged_checks = [], []
    for seed in SEEDS:
        folder = OUTPUT / f"seed{seed}"
        folder.mkdir(exist_ok=True)
        original_folder = ROOT / f"outputs/seed{seed}"
        detector = SharedMahalanobisDetector.load(original_folder / "distance_model.npz")
        thresholds = read_json(original_folder / "frozen_thresholds.json")
        clean_features = np.load(original_folder / "test_cls.npy", allow_pickle=False)
        clean_logits = np.load(original_folder / "test_logits_fp32.npy", allow_pickle=False)
        original_clean = read_frame(original_folder / "test_predictions.csv")
        for col in ("row_id", "text", "true_label", "label_id", "is_oos"):
            assert frames["clean"][col].equals(original_clean[col]), (seed, col)
        clean, metric_rows = evaluate_condition(seed, "clean", frames["clean"], clean_features, clean_logits, detector, thresholds)
        for score in SCORES:
            np.testing.assert_allclose(clean[f"{score}_score"], original_clean[f"{score}_score"], rtol=1e-12, atol=1e-9)
            for policy in POLICIES:
                col = f"{score}_{policy}_reject"
                assert clean[col].equals(original_clean[col]), (seed, col)
        rows += metric_rows
        model, tokenizer = base.load_model(seed)
        for condition in CONDITIONS[1:]:
            features, logits = extract(seed, condition, frames[condition], model, tokenizer, frozen)
            pred, metric_rows = evaluate_condition(seed, condition, frames[condition], features, logits, detector, thresholds)
            rows += metric_rows
            unchanged = ~frames[condition].changed.to_numpy()
            unchanged_checks.append(dict(seed=seed, condition=condition, unchanged_rows=int(unchanged.sum()),
                max_abs_logit_delta=float(np.max(np.abs(logits[unchanged]-clean_logits[unchanged]))),
                changed_argmax=int(np.sum(logits[unchanged].argmax(1) != clean_logits[unchanged].argmax(1))),
                rejection_decision_changes={f"{s}/{p}":int((pred.loc[unchanged,f"{s}_{p}_reject"] != clean.loc[unchanged,f"{s}_{p}_reject"]).sum()) for s in SCORES for p in POLICIES}))
        del model, tokenizer, detector, clean_features, clean_logits
        gc.collect()
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(OUTPUT / "metrics_by_seed.csv", index=False)
    base.write_json(OUTPUT / "unchanged_text_checks.json", unchanged_checks)
    check_hashes(frozen["input_hashes"])
    generated = {p.relative_to(OUTPUT).as_posix():base.digest(p) for p in sorted(OUTPUT.rglob("*")) if p.is_file()}
    base.write_json(OUTPUT / "TEST_COMPLETE.json", dict(completed_at=base.now(),
                    inputs_frozen_sha256=base.digest(STUDY / "INPUTS_FROZEN.json"),
                    output_hashes=generated, frozen_inputs_unchanged=True, evaluated_settings=len(rows)))
    print("TEST_COMPLETE: 54 settings; original model/statistics/thresholds/clean outputs unchanged", flush=True)


if __name__ == "__main__":
    main()
