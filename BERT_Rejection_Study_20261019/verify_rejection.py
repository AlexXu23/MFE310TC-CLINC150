"""Independently verify frozen rejection artifacts without model inference.

This module deliberately imports none of run_rejection_study, distance_model,
report_rejection, or evaluate_oos. It reads completed artifacts and writes only
outputs/verification.json. Missing test artifacts produce NOT_READY, not PASS.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent / "CLINC150_Team_20261019"
SEEDS = (42, 43, 44)
SCORES = ("msp", "entropy", "mahalanobis")
POLICIES = ("max_oos_f1", "is_acceptance_95")
SPLITS = ("validation", "test")
CURVE_FIELDS = ("threshold", "true_oos_rejected", "in_scope_rejected",
                "oos_precision", "oos_recall", "oos_f1", "in_scope_acceptance")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    return pd.read_csv(path, keep_default_na=False, float_precision="round_trip")


def close(actual, expected, context, *, atol=1e-10, rtol=1e-8):
    if expected is None or expected == "":
        require(actual is None or (np.isscalar(actual) and pd.isna(actual)),
                f"{context}: expected unavailable value, got {actual}")
        return
    require(np.isclose(float(actual), float(expected), atol=atol, rtol=rtol),
            f"{context}: {actual} != {expected}")


def booleans(values, context):
    normalized = values.astype(str).str.lower().str.strip()
    require(normalized.isin(("true", "false", "1", "0")).all(),
            f"{context}: invalid Boolean values")
    return normalized.isin(("true", "1")).to_numpy(dtype=bool)


def array_close(actual, expected, context, *, atol=1e-10, rtol=1e-8):
    require(np.shape(actual) == np.shape(expected), f"{context}: shape mismatch")
    require(np.allclose(actual, expected, atol=atol, rtol=rtol),
            f"{context}: maximum difference {np.max(np.abs(actual - expected))}")


def verify_hashes(output):
    freeze_path = output / "DEVELOPMENT_FROZEN.json"
    freeze = read_json(freeze_path)
    completion = read_json(output / "TEST_COMPLETE.json")
    require(completion["development_freeze_sha256"] == digest(freeze_path),
            "Development manifest hash differs from test completion record")
    require(completion["metrics_sha256"] == digest(output / "metrics_by_seed.csv"),
            "Completed metrics hash changed")
    require(completion["source_inputs_unchanged"] is True,
            "Test completion does not record unchanged source inputs")
    require(datetime.fromisoformat(completion["completed_at"]) >=
            datetime.fromisoformat(freeze["frozen_at"]), "Completion predates freeze")
    counts = {}
    for key, base in (("source_hashes", PROJECT), ("code_hashes", ROOT),
                      ("development_hashes", output)):
        require(bool(freeze[key]), f"Empty {key}")
        for relative, expected in freeze[key].items():
            target = (base / relative).resolve()
            require(target.is_relative_to(base.resolve()), f"Unexpected manifest path: {relative}")
            require(digest(target) == expected, f"Frozen hash changed: {key}:{relative}")
        counts[key] = len(freeze[key])
    require(set(freeze["code_hashes"]) ==
            {"run_rejection_study.py", "distance_model.py", "PROTOCOL.json"},
            "Unexpected frozen code/protocol inventory")
    for seed in SEEDS:
        for filename in ("distance_model.npz", "frozen_thresholds.json",
                         "train_extraction.json", "validation_extraction.json"):
            require(f"seed{seed}/{filename}" in freeze["development_hashes"],
                    f"Missing frozen dependency seed{seed}/{filename}")
    protocol = read_json(ROOT / "PROTOCOL.json")
    require(protocol["seeds"] == list(SEEDS) and protocol["training_rows"] == 15000,
            "Protocol seed or training-row mismatch")
    require(protocol["oos_train_used"] is False, "Protocol uses OOS training rows")
    require(protocol["primary"] == {"comparison": "mahalanobis minus msp",
            "threshold_policy": "max_oos_f1", "metric": "oos_f1"},
            "Primary comparison differs from fixed protocol")
    return freeze, counts


def verify_metadata(frame, expected_rows, expected_sources, mapping, context):
    n = len(expected_rows)
    require(len(frame) == n, f"{context}: row count")
    require(np.array_equal(frame.row_id.to_numpy(), np.arange(n)), f"{context}: row order")
    require(frame[["text", "true_label"]].values.tolist() == expected_rows,
            f"{context}: original JSON text/label alignment")
    require(frame.source_split.tolist() == expected_sources, f"{context}: source split")
    labels = np.array([mapping.get(row[1], -1) for row in expected_rows], dtype=np.int64)
    require(np.array_equal(frame.label_id.to_numpy(), labels), f"{context}: label IDs")
    oos = booleans(frame.is_oos, context)
    require(np.array_equal(oos, labels == -1), f"{context}: OOS flags")
    return labels, oos


def verify_extraction(folder, split, expected_rows, expected_sources, mapping, freeze):
    frame = read_csv(folder / f"{split}_metadata.csv")
    labels, oos = verify_metadata(frame, expected_rows, expected_sources, mapping,
                                  f"{folder.name}/{split} metadata")
    record = read_json(folder / f"{split}_extraction.json")
    feature_path, logit_path = folder / f"{split}_cls.npy", folder / f"{split}_logits_fp32.npy"
    require(record["feature_sha256"] == digest(feature_path), f"{split}: feature hash")
    require(record["logit_sha256"] == digest(logit_path), f"{split}: logit hash")
    require(record["protocol_sha256"] == digest(ROOT / "PROTOCOL.json"), f"{split}: protocol hash")
    text_hash = hashlib.sha256(json.dumps([r[0] for r in expected_rows],
                                       ensure_ascii=False).encode()).hexdigest()
    require(record["text_sha256"] == text_hash, f"{split}: extraction text hash")
    require(record["rows"] == len(expected_rows), f"{split}: extraction row count")
    if split == "test":
        require(datetime.fromisoformat(record["completed_at"]) >=
                datetime.fromisoformat(freeze["frozen_at"]), "Test extraction predates freeze")
    features = np.load(feature_path, allow_pickle=False, mmap_mode="r")
    logits = np.load(logit_path, allow_pickle=False, mmap_mode="r")
    require(features.shape == (len(frame), 768) and features.dtype == np.float32,
            f"{split}: expected FP32 CLS [N,768]")
    require(logits.shape == (len(frame), 150) and logits.dtype == np.float32,
            f"{split}: expected FP32 logits [N,150]")
    require(np.isfinite(features).all() and np.isfinite(logits).all(), f"{split}: nonfinite arrays")
    return features, logits, labels, oos


def verify_distance_statistics(folder, training_features, training_labels):
    with np.load(folder / "distance_model.npz", allow_pickle=False) as archive:
        model = {name: np.array(archive[name], copy=True) for name in archive.files}
    require(int(model["format_version"]) == 1, "Distance archive format")
    require(int(model["n_samples"]) == 15000 and int(model["n_features"]) == 768,
            "Distance archive dimensions")
    require(np.array_equal(model["classes"], np.arange(150)), "Distance class IDs")
    require(np.array_equal(model["class_counts"], np.full(150, 100)), "Distance class counts")
    # Use direct per-class means, not the detector's centered accumulation.
    train = np.asarray(training_features, dtype=np.float64)
    means = np.stack([train[training_labels == c].mean(axis=0) for c in range(150)])
    array_close(model["class_means"], means, "Train-only class means", atol=1e-12, rtol=1e-12)
    array_close(model["origin"], train.mean(axis=0), "Training origin", atol=1e-12, rtol=1e-12)
    array_close(model["class_offsets"] + model["origin"], means,
                "Class offsets", atol=1e-12, rtol=1e-12)
    residuals = train - means[training_labels]
    empirical = residuals.T @ residuals / len(train)
    shrinkage = float(model["shrinkage"])
    require(np.isfinite(shrinkage) and 0 <= shrinkage <= 1, "Invalid shrinkage")
    reconstructed = (1 - shrinkage) * empirical
    reconstructed.flat[::769] += shrinkage * np.trace(empirical) / 768
    array_close(model["covariance"], reconstructed, "Train-residual shrunk covariance",
                atol=1e-11, rtol=1e-9)
    np.linalg.cholesky(model["covariance"])
    array_close(model["covariance"] @ model["precision"], np.eye(768),
                "Covariance/precision inverse", atol=1e-5, rtol=1e-5)
    return model


def verify_sample_distances(features, frame, model, seed, split):
    labels = frame.label_id.to_numpy(dtype=np.int64)
    rng = np.random.default_rng(20261019 + seed + (1 if split == "test" else 0))
    indices = np.sort(np.r_[rng.choice(np.flatnonzero(labels >= 0), 3, replace=False),
                            rng.choice(np.flatnonzero(labels == -1), 3, replace=False)])
    differences = np.asarray(features[indices], dtype=np.float64)[:, None, :] - model["class_means"][None]
    flat = differences.reshape(-1, 768)
    # Direct solve is independent of the scorer's precision quadratic expansion.
    solved = np.linalg.solve(model["covariance"], flat.T).T
    distances = np.sum(flat * solved, axis=1).reshape(len(indices), 150)
    minimum = distances.min(axis=1)
    observed = frame.mahalanobis_score.to_numpy(dtype=np.float64)[indices]
    array_close(observed, minimum, f"seed{seed}/{split}: direct-solve distance",
                atol=1e-7, rtol=1e-8)
    nearest = model["classes"][distances.argmin(axis=1)]
    require(np.array_equal(nearest, frame.nearest_centroid_label_id_diagnostic_only.to_numpy()[indices]),
            f"seed{seed}/{split}: nearest centroid diagnostic")
    return {"seed": seed, "split": split, "row_ids": indices.tolist(),
            "max_abs_distance_error": float(np.max(np.abs(observed - minimum)))}


def manual_metrics(labels, prediction, oos, rejected, scores):
    tp = int(np.count_nonzero(oos & rejected))
    fp = int(np.count_nonzero(~oos & rejected))
    fn = int(np.count_nonzero(oos & ~rejected))
    n_is, n_oos = int((~oos).sum()), int(oos.sum())
    accepted = ~oos & ~rejected
    correctly_accepted = int(np.count_nonzero(accepted & (prediction == labels)))
    combined = np.where(rejected, -1, prediction)
    support = np.bincount(labels + 1, minlength=151)
    predicted = np.bincount(combined + 1, minlength=151)
    correct = np.bincount(labels[labels == combined] + 1, minlength=151)
    f1_by_class = np.divide(2 * correct, support + predicted,
                            out=np.zeros(151), where=support + predicted > 0)
    # Mann-Whitney AUC with half credit for score ties, using unique score bins.
    values, bins = np.unique(scores, return_inverse=True)
    positives = np.bincount(bins[oos], minlength=len(values))
    negatives = np.bincount(bins[~oos], minlength=len(values))
    below = np.cumsum(negatives) - negatives
    auc = np.sum(positives * (below + .5 * negatives)) / (n_is * n_oos)
    true_positive_desc = np.cumsum(positives[::-1])
    rejected_desc = np.cumsum((positives + negatives)[::-1])
    ap = np.sum((positives[::-1] / n_oos) * (true_positive_desc / rejected_desc))
    return {"oos_precision": tp / (tp + fp) if tp + fp else 0.,
            "oos_recall": tp / n_oos, "oos_f1": 2 * tp / (2 * tp + fp + fn),
            "oos_auroc": float(auc), "oos_aupr_average_precision": float(ap),
            "in_scope_acceptance": float(accepted.sum() / n_is),
            "in_scope_false_rejection_rate": fp / n_is,
            "in_scope_correct_and_accepted_rate": correctly_accepted / n_is,
            "accepted_in_scope_accuracy": correctly_accepted / int(accepted.sum()) if accepted.any() else None,
            "combined_151_accuracy": (correctly_accepted + tp) / len(labels),
            "combined_151_macro_f1": float(f1_by_class.mean()),
            "oos_true_positive": tp, "in_scope_false_rejections": fp, "oos_missed": fn}


def verify_validation_thresholds(scores, oos, decisions, stored_curve, context):
    unique = np.unique(scores)
    thresholds = np.r_[unique, np.nextafter(unique[-1], np.inf)]
    # Enumerate every decision directly, not the production suffix-count algorithm.
    decisions_matrix = scores[:, None] >= thresholds[None, :]
    tp = decisions_matrix[oos].sum(axis=0)
    fp = decisions_matrix[~oos].sum(axis=0)
    precision = np.divide(tp, tp + fp, out=np.zeros(len(tp)), where=tp + fp > 0)
    recall = tp / int(oos.sum())
    # Same published harmonic-mean arithmetic preserves exact floating-point ties.
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(len(tp)), where=precision + recall > 0)
    acceptance = 1 - fp / int((~oos).sum())
    columns = {"threshold": thresholds, "true_oos_rejected": tp,
               "in_scope_rejected": fp, "oos_precision": precision,
               "oos_recall": recall, "oos_f1": f1, "in_scope_acceptance": acceptance}
    require(len(stored_curve) == len(thresholds), f"{context}: candidate count")
    require(np.array_equal(stored_curve.threshold.to_numpy(), thresholds),
            f"{context}: exact candidate thresholds")
    for field, values in columns.items():
        array_close(stored_curve[field].to_numpy(), values, f"{context}: all candidates/{field}",
                    atol=1e-12, rtol=1e-12)
    for policy in POLICIES:
        selected = decisions[policy]
        if policy == "max_oos_f1":
            best = int(np.lexsort((thresholds, acceptance, f1))[-1])
        else:
            candidates = np.flatnonzero(acceptance >= .95 - 1e-12)
            best = int(candidates[np.lexsort((thresholds[candidates], acceptance[candidates],
                                               f1[candidates], recall[candidates]))[-1]])
        require(selected["threshold"] == float(thresholds[best]),
                f"{context}/{policy}: selected threshold is not optimal under frozen tie rules")
        for field in CURVE_FIELDS:
            close(selected[field], columns[field][best], f"{context}/{policy}/{field}",
                  atol=1e-12, rtol=1e-12)
    return len(thresholds)


def verify(output):
    freeze, hash_counts = verify_hashes(output)
    raw = read_json(PROJECT / "data/raw/data_full.json")
    metrics = read_csv(output / "metrics_by_seed.csv")
    keys = ("seed", "split", "score", "policy")
    expected_keys = {(seed, split, score, policy) for seed in SEEDS for split in SPLITS
                     for score in SCORES for policy in POLICIES}
    require(not metrics.duplicated(list(keys)).any(), "Duplicate metric settings")
    require(set(metrics[list(keys)].itertuples(index=False, name=None)) == expected_keys,
            "Expected exactly 36 validation/test metric settings")
    indexed = metrics.set_index(list(keys))
    settings_checked, thresholds_checked = 0, 0
    distance_checks, artifact_hashes = [], {}
    canonical_mapping = None
    for seed in SEEDS:
        folder = output / f"seed{seed}"
        mapping_document = read_json(PROJECT / f"outputs/bert/seed{seed}/label_mapping.json")
        mapping = mapping_document["label_to_id"]
        require(sorted(mapping.values()) == list(range(150)), "Invalid label mapping")
        if canonical_mapping is None:
            canonical_mapping = mapping
        require(mapping == canonical_mapping, "Label map differs between seeds")
        inverse = {value: key for key, value in mapping.items()}
        decisions = read_json(folder / "frozen_thresholds.json")
        require(set(decisions) == {f"{s}/{p}" for s in SCORES for p in POLICIES},
                f"seed{seed}: threshold settings")
        train_features, _, train_labels, train_oos = verify_extraction(
            folder, "train", raw["train"], ["train"] * 15000, mapping, freeze)
        require(not train_oos.any() and np.array_equal(np.bincount(train_labels), np.full(150, 100)),
                f"seed{seed}: training scope/counts")
        model = verify_distance_statistics(folder, train_features, train_labels)
        del train_features
        for split, source, n_is, n_oos in (("validation", "val", 3000, 100),
                                           ("test", "test", 4500, 1000)):
            expected_rows = raw[source] + raw[f"oos_{source}"]
            expected_sources = [source] * n_is + [f"oos_{source}"] * n_oos
            features, logits, labels, oos = verify_extraction(
                folder, split, expected_rows, expected_sources, mapping, freeze)
            frame_path = folder / f"{split}_predictions.csv"
            frame = read_csv(frame_path)
            verify_metadata(frame, expected_rows, expected_sources, mapping, f"seed{seed}/{split} predictions")
            require(int(oos.sum()) == n_oos and len(labels) == n_is + n_oos, "Split counts")
            require(np.array_equal(np.bincount(labels[~oos], minlength=150), np.full(150, n_is // 150)),
                    f"seed{seed}/{split}: class counts")
            prediction = np.asarray(logits).argmax(axis=1)
            require(np.array_equal(frame.predicted_label_id.to_numpy(), prediction),
                    f"seed{seed}/{split}: original classifier head changed")
            require(frame.predicted_label.tolist() == [inverse[int(k)] for k in prediction],
                    f"seed{seed}/{split}: decoded predictions")
            z = np.asarray(logits, dtype=np.float64)
            exponentials = np.exp(z - z.max(axis=1, keepdims=True))
            probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
            recomputed = {"msp": 1 - probabilities.max(axis=1),
                          "entropy": -np.sum(probabilities * np.log(probabilities), axis=1)}
            for score, values in recomputed.items():
                array_close(frame[f"{score}_score"].to_numpy(dtype=float), values,
                            f"seed{seed}/{split}: independent {score}", atol=1e-13, rtol=1e-12)
            distance_checks.append(verify_sample_distances(features, frame, model, seed, split))
            for score in SCORES:
                values = frame[f"{score}_score"].to_numpy(dtype=np.float64)
                require(np.isfinite(values).all(), f"seed{seed}/{split}/{score}: finite scores")
                if split == "validation":
                    curve = read_csv(folder / f"validation_{score}_threshold_curve.csv")
                    thresholds_checked += verify_validation_thresholds(
                        values, oos, {p: decisions[f"{score}/{p}"] for p in POLICIES},
                        curve, f"seed{seed}/{score}")
                for policy in POLICIES:
                    threshold = decisions[f"{score}/{policy}"]["threshold"]
                    rejected = values >= threshold
                    require(np.array_equal(booleans(frame[f"{score}_{policy}_reject"], "rejection"), rejected),
                            f"seed{seed}/{split}/{score}/{policy}: frozen decision mismatch")
                    recorded = indexed.loc[(seed, split, score, policy)]
                    require(float(recorded.threshold) == threshold, "Metrics threshold lost precision")
                    recalculated = manual_metrics(labels, prediction, oos, rejected, values)
                    for name, value in recalculated.items():
                        close(value, recorded[name], f"seed{seed}/{split}/{score}/{policy}/{name}")
                    settings_checked += 1
            artifact_hashes[frame_path.relative_to(output).as_posix()] = digest(frame_path)
        del model
    # Ensure the immutable boundary and source files remained unchanged while verifying.
    _, ending_counts = verify_hashes(output)
    require(ending_counts == hash_counts, "Manifest inventory changed during verification")
    return {"status": "PASS", "verified_settings": settings_checked,
            "validation_operating_points": 18, "validation_candidates_checked": thresholds_checked,
            "original_json_row_alignment": True, "shared_classifier_predictions_verified": True,
            "independent_manual_confusion_metrics": True, "independent_auc_average_precision": True,
            "all_validation_thresholds_and_tie_rules_verified": True,
            "train_only_means_and_recorded_shrinkage_covariance_verified": True,
            "distance_direct_solve_checks": distance_checks, "hash_inventory_counts": hash_counts,
            "frozen_manifest_and_source_hashes_verified": True,
            "development_manifest_sha256": digest(output / "DEVELOPMENT_FROZEN.json"),
            "test_completion_sha256": digest(output / "TEST_COMPLETE.json"),
            "metrics_sha256": digest(output / "metrics_by_seed.csv"),
            "verified_prediction_sha256": artifact_hashes,
            "verifier_sha256": digest(Path(__file__)),
            "limitations": ["Verifies saved features/logits and their hashes; does not rerun BERT inference.",
                            "Reconstructs covariance using the frozen shrinkage coefficient; does not independently re-estimate that coefficient.",
                            "Direct covariance solves sample six rows per seed/split, all 150 centers per sampled row."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    output = args.output.resolve()
    required = [output / name for name in ("DEVELOPMENT_FROZEN.json", "TEST_COMPLETE.json", "metrics_by_seed.csv")]
    required += [output / f"seed{seed}" / f"{split}_predictions.csv" for seed in SEEDS for split in SPLITS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print(json.dumps({"status": "NOT_READY", "missing": missing}, ensure_ascii=False, indent=2))
        return 2
    started = datetime.now(timezone.utc).isoformat()
    try:
        context = threadpool_limits(limits=4) if threadpool_limits else nullcontext()
        with context:
            result = verify(output)
    except Exception as error:
        result = {"status": "FAIL", "error_type": type(error).__name__, "error": str(error)}
    result.update(started_at_utc=started, completed_at_utc=datetime.now(timezone.utc).isoformat())
    destination = output / "verification.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
