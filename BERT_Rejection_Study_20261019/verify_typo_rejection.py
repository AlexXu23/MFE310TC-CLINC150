"""Independently verify the frozen-threshold typo rejection follow-up.

No experiment, report, distance-model, or previous verifier module is imported.
The only write is typo_stress_test/outputs/verification.json, after artifacts
are complete. There is no training, inference, threshold selection, or refit.
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
STUDY = ROOT / "typo_stress_test"
SEEDS = (42, 43, 44)
CONDITIONS = ("clean", "typo_light", "typo_medium")
SCORES = ("msp", "entropy", "mahalanobis")
POLICIES = ("max_oos_f1", "is_acceptance_95")
IDENTITY = ("row_id", "original_text", "text", "true_label", "label_id")


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


def booleans(values, context):
    normalized = values.astype(str).str.strip().str.lower()
    require(normalized.isin(("true", "false", "1", "0")).all(), f"{context}: invalid booleans")
    return normalized.isin(("true", "1")).to_numpy(dtype=bool)


def close(actual, expected, context, *, atol=1e-10, rtol=1e-8):
    if expected is None or expected == "":
        require(actual is None or pd.isna(actual), f"{context}: expected unavailable value")
    else:
        require(np.isclose(float(actual), float(expected), atol=atol, rtol=rtol),
                f"{context}: {actual} != {expected}")


def array_close(actual, expected, context, *, atol=1e-10, rtol=1e-8):
    require(np.shape(actual) == np.shape(expected), f"{context}: shape mismatch")
    require(np.allclose(actual, expected, atol=atol, rtol=rtol),
            f"{context}: maximum error {np.max(np.abs(actual - expected))}")


def verify_hash_map(mapping, base, context):
    require(isinstance(mapping, dict) and bool(mapping), f"{context}: empty hash inventory")
    for relative, expected in mapping.items():
        target = (base / relative).resolve()
        require(target.is_relative_to(base.resolve()), f"{context}: path outside declared root")
        require(target.is_file(), f"{context}: missing {relative}")
        require(digest(target) == expected, f"{context}: changed {relative}")
    return len(mapping)


def verify_original_freeze():
    output = ROOT / "outputs"
    frozen = read_json(output / "DEVELOPMENT_FROZEN.json")
    complete = read_json(output / "TEST_COMPLETE.json")
    require(complete["development_freeze_sha256"] == digest(output / "DEVELOPMENT_FROZEN.json"),
            "Original development freeze changed")
    require(complete["metrics_sha256"] == digest(output / "metrics_by_seed.csv"),
            "Original clean metrics changed")
    counts = {}
    for key, base in (("source_hashes", PROJECT), ("code_hashes", ROOT),
                      ("development_hashes", output)):
        counts[key] = verify_hash_map(frozen[key], base, f"original/{key}")
    return counts


def verify_followup_freeze(study, output):
    """Validate the follow-up manifest; its exact schema is pinned below."""
    frozen = read_json(study / "INPUTS_FROZEN.json")
    complete = read_json(output / "TEST_COMPLETE.json")
    require(set(frozen) == {"input_hashes", "runtime", "protocol_sha256", "runner_sha256", "frozen_at"},
            "Unexpected INPUTS_FROZEN manifest fields")
    require(complete["inputs_frozen_sha256"] == digest(study / "INPUTS_FROZEN.json"),
            "Follow-up frozen-input manifest changed")
    require(complete["frozen_inputs_unchanged"] is True and complete["evaluated_settings"] == 54,
            "Follow-up completion status/settings")
    require(datetime.fromisoformat(complete["completed_at"]) >= datetime.fromisoformat(frozen["frozen_at"]),
            "Follow-up completion predates its input freeze")
    require(frozen["protocol_sha256"] == digest(study / "PROTOCOL.json"), "Follow-up protocol changed")
    require(frozen["runner_sha256"] == digest(ROOT / "run_typo_rejection.py"), "Follow-up runner changed")
    require(frozen["runtime"] == read_json(ROOT / "outputs/environment.json"), "Follow-up inference runtime changed")
    inventory = frozen["input_hashes"]
    require(bool(inventory), "Empty frozen input inventory")
    for absolute, expected in inventory.items():
        target = Path(absolute)
        require(target.is_absolute() and target.is_file(), f"Invalid absolute frozen input: {absolute}")
        require(target.resolve().is_relative_to(ROOT.resolve()) or target.resolve().is_relative_to(PROJECT.resolve()),
                f"Frozen input outside study/source roots: {absolute}")
        require(digest(target) == expected, f"Frozen input changed: {absolute}")
    required_inputs = [ROOT / "outputs/DEVELOPMENT_FROZEN.json", ROOT / "outputs/TEST_COMPLETE.json",
                       ROOT / "outputs/environment.json", PROJECT / "data/raw/data_full.json",
                       ROOT / "run_typo_rejection.py", study / "PROTOCOL.json"]
    required_inputs += [PROJECT / f"outputs/robustness/typo_data/{c}.csv" for c in CONDITIONS]
    for seed in SEEDS:
        required_inputs += [ROOT / f"outputs/seed{seed}/{name}" for name in
                            ("frozen_thresholds.json", "distance_model.npz", "test_cls.npy",
                             "test_logits_fp32.npy", "test_predictions.csv", "test_extraction.json")]
    for path in required_inputs:
        require(path.resolve().as_posix() in inventory, f"Required frozen input absent: {path}")
    output_count = verify_hash_map(complete["output_hashes"], output, "follow-up completed outputs")
    required_outputs = ["metrics_by_seed.csv", "unchanged_text_checks.json"]
    for seed in SEEDS:
        required_outputs += [f"seed{seed}/{c}_predictions.csv" for c in CONDITIONS]
        required_outputs += [f"seed{seed}/{c}_{suffix}" for c in CONDITIONS[1:]
                             for suffix in ("cls.npy", "logits_fp32.npy", "extraction.json")]
    require(set(required_outputs).issubset(complete["output_hashes"]), "Incomplete output hash inventory")
    protocol = read_json(study / "PROTOCOL.json")
    require(protocol["seeds"] == list(SEEDS) and protocol["conditions"] == list(CONDITIONS), "Protocol experiment grid")
    require(protocol["scores"] == list(SCORES) and protocol["policies"] == list(POLICIES), "Protocol score/policy grid")
    require(protocol["no_fit_or_selection"] is True, "Protocol permits refitting")
    require(protocol["inference"] == read_json(ROOT / "PROTOCOL.json")["inference"], "Protocol inference differs from clean study")
    for seed in SEEDS:
        for condition in CONDITIONS[1:]:
            folder = output / f"seed{seed}"
            extraction = read_json(folder / f"{condition}_extraction.json")
            require(extraction["inputs_frozen_sha256"] == complete["inputs_frozen_sha256"], "Extraction input-freeze binding")
            require(extraction["input_csv_sha256"] == digest(PROJECT / f"outputs/robustness/typo_data/{condition}.csv"),
                    "Extraction perturbation binding")
            require(extraction["seed"] == seed and extraction["condition"] == condition, "Extraction seed/condition")
            require(extraction["runtime"] == frozen["runtime"], "Extraction runtime")
            require(extraction["feature_sha256"] == digest(folder / f"{condition}_cls.npy"), "Extraction feature hash")
            require(extraction["logit_sha256"] == digest(folder / f"{condition}_logits_fp32.npy"), "Extraction logit hash")
            require(datetime.fromisoformat(extraction["completed_at"]) >= datetime.fromisoformat(frozen["frozen_at"]),
                    "Perturbed extraction completion predates freeze")
    return {"frozen_input_files": len(inventory), "completed_output_files": output_count}


def manual_metrics(labels, prediction, oos, rejected, scores):
    n_is, n_oos = int((~oos).sum()), int(oos.sum())
    tp = int(np.count_nonzero(oos & rejected))
    fp = int(np.count_nonzero(~oos & rejected))
    fn = n_oos - tp
    accepted = ~oos & ~rejected
    correct_accepted = int(np.count_nonzero(accepted & (prediction == labels)))
    combined = np.where(rejected, -1, prediction)
    support = np.bincount(labels + 1, minlength=151)
    predicted = np.bincount(combined + 1, minlength=151)
    true_positive = np.bincount(labels[labels == combined] + 1, minlength=151)
    f1 = np.divide(2 * true_positive, support + predicted,
                   out=np.zeros(151), where=support + predicted > 0)
    unique, indices = np.unique(scores, return_inverse=True)
    positives = np.bincount(indices[oos], minlength=len(unique))
    negatives = np.bincount(indices[~oos], minlength=len(unique))
    below = np.cumsum(negatives) - negatives
    auc = float(np.sum(positives * (below + .5 * negatives)) / (n_is * n_oos))
    cumulative_tp = np.cumsum(positives[::-1])
    cumulative_rejected = np.cumsum((positives + negatives)[::-1])
    ap = float(np.sum((positives[::-1] / n_oos) * cumulative_tp / cumulative_rejected))
    return {"oos_precision": tp / (tp + fp) if tp + fp else 0.,
            "oos_recall": tp / n_oos, "oos_f1": 2 * tp / (2 * tp + fp + fn),
            "oos_auroc": auc, "oos_aupr_average_precision": ap,
            "in_scope_acceptance": float(accepted.sum() / n_is),
            "in_scope_false_rejection_rate": fp / n_is,
            "in_scope_correct_and_accepted_rate": correct_accepted / n_is,
            "accepted_in_scope_accuracy": correct_accepted / int(accepted.sum()) if accepted.any() else None,
            "combined_151_accuracy": (correct_accepted + tp) / len(labels),
            "combined_151_macro_f1": float(f1.mean()),
            "oos_true_positive": tp, "in_scope_false_rejections": fp, "oos_missed": fn}


def verify_input_frame(frame, source, original_rows, mapping, condition, context):
    require(len(frame) == 5500, f"{context}: expected 5500 rows")
    require(np.array_equal(frame.row_id.to_numpy(), np.arange(5500)), f"{context}: row IDs")
    require(frame.original_text.tolist() == [row[0] for row in original_rows], f"{context}: original text")
    require(frame.true_label.tolist() == [row[1] for row in original_rows], f"{context}: inherited gold labels")
    labels = np.array([mapping.get(row[1], -1) for row in original_rows], dtype=np.int64)
    require(np.array_equal(frame.label_id.to_numpy(), labels), f"{context}: label IDs")
    oos = booleans(frame.is_oos, context)
    require(np.array_equal(oos, labels == -1) and int(oos.sum()) == 1000, f"{context}: OOS flags")
    require(np.array_equal(np.bincount(labels[~oos], minlength=150), np.full(150, 30)),
            f"{context}: 150 balanced IS classes")
    for column in IDENTITY:
        require(frame[column].tolist() == source[column].tolist(), f"{context}: frozen typo input/{column}")
    changed = booleans(frame.changed, context)
    require(np.array_equal(changed, booleans(source.changed, context)), f"{context}: changed metadata")
    require(np.array_equal(changed, frame.text.to_numpy() != frame.original_text.to_numpy()),
            f"{context}: changed flag does not describe the text")
    require(np.array_equal(frame.modified_tokens.to_numpy(), source.modified_tokens.to_numpy()),
            f"{context}: modified-token metadata")
    require(np.all(frame.modified_tokens.to_numpy() >= 0), f"{context}: negative modifications")
    if "condition" in frame:
        require(frame.condition.eq(condition).all(), f"{context}: condition column")
    if condition == "clean":
        require(not changed.any() and frame.modified_tokens.eq(0).all(), f"{context}: clean text modified")
    return labels, oos


def direct_distance_sample(features, frame, model, seed, condition):
    rng = np.random.default_rng(20261019 + seed + CONDITIONS.index(condition))
    labels = frame.label_id.to_numpy(dtype=np.int64)
    sampled = np.sort(np.r_[rng.choice(np.flatnonzero(labels >= 0), 3, replace=False),
                            rng.choice(np.flatnonzero(labels == -1), 3, replace=False)])
    difference = np.asarray(features[sampled], dtype=np.float64)[:, None, :] - model["class_means"][None]
    flat = difference.reshape(-1, 768)
    solved = np.linalg.solve(model["covariance"], flat.T).T
    distances = np.sum(flat * solved, axis=1).reshape(len(sampled), 150)
    minimum = distances.min(axis=1)
    observed = frame.mahalanobis_score.to_numpy(dtype=np.float64)[sampled]
    array_close(observed, minimum, f"seed{seed}/{condition}: direct covariance solve",
                atol=1e-7, rtol=1e-8)
    if "nearest_centroid_label_id_diagnostic_only" in frame:
        require(np.array_equal(model["classes"][distances.argmin(axis=1)],
                               frame.nearest_centroid_label_id_diagnostic_only.to_numpy()[sampled]),
                f"seed{seed}/{condition}: nearest-center diagnostic")
    return {"seed": seed, "condition": condition, "row_ids": sampled.tolist(),
            "max_abs_distance_error": float(np.max(np.abs(observed - minimum)))}


def verify_optional_transitions(output, frames):
    path = output / "normal_request_transitions.csv"
    if not path.is_file():
        return {"status": "NOT_PRESENT", "checked": False}
    table = read_csv(path)
    keys = ["seed", "condition", "score", "policy"]
    expected = {(s, c, method, p) for s in SEEDS for c in CONDITIONS for method in SCORES for p in POLICIES}
    require(not table.duplicated(keys).any(), "Duplicate transition settings")
    require(set(table[keys].itertuples(index=False, name=None)) == expected,
            "Expected all 54 four-state transition settings")
    indexed = table.set_index(keys)
    state_names = ("accepted_correct", "accepted_wrong", "rejected_correct", "rejected_wrong")
    for seed, condition, score, policy in sorted(expected):
        clean, frame = frames[seed, "clean"], frames[seed, condition]
        normal = ~booleans(frame.is_oos, "transition scope")
        n = int(normal.sum())
        correct = frame.predicted_label_id.to_numpy()[normal] == frame.label_id.to_numpy()[normal]
        clean_correct = clean.predicted_label_id.to_numpy()[normal] == clean.label_id.to_numpy()[normal]
        rejected = booleans(frame[f"{score}_{policy}_reject"], "transition decisions")[normal]
        clean_rejected = booleans(clean[f"{score}_{policy}_reject"], "clean transition decisions")[normal]
        changed = booleans(frame.changed, "transition changed")[normal]
        clean_ac = clean_correct & ~clean_rejected
        states = {"accepted_correct": correct & ~rejected,
                  "accepted_wrong": ~correct & ~rejected,
                  "rejected_correct": correct & rejected,
                  "rejected_wrong": ~correct & rejected}
        require(np.all(np.stack(list(states.values())).sum(axis=0) == 1), "States do not partition IS")
        clean_accepted_n, clean_rejected_n = int((~clean_rejected).sum()), int(clean_rejected.sum())
        new = int((~clean_rejected & rejected).sum())
        recovered = int((clean_rejected & ~rejected).sum())
        require(int(rejected.sum()) - clean_rejected_n == new - recovered,
                "Net rejection change is not new minus recovered")
        def ratio(numerator, denominator):
            return numerator / denominator if denominator else None
        expected_fields = {
            "is_n": n, "changed_is_n": int(changed.sum()),
            "clean_accepted_n": clean_accepted_n, "clean_accepted_correct_n": int(clean_ac.sum()),
            "closed_accuracy": float(correct.mean()), "is_rejection_rate": float(rejected.mean()),
            "rejection_rate_delta_vs_clean": float(rejected.mean() - clean_rejected.mean()),
            "new_rejection_n": new, "new_rejection_rate_over_all_is": new / n,
            "new_rejection_rate_over_clean_accepted": ratio(new, clean_accepted_n),
            "recovered_acceptance_n": recovered, "recovered_acceptance_rate_over_all_is": recovered / n,
            "recovered_acceptance_rate_over_clean_rejected": ratio(recovered, clean_rejected_n),
            "changed_only_rejection_n": int((changed & rejected).sum()),
            "changed_only_rejection_rate": ratio(int((changed & rejected).sum()), int(changed.sum())),
        }
        for state in state_names:
            count = int(states[state].sum())
            flow = int((clean_ac & states[state]).sum())
            expected_fields.update({f"{state}_n": count, f"{state}_rate_over_all_is": count / n,
                                    f"clean_ac_to_{state}_n": flow,
                                    f"clean_ac_to_{state}_rate_over_all_is": flow / n,
                                    f"clean_ac_to_{state}_rate_over_clean_accepted_correct": ratio(flow, int(clean_ac.sum()))})
        require(sum(expected_fields[f"clean_ac_to_{state}_n"] for state in state_names) == int(clean_ac.sum()),
                "Clean-correct-and-accepted flows do not partition their source population")
        recorded = indexed.loc[(seed, condition, score, policy)]
        for field, value in expected_fields.items():
            require(field in recorded, f"Missing transition field: {field}")
            close(value, recorded[field], f"transition/{seed}/{condition}/{score}/{policy}/{field}")
        # Optional explicit totals remain checkable if included by the report.
        for field, value in (("clean_rejected_n", clean_rejected_n),
                             ("net_rejection_change_n", new - recovered)):
            if field in recorded:
                close(value, recorded[field], f"transition/{field}")
    return {"status": "PASS", "checked": True, "verified_rows": len(table),
            "four_current_condition_states_partition_4500_is": True,
            "clean_correct_accepted_flows_verified": True,
            "new_minus_recovered_equals_net_rejection_change": True,
            "sha256": digest(path)}


def verify_unchanged_text_checks(output, frames):
    records = read_json(output / "unchanged_text_checks.json")
    expected_keys = {(seed, condition) for seed in SEEDS for condition in CONDITIONS[1:]}
    require(len(records) == 6 and {(r["seed"], r["condition"]) for r in records} == expected_keys,
            "Unchanged-text check settings")
    summaries = []
    for record in records:
        seed, condition = record["seed"], record["condition"]
        frame, clean = frames[seed, condition], frames[seed, "clean"]
        unchanged = ~booleans(frame.changed, "unchanged-text flags")
        require(np.array_equal(frame.text.to_numpy()[unchanged], clean.text.to_numpy()[unchanged]),
                "Supposedly unchanged text differs from clean input")
        current_logits = np.load(output / f"seed{seed}/{condition}_logits_fp32.npy", allow_pickle=False, mmap_mode="r")
        clean_logits = np.load(ROOT / f"outputs/seed{seed}/test_logits_fp32.npy", allow_pickle=False, mmap_mode="r")
        maximum_delta = float(np.max(np.abs(current_logits[unchanged] - clean_logits[unchanged])))
        argmax_changes = int(np.sum(current_logits[unchanged].argmax(1) != clean_logits[unchanged].argmax(1)))
        close(int(unchanged.sum()), record["unchanged_rows"], "Unchanged row count")
        close(maximum_delta, record["max_abs_logit_delta"], "Unchanged logit delta", atol=1e-12, rtol=1e-12)
        close(argmax_changes, record["changed_argmax"], "Unchanged argmax drift")
        require(set(record["rejection_decision_changes"]) == {f"{s}/{p}" for s in SCORES for p in POLICIES},
                "Unchanged-text rejection inventory")
        for score in SCORES:
            for policy in POLICIES:
                column = f"{score}_{policy}_reject"
                differences = int(np.sum(booleans(frame[column], column)[unchanged] != booleans(clean[column], column)[unchanged]))
                close(differences, record["rejection_decision_changes"][f"{score}/{policy}"], "Unchanged rejection drift")
        summaries.append({"seed": seed, "condition": condition, "unchanged_rows": int(unchanged.sum()),
                          "max_abs_logit_delta": maximum_delta, "changed_argmax": argmax_changes})
    return {"status": "PASS", "verified_rows": 6, "records": summaries,
            "sha256": digest(output / "unchanged_text_checks.json")}


def verify(study):
    output = study / "outputs"
    original_hash_counts = verify_original_freeze()
    followup_hash_counts = verify_followup_freeze(study, output)
    raw = read_json(PROJECT / "data/raw/data_full.json")
    original_rows = raw["test"] + raw["oos_test"]
    inputs = {condition: read_csv(PROJECT / f"outputs/robustness/typo_data/{condition}.csv")
              for condition in CONDITIONS}
    metrics = read_csv(output / "metrics_by_seed.csv")
    key_fields = ["seed", "condition", "score", "policy"]
    expected = {(seed, condition, score, policy) for seed in SEEDS for condition in CONDITIONS
                for score in SCORES for policy in POLICIES}
    require(not metrics.duplicated(key_fields).any(), "Duplicate metrics settings")
    require(set(metrics[key_fields].itertuples(index=False, name=None)) == expected,
            "Expected exactly 54 seed/condition/score/policy settings")
    indexed = metrics.set_index(key_fields)
    frames, distance_checks, prediction_hashes = {}, [], {}
    canonical_mapping = None
    settings_checked = 0
    for seed in SEEDS:
        folder = output / f"seed{seed}"
        original_folder = ROOT / "outputs" / f"seed{seed}"
        mapping = read_json(PROJECT / f"outputs/bert/seed{seed}/label_mapping.json")["label_to_id"]
        require(sorted(mapping.values()) == list(range(150)), "Label mapping")
        if canonical_mapping is None:
            canonical_mapping = mapping
        require(mapping == canonical_mapping, "Mapping differs across seeds")
        inverse = {value: key for key, value in mapping.items()}
        thresholds = read_json(original_folder / "frozen_thresholds.json")
        require(set(thresholds) == {f"{s}/{p}" for s in SCORES for p in POLICIES},
                f"seed{seed}: original threshold inventory")
        with np.load(original_folder / "distance_model.npz", allow_pickle=False) as archive:
            model = {name: np.array(archive[name], copy=True) for name in archive.files}
        require(np.array_equal(model["classes"], np.arange(150)), "Distance model classes")
        require(int(model["n_samples"]) == 15000 and int(model["n_features"]) == 768,
                "Original distance model dimensions")
        np.linalg.cholesky(model["covariance"])
        clean_original = read_csv(original_folder / "test_predictions.csv")
        for condition in CONDITIONS:
            context = f"seed{seed}/{condition}"
            path = folder / f"{condition}_predictions.csv"
            frame = read_csv(path)
            labels, oos = verify_input_frame(frame, inputs[condition], original_rows,
                                             mapping, condition, context)
            array_folder = original_folder if condition == "clean" else folder
            stem = "test" if condition == "clean" else condition
            features = np.load(array_folder / f"{stem}_cls.npy", allow_pickle=False, mmap_mode="r")
            logits = np.load(array_folder / f"{stem}_logits_fp32.npy", allow_pickle=False, mmap_mode="r")
            require(features.shape == (5500, 768) and features.dtype == np.float32, f"{context}: FP32 features")
            require(logits.shape == (5500, 150) and logits.dtype == np.float32, f"{context}: FP32 logits")
            require(np.isfinite(features).all() and np.isfinite(logits).all(), f"{context}: finite arrays")
            prediction = np.asarray(logits).argmax(axis=1)
            require(np.array_equal(frame.predicted_label_id.to_numpy(), prediction), f"{context}: original classifier argmax")
            if "predicted_label" in frame:
                require(frame.predicted_label.tolist() == [inverse[int(k)] for k in prediction], f"{context}: class names")
            z = np.asarray(logits, dtype=np.float64)
            exponentials = np.exp(z - z.max(axis=1, keepdims=True))
            probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
            recomputed = {"msp": 1 - probabilities.max(axis=1),
                          "entropy": -np.sum(probabilities * np.log(probabilities), axis=1)}
            for score, values in recomputed.items():
                array_close(frame[f"{score}_score"].to_numpy(dtype=float), values,
                            f"{context}: independent {score}", atol=1e-13, rtol=1e-12)
            distance_checks.append(direct_distance_sample(features, frame, model, seed, condition))
            if condition == "clean":
                require(np.array_equal(frame.predicted_label_id.to_numpy(), clean_original.predicted_label_id.to_numpy()),
                        f"{context}: clean classifier reuse")
            for score in SCORES:
                values = frame[f"{score}_score"].to_numpy(dtype=np.float64)
                require(np.isfinite(values).all(), f"{context}/{score}: finite scores")
                if condition == "clean":
                    array_close(values, clean_original[f"{score}_score"].to_numpy(),
                                f"{context}/{score}: clean score reuse", atol=1e-9, rtol=1e-12)
                for policy in POLICIES:
                    threshold = thresholds[f"{score}/{policy}"]["threshold"]
                    rejected = values >= threshold
                    field = f"{score}_{policy}_reject"
                    require(np.array_equal(booleans(frame[field], context), rejected),
                            f"{context}/{score}/{policy}: original frozen rejection rule")
                    if condition == "clean":
                        require(np.array_equal(rejected, booleans(clean_original[field], context)),
                                f"{context}/{score}/{policy}: clean decisions changed")
                    recorded = indexed.loc[(seed, condition, score, policy)]
                    require(float(recorded.threshold) == threshold,
                            f"{context}/{score}/{policy}: threshold was changed or refitted")
                    for name, value in manual_metrics(labels, prediction, oos, rejected, values).items():
                        close(value, recorded[name], f"{context}/{score}/{policy}/{name}")
                    known_gold, known_prediction = labels[~oos], prediction[~oos]
                    support = np.bincount(known_gold, minlength=150)
                    predicted = np.bincount(known_prediction, minlength=150)
                    true_positive = np.bincount(known_gold[known_gold == known_prediction], minlength=150)
                    class_f1 = np.divide(2 * true_positive, support + predicted,
                                         out=np.zeros(150), where=support + predicted > 0)
                    close(np.mean(known_gold == known_prediction), recorded["closed_set_accuracy"],
                          f"{context}/{score}/{policy}/closed_set_accuracy")
                    close(class_f1.mean(), recorded["closed_set_macro_f1"],
                          f"{context}/{score}/{policy}/closed_set_macro_f1")
                    settings_checked += 1
            frames[seed, condition] = frame
            prediction_hashes[path.relative_to(output).as_posix()] = digest(path)
    transition_check = verify_optional_transitions(output, frames)
    unchanged_check = verify_unchanged_text_checks(output, frames)
    verify_original_freeze()
    verify_followup_freeze(study, output)
    return {"status": "PASS", "verified_settings": settings_checked,
            "frozen_input_rows_checked": 9 * 5500,
            "original_text_and_inherited_labels_aligned": True,
            "same_frozen_typo_inputs_across_seeds": True,
            "same_classifier_head_for_all_scores": True,
            "unchanged_original_validation_thresholds": True,
            "clean_classifier_and_rejection_decisions_exactly_reused": True,
            "clean_scores_numerically_equal_to_frozen_original": True,
            "independent_manual_confusion_metrics": True,
            "independent_auc_average_precision": True,
            "distance_direct_solve_checks": distance_checks,
            "original_hash_inventory_counts": original_hash_counts,
            "followup_hash_inventory_counts": followup_hash_counts,
            "normal_request_transitions": transition_check,
            "unchanged_text_checks": unchanged_check,
            "verified_prediction_sha256": prediction_hashes,
            "metrics_sha256": digest(output / "metrics_by_seed.csv"),
            "verifier_sha256": digest(Path(__file__)),
            "limitations": ["Saved features/logits are verified; no BERT inference is rerun.",
                            "Inherited labels are checked against the frozen dataset, not manually judged for semantic preservation.",
                            "Direct covariance solves sample six rows per seed/condition, comparing all 150 class centers."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    args = parser.parse_args()
    study = args.study.resolve()
    output = study / "outputs"
    needed = [study / "PROTOCOL.json", study / "INPUTS_FROZEN.json",
              output / "TEST_COMPLETE.json", output / "metrics_by_seed.csv"]
    needed += [output / f"seed{seed}" / f"{condition}_predictions.csv"
               for seed in SEEDS for condition in CONDITIONS]
    missing = [str(path) for path in needed if not path.is_file()]
    if missing:
        print(json.dumps({"status": "NOT_READY", "missing": missing}, ensure_ascii=False, indent=2))
        return 2
    started = datetime.now(timezone.utc).isoformat()
    try:
        context = threadpool_limits(limits=4) if threadpool_limits else nullcontext()
        with context:
            result = verify(study)
    except Exception as error:
        result = {"status": "FAIL", "error_type": type(error).__name__, "error": str(error)}
    result.update(started_at_utc=started, completed_at_utc=datetime.now(timezone.utc).isoformat())
    (output / "verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
