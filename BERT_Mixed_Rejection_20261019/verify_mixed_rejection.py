"""Independent verification of fixed-BERT, mixed-data Mahalanobis fitting.

No experiment runner, distance implementation, report, PyTorch, or sklearn is
imported. Ledoit-Wolf shrinkage is recomputed from NumPy residual moments.
Generic confusion metrics and exhaustive threshold checks are reused only from
the earlier independent verifier, whose source hash is recorded. The only write
is outputs/verification.json; no fitting artifact or prediction is modified.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

ROOT = Path(__file__).resolve().parent
AUGMENTED = ROOT.parent / "BERT_Typo_Augmentation_20261019"
PREVIOUS = ROOT.parent / "BERT_Rejection_Study_20261019"
PROJECT = ROOT.parent / "CLINC150_Team_20261019"
SEEDS = (42, 43, 44)
ARMS = ("clean_only", "clean_duplicate", "clean_plus_typo")
CONDITIONS = ("clean", "typo_light", "typo_medium")
POLICIES = ("max_oos_f1", "is_acceptance_95")
AUGMENTATION_SEED = 31020261020
SCORE = "mahalanobis"


def require(condition, context):
    if not condition:
        raise AssertionError(context)


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


def helpers():
    source = PREVIOUS / "verify_rejection.py"
    spec = importlib.util.spec_from_file_location("independent_base_verifier", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, {source.resolve().as_posix(): digest(source)}


def verify_hashes(inventory, context):
    require(isinstance(inventory, dict) and bool(inventory), f"{context}: empty inventory")
    for absolute, expected in inventory.items():
        path = Path(absolute)
        require(path.is_absolute() and path.is_file(), f"{context}: invalid absolute file {absolute}")
        require(digest(path) == expected, f"{context}: changed {absolute}")
    return len(inventory)


def required_hashes(paths, inventory, context):
    for path in paths:
        require(path.resolve().as_posix() in inventory, f"{context}: unbound file {path}")


def same_columns(observed, expected, columns, context):
    require(len(observed) == len(expected), f"{context}: row count")
    for column in columns:
        require(column in observed and observed[column].tolist() == expected[column].tolist(), f"{context}/{column}: mismatch")


def replay_noise(text, row_id, policy):
    spans = list(re.finditer(r"(?<![A-Za-z0-9])[A-Za-z]{5,}(?![A-Za-z0-9])", text))
    random_seed = int.from_bytes(hashlib.sha256(f"{AUGMENTATION_SEED}\0{row_id}\0{text}".encode()).digest(), "big")
    rng = random.Random(random_seed)
    rng.shuffle(spans)
    mutations = []
    for span in spans[:2]:
        token = span.group()
        swap_indices = [i for i in range(1, len(token) - 2) if token[i] != token[i + 1]]
        if swap_indices and rng.randrange(2) == 0:
            i = rng.choice(swap_indices)
            replacement = token[:i] + token[i + 1] + token[i] + token[i + 2:]
            operation = "swap"
        else:
            i = rng.randrange(1, len(token) - 1)
            replacement, operation = token[:i] + token[i + 1:], "delete"
        mutations.append(dict(start=span.start(), end=span.end(), original=token,
                              replacement=replacement, operation=operation, character_index=i))
    selected = mutations[:1 if policy == "typo_light" else 2]
    changed = text
    for mutation in sorted(selected, key=lambda item: item["start"], reverse=True):
        changed = changed[:mutation["start"]] + mutation["replacement"] + changed[mutation["end"]:]
    return changed, len(spans), selected


def verify_fit_data(h):
    raw = read_json(PROJECT / "data/raw/data_full.json")
    mapping = read_json(PROJECT / "outputs/bert/seed42/label_mapping.json")["label_to_id"]
    require(len(raw["train"]) == 15000 and sorted(mapping.values()) == list(range(150)), "Train/map dimensions")
    noise = read_csv(ROOT / "data/noisy_train.csv")
    require(len(noise) == 15000 and np.array_equal(noise.source_row.to_numpy(), np.arange(15000)), "Noisy source row ownership")
    require(noise.original_text.tolist() == [r[0] for r in raw["train"]], "Noisy source is not official train")
    require(noise.true_label.tolist() == [r[1] for r in raw["train"]], "Noisy inherited labels")
    labels = np.array([mapping[label] for _, label in raw["train"]], dtype=np.int64)
    require(np.array_equal(noise.label_id.to_numpy(), labels), "Noisy label mapping")
    require(np.array_equal(np.bincount(labels), np.full(150, 100)), "Train class counts")
    assigned = np.empty(15000, dtype=object)
    for label in range(150):
        indices = np.random.default_rng(AUGMENTATION_SEED + label).permutation(np.flatnonzero(labels == label))
        assigned[indices[:50]], assigned[indices[50:]] = "typo_light", "typo_medium"
    require(np.array_equal(noise.policy.to_numpy(), assigned), "Noise 50-light/50-medium per-class seed allocation")
    actual_changed = noise.text.to_numpy() != noise.original_text.to_numpy()
    require(np.array_equal(h.booleans(noise.changed, "Noisy training changed"), actual_changed), "Actual noisy changes")
    for row in noise.itertuples(index=False):
        text, eligible, mutations = replay_noise(row.original_text, int(row.source_row), row.policy)
        require(row.text == text and row.eligible_tokens == eligible and row.modified_tokens == len(mutations),
                f"Noise row {row.source_row}: deterministic mutation/counts")
        require(json.loads(row.mutations_json) == mutations, f"Noise row {row.source_row}: mutation trace")
    manifest = read_csv(ROOT / "data/fit_manifest.csv")
    require(set(manifest.arm) == set(ARMS) and len(manifest) == 75000, "Fit manifest arms/row count")
    require(not manifest.duplicated(["arm", "fit_row"]).any(), "Fit manifest duplicate positions")
    for arm in ARMS:
        rows = manifest[manifest.arm == arm]
        n = 15000 if arm == "clean_only" else 30000
        require(np.array_equal(rows.fit_row.to_numpy(), np.arange(n)), f"{arm}: fit row sequence")
        source_rows = np.arange(15000) if n == 15000 else np.tile(np.arange(15000), 2)
        require(np.array_equal(rows.source_row.to_numpy(), source_rows), f"{arm}: fit row source ownership")
        expected_views = ["clean"] * n if arm != "clean_plus_typo" else ["clean"] * 15000 + ["noisy"] * 15000
        require(rows.view.tolist() == expected_views, f"{arm}: fit view ordering")
        require(np.array_equal(rows.label_id.to_numpy(), labels[source_rows]), f"{arm}: fit labels")
    summary = read_json(ROOT / "data/data_summary.json")
    expected_summary = {"source_rows": 15000, "noise_seed": AUGMENTATION_SEED,
        "policy_counts": noise.policy.value_counts().to_dict(), "actual_changed_rows": int(actual_changed.sum()),
        "modified_token_counts": {str(k): int(v) for k, v in noise.modified_tokens.value_counts().items()},
        "planned_mixed_pool_counts": {"clean": 15000, "typo_light": 7500, "typo_medium": 7500},
        "actual_mixed_pool_unchanged_rows": 15000 + int((~actual_changed).sum()),
        "fit_rows": {"clean_only": 15000, "clean_duplicate": 30000, "clean_plus_typo": 30000}}
    for key, expected in expected_summary.items():
        require(summary[key] == expected, f"Data summary/{key}")
    h.close(summary["noisy_view_changed_fraction"], actual_changed.mean(), "Noisy changed fraction")
    h.close(summary["mixed_pool_changed_fraction"], actual_changed.sum() / 30000, "Mixed pool changed fraction")
    return {"raw": raw, "mapping": mapping, "labels": labels, "noise": noise,
            "data_checks": {"noise_rows": 15000, "fit_manifest_rows": 75000,
                "policy_counts": noise.policy.value_counts().to_dict(), "actual_changed_rows": int(actual_changed.sum()),
                "actual_unchanged_rows_retained": int((~actual_changed).sum()),
                "source_and_mutation_replay_verified": True}}


def independent_statistics(features, labels):
    """Use residual moments, not sklearn's blocked fourth-moment product."""
    values = np.asarray(features, dtype=np.float64)
    n, p = values.shape
    require(p == 768 and n in (15000, 30000), "Fit feature dimensions")
    means = np.stack([values[labels == c].mean(axis=0) for c in range(150)])
    residual = values - means[labels]
    covariance = residual.T @ residual / n
    mu = np.trace(covariance) / p
    squared_frobenius = np.square(covariance).sum()
    squared_row_norms = np.einsum("ij,ij->i", residual, residual)
    beta = (np.mean(np.square(squared_row_norms)) - squared_frobenius) / (p * n)
    delta = np.square(covariance - mu * np.eye(p)).sum() / p
    beta = min(beta, delta)
    shrinkage = 0. if beta == 0 else beta / delta
    shrunk = (1 - shrinkage) * covariance + shrinkage * mu * np.eye(p)
    return {"means": means, "origin": values.mean(axis=0), "empirical": covariance,
            "shrinkage": float(shrinkage), "covariance": shrunk,
            "counts": np.bincount(labels, minlength=150), "n": n}


def verify_distance_fit(path, expected, h, context):
    with np.load(path, allow_pickle=False) as archive:
        model = {name: np.array(archive[name], copy=True) for name in archive.files}
    require(int(model["format_version"]) == 1 and int(model["n_samples"]) == expected["n"]
            and int(model["n_features"]) == 768, f"{context}: distance dimensions")
    require(np.array_equal(model["classes"], np.arange(150)) and np.array_equal(model["class_counts"], expected["counts"]), f"{context}: distance class counts")
    h.array_close(model["class_means"], expected["means"], f"{context}: train-only means", atol=2e-12, rtol=2e-12)
    h.array_close(model["origin"], expected["origin"], f"{context}: origin", atol=2e-12, rtol=2e-12)
    h.array_close(model["class_offsets"] + model["origin"], expected["means"], f"{context}: offsets", atol=2e-12, rtol=2e-12)
    h.close(float(model["shrinkage"]), expected["shrinkage"], f"{context}: independent Ledoit-Wolf coefficient", atol=1e-12, rtol=1e-10)
    h.array_close(model["covariance"], expected["covariance"], f"{context}: independently shrunk covariance", atol=1e-11, rtol=1e-9)
    np.linalg.cholesky(model["covariance"])
    h.array_close(model["covariance"] @ model["precision"], np.eye(768), f"{context}: inverse", atol=1e-5, rtol=1e-5)
    return model


def verify_direct_score(features, frame, model, h, context):
    labels = frame.label_id.to_numpy(dtype=np.int64)
    rng = np.random.default_rng(20261020)
    selected = np.r_[rng.choice(np.flatnonzero(labels >= 0), 3, replace=False),
                     rng.choice(np.flatnonzero(labels < 0), 3, replace=False)]
    difference = np.asarray(features[selected], dtype=np.float64)[:, None] - model["class_means"][None]
    flat = difference.reshape(-1, 768)
    solution = np.linalg.solve(model["covariance"], flat.T).T
    distances = np.sum(flat * solution, axis=1).reshape(6, 150)
    actual = frame.mahalanobis_score.to_numpy(dtype=np.float64)[selected]
    expected = distances.min(axis=1)
    h.array_close(actual, expected, f"{context}: direct covariance solve", atol=1e-7, rtol=1e-8)
    require(np.array_equal(frame.nearest_centroid_label_id_diagnostic_only.to_numpy()[selected], distances.argmin(axis=1)), f"{context}: nearest-center diagnostic")
    return {"context": context, "row_ids": selected.tolist(), "max_abs_distance_error": float(np.max(np.abs(actual - expected)))}


def index_metrics(table, conditions, context):
    keys = ["arm", "seed", "condition", "score", "policy"]
    expected = {(a, s, c, SCORE, p) for a in ARMS for s in SEEDS for c in conditions for p in POLICIES}
    require(not table.duplicated(keys).any() and set(table[keys].itertuples(index=False, name=None)) == expected,
            f"{context}: expected {len(expected)} unique settings")
    return table.set_index(keys).sort_index()


def verify_metrics(frame, logits, thresholds, recorded, h, context):
    labels = frame.label_id.to_numpy(dtype=np.int64)
    oos = h.booleans(frame.is_oos, context)
    require(np.array_equal(oos, labels == -1), f"{context}: scope flags")
    prediction = logits.argmax(axis=1)
    require(np.array_equal(frame.predicted_label_id.to_numpy(), prediction), f"{context}: fixed classifier head")
    scores = frame.mahalanobis_score.to_numpy(dtype=np.float64)
    require(np.isfinite(scores).all(), f"{context}: finite distance scores")
    normal = ~oos
    true_counts = np.bincount(labels[normal], minlength=150)
    prediction_counts = np.bincount(prediction[normal], minlength=150)
    correct_counts = np.bincount(labels[normal & (labels == prediction)], minlength=150)
    closed_f1 = np.mean(np.divide(2 * correct_counts, true_counts + prediction_counts,
                                out=np.zeros(150), where=true_counts + prediction_counts > 0))
    require(set(thresholds) == {f"mahalanobis/{p}" for p in POLICIES}, f"{context}: threshold policies")
    for policy in POLICIES:
        threshold = thresholds[f"mahalanobis/{policy}"]["threshold"]
        rejected = scores >= threshold
        require(np.array_equal(h.booleans(frame[f"mahalanobis_{policy}_reject"], context), rejected), f"{context}: >= decisions")
        row = recorded.loc[(SCORE, policy)]
        require(float(row.threshold) == threshold, f"{context}: own frozen threshold")
        for name, value in h.manual_metrics(labels, prediction, oos, rejected, scores).items():
            h.close(value, row[name], f"{context}/{policy}/{name}")
        h.close(float((prediction[normal] == labels[normal]).mean()), row.closed_set_accuracy, f"{context}: fixed closed accuracy")
        h.close(closed_f1, row.closed_set_macro_f1, f"{context}: fixed closed macro F1")
    return 2


def verify_manifest_chain(output, h):
    protocol = read_json(ROOT / "PROTOCOL.json")
    require(protocol["seeds"] == list(SEEDS) and protocol["arms"] == list(ARMS)
            and protocol["conditions"] == list(CONDITIONS), "Protocol experiment grid")
    require(protocol["augmentation"]["seed"] == AUGMENTATION_SEED and
            protocol["augmentation"]["policy_counts"] == {"typo_light": 7500, "typo_medium": 7500}, "Protocol training noise seed/counts")
    require(protocol["augmentation"]["independent_of_model_seed"] is True, "Noise allocation depends on model seed")
    require(protocol["inference"] == read_json(AUGMENTED / "PROTOCOL.json")["inference"], "Protocol FP32 inference")
    require(protocol["clean_guardrails"] == ["oos_f1", "oos_recall", "in_scope_acceptance"], "Protocol clean guardrails")
    frozen_path, development_path = ROOT / "INPUTS_FROZEN.json", output / "DEVELOPMENT_FROZEN.json"
    frozen, development, complete = read_json(frozen_path), read_json(development_path), read_json(output / "TEST_COMPLETE.json")
    require(set(frozen) == {"frozen_at", "runtime", "input_hashes"}, "Frozen input manifest schema")
    require(frozen["runtime"] == read_json(PREVIOUS / "outputs/environment.json"), "Runtime changed")
    input_hash, dev_hash = digest(frozen_path), digest(development_path)
    require(development["inputs_frozen_sha256"] == complete["inputs_frozen_sha256"] == input_hash, "Input-freeze bindings")
    require(complete["development_frozen_sha256"] == dev_hash, "Development-freeze binding")
    require(development["detectors"] == 9 and development["validation_thresholds"] == 18, "Development totals")
    require(complete["test_settings"] == 54 and complete["original_inputs_and_models_unchanged"] is True, "Test totals/immutability flag")
    time = datetime.fromisoformat
    require(time(frozen["frozen_at"]) <= time(development["frozen_at"]) <= time(complete["started_at"]) <= time(complete["completed_at"]),
            "Test scoring started before all development was frozen")
    counts = {"frozen_inputs": verify_hashes(frozen["input_hashes"], "Inputs"),
        "development_outputs": verify_hashes(development["development_hashes"], "Development"),
        "test_outputs": verify_hashes(complete["output_hashes"], "Test")}
    olddev = read_json(AUGMENTED / "outputs/DEVELOPMENT_FROZEN.json")
    oldtest = read_json(AUGMENTED / "outputs/TEST_COMPLETE.json")
    require(oldtest["development_frozen_sha256"] == digest(AUGMENTED / "outputs/DEVELOPMENT_FROZEN.json"), "Augmented-model development binding")
    require(olddev["inputs_frozen_sha256"] == oldtest["inputs_frozen_sha256"] == digest(AUGMENTED / "INPUTS_FROZEN.json"), "Augmented-model input binding")
    required_input = [ROOT / "PROTOCOL.json", ROOT / "run_mixed_rejection.py", PROJECT / "data/raw/data_full.json",
        PROJECT / "src/typo_data.py", PROJECT / "src/evaluate_oos.py", PROJECT / "src/common.py",
        PREVIOUS / "run_rejection_study.py", PREVIOUS / "distance_model.py", PREVIOUS / "outputs/environment.json",
        AUGMENTED / "PROTOCOL.json", AUGMENTED / "outputs/DEVELOPMENT_FROZEN.json", AUGMENTED / "outputs/TEST_COMPLETE.json"]
    required_input += [ROOT / "data" / name for name in ("noisy_train.csv", "fit_manifest.csv", "data_summary.json")]
    source_bound = 0
    for seed in SEEDS:
        source = AUGMENTED / f"outputs/typo_mix/seed{seed}"
        required_input += [PROJECT / f"outputs/bert/seed{seed}/label_mapping.json"]
        for path in source.rglob("*"):
            if path.is_file():
                key = path.resolve().as_posix()
                reference = olddev["development_hashes"].get(key, oldtest["output_hashes"].get(key))
                require(reference is not None and digest(path) == reference, f"Source changed against augmented freeze: {path}")
                required_input.append(path)
                source_bound += 1
        required_input += [source / "final_model/model.safetensors", source / "distance_model.npz", source / "frozen_thresholds.json"]
        for arm in ARMS:
            folder = output / arm / f"seed{seed}"
            local_path = folder / "DEVELOPMENT_COMPLETE.json"
            local = read_json(local_path)
            require(local["arm"] == arm and local["seed"] == seed and local["inputs_frozen_sha256"] == input_hash, f"{arm}/{seed}: development identity/binding")
            n = 15000 if arm == "clean_only" else 30000
            require(local["fit_rows"] == n and local["class_counts"] == [n // 150] * 150, f"{arm}/{seed}: fit counts")
            require(time(frozen["frozen_at"]) <= time(local["completed_at"]) <= time(development["frozen_at"]), f"{arm}/{seed}: development chronology")
            verify_hashes(local["output_hashes"], f"{arm}/{seed}: development output")
            required_local = [folder / name for name in ("distance_model.npz", "frozen_thresholds.json",
                "validation_mahalanobis_threshold_curve.csv", "validation_predictions.csv", "validation_metrics.csv")]
            required_hashes(required_local, local["output_hashes"], f"{arm}/{seed}: local development")
            required_hashes(required_local + [local_path], development["development_hashes"], f"{arm}/{seed}: global development")
            require(set(local["output_hashes"]).issubset(development["development_hashes"]), f"{arm}/{seed}: omitted development dependency")
            required_hashes([folder / f"{c}_{suffix}.csv" for c in CONDITIONS for suffix in ("predictions", "metrics")],
                            complete["output_hashes"], f"{arm}/{seed}: completed tests")
    required_hashes(required_input, frozen["input_hashes"], "Frozen inputs")
    required_hashes([output / "validation_metrics_by_seed.csv"], development["development_hashes"], "Validation metrics")
    required_hashes([output / "metrics_by_seed.csv"], complete["output_hashes"], "Test metrics")
    state = verify_fit_data(h)
    for seed in SEEDS:
        require(read_json(PROJECT / f"outputs/bert/seed{seed}/label_mapping.json")["label_to_id"] == state["mapping"], "Mapping differs by seed")
    state.update(frozen=frozen, development=development, complete=complete, input_hash=input_hash, development_hash=dev_hash,
                 hash_inventory_counts=counts, immutable_source_artifacts=source_bound)
    return state


def verify_noise_features(seed, state, h):
    folder = ROOT / f"features/seed{seed}"
    paths = [folder / f"noisy_train_{suffix}" for suffix in ("cls.npy", "logits_fp32.npy", "metadata.csv", "extraction.json")]
    required_hashes(paths, state["development"]["development_hashes"], "Noisy train features")
    record = read_json(folder / "noisy_train_extraction.json")
    metadata = read_csv(folder / "noisy_train_metadata.csv")
    same_columns(metadata, state["noise"], list(state["noise"].columns), f"Noisy feature metadata seed{seed}")
    require(record["inputs_frozen_sha256"] == state["input_hash"], "Noisy extraction input binding")
    require(record["model_weights_sha256"] == digest(AUGMENTED / f"outputs/typo_mix/seed{seed}/final_model/model.safetensors"), "Noisy extraction fixed model binding")
    expected_text_hash = hashlib.sha256(json.dumps(state["noise"].text.tolist(), ensure_ascii=False).encode()).hexdigest()
    require(record["text_sha256"] == expected_text_hash and record["rows"] == 15000, "Noisy extraction text binding")
    require(record["inference"] == read_json(ROOT / "PROTOCOL.json")["inference"], "Noisy extraction FP32 protocol")
    require(record["features_sha256"] == digest(paths[0]) and record["logits_sha256"] == digest(paths[1]), "Noisy extraction array binding")
    time = datetime.fromisoformat
    require(time(state["frozen"]["frozen_at"]) <= time(record["completed_at"]) <= time(state["development"]["frozen_at"]), "Noisy extraction freeze order")
    for arm in ARMS:
        local = read_json(ROOT / f"outputs/{arm}/seed{seed}/DEVELOPMENT_COMPLETE.json")
        require(time(record["completed_at"]) <= time(local["completed_at"]), "Detector completed before noisy feature extraction")
    features = np.load(paths[0], mmap_mode="r", allow_pickle=False)
    logits = np.load(paths[1], mmap_mode="r", allow_pickle=False)
    require(features.shape == (15000, 768) and features.dtype == np.float32, "Noisy CLS dimensions/precision")
    require(logits.shape == (15000, 150) and logits.dtype == np.float32, "Noisy logits dimensions/precision")
    require(np.isfinite(features).all() and np.isfinite(logits).all(), "Noisy extraction finite outputs")
    return features, logits


def compare_local_metrics(path, indexed, h, context):
    local = read_csv(path)
    require(len(local) == 2 and not local.duplicated(["score", "policy"]).any(), f"{context}: local metric grid")
    for row in local.to_dict("records"):
        aggregate = indexed.loc[(row["score"], row["policy"])]
        for field, value in row.items():
            if field not in ("arm", "seed", "condition", "score", "policy"):
                h.close(value, aggregate[field], f"{context}/{field}")


def verify_transitions(output, frames, h):
    state_path = output / "normal_states.csv"
    transition_path = output / "rejection_transitions.csv"
    missing = [str(path) for path in (state_path, transition_path) if not path.is_file()]
    if missing:
        return {"status": "NOT_PRESENT", "checked": False, "missing": missing}
    table = index_metrics(read_csv(state_path), CONDITIONS, "Normal states")
    for (arm, seed, condition), frame in frames.items():
        normal = frame.label_id.to_numpy() >= 0
        correct = (frame.predicted_label_id.to_numpy() == frame.label_id.to_numpy())[normal]
        require(int(normal.sum()) == 4500, "Transition IS sample size")
        for policy in POLICIES:
            rejected = h.booleans(frame[f"mahalanobis_{policy}_reject"], "Transition rejection")[normal]
            states = {"accepted_correct": correct & ~rejected, "accepted_wrong": ~correct & ~rejected,
                      "rejected_correct": correct & rejected, "rejected_wrong": ~correct & rejected}
            require(np.all(np.stack(list(states.values())).sum(axis=0) == 1), "Four-state IS partition")
            expected = dict(n_is=4500)
            for name, mask in states.items():
                count = int(mask.sum())
                expected.update({f"{name}_n": count, f"{name}_rate_over_all_is": count / 4500})
            recorded = table.loc[(arm, seed, condition, SCORE, policy)]
            for field, value in expected.items():
                h.close(value, recorded[field], f"State/{arm}/{seed}/{condition}/{policy}/{field}")
    transitions = read_csv(transition_path)
    keys = ["comparison", "reference", "seed", "condition", "score", "policy", "stratum"]
    expected_keys = {(f"clean_plus_typo_minus_{r}", r, s, c, SCORE, p, z) for r in ARMS[:2]
                     for s in SEEDS for c in CONDITIONS for p in POLICIES for z in ("all_is", "head_correct", "head_wrong")}
    require(not transitions.duplicated(keys).any() and set(transitions[keys].itertuples(index=False, name=None)) == expected_keys,
            "Cross-detector transition grid: expected 108 settings")
    indexed = transitions.set_index(keys).sort_index()
    for reference in ARMS[:2]:
        for seed in SEEDS:
            for condition in CONDITIONS:
                source, target = frames[(reference, seed, condition)], frames[("clean_plus_typo", seed, condition)]
                normal = source.label_id.to_numpy() >= 0
                correct = (source.predicted_label_id.to_numpy() == source.label_id.to_numpy())[normal]
                for policy in POLICIES:
                    before = h.booleans(source[f"mahalanobis_{policy}_reject"], "Source decisions")[normal]
                    after = h.booleans(target[f"mahalanobis_{policy}_reject"], "Target decisions")[normal]
                    for stratum, mask in (("all_is", np.ones(4500, dtype=bool)), ("head_correct", correct), ("head_wrong", ~correct)):
                        flows = {"accept_accept": ~before & ~after & mask, "accept_reject": ~before & after & mask,
                                 "reject_accept": before & ~after & mask, "reject_reject": before & after & mask}
                        n = int(mask.sum())
                        require(sum(int(flow.sum()) for flow in flows.values()) == n, "Cross-detector flow partition")
                        net = int((after & mask).sum()) - int((before & mask).sum())
                        require(net == int(flows["accept_reject"].sum()) - int(flows["reject_accept"].sum()), "Cross-detector net identity")
                        expected = dict(n_is=4500, stratum_n=n, source_accepted_n=int((~before & mask).sum()),
                                        source_rejected_n=int((before & mask).sum()), net_rejection_change_n=net,
                                        net_rejection_change_rate_over_all_is=net / 4500)
                        for name, flow in flows.items():
                            count = int(flow.sum())
                            expected.update({f"{name}_n": count, f"{name}_rate_over_all_is": count / 4500,
                                             f"{name}_rate_over_stratum": count / n if n else None})
                        recorded = indexed.loc[(f"clean_plus_typo_minus_{reference}", reference, seed, condition, SCORE, policy, stratum)]
                        for field, value in expected.items():
                            h.close(value, recorded[field], f"Cross-detector transition/{reference}/{seed}/{condition}/{policy}/{stratum}/{field}")
    return {"status": "PASS", "checked": True, "verified_state_rows": 54, "verified_transition_rows": 108,
            "normal_states_sha256": digest(state_path), "rejection_transitions_sha256": digest(transition_path),
            "four_states_and_fixed_head_strata_verified": True, "net_equals_accept_reject_minus_reject_accept": True}


def verify_guardrail_file(output, metrics, h):
    path = output / "clean_guardrails.csv"
    if not path.is_file():
        return {"status": "NOT_PRESENT", "checked": False}
    table = read_csv(path)
    expected_metrics = ("oos_f1", "oos_recall", "in_scope_acceptance")
    require(len(table) == 3 and set(table.metric) == set(expected_metrics), "Clean guardrail metric grid")
    changes, means = {}, {}
    for metric in expected_metrics:
        before = np.mean([metrics.loc[("clean_only", s, "clean", SCORE, "is_acceptance_95"), metric] for s in SEEDS])
        after = np.mean([metrics.loc[("clean_plus_typo", s, "clean", SCORE, "is_acceptance_95"), metric] for s in SEEDS])
        means[metric], changes[metric] = (before, after), after - before
    all_pass = all(value >= 0 for value in changes.values())
    for row in table.itertuples(index=False):
        require(row.reference == "clean_only" and row.target == "clean_plus_typo" and row.condition == "clean"
                and row.score == SCORE and row.policy == "is_acceptance_95", "Guardrail identity")
        before, after = means[row.metric]
        for field, value in (("reference_mean", before), ("target_mean", after), ("delta", changes[row.metric]), ("delta_pp", changes[row.metric] * 100)):
            h.close(getattr(row, field), value, f"Guardrail/{row.metric}/{field}")
        require(bool(h.booleans(pd.Series([row.mean_non_decline]), "Guardrail non-decline")[0]) == (changes[row.metric] >= 0), "Guardrail metric strict sign")
        require(bool(h.booleans(pd.Series([row.all_three_pass]), "Guardrail all-pass")[0]) == all_pass, "Guardrail all-three strict sign")
    return {"status": "PASS", "checked": True, "verified_rows": 3, "all_three_pass": all_pass, "sha256": digest(path)}


def verify_results(output, h, state):
    metrics = index_metrics(read_csv(output / "metrics_by_seed.csv"), CONDITIONS, "Test metrics")
    validation = index_metrics(read_csv(output / "validation_metrics_by_seed.csv"), ("validation",), "Validation metrics")
    labels, raw, mapping = state["labels"], state["raw"], state["mapping"]
    inverse = {v: k for k, v in mapping.items()}
    distance_checks, fitting_checks, frames, fallback_checks = [], [], {}, []
    checked_val = checked_test = candidate_count = 0
    clean_reference_max_error = 0.
    for seed in SEEDS:
        original = AUGMENTED / f"outputs/typo_mix/seed{seed}"
        clean = np.load(original / "train_cls.npy", allow_pickle=False, mmap_mode="r")
        train_meta = read_csv(original / "train_metadata.csv")
        h.verify_metadata(train_meta, raw["train"], ["train"] * 15000, mapping, f"Seed {seed}: original clean fit source")
        require(clean.shape == (15000, 768) and clean.dtype == np.float32 and np.isfinite(clean).all(), "Clean fit features")
        noisy, noisy_logits = verify_noise_features(seed, state, h)
        unchanged = ~h.booleans(state["noise"].changed, "Training unchanged fallback")
        clean_logits = np.load(original / "train_logits_fp32.npy", allow_pickle=False, mmap_mode="r")
        cls_delta = np.abs(np.asarray(noisy[unchanged], dtype=np.float64) - clean[unchanged])
        logit_delta = np.abs(np.asarray(noisy_logits[unchanged], dtype=np.float64) - clean_logits[unchanged])
        fallback_checks.append({"seed": seed, "unchanged_text_rows": int(unchanged.sum()),
            "max_abs_cls_delta": float(cls_delta.max(initial=0)), "mean_abs_cls_delta": float(cls_delta.mean()),
            "max_abs_logit_delta": float(logit_delta.max(initial=0)), "mean_abs_logit_delta": float(logit_delta.mean()),
            "classifier_argmax_changes": int((noisy_logits[unchanged].argmax(axis=1) != clean_logits[unchanged].argmax(axis=1)).sum()),
            "interpretation": "Identical text across different dynamic-padding batches; numeric differences are measured, not required to be bitwise zero."})
        del noisy_logits, clean_logits, cls_delta, logit_delta
        per_arm_statistics = {}
        for arm in ARMS:
            folder = output / arm / f"seed{seed}"
            features = clean if arm == "clean_only" else np.concatenate([clean, clean if arm == "clean_duplicate" else noisy])
            targets = labels if arm == "clean_only" else np.tile(labels, 2)
            statistics = independent_statistics(features, targets)
            del features
            per_arm_statistics[arm] = statistics
            model = verify_distance_fit(folder / "distance_model.npz", statistics, h, f"{arm}/seed{seed}")
            local = read_json(folder / "DEVELOPMENT_COMPLETE.json")
            h.close(local["shrinkage"], statistics["shrinkage"], f"{arm}/{seed}: shrinkage ledger")
            thresholds = read_json(folder / "frozen_thresholds.json")
            if arm == "clean_only":
                require(digest(folder / "distance_model.npz") == digest(original / "distance_model.npz"), "Clean reference detector is not an exact copy")
                expected = {k: v for k, v in read_json(original / "frozen_thresholds.json").items() if k.startswith("mahalanobis/")}
                require(thresholds == expected, "Clean reference thresholds changed")
            fitting_checks.append({"arm": arm, "seed": seed, "rows": statistics["n"],
                                   "independent_shrinkage": statistics["shrinkage"], "stored_shrinkage": float(model["shrinkage"])})
            for condition in ("validation",) + CONDITIONS:
                context = f"{arm}/{seed}/{condition}"
                frame = read_csv(folder / f"{condition}_predictions.csv")
                metadata = read_csv(original / f"{condition}_metadata.csv")
                source_predictions = read_csv(original / f"{condition}_predictions.csv")
                same_columns(frame, metadata, list(metadata.columns), f"{context}: reused input metadata")
                same_columns(frame, source_predictions, ["predicted_label_id", "predicted_label"], f"{context}: frozen original predictions")
                rows = raw["val"] + raw["oos_val"] if condition == "validation" else raw["test"] + raw["oos_test"]
                expected_labels = np.array([mapping.get(y, -1) for _, y in rows], dtype=np.int64)
                require(np.array_equal(frame.row_id.to_numpy(), np.arange(len(rows))) and np.array_equal(frame.label_id.to_numpy(), expected_labels), f"{context}: row/label alignment")
                require(frame.true_label.tolist() == [r[1] for r in rows], f"{context}: inherited labels")
                n_is, n_oos = (3000, 100) if condition == "validation" else (4500, 1000)
                require(len(frame) == n_is + n_oos and int((expected_labels < 0).sum()) == n_oos
                        and np.array_equal(np.bincount(expected_labels[expected_labels >= 0], minlength=150), np.full(150, n_is // 150)), f"{context}: class/scope counts")
                if condition == "validation":
                    require(frame.text.tolist() == [r[0] for r in rows], f"{context}: clean validation text")
                else:
                    require(frame.original_text.tolist() == [r[0] for r in rows], f"{context}: original test text")
                    require(np.array_equal(h.booleans(frame.changed, context), frame.text.to_numpy() != frame.original_text.to_numpy()), f"{context}: changed flag")
                cls = np.load(original / f"{condition}_cls.npy", mmap_mode="r", allow_pickle=False)
                logits = np.load(original / f"{condition}_logits_fp32.npy", mmap_mode="r", allow_pickle=False)
                require(cls.shape == (len(frame), 768) and cls.dtype == np.float32 and logits.shape == (len(frame), 150)
                        and logits.dtype == np.float32 and np.isfinite(cls).all() and np.isfinite(logits).all(), f"{context}: reused FP32 arrays")
                require(frame.predicted_label.tolist() == [inverse[int(i)] for i in logits.argmax(axis=1)], f"{context}: decoded classifier labels")
                distance_checks.append(verify_direct_score(cls, frame, model, h, context))
                if arm == "clean_only":
                    error = float(np.max(np.abs(frame.mahalanobis_score.to_numpy() - source_predictions.mahalanobis_score.to_numpy())))
                    clean_reference_max_error = max(clean_reference_max_error, error)
                    require(error == 0., f"{context}: clean reference scores changed")
                    same_columns(frame, source_predictions, [f"mahalanobis_{p}_reject" for p in POLICIES], f"{context}: clean reference decisions")
                index = validation if condition == "validation" else metrics
                recorded = index.loc[(arm, seed, condition)]
                count = verify_metrics(frame, logits, thresholds, recorded, h, context)
                compare_local_metrics(folder / f"{condition}_metrics.csv", recorded, h, context)
                if condition == "validation":
                    curve = read_csv(folder / "validation_mahalanobis_threshold_curve.csv")
                    candidate_count += h.verify_validation_thresholds(frame.mahalanobis_score.to_numpy(dtype=float),
                        h.booleans(frame.is_oos, context), {p: thresholds[f"mahalanobis/{p}"] for p in POLICIES}, curve, context)
                    checked_val += count
                else:
                    checked_test += count
                    frames[(arm, seed, condition)] = frame
                del cls, logits
            del model
        h.array_close(per_arm_statistics["clean_duplicate"]["means"], per_arm_statistics["clean_only"]["means"], f"Seed {seed}: duplicate means", atol=2e-12, rtol=2e-12)
        h.array_close(per_arm_statistics["clean_duplicate"]["empirical"], per_arm_statistics["clean_only"]["empirical"], f"Seed {seed}: duplicate empirical covariance", atol=1e-12, rtol=1e-10)
        # At the observed nonsaturated coefficient the duplicate estimator's
        # sample-count term halves; this is not a requirement for all datasets.
        if per_arm_statistics["clean_only"]["shrinkage"] < 1:
            h.close(per_arm_statistics["clean_duplicate"]["shrinkage"], per_arm_statistics["clean_only"]["shrinkage"] / 2,
                    f"Seed {seed}: duplicate Ledoit-Wolf sample-count effect", atol=1e-12, rtol=1e-10)
        del clean, noisy, per_arm_statistics
    guardrails = []
    for arm in ARMS[1:]:
        for policy in POLICIES:
            changes = {}
            for metric in ("oos_f1", "oos_recall", "in_scope_acceptance"):
                baseline = np.array([metrics.loc[("clean_only", s, "clean", SCORE, policy), metric] for s in SEEDS], dtype=float)
                proposed = np.array([metrics.loc[(arm, s, "clean", SCORE, policy), metric] for s in SEEDS], dtype=float)
                changes[metric] = float(proposed.mean() - baseline.mean())
            guardrails.append({"arm": arm, "policy": policy, "mean_changes_vs_clean_only": changes,
                "strict_descriptive_retention": all(v >= 0 for v in changes.values()), "classification_identical": True})
    transitions = verify_transitions(output, frames, h)
    guardrail_file = verify_guardrail_file(output, metrics, h)
    require(digest(ROOT / "INPUTS_FROZEN.json") == state["input_hash"] and digest(output / "DEVELOPMENT_FROZEN.json") == state["development_hash"], "Manifest changed during verification")
    verify_hashes(state["frozen"]["input_hashes"], "Ending inputs")
    verify_hashes(state["development"]["development_hashes"], "Ending development")
    verify_hashes(state["complete"]["output_hashes"], "Ending test")
    return {"verified_test_settings": checked_test, "verified_validation_settings": checked_val,
        "validation_candidates_checked": candidate_count, "test_prediction_rows_checked": 27 * 5500,
        "data_checks": state["data_checks"], "independent_fit_statistics": fitting_checks,
        "class_means_empirical_covariance_and_ledoit_wolf_independently_verified": True,
        "clean_duplicate_empirical_statistics_and_shrinkage_sample_count_effect_verified": True,
        "all_validation_candidates_and_tie_rules_verified": True,
        "fixed_classifier_and_cached_predictions_verified": True,
        "clean_only_reference_max_abs_score_difference": clean_reference_max_error,
        "manual_confusion_auc_ap_and_closed_metrics_verified": True,
        "direct_covariance_solve_checks": distance_checks, "normal_transitions": transitions,
        "unchanged_noisy_training_fallback_checks": fallback_checks, "clean_guardrail_file": guardrail_file,
        "independently_computed_clean_guardrails": guardrails,
        "immutable_source_artifacts": state["immutable_source_artifacts"], "hash_inventory_counts": state["hash_inventory_counts"],
        "inputs_manifest_sha256": state["input_hash"], "development_manifest_sha256": state["development_hash"],
        "test_completion_sha256": digest(output / "TEST_COMPLETE.json"), "metrics_sha256": digest(output / "metrics_by_seed.csv"),
        "limitations": ["Verifies saved FP32 features, classifier logits, source model hashes and inference bindings; does not rerun BERT inference.",
            "Direct covariance solves sample six rows per detector/condition, checking all 150 centers for each sampled row.",
            "Strict retention is a signed mean comparison of three frozen seeds, not a statistical noninferiority conclusion."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    output = args.output.resolve()
    required = [ROOT / "PROTOCOL.json", ROOT / "INPUTS_FROZEN.json", output / "DEVELOPMENT_FROZEN.json",
                output / "TEST_COMPLETE.json", output / "metrics_by_seed.csv"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print(json.dumps({"status": "NOT_READY", "missing": missing}, ensure_ascii=False, indent=2))
        return 2
    started = datetime.now(timezone.utc).isoformat()
    try:
        h, helper_hashes = helpers()
        context = threadpool_limits(limits=4) if threadpool_limits else nullcontext()
        with context:
            state = verify_manifest_chain(output, h)
            result = verify_results(output, h, state)
        result.update(status="PASS", independent_helper_sha256=helper_hashes, verifier_sha256=digest(Path(__file__)))
    except Exception as error:
        result = {"status": "FAIL", "error_type": type(error).__name__, "error": str(error)}
    result.update(started_at_utc=started, completed_at_utc=datetime.now(timezone.utc).isoformat())
    (output / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
