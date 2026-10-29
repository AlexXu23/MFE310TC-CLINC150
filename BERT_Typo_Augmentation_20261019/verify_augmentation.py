"""Independent artifact verifier for paired clean/typo continuation training.

No runner, model/scoring implementation, or report is imported. Generic checks
come from the preceding independently implemented verifier, whose source hash
is included in this verifier's result. No training or inference is performed.
Only this study's outputs/verification.json is written after completion.
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
PROJECT = ROOT.parent / "CLINC150_Team_20261019"
PREVIOUS = ROOT.parent / "BERT_Rejection_Study_20261019"
SEEDS = (42, 43, 44)
ARMS = ("original", "clean_continue", "typo_mix")
TRAINED_ARMS = ARMS[1:]
CONDITIONS = ("clean", "typo_light", "typo_medium")
SCORES = ("msp", "entropy", "mahalanobis")
POLICIES = ("max_oos_f1", "is_acceptance_95")
AUGMENTATION_SEED = 31020261019


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


def load_independent_helpers():
    path = PREVIOUS / "verify_rejection.py"
    spec = importlib.util.spec_from_file_location("independent_rejection_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, {path.resolve().as_posix(): digest(path)}


def verify_hash_map(mapping, base, context, *, absolute=False):
    require(isinstance(mapping, dict) and mapping, f"{context}: empty hash map")
    for name, expected in mapping.items():
        path = Path(name) if absolute else base / name
        require(path.is_file(), f"{context}: missing {path}")
        if absolute:
            require(path.is_absolute(), f"{context}: path must be absolute")
        else:
            require(path.resolve().is_relative_to(base.resolve()), f"{context}: path escapes root")
        require(digest(path) == expected, f"{context}: changed {path}")
    return len(mapping)


def verify_training_table(table, original_rows, mapping, arm, policy_column,
                          *, row_id_column="row_id", text_column="text"):
    """Check one complete epoch's source ownership, labels and mixture quotas."""
    require(len(table) == 15000, f"{arm}: expected 15000 training rows")
    ids = table[row_id_column].to_numpy(dtype=np.int64)
    require(np.array_equal(np.sort(ids), np.arange(15000)),
            f"{arm}: every original training row must occur exactly once")
    source = [original_rows[int(i)] for i in ids]
    require(table.original_text.tolist() == [r[0] for r in source], f"{arm}: source train text mismatch")
    require(table.true_label.tolist() == [r[1] for r in source], f"{arm}: source train labels mismatch")
    expected_ids = np.array([mapping[r[1]] for r in source], dtype=np.int64)
    require(np.array_equal(table.label_id.to_numpy(), expected_ids), f"{arm}: label map")
    require(np.array_equal(np.bincount(expected_ids, minlength=150), np.full(150, 100)),
            f"{arm}: class balance")
    if "source_split" in table:
        require(table.source_split.eq("train").all(), f"{arm}: non-training source rows")
    policies = table[policy_column].astype(str).to_numpy()
    if arm == "clean_continue":
        require(np.all(policies == "clean"), "Clean continuation includes perturbation policies")
    else:
        require(set(policies) == set(CONDITIONS), "Unexpected training augmentation policies")
        for c in range(150):
            selected = policies[expected_ids == c]
            require({p: int(np.sum(selected == p)) for p in CONDITIONS} ==
                    {"clean": 50, "typo_light": 25, "typo_medium": 25},
                    f"Class {c}: expected 50/25/25 policy assignment")
    changed = table[text_column].to_numpy() != table.original_text.to_numpy()
    require(not changed[policies == "clean"].any(), "Clean-assigned training text was changed")
    if "changed" in table:
        declared = table.changed.astype(str).str.lower()
        require(declared.isin(("true", "false", "1", "0")).all(), "Invalid changed flags")
        require(np.array_equal(declared.isin(("true", "1")).to_numpy(), changed),
                "Actual training text changes differ from declared flags")
    if "modified_tokens" in table:
        count = table.modified_tokens.to_numpy(dtype=np.int64)
        require(np.all(count >= 0), "Negative modified-token count")
        require(np.all(count[policies == "clean"] == 0), "Clean policy modified tokens")
        require(np.all(count[policies == "typo_light"] <= 1), "Light policy modified more than one token")
        require(np.all(count[policies == "typo_medium"] <= 2), "Medium policy modified more than two tokens")
    return {"arm": arm, "training_rows": len(table), "policy_counts":
            {p: int(np.sum(policies == p)) for p in CONDITIONS},
            "actual_changed_rows": int(changed.sum())}, ids


def verify_training_order(row_ids, batch_sizes, context):
    order = np.asarray(row_ids, dtype=np.int64)
    batches = np.asarray(batch_sizes, dtype=np.int64)
    require(np.array_equal(np.sort(order), np.arange(15000)), f"{context}: row order is not one full epoch")
    require(len(batches) == 938 and np.array_equal(batches, np.r_[np.full(937, 16), 8]),
            f"{context}: expected 938 optimizer steps with final batch 8")
    require(int(batches.sum()) == len(order), f"{context}: batch sizes do not cover all rows")
    return order


def verify_prediction_frame(frame, rows, mapping, helpers, context, *, condition=None):
    n = len(rows)
    require(len(frame) == n and np.array_equal(frame.row_id.to_numpy(), np.arange(n)),
            f"{context}: row count/order")
    labels = np.array([mapping.get(row[1], -1) for row in rows], dtype=np.int64)
    require(frame.true_label.tolist() == [r[1] for r in rows], f"{context}: inherited true labels")
    require(np.array_equal(frame.label_id.to_numpy(), labels), f"{context}: label IDs")
    oos = helpers.booleans(frame.is_oos, context)
    require(np.array_equal(oos, labels == -1), f"{context}: OOS flags")
    if condition is None:
        require(frame.text.tolist() == [r[0] for r in rows], f"{context}: clean validation text")
    else:
        source = read_csv(PROJECT / f"outputs/robustness/typo_data/{condition}.csv")
        require(frame.text.tolist() == source.text.tolist(), f"{context}: frozen test text")
        if "original_text" in frame:
            require(frame.original_text.tolist() == [r[0] for r in rows], f"{context}: original test text")
        for column in ("changed", "modified_tokens"):
            if column in frame:
                require(frame[column].tolist() == source[column].tolist(), f"{context}: test perturbation metadata/{column}")
    return labels, oos


def verify_scoring_arrays(frame, logits, features, model, helpers, context):
    n = len(frame)
    require(logits.shape == (n, 150) and logits.dtype == np.float32, f"{context}: FP32 logits")
    require(features.shape == (n, 768) and features.dtype == np.float32, f"{context}: FP32 CLS")
    require(np.isfinite(logits).all() and np.isfinite(features).all(), f"{context}: finite arrays")
    prediction = np.asarray(logits).argmax(axis=1)
    require(np.array_equal(frame.predicted_label_id.to_numpy(), prediction), f"{context}: shared classifier head")
    z = np.asarray(logits, dtype=np.float64)
    exp = np.exp(z - z.max(axis=1, keepdims=True))
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    msp = 1 - probabilities.max(axis=1)
    entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
    for score, values in (("msp", msp), ("entropy", entropy)):
        helpers.array_close(frame[f"{score}_score"].to_numpy(dtype=float), values,
                            f"{context}: independent {score}", atol=1e-13, rtol=1e-12)
    # Select a fixed six rows spanning IS and OOS; solve against covariance,
    # independently of the production precision-matrix quadratic expansion.
    oos = helpers.booleans(frame.is_oos, context)
    rng = np.random.default_rng(20261019)
    sample = np.r_[rng.choice(np.flatnonzero(~oos), 3, replace=False),
                   rng.choice(np.flatnonzero(oos), 3, replace=False)]
    differences = np.asarray(features[sample], dtype=np.float64)[:, None] - model["class_means"][None]
    flattened = differences.reshape(-1, 768)
    solution = np.linalg.solve(model["covariance"], flattened.T).T
    distances = np.sum(flattened * solution, axis=1).reshape(6, 150)
    minimum = distances.min(axis=1)
    stored = frame.mahalanobis_score.to_numpy(dtype=float)[sample]
    helpers.array_close(stored, minimum, f"{context}: direct distance solve", atol=1e-7, rtol=1e-8)
    require(np.array_equal(frame.nearest_centroid_label_id_diagnostic_only.to_numpy()[sample],
                           model["classes"][distances.argmin(axis=1)]), f"{context}: nearest-center diagnostic")
    return prediction, {"context": context, "row_ids": sample.tolist(),
                        "max_abs_distance_error": float(np.max(np.abs(stored - minimum)))}


def verify_operating_points(frame, labels, oos, prediction, thresholds, metric_rows, helpers, context):
    """Use this model's own clean-validation thresholds, never another model's."""
    require(set(thresholds) == {f"{s}/{p}" for s in SCORES for p in POLICIES},
            f"{context}: six threshold settings")
    count = 0
    for score in SCORES:
        values = frame[f"{score}_score"].to_numpy(dtype=np.float64)
        require(np.isfinite(values).all(), f"{context}: finite {score}")
        for policy in POLICIES:
            threshold = thresholds[f"{score}/{policy}"]["threshold"]
            decisions = values >= threshold
            require(np.array_equal(helpers.booleans(frame[f"{score}_{policy}_reject"], context), decisions),
                    f"{context}/{score}/{policy}: frozen >= decision")
            row = metric_rows.loc[(score, policy)]
            require(float(row.threshold) == threshold, f"{context}: model-specific threshold changed")
            for name, value in helpers.manual_metrics(labels, prediction, oos, decisions, values).items():
                helpers.close(value, row[name], f"{context}/{score}/{policy}/{name}")
            support = np.bincount(labels[~oos], minlength=150)
            predicted = np.bincount(prediction[~oos], minlength=150)
            correct = np.bincount(labels[(prediction == labels) & ~oos], minlength=150)
            class_f1 = np.divide(2 * correct, support + predicted,
                                 out=np.zeros(150), where=support + predicted > 0)
            if "closed_set_accuracy" in row:
                helpers.close(np.mean(prediction[~oos] == labels[~oos]), row.closed_set_accuracy,
                              f"{context}/{score}/{policy}: closed accuracy")
            if "closed_set_macro_f1" in row:
                helpers.close(class_f1.mean(), row.closed_set_macro_f1,
                              f"{context}/{score}/{policy}: closed macro F1")
            count += 1
    return count


def require_inventory(paths, inventory, context):
    for path in paths:
        require(path.resolve().as_posix() in inventory, f"{context}: missing frozen dependency {path}")


def timestamp(value):
    return datetime.fromisoformat(value)


def replay_training_perturbation(text, source_row, policy):
    """Replay the published character operation without importing its generator."""
    spans = list(re.finditer(r"(?<![A-Za-z0-9])[A-Za-z]{5,}(?![A-Za-z0-9])", text))
    seed_bytes = hashlib.sha256(f"{AUGMENTATION_SEED}\0{source_row}\0{text}".encode()).digest()
    rng = random.Random(int.from_bytes(seed_bytes, "big"))
    rng.shuffle(spans)
    mutations = []
    # Both prospective mutations are generated before selecting light/medium.
    for span in spans[:2]:
        token = span.group()
        possible_swaps = [k for k in range(1, len(token) - 2) if token[k] != token[k + 1]]
        if possible_swaps and rng.randrange(2) == 0:
            index = rng.choice(possible_swaps)
            replacement = token[:index] + token[index + 1] + token[index] + token[index + 2:]
            operation = "swap"
        else:
            index = rng.randrange(1, len(token) - 1)
            replacement = token[:index] + token[index + 1:]
            operation = "delete"
        mutations.append(dict(start=span.start(), end=span.end(), original=token,
                              replacement=replacement, operation=operation, character_index=index))
    selected = mutations[:CONDITIONS.index(policy)]
    changed = text
    for mutation in sorted(selected, key=lambda x: x["start"], reverse=True):
        changed = changed[:mutation["start"]] + mutation["replacement"] + changed[mutation["end"]:]
    return changed, len(spans), selected


def verify_training_data(raw, mapping, helpers):
    tables = {"clean_continue": read_csv(ROOT / "data/train_clean.csv"),
              "typo_mix": read_csv(ROOT / "data/train_mix.csv")}
    summaries = []
    for arm, table in tables.items():
        summary, ids = verify_training_table(table, raw["train"], mapping, arm, "policy", row_id_column="source_row")
        require(np.array_equal(ids, np.arange(15000)), f"{arm}: canonical training table order")
        expected_policy = np.full(15000, "clean", dtype=object)
        if arm == "typo_mix":
            for label_id in range(150):
                indices = np.flatnonzero(table.label_id.to_numpy() == label_id)
                allocation = np.random.default_rng(AUGMENTATION_SEED + label_id).permutation(indices)
                expected_policy[allocation[50:75]] = "typo_light"
                expected_policy[allocation[75:]] = "typo_medium"
        require(np.array_equal(table.policy.to_numpy(), expected_policy), f"{arm}: augmentation-seed allocation")
        for row in table.itertuples(index=False):
            text, eligible, mutations = replay_training_perturbation(row.original_text, int(row.source_row), row.policy)
            require(row.text == text, f"{arm}/{row.source_row}: deterministic training mutation")
            require(row.eligible_tokens == eligible and row.modified_tokens == len(mutations),
                    f"{arm}/{row.source_row}: mutation counts")
            require(json.loads(row.mutations_json) == mutations, f"{arm}/{row.source_row}: mutation trace")
        summaries.append(summary)
    mixed = tables["typo_mix"]
    summary = read_json(ROOT / "data/data_summary.json")
    require(summary["train_rows"] == 15000 and summary["augmentation_seed"] == AUGMENTATION_SEED,
            "Training data summary dimensions/seed")
    require(summary["policy_counts"] == mixed.policy.value_counts().to_dict(), "Training policy summary")
    require(summary["changed_rows"] == int(helpers.booleans(mixed.changed, "train mixture").sum()), "Training changed summary")
    require(summary["modified_token_counts"] == {str(k): int(v) for k, v in mixed.modified_tokens.value_counts().items()},
            "Training token-count summary")
    require(summary["train_mix_sha256"] == digest(ROOT / "data/train_mix.csv"), "Training mixture hash")
    coverage = read_csv(ROOT / "data/augmentation_coverage.csv").set_index("policy")
    require(set(coverage.index) == set(CONDITIONS), "Augmentation coverage policies")
    for policy, group in mixed.groupby("policy"):
        for field, value in {"rows": len(group), "changed": int(helpers.booleans(group.changed, policy).sum()),
                             "modified_tokens": int(group.modified_tokens.sum()),
                             "two_tokens_changed": int((group.modified_tokens == 2).sum())}.items():
            helpers.close(coverage.loc[policy, field], value, f"Coverage/{policy}/{field}")
    return summaries, tables


def verify_freezes_and_training(output, helpers):
    """Verify the frozen dependency graph and the six complete training ledgers."""
    protocol = read_json(ROOT / "PROTOCOL.json")
    require(protocol["seeds"] == list(SEEDS) and protocol["arms"] == list(TRAINED_ARMS)
            and protocol["conditions"] == list(CONDITIONS), "Protocol experiment grid")
    expected_train = {"epochs": 1, "examples": 15000, "batch_size": 16, "optimizer_updates": 938,
                      "last_batch_size": 8, "learning_rate": 1e-5, "optimizer": "AdamW",
                      "weight_decay": .01, "warmup_ratio": .1, "gradient_clip_norm": 1.}
    for key, value in expected_train.items():
        require(protocol["train"][key] == value, f"Protocol training/{key}")
    require(protocol["augmentation"]["seed"] == AUGMENTATION_SEED, "Protocol augmentation seed")
    require(protocol["augmentation"]["per_class_policy_counts"] == {"clean": 50, "typo_light": 25, "typo_medium": 25},
            "Protocol mixture quotas")
    require(protocol["augmentation"]["same_frozen_data_across_model_seeds"] is True, "Mixture varies with model seed")
    require(protocol["inference"] == read_json(PREVIOUS / "PROTOCOL.json")["inference"], "FP32 inference protocol differs")
    frozen_path, development_path = ROOT / "INPUTS_FROZEN.json", output / "DEVELOPMENT_FROZEN.json"
    frozen, development, complete = read_json(frozen_path), read_json(development_path), read_json(output / "TEST_COMPLETE.json")
    require(set(frozen) == {"input_hashes", "runtime", "frozen_at"}, "INPUTS_FROZEN schema")
    require(frozen["runtime"] == read_json(PREVIOUS / "outputs/environment.json"), "Frozen runtime differs from original")
    input_hash, development_hash = digest(frozen_path), digest(development_path)
    require(development["inputs_frozen_sha256"] == complete["inputs_frozen_sha256"] == input_hash, "Frozen-input bindings")
    require(complete["development_frozen_sha256"] == development_hash, "Frozen-development binding")
    require(development["models"] == 6 and development["validation_thresholds"] == 36, "Development totals")
    require(complete["evaluated_new_settings"] == 108 and complete["total_test_settings"] == 162, "Test totals")
    require(complete["original_inputs_and_new_development_unchanged"] is True, "Test immutability flag")
    require(timestamp(frozen["frozen_at"]) <= timestamp(development["frozen_at"]) <= timestamp(complete["completed_at"]),
            "Freeze/completion chronological order")
    counts = {"frozen_inputs": verify_hash_map(frozen["input_hashes"], ROOT, "inputs", absolute=True),
              "development_outputs": verify_hash_map(development["development_hashes"], output, "development", absolute=True),
              "test_outputs": verify_hash_map(complete["output_hashes"], output, "test", absolute=True)}
    prior_freeze = read_json(PREVIOUS / "outputs/DEVELOPMENT_FROZEN.json")
    original_paths = []
    for category, base in (("source_hashes", PROJECT), ("code_hashes", PREVIOUS),
                           ("development_hashes", PREVIOUS / "outputs")):
        counts[f"original_{category}"] = verify_hash_map(prior_freeze[category], base, category)
        original_paths += [base / name for name in prior_freeze[category]]
    previous_typo = PREVIOUS / "typo_stress_test/outputs"
    prior_complete = read_json(previous_typo / "TEST_COMPLETE.json")
    require(prior_complete["inputs_frozen_sha256"] == digest(PREVIOUS / "typo_stress_test/INPUTS_FROZEN.json"),
            "Previous typo input freeze changed")
    counts["original_typo_outputs"] = verify_hash_map(prior_complete["output_hashes"], previous_typo, "original typo outputs")
    mandatory_inputs = original_paths + [ROOT / "PROTOCOL.json", ROOT / "run_augmentation.py",
        PROJECT / "data/raw/data_full.json", PROJECT / "src/typo_data.py", PROJECT / "src/train_bert.py", PROJECT / "src/common.py",
        PREVIOUS / "outputs/DEVELOPMENT_FROZEN.json", PREVIOUS / "outputs/environment.json",
        previous_typo / "TEST_COMPLETE.json", previous_typo / "metrics_by_seed.csv"]
    mandatory_inputs += [ROOT / "data" / name for name in ("train_clean.csv", "train_mix.csv", "augmentation_coverage.csv", "data_summary.json")]
    mandatory_inputs += [PROJECT / f"outputs/robustness/typo_data/{condition}.csv" for condition in CONDITIONS]
    mandatory_inputs += [previous_typo / f"seed{seed}/{condition}_predictions.csv" for seed in SEEDS for condition in CONDITIONS]
    require_inventory(mandatory_inputs, frozen["input_hashes"], "Input inventory")
    require_inventory([output / "validation_metrics_by_seed.csv"], development["development_hashes"], "Development inventory")
    require_inventory([output / "metrics_by_seed.csv"], complete["output_hashes"], "Test inventory")
    raw = read_json(PROJECT / "data/raw/data_full.json")
    mapping = read_json(PROJECT / "outputs/bert/seed42/label_mapping.json")["label_to_id"]
    require(len(raw["train"]) == 15000 and sorted(mapping.values()) == list(range(150)), "Original train/map dimensions")
    summaries, tables = verify_training_data(raw, mapping, helpers)
    training_summary, orders = [], {}
    for seed in SEEDS:
        require(read_json(PROJECT / f"outputs/bert/seed{seed}/label_mapping.json")["label_to_id"] == mapping,
                f"Seed {seed}: label map changed")
        for arm in TRAINED_ARMS:
            folder = output / arm / f"seed{seed}"
            training = read_json(folder / "TRAINING_COMPLETE.json")
            done = read_json(folder / "DEVELOPMENT_COMPLETE.json")
            context = f"{arm}/seed{seed}"
            require(training["arm"] == arm and training["seed"] == seed, f"{context}: training identity")
            require(training["epochs"] == 1 and training["presentations"] == 15000 and training["optimizer_updates"] == 938,
                    f"{context}: training exposure")
            require(training["inputs_frozen_sha256"] == done["inputs_frozen_sha256"] == input_hash, f"{context}: input bindings")
            require(training["train_precision"] == "bfloat16 autocast, float32 parameters", f"{context}: BF16 training record")
            require(timestamp(frozen["frozen_at"]) <= timestamp(training["completed_at"]) <= timestamp(done["completed_at"])
                    <= timestamp(development["frozen_at"]), f"{context}: development chronology")
            verify_hash_map(training["output_hashes"], folder, f"{context} training", absolute=True)
            verify_hash_map(done["output_hashes"], folder, f"{context} development", absolute=True)
            training_paths = [folder / "training_order.csv", folder / "training_steps.csv"] + list((folder / "final_model").glob("*"))
            require((folder / "final_model/model.safetensors").is_file(), f"{context}: missing final model")
            require_inventory(training_paths, training["output_hashes"], f"{context}: training inventory")
            require(set(training["output_hashes"]).issubset(done["output_hashes"]), f"{context}: development omits training outputs")
            required_development = [folder / name for name in ("TRAINING_COMPLETE.json", "DEVELOPMENT_COMPLETE.json", "distance_model.npz", "frozen_thresholds.json")]
            required_development += [folder / f"{name}_{suffix}" for name in ("train", "validation")
                                     for suffix in ("cls.npy", "logits_fp32.npy", "metadata.csv", "extraction.json")]
            required_development += [folder / f"validation_{s}_threshold_curve.csv" for s in SCORES]
            required_development += [folder / "validation_predictions.csv", folder / "validation_metrics.csv"]
            require_inventory(required_development + training_paths, development["development_hashes"], f"{context}: global development inventory")
            require(set(done["output_hashes"]).issubset(development["development_hashes"]), f"{context}: global freeze omits per-run development")
            order = read_csv(folder / "training_order.csv")
            steps = read_csv(folder / "training_steps.csv")
            require(np.array_equal(order.presentation_index.to_numpy(), np.arange(15000)), f"{context}: presentation indices")
            actual_order = verify_training_order(order.source_row, steps.batch_size, context)
            require(np.array_equal(actual_order, np.random.default_rng(seed).permutation(15000)), f"{context}: seed row permutation")
            require(np.array_equal(steps.step.to_numpy(), np.arange(1, 939)), f"{context}: optimizer step numbers")
            require(np.array_equal(steps.samples_seen.to_numpy(), steps.batch_size.cumsum().to_numpy()), f"{context}: presentations ledger")
            scheduler_step = np.arange(938)
            expected_lr = 1e-5 * np.where(scheduler_step < 94, scheduler_step / 94, (938 - scheduler_step) / 844)
            helpers.array_close(steps.learning_rate.to_numpy(), expected_lr, f"{context}: LR schedule", atol=1e-18, rtol=1e-12)
            require(np.isfinite(steps[["loss", "grad_norm"]].to_numpy()).all()
                    and (steps[["loss", "grad_norm"]].to_numpy() >= 0).all(), f"{context}: finite loss/preclip gradient norm")
            helpers.close(steps.loss.mean(), training["mean_batch_loss"], f"{context}: mean batch loss", atol=1e-12, rtol=1e-12)
            require(training["elapsed_seconds"] > 0 and training["peak_allocated_gpu_bytes"] > 0, f"{context}: training resource records")
            require(done["class_counts"] == [100] * 150 and 0 <= done["shrinkage"] <= 1, f"{context}: clean train detector record")
            original_weight = PROJECT / f"outputs/bert/seed{seed}/best_model/model.safetensors"
            require_inventory([original_weight], frozen["input_hashes"], "Initialization checkpoint")
            require(digest(folder / "final_model/model.safetensors") != digest(original_weight), f"{context}: final model identical to initialization")
            orders[(arm, seed)] = actual_order
            training_summary.append({"arm": arm, "seed": seed, "presentations": 15000, "optimizer_updates": 938,
                "warmup_updates": 94, "mean_batch_loss": training["mean_batch_loss"], "training_complete_sha256": digest(folder / "TRAINING_COMPLETE.json")})
        require(np.array_equal(orders[("clean_continue", seed)], orders[("typo_mix", seed)]), f"Seed {seed}: paired arms have different order")
    return dict(raw=raw, mapping=mapping, frozen=frozen, development=development, complete=complete,
                input_hash=input_hash, development_hash=development_hash, hash_counts=counts,
                training_summary=training_summary, training_data_summary=summaries)


def verify_new_extraction(folder, split, frame, state, helpers):
    record = read_json(folder / f"{split}_extraction.json")
    require(record["training_complete_sha256"] == digest(folder / "TRAINING_COMPLETE.json"), f"{folder}/{split}: checkpoint binding")
    require(record["inputs_frozen_sha256"] == state["input_hash"], f"{folder}/{split}: inputs binding")
    text_hash = hashlib.sha256(json.dumps(frame.text.tolist(), ensure_ascii=False).encode()).hexdigest()
    require(record["text_sha256"] == text_hash and record["rows"] == len(frame), f"{folder}/{split}: text/row binding")
    metadata = read_csv(folder / f"{split}_metadata.csv")
    for column in metadata:
        require(column in frame and metadata[column].tolist() == frame[column].tolist(), f"{folder}/{split}: metadata/{column}")
    feature_path, logit_path = folder / f"{split}_cls.npy", folder / f"{split}_logits_fp32.npy"
    require(record["features_sha256"] == digest(feature_path) and record["logits_sha256"] == digest(logit_path), f"{folder}/{split}: array bindings")
    completion = timestamp(record["completed_at"])
    require(completion >= timestamp(read_json(folder / "TRAINING_COMPLETE.json")["completed_at"]), f"{folder}/{split}: extraction predates training")
    if split in CONDITIONS:
        require(timestamp(state["development"]["frozen_at"]) <= completion <= timestamp(state["complete"]["completed_at"]),
                f"{folder}/{split}: test completion outside frozen test stage")
        require_inventory([folder / f"{split}_{suffix}" for suffix in
            ("cls.npy", "logits_fp32.npy", "metadata.csv", "extraction.json", "predictions.csv", "metrics.csv")],
            state["complete"]["output_hashes"], f"{folder}/{split}: test inventory")
    else:
        require(completion <= timestamp(read_json(folder / "DEVELOPMENT_COMPLETE.json")["completed_at"]), f"{folder}/{split}: extraction after development completion")
    features = np.load(feature_path, allow_pickle=False, mmap_mode="r")
    logits = np.load(logit_path, allow_pickle=False, mmap_mode="r")
    require(features.shape == (len(frame), 768) and features.dtype == np.float32, f"{folder}/{split}: CLS dimensions")
    require(logits.shape == (len(frame), 150) and logits.dtype == np.float32, f"{folder}/{split}: logits dimensions")
    require(np.isfinite(features).all() and np.isfinite(logits).all(), f"{folder}/{split}: nonfinite arrays")
    return features, logits


def keyed_metrics(table, arms, conditions, context):
    keys = ["arm", "seed", "condition", "score", "policy"]
    expected = {(a, s, c, m, p) for a in arms for s in SEEDS for c in conditions for m in SCORES for p in POLICIES}
    require(not table.duplicated(keys).any() and set(table[keys].itertuples(index=False, name=None)) == expected,
            f"{context}: metric grid does not match {len(expected)} settings")
    return table.set_index(keys)


def compare_metric_table(local, aggregate, helpers, context):
    require(not local.duplicated(["score", "policy"]).any() and len(local) == 6, f"{context}: per-condition metric table")
    for row in local.to_dict("records"):
        recorded = aggregate.loc[(row["score"], row["policy"])]
        for field, value in row.items():
            if field not in ("arm", "seed", "condition", "score", "policy"):
                helpers.close(value, recorded[field], f"{context}/{field}")


def verify_optional_transitions(output, frames, helpers):
    path = output / "normal_transitions.csv"
    if not path.is_file():
        return {"status": "NOT_PRESENT", "checked": False}
    table = keyed_metrics(read_csv(path), ARMS, CONDITIONS, "Normal transitions")
    state_names = ("accepted_correct", "accepted_wrong", "rejected_correct", "rejected_wrong")
    for (arm, seed, condition), frame in frames.items():
        clean = frames[(arm, seed, "clean")]
        normal = frame.label_id.to_numpy() >= 0
        correct = (frame.predicted_label_id.to_numpy() == frame.label_id.to_numpy())[normal]
        clean_correct = (clean.predicted_label_id.to_numpy() == clean.label_id.to_numpy())[normal]
        changed = helpers.booleans(frame.changed, "Transition changed")[normal]
        n = int(normal.sum())
        require(n == 4500, "Transition IS size")
        for score in SCORES:
            for policy in POLICIES:
                rejected = helpers.booleans(frame[f"{score}_{policy}_reject"], "Transition reject")[normal]
                clean_rejected = helpers.booleans(clean[f"{score}_{policy}_reject"], "Transition clean reject")[normal]
                clean_ac = clean_correct & ~clean_rejected
                states = {"accepted_correct": correct & ~rejected, "accepted_wrong": ~correct & ~rejected,
                          "rejected_correct": correct & rejected, "rejected_wrong": ~correct & rejected}
                require(np.all(np.stack(list(states.values())).sum(axis=0) == 1), "Four states fail to partition IS")
                clean_accepted_n, clean_rejected_n = int((~clean_rejected).sum()), int(clean_rejected.sum())
                new, recovered = int((~clean_rejected & rejected).sum()), int((clean_rejected & ~rejected).sum())
                require(int(rejected.sum()) - clean_rejected_n == new - recovered, "Net rejection identity")
                ratio = lambda x, y: x / y if y else None
                expected = dict(is_n=n, changed_is_n=int(changed.sum()), clean_accepted_n=clean_accepted_n,
                    clean_accepted_correct_n=int(clean_ac.sum()), clean_rejected_n=clean_rejected_n,
                    closed_accuracy=float(correct.mean()), is_rejection_rate=float(rejected.mean()),
                    rejection_rate_delta_vs_clean=float(rejected.mean() - clean_rejected.mean()),
                    new_rejection_n=new, new_rejection_rate_over_all_is=new / n,
                    new_rejection_rate_over_clean_accepted=ratio(new, clean_accepted_n),
                    recovered_acceptance_n=recovered, recovered_acceptance_rate_over_all_is=recovered / n,
                    recovered_acceptance_rate_over_clean_rejected=ratio(recovered, clean_rejected_n),
                    net_rejection_change_n=new - recovered, changed_only_rejection_n=int((changed & rejected).sum()),
                    changed_only_rejection_rate=ratio(int((changed & rejected).sum()), int(changed.sum())))
                for name in state_names:
                    count, flow = int(states[name].sum()), int((clean_ac & states[name]).sum())
                    expected.update({f"{name}_n": count, f"{name}_rate_over_all_is": count / n,
                        f"clean_ac_to_{name}_n": flow, f"clean_ac_to_{name}_rate_over_all_is": flow / n,
                        f"clean_ac_to_{name}_rate_over_clean_accepted_correct": ratio(flow, int(clean_ac.sum()))})
                recorded = table.loc[(arm, seed, condition, score, policy)]
                for field, value in expected.items():
                    helpers.close(value, recorded[field], f"Transition/{arm}/{seed}/{condition}/{score}/{policy}/{field}")
    return {"status": "PASS", "checked": True, "verified_rows": len(table), "sha256": digest(path),
            "four_current_condition_states_and_clean_ac_flows_verified": True,
            "new_minus_recovered_equals_net_rejection_change": True}


def verify_results(output, helpers, state):
    """Independently check 36 clean-validation and 162 frozen test settings."""
    metrics = keyed_metrics(read_csv(output / "metrics_by_seed.csv"), ARMS, CONDITIONS, "Test")
    validation = keyed_metrics(read_csv(output / "validation_metrics_by_seed.csv"), TRAINED_ARMS, ("validation",), "Validation")
    raw, mapping = state["raw"], state["mapping"]
    inverse = {v: k for k, v in mapping.items()}
    val_rows, test_rows = raw["val"] + raw["oos_val"], raw["test"] + raw["oos_test"]
    train_labels = np.array([mapping[y] for _, y in raw["train"]], dtype=np.int64)
    checked_validation = checked_test = candidate_count = 0
    distance_checks, frames = [], {}
    previous_typo = PREVIOUS / "typo_stress_test/outputs"
    original_metrics = read_csv(previous_typo / "metrics_by_seed.csv")
    original_metrics.insert(0, "arm", "original")
    historical_index = keyed_metrics(original_metrics, ("original",), CONDITIONS, "Historical test")
    for key, row in historical_index.iterrows():
        for field, value in row.items():
            helpers.close(value, metrics.loc[key, field], f"Historical exact reuse/{key}/{field}", atol=0, rtol=0)
    for arm in ARMS:
        for seed in SEEDS:
            folder = PREVIOUS / f"outputs/seed{seed}" if arm == "original" else output / arm / f"seed{seed}"
            thresholds = read_json(folder / "frozen_thresholds.json")
            if arm == "original":
                training_features = np.load(folder / "train_cls.npy", allow_pickle=False, mmap_mode="r")
            else:
                train_frame = read_csv(folder / "train_metadata.csv")
                helpers.verify_metadata(train_frame, raw["train"], ["train"] * 15000, mapping, f"{arm}/{seed}: clean detector train")
                training_features, _ = verify_new_extraction(folder, "train", train_frame, state, helpers)
            model = helpers.verify_distance_statistics(folder, training_features, train_labels)
            del training_features
            if arm != "original":
                helpers.close(float(model["shrinkage"]), read_json(folder / "DEVELOPMENT_COMPLETE.json")["shrinkage"], f"{arm}/{seed}: shrinkage record")
                frame = read_csv(folder / "validation_predictions.csv")
                labels, oos = verify_prediction_frame(frame, val_rows, mapping, helpers, f"{arm}/{seed}/validation")
                require(int(oos.sum()) == 100 and np.array_equal(np.bincount(labels[~oos], minlength=150), np.full(150, 20)), "Validation class counts")
                features, logits = verify_new_extraction(folder, "validation", frame, state, helpers)
                prediction, distance_record = verify_scoring_arrays(frame, logits, features, model, helpers, f"{arm}/{seed}/validation")
                require(frame.predicted_label.tolist() == [inverse[int(k)] for k in prediction], "Validation decoded labels")
                distance_checks.append(distance_record)
                for score in SCORES:
                    curve = read_csv(folder / f"validation_{score}_threshold_curve.csv")
                    candidate_count += helpers.verify_validation_thresholds(frame[f"{score}_score"].to_numpy(dtype=float), oos,
                        {p: thresholds[f"{score}/{p}"] for p in POLICIES}, curve, f"{arm}/{seed}/{score}")
                selected = validation.loc[(arm, seed, "validation")]
                checked_validation += verify_operating_points(frame, labels, oos, prediction, thresholds, selected, helpers, f"{arm}/{seed}/validation")
                compare_metric_table(read_csv(folder / "validation_metrics.csv"), selected, helpers, f"{arm}/{seed}/validation local metrics")
                del features, logits
            for condition in CONDITIONS:
                context = f"{arm}/{seed}/{condition}"
                frame_path = (previous_typo / f"seed{seed}" if arm == "original" else folder) / f"{condition}_predictions.csv"
                frame = read_csv(frame_path)
                labels, oos = verify_prediction_frame(frame, test_rows, mapping, helpers, context, condition=condition)
                require(int(oos.sum()) == 1000 and np.array_equal(np.bincount(labels[~oos], minlength=150), np.full(150, 30)), f"{context}: test class counts")
                require(np.array_equal(helpers.booleans(frame.changed, context), frame.text.to_numpy() != frame.original_text.to_numpy()), f"{context}: actual changed text")
                if arm == "original":
                    array_folder = folder if condition == "clean" else previous_typo / f"seed{seed}"
                    prefix = "test" if condition == "clean" else condition
                    features = np.load(array_folder / f"{prefix}_cls.npy", allow_pickle=False, mmap_mode="r")
                    logits = np.load(array_folder / f"{prefix}_logits_fp32.npy", allow_pickle=False, mmap_mode="r")
                else:
                    features, logits = verify_new_extraction(folder, condition, frame, state, helpers)
                prediction, distance_record = verify_scoring_arrays(frame, logits, features, model, helpers, context)
                require(frame.predicted_label.tolist() == [inverse[int(k)] for k in prediction], f"{context}: decoded labels")
                distance_checks.append(distance_record)
                selected = metrics.loc[(arm, seed, condition)]
                checked_test += verify_operating_points(frame, labels, oos, prediction, thresholds, selected, helpers, context)
                if arm != "original":
                    compare_metric_table(read_csv(folder / f"{condition}_metrics.csv"), selected, helpers, f"{context}: local metrics")
                frames[(arm, seed, condition)] = frame
                del features, logits
            del model
    transitions = verify_optional_transitions(output, frames, helpers)
    require(digest(ROOT / "INPUTS_FROZEN.json") == state["input_hash"] and digest(output / "DEVELOPMENT_FROZEN.json") == state["development_hash"],
            "Frozen manifests changed during verification")
    # Recheck all experiment hashes to detect writes during the independent audit.
    verify_hash_map(state["frozen"]["input_hashes"], ROOT, "ending inputs", absolute=True)
    verify_hash_map(state["development"]["development_hashes"], output, "ending development", absolute=True)
    verify_hash_map(state["complete"]["output_hashes"], output, "ending test", absolute=True)
    return dict(verified_test_settings=checked_test, verified_validation_settings=checked_validation,
        verified_test_prediction_rows=27 * 5500, validation_candidates_checked=candidate_count,
        validation_operating_points=36, all_validation_thresholds_and_tie_rules_verified=True,
        model_specific_thresholds_frozen_before_new_model_test=True,
        training_data_source_and_deterministic_mutations_verified=True,
        paired_training_order_and_938_update_ledgers_verified=True,
        training_data_summary=state["training_data_summary"], training_runs=state["training_summary"],
        clean_original_train_only_distance_statistics_verified=True,
        original_json_row_alignment=True, shared_classifier_predictions_verified=True,
        independent_manual_confusion_metrics=True, independent_auc_average_precision=True,
        distance_direct_solve_checks=distance_checks, normal_transitions=transitions,
        hash_inventory_counts=state["hash_counts"], frozen_manifest_and_source_hashes_verified=True,
        inputs_manifest_sha256=state["input_hash"], development_manifest_sha256=state["development_hash"],
        test_completion_sha256=digest(output / "TEST_COMPLETE.json"), metrics_sha256=digest(output / "metrics_by_seed.csv"),
        limitations=["Verifies frozen training ledgers, model/data bindings and saved FP32 outputs; does not rerun training or BERT inference.",
          "Training source review and frozen ledgers support the prescribed optimizer/precision setup; ledgers alone cannot prove every GPU operation.",
          "Reconstructs covariance using each recorded shrinkage coefficient; does not independently re-estimate that coefficient.",
          "Direct covariance solves sample six rows per model/split or condition, all 150 centers for each sampled row.",
          "Test-extraction completion times and frozen runner gate are checked; records do not separately timestamp extraction start."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    output = args.output.resolve()
    required = [ROOT / "PROTOCOL.json", ROOT / "INPUTS_FROZEN.json",
                output / "DEVELOPMENT_FROZEN.json", output / "TEST_COMPLETE.json",
                output / "metrics_by_seed.csv"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print(json.dumps({"status": "NOT_READY", "missing": missing}, ensure_ascii=False, indent=2))
        return 2
    started = datetime.now(timezone.utc).isoformat()
    try:
        helpers, helper_hashes = load_independent_helpers()
        context = threadpool_limits(limits=4) if threadpool_limits else nullcontext()
        with context:
            state = verify_freezes_and_training(output, helpers)
            result = verify_results(output, helpers, state)
        result.update(status="PASS", independent_helper_sha256=helper_hashes,
                      verifier_sha256=digest(Path(__file__)))
    except Exception as error:
        result = {"status": "FAIL", "error_type": type(error).__name__, "error": str(error)}
    result.update(started_at_utc=started, completed_at_utc=datetime.now(timezone.utc).isoformat())
    (output / "verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
