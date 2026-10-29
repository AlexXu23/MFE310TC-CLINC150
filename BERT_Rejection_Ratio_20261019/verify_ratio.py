"""Independent audit of frozen-feature detector noise-ratio experiments.

Imports only earlier independent verification modules, never a producer runner,
distance estimator, report, sklearn, or torch. Only outputs/verification.json
is written. Old experiment directories remain read-only.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "BERT_Mixed_Rejection_20261019"
AUGMENTED = ROOT.parent / "BERT_Typo_Augmentation_20261019"
PROJECT = ROOT.parent / "CLINC150_Team_20261019"
SEEDS = (42, 43, 44)
ARMS = ("clean_reference", "ratio_00", "ratio_10", "ratio_25", "ratio_50")
CONDITIONS = ("clean", "typo_light", "typo_medium")
POLICIES = ("max_oos_f1", "is_acceptance_95")
REFERENCES = {"clean_reference": "clean_only", "ratio_00": "clean_duplicate", "ratio_50": "clean_plus_typo"}
RATIOS = {"clean_reference": 0., "ratio_00": 0., "ratio_10": .1, "ratio_25": .25, "ratio_50": .5}
ALLOCATION_SEED = 31020261021


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
    path = PREVIOUS / "verify_mixed_rejection.py"
    spec = importlib.util.spec_from_file_location("independent_mixed_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base, hashes = module.helpers()
    hashes[path.resolve().as_posix()] = digest(path)
    return module, base, hashes


def index_metrics(table, conditions, context):
    keys = ["arm", "seed", "condition", "score", "policy"]
    expected = {(a, s, c, "mahalanobis", p) for a in ARMS for s in SEEDS for c in conditions for p in POLICIES}
    require(not table.duplicated(keys).any() and set(table[keys].itertuples(index=False, name=None)) == expected,
            f"{context}: expected {len(expected)} unique settings")
    return table.set_index(keys).sort_index()


def verify_data(m, h):
    raw = read_json(PROJECT / "data/raw/data_full.json")
    mapping = read_json(PROJECT / "outputs/bert/seed42/label_mapping.json")["label_to_id"]
    noise = read_csv(PREVIOUS / "data/noisy_train.csv")
    selection = read_csv(ROOT / "data/selection_manifest.csv")
    require(len(noise) == len(selection) == 15000, "Source/selection row counts")
    require(np.array_equal(selection.source_row.to_numpy(), np.arange(15000)), "Selection source row order")
    m.same_columns(selection, noise, ["source_row", "label_id", "policy"], "Selection source")
    require(np.array_equal(h.booleans(selection.changed_existing, "Selection changed flag"), h.booleans(noise.changed, "Source noise changed")), "Selection changed flag alignment")
    require(noise.original_text.tolist() == [r[0] for r in raw["train"]] and noise.true_label.tolist() == [r[1] for r in raw["train"]], "Official training source ownership")
    labels = np.array([mapping[y] for _, y in raw["train"]], dtype=np.int64)
    require(np.array_equal(selection.label_id.to_numpy(), labels), "Selection source labels")
    require(np.array_equal(np.bincount(labels, minlength=150), np.full(150, 100)), "Train class balance")
    ranks = np.empty(15000, dtype=np.int64)
    for label in range(150):
        for index, policy in enumerate(("typo_light", "typo_medium")):
            source = np.flatnonzero((labels == label) & (noise.policy.to_numpy() == policy))
            require(len(source) == 50, f"Class {label}/{policy}: original policy count")
            order = np.random.default_rng(ALLOCATION_SEED + label * 2 + index).permutation(source)
            ranks[order] = np.arange(50)
    require(np.array_equal(selection.within_group_rank.to_numpy(), ranks), "Independent subset RNG/ranking")
    masks = {"clean_reference": np.zeros(15000, dtype=bool), "ratio_00": np.zeros(15000, dtype=bool)}
    for arm, k in (("ratio_10", 10), ("ratio_25", 25), ("ratio_50", 50)):
        masks[arm] = ranks < k
        require(np.array_equal(h.booleans(selection[f"selected_{arm}"], arm), masks[arm]), f"{arm}: fixed subset selection")
        for label in range(150):
            for policy in ("typo_light", "typo_medium"):
                require(int((masks[arm] & (labels == label) & (noise.policy.to_numpy() == policy)).sum()) == k,
                        f"{arm}/{label}/{policy}: per-class policy selection")
    require(not (masks["ratio_10"] & ~masks["ratio_25"]).any() and not (masks["ratio_25"] & ~masks["ratio_50"]).any(), "Nested subset relation")
    manifest = read_csv(ROOT / "data/fit_manifest.csv")
    require(len(manifest) == 135000 and set(manifest.arm) == set(ARMS), "Fit manifest rows/arms")
    require(not manifest.duplicated(["arm", "fit_row"]).any(), "Duplicate fit positions")
    summary = read_json(ROOT / "data/data_summary.json")
    require(summary["allocation_seed"] == ALLOCATION_SEED and summary["source_noise_seed"] == 31020261020, "Data summary RNG seeds")
    require(summary["source_noise_file_sha256"] == digest(PREVIOUS / "data/noisy_train.csv"), "Source noise hash binding")
    require(summary["nested_10_in_25_in_50"] is True and summary["one_shared_selection_across_model_seeds"] is True, "Subset nesting/shared-seed declaration")
    require(set(summary["arms"]) == set(ARMS), "Data summary arm grid")
    checked = []
    changed = h.booleans(noise.changed, "Noise changed")
    for arm in ARMS:
        n = 15000 if arm == "clean_reference" else 30000
        frame = manifest[manifest.arm == arm]
        sources = np.arange(15000) if n == 15000 else np.tile(np.arange(15000), 2)
        views = np.full(n, "clean", dtype=object)
        if n == 30000:
            views[15000:] = np.where(masks[arm], "noisy", "clean")
        require(np.array_equal(frame.fit_row.to_numpy(), np.arange(n)), f"{arm}: fit row order")
        require(np.array_equal(frame.source_row.to_numpy(), sources) and np.array_equal(frame.label_id.to_numpy(), labels[sources]), f"{arm}: exactly two source views and inherited labels")
        require(np.array_equal(frame.view.to_numpy(), views), f"{arm}: clean-first and selected cached-noise second view")
        mask = masks[arm]
        actual_changed = int((mask & changed).sum())
        expected = dict(fit_rows=n, independent_source_rows=15000, planned_noisy_ratio=RATIOS[arm],
            selected_noisy_views=int(mask.sum()), selected_light=int((mask & noise.policy.eq("typo_light").to_numpy()).sum()),
            selected_medium=int((mask & noise.policy.eq("typo_medium").to_numpy()).sum()),
            actual_changed_rows=actual_changed, actual_changed_pool_fraction=actual_changed / n,
            unchanged_pool_rows=n - actual_changed,
            selected_modified_token_counts={str(k): int(v) for k, v in noise.loc[mask, "modified_tokens"].value_counts().items()})
        for key, value in expected.items():
            require(summary["arms"][arm][key] == value, f"{arm}: coverage summary/{key}")
        checked.append(dict(arm=arm, **expected))
    return dict(raw=raw, mapping=mapping, labels=labels, masks=masks, data_checks=checked)


def verify_freezes(output, m, h):
    protocol = read_json(ROOT / "PROTOCOL.json")
    require(protocol["seeds"] == list(SEEDS) and protocol["arms"] == list(ARMS) and protocol["conditions"] == list(CONDITIONS), "Protocol grid")
    require(protocol["planned_noisy_pool_ratios"] == RATIOS and protocol["reference_map"] == REFERENCES, "Protocol ratios/references")
    allocation = protocol["allocation"]
    require(allocation["seed"] == ALLOCATION_SEED and allocation["policy_indices"] == {"typo_light": 0, "typo_medium": 1}, "Protocol selection seed")
    require(allocation["nested"] is True and allocation["shared_across_model_seeds"] is True, "Protocol nested shared subset")
    require(allocation["per_class_noisy_counts"] == {"ratio_00": 0, "ratio_10": 20, "ratio_25": 50, "ratio_50": 100}, "Protocol counts")
    frozen = read_json(ROOT / "INPUTS_FROZEN.json")
    development = read_json(output / "DEVELOPMENT_FROZEN.json")
    complete = read_json(output / "TEST_COMPLETE.json")
    input_hash, dev_hash = digest(ROOT / "INPUTS_FROZEN.json"), digest(output / "DEVELOPMENT_FROZEN.json")
    require(set(frozen) == {"frozen_at", "execution", "prior_runtime", "input_hashes"}, "Input manifest schema")
    require(development["inputs_frozen_sha256"] == complete["inputs_frozen_sha256"] == input_hash, "Input freeze bindings")
    require(complete["development_frozen_sha256"] == dev_hash, "Development freeze binding")
    require(development["detectors"] == 15 and development["validation_thresholds"] == 30
            and development["new_detectors"] == 6 and development["new_validation_thresholds"] == 12, "Development count")
    require(complete["test_settings"] == 90 and complete["original_models_inputs_and_reference_results_unchanged"] is True, "Test count/unchanged assertion")
    time = datetime.fromisoformat
    require(time(frozen["frozen_at"]) <= time(development["frozen_at"]) <= time(complete["started_at"]) <= time(complete["completed_at"]), "All-development-before-test gate")
    counts = {"inputs": m.verify_hashes(frozen["input_hashes"], "Frozen inputs"),
              "development": m.verify_hashes(development["development_hashes"], "Frozen development"),
              "test": m.verify_hashes(complete["output_hashes"], "Test outputs")}
    old_inputs = read_json(PREVIOUS / "INPUTS_FROZEN.json")
    old_dev = read_json(PREVIOUS / "outputs/DEVELOPMENT_FROZEN.json")
    old_test = read_json(PREVIOUS / "outputs/TEST_COMPLETE.json")
    require(old_test["inputs_frozen_sha256"] == old_dev["inputs_frozen_sha256"] == digest(PREVIOUS / "INPUTS_FROZEN.json"), "Previous input chain")
    require(old_test["development_frozen_sha256"] == digest(PREVIOUS / "outputs/DEVELOPMENT_FROZEN.json"), "Previous development chain")
    require(frozen["prior_runtime"] == old_inputs["runtime"], "Prior frozen runtime")
    required_input = [ROOT / "PROTOCOL.json", ROOT / "run_ratio.py", PREVIOUS / "INPUTS_FROZEN.json",
                      PREVIOUS / "outputs/DEVELOPMENT_FROZEN.json", PREVIOUS / "outputs/TEST_COMPLETE.json"]
    for category, inventory in (("inputs", old_inputs["input_hashes"]), ("development", old_dev["development_hashes"]), ("test", old_test["output_hashes"])):
        counts[f"prior_{category}"] = m.verify_hashes(inventory, f"Previous {category}")
        required_input += [Path(path) for path in inventory]
    required_input += [ROOT / "data" / name for name in ("selection_manifest.csv", "fit_manifest.csv", "data_summary.json")]
    m.required_hashes(required_input, frozen["input_hashes"], "Source and data inventory")
    m.required_hashes([output / "validation_metrics_by_seed.csv"], development["development_hashes"], "Aggregate validation")
    m.required_hashes([output / "metrics_by_seed.csv"], complete["output_hashes"], "Aggregate test")
    for arm in ARMS:
        for seed in SEEDS:
            folder = output / arm / f"seed{seed}"
            local = read_json(folder / "DEVELOPMENT_COMPLETE.json")
            n = 15000 if arm == "clean_reference" else 30000
            require(local["arm"] == arm and local["seed"] == seed and local["fit_rows"] == n
                    and local["planned_noisy_ratio"] == RATIOS[arm] and local["class_counts"] == [n // 150] * 150, f"{arm}/{seed}: fit ledger")
            require(local["inputs_frozen_sha256"] == input_hash, f"{arm}/{seed}: input binding")
            require(time(frozen["frozen_at"]) <= time(local["completed_at"]) <= time(development["frozen_at"]), f"{arm}/{seed}: local freeze chronology")
            m.verify_hashes(local["output_hashes"], f"{arm}/{seed}: development outputs")
            required_local = [folder / name for name in ("distance_model.npz", "frozen_thresholds.json", "validation_mahalanobis_threshold_curve.csv",
                                                         "validation_predictions.csv", "validation_metrics.csv")]
            m.required_hashes(required_local, local["output_hashes"], f"{arm}/{seed}: local inventory")
            m.required_hashes(required_local + [folder / "DEVELOPMENT_COMPLETE.json"], development["development_hashes"], f"{arm}/{seed}: global inventory")
            require(set(local["output_hashes"]).issubset(development["development_hashes"]), f"{arm}/{seed}: omitted dependency")
            m.required_hashes([folder / f"{c}_{suffix}.csv" for c in CONDITIONS for suffix in ("metrics", "predictions")], complete["output_hashes"], "Test prediction inventory")
    state = verify_data(m, h)
    for seed in SEEDS:
        require(read_json(PROJECT / f"outputs/bert/seed{seed}/label_mapping.json")["label_to_id"] == state["mapping"], "Shared seed label mapping")
    state.update(frozen=frozen, development=development, complete=complete, input_hash=input_hash,
                 development_hash=dev_hash, hash_inventory_counts=counts)
    return state


def verify_reports(output, metrics, frames, m, h):
    names = ("normal_states.csv", "rejection_transitions.csv", "clean_guardrails.csv", "benefit_retention.csv", "summary_mean_sd.csv")
    missing = [name for name in names if not (output / name).is_file()]
    if missing:
        return {"status": "NOT_PRESENT", "checked": False, "missing": missing}
    states = index_metrics(read_csv(output / names[0]), CONDITIONS, "Report states")
    for (arm, seed, condition), frame in frames.items():
        normal = frame.label_id.to_numpy() >= 0
        correct = (frame.predicted_label_id.to_numpy() == frame.label_id.to_numpy())[normal]
        for policy in POLICIES:
            rejected = h.booleans(frame[f"mahalanobis_{policy}_reject"], "State reject")[normal]
            events = {"accepted_correct": correct & ~rejected, "accepted_wrong": ~correct & ~rejected,
                      "rejected_correct": correct & rejected, "rejected_wrong": ~correct & rejected}
            row = states.loc[(arm, seed, condition, "mahalanobis", policy)]
            h.close(row.n_is, 4500, "State IS population")
            require(sum(int(event.sum()) for event in events.values()) == 4500, "State partition")
            for name, event in events.items():
                h.close(int(event.sum()), row[f"{name}_n"], f"State/{arm}/{seed}/{condition}/{policy}/{name}")
                h.close(event.mean(), row[f"{name}_rate_over_all_is"], f"State rate/{name}")
    table = read_csv(output / names[1])
    keys = ["candidate", "reference", "seed", "condition", "score", "policy", "stratum"]
    expected = {(a, r, s, c, "mahalanobis", p, z) for a in ("ratio_10", "ratio_25") for r in ("ratio_00", "ratio_50")
                for s in SEEDS for c in CONDITIONS for p in POLICIES for z in ("all_is", "head_correct", "head_wrong")}
    require(not table.duplicated(keys).any() and set(table[keys].itertuples(index=False, name=None)) == expected, "Report transition grid")
    table = table.set_index(keys).sort_index()
    for key, row in table.iterrows():
        arm, reference, seed, condition, score, policy, stratum = key
        target, source = frames[(arm, seed, condition)], frames[(reference, seed, condition)]
        scope = source.label_id.to_numpy() >= 0
        correct = source.predicted_label_id.to_numpy() == source.label_id.to_numpy()
        mask = scope if stratum == "all_is" else scope & (correct if stratum == "head_correct" else ~correct)
        before = h.booleans(source[f"mahalanobis_{policy}_reject"], "Before reject")
        after = h.booleans(target[f"mahalanobis_{policy}_reject"], "After reject")
        events = {"accept_accept": ~before & ~after, "accept_reject": ~before & after,
                  "reject_accept": before & ~after, "reject_reject": before & after}
        n = int(mask.sum())
        net = int((mask & after).sum()) - int((mask & before).sum())
        expected = dict(n_is=4500, stratum_n=n, source_accepted_n=int((mask & ~before).sum()),
            source_rejected_n=int((mask & before).sum()), net_rejection_change_n=net,
            net_rejection_change_rate_over_all_is=net / 4500)
        require(row.comparison == f"{arm}_minus_{reference}", "Report transition direction")
        require(net == int((mask & events["accept_reject"]).sum()) - int((mask & events["reject_accept"]).sum()), "Net rejection identity")
        for name, event in events.items():
            count = int((mask & event).sum())
            expected.update({f"{name}_n": count, f"{name}_rate_over_all_is": count / 4500, f"{name}_rate_over_stratum": count / n if n else None})
        for name, value in expected.items():
            h.close(value, row[name], f"Transition/{key}/{name}")
    mean = lambda a, c, p, metric: float(np.mean([metrics.loc[(a, s, c, "mahalanobis", p), metric] for s in SEEDS]))
    guardrails = read_csv(output / names[2])
    keys = ["candidate", "reference", "condition", "score", "policy", "metric"]
    guard_metrics = ("oos_f1", "oos_recall", "in_scope_acceptance")
    expected = {(a, r, "clean", "mahalanobis", p, metric) for a in ("ratio_10", "ratio_25")
                for r in ("ratio_00", "ratio_50", "clean_reference") for p in POLICIES for metric in guard_metrics}
    require(not guardrails.duplicated(keys).any() and set(guardrails[keys].itertuples(index=False, name=None)) == expected, "Report guardrail grid")
    for row in guardrails.itertuples(index=False):
        before, after = mean(row.reference, "clean", row.policy, row.metric), mean(row.candidate, "clean", row.policy, row.metric)
        for field, value in (("reference_mean", before), ("candidate_mean", after), ("delta", after - before), ("delta_pp", 100 * (after - before))):
            h.close(value, getattr(row, field), f"Guardrail/{row.candidate}/{row.reference}/{row.policy}/{row.metric}/{field}")
        all_pass = all(mean(row.candidate, "clean", row.policy, metric) >= mean(row.reference, "clean", row.policy, metric) for metric in guard_metrics)
        require(bool(h.booleans(pd.Series([row.mean_non_decline]), "Guardrail sign")[0]) == (after >= before), "Guardrail per-metric sign")
        require(bool(h.booleans(pd.Series([row.all_three_pass]), "Guardrail all pass")[0]) == all_pass, "Guardrail all-three decision")
        require(row.interpretation == ("improvement_relative_to_50" if row.reference == "ratio_50" else "strict_preservation"), "Guardrail comparison interpretation")
    retention = read_csv(output / names[3])
    keys = ["candidate", "condition", "score", "policy"]
    expected = {(a, c, "mahalanobis", p) for a in ("ratio_10", "ratio_25") for c in CONDITIONS for p in POLICIES}
    require(not retention.duplicated(keys).any() and set(retention[keys].itertuples(index=False, name=None)) == expected, "Report retention grid")
    for row in retention.itertuples(index=False):
        totals = {a: sum(int(metrics.loc[(a, s, row.condition, "mahalanobis", row.policy), "in_scope_false_rejections"]) for s in SEEDS) for a in ARMS}
        numerator_count, denominator_count = totals["ratio_00"] - totals[row.candidate], totals["ratio_00"] - totals["ratio_50"]
        require(row.baseline == "ratio_00" and row.benefit_reference == "ratio_50", "Retention correct references")
        expected_fields = {"baseline_fr_mean": totals["ratio_00"] / 13500, "candidate_fr_mean": totals[row.candidate] / 13500,
            "ratio50_fr_mean": totals["ratio_50"] / 13500, "historical_clean_reference_fr_mean": totals["clean_reference"] / 13500,
            "numerator": numerator_count / 13500, "denominator": denominator_count / 13500,
            "benefit_retention_percent": 100 * numerator_count / denominator_count if denominator_count > 0 else None}
        for field, value in expected_fields.items():
            h.close(value, getattr(row, field), f"Retention/{row.candidate}/{row.condition}/{row.policy}/{field}")
        require(bool(h.booleans(pd.Series([row.defined]), "Retention defined")[0]) == (denominator_count > 0), "Retention domain must use positive exact count difference")
    summary = read_csv(output / names[4])
    keys = ["arm", "condition", "score", "policy"]
    expected = {(a, c, "mahalanobis", p) for a in ARMS for c in CONDITIONS for p in POLICIES}
    require(not summary.duplicated(keys).any() and set(summary[keys].itertuples(index=False, name=None)) == expected, "Report summary grid")
    summary = summary.set_index(keys).sort_index()
    for key, row in summary.iterrows():
        arm, condition, score, policy = key
        require(row.n_seeds == 3, "Summary fixed three seeds")
        for field in row.index:
            if not field.endswith("_mean"):
                continue
            name = field[:-5]
            if name == "in_scope_wrong_and_accepted_rate":
                values = []
                for seed in SEEDS:
                    frame = frames[(arm, seed, condition)]
                    normal = frame.label_id.to_numpy() >= 0
                    wrong = frame.predicted_label_id.to_numpy() != frame.label_id.to_numpy()
                    rejected = h.booleans(frame[f"mahalanobis_{policy}_reject"], "Summary reject")
                    values.append(int((normal & wrong & ~rejected).sum()) / 4500)
            elif name == "oos_false_acceptance_rate":
                values = [int(metrics.loc[(arm, seed, condition, score, policy), "oos_missed"]) / 1000 for seed in SEEDS]
            else:
                values = [metrics.loc[(arm, seed, condition, score, policy), name] for seed in SEEDS]
            h.close(float(np.mean(values)), row[field], f"Summary/{key}/{name}/mean")
            h.close(float(np.std(values, ddof=1)), row[f"{name}_sd"], f"Summary/{key}/{name}/sample SD")
    return {"status": "PASS", "checked": True, "state_rows": 90, "transition_rows": 216,
            "guardrail_rows": 36, "benefit_retention_rows": 12, "summary_rows": 30,
            "zero_benefit_denominator_checked_from_integer_confusion_counts": True,
            "derived_wrong_acceptance_and_oos_false_acceptance_verified": True,
            "artifact_sha256": {name: digest(output / name) for name in names}}


def verify_results(output, state, m, h):
    metrics = index_metrics(read_csv(output / "metrics_by_seed.csv"), CONDITIONS, "Test metrics")
    validation = index_metrics(read_csv(output / "validation_metrics_by_seed.csv"), ("validation",), "Validation metrics")
    checked_test = checked_val = candidates = 0
    direct_checks, fits, reference_checks, frames = [], [], [], {}
    labels, mapping, raw = state["labels"], state["mapping"], state["raw"]
    inverse = {v: k for k, v in mapping.items()}
    for seed in SEEDS:
        source = AUGMENTED / f"outputs/typo_mix/seed{seed}"
        clean = np.load(source / "train_cls.npy", mmap_mode="r", allow_pickle=False)
        noisy = np.load(PREVIOUS / f"features/seed{seed}/noisy_train_cls.npy", mmap_mode="r", allow_pickle=False)
        train_meta = read_csv(source / "train_metadata.csv")
        h.verify_metadata(train_meta, raw["train"], ["train"] * 15000, mapping, f"Seed {seed}: clean train source")
        require(clean.shape == noisy.shape == (15000, 768) and clean.dtype == noisy.dtype == np.float32, "Fixed cached fitting arrays")
        require(np.isfinite(clean).all() and np.isfinite(noisy).all(), "Finite fitting features")
        for arm in ARMS:
            folder = output / arm / f"seed{seed}"
            thresholds = read_json(folder / "frozen_thresholds.json")
            if arm in REFERENCES:
                reference = PREVIOUS / "outputs" / REFERENCES[arm] / f"seed{seed}"
                for name in ("distance_model.npz", "frozen_thresholds.json", "validation_mahalanobis_threshold_curve.csv"):
                    require(digest(folder / name) == digest(reference / name), f"{arm}/{seed}: reference {name} differs")
                with np.load(folder / "distance_model.npz", allow_pickle=False) as archive:
                    model = {key: np.array(archive[key], copy=True) for key in archive.files}
                reference_checks.append(dict(arm=arm, seed=seed, old_arm=REFERENCES[arm], detector_and_thresholds_identical=True))
            else:
                second = np.array(clean, copy=True)
                mask = state["masks"][arm]
                second[mask] = noisy[mask]
                fitted = np.concatenate([clean, second])
                expected = m.independent_statistics(fitted, np.tile(labels, 2))
                model = m.verify_distance_fit(folder / "distance_model.npz", expected, h, f"{arm}/{seed}")
                local = read_json(folder / "DEVELOPMENT_COMPLETE.json")
                h.close(local["shrinkage"], expected["shrinkage"], f"{arm}/{seed}: shrinkage ledger")
                fits.append(dict(arm=arm, seed=seed, fit_rows=30000, selected_noisy_views=int(mask.sum()),
                                 independent_shrinkage=expected["shrinkage"], stored_shrinkage=float(model["shrinkage"])))
                del second, fitted, expected
            for condition in ("validation",) + CONDITIONS:
                context = f"{arm}/{seed}/{condition}"
                frame = read_csv(folder / f"{condition}_predictions.csv")
                metadata = read_csv(source / f"{condition}_metadata.csv")
                old_prediction = read_csv(source / f"{condition}_predictions.csv")
                m.same_columns(frame, metadata, list(metadata.columns), f"{context}: cached input metadata")
                m.same_columns(frame, old_prediction, ["predicted_label_id", "predicted_label"], f"{context}: unchanged classifier predictions")
                rows = raw["val"] + raw["oos_val"] if condition == "validation" else raw["test"] + raw["oos_test"]
                expected_labels = np.array([mapping.get(y, -1) for _, y in rows], dtype=np.int64)
                require(np.array_equal(frame.row_id.to_numpy(), np.arange(len(rows))) and np.array_equal(frame.label_id.to_numpy(), expected_labels), f"{context}: original row/label alignment")
                require(frame.true_label.tolist() == [r[1] for r in rows], f"{context}: inherited labels")
                n_is, n_oos = (3000, 100) if condition == "validation" else (4500, 1000)
                require(len(frame) == n_is + n_oos and int((expected_labels < 0).sum()) == n_oos and
                        np.array_equal(np.bincount(expected_labels[expected_labels >= 0], minlength=150), np.full(150, n_is // 150)), f"{context}: IS/OOS class counts")
                features = np.load(source / f"{condition}_cls.npy", allow_pickle=False, mmap_mode="r")
                logits = np.load(source / f"{condition}_logits_fp32.npy", allow_pickle=False, mmap_mode="r")
                require(features.shape == (len(frame), 768) and features.dtype == np.float32 and
                        logits.shape == (len(frame), 150) and logits.dtype == np.float32, f"{context}: cached FP32 dimensions")
                require(frame.predicted_label.tolist() == [inverse[int(k)] for k in logits.argmax(axis=1)], f"{context}: decoded classifier labels")
                direct_checks.append(m.verify_direct_score(features, frame, model, h, context))
                index = validation if condition == "validation" else metrics
                selected = index.loc[(arm, seed, condition)]
                checked = m.verify_metrics(frame, logits, thresholds, selected, h, context)
                m.compare_local_metrics(folder / f"{condition}_metrics.csv", selected, h, context)
                if arm in REFERENCES:
                    require(digest(folder / f"{condition}_predictions.csv") == digest(reference / f"{condition}_predictions.csv"), f"{context}: reference predictions are not byte-identical")
                    previous_metrics = read_csv(reference / f"{condition}_metrics.csv").set_index(["score", "policy"])
                    for key, row in previous_metrics.iterrows():
                        for field, value in row.items():
                            if field not in ("arm", "seed", "condition"):
                                h.close(value, selected.loc[key, field], f"{context}: previous metric/{field}", atol=0, rtol=0)
                if condition == "validation":
                    curve = read_csv(folder / "validation_mahalanobis_threshold_curve.csv")
                    candidates += h.verify_validation_thresholds(frame.mahalanobis_score.to_numpy(dtype=float), h.booleans(frame.is_oos, context),
                        {p: thresholds[f"mahalanobis/{p}"] for p in POLICIES}, curve, context)
                    checked_val += checked
                else:
                    checked_test += checked
                    frames[(arm, seed, condition)] = frame
                del features, logits
            del model
        del clean, noisy
    reports = verify_reports(output, metrics, frames, m, h)
    guardrails, retention = [], []
    for arm in ("ratio_10", "ratio_25"):
        for policy in POLICIES:
            for reference in ("clean_reference", "ratio_00"):
                changes = {metric: float(np.mean([metrics.loc[(arm, s, "clean", "mahalanobis", policy), metric] for s in SEEDS])
                    - np.mean([metrics.loc[(reference, s, "clean", "mahalanobis", policy), metric] for s in SEEDS]))
                    for metric in ("oos_f1", "oos_recall", "in_scope_acceptance")}
                guardrails.append(dict(arm=arm, reference=reference, policy=policy, mean_changes=changes,
                                       strict_descriptive_retention=all(v >= 0 for v in changes.values())))
            for condition in CONDITIONS:
                rate = lambda a: float(np.mean([metrics.loc[(a, s, condition, "mahalanobis", policy), "in_scope_false_rejection_rate"] for s in SEEDS]))
                zero, fifty, candidate = rate("ratio_00"), rate("ratio_50"), rate(arm)
                counts = {a: sum(int(metrics.loc[(a, s, condition, "mahalanobis", policy), "in_scope_false_rejections"]) for s in SEEDS) for a in ("ratio_00", "ratio_50", arm)}
                denominator = counts["ratio_00"] - counts["ratio_50"]
                numerator = counts["ratio_00"] - counts[arm]
                retention.append(dict(arm=arm, condition=condition, policy=policy, ratio00_fr=zero,
                    ratio50_fr=fifty, candidate_fr=candidate,
                    benefit_retention_percent=100 * numerator / denominator if denominator > 0 else None))
    require(digest(ROOT / "INPUTS_FROZEN.json") == state["input_hash"] and digest(output / "DEVELOPMENT_FROZEN.json") == state["development_hash"], "Manifest changed during verification")
    m.verify_hashes(state["frozen"]["input_hashes"], "Ending input hashes")
    m.verify_hashes(state["development"]["development_hashes"], "Ending development hashes")
    m.verify_hashes(state["complete"]["output_hashes"], "Ending test hashes")
    return dict(verified_test_settings=checked_test, verified_validation_settings=checked_val,
        validation_candidates_checked=candidates, test_prediction_rows_checked=45 * 5500,
        new_detectors_independently_recomputed=len(fits), independent_fit_statistics=fits,
        copied_references_checked=reference_checks, references_predictions_and_metrics_exactly_equal=True,
        subset_ranking_and_nesting_verified=True, data_coverage=state["data_checks"],
        all_validation_candidates_and_tie_rules_verified=True, fixed_classifier_predictions_verified=True,
        manual_confusion_and_closed_metrics_verified=True, direct_covariance_solve_checks=direct_checks,
        report_checks=reports, independent_clean_guardrails=guardrails, independent_benefit_retention=retention,
        hash_inventory_counts=state["hash_inventory_counts"], inputs_manifest_sha256=state["input_hash"],
        development_manifest_sha256=state["development_hash"], test_completion_sha256=digest(output / "TEST_COMPLETE.json"),
        metrics_sha256=digest(output / "metrics_by_seed.csv"),
        limitations=["No BERT inference or training is run; saved feature provenance and immutable model/input hashes are verified.",
            "Six new detector fits are independently reconstructed; nine reference detectors must match prior frozen artifacts byte-for-byte.",
            "Direct covariance solves sample six rows per detector/condition and all 150 class centers per sampled row.",
            "Strict clean retention is a signed frozen-seed mean comparison, not statistical noninferiority."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    output = args.output.resolve()
    required = [ROOT / "PROTOCOL.json", ROOT / "INPUTS_FROZEN.json", output / "DEVELOPMENT_FROZEN.json",
                output / "TEST_COMPLETE.json", output / "metrics_by_seed.csv"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print(json.dumps(dict(status="NOT_READY", missing=missing), ensure_ascii=False, indent=2))
        return 2
    started = datetime.now(timezone.utc).isoformat()
    try:
        m, h, helper_hashes = load_independent_helpers()
        context = threadpool_limits(limits=4) if threadpool_limits else nullcontext()
        with context:
            state = verify_freezes(output, m, h)
            result = verify_results(output, state, m, h)
        result.update(status="PASS", independent_helper_sha256=helper_hashes, verifier_sha256=digest(Path(__file__)))
    except Exception as error:
        result = dict(status="FAIL", error_type=type(error).__name__, error=str(error))
    result.update(started_at_utc=started, completed_at_utc=datetime.now(timezone.utc).isoformat())
    (output / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
