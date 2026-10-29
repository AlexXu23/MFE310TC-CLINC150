"""Independent audit of IS-only acceptance alignment on immutable scores.

Only earlier independent verifier helpers are imported. No runner, report,
detector fitting, sklearn or BERT inference is used. This file writes only
outputs/verification.json; all upstream experiments remain read-only.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "BERT_Rejection_Ratio_20261019"
PROJECT = ROOT.parent / "CLINC150_Team_20261019"
ARMS = ("clean_reference", "ratio_00", "ratio_10", "ratio_25", "ratio_50")
SEEDS = (42, 43, 44)
CONDITIONS = ("clean", "typo_light", "typo_medium")
OLD = "is_acceptance_95"
NEW = "aligned_is_acceptance_95"
POLICIES = (OLD, NEW)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    return pd.read_csv(path, keep_default_na=False, float_precision="round_trip")


def load_helpers():
    path = ROOT.parent / "BERT_Mixed_Rejection_20261019/verify_mixed_rejection.py"
    spec = importlib.util.spec_from_file_location("independent_mixed_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base, hashes = module.helpers()
    hashes[path.resolve().as_posix()] = digest(path)
    return module, base, hashes


def index_table(table, conditions, context):
    keys = ["arm", "seed", "condition", "score", "policy"]
    expected = {(a, s, c, "mahalanobis", p) for a in ARMS for s in SEEDS for c in conditions for p in POLICIES}
    require(not table.duplicated(keys).any() and set(table[keys].itertuples(index=False, name=None)) == expected,
            f"{context}: expected {len(expected)} unique settings")
    return table.set_index(keys).sort_index()


def verify_gates(output, m):
    protocol = read_json(ROOT / "PROTOCOL.json")
    require(protocol["arms"] == list(ARMS) and protocol["seeds"] == list(SEEDS) and protocol["conditions"] == list(CONDITIONS)
            and protocol["policies"] == list(POLICIES), "Protocol grid")
    require(protocol["alignment"]["target_accepted"] == 2850 and protocol["alignment"]["normal_validation_rows"] == 3000
            and protocol["alignment"]["target_acceptance"] == .95, "Protocol acceptance target")
    frozen = read_json(ROOT / "INPUTS_FROZEN.json")
    development = read_json(output / "DEVELOPMENT_FROZEN.json")
    complete = read_json(output / "TEST_COMPLETE.json")
    input_hash, dev_hash = digest(ROOT / "INPUTS_FROZEN.json"), digest(output / "DEVELOPMENT_FROZEN.json")
    require(set(frozen) == {"frozen_at", "execution", "input_hashes"}, "Input manifest schema")
    require(development["inputs_frozen_sha256"] == complete["inputs_frozen_sha256"] == input_hash, "Input bindings")
    require(complete["development_frozen_sha256"] == dev_hash, "Development binding")
    require(development["existing_detectors"] == 15 and development["new_thresholds"] == 15
            and development["total_thresholds_including_original"] == 30, "Development totals")
    require(development["target_accepted"] == 2850 and development["selection_used_oos"] is False, "IS-only development rule")
    require(complete["test_settings"] == 90 and complete["original_models_detectors_features_scores_unchanged"] is True, "Test totals/immutability")
    time = datetime.fromisoformat
    require(time(frozen["frozen_at"]) <= time(development["frozen_at"]) <= time(complete["started_at"]) <= time(complete["completed_at"]), "Threshold freeze before test rethresholding")
    counts = {"inputs": m.verify_hashes(frozen["input_hashes"], "Input hashes"),
        "development": m.verify_hashes(development["development_hashes"], "Development hashes"),
        "test": m.verify_hashes(complete["output_hashes"], "Test hashes")}
    previous = [read_json(PRIOR / "INPUTS_FROZEN.json"), read_json(PRIOR / "outputs/DEVELOPMENT_FROZEN.json"), read_json(PRIOR / "outputs/TEST_COMPLETE.json")]
    require(previous[2]["inputs_frozen_sha256"] == previous[1]["inputs_frozen_sha256"] == digest(PRIOR / "INPUTS_FROZEN.json"), "Prior input chain")
    require(previous[2]["development_frozen_sha256"] == digest(PRIOR / "outputs/DEVELOPMENT_FROZEN.json"), "Prior development chain")
    dependencies = [ROOT / "PROTOCOL.json", ROOT / "run_alignment.py", PRIOR / "INPUTS_FROZEN.json",
                    PRIOR / "outputs/DEVELOPMENT_FROZEN.json", PRIOR / "outputs/TEST_COMPLETE.json"]
    for document, key in zip(previous, ("input_hashes", "development_hashes", "output_hashes")):
        m.verify_hashes(document[key], f"Prior {key}")
        dependencies += [Path(path) for path in document[key]]
    m.required_hashes(dependencies, frozen["input_hashes"], "Inherited source inventory")
    m.required_hashes([output / name for name in ("alignment_summary.json", "operating_points.csv", "validation_metrics_by_seed.csv")], development["development_hashes"], "Global development inventory")
    m.required_hashes([output / "metrics_by_seed.csv"], complete["output_hashes"], "Test metric inventory")
    for arm in ARMS:
        for seed in SEEDS:
            folder = output / arm / f"seed{seed}"
            local = read_json(folder / "DEVELOPMENT_COMPLETE.json")
            require(local["arm"] == arm and local["seed"] == seed and local["inputs_frozen_sha256"] == input_hash, "Local development identity")
            detector = PRIOR / f"outputs/{arm}/seed{seed}/distance_model.npz"
            require(Path(local["detector_path"]).resolve() == detector.resolve() and local["detector_sha256"] == digest(detector), "Existing fixed detector binding")
            require(time(frozen["frozen_at"]) <= time(local["completed_at"]) <= time(development["frozen_at"]), "Local freeze chronology")
            m.verify_hashes(local["output_hashes"], f"Local {arm}/{seed} hashes")
            required = [folder / name for name in ("frozen_thresholds.json", "validation_is_threshold_candidates.csv", "validation_predictions.csv", "validation_metrics.csv")]
            m.required_hashes(required, local["output_hashes"], "Local development inventory")
            m.required_hashes(required + [folder / "DEVELOPMENT_COMPLETE.json"], development["development_hashes"], "Global development inventory")
            m.required_hashes([folder / f"{c}_{suffix}.csv" for c in CONDITIONS for suffix in ("predictions", "metrics")], complete["output_hashes"], "Test output inventory")
    return dict(frozen=frozen, development=development, complete=complete, input_hash=input_hash, development_hash=dev_hash, hash_counts=counts)


def crosscheck_alignment(output, state, h):
    options, common = {}, None
    for arm in ARMS:
        for seed in SEEDS:
            old = read_csv(PRIOR / f"outputs/{arm}/seed{seed}/validation_predictions.csv")
            scope = ~h.booleans(old.is_oos, "Validation scope")
            require(int(scope.sum()) == 3000 and int((~scope).sum()) == 100, "Validation scope counts")
            # Group IS scores only. Counts below each unique value follow from
            # group multiplicities; no OOS value or outcome enters this check.
            values, counts = np.unique(old.mahalanobis_score.to_numpy(dtype=float)[scope], return_counts=True)
            thresholds = np.r_[values, np.nextafter(values[-1], np.inf)]
            accepted = np.r_[0, np.cumsum(counts)]
            options[(arm, seed)] = (thresholds, accepted, old, scope)
            attainable = set(map(int, accepted))
            common = attainable if common is None else common & attainable
    chosen_count = min(value for value in common if value >= 2850)
    summary = read_json(output / "alignment_summary.json")
    require(summary["target_accepted"] == 2850 and summary["actual_common_accepted"] == chosen_count
            and summary["validation_is_rows"] == 3000 and summary["selection_used_oos"] is False, "Alignment joint target")
    require(summary["common_feasible_counts"] == sorted(common) and summary["common_feasible_count_count"] == len(common), "Common attainable IS-count intersection")
    require(summary["target_acceptance"] == .95 and summary["actual_common_acceptance"] == chosen_count / 3000, "Requested/achieved acceptance")
    require(state["development"]["actual_common_accepted"] == chosen_count, "Development joint acceptance")
    require(datetime.fromisoformat(summary["selected_at"]) <= datetime.fromisoformat(state["development"]["frozen_at"]), "Selection after development freeze")
    table = read_csv(output / "operating_points.csv")
    require(len(table) == 15 and not table.duplicated(["arm", "seed"]).any(), "Operating point grid")
    indexed = table.set_index(["arm", "seed"])
    require(set(indexed.index) == set(options), "Operating point identities")
    require(len(summary["models"]) == 15, "Alignment model summary count")
    records = {(row["arm"], row["seed"]): row for row in summary["models"]}
    require(set(records) == set(options), "Alignment summary identities")
    cutoffs = {}
    for (arm, seed), (thresholds, accepted, old, scope) in options.items():
        folder = output / arm / f"seed{seed}"
        chosen = float(thresholds[accepted == chosen_count].min())
        source_thresholds = read_json(PRIOR / f"outputs/{arm}/seed{seed}/frozen_thresholds.json")
        frozen = read_json(folder / "frozen_thresholds.json")
        require(set(frozen) == {f"mahalanobis/{p}" for p in POLICIES}, "Frozen policy set")
        require(frozen[f"mahalanobis/{OLD}"] == source_thresholds[f"mahalanobis/{OLD}"], "Old threshold dictionary changed")
        new = frozen[f"mahalanobis/{NEW}"]
        require(new["threshold"] == chosen and new["actual_accepted"] == chosen_count and new["target_accepted"] == 2850
                and new["validation_is_rows"] == 3000 and new["in_scope_acceptance"] == chosen_count / 3000, "Independent IS-only chosen cutoff")
        require(new["validation_predictions_sha256"] == digest(PRIOR / f"outputs/{arm}/seed{seed}/validation_predictions.csv"), "IS cutoff source binding")
        candidates = read_csv(folder / "validation_is_threshold_candidates.csv")
        require(np.array_equal(candidates.threshold.to_numpy(), thresholds) and np.array_equal(candidates.accepted_is.to_numpy(), accepted)
                and candidates.validation_is_rows.eq(3000).all(), "IS-only candidate representation")
        old_threshold = frozen[f"mahalanobis/{OLD}"]["threshold"]
        old_count = int((old.mahalanobis_score.to_numpy()[scope] < old_threshold).sum())
        expected = dict(requested_accepted=2850, actual_accepted=chosen_count, requested_acceptance=.95,
            actual_acceptance=chosen_count / 3000, original_accepted=old_count, original_threshold=old_threshold,
            aligned_threshold=chosen, unique_is_scores=len(thresholds) - 1, candidate_count=len(thresholds), threshold_delta=chosen - old_threshold)
        for field, value in expected.items():
            h.close(value, indexed.loc[(arm, seed), field], f"Operating point/{arm}/{seed}/{field}")
            h.close(value, records[(arm, seed)][field], f"Alignment summary/{arm}/{seed}/{field}")
        cutoffs[(arm, seed)] = frozen
    return cutoffs, chosen_count, {"models": 15, "requested_is_accepted": 2850, "actual_common_is_accepted": chosen_count,
        "exact_requested_95_percent": chosen_count == 2850, "common_attainable_counts": len(common),
        "is_only_cutoffs_and_minimum_candidate_crosschecked": True}


def verify_reports(output, metrics, frames, h):
    names = ("summary_mean_sd.csv", "within_arm_changes.csv", "cross_arm_gaps.csv", "normal_states.csv", "policy_transitions.csv")
    missing = [name for name in names if not (output / name).is_file()]
    if missing:
        return {"status": "NOT_PRESENT", "checked": False, "missing": missing}
    metric_names = ("in_scope_false_rejection_rate", "in_scope_correct_and_accepted_rate", "in_scope_wrong_and_accepted_rate",
                    "oos_false_acceptance_rate", "oos_f1", "oos_recall", "in_scope_acceptance", "combined_151_accuracy")
    values = {}
    for (arm, seed, condition), frame in frames.items():
        labels, pred = frame.label_id.to_numpy(dtype=np.int64), frame.predicted_label_id.to_numpy(dtype=np.int64)
        normal = labels >= 0
        for policy in POLICIES:
            reject = h.booleans(frame[f"mahalanobis_{policy}_reject"], "Report value rejection")
            computed = h.manual_metrics(labels, pred, ~normal, reject, frame.mahalanobis_score.to_numpy(dtype=float))
            computed["in_scope_wrong_and_accepted_rate"] = int((normal & (pred != labels) & ~reject).sum()) / 4500
            computed["oos_false_acceptance_rate"] = int((~normal & ~reject).sum()) / 1000
            for name, value in computed.items():
                values[(arm, seed, condition, policy, name)] = value
    within = read_csv(output / names[1])
    keys = ["arm", "seed", "condition", "metric"]
    expected_keys = {(a, s, c, metric) for a in ARMS for s in SEEDS for c in CONDITIONS for metric in metric_names}
    require(not within.duplicated(keys).any() and set(within[keys].itertuples(index=False, name=None)) == expected_keys, "Within-arm change grid: expected 360")
    for row in within.itertuples(index=False):
        old = values[(row.arm, row.seed, row.condition, OLD, row.metric)]
        aligned = values[(row.arm, row.seed, row.condition, NEW, row.metric)]
        for field, value in (("old_value", old), ("aligned_value", aligned), ("aligned_minus_original", aligned - old),
                             ("aligned_minus_original_pp", 100 * (aligned - old))):
            h.close(value, getattr(row, field), f"Within change/{row.arm}/{row.seed}/{row.condition}/{row.metric}/{field}")
    gaps = read_csv(output / names[2])
    keys = ["candidate", "reference", "seed", "condition", "metric"]
    expected_keys = {(a, r, s, c, metric) for a in ("ratio_10", "ratio_25") for r in ("ratio_00", "ratio_50")
                     for s in SEEDS for c in CONDITIONS for metric in metric_names}
    require(not gaps.duplicated(keys).any() and set(gaps[keys].itertuples(index=False, name=None)) == expected_keys, "Cross-arm gap grid: expected 288")
    for row in gaps.itertuples(index=False):
        old_gap = values[(row.candidate, row.seed, row.condition, OLD, row.metric)] - values[(row.reference, row.seed, row.condition, OLD, row.metric)]
        aligned_gap = values[(row.candidate, row.seed, row.condition, NEW, row.metric)] - values[(row.reference, row.seed, row.condition, NEW, row.metric)]
        gap_change = old_gap - aligned_gap
        # The gap change is reference policy shift minus candidate policy shift.
        shift_identity = (values[(row.reference, row.seed, row.condition, NEW, row.metric)] - values[(row.reference, row.seed, row.condition, OLD, row.metric)]) - (
            values[(row.candidate, row.seed, row.condition, NEW, row.metric)] - values[(row.candidate, row.seed, row.condition, OLD, row.metric)])
        h.close(gap_change, shift_identity, "Gap decomposition accounting identity")
        for field, value in (("old_gap", old_gap), ("aligned_gap", aligned_gap), ("gap_change", gap_change)):
            h.close(value, getattr(row, field), f"Cross-arm/{row.candidate}/{row.reference}/{row.seed}/{row.condition}/{row.metric}/{field}")
            h.close(value * 100, getattr(row, field + "_pp"), f"Cross-arm/{field}_pp")
    states = index_table(read_csv(output / names[3]), CONDITIONS, "Report normal states")
    for (arm, seed, condition), frame in frames.items():
        normal = frame.label_id.to_numpy() >= 0
        correct = frame.predicted_label_id.to_numpy() == frame.label_id.to_numpy()
        for policy in POLICIES:
            reject = h.booleans(frame[f"mahalanobis_{policy}_reject"], "Normal states rejection")
            masks = {"accepted_correct": normal & ~reject & correct, "accepted_wrong": normal & ~reject & ~correct,
                     "rejected_correct": normal & reject & correct, "rejected_wrong": normal & reject & ~correct}
            row = states.loc[(arm, seed, condition, "mahalanobis", policy)]
            require(sum(int(mask.sum()) for mask in masks.values()) == 4500, "Normal states partition")
            for name, mask in masks.items():
                h.close(int(mask.sum()), row[name + "_n"], f"Normal state/{name}")
                h.close(mask.sum() / 4500, row[name + "_rate_over_all_is"], f"Normal state rate/{name}")
    flows = read_csv(output / names[4])
    keys = ["arm", "seed", "condition", "stratum"]
    expected_keys = {(a, s, c, stratum) for a in ARMS for s in SEEDS for c in CONDITIONS for stratum in ("all_is", "head_correct", "head_wrong", "oos")}
    require(not flows.duplicated(keys).any() and set(flows[keys].itertuples(index=False, name=None)) == expected_keys, "Policy transition grid: expected 180")
    for row in flows.itertuples(index=False):
        require(row.old_policy == OLD and row.new_policy == NEW, "Policy flow direction labels")
        frame = frames[(row.arm, row.seed, row.condition)]
        normal = frame.label_id.to_numpy() >= 0
        correct = frame.predicted_label_id.to_numpy() == frame.label_id.to_numpy()
        groups = {"all_is": normal, "head_correct": normal & correct, "head_wrong": normal & ~correct, "oos": ~normal}
        mask = groups[row.stratum]
        before = h.booleans(frame[f"mahalanobis_{OLD}_reject"], "Old policy transition")
        after = h.booleans(frame[f"mahalanobis_{NEW}_reject"], "Aligned policy transition")
        group_n, scope_n = int(mask.sum()), 1000 if row.stratum == "oos" else 4500
        events = {"accept_accept": mask & ~before & ~after, "accept_reject": mask & ~before & after,
                  "reject_accept": mask & before & ~after, "reject_reject": mask & before & after}
        require(sum(int(event.sum()) for event in events.values()) == group_n, "Policy transition partition")
        h.close(group_n, row.group_n, "Policy transition group count")
        h.close(scope_n, row.scope_n, "Policy transition scope denominator")
        for name, event in events.items():
            count = int(event.sum())
            h.close(count, getattr(row, name + "_n"), f"Policy flow/{name}")
            h.close(count / group_n if group_n else None, getattr(row, name + "_rate_over_group"), f"Policy flow/{name}/group")
            h.close(count / scope_n, getattr(row, name + "_rate_over_scope"), f"Policy flow/{name}/scope")
        net = int((mask & after).sum()) - int((mask & before).sum())
        require(net == int(events["accept_reject"].sum()) - int(events["reject_accept"].sum()), "Policy net transition identity")
        h.close(net, row.net_rejection_change_n, "Policy net rejection")
        if hasattr(row, "net_rejection_change_rate_over_scope"):
            h.close(net / scope_n, row.net_rejection_change_rate_over_scope, "Policy net rejection rate")
    summary = read_csv(output / names[0])
    keys = ["arm", "condition", "score", "policy"]
    expected_keys = {(a, c, "mahalanobis", p) for a in ARMS for c in CONDITIONS for p in POLICIES}
    require(not summary.duplicated(keys).any() and set(summary[keys].itertuples(index=False, name=None)) == expected_keys, "Summary grid")
    for row in summary.to_dict("records"):
        arm, condition, policy = row["arm"], row["condition"], row["policy"]
        require(row["n_seeds"] == 3, "Summary three fixed models")
        for field, recorded in row.items():
            if not field.endswith("_mean"):
                continue
            metric = field[:-5]
            if metric in ("in_scope_wrong_and_accepted_rate", "oos_false_acceptance_rate"):
                observations = [values[(arm, s, condition, policy, metric)] for s in SEEDS]
            else:
                observations = [metrics.loc[(arm, s, condition, "mahalanobis", policy), metric] for s in SEEDS]
            h.close(float(np.mean(observations)), recorded, f"Summary/{arm}/{condition}/{policy}/{metric}/mean")
            h.close(float(np.std(observations, ddof=1)), row[metric + "_sd"], f"Summary/{metric}/sample SD")
    interval_check = {"status": "NOT_PRESENT", "point_estimates_checked": 0}
    interval_path = output / "paired_intervals.csv"
    if interval_path.is_file():
        intervals = read_csv(interval_path)
        keys = ["contrast_type", "arm", "candidate", "reference", "condition", "metric", "estimand"]
        expected = {("within_arm", a, "", "", c, metric, "aligned_minus_original") for a in ARMS for c in CONDITIONS for metric in metric_names}
        expected |= {("cross_arm", "", a, r, c, metric, estimand) for a in ("ratio_10", "ratio_25") for r in ("ratio_00", "ratio_50")
                     for c in CONDITIONS for metric in metric_names for estimand in ("old_gap", "aligned_gap", "gap_change")}
        require(not intervals.duplicated(keys).any() and set(intervals[keys].itertuples(index=False, name=None)) == expected, "Paired interval contrast grid: expected 408")
        average = lambda a, c, p, metric: float(np.mean([values[(a, s, c, p, metric)] for s in SEEDS]))
        for row in intervals.itertuples(index=False):
            if row.contrast_type == "within_arm":
                estimate = average(row.arm, row.condition, NEW, row.metric) - average(row.arm, row.condition, OLD, row.metric)
            else:
                old = average(row.candidate, row.condition, OLD, row.metric) - average(row.reference, row.condition, OLD, row.metric)
                aligned = average(row.candidate, row.condition, NEW, row.metric) - average(row.reference, row.condition, NEW, row.metric)
                estimate = {"old_gap": old, "aligned_gap": aligned, "gap_change": old - aligned}[row.estimand]
            h.close(estimate, row.estimate, "Paired interval independent point estimate")
            require(np.isfinite([row.ci_low, row.ci_high]).all() and row.ci_low <= row.ci_high, "Paired interval bounds")
            require(row.bootstrap_repeats == 2000 and row.bootstrap_seed == 20261022 and row.n_is == 4500
                    and row.n_oos == 1000 and row.n_fixed_models == 3, "Paired interval protocol metadata")
            for name in ("estimate", "ci_low", "ci_high"):
                h.close(getattr(row, name) * 100, getattr(row, name + "_pp"), f"Paired interval percentage-point conversion/{name}")
        interval_check = dict(status="PASS", point_estimates_checked=408, units_bounds_and_metadata_checked=True,
                              bootstrap_quantiles_independently_resampled=False, sha256=digest(interval_path))
        # Reproduce the amended zero-boundary contrast independently using an
        # integer per-text difference and direct indexed sums, not the report's
        # large event tensor or weighted matrix multiplication.
        first = frames[("ratio_25", SEEDS[0], "clean")]
        labels = first.label_id.to_numpy(dtype=np.int64)
        difference = np.zeros(len(labels), dtype=np.int64)
        for seed in SEEDS:
            candidate = frames[("ratio_25", seed, "clean")]
            reference = frames[("ratio_50", seed, "clean")]
            a = h.booleans(candidate[f"mahalanobis_{NEW}_reject"], "Targeted CI candidate").astype(np.int64)
            b = h.booleans(reference[f"mahalanobis_{NEW}_reject"], "Targeted CI reference").astype(np.int64)
            difference += (a - b) * (labels >= 0)
        strata = [np.flatnonzero(labels == label) for label in np.unique(labels)]
        rng = np.random.default_rng(20261022)
        draws = []
        for start in range(0, 2000, 64):
            size = min(64, 2000 - start)
            sampled = np.concatenate([rng.choice(indices, size=(size, len(indices)), replace=True) for indices in strata], axis=1)
            draws.append(difference[sampled].sum(axis=1, dtype=np.int64) / 13500)
        low, high = np.quantile(np.concatenate(draws), [.025, .975])
        focus = intervals[(intervals.contrast_type == "cross_arm") & (intervals.candidate == "ratio_25")
            & (intervals.reference == "ratio_50") & (intervals.condition == "clean")
            & (intervals.metric == "in_scope_false_rejection_rate") & (intervals.estimand == "aligned_gap")]
        require(len(focus) == 1, "Targeted CI contrast identity")
        row = focus.iloc[0]
        h.close(int(difference.sum()) / 13500, row.estimate, "Targeted CI integer-count point", atol=1e-15, rtol=1e-12)
        h.close(low, row.ci_low, "Targeted CI independently resampled lower endpoint", atol=1e-15, rtol=1e-12)
        h.close(high, row.ci_high, "Targeted CI independently resampled upper endpoint", atol=1e-15, rtol=1e-12)
        require(float(high) == float(row.ci_high) == float(row.ci_high_pp) == 0., "Targeted CI upper endpoint must be exact zero")
        interval_check["targeted_zero_boundary_audit"] = dict(status="PASS", candidate="ratio_25", reference="ratio_50",
            condition="clean", metric="in_scope_false_rejection_rate", estimand="aligned_gap", draws=2000,
            random_seed=20261022, estimate=float(row.estimate), independently_resampled_ci_low=float(low),
            independently_resampled_ci_high=float(high), exact_zero_upper_boundary=True,
            method="Shared original-text stratified samples; sum integer rejection differences across three fixed seeds before division")
    return dict(status="PASS", checked=True, summary_rows=30, within_arm_changes=360, cross_arm_gaps=288,
                normal_states=90, policy_transitions=180, gap_change_old_minus_aligned_identity_verified=True,
                all_state_flow_denominators_verified=True, paired_intervals=interval_check,
                report_sha256={name: digest(output / name) for name in names})


def verify_results(output, state, m, h):
    cutoffs, joint_count, alignment_check = crosscheck_alignment(output, state, h)
    metrics = index_table(read_csv(output / "metrics_by_seed.csv"), CONDITIONS, "Test")
    validation = index_table(read_csv(output / "validation_metrics_by_seed.csv"), ("validation",), "Validation")
    old_test = read_csv(PRIOR / "outputs/metrics_by_seed.csv").set_index(["arm", "seed", "condition", "score", "policy"])
    old_val = read_csv(PRIOR / "outputs/validation_metrics_by_seed.csv").set_index(["arm", "seed", "condition", "score", "policy"])
    raw = read_json(PROJECT / "data/raw/data_full.json")
    mapping = read_json(PROJECT / "outputs/bert/seed42/label_mapping.json")["label_to_id"]
    checked_test = checked_val = 0
    direction_checks, frames, classifiers = [], {}, {}
    for arm in ARMS:
        for seed in SEEDS:
            folder = output / arm / f"seed{seed}"
            frozen = cutoffs[(arm, seed)]
            old_threshold, new_threshold = (frozen[f"mahalanobis/{p}"]["threshold"] for p in POLICIES)
            for condition in ("validation",) + CONDITIONS:
                context = f"{arm}/{seed}/{condition}"
                frame = read_csv(folder / f"{condition}_predictions.csv")
                old = read_csv(PRIOR / f"outputs/{arm}/seed{seed}/{condition}_predictions.csv")
                require(set(frame.columns) == set(old.columns) | {f"mahalanobis_{NEW}_reject"}, f"{context}: only aligned decision column added")
                m.same_columns(frame, old, list(old.columns), f"{context}: all original columns unchanged")
                rows = raw["val"] + raw["oos_val"] if condition == "validation" else raw["test"] + raw["oos_test"]
                labels = np.array([mapping.get(y, -1) for _, y in rows], dtype=np.int64)
                require(np.array_equal(frame.row_id.to_numpy(), np.arange(len(rows))) and np.array_equal(frame.label_id.to_numpy(), labels), f"{context}: original row/label order")
                require(frame.true_label.tolist() == [r[1] for r in rows], f"{context}: original labels")
                scope = labels >= 0
                is_n, oos_n = (3000, 100) if condition == "validation" else (4500, 1000)
                require(np.array_equal(h.booleans(frame.is_oos, context), ~scope) and int(scope.sum()) == is_n
                        and int((~scope).sum()) == oos_n and np.array_equal(np.bincount(labels[scope]), np.full(150, is_n // 150)), f"{context}: IS/OOS counts")
                predicted = frame.predicted_label_id.to_numpy(dtype=np.int64)
                key = (seed, condition)
                if key not in classifiers:
                    classifiers[key] = predicted
                require(np.array_equal(classifiers[key], predicted), f"{context}: classifier changed across arms")
                scores = frame.mahalanobis_score.to_numpy(dtype=np.float64)
                require(np.isfinite(scores).all(), f"{context}: finite cached scores")
                old_reject, new_reject = scores >= old_threshold, scores >= new_threshold
                if new_threshold < old_threshold:
                    require(not (old_reject & ~new_reject).any(), f"{context}: downward cutoff removed rejection")
                elif new_threshold > old_threshold:
                    require(not (~old_reject & new_reject).any(), f"{context}: upward cutoff added rejection")
                else:
                    require(np.array_equal(old_reject, new_reject), f"{context}: equal cutoff decisions differ")
                direction_checks.append(dict(arm=arm, seed=seed, condition=condition, threshold_delta=new_threshold - old_threshold,
                    added_rejections=int((~old_reject & new_reject).sum()), removed_rejections=int((old_reject & ~new_reject).sum()),
                    old_is_accepted=int((scope & ~old_reject).sum()), aligned_is_accepted=int((scope & ~new_reject).sum())))
                if condition == "validation":
                    require(int((scope & ~new_reject).sum()) == joint_count, f"{context}: exact aligned validation count")
                true_counts, pred_counts = np.bincount(labels[scope], minlength=150), np.bincount(predicted[scope], minlength=150)
                correct_counts = np.bincount(labels[scope & (labels == predicted)], minlength=150)
                closed_f1 = np.mean(np.divide(2 * correct_counts, true_counts + pred_counts, out=np.zeros(150), where=true_counts + pred_counts > 0))
                index = validation if condition == "validation" else metrics
                previous = old_val if condition == "validation" else old_test
                selected = index.loc[(arm, seed, condition)]
                for policy in POLICIES:
                    threshold = frozen[f"mahalanobis/{policy}"]["threshold"]
                    rejected = scores >= threshold
                    require(np.array_equal(h.booleans(frame[f"mahalanobis_{policy}_reject"], context), rejected), f"{context}/{policy}: >= decisions")
                    row = selected.loc[("mahalanobis", policy)]
                    require(float(row.threshold) == threshold, f"{context}: frozen cutoff differs")
                    expected = h.manual_metrics(labels, predicted, ~scope, rejected, scores)
                    expected.update(closed_set_accuracy=float((predicted[scope] == labels[scope]).mean()), closed_set_macro_f1=float(closed_f1))
                    for field, value in expected.items():
                        h.close(value, row[field], f"{context}/{policy}/{field}")
                    if policy == OLD:
                        old_row = previous.loc[(arm, seed, condition, "mahalanobis", OLD)]
                        for field in row.index:
                            h.close(row[field], old_row[field], f"{context}: exact old metric/{field}", atol=0, rtol=0)
                m.compare_local_metrics(folder / f"{condition}_metrics.csv", selected, h, context)
                if condition == "validation":
                    checked_val += 2
                else:
                    checked_test += 2
                    frames[(arm, seed, condition)] = frame
    reports = verify_reports(output, metrics, frames, h)
    require(digest(ROOT / "INPUTS_FROZEN.json") == state["input_hash"] and digest(output / "DEVELOPMENT_FROZEN.json") == state["development_hash"], "Manifests changed during verification")
    m.verify_hashes(state["frozen"]["input_hashes"], "Ending input hashes")
    m.verify_hashes(state["development"]["development_hashes"], "Ending development hashes")
    m.verify_hashes(state["complete"]["output_hashes"], "Ending test hashes")
    return dict(verified_test_settings=checked_test, verified_validation_settings=checked_val,
        test_prediction_rows_checked=45 * 5500, validation_prediction_rows_checked=15 * 3100,
        alignment_check=alignment_check, all_original_columns_and_scores_exactly_unchanged=True,
        classifier_identical_across_arms=True, old_policy_decisions_and_metrics_exactly_reproduced=True,
        manual_confusion_metrics_and_closed_metrics_verified=True, threshold_direction_and_decision_sets=direction_checks,
        report_checks=reports, hash_inventory_counts=state["hash_counts"], inputs_manifest_sha256=state["input_hash"],
        development_manifest_sha256=state["development_hash"], test_completion_sha256=digest(output / "TEST_COMPLETE.json"),
        metrics_sha256=digest(output / "metrics_by_seed.csv"),
        limitations=["Only saved scores and frozen artifacts are audited; no BERT inference or detector fitting is performed.",
            "IS-only cutoffs are independently reconstructed; the separate selection audit supplies additional source and candidate scrutiny.",
            "Exact validation acceptance alignment does not imply equal acceptance on clean or perturbed test inputs."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    output = args.output.resolve()
    required = [ROOT / "PROTOCOL.json", ROOT / "INPUTS_FROZEN.json", output / "DEVELOPMENT_FROZEN.json", output / "TEST_COMPLETE.json", output / "metrics_by_seed.csv"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print(json.dumps(dict(status="NOT_READY", missing=missing), ensure_ascii=False, indent=2))
        return 2
    started = datetime.now(timezone.utc).isoformat()
    try:
        m, h, helper_hashes = load_helpers()
        state = verify_gates(output, m)
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
