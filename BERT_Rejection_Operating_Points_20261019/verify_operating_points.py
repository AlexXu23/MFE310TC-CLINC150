"""Independent verifier for a frozen-score IS-acceptance operating-point grid.

No producer runner/report, estimator, model inference or refitting is imported.
The only write is outputs/verification.json. All preceding studies are read-only.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
for thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[thread_variable] = "4"
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ALIGNMENT = ROOT.parent / "BERT_Rejection_Alignment_20261019"
RATIO = ROOT.parent / "BERT_Rejection_Ratio_20261019"
PROJECT = ROOT.parent / "CLINC150_Team_20261019"
ARMS = ("clean_reference", "ratio_00", "ratio_10", "ratio_25", "ratio_50")
SEEDS = (42, 43, 44)
CONDITIONS = ("clean", "typo_light", "typo_medium")
TARGETS = (("aligned_is_acceptance_90", 2700), ("aligned_is_acceptance_92p5", 2775),
           ("aligned_is_acceptance_95", 2850), ("aligned_is_acceptance_97p5", 2925),
           ("aligned_is_acceptance_99", 2970))
POLICIES = tuple(policy for policy, _ in TARGETS)


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
    spec = importlib.util.spec_from_file_location("independent_mixed_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base, hashes = module.helpers()
    hashes[path.resolve().as_posix()] = digest(path)
    return module, base, hashes


def index_metrics(frame, conditions, context):
    keys = ["arm", "seed", "condition", "score", "policy"]
    expected = {(a, s, c, "mahalanobis", p) for a in ARMS for s in SEEDS for c in conditions for p in POLICIES}
    require(not frame.duplicated(keys).any() and set(frame[keys].itertuples(index=False, name=None)) == expected,
            f"{context}: expected {len(expected)} unique operating settings")
    return frame.set_index(keys).sort_index()


def independently_rebuild_is_candidates(h):
    options, common = {}, None
    for arm in ARMS:
        for seed in SEEDS:
            frame = read_csv(ALIGNMENT / f"outputs/{arm}/seed{seed}/validation_predictions.csv")
            is_mask = ~h.booleans(frame.is_oos, "IS-only candidate source")
            require(int(is_mask.sum()) == 3000 and int((~is_mask).sum()) == 100, "Validation scope")
            scores = frame.mahalanobis_score.to_numpy(dtype=np.float64)[is_mask]
            require(np.isfinite(scores).all(), "Finite IS validation scores")
            values, multiplicities = np.unique(scores, return_counts=True)
            thresholds = np.r_[values, np.nextafter(values[-1], np.inf)]
            accepted = np.r_[0, np.cumsum(multiplicities)]
            options[(arm, seed)] = (thresholds, accepted)
            attainable = set(map(int, accepted))
            common = attainable if common is None else common & attainable
    chosen_counts = {policy: min(count for count in common if count >= requested) for policy, requested in TARGETS}
    chosen_thresholds = {(arm, seed, policy): float(thresholds[counts == chosen_counts[policy]].min())
        for (arm, seed), (thresholds, counts) in options.items() for policy in POLICIES}
    return options, common, chosen_counts, chosen_thresholds


def verify_gates(output, m, h):
    protocol = read_json(ROOT / "PROTOCOL.json")
    expected_targets = [dict(policy=p, target_acceptance=n / 3000, target_accepted=n) for p, n in TARGETS]
    require(protocol["arms"] == list(ARMS) and protocol["seeds"] == list(SEEDS) and protocol["conditions"] == list(CONDITIONS)
            and protocol["targets"] == expected_targets, "Protocol declared experiment grid")
    require(protocol["bootstrap"]["repeats"] == 2000 and protocol["bootstrap"]["seed"] == 20261023, "Bootstrap protocol")
    frozen = read_json(ROOT / "INPUTS_FROZEN.json")
    development = read_json(output / "DEVELOPMENT_FROZEN.json")
    complete = read_json(output / "TEST_COMPLETE.json")
    input_hash, dev_hash = digest(ROOT / "INPUTS_FROZEN.json"), digest(output / "DEVELOPMENT_FROZEN.json")
    require(set(frozen) == {"frozen_at", "execution", "input_hashes"}, "Frozen input schema")
    require(development["inputs_frozen_sha256"] == complete["inputs_frozen_sha256"] == input_hash, "Input bindings")
    require(complete["development_frozen_sha256"] == dev_hash, "Development binding")
    require(development["existing_detectors"] == 15 and development["total_thresholds"] == 75
            and development["selection_used_oos"] is False, "Development counts/IS-only rule")
    require(complete["test_settings"] == 225 and complete["original_models_detectors_features_scores_unchanged"] is True
            and complete["anchor_95_decisions_identical"] is True, "Test count/invariants")
    time = datetime.fromisoformat
    require(time(frozen["frozen_at"]) <= time(development["frozen_at"]) <= time(complete["started_at"]) <= time(complete["completed_at"]), "All cutoffs frozen before test")
    counts = {"inputs": m.verify_hashes(frozen["input_hashes"], "Input hashes"),
        "development": m.verify_hashes(development["development_hashes"], "Development hashes"),
        "test": m.verify_hashes(complete["output_hashes"], "Test hashes")}
    prior = [read_json(ALIGNMENT / "INPUTS_FROZEN.json"), read_json(ALIGNMENT / "outputs/DEVELOPMENT_FROZEN.json"), read_json(ALIGNMENT / "outputs/TEST_COMPLETE.json")]
    require(prior[2]["inputs_frozen_sha256"] == prior[1]["inputs_frozen_sha256"] == digest(ALIGNMENT / "INPUTS_FROZEN.json"), "Prior input chain")
    require(prior[2]["development_frozen_sha256"] == digest(ALIGNMENT / "outputs/DEVELOPMENT_FROZEN.json"), "Prior development chain")
    dependencies = [ROOT / "PROTOCOL.json", ROOT / "run_operating_points.py", ALIGNMENT / "INPUTS_FROZEN.json",
                    ALIGNMENT / "outputs/DEVELOPMENT_FROZEN.json", ALIGNMENT / "outputs/TEST_COMPLETE.json"]
    for document, field in zip(prior, ("input_hashes", "development_hashes", "output_hashes")):
        m.verify_hashes(document[field], f"Prior {field}")
        dependencies += [Path(path) for path in document[field]]
    m.required_hashes(dependencies, frozen["input_hashes"], "Complete inherited inventory")
    m.required_hashes([output / name for name in ("alignment_summary.json", "operating_points.csv", "validation_metrics_by_seed.csv")], development["development_hashes"], "Global development files")
    m.required_hashes([output / "metrics_by_seed.csv"], complete["output_hashes"], "Global test file")
    for arm in ARMS:
        for seed in SEEDS:
            folder = output / arm / f"seed{seed}"
            local = read_json(folder / "DEVELOPMENT_COMPLETE.json")
            require(local["arm"] == arm and local["seed"] == seed and local["inputs_frozen_sha256"] == input_hash, "Local identity/binding")
            detector = RATIO / f"outputs/{arm}/seed{seed}/distance_model.npz"
            require(Path(local["detector_path"]).resolve() == detector.resolve() and local["detector_sha256"] == digest(detector), "Unchanged source detector")
            require(time(frozen["frozen_at"]) <= time(local["completed_at"]) <= time(development["frozen_at"]), "Local freeze timing")
            m.verify_hashes(local["output_hashes"], f"{arm}/{seed}: local development")
            paths = [folder / name for name in ("frozen_thresholds.json", "validation_is_threshold_candidates.csv", "validation_predictions.csv", "validation_metrics.csv")]
            m.required_hashes(paths, local["output_hashes"], "Local development inventory")
            m.required_hashes(paths + [folder / "DEVELOPMENT_COMPLETE.json"], development["development_hashes"], "Global development inventory")
            m.required_hashes([folder / f"{c}_{kind}.csv" for c in CONDITIONS for kind in ("predictions", "metrics")], complete["output_hashes"], "Test decision inventory")
    return dict(frozen=frozen, development=development, complete=complete, input_hash=input_hash, development_hash=dev_hash,
                hash_inventory_counts=counts)


def verify_cutoffs(output, state, h):
    options, common, chosen_counts, chosen_thresholds = independently_rebuild_is_candidates(h)
    summary = read_json(output / "alignment_summary.json")
    require(summary["selection_used_oos"] is False and summary["validation_is_rows"] == 3000, "IS-only selection summary")
    require(summary["common_feasible_counts"] == sorted(common) and summary["common_feasible_count_count"] == len(common), "Common attainable-count set")
    expected_targets = [dict(policy=p, target_acceptance=n / 3000, target_accepted=n,
        actual_common_accepted=chosen_counts[p], actual_common_acceptance=chosen_counts[p] / 3000) for p, n in TARGETS]
    require(summary["targets"] == state["development"]["targets"] == expected_targets, "Requested and actual common targets")
    table = read_csv(output / "operating_points.csv")
    keys = ["arm", "seed", "policy"]
    expected_keys = {(a, s, p) for a in ARMS for s in SEEDS for p in POLICIES}
    require(not table.duplicated(keys).any() and set(table[keys].itertuples(index=False, name=None)) == expected_keys, "75 operating points")
    indexed = table.set_index(keys).sort_index()
    require(len(summary["models"]) == 75, "75 model/cutoff records")
    models = {(r["arm"], r["seed"], r["policy"]): r for r in summary["models"]}
    require(set(models) == expected_keys, "Summary operating point identities")
    all_cutoffs = {}
    for arm in ARMS:
        for seed in SEEDS:
            folder = output / arm / f"seed{seed}"
            local = read_json(folder / "DEVELOPMENT_COMPLETE.json")
            require(datetime.fromisoformat(summary["selected_at"]) <= datetime.fromisoformat(local["completed_at"]), "All selections precede validation completion")
            thresholds, counts = options[(arm, seed)]
            curve = read_csv(folder / "validation_is_threshold_candidates.csv")
            require(np.array_equal(curve.threshold.to_numpy(), thresholds) and np.array_equal(curve.accepted_is.to_numpy(), counts)
                    and curve.validation_is_rows.eq(3000).all(), "IS candidate values and grouped counts")
            frozen = read_json(folder / "frozen_thresholds.json")
            require(set(frozen) == {f"mahalanobis/{p}" for p in POLICIES}, "Five frozen policies")
            anchor = read_json(ALIGNMENT / f"outputs/{arm}/seed{seed}/frozen_thresholds.json")["mahalanobis/aligned_is_acceptance_95"]["threshold"]
            previous = None
            for policy, requested in TARGETS:
                chosen = chosen_thresholds[(arm, seed, policy)]
                actual = chosen_counts[policy]
                item = frozen[f"mahalanobis/{policy}"]
                require(item["threshold"] == chosen and item["target_acceptance"] == requested / 3000 and item["target_accepted"] == requested
                        and item["actual_accepted"] == actual and item["validation_is_rows"] == 3000 and item["in_scope_acceptance"] == actual / 3000, "Independent cutoff/target reconstruction")
                require(item["validation_predictions_sha256"] == digest(ALIGNMENT / f"outputs/{arm}/seed{seed}/validation_predictions.csv"), "Cutoff cached validation binding")
                require(previous is None or previous <= chosen, "Acceptance targets require monotone thresholds")
                previous = chosen
                if policy == "aligned_is_acceptance_95":
                    require(chosen == anchor, "95 percent cutoff regression")
                expected = dict(target_acceptance=requested / 3000, target_accepted=requested, actual_accepted=actual,
                    actual_acceptance=actual / 3000, threshold=chosen, unique_is_scores=len(thresholds) - 1,
                    candidate_count=len(thresholds), anchor_95_threshold=anchor, threshold_minus_95=chosen - anchor)
                for field, value in expected.items():
                    h.close(value, indexed.loc[(arm, seed, policy), field], f"Operating points/{arm}/{seed}/{policy}/{field}")
                    h.close(value, models[(arm, seed, policy)][field], f"Summary/{arm}/{seed}/{policy}/{field}")
            all_cutoffs[(arm, seed)] = frozen
    return all_cutoffs, chosen_counts, dict(verified_cutoffs=75, independent_is_candidate_count=sum(len(v[0]) for v in options.values()),
        common_attainable_counts=len(common), targets=expected_targets, anchor_95_thresholds_identical=True,
        all_five_requested_acceptances_exact=all(chosen_counts[p] == n for p, n in TARGETS))


def verify_reports(output, metrics, frames, h):
    names = ("summary_mean_sd.csv", "gap_by_seed.csv", "gap_summary.csv", "clean_guardrails.csv",
             "stability_summary.csv", "primary_paired_intervals.csv", "ordering_checks.csv")
    missing = [name for name in names if not (output / name).is_file()]
    if missing:
        return dict(status="PENDING_REPORT", checked=False, missing=missing)
    measures = ("in_scope_false_rejection_rate", "in_scope_correct_and_accepted_rate", "in_scope_wrong_and_accepted_rate",
                "oos_false_acceptance_rate", "oos_f1", "oos_recall", "in_scope_acceptance", "combined_151_accuracy")
    denominators = dict(zip(measures, (4500, 4500, 4500, 1000, None, 1000, 4500, 5500)))
    candidates, references, anchor = ("ratio_10", "ratio_25"), ("ratio_00", "ratio_50"), POLICIES[2]
    target_map = {policy: count / 3000 for policy, count in TARGETS}
    raw_metrics = metrics.reset_index().copy()
    event_counts, numerators = {}, {}
    for (arm, seed, condition), frame in frames.items():
        oos = h.booleans(frame.is_oos, "Report OOS")
        correct = frame.predicted_label_id.to_numpy() == frame.label_id.to_numpy()
        for policy in POLICIES:
            reject = h.booleans(frame[f"mahalanobis_{policy}_reject"], "Report decision")
            tp, fp, ac, aw = (int(mask.sum()) for mask in (oos & reject, ~oos & reject,
                    ~oos & ~reject & correct, ~oos & ~reject & ~correct))
            key = (arm, seed, condition, policy)
            event_counts[key] = (tp, fp, ac, aw)
            numerators[key] = dict(zip(measures, (fp, ac, aw, 1000-tp, None, tp, 4500-fp, tp+ac)))
            mask = raw_metrics.arm.eq(arm) & raw_metrics.seed.eq(seed) & raw_metrics.condition.eq(condition) & raw_metrics.policy.eq(policy)
            raw_metrics.loc[mask, "report_correct_accepted_n"] = ac
            raw_metrics.loc[mask, "report_wrong_accepted_n"] = aw

    def grid(frame, keys, expected, context):
        require(not frame.duplicated(keys).any() and set(frame[keys].itertuples(index=False, name=None)) == set(expected), context)
        return frame.set_index(keys)

    def close(actual, expected, context):
        if pd.isna(expected):
            require(actual == "" or pd.isna(actual), context + ": expected undefined")
        else:
            h.close(actual, expected, context, atol=2e-13, rtol=2e-12)

    summary = read_csv(output / names[0])
    s_keys = ["arm", "condition", "score", "policy"]
    s_index = grid(summary, s_keys, ((a, c, "mahalanobis", p) for a in ARMS for c in CONDITIONS for p in POLICIES), "75 summary groups")
    numeric = [c for c in raw_metrics if c not in {*s_keys, "seed", "threshold"} and pd.api.types.is_numeric_dtype(raw_metrics[c])]
    for key, group in raw_metrics.groupby(s_keys, sort=False):
        row = s_index.loc[key]
        require(int(row.n_seeds) == 3 and set(group.seed) == set(SEEDS), "Three fixed models in summary")
        for field in numeric:
            values = group[field].to_numpy(dtype=float)
            close(row[field + "_mean"], values.mean(), f"Summary {key}/{field} mean")
            close(row[field + "_sd"], values.std(ddof=1), f"Summary {key}/{field} sample SD")

    def gap(candidate, reference, seed, condition, policy, measure):
        denominator = denominators[measure]
        if denominator:
            count = numerators[(candidate, seed, condition, policy)][measure] - numerators[(reference, seed, condition, policy)][measure]
            return count / denominator, count
        return float(metrics.loc[(candidate, seed, condition, "mahalanobis", policy), measure]
                     - metrics.loc[(reference, seed, condition, "mahalanobis", policy), measure]), None

    gap_raw = read_csv(output / names[1])
    g_keys = ["candidate", "reference", "seed", "condition", "policy", "metric"]
    expected_gap_keys = [(a, r, s, c, p, metric) for a in candidates for r in references for s in SEEDS
                         for c in CONDITIONS for p in POLICIES for metric in measures]
    g_index = grid(gap_raw, g_keys, expected_gap_keys, "1440 per-seed gaps")
    computed_gaps = {}
    for key in expected_gap_keys:
        a, r, s, c, p, metric = key
        value, count = gap(a, r, s, c, p, metric)
        value95, count95 = gap(a, r, s, c, anchor, metric)
        denominator = denominators[metric]
        change = (count-count95)/denominator if denominator else value-value95
        expected = dict(gap=value, gap_pp=value*100, gap_at95=value95, gap_at95_pp=value95*100,
                        gap_change_from95=change, gap_change_from95_pp=change*100,
                        gap_count_difference=count, gap95_count_difference=count95, rate_denominator=denominator)
        row = g_index.loc[key]
        require(row.score == "mahalanobis" and float(row.target_acceptance) == target_map[p], "Gap metadata")
        for name, number in expected.items():
            close(row[name], np.nan if number is None else number, f"Gap {key}/{name}")
        if p == anchor:
            require(float(row.gap_change_from95) == 0 and float(row.gap_change_from95_pp) == 0, "Anchor gap change identically zero")
        computed_gaps[key] = expected

    gap_summary = read_csv(output / names[2])
    gs_keys = ["candidate", "reference", "condition", "policy", "metric"]
    expected_groups = [(a, r, c, p, metric) for a in candidates for r in references for c in CONDITIONS for p in POLICIES for metric in measures]
    gs_index = grid(gap_summary, gs_keys, expected_groups, "480 gap summary groups")
    mean_gaps = {}
    for key in expected_groups:
        a, r, c, p, metric = key
        records = [computed_gaps[(a, r, s, c, p, metric)] for s in SEEDS]
        row = gs_index.loc[key]
        require(int(row.n_seeds) == 3, "Gap summary seed count")
        denominator = denominators[metric]
        for field in records[0]:
            values = np.array([np.nan if record[field] is None else record[field] for record in records], dtype=float)
            mean = values.mean()
            if denominator and field in ("gap", "gap_pp", "gap_at95", "gap_at95_pp", "gap_change_from95", "gap_change_from95_pp"):
                count = sum(record["gap_count_difference"] for record in records)
                count95 = sum(record["gap95_count_difference"] for record in records)
                mean = (count95 if field.startswith("gap_at95") else count-count95 if field.startswith("gap_change") else count) / (3*denominator)
                if field.endswith("_pp"):
                    mean *= 100
            close(row[field+"_mean"], mean, f"Gap summary {key}/{field} mean")
            close(row[field+"_sd"], values.std(ddof=1), f"Gap summary {key}/{field} SD")
        mean_gaps[key] = dict(gap=float(row.gap_mean), gap_change_from95=float(row.gap_change_from95_mean))

    guard = read_csv(output / names[3])
    guard_keys = ["candidate", "reference", "policy", "metric"]
    guard_metrics = ("oos_f1", "oos_recall", "in_scope_acceptance")
    guard_refs = references + ("clean_reference",)
    guard_index = grid(guard, guard_keys, ((a,r,p,m) for a in candidates for r in guard_refs for p in POLICIES for m in guard_metrics), "90 clean guardrails")
    for a in candidates:
        for r in guard_refs:
            for p in POLICIES:
                flags = []
                for metric in guard_metrics:
                    denominator = denominators[metric]
                    if denominator:
                        before_count = sum(numerators[(r,s,"clean",p)][metric] for s in SEEDS)
                        after_count = sum(numerators[(a,s,"clean",p)][metric] for s in SEEDS)
                        before, after = before_count/(3*denominator), after_count/(3*denominator)
                        change, flag = (after_count-before_count)/(3*denominator), after_count >= before_count
                    else:
                        before, after = [float(np.mean([metrics.loc[(arm,s,"clean","mahalanobis",p),metric] for s in SEEDS])) for arm in (r,a)]
                        change, flag = after-before, after >= before
                    row = guard_index.loc[(a,r,p,metric)]
                    require(row.condition == "clean" and float(row.target_acceptance) == target_map[p], "Guardrail same-condition/target")
                    for field, value in (("reference_mean",before),("candidate_mean",after),("delta_pp",100*change)):
                        close(row[field],value,f"Guardrail {a}/{r}/{p}/{metric}/{field}")
                    require(bool(h.booleans(pd.Series([row.mean_non_decline]), "Guardrail flag")[0]) == flag, "Strict guardrail direction uses exact rate counts")
                    if denominator and change == 0:
                        require(float(row.delta_pp) == 0, "Mathematically equal rate must report zero")
                    flags.append(flag)
                for metric in guard_metrics:
                    require(bool(h.booleans(pd.Series([guard_index.loc[(a,r,p,metric)].all_three_pass]),"Three metric guardrail")[0]) == all(flags), "Joint clean guardrail")

    intervals = read_csv(output / names[5])
    ci_keys = ["candidate", "reference", "condition", "policy", "metric", "estimand"]
    expected_ci = [("ratio_25",r,c,p,m,e) for r in references for c in CONDITIONS for p in POLICIES for m in measures for e in ("gap","gap_change_from95")]
    ci_index = grid(intervals, ci_keys, expected_ci, "480 prespecified primary intervals")
    bootstrap = independently_bootstrap(frames, h, measures, denominators)
    zero_anchor_intervals = 0
    for key in expected_ci:
        a,r,c,p,metric,estimand = key
        row = ci_index.loc[key]
        require(int(row.bootstrap_repeats)==2000 and int(row.bootstrap_seed)==20261023 and int(row.n_is)==4500
                and int(row.n_oos)==1000 and int(row.n_fixed_models)==3, "Fixed-model stratified bootstrap metadata")
        estimate = mean_gaps[(a,r,c,p,metric)][estimand]
        low, high = bootstrap[(r,c,p,metric,estimand)]
        for field, value in (("estimate",estimate),("ci_low",low),("ci_high",high)):
            close(row[field],value,f"Independent bootstrap {key}/{field}")
            close(row[field+"_pp"],100*value,f"Bootstrap pp conversion {key}/{field}")
            if value == 0 and (field != "estimate" or denominators[metric]):
                require(float(row[field]) == 0 and float(row[field+"_pp"]) == 0, "Exact zero bootstrap boundary")
        require(float(row.ci_low) <= float(row.ci_high), "Ordered confidence interval")
        if p == anchor and estimand == "gap_change_from95":
            require(all(float(row[field]) == 0 for field in ("estimate","ci_low","ci_high")), "95% anchor bootstrap contrast is identically zero")
            zero_anchor_intervals += 1

    stability = read_csv(output / names[4])
    st_keys = ["candidate", "reference", "condition", "metric"]
    st_index = grid(stability,st_keys,((a,r,c,m) for a in candidates for r in references for c in CONDITIONS for m in measures),"96 stability records")
    for key,row in st_index.iterrows():
        a,r,c,metric=key
        values=np.array([mean_gaps[(a,r,c,p,metric)]["gap"]*100 for p in POLICIES])
        changes=np.array([mean_gaps[(a,r,c,p,metric)]["gap_change_from95"]*100 for p in POLICIES])
        direction="all_positive" if (values>0).all() else "all_negative" if (values<0).all() else "all_zero" if (values==0).all() else "includes_zero_or_changes_sign"
        require(row.direction_across_targets==direction,"Point-estimate direction across all five targets")
        close(row.minimum_gap_pp,values.min(),"Minimum signed gap")
        close(row.maximum_gap_pp,values.max(),"Maximum signed gap")
        close(row.max_absolute_change_from95_pp,np.abs(changes).max(),"Largest change from95")
        ci_direction="not_bootstrapped_10_percent_supplement"
        if a=="ratio_25":
            selected=[ci_index.loc[(a,r,c,p,metric,"gap")] for p in POLICIES]
            ci_direction="all_above_zero" if all(float(t.ci_low)>0 for t in selected) else "all_below_zero" if all(float(t.ci_high)<0 for t in selected) else "not_same_strict_interval_direction_at_every_target"
        require(row.interval_direction==ci_direction,"Interval direction across every target")

    ordering_result = verify_ordering_table(output / names[6], numerators, h)
    return dict(status="PASS",checked=True,summary_groups=75,per_seed_gaps=1440,gap_summary_groups=480,
                strict_clean_guardrails=90,stability_records=96,primary_intervals=480,
                all_480_bootstrap_quantiles_independently_resampled=True,bootstrap_repeats=2000,bootstrap_seed=20261023,
                identically_zero_95_change_intervals=zero_anchor_intervals,ordering_checks=ordering_result,
                report_artifact_sha256={name:digest(output/name) for name in names})


def independently_bootstrap(frames, h, measures, denominators):
    """Independent count-based resampling, without any report implementation imports."""
    arms=("ratio_00","ratio_25","ratio_50")
    labels=frames[(arms[0],42,"clean")].label_id.to_numpy()
    oos=labels<0
    events=np.empty((3,3,3,5,4,len(labels)),dtype=np.float64)
    for ai,arm in enumerate(arms):
        for si,seed in enumerate(SEEDS):
            for ci,condition in enumerate(CONDITIONS):
                frame=frames[(arm,seed,condition)]
                correct=frame.predicted_label_id.to_numpy()==labels
                for pi,policy in enumerate(POLICIES):
                    rejected=h.booleans(frame[f"mahalanobis_{policy}_reject"],"Bootstrap event")
                    events[ai,si,ci,pi]=np.stack((oos&rejected,~oos&rejected,~oos&~rejected&correct,~oos&~rejected&~correct))
    rng=np.random.default_rng(20261023)
    weights=np.empty((2000,len(labels)),dtype=np.float64)
    strata=[np.flatnonzero(labels==label) for label in np.unique(labels)]
    require([len(s) for s in strata]==[1000]+[30]*150,"Bootstrap original-text strata")
    for start in range(0,2000,64):
        size=min(64,2000-start)
        indices=np.concatenate([stratum[rng.integers(0,len(stratum),size=(size,len(stratum)))] for stratum in strata],axis=1)
        for offset,sample in enumerate(indices):
            weights[start+offset]=np.bincount(sample,minlength=len(labels))
    counts=(events.reshape(-1,len(labels)) @ weights.T).reshape(3,3,3,5,4,2000)
    require(np.equal(counts,np.rint(counts)).all(),"Bootstrap exact integer event sums")
    tp,fp,ac,aw=[counts[...,i,:] for i in range(4)]
    linear_counts=dict(zip(measures,(fp,ac,aw,1000-tp,None,tp,4500-fp,tp+ac)))
    f1=np.divide(2*tp,1000+tp+fp).mean(axis=1)
    results={}
    for reference in ("ratio_00","ratio_50"):
        ri=arms.index(reference)
        for ci,condition in enumerate(CONDITIONS):
            for mi,measure in enumerate(measures):
                denominator=denominators[measure]
                if denominator:
                    summed=linear_counts[measure].sum(axis=1)
                    delta=summed[1,ci]-summed[ri,ci]
                    distribution=delta/(3*denominator)
                    changed=(delta-delta[2])/(3*denominator)
                else:
                    distribution=f1[1,ci]-f1[ri,ci]
                    changed=distribution-distribution[2]
                for pi,policy in enumerate(POLICIES):
                    for estimand,values in (("gap",distribution[pi]),("gap_change_from95",changed[pi])):
                        results[(reference,condition,policy,measure,estimand)]=np.quantile(values,[.025,.975])
    return results


def verify_ordering_table(path, numerators, h):
    table=read_csv(path)
    require(len(table)==40,"40 noisy ordering records")
    expected={(scope,str(seed) if seed is not None else "",c,p) for c in CONDITIONS[1:] for p in POLICIES
              for scope,seed in [("seed",42),("seed",43),("seed",44),("mean",None)]}
    observed=set()
    exceptions=[]
    measures=(("fr","in_scope_false_rejection_rate",4500),
              ("wrong_is_accept","in_scope_wrong_and_accepted_rate",4500),
              ("oos_false_accept","oos_false_acceptance_rate",1000))
    target_map={p:n/3000 for p,n in TARGETS}
    for row in table.itertuples(index=False):
        seed="" if row.scope=="mean" else str(int(float(row.seed)))
        key=(row.scope,seed,row.condition,row.policy)
        require(key in expected and key not in observed,"Unique predeclared ordering record")
        observed.add(key)
        require(float(row.target_acceptance)==target_map[row.policy],"Ordering target")
        selected=SEEDS if row.scope=="mean" else (int(seed),)
        if row.scope=="mean":
            require(row.seed=="","Mean ordering has no invented seed")
        for prefix,metric,denominator in measures:
            counts={arm:sum(numerators[(arm,s,row.condition,row.policy)][metric] for s in selected)
                    for arm in ("ratio_00","ratio_25","ratio_50")}
            differences=(counts["ratio_25"]-counts["ratio_00"],counts["ratio_25"]-counts["ratio_50"])
            for reference,difference in zip(("00","50"),differences):
                actual=getattr(row,f"{prefix}_gap25_minus{reference}_pp")
                h.close(actual,100*difference/(len(selected)*denominator),"Noisy ordering signed gap",atol=2e-13,rtol=2e-12)
                if difference==0:
                    require(float(actual)==0,"Exact zero ordering gap")
            between=differences[0]*differences[1]<=0
            actual=bool(h.booleans(pd.Series([getattr(row,f"{prefix}_between_00_and_50")]),"Ordering flag")[0])
            require(actual==between,"Actual intermediate ordering including ties")
            if not between:
                exceptions.append(dict(scope=row.scope,seed=None if seed=="" else int(seed),condition=row.condition,
                                       policy=row.policy,metric=metric,count_differences=list(differences)))
    require(observed==expected,"Complete noisy ordering grid")
    return dict(status="PASS",rows=40,scalar_ordering_checks=120,exceptions=exceptions)


def verify_results(output, state, m, h):
    cutoffs, counts, cutoff_checks = verify_cutoffs(output, state, h)
    metrics = index_metrics(read_csv(output / "metrics_by_seed.csv"), CONDITIONS, "Test metrics")
    validation = index_metrics(read_csv(output / "validation_metrics_by_seed.csv"), ("validation",), "Validation metrics")
    old_test = read_csv(ALIGNMENT / "outputs/metrics_by_seed.csv").set_index(["arm", "seed", "condition", "score", "policy"])
    old_val = read_csv(ALIGNMENT / "outputs/validation_metrics_by_seed.csv").set_index(["arm", "seed", "condition", "score", "policy"])
    raw = read_json(PROJECT / "data/raw/data_full.json")
    mapping = read_json(PROJECT / "outputs/bert/seed42/label_mapping.json")["label_to_id"]
    checked_test = checked_val = monotonic_pairs = 0
    frames, shared_classifiers, monotonic_records = {}, {}, []
    for arm in ARMS:
        for seed in SEEDS:
            folder = output / arm / f"seed{seed}"
            frozen = cutoffs[(arm, seed)]
            for condition in ("validation",) + CONDITIONS:
                context = f"{arm}/{seed}/{condition}"
                frame = read_csv(folder / f"{condition}_predictions.csv")
                original = read_csv(ALIGNMENT / f"outputs/{arm}/seed{seed}/{condition}_predictions.csv")
                require(set(frame.columns) == set(original.columns) | {f"mahalanobis_{p}_reject" for p in POLICIES}, "Only prespecified rejection columns added")
                m.same_columns(frame, original, list(original.columns), f"{context}: exact original columns")
                rows = raw["val"] + raw["oos_val"] if condition == "validation" else raw["test"] + raw["oos_test"]
                labels = np.array([mapping.get(y, -1) for _, y in rows], dtype=np.int64)
                require(np.array_equal(frame.row_id.to_numpy(), np.arange(len(rows))) and np.array_equal(frame.label_id.to_numpy(), labels), f"{context}: original rows/labels")
                require(frame.true_label.tolist() == [r[1] for r in rows], f"{context}: labels unchanged")
                normal = labels >= 0
                n_is, n_oos = (3000, 100) if condition == "validation" else (4500, 1000)
                require(np.array_equal(h.booleans(frame.is_oos, context), ~normal) and int(normal.sum()) == n_is
                        and int((~normal).sum()) == n_oos and np.array_equal(np.bincount(labels[normal]), np.full(150, n_is // 150)), f"{context}: class/scope balance")
                pred = frame.predicted_label_id.to_numpy(dtype=np.int64)
                identity = (seed, condition)
                if identity not in shared_classifiers:
                    shared_classifiers[identity] = pred
                require(np.array_equal(pred, shared_classifiers[identity]), f"{context}: fixed classifier across arms")
                scores = frame.mahalanobis_score.to_numpy(dtype=np.float64)
                require(np.isfinite(scores).all(), "Finite cached scores")
                index = validation if condition == "validation" else metrics
                previous_index = old_val if condition == "validation" else old_test
                selected = index.loc[(arm, seed, condition)]
                local_metrics = read_csv(folder / f"{condition}_metrics.csv")
                require(len(local_metrics) == 5 and not local_metrics.duplicated(["score", "policy"]).any(), "Local five-policy grid")
                for local in local_metrics.to_dict("records"):
                    aggregate = selected.loc[(local["score"], local["policy"])]
                    for field, value in local.items():
                        if field not in ("arm", "seed", "condition", "score", "policy"):
                            h.close(value, aggregate[field], f"{context}: local aggregate/{field}")
                support, prediction_count = np.bincount(labels[normal], minlength=150), np.bincount(pred[normal], minlength=150)
                correct_count = np.bincount(labels[normal & (labels == pred)], minlength=150)
                closed_f1 = float(np.mean(np.divide(2 * correct_count, support + prediction_count, out=np.zeros(150), where=support + prediction_count > 0)))
                prior_accept, accepted_counts, cutoff_values = None, [], []
                for policy, requested in TARGETS:
                    cutoff = frozen[f"mahalanobis/{policy}"]["threshold"]
                    rejected = scores >= cutoff
                    require(np.array_equal(h.booleans(frame[f"mahalanobis_{policy}_reject"], context), rejected), f"{context}/{policy}: >= decision")
                    if prior_accept is not None:
                        require(not (prior_accept & rejected).any(), f"{context}: increasing target removed accepted sample")
                        monotonic_pairs += 1
                    prior_accept = ~rejected
                    accepted_counts.append(int((normal & ~rejected).sum()))
                    cutoff_values.append(cutoff)
                    require(len(cutoff_values) < 2 or cutoff_values[-1] >= cutoff_values[-2], "Threshold monotonicity")
                    if condition == "validation":
                        require(accepted_counts[-1] == counts[policy], "Exact common validation acceptance")
                    row = selected.loc[("mahalanobis", policy)]
                    expected = h.manual_metrics(labels, pred, ~normal, rejected, scores)
                    expected.update(closed_set_accuracy=float((pred[normal] == labels[normal]).mean()), closed_set_macro_f1=closed_f1,
                        in_scope_wrong_and_accepted_rate=int((normal & (pred != labels) & ~rejected).sum()) / n_is,
                        oos_false_acceptance_rate=int((~normal & ~rejected).sum()) / n_oos,
                        target_acceptance=requested / 3000, target_accepted=requested,
                        actual_validation_accepted=counts[policy], actual_validation_acceptance=counts[policy] / 3000)
                    require(float(row.threshold) == cutoff, "Frozen threshold in metrics")
                    for field, value in expected.items():
                        h.close(value, row[field], f"{context}/{policy}/{field}")
                    if policy == "aligned_is_acceptance_95":
                        old_row = previous_index.loc[(arm, seed, condition, "mahalanobis", policy)]
                        for field, value in old_row.items():
                            h.close(value, row[field], f"{context}: exact95 regression/{field}", atol=0, rtol=0)
                monotonic_records.append(dict(arm=arm, seed=seed, condition=condition, is_accepted_counts=accepted_counts,
                                              all_accepted_sets_nested=True))
                if condition == "validation":
                    checked_val += 5
                else:
                    checked_test += 5
                    frames[(arm, seed, condition)] = frame
    reports = verify_reports(output, metrics, frames, h)
    require(digest(ROOT / "INPUTS_FROZEN.json") == state["input_hash"] and digest(output / "DEVELOPMENT_FROZEN.json") == state["development_hash"], "Frozen manifests changed during verification")
    m.verify_hashes(state["frozen"]["input_hashes"], "Ending inputs")
    m.verify_hashes(state["development"]["development_hashes"], "Ending development")
    m.verify_hashes(state["complete"]["output_hashes"], "Ending test")
    return dict(verified_test_settings=checked_test, verified_validation_settings=checked_val,
        test_prediction_rows_checked=45 * 5500, validation_prediction_rows_checked=15 * 3100,
        cutoff_checks=cutoff_checks, every_original_column_exactly_unchanged=True, fixed_classifier_across_arms=True,
        anchor_95_decisions_and_metrics_exactly_reproduced=True, manual_confusion_and_wrong_acceptance_metrics_verified=True,
        acceptance_monotonicity_adjacent_pairs=monotonic_pairs, acceptance_monotonicity=monotonic_records,
        report_checks=reports, hash_inventory_counts=state["hash_inventory_counts"], inputs_manifest_sha256=state["input_hash"],
        development_manifest_sha256=state["development_hash"], test_completion_sha256=digest(output / "TEST_COMPLETE.json"),
        metrics_sha256=digest(output / "metrics_by_seed.csv"),
        limitations=["Saved scores/decisions and immutable source hashes are audited; no inference or refitting is performed.",
            "Exact common acceptance holds on validation; test acceptances can differ across detector arms.",
            "Accepted sets must expand with target acceptance; F1 and combined accuracy are not assumed monotone."])


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
        m, h, helper_hashes = load_helpers()
        state = verify_gates(output, m, h)
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
