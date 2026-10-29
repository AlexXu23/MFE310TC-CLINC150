"""Read frozen rejection results, verify them, and create a Chinese report.

No model inference, model/threshold selection, or training occurs here. Run only
after metrics_by_seed.csv and all three test_predictions.csv files exist:

    python report_rejection.py

Bootstrap units are matched original texts, jointly resampled across all three
seeds and methods, stratified by each of 150 IS intents plus the OOS partition.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


SEEDS = (42, 43, 44)
SCORES = ("msp", "entropy", "mahalanobis")
POLICIES = ("max_oos_f1", "is_acceptance_95")
SCORE_NAMES = {"msp": "MSP", "entropy": "Entropy", "mahalanobis": "Mahalanobis"}
POLICY_NAMES = {
    "max_oos_f1": "验证集 OOS F1 最大",
    "is_acceptance_95": "验证集 IS 接受率 ≥95%",
}
INTERVAL_METRICS = (
    "oos_f1", "oos_recall", "in_scope_acceptance", "combined_151_accuracy"
)
METRIC_NAMES = {
    "oos_f1": "OOS F1", "oos_recall": "OOS recall",
    "in_scope_acceptance": "正常请求接受率",
    "combined_151_accuracy": "系统准确率",
}
IDENTITY_COLUMNS = ("row_id", "text", "true_label", "label_id", "is_oos")


def booleans(values: pd.Series) -> np.ndarray:
    normalized = values.astype(str).str.lower().str.strip()
    if not normalized.isin(("true", "false", "1", "0")).all():
        raise ValueError(f"Invalid Boolean values in {values.name}")
    return normalized.isin(("true", "1")).to_numpy(dtype=bool)


def evaluate_predictions(labels, prediction, is_oos, scores, rejected):
    """Independent recomputation of the original evaluator's test metrics."""
    accepted_is = ~is_oos & ~rejected
    tp = int(np.sum(is_oos & rejected))
    fp = int(np.sum(~is_oos & rejected))
    fn = int(np.sum(is_oos & ~rejected))
    final = np.where(rejected, -1, prediction)
    return {
        "oos_precision": tp / (tp + fp) if tp + fp else 0.0,
        "oos_recall": tp / (tp + fn),
        "oos_f1": 2 * tp / (2 * tp + fp + fn),
        "oos_auroc": float(roc_auc_score(is_oos, scores)),
        "oos_aupr_average_precision": float(average_precision_score(is_oos, scores)),
        "in_scope_acceptance": float(accepted_is.sum() / (~is_oos).sum()),
        "in_scope_false_rejection_rate": float(fp / (~is_oos).sum()),
        "in_scope_correct_and_accepted_rate": float(
            ((prediction == labels) & accepted_is).sum() / (~is_oos).sum()
        ),
        "accepted_in_scope_accuracy": float((prediction[accepted_is] == labels[accepted_is]).mean())
        if accepted_is.any() else np.nan,
        "combined_151_accuracy": float((final == labels).mean()),
        "combined_151_macro_f1": float(f1_score(
            labels, final, labels=np.arange(-1, 150), average="macro", zero_division=0
        )),
        "oos_true_positive": tp, "in_scope_false_rejections": fp, "oos_missed": fn,
    }


def load_and_verify(output: Path):
    metrics_path = output / "metrics_by_seed.csv"
    metrics = pd.read_csv(metrics_path, float_precision="round_trip")
    required = {"seed", "split", "score", "policy", "threshold"}
    if not required.issubset(metrics):
        raise ValueError(f"Metrics missing fields: {required - set(metrics)}")
    metrics["seed"] = pd.to_numeric(metrics["seed"], errors="raise").astype(int)
    if metrics.duplicated(["seed", "split", "score", "policy"]).any():
        raise ValueError("Duplicate seed/split/score/policy rows")
    if set(metrics["seed"]) != set(SEEDS):
        raise ValueError("Exactly seeds 42, 43, 44 are required")
    if set(metrics["score"]) != set(SCORES) or set(metrics["policy"]) != set(POLICIES):
        raise ValueError("Unexpected method or policy set")
    test = metrics[metrics["split"] == "test"]
    if len(test) != len(SEEDS) * len(SCORES) * len(POLICIES):
        raise ValueError("Expected all 18 test result rows")
    inputs = [metrics_path]
    frames, canonical = {}, None
    for seed in SEEDS:
        path = output / f"seed{seed}" / "test_predictions.csv"
        frame = pd.read_csv(path, float_precision="round_trip")
        required_predictions = set(IDENTITY_COLUMNS) | {"predicted_label_id", "predicted_label"}
        required_predictions |= {f"{s}_score" for s in SCORES}
        required_predictions |= {f"{s}_{p}_reject" for s in SCORES for p in POLICIES}
        if not required_predictions.issubset(frame):
            raise ValueError(f"Seed {seed} missing fields: {required_predictions - set(frame)}")
        if frame[list(IDENTITY_COLUMNS)].isna().any().any() or frame["row_id"].duplicated().any():
            raise ValueError(f"Seed {seed} has null metadata or duplicate row IDs")
        frame["is_oos"] = booleans(frame["is_oos"])
        frame["label_id"] = pd.to_numeric(frame["label_id"], errors="raise").astype(int)
        frame["predicted_label_id"] = pd.to_numeric(frame["predicted_label_id"], errors="raise").astype(int)
        if canonical is None:
            canonical = frame[list(IDENTITY_COLUMNS)].reset_index(drop=True).copy()
        else:
            if set(frame["row_id"]) != set(canonical["row_id"]):
                raise ValueError("Test row sets differ across seeds")
            frame = frame.set_index("row_id").loc[canonical["row_id"]].reset_index()
            if not frame[list(IDENTITY_COLUMNS)].equals(canonical):
                raise ValueError("Original text or label metadata differs across seeds")
        labels = frame["label_id"].to_numpy()
        is_oos = frame["is_oos"].to_numpy()
        known_labels, counts = np.unique(labels[~is_oos], return_counts=True)
        if len(frame) != 5500 or is_oos.sum() != 1000:
            raise ValueError("Expected 4,500 IS and 1,000 OOS test rows")
        if not np.array_equal(known_labels, np.arange(150)) or not np.all(counts == 30):
            raise ValueError("Expected exactly 30 test rows for each of 150 IS intents")
        if not np.all(labels[is_oos] == -1):
            raise ValueError("OOS label IDs must be -1")
        mapping_rows = frame.loc[~is_oos, ["label_id", "true_label"]].drop_duplicates()
        if len(mapping_rows) != 150 or mapping_rows["true_label"].nunique() != 150:
            raise ValueError("Inconsistent intent names and IDs")
        mapping = dict(zip(mapping_rows["label_id"], mapping_rows["true_label"]))
        if not frame["predicted_label_id"].between(0, 149).all():
            raise ValueError("Predictions must remain original 150-class classifier outputs")
        if not frame["predicted_label_id"].map(mapping).equals(frame["predicted_label"]):
            raise ValueError("Prediction labels disagree with classifier label IDs")
        prediction = frame["predicted_label_id"].to_numpy()
        for score in SCORES:
            values = pd.to_numeric(frame[f"{score}_score"], errors="raise").to_numpy()
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite {score} score")
            for policy in POLICIES:
                selected = test[(test["seed"] == seed) & (test["score"] == score)
                                & (test["policy"] == policy)]
                if len(selected) != 1:
                    raise ValueError("Missing unique test method/policy row")
                row = selected.iloc[0]
                threshold = float(row["threshold"])
                if np.isnan(threshold):
                    raise ValueError("NaN threshold")
                column = f"{score}_{policy}_reject"
                rejected = booleans(frame[column])
                if not np.array_equal(rejected, values >= threshold):
                    raise ValueError(f"Stored rejection does not match frozen threshold: {seed}/{column}")
                frame[column] = rejected
                calculated = evaluate_predictions(labels, prediction, is_oos, values, rejected)
                for key, actual in calculated.items():
                    if key not in metrics:
                        raise ValueError(f"Metrics table missing original evaluator field {key}")
                    expected = float(row[key])
                    if not np.isclose(actual, expected, rtol=1e-8, atol=1e-10, equal_nan=True):
                        raise ValueError(f"Metric mismatch: {seed}/{score}/{policy}/{key}: {actual} != {expected}")
        frames[seed] = frame
        inputs.append(path)
    for key, group in metrics.groupby(["split", "score", "policy"]):
        if set(group["seed"]) != set(SEEDS):
            raise ValueError(f"Incomplete three-seed group: {key}")
    return metrics, frames, inputs


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    excluded = {"seed", "split", "score", "policy", "threshold"}
    columns = [c for c in metrics if c not in excluded
               and pd.api.types.is_numeric_dtype(metrics[c])]
    records = []
    for (split, score, policy), group in metrics.groupby(["split", "score", "policy"], sort=False):
        record = {"split": split, "score": score, "policy": policy, "n_seeds": len(group)}
        for column in columns:
            values = group[column].to_numpy(dtype=float)
            # Missing conditional metrics are retained as missing, never zero-filled.
            record[f"{column}_mean"] = float(np.mean(values))
            record[f"{column}_sd"] = float(np.std(values, ddof=1))
        records.append(record)
    return pd.DataFrame(records)


def events_to_metrics(counts: np.ndarray, n_is: int, n_oos: int) -> np.ndarray:
    """Counts end in [TP, FP, IS-correct-and-accepted, replicate]."""
    tp, fp, accepted_correct = (counts[..., i, :] for i in range(3))
    return np.stack((2 * tp / (tp + fp + n_oos), tp / n_oos,
                     1 - fp / n_is, (tp + accepted_correct) / (n_is + n_oos)), axis=-2)


def paired_bootstrap(frames, repeats=2000, random_seed=20261019, batch_size=64):
    if repeats < 2 or batch_size < 1:
        raise ValueError("Bootstrap needs at least two replicates and a positive batch size")
    first = frames[SEEDS[0]]
    labels = first["label_id"].to_numpy()
    is_oos = first["is_oos"].to_numpy(dtype=bool)
    n_is, n_oos = int((~is_oos).sum()), int(is_oos.sum())
    strata = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    if any(len(indices) == 0 for indices in strata):
        raise ValueError("Empty bootstrap stratum")
    # Shape: seed, method, policy, event, original text.
    events = np.zeros((len(SEEDS), len(SCORES), len(POLICIES), 3, len(first)), dtype=float)
    for si, seed in enumerate(SEEDS):
        prediction = frames[seed]["predicted_label_id"].to_numpy()
        correct = (prediction == labels) & ~is_oos
        for mi, score in enumerate(SCORES):
            for pi, policy in enumerate(POLICIES):
                rejected = frames[seed][f"{score}_{policy}_reject"].to_numpy(dtype=bool)
                events[si, mi, pi, 0] = rejected & is_oos
                events[si, mi, pi, 1] = rejected & ~is_oos
                events[si, mi, pi, 2] = ~rejected & correct
    point = events_to_metrics(events.sum(axis=-1)[..., None], n_is, n_oos).mean(axis=0)[..., 0]
    draws = np.empty((len(SCORES), len(POLICIES), len(INTERVAL_METRICS), repeats))
    rng = np.random.default_rng(random_seed)
    flattened = events.reshape(-1, len(first))
    for start in range(0, repeats, batch_size):
        size = min(batch_size, repeats - start)
        # The SAME resampled original rows are used by every seed and method.
        sample = np.concatenate([rng.choice(indices, size=(size, len(indices)), replace=True)
                                 for indices in strata], axis=1)
        offsets = len(first) * np.arange(size)[:, None]
        weights = np.bincount((sample + offsets).ravel(), minlength=size * len(first))
        weights = weights.reshape(size, len(first)).astype(float)
        counts = (flattened @ weights.T).reshape(*events.shape[:-1], size)
        # Compute nonlinear F1 within EACH seed before averaging across seeds.
        draws[..., start:start + size] = events_to_metrics(counts, n_is, n_oos).mean(axis=0)
    records = []
    distance_index = SCORES.index("mahalanobis")
    for reference in ("msp", "entropy"):
        reference_index = SCORES.index(reference)
        for pi, policy in enumerate(POLICIES):
            for ki, metric in enumerate(INTERVAL_METRICS):
                delta = draws[distance_index, pi, ki] - draws[reference_index, pi, ki]
                low, high = np.quantile(delta, [0.025, 0.975])
                estimate = point[distance_index, pi, ki] - point[reference_index, pi, ki]
                records.append({
                    "comparison": f"mahalanobis_minus_{reference}",
                    "comparison_role": "primary" if (
                        reference == "msp" and policy == "max_oos_f1" and metric == "oos_f1"
                    ) else "secondary",
                    "policy": policy, "metric": metric,
                    "estimate_delta": estimate, "ci_low": low, "ci_high": high,
                    "estimate_delta_pp": estimate * 100, "ci_low_pp": low * 100, "ci_high_pp": high * 100,
                    "bootstrap_repeats": repeats, "bootstrap_seed": random_seed,
                    "n_is": n_is, "n_oos": n_oos, "n_fixed_models": len(SEEDS),
                    "strata": "150 IS intents (30 rows each) plus 1000 OOS rows",
                    "scope": "fixed models and thresholds; shared text resampling across seeds and methods",
                })
    return pd.DataFrame(records)


def create_plots(summary, intervals, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#64748b", "#21918c", "#d17720")
    test = summary[summary["split"] == "test"].set_index(["score", "policy"])
    panels = (("oos_f1", "OOS F1 (%)"),
              ("in_scope_false_rejection_rate", "IS false rejection (%)"),
              ("combined_151_accuracy", "System accuracy (%)"))
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for pi, policy in enumerate(POLICIES):
        for ki, (metric, title) in enumerate(panels):
            ax = axes[pi, ki]
            means = [test.loc[(s, policy), f"{metric}_mean"] * 100 for s in SCORES]
            sd = [test.loc[(s, policy), f"{metric}_sd"] * 100 for s in SCORES]
            bars = ax.bar(range(3), means, yerr=sd, capsize=4, color=colors, width=.65)
            ax.set_xticks(range(3), [SCORE_NAMES[s] for s in SCORES], rotation=12)
            ax.set_ylabel(title)
            ax.set_title("Validation max OOS F1" if pi == 0 else "Validation IS acceptance >=95%")
            ax.set_ylim(0, min(105, max(10, max(m + e for m, e in zip(means, sd)) * 1.22)))
            ax.spines[["top", "right"]].set_visible(False)
            ax.bar_label(bars, labels=[f"{v:.2f}" for v in means], padding=5, fontsize=9)
    fig.suptitle("Frozen BERT rejection: test mean and sample SD across 3 seeds", fontsize=14)
    comparison_path = output / "rejection_comparison.png"
    fig.savefig(comparison_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ri, reference in enumerate(("msp", "entropy")):
        for pi, policy in enumerate(POLICIES):
            ax = axes[ri, pi]
            rows = intervals[(intervals["comparison"] == f"mahalanobis_minus_{reference}")
                             & (intervals["policy"] == policy)].set_index("metric").loc[list(INTERVAL_METRICS)]
            estimates = rows["estimate_delta_pp"].to_numpy()
            lows, highs = rows["ci_low_pp"].to_numpy(), rows["ci_high_pp"].to_numpy()
            positions = np.arange(len(rows))
            # Percentile intervals need not contain the point estimate; draw independently.
            ax.hlines(positions, lows, highs, color=colors[2], linewidth=2.5)
            ax.scatter(estimates, positions, color=colors[2], zorder=3)
            ax.axvline(0, color="#64748b", linestyle="--", linewidth=1)
            ax.set_yticks(positions, ["OOS F1", "OOS recall", "IS acceptance", "System accuracy"])
            ax.invert_yaxis()
            ax.set_xlabel("Mahalanobis minus reference (percentage points)")
            ax.set_title(f"vs {SCORE_NAMES[reference]} | " + ("max F1" if pi == 0 else "IS >=95%"))
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Paired 95% percentile intervals, conditional on 3 fixed models", fontsize=14)
    interval_path = output / "paired_intervals.png"
    fig.savefig(interval_path, dpi=180)
    plt.close(fig)
    return comparison_path, interval_path


def htable(headers, rows):
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in rows)
    return f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def mean_sd(row, name, percent=True):
    multiplier = 100 if percent else 1
    return f"{row[name + '_mean'] * multiplier:.2f} ± {row[name + '_sd'] * multiplier:.2f}"


def image_tag(path):
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img alt="{html.escape(path.stem)}" src="data:image/png;base64,{encoded}">'


def load_run_details(output, frames):
    """Read observed forward drift and training-statistic provenance."""
    details, paths = {}, []
    for seed in SEEDS:
        development_path = output / f"seed{seed}" / "development_summary.json"
        test_path = output / f"seed{seed}" / "test_drift.json"
        development = json.loads(development_path.read_text(encoding="utf-8"))
        test_drift = json.loads(test_path.read_text(encoding="utf-8"))
        if development["seed"] != seed:
            raise ValueError("Development-summary seed mismatch")
        if sum(development["class_counts"]) != development["training_rows"]:
            raise ValueError("Training-class counts do not match development summary")
        for split, drift in (("validation", development["validation_drift"]), ("test", test_drift)):
            changed = drift["changed_top1_row_ids"]
            if drift["changed_top1_count"] != len(changed) or len(set(changed)) != len(changed):
                raise ValueError(f"Inconsistent drift count/IDs for seed {seed}, {split}")
            if not 0 <= len(changed) <= drift["rows"] or any(i < 0 or i >= drift["rows"] for i in changed):
                raise ValueError("Invalid drift row IDs")
        if test_drift["rows"] != len(frames[seed]):
            raise ValueError("Drift report does not describe the same test population")
        details[seed] = {"development": development, "test_drift": test_drift}
        paths.extend((development_path, test_path))
    verification_path = output / "verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification["status"] != "PASS":
        raise ValueError("Independent verification did not pass")
    if verification["metrics_sha256"] != hashlib.sha256((output / "metrics_by_seed.csv").read_bytes()).hexdigest():
        raise ValueError("Independent verification refers to a different metrics file")
    details["verification"] = verification
    paths.append(verification_path)
    return details, paths


def build_html(metrics, frames, summary, intervals, plots, inputs, repeats, random_seed, run_details):
    test = summary[summary["split"] == "test"].set_index(["score", "policy"])
    primary = intervals[(intervals["comparison"] == "mahalanobis_minus_msp")
                        & (intervals["policy"] == "max_oos_f1") & (intervals["metric"] == "oos_f1")].iloc[0]
    delta = primary["estimate_delta_pp"]
    direction = "提高" if delta > 0 else "降低" if delta < 0 else "不变"
    interval_text = ("该条件区间高于 0" if primary["ci_low_pp"] > 0 else
                     "该条件区间低于 0" if primary["ci_high_pp"] < 0 else "该条件区间包含 0")
    headline = (f"主比较中，马氏距离相对 MSP 的 OOS F1 {direction} {abs(delta):.2f} 个百分点；"
                f"95% 条件区间 [{primary['ci_low_pp']:.2f}, {primary['ci_high_pp']:.2f}]，{interval_text}。")
    secondary_f1 = intervals[(intervals["comparison"] == "mahalanobis_minus_entropy")
                             & (intervals["metric"] == "oos_f1")].set_index("policy")
    entropy_max = secondary_f1.loc["max_oos_f1"]
    entropy_95 = secondary_f1.loc["is_acceptance_95"]
    max_distance, max_msp, max_entropy = [test.loc[(s, "max_oos_f1")]
                                        for s in ("mahalanobis", "msp", "entropy")]
    fixed_distance = test.loc[("mahalanobis", "is_acceptance_95")]
    fixed_entropy = test.loc[("entropy", "is_acceptance_95")]
    entropy_interval_statement = (
        "该区间包含 0，当前条件分析不能确认马氏距离的 OOS F1 高于 Entropy"
        if entropy_max["ci_low_pp"] <= 0 <= entropy_max["ci_high_pp"] else
        "该区间不包含 0，方向和大小限于这一冻结策略下的条件比较"
    )
    nuanced_conclusion = (
        f"在主策略“验证集 OOS F1 最大”下，马氏距离 F1 为 {max_distance['oos_f1_mean']*100:.2f}%，"
        f"MSP 为 {max_msp['oos_f1_mean']*100:.2f}%。但同一策略的 Entropy 为 {max_entropy['oos_f1_mean']*100:.2f}%，"
        f"距离 − Entropy 仅为 {entropy_max['estimate_delta_pp']:+.2f} 个百分点，"
        f"95% 条件区间 [{entropy_max['ci_low_pp']:+.2f}, {entropy_max['ci_high_pp']:+.2f}]；"
        f"{entropy_interval_statement}。"
        f"在另一预设策略“验证集 IS 接受率 ≥95%”下，马氏距离 F1 为 {fixed_distance['oos_f1_mean']*100:.2f}%，"
        f"Entropy 为 {fixed_entropy['oos_f1_mean']*100:.2f}%，差值 {entropy_95['estimate_delta_pp']:+.2f} 个百分点，"
        f"95% 条件区间 [{entropy_95['ci_low_pp']:+.2f}, {entropy_95['ci_high_pp']:+.2f}]；"
        f"对应测试正常接受率分别为 {fixed_distance['in_scope_acceptance_mean']*100:.2f}% 和 "
        f"{fixed_entropy['in_scope_acceptance_mean']*100:.2f}%。"
        "因此收益需要同时注明参照方法与阈值策略，不能概括成对所有方法和运行点全面优于。"
        "这项补充对照不改变原定的唯一主终点：验证集最大 F1 策略下，距离 − MSP 的测试 OOS F1。"
    )
    chunks = []
    for policy in POLICIES:
        rows = []
        for score in SCORES:
            row = test.loc[(score, policy)]
            rows.append([SCORE_NAMES[score], mean_sd(row, "oos_precision"), mean_sd(row, "oos_recall"),
                         mean_sd(row, "oos_f1"), mean_sd(row, "in_scope_acceptance"),
                         mean_sd(row, "in_scope_correct_and_accepted_rate"), mean_sd(row, "combined_151_accuracy")])
        chunks.append(f"<h3>{POLICY_NAMES[policy]}</h3>" + htable(
            ["分数", "OOS precision %", "OOS recall %", "OOS F1 %", "正常接受率 %", "正常正确且接受 %", "系统准确率 %"], rows))
        distance = test.loc[("mahalanobis", policy)]
        reference = test.loc[("msp", policy)]
        extra = distance["in_scope_false_rejections_mean"] - reference["in_scope_false_rejections_mean"]
        chunks.append(f"<p>在 4,500 条正常请求中，马氏距离平均误拒 {distance['in_scope_false_rejections_mean']:.2f} 条，"
                      f"MSP 为 {reference['in_scope_false_rejections_mean']:.2f} 条；差值 {extra:+.2f} 条。"
                      f"马氏距离正常请求误拒率为 {mean_sd(distance, 'in_scope_false_rejection_rate')}%。"
                      "这些是三个固定模型的平均计数，不是单次运行中的小数样本。</p>")
    ranking = htable(["分数", "OOS AUROC %", "OOS average precision %"], [
        [SCORE_NAMES[s], mean_sd(test.loc[(s, POLICIES[0])], "oos_auroc"),
         mean_sd(test.loc[(s, POLICIES[0])], "oos_aupr_average_precision")] for s in SCORES])
    ci_rows = [[("主终点：" if r.comparison_role == "primary" else "次分析：")
                + ("距离 − MSP" if r.comparison == "mahalanobis_minus_msp" else "距离 − Entropy"),
                POLICY_NAMES[r.policy], METRIC_NAMES[r.metric], f"{r.estimate_delta_pp:+.2f}",
                f"[{r.ci_low_pp:+.2f}, {r.ci_high_pp:+.2f}]"] for r in intervals.itertuples()]
    ci_table = htable(["比较", "阈值策略", "指标", "差值 pp", "95% 条件区间 pp"], ci_rows)
    classification = []
    for seed, frame in frames.items():
        known = ~frame["is_oos"].to_numpy()
        labels = frame.loc[known, "label_id"].to_numpy()
        pred = frame.loc[known, "predicted_label_id"].to_numpy()
        classification.append([seed, f"{(labels == pred).mean()*100:.2f}%",
                               f"{f1_score(labels, pred, average='macro')*100:.2f}%"])
    thresholds = metrics[metrics["split"] == "test"].sort_values(["seed", "score", "policy"])
    threshold_table = htable(["seed", "分数", "验证选择规则", "冻结阈值"], [
        [r.seed, SCORE_NAMES[r.score], POLICY_NAMES[r.policy], f"{r.threshold:.10g}"]
        for r in thresholds.itertuples()])
    provenance = htable(["输入文件", "SHA256"], [
        [str(path.relative_to(inputs[0].parent)), hashlib.sha256(path.read_bytes()).hexdigest()] for path in inputs])
    drift_rows, development_rows = [], []
    for seed in SEEDS:
        details = run_details[seed]
        development = details["development"]
        for split, drift in (("validation", development["validation_drift"]), ("test", details["test_drift"])):
            drift_rows.append([
                seed, split, f"{drift['changed_top1_count']} / {drift['rows']}",
                f"{100*drift['changed_top1_count']/drift['rows']:.4f}%",
                ", ".join(str(i) for i in drift["changed_top1_row_ids"]) or "无",
                f"{drift['max_abs_logit_difference']:.8f}",
            ])
        counts = development["class_counts"]
        development_rows.append([seed, development["feature_dim"], development["training_rows"],
                                 len(counts), f"{min(counts)}–{max(counts)}",
                                 f"{development['shrinkage']:.8f}"])
    drift_table = htable(["seed", "划分", "top-1 改变 / 总行数", "改变比例", "改变的 row_id", "最大 logit 绝对差"], drift_rows)
    development_table = htable(["seed", "特征维度", "训练样本", "已知类别", "每类训练数量范围", "Ledoit–Wolf shrinkage"], development_rows)
    validation_total = run_details[SEEDS[0]]["development"]["validation_drift"]["rows"]
    if validation_total != 3100:
        raise ValueError("Expected original validation split with 3,000 IS and 100 OOS rows")
    test_total = len(frames[SEEDS[0]])
    test_oos = int(frames[SEEDS[0]]["is_oos"].sum())
    verification = run_details["verification"]
    direct_checks = verification["distance_direct_solve_checks"]
    checked_rows = sum(len(check["row_ids"]) for check in direct_checks)
    max_distance_error = max(check["max_abs_distance_error"] for check in direct_checks)
    verification_text = (
        f"独立 verify_rejection.py 核验记录为 {verification['status']}：复算 {verification['verified_settings']} 个设置，"
        f"检查 {verification['validation_candidates_checked']:,} 个验证候选阈值及 "
        f"{verification['validation_operating_points']} 个最终验证运行点；对 {checked_rows} 个真实 CLS 向量"
        f"逐一计算到全部 150 个类中心的直接线性求解距离，最大绝对差为 {max_distance_error:.2e}。"
        "记录还确认原始 JSON 行对齐、共享分类头预测、train-only 距离统计和冻结清单/源文件哈希；"
        "本次报告再次核对该验证记录的指标文件 SHA256。"
    )
    generated = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M UTC+08:00")
    return f'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>冻结 BERT 的马氏距离拒识实验</title>
<style>
:root{{color-scheme:light;--ink:#153344;--muted:#536a75;--line:#d9e3e7;--accent:#b86622}}
*{{box-sizing:border-box}}body{{margin:0;background:#f3f6f7;color:var(--ink);font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1190px;margin:30px auto;padding:34px 42px;background:white;border:1px solid var(--line);border-radius:16px}}
h1{{font-size:30px;line-height:1.35;margin:0 0 12px}}h2{{font-size:23px;margin-top:36px}}h3{{font-size:18px;margin-top:26px}}
p{{margin:12px 0}}.meta{{color:var(--muted);font-size:13px}}.lead{{border-left:5px solid var(--accent);padding:14px 20px;background:#fff6eb;font-size:18px}}
.table-scroll{{overflow-x:auto;margin:16px 0}}table{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}}
th,td{{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}}th{{background:#eaf0f3;white-space:nowrap}}
td{{overflow-wrap:anywhere}}img{{max-width:100%;height:auto;border:1px solid var(--line);border-radius:8px;margin:14px 0}}
details{{margin-top:24px}}summary{{cursor:pointer;font-weight:650}}code{{background:#edf2f5;padding:2px 5px;border-radius:4px}}
@media(max-width:700px){{main{{padding:24px 18px;margin:0;border-radius:0}}h1{{font-size:25px}}}}
@media print{{body{{background:white}}main{{border:0;margin:0;padding:12px}}details{{display:block}}img,table{{break-inside:avoid}}}}
</style><main>
<p class="meta">MFE310TC · 探索性实验 · {generated}</p>
<h1>冻结 BERT：马氏距离能否改善未知请求拒识？</h1>
<p class="lead">{html.escape(headline)}</p>
<p>{html.escape(nuanced_conclusion)}</p>
<p>本次只替换拒识分数。原 BERT 分类器权重保持不变，正常意图由本次前向传播的原分类头 argmax 决定。
最近类别中心仅用于计算距离，不充当新的分类预测。是否更好需要同时看未知请求检出、正常请求误拒和系统正确率。</p>
<h2>比较范围与实验口径</h2>
<p>MSP、Entropy 和 Mahalanobis 使用每个 seed 同一次 FP32 前向得到的输出。
本实验运行口径为 PyTorch 2.5.1+cu121、Transformers 4.57.6；并非旧 FP16 实验日志的逐位复现。
因此三种方法之间使用本次结果作公平对照，不能把新旧环境差异当成拒识方法带来的收益。</p>
<p>马氏距离使用冻结 BERT 最后一层的 [CLS] 表示。仅用正常训练文本估计每类中心，
对类内残差拟合 Ledoit–Wolf 共享协方差，取到任一类中心的最小平方马氏距离作为异常分数。
训练统计只来自 train；两种阈值规则仅使用 validation；test 用于冻结系统的评价。
所有分数越大越倾向 OOS，拒识规则为 <code>score ≥ threshold</code>。</p>
<p>距离形式参考 <a href="https://arxiv.org/abs/2101.03778" target="_blank" rel="noopener noreferrer">原始研究（arXiv:2101.03778）</a>。
Ledoit–Wolf 是本实验预先规定的协方差估计选择，不声称与原论文实现完全相同，也没有根据本次测试结果更换估计器。</p>
<p>这是在既有测试结果已被查看后提出的探索性新实验，不是全新未曝光确认集。
原温度缩放实验的负结果保留，本实验不会替换其主终点或把它改写成成功。
唯一主终点是“验证集 OOS F1 最大”策略下，距离 − MSP 的测试 OOS F1。
其余所有指标和策略（包括 95% 接受率策略及距离 − Entropy）均为次分析；两个预设阈值策略全部报告，不按 test 结果挑选。</p>
<h3>与课程内容的连接</h3>
<p>BERT 的表示学习把文本映射成向量；同类训练文本的中心描述该意图的典型位置；类内协方差描述不同方向的变化尺度及相互关系。
马氏距离因此衡量“当前表示离已知类分布有多远”，再由验证集选择的阈值转成接收或拒绝的决策。
这条链是“文本表示 → 类中心与类内协方差 → 异常分数 → 验证集阈值”。
与之相比，softmax 只表示 150 个已知类别之间的相对偏好；这些概率总和为 1，
并没有额外的“未知类概率”。因此最大概率高并不自动证明请求属于支持范围。</p>
<p>验证集 OOS 为 100/{validation_total:,}（{100/validation_total*100:.2f}%），
测试集 OOS 为 {test_oos:,}/{test_total:,}（{test_oos/test_total*100:.2f}%）。
OOS precision 和 F1 会随这一先验比例变化，因此在验证集使 F1 最大的阈值不一定在测试集仍是同一最优运行点。
但本实验没有单独固定其他因素、只改变混合比例；两种策略的差异不能全部归因于比例变化，
还可能涉及验证样本的有限数量、分数分布与两种选择目标的差异。测试集不参与重新选阈值。</p>
<h2>测试结果：拒识收益与正常请求代价</h2>
<p>测试集为 4,500 条正常请求与 1,000 条 OOS。表格为 seeds 42/43/44 的均值 ± 样本标准差（ddof=1），
百分比指标的标准差单位是百分点。OOS F1 在完整混合测试集上计算；正常正确且接受率以全部正常请求为分母；
系统准确率要求 OOS 被拒绝，或正常请求被接收且原分类器意图预测正确。</p>
{''.join(chunks)}
<p><strong>95% 是验证集上的正常接受率约束，不能保证 test、扰动输入或真实部署也达到 95%。</strong>
OOS precision 和 F1 依赖 IS/OOS 混合比例，不应把本测试集的收益直接等同于部署收益。</p>
{image_tag(plots[0])}
<h3>与具体阈值无关的排序表现</h3>{ranking}
<p>AUROC 和 average precision 描述分数排序；F1、误拒率和系统准确率描述冻结阈值下的实际决策。
三类证据要分别解释，不用排序指标代替正常请求代价。</p>
<h2>配对统计证据</h2>
<p>进行 {repeats:,} 次配对 bootstrap，随机种子为 {random_seed}。
每次在 150 个 IS intent 中各有放回抽取 30 条，再在 OOS 中抽取 1,000 条。
同一原始文本的抽样权重共同应用于三个 seed、三种分数和两种策略。
先为每个固定模型单独计算指标，再求三模型均值并作方法差值；不把三个 seed 的重复文本当成独立样本，
也不以合并混淆计数后的 F1 替代各模型 F1 的均值。</p>
<p>区间取重采样差值的 2.5% 和 97.5% 分位数。它们仅描述固定模型、固定训练距离统计、固定验证阈值条件下的测试样本不确定性，
不包含重新训练、验证集阈值选择或协方差拟合的不确定性；没有多重比较校正。区间跨 0 表示该条件分析尚不能排除无差异。</p>
{ci_table}{image_tag(plots[1])}
<h2>分类器与可追溯性</h2>
<p>以下是本次 FP32 前向的闭集分类结果，三种拒识分数共用这些原分类头预测；拒识不会改变分类头本身。</p>
{htable(['seed','IS Accuracy','IS Macro-F1'],classification)}
<h3>实际记录的新旧前向差异</h3>
<p>下表读取每个 seed 的 <code>test_drift.json</code> 与 <code>development_summary.json</code>，
比较归档 FP16 与本次统一 FP32 eager 前向。top-1 改变是类别输出发生变化，不等于正确率按同样数量下降；
OOS 行也参与这项类别一致性检查。这些差异说明新旧结果不能宣称逐位相同，但本次三种分数仍共享同一个当前前向。</p>
{drift_table}
<p>距离统计的实际拟合规模与收缩系数也来自各 seed 的 development summary：</p>
{development_table}
<p>报告生成前已从三份逐样本预测独立复算全部 18 个 test 设置的原评价指标，检查文本/标签跨 seed 对齐、
150 类各 30 条、1,000 条 OOS，以及每个布尔拒识列与冻结阈值的一致性。数值一致不等于证明语义正确或部署有效。</p>
<p>{html.escape(verification_text)}</p>
<details><summary>每个 seed 的冻结阈值</summary>{threshold_table}</details>
<details><summary>本报告输入文件与 SHA256</summary>{provenance}</details>
<p class="meta">同目录产物：summary_mean_sd.csv、paired_intervals.csv、rejection_comparison.png、paired_intervals.png。
报告图片已内嵌，可独立复制 HTML 阅读。所有关键结果从现存预测和指标表读取，不预设马氏距离优于参照。</p>
<p class="meta">AI 协助了实现、统计检查与报告整理；学生仍需核验结果解释，并按课程要求披露实际 AI 用途与本人贡献。</p>
</main></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs",
                        help="Directory containing frozen results; derived reports are saved here")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261019)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    output = args.output.resolve()
    metrics, frames, inputs = load_and_verify(output)
    run_details, detail_paths = load_run_details(output, frames)
    inputs.extend(detail_paths)
    summary = summarize(metrics)
    intervals = paired_bootstrap(frames, args.bootstrap_repeats, args.bootstrap_seed, args.batch_size)
    plots = create_plots(summary, intervals, output)
    document = build_html(metrics, frames, summary, intervals, plots, inputs,
                          args.bootstrap_repeats, args.bootstrap_seed, run_details)
    summary.to_csv(output / "summary_mean_sd.csv", index=False)
    intervals.to_csv(output / "paired_intervals.csv", index=False)
    report = output / "rejection_report_cn.html"
    report.write_text(document, encoding="utf-8")
    print(json.dumps({"status": "PASS", "verified_test_settings": 18,
                      "bootstrap_repeats": args.bootstrap_repeats,
                      "report": str(report), "outputs": [str(p) for p in plots]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
