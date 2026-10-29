"""Report frozen BERT rejection under existing deterministic typo stress tests.

Run after every prediction and metric file has been generated:
    python report_typo_rejection.py

This script only reads frozen results and writes derived tables, figures and an
HTML report. It never fits a model, changes a threshold or performs inference.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from report_rejection import (
    SEEDS, SCORES, POLICIES, SCORE_NAMES, POLICY_NAMES,
    booleans, evaluate_predictions, htable, image_tag,
)


CONDITIONS = ("clean", "typo_light", "typo_medium")
CONDITION_NAMES = {"clean": "Clean", "typo_light": "Light", "typo_medium": "Medium"}
DISPLAY_POLICIES = ("is_acceptance_95", "max_oos_f1")
STATES = ("accepted_correct", "accepted_wrong", "rejected_correct", "rejected_wrong")
STATE_NAMES = ("接受且分类正确", "接受但分类错误", "拒绝但分类头正确", "拒绝且分类头错误")


def load_results(project, output):
    metric_path = output / "metrics_by_seed.csv"
    metrics = pd.read_csv(metric_path, float_precision="round_trip")
    keys = ["seed", "condition", "score", "policy"]
    if not set(keys + ["threshold"]).issubset(metrics):
        raise ValueError("Metrics table lacks required identifiers or threshold")
    if metrics.duplicated(keys).any() or len(metrics) != 54:
        raise ValueError("Expected 54 unique seed/condition/score/policy settings")
    if set(metrics.seed) != set(SEEDS) or set(metrics.condition) != set(CONDITIONS):
        raise ValueError("Unexpected seeds or conditions")
    if set(metrics.score) != set(SCORES) or set(metrics.policy) != set(POLICIES):
        raise ValueError("Unexpected methods or policies")
    frozen_path = project / "outputs" / "metrics_by_seed.csv"
    frozen = pd.read_csv(frozen_path, float_precision="round_trip")
    frozen = frozen[frozen.split == "test"].set_index(["seed", "score", "policy"])
    if len(frozen) != 18 or frozen.index.duplicated().any():
        raise ValueError("Original clean experiment has no unique 18 frozen settings")
    inputs = [metric_path, frozen_path]
    frames, canonical = {}, None
    shared_columns = ["row_id", "original_text", "true_label", "label_id", "is_oos"]
    for seed in SEEDS:
        frames[seed] = {}
        for condition in CONDITIONS:
            path = output / f"seed{seed}" / f"{condition}_predictions.csv"
            frame = pd.read_csv(path, float_precision="round_trip")
            required = set(shared_columns + ["text", "changed", "modified_tokens", "predicted_label_id"])
            required |= {f"{s}_score" for s in SCORES}
            required |= {f"{s}_{p}_reject" for s in SCORES for p in POLICIES}
            if not required.issubset(frame):
                raise ValueError(f"Missing columns in {path}: {required - set(frame)}")
            if frame[list(required)].isna().any().any() or frame.row_id.duplicated().any():
                raise ValueError(f"Null values or duplicate row IDs in {path}")
            frame["is_oos"] = booleans(frame.is_oos)
            frame["changed"] = booleans(frame.changed)
            for column in ("label_id", "predicted_label_id", "modified_tokens"):
                numeric = pd.to_numeric(frame[column], errors="raise")
                if not np.all(numeric == numeric.astype(int)):
                    raise ValueError(f"Non-integer {column}")
                frame[column] = numeric.astype(int)
            if canonical is None:
                canonical = frame[shared_columns].reset_index(drop=True).copy()
            else:
                if set(frame.row_id) != set(canonical.row_id):
                    raise ValueError("Original row set differs across seed or condition")
                frame = frame.set_index("row_id").loc[canonical.row_id].reset_index()
                if not frame[shared_columns].equals(canonical):
                    raise ValueError("Original text or inherited gold labels differ")
            if not np.array_equal(frame.changed, frame.text != frame.original_text):
                raise ValueError("Changed flags disagree with actual text differences")
            if not np.array_equal(frame.changed, frame.modified_tokens > 0):
                raise ValueError("Modified-token counts disagree with changed flags")
            if condition == "clean" and (frame.changed.any() or frame.modified_tokens.any()):
                raise ValueError("Clean input contains mutations")
            if seed != SEEDS[0]:
                comparison_columns = ["text", "changed", "modified_tokens"]
                if not frame[comparison_columns].equals(frames[SEEDS[0]][condition][comparison_columns]):
                    raise ValueError("Different seeds received different perturbations")
            labels = frame.label_id.to_numpy()
            is_oos = frame.is_oos.to_numpy()
            unique, counts = np.unique(labels[~is_oos], return_counts=True)
            if len(frame) != 5500 or is_oos.sum() != 1000 or not np.all(labels[is_oos] == -1):
                raise ValueError("Expected 4,500 IS rows and 1,000 OOS rows with ID -1")
            if not np.array_equal(unique, np.arange(150)) or not np.all(counts == 30):
                raise ValueError("Expected exactly 30 rows per IS intent")
            prediction = frame.predicted_label_id.to_numpy()
            if np.any((prediction < 0) | (prediction >= 150)):
                raise ValueError("Predictions must remain original 150-class classifier outputs")
            for score in SCORES:
                values = pd.to_numeric(frame[f"{score}_score"], errors="raise").to_numpy()
                if not np.isfinite(values).all():
                    raise ValueError("Non-finite anomaly scores")
                for policy in POLICIES:
                    selected = metrics[(metrics.seed == seed) & (metrics.condition == condition)
                                       & (metrics.score == score) & (metrics.policy == policy)]
                    if len(selected) != 1:
                        raise ValueError("Missing unique metric setting")
                    row = selected.iloc[0]
                    threshold = float(row.threshold)
                    if threshold != float(frozen.loc[(seed, score, policy), "threshold"]):
                        raise ValueError("A stress-test threshold differs from the original frozen threshold")
                    column = f"{score}_{policy}_reject"
                    rejected = booleans(frame[column])
                    if not np.array_equal(rejected, values >= threshold):
                        raise ValueError("Rejection flags disagree with the frozen threshold")
                    frame[column] = rejected
                    calculated = evaluate_predictions(labels, prediction, is_oos, values, rejected)
                    for metric, actual in calculated.items():
                        if metric not in metrics or not np.isclose(
                            actual, float(row[metric]), rtol=1e-8, atol=1e-10, equal_nan=True
                        ):
                            raise ValueError(f"Metric mismatch: {seed}/{condition}/{score}/{policy}/{metric}")
            frames[seed][condition] = frame
            inputs.append(path)
    # Anchor clean classification and scores to the already completed FP32 study.
    for seed in SEEDS:
        path = project / "outputs" / f"seed{seed}" / "test_predictions.csv"
        original = pd.read_csv(path, float_precision="round_trip").set_index("row_id")
        current = frames[seed]["clean"]
        if set(original.index) != set(current.row_id):
            raise ValueError("Clean row set differs from the original FP32 study")
        original = original.loc[current.row_id].reset_index()
        if not np.array_equal(original.predicted_label_id, current.predicted_label_id):
            raise ValueError("Clean classifier predictions changed from the frozen study")
        if not np.array_equal(original.text, current.text):
            raise ValueError("Clean text differs from the original FP32 study")
        for score in SCORES:
            if not np.allclose(original[f"{score}_score"], current[f"{score}_score"], rtol=1e-10, atol=1e-10):
                raise ValueError("Clean score differs from the original FP32 study")
        inputs.append(path)
    return metrics, frames, inputs


def aggregate(frame, group_columns):
    excluded = set(group_columns) | {"seed", "threshold"}
    columns = [c for c in frame if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])]
    rows = []
    for values, group in frame.groupby(group_columns, sort=False):
        if set(group.seed) != set(SEEDS) or len(group) != len(SEEDS):
            raise ValueError("Incomplete three-seed aggregate")
        row = dict(zip(group_columns, values))
        row["n_seeds"] = len(group)
        for column in columns:
            values_array = group[column].to_numpy(dtype=float)
            row[column + "_mean"] = float(values_array.mean())
            row[column + "_sd"] = float(values_array.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def state_statistics(clean_correct, current_correct, clean_rejected, current_rejected, changed):
    """All five arrays refer only to the SAME independent IS test texts."""
    n = len(current_correct)
    if not n or any(len(a) != n for a in (clean_correct, clean_rejected, current_rejected, changed)):
        raise ValueError("State arrays need matching non-empty IS rows")
    clean_accepted = ~clean_rejected
    clean_ac = clean_correct & clean_accepted
    states = (
        current_correct & ~current_rejected, ~current_correct & ~current_rejected,
        current_correct & current_rejected, ~current_correct & current_rejected,
    )
    if not np.all(np.sum(np.stack(states), axis=0) == 1):
        raise AssertionError("Normal-request states must be mutually exclusive and exhaustive")
    result = {
        "is_n": n, "changed_is_n": int(changed.sum()),
        "clean_accepted_n": int(clean_accepted.sum()),
        "clean_rejected_n": int(clean_rejected.sum()),
        "clean_accepted_correct_n": int(clean_ac.sum()),
        "closed_accuracy": float(current_correct.mean()),
        "is_rejection_rate": float(current_rejected.mean()),
        "rejection_rate_delta_vs_clean": float(current_rejected.mean() - clean_rejected.mean()),
        "new_rejection_n": int((clean_accepted & current_rejected).sum()),
        "new_rejection_rate_over_all_is": float((clean_accepted & current_rejected).mean()),
        "new_rejection_rate_over_clean_accepted": float((clean_accepted & current_rejected).sum() / clean_accepted.sum())
        if clean_accepted.any() else np.nan,
        "recovered_acceptance_n": int((clean_rejected & ~current_rejected).sum()),
        "recovered_acceptance_rate_over_all_is": float((clean_rejected & ~current_rejected).mean()),
        "recovered_acceptance_rate_over_clean_rejected": float((clean_rejected & ~current_rejected).sum() / clean_rejected.sum())
        if clean_rejected.any() else np.nan,
        "net_rejection_change_n": int(current_rejected.sum() - clean_rejected.sum()),
        "changed_only_rejection_n": int((changed & current_rejected).sum()),
        "changed_only_rejection_rate": float((changed & current_rejected).sum() / changed.sum()) if changed.any() else np.nan,
    }
    if result["net_rejection_change_n"] != result["new_rejection_n"] - result["recovered_acceptance_n"]:
        raise AssertionError("Net rejection change must equal new rejection minus recovered acceptance")
    for name, state in zip(STATES, states):
        flow = clean_ac & state
        result.update({
            name + "_n": int(state.sum()), name + "_rate_over_all_is": float(state.mean()),
            "clean_ac_to_" + name + "_n": int(flow.sum()),
            "clean_ac_to_" + name + "_rate_over_all_is": float(flow.mean()),
            "clean_ac_to_" + name + "_rate_over_clean_accepted_correct": float(flow.sum() / clean_ac.sum())
            if clean_ac.any() else np.nan,
        })
    return result


def normal_transitions(frames):
    rows = []
    for seed in SEEDS:
        clean = frames[seed]["clean"]
        known = ~clean.is_oos.to_numpy()
        clean_correct = clean.predicted_label_id.to_numpy()[known] == clean.label_id.to_numpy()[known]
        for condition in CONDITIONS:
            current = frames[seed][condition]
            correct = current.predicted_label_id.to_numpy()[known] == current.label_id.to_numpy()[known]
            for score in SCORES:
                for policy in POLICIES:
                    column = f"{score}_{policy}_reject"
                    stats = state_statistics(clean_correct, correct, clean[column].to_numpy()[known],
                                             current[column].to_numpy()[known], current.changed.to_numpy()[known])
                    rows.append({"seed": seed, "condition": condition, "score": score, "policy": policy, **stats})
    return pd.DataFrame(rows)


def paired_intervals(frames, repeats=2000, random_seed=20261019, batch_size=64):
    if repeats < 2 or batch_size < 1:
        raise ValueError("Bootstrap requires at least two draws and a positive batch size")
    first = frames[SEEDS[0]]["clean"]
    known = ~first.is_oos.to_numpy()
    labels = first.label_id.to_numpy()[known]
    strata = [np.flatnonzero(labels == label) for label in range(150)]
    if len(labels) != 4500 or any(len(indices) != 30 for indices in strata):
        raise ValueError("Bootstrap expects 150 IS intent strata, 30 rows each")
    events = np.zeros((3, 3, 3, 2, 4500), dtype=float)
    for si, seed in enumerate(SEEDS):
        for ci, condition in enumerate(CONDITIONS):
            for mi, score in enumerate(SCORES):
                for pi, policy in enumerate(POLICIES):
                    events[si, ci, mi, pi] = frames[seed][condition][f"{score}_{policy}_reject"].to_numpy()[known]
    point = events.mean(axis=-1).mean(axis=0)
    draws = np.empty((3, 3, 2, repeats))
    flattened = events.reshape(-1, 4500)
    rng = np.random.default_rng(random_seed)
    for start in range(0, repeats, batch_size):
        size = min(batch_size, repeats - start)
        sample = np.concatenate([rng.choice(indices, size=(size, 30), replace=True) for indices in strata], axis=1)
        offsets = 4500 * np.arange(size)[:, None]
        weights = np.bincount((sample + offsets).ravel(), minlength=size * 4500).reshape(size, 4500).astype(float)
        per_seed_rates = (flattened @ weights.T).reshape(*events.shape[:-1], size) / 4500
        draws[..., start:start + size] = per_seed_rates.mean(axis=0)
    comparisons = []
    distance = SCORES.index("mahalanobis")
    for ci, condition in enumerate(CONDITIONS[1:], start=1):
        for pi, policy in enumerate(POLICIES):
            comparisons.append(("mahalanobis_noisy_minus_clean", condition, policy,
                                (ci, distance, pi), (0, distance, pi)))
            for reference in ("entropy", "msp"):
                comparisons.append((f"mahalanobis_minus_{reference}", condition, policy,
                                    (ci, distance, pi), (ci, SCORES.index(reference), pi)))
    records = []
    for name, condition, policy, target, reference in comparisons:
        delta = draws[target] - draws[reference]
        low, high = np.quantile(delta, [.025, .975])
        estimate = point[target] - point[reference]
        records.append({
            "comparison": name, "condition": condition, "policy": policy,
            "analysis_role": "requested_operating_point" if policy == "is_acceptance_95" else "additional_operating_point",
            "metric": "in_scope_false_rejection_rate", "estimate_delta": estimate,
            "ci_low": low, "ci_high": high, "estimate_delta_pp": estimate * 100,
            "ci_low_pp": low * 100, "ci_high_pp": high * 100,
            "positive_direction": "more rejection of supported requests",
            "denominator": "4500 original IS texts", "n_fixed_models": 3,
            "bootstrap_repeats": repeats, "bootstrap_seed": random_seed,
            "strata": "150 IS intents, 30 rows per intent; same draws across conditions, seeds and methods",
            "scope": "fixed BERT weights, training distance statistics and clean validation thresholds",
        })
    return pd.DataFrame(records)


def create_plots(transition_summary, intervals, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = transition_summary.set_index(["condition", "score", "policy"])
    colors = ("#19847c", "#748b9e", "#e6a449", "#bd5050")
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for pi, policy in enumerate(DISPLAY_POLICIES):
        for mi, score in enumerate(SCORES):
            ax = axes[pi, mi]
            bottom = np.zeros(3)
            for state, color, title in zip(STATES, colors, ("Accepted / head correct", "Accepted / head wrong",
                                                          "Rejected / head correct", "Rejected / head wrong")):
                values = np.array([table.loc[(c, score, policy), state + "_rate_over_all_is_mean"] * 100 for c in CONDITIONS])
                ax.bar(range(3), values, bottom=bottom, color=color, label=title)
                bottom += values
            ax.set_xticks(range(3), [CONDITION_NAMES[c] for c in CONDITIONS])
            ax.set_ylim(0, 100)
            ax.set_ylabel("% of all 4,500 supported requests")
            ax.set_title(SCORE_NAMES[score] + (" | clean val IS >=95%" if pi == 0 else " | clean val max OOS F1"))
            ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    fig.suptitle("Frozen classifier and thresholds: normal-request outcomes (three-seed means)", fontsize=14)
    state_path = output / "normal_request_states.png"
    fig.savefig(state_path, dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), constrained_layout=True)
    for ax, policy in zip(axes, DISPLAY_POLICIES):
        subset = intervals[intervals.policy == policy]
        y = np.arange(len(subset))
        labels = []
        for row in subset.itertuples():
            reference = {"mahalanobis_noisy_minus_clean": "distance noisy - clean",
                         "mahalanobis_minus_entropy": "distance - entropy",
                         "mahalanobis_minus_msp": "distance - MSP"}[row.comparison]
            labels.append(CONDITION_NAMES[row.condition] + ": " + reference)
        ax.hlines(y, subset.ci_low_pp, subset.ci_high_pp, color="#bc6924", linewidth=2.5)
        ax.scatter(subset.estimate_delta_pp, y, color="#bc6924", zorder=3)
        ax.axvline(0, color="#64748b", linestyle="--", linewidth=1)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_title("Clean validation IS >=95%" if policy == "is_acceptance_95" else "Clean validation max OOS F1")
        ax.set_xlabel("IS rejection-rate difference (pp); positive = more rejection")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Paired 95% intervals, conditional on frozen models and thresholds", fontsize=14)
    interval_path = output / "typo_rejection_intervals.png"
    fig.savefig(interval_path, dpi=180)
    plt.close(fig)
    return state_path, interval_path


def formatted(row, name, multiplier=100):
    mean, sd = row[name + "_mean"], row[name + "_sd"]
    if not np.isfinite(mean) or not np.isfinite(sd):
        return "—"
    return f"{mean*multiplier:.2f} ± {sd*multiplier:.2f}"


def build_html(frames, summary, transitions, transition_summary, intervals, plots, inputs, repeats, random_seed, input_audit, cases):
    table = transition_summary.set_index(["condition", "score", "policy"])
    metrics = summary.set_index(["condition", "score", "policy"])
    focus = [table.loc[(condition, "mahalanobis", "is_acceptance_95")] for condition in CONDITIONS]
    headline = ("固定 clean validation 95% 接受率策略下，马氏距离对正常请求的误拒率从 clean 的 "
                f"{focus[0]['is_rejection_rate_mean']*100:.2f}% 变为 light 的 {focus[1]['is_rejection_rate_mean']*100:.2f}%、"
                f"medium 的 {focus[2]['is_rejection_rate_mean']*100:.2f}%。"
                "下面将分类本身变错与拒识造成的服务损失分开检查。")
    comparison_sentences = []
    for condition in CONDITIONS[1:]:
        distance = table.loc[(condition, "mahalanobis", "is_acceptance_95")]
        entropy = table.loc[(condition, "entropy", "is_acceptance_95")]
        msp = table.loc[(condition, "msp", "is_acceptance_95")]
        comparison_sentences.append(
            f"{CONDITION_NAMES[condition]} 下正常误拒率分别为距离 {distance['is_rejection_rate_mean']*100:.2f}%、"
            f"Entropy {entropy['is_rejection_rate_mean']*100:.2f}%、MSP {msp['is_rejection_rate_mean']*100:.2f}%；"
            f"距离的正确且接受率为 {distance['accepted_correct_rate_over_all_is_mean']*100:.2f}%，"
            f"另有 {distance['rejected_correct_rate_over_all_is_mean']*100:.2f}% 的正常输入属于分类头本可正确却被拒绝。"
            f"其中 clean 原本正确且被接受、扰动后分类仍正确却新被拒绝的平均为 "
            f"{distance['clean_ac_to_rejected_correct_n_mean']:.2f} 条（占全部IS的 "
            f"{distance['clean_ac_to_rejected_correct_rate_over_all_is_mean']*100:.2f}%），这是更直接的新增服务损失。"
        )
    comparison_sentences.append(
        "这组95%策略结果必须保留，不能通过改挑max-F1运行点掩盖它。"
        "同时，各种分数的阈值位置不同，不能仅据这个运行点断言某种分数在所有阈值下都更差。"
    )
    for condition in CONDITIONS[1:]:
        distance = metrics.loc[(condition, "mahalanobis", "max_oos_f1")]
        entropy = metrics.loc[(condition, "entropy", "max_oos_f1")]
        comparison_sentences.append(
            f"保留的max-F1策略中，{CONDITION_NAMES[condition]} 的距离/Entropy正常误拒率为 "
            f"{distance['in_scope_false_rejection_rate_mean']*100:.2f}% / {entropy['in_scope_false_rejection_rate_mean']*100:.2f}%，"
            f"但对应OOS recall也为 {distance['oos_recall_mean']*100:.2f}% / {entropy['oos_recall_mean']*100:.2f}%，"
            "应共同评价误拒和未知漏检的取舍。"
        )
    coverage_rows, classification_rows = [], []
    for condition in CONDITIONS:
        frame = frames[SEEDS[0]][condition]
        known = ~frame.is_oos.to_numpy()
        for scope, mask in (("IS", known), ("OOS", ~known)):
            modified = frame.modified_tokens.to_numpy()[mask]
            coverage_rows.append([CONDITION_NAMES[condition], scope, int(mask.sum()), int((modified == 0).sum()),
                                  int((modified == 1).sum()), int((modified >= 2).sum()),
                                  f"{frame.changed.to_numpy()[mask].mean()*100:.2f}%"])
        closed = []
        for seed in SEEDS:
            current = frames[seed][condition]
            closed.append(np.mean(current.predicted_label_id.to_numpy()[known] == current.label_id.to_numpy()[known]))
        classification_rows.append([CONDITION_NAMES[condition], f"{np.mean(closed)*100:.2f} ± {np.std(closed,ddof=1)*100:.2f}"])
    sections = []
    for policy in DISPLAY_POLICIES:
        outcome_rows, transition_rows, operating_rows = [], [], []
        for score in SCORES:
            for condition in CONDITIONS:
                row = table.loc[(condition, score, policy)]
                outcome_rows.append([SCORE_NAMES[score], CONDITION_NAMES[condition],
                                     *[formatted(row, state + "_rate_over_all_is") for state in STATES],
                                     formatted(row, "is_rejection_rate"), formatted(row, "changed_only_rejection_rate")])
                if condition != "clean":
                    transition_rows.append([SCORE_NAMES[score], CONDITION_NAMES[condition],
                                            formatted(row, "clean_accepted_correct_n", 1),
                                            *[formatted(row, "clean_ac_to_" + state + "_rate_over_all_is") for state in STATES],
                                            formatted(row, "new_rejection_n", 1),
                                            formatted(row, "recovered_acceptance_n", 1),
                                            formatted(row, "net_rejection_change_n", 1),
                                            formatted(row, "new_rejection_rate_over_all_is"),
                                            formatted(row, "new_rejection_rate_over_clean_accepted")])
                result = metrics.loc[(condition, score, policy)]
                operating_rows.append([SCORE_NAMES[score], CONDITION_NAMES[condition], formatted(result, "oos_f1"),
                                       formatted(result, "oos_recall"), formatted(result, "in_scope_acceptance"),
                                       formatted(result, "combined_151_accuracy")])
        sections.append(f"<h3>{POLICY_NAMES[policy]}：阈值始终来自 clean validation</h3>")
        sections.append(htable(["分数", "条件", *[name + " %" for name in STATE_NAMES], "IS 总误拒 %", "仅被修改 IS 误拒 %"], outcome_rows))
        sections.append("<p>四状态的分母均为全部 4,500 条 IS；横向相加为 100%。分类正确/错误使用当前条件下的原分类头预测，"
                        "包括被拒绝的请求。最后一列分母仅为实际发生修改的正常文本，clean 无该分母，显示为“—”。</p>")
        sections.append("<h4>原本正确服务的请求去了哪里？</h4>")
        sections.append(htable(["分数", "条件", "clean 正确且接受数", "→接受正确 /4500 %", "→接受错误 /4500 %",
                                "→拒绝正确 /4500 %", "→拒绝错误 /4500 %", "新误拒数", "恢复接受数", "净增加拒绝数",
                                "新误拒 /4500 %", "新误拒 /clean 接受IS %"], transition_rows))
        sections.append("<p>四个箭头仅跟踪 clean 时“接受且分类正确”的同一批请求，其比例均除 4,500，而非各自去向样本数。"
                        "新误拒跟踪更宽的来源池：clean 时被接受的所有 IS（包括当时分类错误的请求）→扰动后被拒绝。"
                        "“新误拒 /clean 接受IS”逐 seed 使用各自 clean 接受数量为分母后再求均值。"
                        "恢复接受为 clean 被拒→扰动后被接受，并不保证恢复后的分类正确。"
                        "净增加拒绝数 = 新误拒数 − 恢复接受数，不能把净增称作全部新增误拒。</p>")
        sections.append("<details><summary>同一冻结运行点的 OOS 与混合系统结果</summary>" + htable(
            ["分数", "条件", "OOS F1 %", "OOS recall %", "IS 接受率 %", "系统准确率 %"], operating_rows) + "</details>")
    ci_rows = []
    for row in intervals.itertuples():
        comparison = {"mahalanobis_noisy_minus_clean": "距离：扰动 − clean",
                      "mahalanobis_minus_entropy": "同条件：距离 − Entropy",
                      "mahalanobis_minus_msp": "同条件：距离 − MSP"}[row.comparison]
        ci_rows.append([POLICY_NAMES[row.policy], CONDITION_NAMES[row.condition], comparison,
                        f"{row.estimate_delta_pp:+.2f}", f"[{row.ci_low_pp:+.2f}, {row.ci_high_pp:+.2f}]"])
    input_base = inputs[0].parent
    provenance = htable(["输入", "SHA256"], [[str(path.relative_to(input_base)) if path.is_relative_to(input_base) else str(path),
                       hashlib.sha256(path.read_bytes()).hexdigest()] for path in inputs])
    unchanged_path = input_base / "unchanged_text_checks.json"
    unchanged = json.loads(unchanged_path.read_text(encoding="utf-8"))
    unchanged_table = htable(["seed", "条件", "未改变文本数（IS+OOS）", "最大logit绝对差", "argmax改变数", "拒识改变数合计"], [
        [row["seed"], CONDITION_NAMES[row["condition"]], row["unchanged_rows"], f"{row['max_abs_logit_delta']:.3g}",
         row["changed_argmax"], sum(row["rejection_decision_changes"].values())] for row in unchanged])
    tokenizer_rows = input_audit["tokenizer_audit"]["statistics"]
    tokenizer_table = htable(["条件", "检查行数", "最大WordPiece长度（含特殊token）", "超过64的行数", "含UNK的行数"], [
        [CONDITION_NAMES[row["condition"]], row["rows"], row["max_wordpiece_length"],
         row["rows_over_max_length_64"], row["rows_with_unk"]] for row in tokenizer_rows])
    if any(row["rows_over_max_length_64"] or row["rows_with_unk"] for row in tokenizer_rows):
        tokenizer_conclusion = "输入审计发现超长或UNK情况；需要保留该限制，不能排除它们与错误有关。"
    else:
        tokenizer_conclusion = ("三种条件均未超过max_length=64且没有UNK，因此不能把本次性能变化归因于截断或未知token替代。"
                                 "这不表示分词未变化：有效WordPiece组成及其上下文表示仍可能改变。")
    case_rows = []
    for row in cases.itertuples():
        if row.method != "mahalanobis" or row.policy != "is_acceptance_95":
            raise ValueError("Illustrative cases refer to another method or policy")
        for seed in SEEDS:
            clean = frames[seed]["clean"].set_index("row_id").loc[row.row_id]
            noisy = frames[seed][row.condition].set_index("row_id").loc[row.row_id]
            if (clean.text != row.original_text or noisy.text != row.text or noisy.true_label != row.true_label
                    or clean.predicted_label_id != clean.label_id or noisy.predicted_label_id != noisy.label_id
                    or clean.mahalanobis_is_acceptance_95_reject or not noisy.mahalanobis_is_acceptance_95_reject):
                raise ValueError("Casebook's claimed accepted-correct to rejected-correct transition is inconsistent")
        case_rows.append([CONDITION_NAMES[row.condition], row.row_id, row.original_text, row.text,
                          row.true_label, "三个seed：分类仍正确，但从接受转为拒绝",
                          f"{row.seed42_clean_distance:.1f} → {row.seed42_typo_distance:.1f} / {row.seed42_frozen_threshold:.1f}"])
    case_table = htable(["条件", "row_id", "原文", "扰动后", "Gold及正确预测", "结果", "seed42距离变化 / 冻结阈值"], case_rows)
    generated = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M UTC+08:00")
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>冻结拒识系统：拼写扰动下的正常请求代价</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f2f5f6;color:#183846;font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1250px;margin:28px auto;padding:34px 40px;background:white;border:1px solid #dce5e8;border-radius:16px}}
h1{{font-size:29px;line-height:1.4}}h2{{font-size:23px;margin-top:35px}}h3{{font-size:18px;margin-top:26px}}h4{{font-size:16px}}
.meta{{font-size:13px;color:#5d7380}}.lead{{background:#fff5e9;border-left:5px solid #bd712e;padding:16px 20px;font-size:18px}}
.table-scroll{{overflow-x:auto;margin:16px 0}}table{{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}}
th,td{{padding:9px 10px;border-bottom:1px solid #dce5e8;text-align:left;vertical-align:top}}th{{background:#eaf0f2;white-space:nowrap}}
td{{overflow-wrap:anywhere}}img{{max-width:100%;height:auto;border:1px solid #dce5e8;border-radius:8px;margin:16px 0}}
summary{{cursor:pointer;font-weight:650}}details{{margin:18px 0}}code{{background:#edf2f5;padding:2px 5px}}
@media(max-width:700px){{main{{padding:24px 16px;margin:0;border-radius:0}}h1{{font-size:24px}}}}
</style><main><p class="meta">MFE310TC · 固定系统的 typo 压力测试 · {generated}</p>
<h1>正常请求拼错以后，系统还能正确服务吗？</h1><p class="lead">{html.escape(headline)}</p>
<p>{html.escape(' '.join(comparison_sentences))}</p>
<p>本轮保留现有 BERT 权重、训练得到的类中心与协方差、所有 clean validation 阈值。
MSP、Entropy、Mahalanobis 在相同原始文本及同一套 light/medium 扰动上比较。
95% 接受率策略是本轮关注的运行点；原 clean 实验的唯一主终点仍为“max-OOS-F1 策略下，距离 − MSP 的 OOS F1”，没有重定义。</p>
<h2>先区分分类错误与拒识误伤</h2>
<p>闭集分类检查原分类头是否给出正确意图，即使请求最终被拒绝也照常评价。
拒识决定是否向用户提供该预测。正常文本拼错后仍继承原 IS 标签，因此被拒绝计为正常请求误拒；
其中“分类头本来正确但被拒绝”和“分类头已分错且被拒绝”分开呈现。拒绝错误分类可以阻止错误执行，但仍没有完成原正常请求，不能混同于正确服务。
只在被接受子集上的准确率提高，可能只是拒绝了更多困难请求，不能据此称鲁棒性提升。</p>
{htable(['条件','闭集 IS Accuracy %（三个seed均值 ± SD）'],classification_rows)}
<p>上述闭集准确率在同一条件下为三种拒识分数共用；四状态进一步说明分类结果如何与接收/拒绝决策组合。</p>
<h2>扰动覆盖与标签边界</h2>
{htable(['条件','范围','总数','未改变','修改1词','修改至少2词','修改覆盖率'],coverage_rows)}
<p>这些是既有自动字符扰动，不是人工确认的语义保持改写。修改可能偶然形成其他词，继承原标签是一项评估约定。
没有合适词的文本保持原样，medium中也并非所有请求均修改两个词；总体指标混合了不同扰动覆盖。
因此同时报告全部 4,500 条 IS 与 changed-only 的误拒率。
本轮压力测试测量的是冻结系统对这套 synthetic typo 的敏感性，不直接代表真实用户打字错误分布。</p>
{tokenizer_table}<p>{html.escape(tokenizer_conclusion)}</p>
<h2>正常请求的四种结局</h2>
<p>所有表格为三个固定 seed 的均值 ± 样本标准差（ddof=1）；百分比指标的 SD 单位为百分点。
计数表中的小数是三个模型的平均计数，不能解释为单次运行中的小数样本。</p>
{''.join(sections)}{image_tag(plots[0])}
<p><strong>95% 仅约束 clean validation 的 IS 接受率，不能保证 clean test，更不能保证 typo test 也达到95%。</strong>
拼写变化可能改变 BERT 表示以及其到已知类中心的距离；这可以解释为何需要测试阈值迁移，但结果方向和大小必须由表中观测支持。</p>
<h3>分类仍正确却新被拒绝的实际例子</h3>{case_table}
<p>这些案例来自 illustrative_false_rejections.csv：每种扰动在三个seed都满足clean accepted_correct→typo rejected_correct的池中，
按原row_id选前三条。它们用于展示已观测到的错误类型，不估计其总体占比，也不是内部因果机制证明。
表中完整的三seed距离与阈值保留在原CSV，正文距离列只展示seed42。</p>
<h2>配对比较与区间</h2>
<p>以下差值均为“正常请求误拒率”，正值表示左侧拒绝更多正常请求。距离的 light/medium − clean 衡量净误拒率变化；
同条件距离 − Entropy/MSP 衡量相同文本上的方法差异。它们不是 OOS F1，也不是新误拒数量。</p>
{htable(['冻结阈值策略','条件','比较','差值 pp','95% 条件区间 pp'],ci_rows)}{image_tag(plots[1])}
<p>使用 {repeats:,} 次配对 bootstrap，随机种子 {random_seed}。仅对 4,500 条 IS 原始文本按150个intent分层，
每类有放回抽取30条；同一抽样权重共同应用于 clean/light/medium、三个 seed 和三种分数。
每个固定模型先计算误拒率，再对三个模型取均值并计算差值；不把同一文本的不同 seed 预测当作独立样本。
区间是差值分布的2.5%和97.5%分位数，条件于已固定的模型、训练距离统计与阈值，不包含重新训练、重新拟合或验证选阈值的不确定性。
这些是事后诊断的多个比较，未进行多重比较校正。</p>
<h2>复核范围</h2>
<p>生成报告前逐样本复算54组评价指标，检查原文与标签跨条件/seed对齐、不同seed使用相同扰动、修改标记与文本一致、
所有阈值与原clean研究一致，并确认clean原分类头预测及三种分数与原FP32结果一致。
这些数值检查不证明自动扰动完全保义，也不把统计关联当成模型内部因果解释。</p>
<h3>未改变文本的前向检查</h3>{unchanged_table}
<p>上表来自 unchanged_text_checks.json。未改变文本指正常与OOS两类合计，不应把该数误当成未改变IS数量。
微小浮点差异与分类或拒识决策变化分别记录。</p>
<details><summary>输入文件与 SHA256</summary>{provenance}</details>
<p class="meta">衍生产物：summary_mean_sd.csv、normal_request_transitions.csv（逐seed四状态、流向、分母与新误拒）、paired_intervals.csv及两张PNG。
图片已内嵌，HTML可独立阅读。AI协助实现、统计检查与整理；学生须核验解释并按课程要求披露实际使用与本人贡献。</p>
</main></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parent
    parser.add_argument("--output", type=Path, default=project / "typo_stress_test" / "outputs")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261019)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    output = args.output.resolve()
    metrics, frames, inputs = load_results(project, output)
    inputs.append(output / "unchanged_text_checks.json")
    audit_path = output.parent / "input_audit.json"
    input_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not all(input_audit["verified"].values()):
        raise ValueError("Input audit contains unsuccessful checks")
    cases_path = output / "illustrative_false_rejections.csv"
    cases = pd.read_csv(cases_path, float_precision="round_trip")
    inputs.extend((audit_path, cases_path))
    summary = aggregate(metrics, ["condition", "score", "policy"])
    transitions = normal_transitions(frames)
    transition_summary = aggregate(transitions, ["condition", "score", "policy"])
    intervals = paired_intervals(frames, args.bootstrap_repeats, args.bootstrap_seed, args.batch_size)
    plots = create_plots(transition_summary, intervals, output)
    document = build_html(frames, summary, transitions, transition_summary, intervals, plots,
                          inputs, args.bootstrap_repeats, args.bootstrap_seed, input_audit, cases)
    summary.to_csv(output / "summary_mean_sd.csv", index=False)
    transitions.to_csv(output / "normal_request_transitions.csv", index=False)
    intervals.to_csv(output / "paired_intervals.csv", index=False)
    report = output / "typo_rejection_report_cn.html"
    report.write_text(document, encoding="utf-8")
    print(json.dumps({"status": "PASS", "verified_settings": 54, "transition_rows": len(transitions),
                      "paired_intervals": len(intervals), "bootstrap_repeats": args.bootstrap_repeats,
                      "report": str(report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
