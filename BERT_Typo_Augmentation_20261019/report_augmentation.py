"""Report the fixed one-epoch clean-control versus typo-mixture experiment.

Run only when all six models and their frozen test results have completed:
    python report_augmentation.py

Only derived reports are written. Training, inference and threshold fitting are
outside this script. Historical predictions are read from the earlier study.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT = Path(__file__).resolve().parent
PREVIOUS_PROJECT = PROJECT.parent / "BERT_Rejection_Study_20261019"
# Reuse verified reporting math only; imports perform no inference or writes.
sys.path.insert(0, str(PREVIOUS_PROJECT))
from report_rejection import SEEDS, SCORES, POLICIES, SCORE_NAMES, POLICY_NAMES, booleans, evaluate_predictions, htable, image_tag
from report_typo_rejection import CONDITIONS, CONDITION_NAMES, STATES, STATE_NAMES, aggregate, normal_transitions


ARMS = ("original", "clean_continue", "typo_mix")
ARM_NAMES = {"original": "原 BERT", "clean_continue": "继续 clean 训练", "typo_mix": "混合 typo 训练"}
FOCUS_SCORE = "mahalanobis"
FOCUS_POLICY = "is_acceptance_95"
BOOTSTRAP_METRICS = (
    "in_scope_false_rejection_rate", "closed_set_accuracy", "in_scope_correct_and_accepted_rate",
    "oos_f1", "oos_recall", "in_scope_acceptance", "combined_151_accuracy",
)
METRIC_NAMES = {
    "in_scope_false_rejection_rate": "正常请求误拒率",
    "closed_set_accuracy": "闭集分类准确率",
    "in_scope_correct_and_accepted_rate": "正常请求正确且接受率",
    "oos_f1": "OOS F1", "oos_recall": "OOS recall",
    "in_scope_acceptance": "正常请求接受率", "combined_151_accuracy": "混合系统准确率",
}


def load_results(output):
    metric_path = output / "metrics_by_seed.csv"
    metrics = pd.read_csv(metric_path, float_precision="round_trip")
    keys = ["arm", "seed", "condition", "score", "policy"]
    if not set(keys + ["threshold", "closed_set_accuracy", "closed_set_macro_f1"]).issubset(metrics):
        raise ValueError("Metrics table lacks expected identifiers or closed-set metrics")
    if len(metrics) != 162 or metrics.duplicated(keys).any():
        raise ValueError("Expected 162 unique test settings")
    expected_sets = {"arm": ARMS, "seed": SEEDS, "condition": CONDITIONS, "score": SCORES, "policy": POLICIES}
    for key, expected in expected_sets.items():
        if set(metrics[key]) != set(expected):
            raise ValueError(f"Unexpected {key} values")
    original_output = PREVIOUS_PROJECT / "typo_stress_test" / "outputs"
    original_metrics_path = original_output / "metrics_by_seed.csv"
    original_metrics = pd.read_csv(original_metrics_path, float_precision="round_trip").set_index(
        ["seed", "condition", "score", "policy"])
    inputs = [metric_path, original_metrics_path]
    identity = ["row_id", "original_text", "true_label", "label_id", "is_oos"]
    frames, canonical, reference_texts = {}, None, {}
    for arm in ARMS:
        frames[arm] = {}
        for seed in SEEDS:
            frames[arm][seed] = {}
            for condition in CONDITIONS:
                folder = original_output / f"seed{seed}" if arm == "original" else output / arm / f"seed{seed}"
                path = folder / f"{condition}_predictions.csv"
                frame = pd.read_csv(path, float_precision="round_trip")
                required = set(identity + ["text", "changed", "modified_tokens", "predicted_label_id"])
                required |= {f"{s}_score" for s in SCORES}
                required |= {f"{s}_{p}_reject" for s in SCORES for p in POLICIES}
                if not required.issubset(frame) or frame[list(required)].isna().any().any():
                    raise ValueError(f"Missing fields or null values: {path}")
                if frame.row_id.duplicated().any():
                    raise ValueError("Duplicate prediction row IDs")
                frame["is_oos"], frame["changed"] = booleans(frame.is_oos), booleans(frame.changed)
                for name in ("label_id", "predicted_label_id", "modified_tokens"):
                    numeric = pd.to_numeric(frame[name], errors="raise")
                    if not np.all(numeric == numeric.astype(int)):
                        raise ValueError(f"Noninteger {name}")
                    frame[name] = numeric.astype(int)
                if canonical is None:
                    canonical = frame[identity].copy().reset_index(drop=True)
                else:
                    if set(frame.row_id) != set(canonical.row_id):
                        raise ValueError("Original test rows differ across arms/seeds/conditions")
                    frame = frame.set_index("row_id").loc[canonical.row_id].reset_index()
                    if not frame[identity].equals(canonical):
                        raise ValueError("Original texts or inherited labels differ")
                text_columns = ["text", "changed", "modified_tokens"]
                if condition not in reference_texts:
                    reference_texts[condition] = frame[text_columns].copy()
                elif not frame[text_columns].equals(reference_texts[condition]):
                    raise ValueError("Models received different stress-test texts")
                if not np.array_equal(frame.changed, frame.text != frame.original_text):
                    raise ValueError("Mutation flags do not match text differences")
                if not np.array_equal(frame.changed, frame.modified_tokens > 0):
                    raise ValueError("Mutation counts do not match changed flags")
                if condition == "clean" and frame.changed.any():
                    raise ValueError("Clean rows contain mutations")
                labels = frame.label_id.to_numpy()
                is_oos = frame.is_oos.to_numpy()
                prediction = frame.predicted_label_id.to_numpy()
                unique, counts = np.unique(labels[~is_oos], return_counts=True)
                if (len(frame) != 5500 or is_oos.sum() != 1000 or not np.all(labels[is_oos] == -1)
                        or not np.array_equal(unique, np.arange(150)) or not np.all(counts == 30)):
                    raise ValueError("Expected 150 IS classes with 30 rows each and 1,000 OOS rows")
                if not np.all((prediction >= 0) & (prediction < 150)):
                    raise ValueError("Classifier predictions must remain 150-class IDs")
                closed_accuracy = float((prediction[~is_oos] == labels[~is_oos]).mean())
                closed_f1 = float(f1_score(labels[~is_oos], prediction[~is_oos], labels=np.arange(150), average="macro"))
                for score in SCORES:
                    values = pd.to_numeric(frame[f"{score}_score"], errors="raise").to_numpy()
                    if not np.isfinite(values).all():
                        raise ValueError("Non-finite rejection score")
                    for policy in POLICIES:
                        selected = metrics[(metrics.arm == arm) & (metrics.seed == seed) & (metrics.condition == condition)
                                           & (metrics.score == score) & (metrics.policy == policy)]
                        if len(selected) != 1:
                            raise ValueError("Missing unique experiment setting")
                        row = selected.iloc[0]
                        frozen_group = metrics[(metrics.arm == arm) & (metrics.seed == seed)
                                               & (metrics.score == score) & (metrics.policy == policy)]
                        if frozen_group.threshold.nunique(dropna=False) != 1 or pd.isna(row.threshold):
                            raise ValueError("Threshold changed across test conditions")
                        column = f"{score}_{policy}_reject"
                        rejected = booleans(frame[column])
                        if not np.array_equal(rejected, values >= float(row.threshold)):
                            raise ValueError("Rejection flags differ from frozen threshold")
                        frame[column] = rejected
                        calculated = evaluate_predictions(labels, prediction, is_oos, values, rejected)
                        calculated.update(closed_set_accuracy=closed_accuracy, closed_set_macro_f1=closed_f1)
                        for metric, actual in calculated.items():
                            if metric not in metrics or not np.isclose(actual, float(row[metric]), rtol=1e-8, atol=1e-10, equal_nan=True):
                                raise ValueError(f"Metric mismatch: {arm}/{seed}/{condition}/{score}/{policy}/{metric}")
                        if arm == "original":
                            old = original_metrics.loc[(seed, condition, score, policy)]
                            if float(old.threshold) != float(row.threshold):
                                raise ValueError("Historical threshold was altered")
                            for metric in calculated:
                                if metric in old.index and not np.isclose(float(old[metric]), float(row[metric]), rtol=1e-8, atol=1e-10, equal_nan=True):
                                    raise ValueError("Historical evaluation changed")
                frames[arm][seed][condition] = frame
                inputs.append(path)
    return metrics, frames, inputs


def all_transitions(frames):
    result = []
    for arm in ARMS:
        frame = normal_transitions(frames[arm])
        frame.insert(0, "arm", arm)
        result.append(frame)
    return pd.concat(result, ignore_index=True)


def counts_to_metrics(counts, n_is=4500, n_oos=1000):
    """Counts end in [OOS TP, IS FP, IS head-correct, IS accepted-correct, draw]."""
    tp, fp, correct, accepted_correct = (counts[..., i, :] for i in range(4))
    return np.stack((fp / n_is, correct / n_is, accepted_correct / n_is,
                     2 * tp / (tp + fp + n_oos), tp / n_oos,
                     1 - fp / n_is, (tp + accepted_correct) / (n_is + n_oos)), axis=-2)


def paired_bootstrap(frames, repeats=2000, random_seed=20261019, batch_size=64):
    if repeats < 2 or batch_size < 1:
        raise ValueError("Invalid bootstrap configuration")
    first = frames[ARMS[0]][SEEDS[0]]["clean"]
    labels, is_oos = first.label_id.to_numpy(), first.is_oos.to_numpy()
    strata = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    if len(strata) != 151 or sum(len(x) for x in strata) != 5500:
        raise ValueError("Expected 150 IS strata plus OOS")
    # arm, seed, condition, event, original text. Every method uses the same draw.
    events = np.zeros((3, 3, 3, 4, 5500), dtype=float)
    for ai, arm in enumerate(ARMS):
        for si, seed in enumerate(SEEDS):
            for ci, condition in enumerate(CONDITIONS):
                frame = frames[arm][seed][condition]
                rejected = frame[f"{FOCUS_SCORE}_{FOCUS_POLICY}_reject"].to_numpy(dtype=bool)
                correct = (frame.predicted_label_id.to_numpy() == labels) & ~is_oos
                events[ai, si, ci] = np.stack((rejected & is_oos, rejected & ~is_oos,
                                              correct, ~rejected & correct))
    # Preserve per-seed nonlinear F1; never pool counts before calculating it.
    point = counts_to_metrics(events.sum(axis=-1)[..., None]).mean(axis=1)[..., 0]
    draws = np.empty((3, 3, len(BOOTSTRAP_METRICS), repeats))
    flattened = events.reshape(-1, 5500)
    rng = np.random.default_rng(random_seed)
    for start in range(0, repeats, batch_size):
        size = min(batch_size, repeats - start)
        sample = np.concatenate([rng.choice(indices, size=(size, len(indices)), replace=True) for indices in strata], axis=1)
        offsets = 5500 * np.arange(size)[:, None]
        weights = np.bincount((sample + offsets).ravel(), minlength=size * 5500).reshape(size, 5500).astype(float)
        counts = (flattened @ weights.T).reshape(*events.shape[:-1], size)
        draws[..., start:start + size] = counts_to_metrics(counts).mean(axis=1)
    records = []
    target = ARMS.index("typo_mix")
    for reference in ("clean_continue", "original"):
        ri = ARMS.index(reference)
        for ci, condition in enumerate(CONDITIONS):
            selected_metrics = BOOTSTRAP_METRICS if condition == "clean" else BOOTSTRAP_METRICS[:3]
            for metric in selected_metrics:
                mi = BOOTSTRAP_METRICS.index(metric)
                distribution = draws[target, ci, mi] - draws[ri, ci, mi]
                low, high = np.quantile(distribution, [.025, .975])
                estimate = point[target, ci, mi] - point[ri, ci, mi]
                records.append({
                    "comparison": "typo_mix_minus_" + reference,
                    "comparison_role": "controlled_training_comparison" if reference == "clean_continue" else "historical_reference",
                    "condition": condition, "score": FOCUS_SCORE, "policy": FOCUS_POLICY, "metric": metric,
                    "estimate_delta": estimate, "ci_low": low, "ci_high": high,
                    "estimate_delta_pp": estimate * 100, "ci_low_pp": low * 100, "ci_high_pp": high * 100,
                    "positive_direction": "more false rejection; worse" if metric == "in_scope_false_rejection_rate" else "higher metric; better",
                    "bootstrap_repeats": repeats, "bootstrap_seed": random_seed,
                    "n_is": 4500, "n_oos": 1000, "n_fixed_models_per_arm": 3,
                    "scope": "fixed checkpoints, train distance statistics and clean validation thresholds",
                    "resampling": "150 IS intents x30 plus 1000 OOS; shared original-text weights across arms, seeds and conditions",
                })
    return pd.DataFrame(records)


def create_plots(summary, intervals, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = summary.set_index(["arm", "condition", "score", "policy"])
    colors = ("#6b7d8f", "#308d95", "#c3792e")
    panels = (("in_scope_false_rejection_rate", "IS false rejection (%)"),
              ("closed_set_accuracy", "Closed-set IS accuracy (%)"),
              ("in_scope_correct_and_accepted_rate", "IS correct and accepted (%)"))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    for ax, (metric, label) in zip(axes, panels):
        for ai, (arm, color) in enumerate(zip(ARMS, colors)):
            rows = [table.loc[(arm, c, FOCUS_SCORE, FOCUS_POLICY)] for c in CONDITIONS]
            means = np.array([r[metric + "_mean"] * 100 for r in rows])
            sd = np.array([r[metric + "_sd"] * 100 for r in rows])
            ax.bar(np.arange(3) + (ai - 1) * .24, means, width=.22, color=color, yerr=sd, capsize=3,
                   label={"original": "Original", "clean_continue": "Clean continue", "typo_mix": "Typo mix"}[arm])
        ax.set_xticks(range(3), [CONDITION_NAMES[c] for c in CONDITIONS])
        ax.set_ylabel(label)
        ax.set_ylim(0, 105)
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
    fig.suptitle("Mahalanobis, frozen clean-validation IS acceptance >=95% | mean +/- sample SD")
    robustness = output / "augmentation_robustness.png"
    fig.savefig(robustness, dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.7), constrained_layout=True)
    for ax, metric in zip(axes, ("oos_f1", "oos_recall", "in_scope_acceptance", "closed_set_accuracy")):
        rows = [table.loc[(arm, "clean", FOCUS_SCORE, FOCUS_POLICY)] for arm in ARMS]
        values = [row[metric + "_mean"] * 100 for row in rows]
        errors = [row[metric + "_sd"] * 100 for row in rows]
        bars = ax.bar(range(3), values, color=colors, yerr=errors, capsize=3)
        ax.set_xticks(range(3), ["Original", "Clean\ncontinue", "Typo\nmix"])
        ax.set_title(metric.replace("_", " "))
        ax.set_ylabel("Clean test (%)")
        ax.set_ylim(0, 108)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=4, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Clean capabilities and rejection trade-offs at the same validation policy")
    clean_plot = output / "augmentation_clean_tradeoff.png"
    fig.savefig(clean_plot, dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), constrained_layout=True)
    for ax, reference in zip(axes, ("clean_continue", "original")):
        rows = intervals[intervals.comparison == "typo_mix_minus_" + reference]
        y = np.arange(len(rows))
        labels = [CONDITION_NAMES[r.condition] + ": " + {
            "in_scope_false_rejection_rate": "IS false rejection", "closed_set_accuracy": "IS accuracy",
            "in_scope_correct_and_accepted_rate": "IS correct + accepted", "oos_f1": "OOS F1",
            "oos_recall": "OOS recall", "in_scope_acceptance": "IS acceptance", "combined_151_accuracy": "System accuracy",
        }[r.metric] for r in rows.itertuples()]
        ax.hlines(y, rows.ci_low_pp, rows.ci_high_pp, color=colors[2], linewidth=2.5)
        ax.scatter(rows.estimate_delta_pp, y, color=colors[2], zorder=3)
        ax.axvline(0, color=colors[0], linestyle="--", linewidth=1)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_title("Typo mix minus " + reference.replace("_", " "))
        ax.set_xlabel("Difference (pp); false rejection: negative is better")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Paired 95% conditional intervals | Mahalanobis, clean-validation IS >=95%")
    paired_plot = output / "augmentation_paired_intervals.png"
    fig.savefig(paired_plot, dpi=180)
    plt.close(fig)
    return robustness, clean_plot, paired_plot


def format_metric(row, metric, multiplier=100):
    mean, sd = row[metric + "_mean"], row[metric + "_sd"]
    return "—" if not np.isfinite(mean) or not np.isfinite(sd) else f"{mean*multiplier:.2f} ± {sd*multiplier:.2f}"


def common_correct_diagnostic_html(diagnostic, cases):
    """Describe post hoc composition changes without interpreting distance scales causally."""
    if (diagnostic["score"] != FOCUS_SCORE or diagnostic["policy"] != FOCUS_POLICY
            or diagnostic["comparison"] != "typo_mix_minus_clean_continue"):
        raise ValueError("Unexpected common-correct diagnostic comparison")
    for path, expected in diagnostic["input_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Common-correct diagnostic source changed: {path}")
    if len(diagnostic["per_seed"]) != 6 or not all(row["common_correct_rejection_delta_pp"] < 0 for row in diagnostic["per_seed"]):
        raise ValueError("The fixed common-correct interpretation no longer matches all six paired results")
    rows, decompositions = [], []
    for row in diagnostic["mean_across_fixed_seeds"]:
        name = CONDITION_NAMES[row["condition"]]
        rows.append([
            name, f"{row['both_correct_n_mean']:.2f}",
            f"{100*row['both_correct_control_rejection_rate_mean']:.2f}% → {100*row['both_correct_mix_rejection_rate_mean']:.2f}%",
            f"{row['both_correct_reject_accept_mean']:.2f} / {row['both_correct_accept_reject_mean']:.2f}",
            f"{row['newly_correct_mix_only_n_mean']:.2f}",
            f"{row['newly_correct_mix_only_mix_rejected_mean']:.2f}（{100*row['newly_correct_mix_only_mix_rejection_rate_mean']:.2f}%）",
        ])
        components = (row["common_correct_rejection_delta_count_mean"],
                      row["newly_correct_mix_only_mix_rejected_mean"],
                      row["lost_correct_control_only_control_rejected_mean"])
        if not np.isclose(components[0] + components[1] - components[2], row["correct_rejected_total_delta_mean"]):
            raise ValueError("Common-correct rejection decomposition does not add up")
        decompositions.append(f"{name}：{components[0]:+.2f} + {components[1]:.2f} − {components[2]:.2f} = {row['correct_rejected_total_delta_mean']:+.2f}条")
    example = cases.loc[cases.row_id.eq(82)].sort_values("condition")
    if set(example.condition) != set(CONDITIONS) or example.true_label.nunique() != 1:
        raise ValueError("Expected one row82 example under all three conditions")
    noisy = example.loc[example.condition.ne("clean")]
    if noisy.text.nunique() != 1:
        raise ValueError("row82 noisy conditions no longer describe one identical example")
    case = noisy.iloc[0]
    seed_outcomes = []
    for seed in SEEDS:
        control_rejected = booleans(noisy[f"clean_continue_seed{seed}_rejected"]).all()
        mix_rejected = booleans(noisy[f"typo_mix_seed{seed}_rejected"])
        if np.unique(mix_rejected).size != 1 or not control_rejected:
            raise ValueError("row82 paired conditions do not match the recorded recovery case")
        seed_outcomes.append(f"seed{seed}：" + ("仍拒绝" if mix_rejected[0] else "恢复接受"))
    accepted_count = int(case["typo_mix_correct_acceptance_seed_count"])
    if accepted_count != sum(not bool(booleans(noisy[f"typo_mix_seed{seed}_rejected"])[0]) for seed in SEEDS):
        raise ValueError("row82 acceptance count is inconsistent")
    if (not all(booleans(noisy[f"original_seed{seed}_rejected"]).all() for seed in SEEDS)
            or any(booleans(example.loc[example.condition.eq("clean"), f"{arm}_seed{seed}_rejected"]).any()
                   for arm in ARMS for seed in SEEDS)):
        raise ValueError("row82 original or clean decisions no longer match the recorded case")
    if not all((example[f"{arm}_seed{seed}_prediction"] == example.true_label).all()
               for arm in ARMS for seed in SEEDS):
        raise ValueError("row82 is not classified correctly by every arm and seed")
    return (
        "<h3>共同正确集与增强才分对集：为什么正确却被拒绝的总数会增加？</h3>"
        "<p>下面在同一扰动文本上配对clean_continue与typo_mix，将两者分类头都正确的“共同正确集”单独观察。"
        "它的误拒率在两种强度、三个seed中都下降；但增强模型额外分对的一批请求大多仍被拒绝，"
        "因此“分类头正确且被拒绝”的总数可以同时上升。不能据总数断言原本正确的请求整体遭到更多拒绝。</p>"
        + htable(["条件", "共同正确数", "该集误拒率：对照→混合", "该集拒→受 / 受→拒数", "增强才分对数", "增强才分对但仍拒绝数（该集占比）"], rows)
        + "<p>表中计数是三个固定seed各自计数的均值；比例先在每seed以对应子集为分母计算，再取均值，并非以4,500为分母。"
        "转移方向为clean对照→混合增强。总“正确且拒绝”差值 = 共同正确集内拒绝变化 + 增强才分对且拒绝 − 对照才分对且拒绝："
        + html.escape("；".join(decompositions))
        + "。共同正确集是按两个模型结果事后筛选的条件子集，这个描述性分解不识别拒识边界的纯因果效果。</p>"
        + f"<p><strong>一条具体恢复案例（row_id=82）</strong>：<code>{html.escape(str(case.original_text))}</code> → "
        + f"<code>{html.escape(str(case.text))}</code>，真实意图为<code>{html.escape(str(case.true_label))}</code>。"
        + "三个分支、三个seed的分类头均预测正确；原句全部接受，扰动后原BERT与clean对照均3/3拒绝。"
        + f"混合增强恢复{accepted_count}/3：{html.escape('；'.join(seed_outcomes))}。"
        + "本例light与medium得到相同文本，只算一例，不是两项独立恢复证据，也不是总体效果估计。"
        + "各模型的表示、中心、协方差和验证阈值均不同，不能用跨模型原始马氏距离下降直接证明表示改善机制。</p>"
    )


def build_html(summary, transition_summary, intervals, plots, inputs, repeats, random_seed, data_summary, evidence, diagnostic_html):
    table = summary.set_index(["arm", "condition", "score", "policy"])
    states = transition_summary.set_index(["arm", "condition", "score", "policy"])
    lines = []
    for condition in CONDITIONS[1:]:
        row = intervals[(intervals.comparison == "typo_mix_minus_clean_continue")
                        & (intervals.condition == condition) & (intervals.metric == "in_scope_false_rejection_rate")].iloc[0]
        target = table.loc[("typo_mix", condition, FOCUS_SCORE, FOCUS_POLICY)]
        control = table.loc[("clean_continue", condition, FOCUS_SCORE, FOCUS_POLICY)]
        lines.append(f"{CONDITION_NAMES[condition]} 下，混合训练/继续clean训练的正常误拒率为 "
                     f"{target['in_scope_false_rejection_rate_mean']*100:.2f}% / {control['in_scope_false_rejection_rate_mean']*100:.2f}%，"
                     f"差值 {row.estimate_delta_pp:+.2f} 个百分点，95%条件区间 [{row.ci_low_pp:+.2f}, {row.ci_high_pp:+.2f}]。")
    noisy_comparisons = intervals[(intervals.comparison == "typo_mix_minus_clean_continue")
                                  & intervals.condition.isin(CONDITIONS[1:])
                                  & (intervals.metric == "in_scope_false_rejection_rate")]
    if (noisy_comparisons.ci_high_pp < 0).all():
        noisy_judgment = "两种扰动的条件区间均低于0，支持本次混合训练减少正常误拒的收益超过只多训练一轮clean的对照。"
    elif (noisy_comparisons.estimate_delta_pp < 0).all():
        noisy_judgment = "两种扰动的误拒均值均下降，但至少一个条件区间不能排除无差异；不能笼统宣称两种强度均已证实改善。"
    else:
        noisy_judgment = "两种扰动没有同时呈现误拒均值下降，需要保留各自结果，不能笼统宣称增强训练成功。"
    retention_metrics = ("oos_f1", "oos_recall", "in_scope_acceptance", "closed_set_accuracy")
    original_clean = table.loc[("original", "clean", FOCUS_SCORE, FOCUS_POLICY)]
    augmented_clean = table.loc[("typo_mix", "clean", FOCUS_SCORE, FOCUS_POLICY)]
    retention_rows, declines = [], []
    for metric in retention_metrics:
        original_mean = float(original_clean[metric + "_mean"])
        augmented_mean = float(augmented_clean[metric + "_mean"])
        delta = augmented_mean - original_mean
        status = "下降" if delta < 0 else "提高" if delta > 0 else "相同"
        retention_rows.append([METRIC_NAMES[metric], f"{original_mean*100:.4f}%", f"{augmented_mean*100:.4f}%",
                               f"{delta*100:+.6g}", status])
        if delta < 0:
            declines.append(f"{METRIC_NAMES[metric]}下降{-delta*100:.6g}个百分点")
    retention_conclusion = (
        "按预先规定的严格描述性规则，clean四项指标的三seed均值相对original全部未下降。"
        "这只是观察到的均值比较，不是非劣效、等效或总体能力无损的统计证明。"
        if not declines else
        "按预先规定的严格描述性规则，不能声称clean四项均无下降：" + "；".join(declines)
        + "。这些下降必须与noisy误拒变化同时作为取舍报告，不事后设置容忍margin或改换指标。"
    )
    training_n = int(data_summary["train_rows"])
    changed_n = int(data_summary["changed_rows"])
    distribution = data_summary["modified_token_counts"]
    policies = data_summary["policy_counts"]
    if sum(int(x) for x in distribution.values()) != training_n or changed_n != training_n - int(distribution["0"]):
        raise ValueError("Training augmentation summary has inconsistent actual mutation counts")
    if sum(int(x) for x in policies.values()) != training_n:
        raise ValueError("Training augmentation assignment counts do not cover all presentations")
    training_table = htable(["原训练样本/位置数", "策略分配clean/light/medium", "实际改变", "实际改变比例", "修改0/1/2词"], [[
        training_n, f"{policies['clean']} / {policies['typo_light']} / {policies['typo_medium']}", changed_n,
        f"{changed_n/training_n*100:.2f}%", f"{distribution['0']} / {distribution['1']} / {distribution['2']}",
    ]])
    clean_rows, robustness_rows, state_rows, transition_rows, full_rows = [], [], [], [], []
    for arm in ARMS:
        row = table.loc[(arm, "clean", FOCUS_SCORE, FOCUS_POLICY)]
        clean_rows.append([ARM_NAMES[arm], *[format_metric(row, metric) for metric in
                           ("closed_set_accuracy", "closed_set_macro_f1", "oos_f1", "oos_recall", "in_scope_acceptance", "in_scope_correct_and_accepted_rate")]])
        for condition in CONDITIONS:
            row = table.loc[(arm, condition, FOCUS_SCORE, FOCUS_POLICY)]
            state = states.loc[(arm, condition, FOCUS_SCORE, FOCUS_POLICY)]
            robustness_rows.append([ARM_NAMES[arm], CONDITION_NAMES[condition], *[format_metric(row, metric) for metric in
                                    ("closed_set_accuracy", "in_scope_false_rejection_rate", "in_scope_correct_and_accepted_rate", "oos_f1", "oos_recall")]])
            state_rows.append([ARM_NAMES[arm], CONDITION_NAMES[condition], *[format_metric(state, name + "_rate_over_all_is") for name in STATES]])
            if condition != "clean":
                transition_rows.append([ARM_NAMES[arm], CONDITION_NAMES[condition], format_metric(state, "clean_accepted_correct_n", 1),
                                        format_metric(state, "clean_ac_to_rejected_correct_n", 1),
                                        format_metric(state, "new_rejection_n", 1), format_metric(state, "recovered_acceptance_n", 1),
                                        format_metric(state, "net_rejection_change_n", 1),
                                        format_metric(state, "new_rejection_rate_over_clean_accepted")])
    for row in summary.itertuples():
        values = pd.Series(row._asdict())
        full_rows.append([ARM_NAMES[row.arm], CONDITION_NAMES[row.condition], SCORE_NAMES[row.score], POLICY_NAMES[row.policy],
                          *[format_metric(values, metric) for metric in ("oos_f1", "oos_recall", "in_scope_false_rejection_rate", "in_scope_correct_and_accepted_rate")]])
    ci_rows = [["受控：混合 − 继续clean" if r.comparison_role == "controlled_training_comparison" else "历史：混合 − 原BERT",
                CONDITION_NAMES[r.condition], METRIC_NAMES[r.metric], f"{r.estimate_delta_pp:+.2f}",
                f"[{r.ci_low_pp:+.2f}, {r.ci_high_pp:+.2f}]"] for r in intervals.itertuples()]
    provenance = htable(["输入", "SHA256"], [[str(path), hashlib.sha256(path.read_bytes()).hexdigest()] for path in inputs])
    audit, sensitivity, verification = evidence["audit"], evidence["sensitivity"], evidence["verification"]
    added_collisions = sum(part.get("added_source_eval_pairs", {}).get("row_pairs", 0)
                           for groups in audit["evaluation_collisions"].values() for part in groups.values())
    if not all(audit["checks"].values()):
        raise ValueError("Independent training-data audit contains a failed check")
    if verification["status"] != "PASS" or verification["normal_transitions"]["status"] != "PASS":
        raise ValueError("Independent experiment or state-transition verification failed")
    if verification["metrics_sha256"] != hashlib.sha256(inputs[0].read_bytes()).hexdigest():
        raise ValueError("Verification refers to a different metrics table")
    direct_checks = verification["distance_direct_solve_checks"]
    checked_vectors = sum(len(row["row_ids"]) for row in direct_checks)
    max_distance_error = max(row["max_abs_distance_error"] for row in direct_checks)
    sensitivity_rows = []
    for row in sensitivity["mean_method_differences"]:
        if (row["comparison"] == "typo_mix_minus_clean_continue" and row["condition"] in CONDITIONS[1:]
                and row["metric"] == "in_scope_false_rejection_rate"):
            sensitivity_rows.append([CONDITION_NAMES[row["condition"]], f"{row['official_delta_pp']:+.4f}",
                                     f"{row['excluded_delta_pp']:+.4f}"])
    evidence_html = (
        f"<p>独立训练数据审计检查了两个分支共 {audit['checked_training_rows']:,} 个数据位置，确认每个原训练样本每臂只呈现一次、"
        f"标签与原train一致、每类50/25/25分配和实际扰动可独立重现。新增精确/规范化训练—评价文本碰撞计数为 {added_collisions}。"
        "这不消除原数据的重叠或模型预训练阶段的未知污染。</p>"
        f"<p>冻结评价敏感性对所有分支、seed和三种条件使用相同原始row_id掩码，排除 "
        f"{html.escape(', '.join(str(x) for x in sensitivity['excluded_original_row_ids']))} 两条原train重叠测试文本。"
        f"各项方法差值的最大变化仅为 {sensitivity['max_absolute_change_in_method_difference_pp']:.4f} 个百分点。"
        "这是只重算评价的敏感性检查，没有重训、重拟合距离或重选阈值；原train–validation重叠及历史test曝光仍然存在。"
        "混合训练当前文本少一个精确test重叠，也不代表该信息未进入原checkpoint或clean训练距离统计。</p>"
        + htable(["条件", "混合−clean对照误拒差值 pp：原评估", "排除重叠后 pp"], sensitivity_rows)
        + f"<p>独立验证状态 {verification['status']}：复算 {verification['verified_test_settings']} 组test指标、"
        f"{verification['normal_transitions']['verified_rows']} 行状态转移；核对 {verification['validation_operating_points']} 个验证运行点及 "
        f"{verification['validation_candidates_checked']:,} 个候选阈值。对 {checked_vectors} 个真实CLS向量逐一计算全部150类中心的直接协方差解，"
        f"最大距离绝对差 {max_distance_error:.2e}；训练账本、配对顺序、冻结源文件哈希均由独立验证记录核对。</p>"
    )
    generated = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M UTC+08:00")
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BERT 拼写增强：受控继续训练实验</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f6f7;color:#193746;font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1240px;margin:28px auto;padding:34px 40px;background:white;border:1px solid #dbe5e8;border-radius:16px}}
h1{{font-size:29px;line-height:1.4}}h2{{font-size:23px;margin-top:34px}}h3{{font-size:18px}}
.meta{{color:#5c7280;font-size:13px}}.lead{{background:#fff5e9;border-left:5px solid #c3792e;padding:16px 20px;font-size:18px}}
.table-scroll{{overflow-x:auto;margin:16px 0}}table{{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}}
th,td{{padding:9px 10px;text-align:left;vertical-align:top;border-bottom:1px solid #dce5e8}}th{{background:#eaf0f3;white-space:nowrap}}td{{overflow-wrap:anywhere}}
img{{max-width:100%;height:auto;border:1px solid #dce5e8;border-radius:8px;margin:16px 0}}summary{{cursor:pointer;font-weight:650}}details{{margin:18px 0}}
@media(max-width:700px){{main{{padding:24px 16px;margin:0;border-radius:0}}h1{{font-size:24px}}}}
</style><main><p class="meta">MFE310TC · 探索性受控继续训练 · {generated}</p>
<h1>拼写增强训练能否减少正常请求误拒，并保留干净输入能力？</h1>
<p class="lead">{html.escape(' '.join(lines))}</p>
<p>{html.escape(noisy_judgment)} {html.escape(retention_conclusion)}</p>
<p>关键对照是<strong>混合 typo 训练 − 同样多训练一轮的 clean 对照</strong>。只和旧BERT比较会同时混入额外训练的影响。
历史“混合训练 − 原BERT”另列，不能替代受控对照。以下核心表聚焦Mahalanobis、clean validation接受率≥95%的运行点，
其余分数与阈值策略完整保留为补充，不预设所有方法均改善。</p>
<h2>训练对照与冻结边界</h2>
<p>这是从原已微调模型继续一轮的pilot实验，不是从预训练底座完整重训。
seeds 42/43/44 各自从原checkpoint继续训练，两个分支均固定1 epoch、每条原训练样本只出现一次，共15,000个位置。
clean_continue只使用原文；typo_mix在每类中随机分配50% clean、25% light、25% medium。
这是分配策略，缺少合格词时实际修改数会减少，不能把25% medium直接等同于25%的输入都修改了两个词。</p>
{training_table}
<p>上表直接读取data/data_summary.json。增强分配和实际改变分开记录；未改变样本包括clean分配和无法实际修改的噪声分配。</p>
<p>共同设置为学习率1e-5、batch16、AdamW weight decay0.01、warmup比例0.1、gradient clip1；两个继续训练分支统一使用BF16，推理使用FP32。
两臂具有相同的样本呈现次数与更新步数，但扰动可能改变WordPiece长度和padding，因此不声称严格相同FLOPs。
新模型分别用原始clean train重新拟合CLS类中心与Ledoit–Wolf共享协方差，所有分数的两个阈值策略仅在clean validation选择。
六个模型完成训练及clean validation冻结后才评价三种test条件，固定一轮，没有依据本次test改训练轮数或阈值。
因此主对照评估的是这套增强训练及其按相同规则重新拟合的拒识系统，不是固定旧模型只换测试预处理。</p>
<p>混合干净与噪声训练的研究动机可参考<a href="https://aclanthology.org/2020.acl-main.138/" target="_blank" rel="noopener noreferrer">相关原始研究（ACL 2020）</a>。
该文研究的是sequence labeling（序列标注）；本实验是意图分类与未知拒识，使用自己的固定比例与继续训练方案，并非该论文的精确复现。</p>
<h2>干净输入能力是否保留？</h2>
<p>以下为三个seed均值±样本标准差（ddof=1），百分比指标的SD单位为百分点。
“保留”的预设描述性规则是：在mahalanobis/95策略下，clean OOS F1、OOS recall、IS接受率、闭集Accuracy四项均值相对original全部不下降，才说观察到无下降。
任一项下降都逐项报告，不事后引入容忍margin，也不把区间跨0等同于已经证明等效。</p>
<p><strong>{html.escape(retention_conclusion)}</strong></p>
{htable(['预设clean指标','original均值','typo_mix均值','混合−original pp','描述性方向'],retention_rows)}
{htable(['分支','闭集Accuracy %','闭集Macro-F1 %','OOS F1 %','OOS recall %','正常接受率 %','正常正确且接受 %'],clean_rows)}
{image_tag(plots[1])}
<h2>相同拼写扰动下的表现</h2>
{htable(['分支','条件','闭集Accuracy %','IS误拒率 %','IS正确且接受 %','OOS F1 %','OOS recall %'],robustness_rows)}
{image_tag(plots[0])}
<p>闭集accuracy评价所有正常文本的原分类头预测；IS误拒率把所有仍按原标签视作正常的请求被拒绝计为服务损失。
正确且接受率以全部4,500条IS为分母。接受子集准确率变高可能只因拒绝更多困难输入，不能单独作为鲁棒性改善证据。
95%只约束各模型的clean validation，不保证clean或noisy test也达到95%。</p>
<h3>正常请求四状态</h3>
{htable(['分支','条件',*[name+' /4500 %' for name in STATE_NAMES]],state_rows)}
<p>四状态互斥且覆盖全部4,500条IS。正确/错误指当前条件下的分类头，不指拒绝动作是否“正确”；
“拒绝但分类头正确”直接显示本可给出正确意图但未提供服务的请求。</p>
{diagnostic_html}
<h3>从各自clean状态出发的服务流失</h3>
{htable(['分支','条件','自身clean正确且接受数','该池→拒绝但头正确','新误拒数','恢复接受数','净增加拒绝数','新误拒/自身clean接受IS %'],transition_rows)}
<p>每个模型以自身clean决策为起点，来源池随分支而变化，不能把不同来源池的流向计数直接当成固定同一子集上的训练因果效应。
新误拒=clean接受→扰动拒绝；恢复接受=clean拒绝→扰动接受；净增加拒绝=new−recovered。
完整四方向流量及全部分母保存在normal_transitions.csv；跨分支主比较仍使用相同4,500条原始IS文本。</p>
<h2>配对差值与条件区间</h2>
<p>差值统一为混合训练减参照；正常误拒率为负更好，其余指标为正更好。clean与noisy结果共同判断，不事后挑选表现最好的一项。</p>
{htable(['比较','条件','指标','差值 pp','95%条件区间 pp'],ci_rows)}{image_tag(plots[2])}
<p>{repeats:,}次bootstrap，随机种子{random_seed}：150个IS intent各有放回抽取30条，OOS抽取1,000条。
同一原始文本权重共同用于所有训练分支、seed和条件。每seed先独立计算指标（包括非线性的F1），再求三模型平均并作差；
不把重复seed预测视为独立文本。区间取2.5%与97.5%分位数，条件于固定模型、训练统计及clean validation阈值，
不包含训练随机性总体、验证阈值选择或重新拟合的不确定性；多个比较未做多重检验校正。</p>
<h2>解释范围</h2>
<p>测试集此前已多次查看，本次是探索性后续研究，不能称为新的未曝光确认集。训练增强与测试使用同一种字符扰动家族，
即使结果改善，也只支持该家族下的稳健性，不能直接声称泛化到自然拼写、语音转写或任意分布变化。
扰动继承原标签，不是人工逐例确认的语义保持改写。旧快照和原clean拒识实验主终点保持原样。</p>
<details><summary>全部分数与两种阈值策略</summary>{htable(['分支','条件','分数','clean验证策略','OOS F1 %','OOS recall %','IS误拒 %','IS正确且接受 %'],full_rows)}</details>
<p>报告生成前从27份逐样本预测独立复算162组指标，检查原文、标签和扰动跨分支/seed对齐、阈值跨test条件冻结，
并核对历史原BERT指标没有改变。数字一致不证明自然语义保持或真实部署有效。</p>
<h3>独立数据审计、重复敏感性与验证</h3>{evidence_html}
<details><summary>输入与SHA256</summary>{provenance}</details>
<p class="meta">产物：summary_mean_sd.csv、normal_transitions.csv、paired_intervals.csv及三张PNG。HTML图片已内嵌。
AI协助实现、统计检查与整理；学生应核验解释并按课程要求披露实际AI用途和本人贡献。</p>
</main></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261019)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    output = args.output.resolve()
    metrics, frames, inputs = load_results(output)
    data_summary_path = PROJECT / "data" / "data_summary.json"
    data_summary = json.loads(data_summary_path.read_text(encoding="utf-8"))
    inputs.append(data_summary_path)
    evidence_paths = {"audit": PROJECT / "data_audit.json", "sensitivity": output / "duplicate_sensitivity.json",
                      "verification": output / "verification.json"}
    evidence = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in evidence_paths.items()}
    inputs.extend(evidence_paths.values())
    diagnostic_path = output / "common_correct_rejection_diagnostic.json"
    cases_path = output / "illustrative_recoveries.csv"
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    cases = pd.read_csv(cases_path, float_precision="round_trip")
    diagnostic_html = common_correct_diagnostic_html(diagnostic, cases)
    inputs.extend([diagnostic_path, cases_path])
    summary = aggregate(metrics, ["arm", "condition", "score", "policy"])
    transitions = all_transitions(frames)
    transition_summary = aggregate(transitions, ["arm", "condition", "score", "policy"])
    intervals = paired_bootstrap(frames, args.bootstrap_repeats, args.bootstrap_seed, args.batch_size)
    plots = create_plots(summary, intervals, output)
    document = build_html(summary, transition_summary, intervals, plots, inputs, args.bootstrap_repeats, args.bootstrap_seed, data_summary, evidence, diagnostic_html)
    summary.to_csv(output / "summary_mean_sd.csv", index=False)
    transitions.to_csv(output / "normal_transitions.csv", index=False)
    intervals.to_csv(output / "paired_intervals.csv", index=False)
    report = output / "augmentation_report_cn.html"
    report.write_text(document, encoding="utf-8")
    print(json.dumps({"status": "PASS", "verified_settings": 162, "transitions": len(transitions),
                      "paired_intervals": len(intervals), "bootstrap_repeats": args.bootstrap_repeats,
                      "report": str(report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
