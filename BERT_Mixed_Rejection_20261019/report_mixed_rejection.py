"""Report a frozen-BERT experiment changing only rejection-distribution fitting.

Run after TEST_COMPLETE.json exists. Reads frozen predictions; never trains,
extracts embeddings, fits a detector, or chooses thresholds. Writes derived
tables, scientific PNGs and a self-contained Chinese HTML report only.
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
REJECTION = PROJECT.parent / "BERT_Rejection_Study_20261019"
AUGMENTATION = PROJECT.parent / "BERT_Typo_Augmentation_20261019"
sys.path.insert(0, str(REJECTION))
from report_rejection import SEEDS, POLICIES, POLICY_NAMES, booleans, evaluate_predictions, htable, image_tag
from report_typo_rejection import CONDITIONS, CONDITION_NAMES, aggregate

ARMS = ("clean_only", "clean_duplicate", "clean_plus_typo")
ARM_NAMES = {"clean_only": "原clean拟合", "clean_duplicate": "clean重复两遍", "clean_plus_typo": "clean＋新typo拟合"}
ARM_EN = {"clean_only": "Clean only", "clean_duplicate": "Clean duplicate", "clean_plus_typo": "Clean + typo"}
FOCUS_POLICY = "is_acceptance_95"
METRICS = ("in_scope_false_rejection_rate", "in_scope_correct_and_accepted_rate", "oos_f1",
           "oos_recall", "in_scope_acceptance", "combined_151_accuracy")
NAMES = {"in_scope_false_rejection_rate": "IS误拒率", "in_scope_correct_and_accepted_rate": "IS正确且接受率",
         "oos_f1": "OOS F1", "oos_recall": "OOS recall", "in_scope_acceptance": "IS接受率",
         "combined_151_accuracy": "混合系统准确率", "closed_set_accuracy": "闭集Accuracy"}
STATES = ("accepted_correct", "accepted_wrong", "rejected_correct", "rejected_wrong")
STATE_NAMES = ("接受／头正确", "接受／头错误", "拒绝／头正确", "拒绝／头错误")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_hashes(value):
    """Collect absolute-path SHA256 maps without assuming manifest metadata names."""
    found = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and len(item) == 64 and Path(key).is_absolute():
                found[str(Path(key).resolve())] = item
            elif isinstance(item, (dict, list)):
                found.update(manifest_hashes(item))
    elif isinstance(value, list):
        for item in value:
            found.update(manifest_hashes(item))
    return found


def load_results(output):
    manifests = [PROJECT / "INPUTS_FROZEN.json", output / "DEVELOPMENT_FROZEN.json", output / "TEST_COMPLETE.json"]
    frozen_hashes = {}
    for path in manifests:
        frozen_hashes.update(manifest_hashes(json.loads(path.read_text(encoding="utf-8"))))
    metric_path = output / "metrics_by_seed.csv"
    metrics = pd.read_csv(metric_path, float_precision="round_trip")
    keys = ["arm", "seed", "condition", "score", "policy"]
    if len(metrics) != 54 or metrics.duplicated(keys).any():
        raise ValueError("Expected 54 unique frozen test settings")
    for name, values in (("arm", ARMS), ("seed", SEEDS), ("condition", CONDITIONS),
                         ("score", ("mahalanobis",)), ("policy", POLICIES)):
        if set(metrics[name]) != set(values):
            raise ValueError(f"Unexpected {name} values")
    history_path = AUGMENTATION / "outputs" / "metrics_by_seed.csv"
    history = pd.read_csv(history_path, float_precision="round_trip")
    history = history[(history.arm == "typo_mix") & (history.score == "mahalanobis")].set_index(["seed", "condition", "policy"])
    inputs = [metric_path, history_path, *manifests]
    frames, identity_reference, shared = {}, None, {}
    identity = ["row_id", "original_text", "true_label", "label_id", "is_oos"]
    shared_columns = identity + ["text", "changed", "modified_tokens", "predicted_label_id"]
    for arm in ARMS:
        frames[arm] = {}
        for seed in SEEDS:
            frames[arm][seed] = {}
            for condition in CONDITIONS:
                path = output / arm / f"seed{seed}" / f"{condition}_predictions.csv"
                frame = pd.read_csv(path, float_precision="round_trip")
                required = set(shared_columns + ["mahalanobis_score"] + [f"mahalanobis_{p}_reject" for p in POLICIES])
                if not required.issubset(frame) or frame[list(required)].isna().any().any() or frame.row_id.duplicated().any():
                    raise ValueError(f"Invalid prediction schema: {path}")
                frame["is_oos"], frame["changed"] = booleans(frame.is_oos), booleans(frame.changed)
                if identity_reference is None:
                    identity_reference = frame[identity].copy()
                else:
                    if set(frame.row_id) != set(identity_reference.row_id):
                        raise ValueError("Original row IDs differ")
                    frame = frame.set_index("row_id").loc[identity_reference.row_id].reset_index()
                    if not frame[identity].equals(identity_reference):
                        raise ValueError("Original labels or text identity differ")
                if arm == ARMS[0]:
                    shared[(seed, condition)] = frame[shared_columns].copy()
                elif not frame[shared_columns].equals(shared[(seed, condition)]):
                    raise ValueError("Classifier predictions or test texts changed across detector arms")
                labels, oos = frame.label_id.to_numpy(), frame.is_oos.to_numpy()
                predicted = frame.predicted_label_id.to_numpy()
                _, counts = np.unique(labels[~oos], return_counts=True)
                if (len(frame) != 5500 or oos.sum() != 1000 or len(counts) != 150 or not np.all(counts == 30)
                        or not np.array_equal(np.unique(labels[~oos]), np.arange(150)) or not np.all(labels[oos] == -1)):
                    raise ValueError("Expected 150 intents x30 IS and 1000 OOS")
                if not np.array_equal(frame.changed, frame.text != frame.original_text):
                    raise ValueError("Changed flags disagree with texts")
                values = frame.mahalanobis_score.to_numpy(dtype=float)
                if not np.isfinite(values).all():
                    raise ValueError("Non-finite Mahalanobis scores")
                closed = float((predicted[~oos] == labels[~oos]).mean())
                closed_f1 = float(f1_score(labels[~oos], predicted[~oos], labels=np.arange(150), average="macro"))
                for policy in POLICIES:
                    row = metrics[(metrics.arm == arm) & (metrics.seed == seed) & (metrics.condition == condition)
                                  & (metrics.policy == policy)].iloc[0]
                    frozen = metrics[(metrics.arm == arm) & (metrics.seed == seed) & (metrics.policy == policy)]
                    if frozen.threshold.nunique(dropna=False) != 1 or pd.isna(row.threshold):
                        raise ValueError("Test conditions use different thresholds")
                    column = f"mahalanobis_{policy}_reject"
                    rejected = booleans(frame[column])
                    frame[column] = rejected
                    if not np.array_equal(rejected, values >= float(row.threshold)):
                        raise ValueError("Rejection decisions disagree with threshold")
                    calculated = evaluate_predictions(labels, predicted, oos, values, rejected)
                    calculated.update(closed_set_accuracy=closed, closed_set_macro_f1=closed_f1)
                    for metric, actual in calculated.items():
                        if metric not in row or not np.isclose(actual, float(row[metric]), rtol=1e-8, atol=1e-10, equal_nan=True):
                            raise ValueError(f"Metric mismatch: {arm}/{seed}/{condition}/{policy}/{metric}")
                    if arm == "clean_only":
                        old = history.loc[(seed, condition, policy)]
                        if float(old.threshold) != float(row.threshold):
                            raise ValueError("Original enhanced-BERT clean detector threshold changed")
                        for metric in calculated:
                            if not np.isclose(float(old[metric]), float(row[metric]), rtol=1e-8, atol=1e-10, equal_nan=True):
                                raise ValueError(f"clean_only does not reproduce enhanced-BERT baseline: {metric}")
                if arm == "clean_only":
                    old_path = AUGMENTATION / "outputs" / "typo_mix" / f"seed{seed}" / f"{condition}_predictions.csv"
                    old = pd.read_csv(old_path, float_precision="round_trip").set_index("row_id").loc[frame.row_id].reset_index()
                    if not np.array_equal(old.predicted_label_id, predicted) or not np.array_equal(old.mahalanobis_score, values):
                        raise ValueError("clean_only classifier or scores differ from frozen enhanced-BERT results")
                    inputs.append(old_path)
                frames[arm][seed][condition] = frame
                inputs.append(path)
    for _, group in metrics.groupby(["seed", "condition"]):
        if group.closed_set_accuracy.nunique() != 1 or group.closed_set_macro_f1.nunique() != 1:
            raise ValueError("Closed-set metrics must be exactly identical across all detector arms and policies")
    # Report only hashes of consumed inputs; the independent verifier audits the full chain.
    for path in inputs:
        expected = frozen_hashes.get(str(path.resolve()))
        if expected is not None and sha256(path) != expected:
            raise ValueError(f"Frozen consumed input changed: {path}")
    return metrics, frames, inputs


def states_and_transitions(frames):
    states, transitions = [], []
    for arm in ARMS:
        for seed in SEEDS:
            for condition in CONDITIONS:
                frame = frames[arm][seed][condition]
                is_scope = ~frame.is_oos.to_numpy()
                correct = frame.predicted_label_id.to_numpy() == frame.label_id.to_numpy()
                for policy in POLICIES:
                    rejected = frame[f"mahalanobis_{policy}_reject"].to_numpy()
                    masks = (~rejected & correct, ~rejected & ~correct, rejected & correct, rejected & ~correct)
                    row = dict(arm=arm, seed=seed, condition=condition, score="mahalanobis", policy=policy, n_is=4500)
                    for state, mask in zip(STATES, masks):
                        row[state + "_n"] = int((mask & is_scope).sum())
                        row[state + "_rate_over_all_is"] = row[state + "_n"] / 4500
                    if sum(row[state + "_n"] for state in STATES) != 4500:
                        raise ValueError("IS states do not cover all requests")
                    states.append(row)
    for reference in ARMS[:2]:
        for seed in SEEDS:
            for condition in CONDITIONS:
                target, baseline = frames[ARMS[2]][seed][condition], frames[reference][seed][condition]
                scope = ~target.is_oos.to_numpy()
                correct = target.predicted_label_id.to_numpy() == target.label_id.to_numpy()
                for policy in POLICIES:
                    source = baseline[f"mahalanobis_{policy}_reject"].to_numpy()
                    dest = target[f"mahalanobis_{policy}_reject"].to_numpy()
                    for stratum, mask in (("all_is", scope), ("head_correct", scope & correct), ("head_wrong", scope & ~correct)):
                        n = int(mask.sum())
                        row = dict(comparison="clean_plus_typo_minus_" + reference, reference=reference, seed=seed,
                                   condition=condition, score="mahalanobis", policy=policy, stratum=stratum,
                                   n_is=4500, stratum_n=n, source_accepted_n=int((mask & ~source).sum()),
                                   source_rejected_n=int((mask & source).sum()))
                        for name, event in (("accept_accept", ~source & ~dest), ("accept_reject", ~source & dest),
                                            ("reject_accept", source & ~dest), ("reject_reject", source & dest)):
                            count = int((mask & event).sum())
                            row[name + "_n"] = count
                            row[name + "_rate_over_all_is"] = count / 4500
                            row[name + "_rate_over_stratum"] = count / n if n else np.nan
                        row["net_rejection_change_n"] = row["accept_reject_n"] - row["reject_accept_n"]
                        row["net_rejection_change_rate_over_all_is"] = row["net_rejection_change_n"] / 4500
                        if sum(row[name + "_n"] for name in ("accept_accept", "accept_reject", "reject_accept", "reject_reject")) != n:
                            raise ValueError("Decision flows do not partition their fixed-head stratum")
                        transitions.append(row)
    return pd.DataFrame(states), pd.DataFrame(transitions)


def counts_to_metrics(counts):
    tp, fp, accepted_correct = (counts[..., i, :] for i in range(3))
    return np.stack((fp / 4500, accepted_correct / 4500, 2 * tp / (tp + fp + 1000),
                     tp / 1000, 1 - fp / 4500, (tp + accepted_correct) / 5500), axis=-2)


def paired_bootstrap(frames, repeats=2000, random_seed=20261020, batch_size=64):
    if repeats < 2 or batch_size < 1:
        raise ValueError("Invalid resampling settings")
    first = frames[ARMS[0]][SEEDS[0]]["clean"]
    labels, oos = first.label_id.to_numpy(), first.is_oos.to_numpy()
    strata = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    # arm, seed, condition, threshold policy, event, original text
    events = np.zeros((3, 3, 3, 2, 3, 5500), dtype=float)
    for ai, arm in enumerate(ARMS):
        for si, seed in enumerate(SEEDS):
            for ci, condition in enumerate(CONDITIONS):
                frame = frames[arm][seed][condition]
                correct = (frame.predicted_label_id.to_numpy() == labels) & ~oos
                for pi, policy in enumerate(POLICIES):
                    rejected = frame[f"mahalanobis_{policy}_reject"].to_numpy()
                    events[ai, si, ci, pi] = np.stack((rejected & oos, rejected & ~oos, ~rejected & correct))
    point = counts_to_metrics(events.sum(axis=-1)[..., None]).mean(axis=1)[..., 0]
    draws = np.empty((*point.shape, repeats))
    flat, rng = events.reshape(-1, 5500), np.random.default_rng(random_seed)
    for start in range(0, repeats, batch_size):
        size = min(batch_size, repeats - start)
        sample = np.concatenate([rng.choice(indices, size=(size, len(indices)), replace=True) for indices in strata], axis=1)
        weights = np.bincount((sample + 5500 * np.arange(size)[:, None]).ravel(), minlength=size * 5500).reshape(size, 5500).astype(float)
        counts = (flat @ weights.T).reshape(*events.shape[:-1], size)
        draws[..., start:start + size] = counts_to_metrics(counts).mean(axis=1)
    rows = []
    for ri, reference in enumerate(ARMS[:2]):
        for ci, condition in enumerate(CONDITIONS):
            for pi, policy in enumerate(POLICIES):
                for mi, metric in enumerate(METRICS):
                    delta = draws[2, ci, pi, mi] - draws[ri, ci, pi, mi]
                    low, high = np.quantile(delta, [.025, .975])
                    estimate = point[2, ci, pi, mi] - point[ri, ci, pi, mi]
                    rows.append(dict(comparison="clean_plus_typo_minus_" + reference,
                                     comparison_role="main_comparison" if ri == 0 else "duplicate_count_control",
                                     condition=condition, score="mahalanobis", policy=policy, metric=metric,
                                     estimate_delta=estimate, ci_low=low, ci_high=high,
                                     estimate_delta_pp=100*estimate, ci_low_pp=100*low, ci_high_pp=100*high,
                                     bootstrap_repeats=repeats, bootstrap_seed=random_seed, n_is=4500, n_oos=1000,
                                     n_fixed_models=3, scope="fixed shared classifiers, detector fits and clean-validation thresholds",
                                     resampling="150 intent strata x30 plus 1000 OOS; same original-text weights across arms, seeds, conditions and policies"))
    return pd.DataFrame(rows)


def create_plots(summary, intervals, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    table = summary.set_index(["arm", "condition", "score", "policy"])
    colors = ("#64788b", "#329199", "#c17c36")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    panels = (("in_scope_false_rejection_rate", "IS false rejection (%)"),
              ("in_scope_correct_and_accepted_rate", "IS correct and accepted (%)"),
              ("oos_recall", "OOS recall (%)"))
    for ax, (metric, title) in zip(axes, panels):
        for ai, arm in enumerate(ARMS):
            rows = [table.loc[(arm, c, "mahalanobis", FOCUS_POLICY)] for c in CONDITIONS]
            ax.bar(np.arange(3) + (ai-1)*.24, [100*r[metric+"_mean"] for r in rows], width=.22,
                   yerr=[100*r[metric+"_sd"] for r in rows], capsize=3, color=colors[ai], label=ARM_EN[arm])
        ax.set_xticks(range(3), ["Clean", "Light", "Medium"])
        ax.set_ylabel(title)
        ax.set_ylim(0, 105)
        ax.spines[["right", "top"]].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
    fig.suptitle("Same frozen BERT; different detector-fit data | clean-validation IS >=95% | mean +/- sample SD")
    robustness = output / "mixed_rejection_robustness.png"
    fig.savefig(robustness, dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), constrained_layout=True)
    for ax, metric in zip(axes, ("oos_f1", "oos_recall", "in_scope_acceptance")):
        rows = [table.loc[(arm, "clean", "mahalanobis", FOCUS_POLICY)] for arm in ARMS]
        values = [100*r[metric+"_mean"] for r in rows]
        bars = ax.bar(range(3), values, color=colors, yerr=[100*r[metric+"_sd"] for r in rows], capsize=3)
        ax.set_xticks(range(3), ["Clean only", "Clean\nduplicate", "Clean + typo"])
        ax.set_ylabel("Clean test (%)")
        ax.set_title({"oos_f1":"OOS F1", "oos_recall":"OOS recall", "in_scope_acceptance":"IS acceptance"}[metric])
        ax.set_ylim(0, 105)
        ax.bar_label(bars, labels=[f"{x:.2f}" for x in values], padding=4, fontsize=9)
        ax.spines[["right", "top"]].set_visible(False)
    fig.suptitle("Clean retention relative to the current enhanced-BERT clean-only detector")
    clean = output / "mixed_rejection_clean_tradeoff.png"
    fig.savefig(clean, dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), constrained_layout=True)
    for ax, reference in zip(axes, ARMS[:2]):
        chosen = intervals[(intervals.comparison == "clean_plus_typo_minus_"+reference) & (intervals.policy == FOCUS_POLICY)
                           & (((intervals.condition == "clean") & intervals.metric.isin(("oos_f1", "oos_recall", "in_scope_acceptance")))
                              | ((intervals.condition != "clean") & intervals.metric.isin(("in_scope_false_rejection_rate", "in_scope_correct_and_accepted_rate", "oos_recall"))))]
        labels = [r.condition.replace("typo_", "").title()+": "+{"oos_f1":"OOS F1", "oos_recall":"OOS recall", "in_scope_acceptance":"IS acceptance", "in_scope_false_rejection_rate":"IS false rejection", "in_scope_correct_and_accepted_rate":"IS correct + accepted"}[r.metric] for r in chosen.itertuples()]
        y = np.arange(len(chosen))
        ax.hlines(y, chosen.ci_low_pp, chosen.ci_high_pp, color=colors[2], linewidth=2.5)
        ax.scatter(chosen.estimate_delta_pp, y, color=colors[2], zorder=3)
        ax.axvline(0, color=colors[0], linestyle="--", linewidth=1)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_title("Clean + typo minus " + ARM_EN[reference])
        ax.set_xlabel("Difference (pp); false rejection: negative is better")
        ax.spines[["right", "top"]].set_visible(False)
    fig.suptitle("Paired 95% conditional intervals | clean-validation IS >=95%")
    paired = output / "mixed_rejection_paired_intervals.png"
    fig.savefig(paired, dpi=180)
    plt.close(fig)
    return robustness, clean, paired


def formatted(row, name, multiplier=100):
    return f"{row[name+'_mean']*multiplier:.2f} ± {row[name+'_sd']*multiplier:.2f}"


def clean_guardrails(summary):
    table = summary.set_index(["arm", "condition", "score", "policy"])
    rows = []
    for metric in ("oos_f1", "oos_recall", "in_scope_acceptance"):
        source, target = (float(table.loc[(arm, "clean", "mahalanobis", FOCUS_POLICY)][metric+"_mean"])
                          for arm in ("clean_only", "clean_plus_typo"))
        rows.append(dict(reference="clean_only", target="clean_plus_typo", condition="clean", score="mahalanobis",
                         policy=FOCUS_POLICY, metric=metric, reference_mean=source, target_mean=target,
                         delta=target-source, delta_pp=100*(target-source), mean_non_decline=target >= source))
    result = pd.DataFrame(rows)
    result["all_three_pass"] = bool(result.mean_non_decline.all())
    result["rule"] = "all three target means >= reference means; no tolerance margin; descriptive only"
    return result


def evidence_summary(evidence, inputs):
    sections = []
    audit = evidence.get("data_audit.json")
    if audit:
        if not all(audit["checks"].values()):
            raise ValueError("Independent training-data audit includes a failed check")
        extra_masks = audit["sensitivity_masks"]["normalized"]["additional_row_ids_beyond_original_clean_overlap"]
        sections.append(
            f"<p>独立数据审计复现{audit['checked_noisy_rows']:,}条新noise及{audit['checked_manifest_rows']:,}条拟合清单，"
            "核对每类50/50分配、逐条变体、原标签与来源权重。新扰动未增加相对验证/测试的精确或规范化文本碰撞；"
            f"新增测试排除row_id数为{len(extra_masks)}。仍保留原数据3处train–validation及2处train–test重叠，"
            "不能把“无新增碰撞”说成完全无泄漏，也不能排除语义重复或预训练污染。</p>"
        )
    verification = evidence.get("verification.json")
    if verification:
        if verification["status"] != "PASS" or verification["metrics_sha256"] != sha256(inputs[0]):
            raise ValueError("Independent verification failed or refers to different metrics")
        direct = verification["direct_covariance_solve_checks"]
        vectors = sum(len(item["row_ids"]) for item in direct)
        max_error = max(item["max_abs_distance_error"] for item in direct)
        flow = verification["normal_transitions"]
        sections.append(
            f"<p>独立核验PASS：{verification['verified_test_settings']}个test设置、"
            f"{verification['verified_validation_settings']}个validation运行点及{verification['validation_candidates_checked']:,}个候选阈值；"
            f"独立重算{len(verification['independent_fit_statistics'])}组中心、协方差和Ledoit–Wolf收缩系数。"
            f"{flow['verified_state_rows']}行四状态、{flow['verified_transition_rows']}行流转与3项guardrail通过；"
            f"{vectors}条CLS逐一对150中心直接解协方差，最大误差{max_error:.2e}。"
            "同一输入的分类预测保持一致，clean_only与上一轮分数完全相同；冻结源文件哈希核对通过。</p>"
        )
        fallback = verification["unchanged_noisy_training_fallback_checks"]
        sections.append(
            "<details><summary>未改变文本的浮点核验</summary><p>每seed有"
            f"{fallback[0]['unchanged_text_rows']:,}条noise fallback保持原文。不同动态padding批次下，"
            f"最大CLS差{max(r['max_abs_cls_delta'] for r in fallback):.2e}、最大logit差"
            f"{max(r['max_abs_logit_delta'] for r in fallback):.2e}，分类argmax变化"
            f"{sum(r['classifier_argmax_changes'] for r in fallback)}条。记录微小浮点差异，不要求缓存向量逐位相同。</p></details>"
        )
    sensitivity = evidence.get("duplicate_sensitivity.json")
    if sensitivity:
        excluded = ", ".join(str(x) for x in sensitivity["original_train_test_excluded_row_ids"])
        sections.append(
            f"<p>重复敏感性对所有分支、seed、条件和两策略使用同一原始row_id排除掩码[{excluded}]，"
            f"方法差值的最大绝对变化为{sensitivity['max_absolute_change_in_method_difference_pp']:.6f} pp。"
            "无新增精确碰撞，因此没有用相同掩码另造一份重复敏感性表。"
            "这仅重算固定输出评价，没有重新拟合或选阈值；不能撤销此前训练、拟合、验证重叠或测试曝光的影响。</p>"
        )
    return "".join(sections)


def mechanism_summary(evidence):
    diagnostic = evidence.get("fixed_head_rejection_diagnostic.json")
    if not diagnostic:
        return ""
    if diagnostic["policy"] != FOCUS_POLICY:
        raise ValueError("Mechanism diagnostic has a different threshold policy")
    groups = [row for row in diagnostic["mean_across_fixed_seeds"]
              if row["comparison"] == "clean_plus_typo_minus_clean_only"
              and row["subset"] == "previously_newly_correct_from_augmentation"]
    if {row["condition"] for row in groups} != set(CONDITIONS[1:]):
        raise ValueError("Missing previously newly-correct diagnostic groups")
    rows = [[CONDITION_NAMES[r["condition"]], f"{r['n_mean']:.2f}",
             f"{r['reference_accepted_mean']:.2f} → {r['mixed_accepted_mean']:.2f}",
             f"{r['reject_accept_mean']:.2f} / {r['accept_reject_mean']:.2f}",
             f"{r['net_additional_accepts_mean']:+.2f}", f"{100*r['mixed_rejection_rate_mean']:.2f}%"] for r in groups]
    cases = diagnostic["row82_illustration"]
    if len(cases) != 9 or len({r["text"] for r in cases}) != 1 or not all(r["predicted_label"] == r["true_label"] for r in cases):
        raise ValueError("row82 illustration no longer describes one identical correctly classified text")
    accepted = {arm: sum(not r["rejected"] for r in cases if r["arm"] == arm) for arm in ARMS}
    first = cases[0]
    return (
        "<h3>上一轮增强后才分对的请求，这次是否真正获得服务？</h3>"
        "<p>沿用上一轮的事后子集：同一扰动文本被clean_continue分错、被增强typo_mix分对。"
        "本轮这批分类预测保持固定，观察其拒识决定是否恢复。下表计数是三个固定seed的均值，"
        "最后一列先以各seed该子集大小为分母计算再求平均，不以全部4,500 IS为分母。</p>"
        + htable(["条件", "上轮新分对组大小", "接受数：clean_only→混合拟合", "拒→受 / 受→拒", "正确服务净增数", "混合拟合后仍拒比例"], rows)
        + "<p>因此，补充扰动拟合分布确实恢复了这批样本中的部分正确服务，仍有一部分被拒。"
        "这是已查看测试集上的描述性子集；类别中心、协方差和验证阈值同时改变，不能据此分离每一部分的因果作用。</p>"
        + f"<p>row_id=82的<code>{html.escape(first['original_text'])}</code> → <code>{html.escape(first['text'])}</code>仍预测"
        + f"<code>{html.escape(first['true_label'])}</code>。clean_only正确且接受{accepted['clean_only']}/3 seed，"
        + f"clean_duplicate为{accepted['clean_duplicate']}/3，混合拟合为{accepted['clean_plus_typo']}/3。"
        + "light与medium在这里是相同文本，仅算一个示例，不是两个独立恢复案例。原始距离跨不同协方差不可直接作为机制证明。</p>"
    )


def build_html(summary, state_summary, transition_summary, intervals, plots, inputs, data_summary, repeats, random_seed, optional_evidence):
    table = summary.set_index(["arm", "condition", "score", "policy"])
    states = state_summary.set_index(["arm", "condition", "score", "policy"])
    focus_intervals = intervals[(intervals.comparison == "clean_plus_typo_minus_clean_only")
                                & (intervals.policy == FOCUS_POLICY)].set_index(["condition", "metric"])
    leading = []
    for condition in CONDITIONS[1:]:
        ci = intervals[(intervals.comparison == "clean_plus_typo_minus_clean_only") & (intervals.policy == FOCUS_POLICY)
                       & (intervals.condition == condition) & (intervals.metric == "in_scope_false_rejection_rate")].iloc[0]
        baseline, target = (table.loc[(a, condition, "mahalanobis", FOCUS_POLICY)] for a in ("clean_only", "clean_plus_typo"))
        leading.append(f"{CONDITION_NAMES[condition]}误拒率 {100*baseline.in_scope_false_rejection_rate_mean:.2f}% → {100*target.in_scope_false_rejection_rate_mean:.2f}%，差值{ci.estimate_delta_pp:+.2f} pp，95%条件区间[{ci.ci_low_pp:+.2f}, {ci.ci_high_pp:+.2f}]。")
    guardrails, declines, guardrail_intervals = [], [], []
    for metric in ("oos_f1", "oos_recall", "in_scope_acceptance"):
        baseline, target = (table.loc[(a, "clean", "mahalanobis", FOCUS_POLICY)][metric+"_mean"] for a in ("clean_only", "clean_plus_typo"))
        delta = target-baseline
        interval = focus_intervals.loc[("clean", metric)]
        guardrail_intervals.append(interval)
        guardrails.append([NAMES[metric], f"{baseline*100:.4f}%", f"{target*100:.4f}%", f"{delta*100:+.6g}",
                           f"[{interval.ci_low_pp:+.4f}, {interval.ci_high_pp:+.4f}]", "下降" if delta < 0 else "未下降"])
        if delta < 0:
            declines.append(f"{NAMES[metric]}下降{-delta*100:.6g} pp")
    retention = ("严格描述性clean保留规则通过：三项均值相对当前增强模型的clean_only均未下降。" if not declines else
                 "严格描述性clean保留规则未通过：" + "；".join(declines) + "。")
    uncertainty = ("三项clean差值的条件区间均跨0，未排除零差异；均值回落不等于已证实总体真实退化，也不能反过来解释成已经证明等效或无损。"
                   if all(r.ci_low_pp <= 0 <= r.ci_high_pp for r in guardrail_intervals) else
                   "各clean差值的条件区间见表；均值保留规则与是否排除零差异是不同判断，不能将区间跨0解释为已经证明等效或无损。")
    service_rows = []
    for condition in CONDITIONS[1:]:
        baseline, target = (table.loc[(arm, condition, "mahalanobis", FOCUS_POLICY)] for arm in ("clean_only", "clean_plus_typo"))
        correct_gain = 100 * (target.in_scope_correct_and_accepted_rate_mean - baseline.in_scope_correct_and_accepted_rate_mean)
        total_gain = 100 * (target.in_scope_acceptance_mean - baseline.in_scope_acceptance_mean)
        recall = focus_intervals.loc[(condition, "oos_recall")]
        service_rows.append([CONDITION_NAMES[condition], f"{total_gain:+.3f}", f"{correct_gain:+.3f}",
                             f"{total_gain-correct_gain:+.3f}", f"{recall.estimate_delta_pp:+.3f}",
                             f"[{recall.ci_low_pp:+.3f}, {recall.ci_high_pp:+.3f}]"])
    main_rows, state_rows, flow_rows, full_rows, closed_rows = [], [], [], [], []
    for condition in CONDITIONS:
        shared = table.loc[("clean_only", condition, "mahalanobis", FOCUS_POLICY)]
        closed_rows.append([CONDITION_NAMES[condition], formatted(shared, "closed_set_accuracy"), "三个拒识器逐条预测完全相同"])
        for arm in ARMS:
            row = table.loc[(arm, condition, "mahalanobis", FOCUS_POLICY)]
            main_rows.append([ARM_NAMES[arm], CONDITION_NAMES[condition], *[formatted(row, m) for m in METRICS]])
            row = states.loc[(arm, condition, "mahalanobis", FOCUS_POLICY)]
            state_rows.append([ARM_NAMES[arm], CONDITION_NAMES[condition], *[formatted(row, state+"_rate_over_all_is") for state in STATES]])
    for row in transition_summary.itertuples():
        if row.policy == FOCUS_POLICY:
            value = pd.Series(row._asdict())
            flow_rows.append([ARM_NAMES[row.reference]+" → 混合拟合", CONDITION_NAMES[row.condition],
                              {"all_is":"全部IS", "head_correct":"固定头正确", "head_wrong":"固定头错误"}[row.stratum],
                              formatted(value, "stratum_n", 1), formatted(value, "accept_reject_n", 1),
                              formatted(value, "reject_accept_n", 1), formatted(value, "net_rejection_change_n", 1)])
    for row in summary.itertuples():
        values = pd.Series(row._asdict())
        full_rows.append([ARM_NAMES[row.arm], CONDITION_NAMES[row.condition], POLICY_NAMES[row.policy], *[formatted(values, m) for m in METRICS]])
    ci_rows = [["主比较" if r.comparison_role == "main_comparison" else "重复数对照", CONDITION_NAMES[r.condition],
                POLICY_NAMES[r.policy], NAMES[r.metric], f"{r.estimate_delta_pp:+.2f}", f"[{r.ci_low_pp:+.2f}, {r.ci_high_pp:+.2f}]"]
               for r in intervals.itertuples()]
    source_table = htable(["输入文件", "SHA256"], [[str(p), sha256(p)] for p in inputs])
    evidence_html = evidence_summary(optional_evidence, inputs)
    evidence_html += "".join(f"<details><summary>完整证据：{html.escape(name)}</summary><pre>{html.escape(json.dumps(value, ensure_ascii=False, indent=2))}</pre></details>" for name, value in optional_evidence.items())
    mechanism_html = mechanism_summary(optional_evidence)
    now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M UTC+08:00")
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>固定增强BERT：混合拟合拒识分布</title><style>
*{{box-sizing:border-box}}body{{background:#f3f6f7;color:#193746;margin:0;font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1260px;margin:28px auto;padding:34px 40px;background:white;border:1px solid #dbe5e8;border-radius:16px}}
h1{{font-size:29px;line-height:1.4}}h2{{font-size:23px;margin-top:34px}}h3{{font-size:18px}}.meta{{color:#5c7280;font-size:13px}}
.lead{{background:#fff5e9;border-left:5px solid #c17c36;padding:16px 20px;font-size:18px}}
.table-scroll{{overflow-x:auto;margin:16px 0}}table{{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}}
th,td{{padding:9px 10px;vertical-align:top;text-align:left;border-bottom:1px solid #dce5e8}}th{{background:#eaf0f3;white-space:nowrap}}td{{overflow-wrap:anywhere}}
img{{max-width:100%;height:auto;border:1px solid #dce5e8;border-radius:8px;margin:16px 0}}details{{margin:18px 0}}summary{{font-weight:650;cursor:pointer}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}@media(max-width:700px){{main{{padding:24px 16px;margin:0;border-radius:0}}h1{{font-size:24px}}}}
</style><main><p class="meta">MFE310TC · 探索性冻结模型实验 · {now}</p>
<h1>固定增强BERT，只改变马氏拒识分布的拟合数据</h1>
<p class="lead">{html.escape(' '.join(leading))}</p><p><strong>{html.escape(retention)}</strong>
不事后设容忍margin；这是描述性均值判断，不是非劣效或等效证明。</p>
<p>主比较为clean_plus_typo − clean_only，即相对当前增强BERT原有的clean拒识器；clean_plus_typo − clean_duplicate用于控制重复样本数。
本轮分类模型、同一测试文本的分类预测完全共享。改变的是类别中心、共享协方差及按同一clean validation规则重选的阈值；没有再次训练BERT。</p>
<h2>实验边界：三臂改变了什么？</h2>
{htable(['拒识器','用于拟合的向量','含义'],[[ARM_NAMES['clean_only'],'15,000条原clean CLS','直接复制当前增强模型原detector与阈值'],[ARM_NAMES['clean_duplicate'],'原clean CLS重复两遍，共30,000','控制拟合行数；重复观测不是新增独立样本'],[ARM_NAMES['clean_plus_typo'],'15,000 clean＋15,000新noisy CLS','每条训练原文增加一条noise，每类50 light/50 medium']])}
<p>新noise分配种子31020261020；总体拟合策略为50% clean、25% light、25% medium，实际修改覆盖可能不足，不能把策略比例当实际改变比例。
中心与Ledoit–Wolf共享协方差使用训练数据，阈值仅由clean validation选择；两个新拒识器开发冻结后才读取本次test。
clean重复两遍可能影响收缩估计的有限样本行为，因此保留该对照，而不是预先假设它必与原拒识器完全一样。</p>
<p>实际新noise中有{data_summary['actual_changed_rows']:,}/{data_summary['source_rows']:,}条改变（{data_summary['noisy_view_changed_fraction']*100:.2f}%）；
相对于混合拟合池的30,000个位置，改变覆盖率为{data_summary['mixed_pool_changed_fraction']*100:.2f}%。新noise中修改0/1/2词分别为
{data_summary['modified_token_counts']['0']:,}/{data_summary['modified_token_counts']['1']:,}/{data_summary['modified_token_counts']['2']:,}，缺少合格词时可少改或不改。</p>
<details><summary>实际拟合数据摘要</summary><pre>{html.escape(json.dumps(data_summary, ensure_ascii=False, indent=2))}</pre></details>
<p>BERT表示学习将文本变为CLS向量；类别中心与类内协方差描述已知请求的分布；最小马氏距离形成异常分数；clean验证集确定运行阈值。
最近中心只参与异常评分，意图预测仍由冻结分类头给出。不同拒识器的中心、协方差和阈值不同，原始距离不可直接跨臂用作改善机制证据。</p>
<h2>干净输入能力是否保留？</h2><p>严格规则为95策略下clean OOS F1、OOS recall、IS接受率三项mean相对clean_only全部不下降。
原BERT不是本轮保留基准；闭集accuracy由固定分类预测决定，须完全相同，单独列出。</p>
{htable(['clean指标','clean_only','clean_plus_typo','差值 pp','95%条件区间 pp','方向'],guardrails)}
<p>{html.escape(uncertainty)}</p>{image_tag(plots[1])}
{htable(['输入条件','三个seed闭集Accuracy %：mean±SD','分类不变检查'],closed_rows)}
<h2>拒识效果与正常请求代价</h2><p>以下聚焦clean验证集IS接受率≥95%的策略。百分比为三个固定seed均值±样本SD（ddof=1），SD单位为百分点。
IS误拒及正确且接受以4,500条IS为分母，OOS recall以1,000条OOS为分母。接受率约束只在clean validation成立，不保证test达到95%。</p>
{htable(['拟合臂','条件',*[NAMES[m]+' %' for m in METRICS]],main_rows)}{image_tag(plots[0])}
<p>降低误拒可能伴随漏过更多OOS，必须一起看recall与F1；接受子集accuracy提高也不能单独证明服务改善。
固定分类头使“分类本可正确却被拒绝”可以直接比较，不再混入本轮分类能力变化。</p>
<p>相对clean_only，总接受增量必须拆成“正确分类的接受”与“错误分类的放行”。下表前三个增量以同一4,500条IS为分母；
总增量不是全部正确服务收益。OOS recall同时报告，它的变化以1,000条OOS为分母。</p>
{htable(['条件','总接受净增 pp','正确且接受净增 pp','错误分类放行净增 pp','OOS recall差值 pp','recall的95%条件区间 pp'],service_rows)}
{htable(['拟合臂','条件',*[s+' /4500 %' for s in STATE_NAMES]],state_rows)}
<h3>同一分类结果下：新增误拒与恢复接受</h3><p>方向统一为参照拒识器→混合拟合。受→拒是新增误拒，拒→受是恢复接受；净拒绝变化=受→拒−拒→受。
按固定分类头是否正确分层后，头正确组的拒→受才是恢复正确服务；头错误组的拒→受不等于服务正确。
表内数量为三个seed计数的均值±SD，分组内比率及以全部4,500 IS为分母的比率另存CSV，不把3个seed当成独立文本。</p>
{htable(['比较方向','条件','固定头分层','组大小','受→拒','拒→受','净拒绝变化'],flow_rows)}
{mechanism_html}
<h2>配对区间</h2><p>差值统一为混合拟合减参照。误拒为负更好，其余指标为正更好。主比较与重复数控制同时报告，不根据test挑选阈值策略。</p>
{image_tag(plots[2])}<details><summary>全部配对区间（含max-F1补充策略）</summary>{htable(['比较','条件','策略','指标','差值 pp','95%条件区间 pp'],ci_rows)}</details>
<p>{repeats:,}次bootstrap，随机种子{random_seed}。150个IS intent各有放回抽30条，OOS抽1,000条；同一原文的权重用于全部arms、seeds、条件与策略。
先在每seed分别计算包括F1在内的指标，再对三个固定模型求均值并作差。区间为2.5%/97.5%分位数，条件于固定分类模型、拟合分布与阈值；
不包含重训、重新拟合或阈值选择的不确定性，也不把3seed预测当独立文本；多项比较未校正。</p>
<h2>解释范围与核验</h2><p>本实验使用此前已查看的测试集，是探索性后续实验。训练增强、拒识分布拟合与测试扰动属于同一合成家族；不推出自然拼写或任意分布外输入上的泛化。
扰动继承原标签，未逐条人工确认语义保持。对照仍含原数据已披露的少量重叠，新增拟合数据是否引入碰撞应依据独立数据审计。
旧实验主终点不改变，本轮也不把结果较好的策略追认为原来的主终点。</p>
<p>报告从27份逐样本预测复算54个test设置，核对同一seed/条件下所有分类预测逐条一致、阈值跨test条件冻结，
并与上一轮增强BERT逐条核对clean_only分类预测与马氏分数及指标。报告只重算统计，不修改冻结结果。</p>
{evidence_html}<details><summary>全部两种阈值策略结果</summary>{htable(['拟合臂','条件','策略',*[NAMES[m]+' %' for m in METRICS]],full_rows)}</details>
<details><summary>输入文件与SHA256</summary>{source_table}</details>
<p class="meta">输出：summary_mean_sd.csv、clean_guardrails.csv、normal_states.csv、rejection_transitions.csv、paired_intervals.csv、三张PNG。HTML图像已内嵌。
AI协助实现、统计核验与整理；学生应核验解释并按课程要求披露实际AI用途与本人贡献。</p></main></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261020)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    output = args.output.resolve()
    metrics, frames, inputs = load_results(output)
    data_path = PROJECT / "data" / "data_summary.json"
    data_summary = json.loads(data_path.read_text(encoding="utf-8"))
    inputs.append(data_path)
    optional_evidence = {}
    for path in (PROJECT / "data_audit.json", output / "verification.json", output / "duplicate_sensitivity.json",
                 output / "fixed_head_rejection_diagnostic.json"):
        if path.is_file():
            optional_evidence[path.name] = json.loads(path.read_text(encoding="utf-8"))
            inputs.append(path)
    summary = aggregate(metrics, ["arm", "condition", "score", "policy"])
    states, transitions = states_and_transitions(frames)
    state_summary = aggregate(states, ["arm", "condition", "score", "policy"])
    transition_summary = aggregate(transitions, ["comparison", "reference", "condition", "score", "policy", "stratum"])
    intervals = paired_bootstrap(frames, args.bootstrap_repeats, args.bootstrap_seed, args.batch_size)
    plots = create_plots(summary, intervals, output)
    document = build_html(summary, state_summary, transition_summary, intervals, plots, inputs, data_summary,
                          args.bootstrap_repeats, args.bootstrap_seed, optional_evidence)
    for name, data in (("summary_mean_sd.csv", summary), ("clean_guardrails.csv", clean_guardrails(summary)), ("normal_states.csv", states),
                       ("rejection_transitions.csv", transitions), ("paired_intervals.csv", intervals)):
        data.to_csv(output / name, index=False)
    report = output / "mixed_rejection_report_cn.html"
    report.write_text(document, encoding="utf-8")
    print(json.dumps(dict(status="PASS", verified_test_settings=len(metrics), normal_states=len(states),
                          rejection_transitions=len(transitions), paired_intervals=len(intervals),
                          bootstrap_repeats=args.bootstrap_repeats, report=str(report)), ensure_ascii=False))


if __name__ == "__main__":
    main()
