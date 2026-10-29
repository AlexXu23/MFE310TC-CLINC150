"""Frozen-logit OOS scoring and validation-only temperature scaling.

All anomaly scores increase toward OOS, and rejection uses score >= threshold.
Temperature is fitted by 150-class NLL on in-scope validation rows only.
Thresholds and their tie breaks are frozen before test arrays are loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
import sklearn
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_recall_curve, roc_auc_score, roc_curve,
)


SCORES = ("msp", "entropy", "margin")
CALIBRATIONS = ("raw", "temperature_scaled")
POLICIES = ("max_oos_f1", "is_acceptance_95")
COLORS = {"msp": "#286694", "entropy": "#ce6238", "margin": "#2d8974"}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_split(run: Path, split: str, mapping: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    logits = np.load(run / f"{split}_logits.npy", allow_pickle=False).astype(np.float64)
    with np.load(run / f"{split}_targets.npz", allow_pickle=False) as target_file:
        labels = target_file["label_ids"].copy()
        is_oos = target_file["is_oos"].copy()
    metadata = pd.read_csv(run / f"{split}_metadata.csv", keep_default_na=False)
    expected_is, expected_oos = (3000, 100) if split == "validation" else (4500, 1000)
    if logits.shape != (expected_is + expected_oos, 150) or not np.isfinite(logits).all():
        raise ValueError(f"Invalid {split} logits")
    if is_oos.dtype != np.bool_ or int(is_oos.sum()) != expected_oos:
        raise ValueError(f"Invalid {split} OOS flags")
    if labels.shape != (len(logits),) or len(metadata) != len(logits):
        raise ValueError(f"Invalid {split} row count")
    checks = (
        np.array_equal(metadata["row_id"].to_numpy(), np.arange(len(logits))),
        np.array_equal(metadata["is_oos"].to_numpy(), is_oos),
        np.array_equal(metadata["label_id"].to_numpy(), labels),
        np.array_equal(metadata["true_label"].map(lambda x: mapping.get(x, -1)).to_numpy(), labels),
        np.array_equal(labels == -1, is_oos),
        bool(((labels[~is_oos] >= 0) & (labels[~is_oos] < 150)).all()),
    )
    if not all(checks):
        raise ValueError(f"Misaligned {split} targets / metadata / label mapping")
    return logits, labels, is_oos, metadata


def log_probabilities(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    return scaled - logsumexp(scaled, axis=1, keepdims=True)


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> float:
    log_p = log_probabilities(logits, temperature)
    return float(-log_p[np.arange(len(labels)), labels].mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> dict:
    if np.any(labels < 0):
        raise ValueError("Temperature fit requires in-scope labels only")
    result = minimize_scalar(
        lambda log_t: nll(logits, labels, float(np.exp(log_t))),
        bounds=(float(np.log(0.05)), float(np.log(20.0))),
        method="bounded", options={"xatol": 1e-9},
    )
    if not result.success:
        raise RuntimeError(f"Temperature fitting failed: {result.message}")
    temperature = float(np.exp(result.x))
    if result.fun > nll(logits, labels) + 1e-8:
        raise RuntimeError("Fitted temperature unexpectedly increases calibration NLL")
    return {
        "temperature": temperature, "objective": "in_scope_validation_NLL",
        "fit_examples": len(labels), "bounds": [0.05, 20.0],
        "raw_fit_nll": nll(logits, labels), "fitted_nll": float(result.fun),
        "optimizer_success": bool(result.success), "function_evaluations": int(result.nfev),
    }


def anomaly_scores(logits: np.ndarray, temperature: float = 1.0) -> dict[str, np.ndarray]:
    log_p = log_probabilities(logits, temperature)
    probabilities = np.exp(log_p)
    top_two = np.partition(probabilities, -2, axis=1)[:, -2:]
    largest = top_two.max(axis=1)
    second = top_two.min(axis=1)
    return {
        "msp": 1.0 - largest,
        "entropy": -(probabilities * log_p).sum(axis=1),
        "margin": 1.0 - (largest - second),
    }


def calibration_metrics(logits: np.ndarray, labels: np.ndarray, temperature: float, n_bins: int = 15) -> tuple[dict, pd.DataFrame]:
    log_p = log_probabilities(logits, temperature)
    p = np.exp(log_p)
    prediction = p.argmax(axis=1)
    confidence = p.max(axis=1)
    correct = prediction == labels
    bin_ids = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    bins = []
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        count = int(mask.sum())
        bins.append({
            "bin": bin_id, "lower": bin_id / n_bins, "upper": (bin_id + 1) / n_bins,
            "count": count, "mean_confidence": float(confidence[mask].mean()) if count else 0.0,
            "accuracy": float(correct[mask].mean()) if count else 0.0,
        })
    bin_frame = pd.DataFrame(bins)
    ece = float((bin_frame["count"] / len(labels) * (bin_frame["accuracy"] - bin_frame["mean_confidence"]).abs()).sum())
    # Multiclass Brier: sum over classes, then mean over samples; no /K factor.
    brier = float((np.square(p).sum(axis=1) - 2 * p[np.arange(len(labels)), labels] + 1).mean())
    return {
        "accuracy": float(correct.mean()),
        "macro_f1": float(f1_score(labels, prediction, average="macro", labels=np.arange(p.shape[1]), zero_division=0)),
        "nll": float(-log_p[np.arange(len(labels)), labels].mean()),
        "ece_15_bins": ece, "brier_sum_over_classes": brier,
        "mean_confidence": float(confidence.mean()), "n": len(labels),
    }, bin_frame


def threshold_curve(scores: np.ndarray, is_oos: np.ndarray) -> pd.DataFrame:
    """Exact thresholds, grouping tied scores together before cumulative counts."""
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all() or len(scores) != len(is_oos):
        raise ValueError("Invalid threshold input")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_oos = is_oos[order].astype(np.int64)
    values, starts = np.unique(sorted_scores, return_index=True)
    suffix_tp = np.cumsum(sorted_oos[::-1])[::-1]
    tp = np.r_[suffix_tp[starts], 0]
    rejected = np.r_[len(scores) - starts, 0]
    fp = rejected - tp
    total_oos = int(is_oos.sum())
    total_is = len(scores) - total_oos
    precision = np.divide(tp, rejected, out=np.zeros_like(tp, dtype=float), where=rejected > 0)
    recall = tp / total_oos
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=precision + recall > 0)
    return pd.DataFrame({
        "threshold": np.r_[values, np.nextafter(values[-1], np.inf)],
        "true_oos_rejected": tp, "in_scope_rejected": fp,
        "oos_precision": precision, "oos_recall": recall, "oos_f1": f1,
        "in_scope_acceptance": 1.0 - fp / total_is,
    })


def choose_threshold(curve: pd.DataFrame, policy: str) -> dict:
    if policy == "max_oos_f1":
        candidates = curve.sort_values(
            ["oos_f1", "in_scope_acceptance", "threshold"], ascending=[False, False, False]
        )
    elif policy == "is_acceptance_95":
        candidates = curve[curve["in_scope_acceptance"] >= 0.95 - 1e-12].sort_values(
            ["oos_recall", "oos_f1", "in_scope_acceptance", "threshold"],
            ascending=[False, False, False, False],
        )
    else:
        raise ValueError(f"Unknown policy {policy}")
    return {key: float(value) for key, value in candidates.iloc[0].to_dict().items()}


def evaluate(scores: np.ndarray, is_oos: np.ndarray, labels: np.ndarray, prediction: np.ndarray, threshold: float) -> dict:
    rejected = scores >= threshold
    tp = int((rejected & is_oos).sum())
    fp = int((rejected & ~is_oos).sum())
    fn = int((~rejected & is_oos).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    accepted_is = ~rejected & ~is_oos
    combined_prediction = prediction.copy()
    combined_prediction[rejected] = -1
    return {
        "oos_precision": precision, "oos_recall": recall, "oos_f1": f1,
        "oos_auroc": float(roc_auc_score(is_oos, scores)),
        "oos_aupr_average_precision": float(average_precision_score(is_oos, scores)),
        "in_scope_acceptance": float(accepted_is.sum() / (~is_oos).sum()),
        "in_scope_false_rejection_rate": float(fp / (~is_oos).sum()),
        "in_scope_correct_and_accepted_rate": float(((prediction == labels) & accepted_is).sum() / (~is_oos).sum()),
        "accepted_in_scope_accuracy": float((prediction[accepted_is] == labels[accepted_is]).mean()) if accepted_is.any() else None,
        "combined_151_accuracy": float(accuracy_score(labels, combined_prediction)),
        "combined_151_macro_f1": float(f1_score(labels, combined_prediction, labels=np.arange(-1, 150), average="macro", zero_division=0)),
        "oos_true_positive": tp, "in_scope_false_rejections": fp, "oos_missed": fn,
    }


def paired_bootstrap(is_oos: np.ndarray, raw_rejected: np.ndarray, scaled_rejected: np.ndarray, repeats: int = 2000, seed: int = 2026) -> dict:
    """Conditional paired CI with test scope counts and both thresholds fixed."""
    rng = np.random.default_rng(seed)
    oos_indices = np.flatnonzero(is_oos)
    is_indices = np.flatnonzero(~is_oos)
    deltas = []
    for _ in range(repeats):
        sampled_oos = rng.choice(oos_indices, len(oos_indices), replace=True)
        sampled_is = rng.choice(is_indices, len(is_indices), replace=True)
        metrics = []
        for decisions in (raw_rejected, scaled_rejected):
            tp = decisions[sampled_oos].sum()
            fp = decisions[sampled_is].sum()
            fn = len(sampled_oos) - tp
            metrics.append(np.array([2 * tp / (2 * tp + fp + fn), tp / len(sampled_oos), 1 - fp / len(sampled_is)]))
        deltas.append(metrics[1] - metrics[0])
    deltas = np.asarray(deltas)
    intervals = np.quantile(deltas, [0.025, 0.975], axis=0)
    return {
        "comparison": "temperature_scaled_minus_raw_MSP_max_oos_f1",
        "repeats": repeats, "seed": seed,
        "scope_counts_fixed": True,
        "limitations": "Conditional on one trained checkpoint and frozen validation choices; excludes training and validation-selection uncertainty.",
        "delta_95_percentile_CI": {
            name: [float(intervals[0, i]), float(intervals[1, i])]
            for i, name in enumerate(("oos_f1", "oos_recall", "in_scope_acceptance"))
        },
    }


def save_plots(output: Path, calibration_rows: pd.DataFrame, bins: dict, test_scores: dict, is_oos: np.ndarray, curves: dict) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True, sharey=True)
    for axis, calibration in zip(axes, CALIBRATIONS):
        frame = bins[("test", calibration)]
        populated = frame[frame["count"] > 0]
        values = calibration_rows[(calibration_rows["split"] == "test") & (calibration_rows["calibration"] == calibration)].iloc[0]
        axis.plot([0, 1], [0, 1], "--", color="#9098a0", label="Perfect calibration")
        axis.plot(populated["mean_confidence"], populated["accuracy"], "o-", color="#286694", label="In-scope test bins")
        axis.set(xlabel="Mean confidence", ylabel="Observed accuracy", xlim=(0, 1), ylim=(0, 1), title=f"{calibration.replace('_', ' ').title()}\nECE={values['ece_15_bins']:.3f}, NLL={values['nll']:.3f}")
        axis.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "reliability_test.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for column, calibration in enumerate(CALIBRATIONS):
        for score_name in SCORES:
            scores = test_scores[calibration][score_name]
            fpr, tpr, _ = roc_curve(is_oos, scores)
            precision, recall, _ = precision_recall_curve(is_oos, scores)
            axes[0, column].plot(fpr, tpr, label=f"{score_name.upper()} (AUC {roc_auc_score(is_oos, scores):.3f})", color=COLORS[score_name])
            axes[1, column].plot(recall, precision, label=f"{score_name.upper()} (AP {average_precision_score(is_oos, scores):.3f})", color=COLORS[score_name])
        axes[0, column].plot([0, 1], [0, 1], "--", color="#a5adb4", linewidth=1)
        axes[1, column].axhline(is_oos.mean(), ls="--", color="#a5adb4", linewidth=1)
        axes[0, column].set(xlabel="In-scope false rejection rate", ylabel="OOS recall", title=f"ROC: {calibration}")
        axes[1, column].set(xlabel="OOS recall", ylabel="OOS precision", title=f"Precision-recall: {calibration}")
    for axis in axes.flat:
        axis.set(xlim=(0, 1), ylim=(0, 1.01))
        axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "oos_roc_pr_test.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, score_name in zip(axes, SCORES):
        for calibration, style in zip(CALIBRATIONS, ("-", "--")):
            curve = curves[(calibration, score_name)]
            axis.plot(curve["in_scope_acceptance"], curve["oos_recall"], style, label=calibration.replace("_", " "))
        axis.axvline(.95, ls=":", color="#9a9a9a")
        axis.set(xlabel="In-scope acceptance", ylabel="OOS recall", title=f"Validation trade-off: {score_name.upper()}", xlim=(.85, 1), ylim=(0, 1))
        axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "validation_tradeoffs.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for axis, calibration in zip(axes, CALIBRATIONS):
        msp = 1 - test_scores[calibration]["msp"]
        for mask, label, color in ((~is_oos, "In-scope", "#286694"), (is_oos, "OOS", "#ce6238")):
            axis.hist(msp[mask], bins=np.linspace(0, 1, 31), density=True, alpha=.5, label=label, color=color)
        axis.set(xlabel="Maximum softmax probability", ylabel="Density", title=calibration.replace("_", " ").title(), xlim=(0, 1))
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "test_confidence_distributions.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run, output = Path(args.run), Path(args.output)
    if (output / "summary.json").exists():
        raise FileExistsError("Results already exist; choose a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    training_summary = json.loads((run / "run_summary.json").read_text(encoding="utf-8"))
    source_seed = int(training_summary["config"]["seed"])
    mapping = json.loads((run / "label_mapping.json").read_text(encoding="utf-8"))["label_to_id"]
    protocol = {
        "source_run": str(run.resolve()), "scores": list(SCORES),
        "primary_comparison": "raw_MSP_vs_temperature_scaled_MSP_at_validation_max_OOS_F1",
        "temperature_fit": "all_3000_in_scope_validation_examples_only",
        "calibration_validation_scores_are_in_sample": True,
        "threshold_tuning": "all_3100_validation_examples_including_100_OOS",
        "policies": {"max_oos_f1": "maximize validation OOS F1, tie: higher IS acceptance, higher threshold", "is_acceptance_95": "maximize validation OOS recall with IS acceptance >= 0.95, tie: F1, acceptance, threshold"},
        "anomaly_direction": "larger_means_more_OOS", "reject_rule": "anomaly_score >= threshold",
        "score_formulas": {"msp": "1-max(p)", "entropy": "-sum(p*log(p))", "margin": "1-(p_top1-p_top2)"},
        "test_use": "frozen evaluation only; no test-based threshold, temperature or method selection",
        "existing_test_exposure": "In-scope test metrics were already evaluated in the previous training stage.",
        "ece_bins": "15 equal-width bins on [0,1]", "bootstrap": "2000 paired, stratified by test scope; seed=2026",
    }
    write_json(output / "protocol.json", protocol)
    val_logits, val_labels, val_oos, val_metadata = load_split(run, "validation", mapping)
    temperature_fit = fit_temperature(val_logits[~val_oos], val_labels[~val_oos])
    temperature = temperature_fit["temperature"]
    write_json(output / "temperature_fit.json", temperature_fit)
    temperatures = np.geomspace(.05, 20, 101)
    pd.DataFrame({"temperature": temperatures, "in_scope_validation_nll": [nll(val_logits[~val_oos], val_labels[~val_oos], value) for value in temperatures]}).to_csv(output / "temperature_objective.csv", index=False)

    decisions, curves, val_scores = [], {}, {}
    for calibration in CALIBRATIONS:
        t = 1.0 if calibration == "raw" else temperature
        val_scores[calibration] = anomaly_scores(val_logits, t)
        for score_name in SCORES:
            curve = threshold_curve(val_scores[calibration][score_name], val_oos)
            curves[(calibration, score_name)] = curve
            curve.to_csv(output / f"threshold_curve_{calibration}_{score_name}.csv", index=False)
            for policy in POLICIES:
                selection = choose_threshold(curve, policy)
                decisions.append({"calibration": calibration, "score": score_name, "policy": policy, "temperature": t, **selection})
    write_json(output / "frozen_validation_decisions.json", {"decisions": decisions})
    # No test logit/target reads occur before the preceding file is saved.
    test_logits, test_labels, test_oos, test_metadata = load_split(run, "test", mapping)
    if not np.array_equal(test_logits.argmax(axis=1), (test_logits / temperature).argmax(axis=1)):
        raise AssertionError("Positive scalar temperature changed top-1 predictions")
    test_scores = {calibration: anomaly_scores(test_logits, 1.0 if calibration == "raw" else temperature) for calibration in CALIBRATIONS}

    result_rows = []
    for decision in decisions:
        calibration, score_name = decision["calibration"], decision["score"]
        row = {"calibration": calibration, "score": score_name, "policy": decision["policy"], "temperature": decision["temperature"], "threshold": decision["threshold"]}
        for key in ("oos_precision", "oos_recall", "oos_f1", "in_scope_acceptance"):
            row[f"validation_{key}"] = decision[key]
        row.update(evaluate(test_scores[calibration][score_name], test_oos, test_labels, test_logits.argmax(axis=1), decision["threshold"]))
        result_rows.append(row)
    result_frame = pd.DataFrame(result_rows)
    result_frame.to_csv(output / "oos_results.csv", index=False)

    calibration_rows, bins = [], {}
    for split, logits, labels, is_oos in (("validation", val_logits, val_labels, val_oos), ("test", test_logits, test_labels, test_oos)):
        for calibration in CALIBRATIONS:
            t = 1.0 if calibration == "raw" else temperature
            metrics, bin_frame = calibration_metrics(logits[~is_oos], labels[~is_oos], t)
            calibration_rows.append({"split": split, "calibration": calibration, "temperature": t, **metrics})
            bins[(split, calibration)] = bin_frame
            bin_frame.to_csv(output / f"calibration_bins_{split}_{calibration}.csv", index=False)
    calibration_frame = pd.DataFrame(calibration_rows)
    calibration_frame.to_csv(output / "calibration_metrics.csv", index=False)

    for split, logits, scores, metadata in (("validation", val_logits, val_scores, val_metadata), ("test", test_logits, test_scores, test_metadata)):
        frame = metadata.copy()
        frame["predicted_label_id"] = logits.argmax(axis=1)
        id_to_label = {value: key for key, value in mapping.items()}
        frame["predicted_label"] = frame["predicted_label_id"].map(id_to_label)
        for calibration in CALIBRATIONS:
            for score_name in SCORES:
                frame[f"{calibration}_{score_name}_anomaly"] = scores[calibration][score_name]
        frame.to_csv(output / f"{split}_scores.csv", index=False)
        if split == "test":
            test_score_frame = frame

    primary = result_frame[(result_frame["score"] == "msp") & (result_frame["policy"] == "max_oos_f1")]
    raw_row = primary[primary["calibration"] == "raw"].iloc[0]
    scaled_row = primary[primary["calibration"] == "temperature_scaled"].iloc[0]
    bootstrap = paired_bootstrap(
        test_oos, test_scores["raw"]["msp"] >= raw_row["threshold"],
        test_scores["temperature_scaled"]["msp"] >= scaled_row["threshold"],
    )
    write_json(output / "paired_bootstrap.json", bootstrap)
    for calibration, result_row in (("raw", raw_row), ("temperature_scaled", scaled_row)):
        rejected = test_scores[calibration]["msp"] >= result_row["threshold"]
        frame = test_score_frame.copy()
        frame["rejected_as_oos"] = rejected
        frame["error_type"] = np.select(
            [~test_oos & rejected, test_oos & ~rejected, ~test_oos & ~rejected & (test_logits.argmax(axis=1) != test_labels)],
            ["false_oos", "missed_oos", "accepted_intent_error"], default="correct",
        )
        frame[frame["error_type"] != "correct"].to_csv(output / f"primary_errors_{calibration}.csv", index=False)
    save_plots(output, calibration_frame, bins, test_scores, test_oos, curves)

    input_hashes = {name: file_hash(run / name) for name in ("validation_logits.npy", "test_logits.npy", "validation_targets.npz", "test_targets.npz", "label_mapping.json")}
    summary = {
        "source_seed": source_seed, "temperature_fit": temperature_fit,
        "primary_test_results": primary.to_dict(orient="records"),
        "calibration_metrics": calibration_frame.to_dict(orient="records"),
        "paired_bootstrap": bootstrap, "test_argmax_unchanged": True,
        "limitations": ["Single BERT training seed; multi-seed verification remains outstanding.", "Validation NLL/ECE after fitting are in-sample; test metrics are held out for temperature fitting.", "95% validation acceptance does not guarantee 95% test acceptance.", "OOS precision and F1 depend on scope prevalence; validation and test OOS ratios differ.", "Existing test errors must not be used for subsequent hyperparameter tuning."],
        "input_sha256": input_hashes,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__},
    }
    write_json(output / "summary.json", summary)
    print("Temperature:", temperature)
    print(calibration_frame.to_string(index=False))
    print(result_frame[["calibration", "score", "policy", "oos_precision", "oos_recall", "oos_f1", "oos_auroc", "in_scope_acceptance"]].to_string(index=False))
    print(json.dumps(bootstrap, indent=2))


if __name__ == "__main__":
    main()
