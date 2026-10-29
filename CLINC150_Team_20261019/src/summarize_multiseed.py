"""Aggregate training variability and paired clean/typo robustness results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_oos import write_json
from typo_data import CONDITIONS


def aggregate(frame: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    records = []
    for names, group in frame.groupby(keys, dropna=False, sort=False):
        row = dict(zip(keys, names if isinstance(names, tuple) else (names,)))
        row["n_seeds"] = int(group["seed"].nunique())
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1)) if len(group) > 1 else None
        records.append(row)
    return pd.DataFrame(records)


def plots(classification: pd.DataFrame, rejection: pd.DataFrame, clean_oos: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, metric, title in zip(axes, ("accuracy", "macro_f1"), ("Intent accuracy", "Intent Macro-F1")):
        for model, label, color in (("tfidf_lr", "TF-IDF + LR (one fit)", "#cf683f"), ("bert", "BERT (3 seeds, sample SD)", "#286694")):
            group = classification[(classification["model"] == model) & (classification["calibration"] == "raw")].set_index("condition").loc[list(CONDITIONS)]
            axis.errorbar(x, group[f"{metric}_mean"] * 100, yerr=group[f"{metric}_sd"].fillna(0) * 100, label=label, color=color, marker="o", capsize=4)
        axis.set(xticks=x, xticklabels=["Clean", "Light typo", "Medium typo"], ylabel="Percent", title=title)
        axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "classification_robustness.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    selections = (("tfidf_lr", "raw", "msp", "TF-IDF MSP", "#9a7347"), ("bert", "raw", "msp", "BERT MSP", "#286694"), ("bert", "temperature_scaled", "msp", "BERT scaled MSP", "#9172b4"), ("bert", "raw", "entropy", "BERT entropy", "#248577"))
    for axis, metric, title in zip(axes, ("in_scope_false_rejection_rate", "oos_f1"), ("Normal requests incorrectly rejected", "OOS F1 (both scopes perturbed)")):
        for model, calibration, score, label, color in selections:
            group = rejection[(rejection["model"] == model) & (rejection["calibration"] == calibration) & (rejection["score"] == score) & (rejection["policy"] == "is_acceptance_95")].set_index("condition").loc[list(CONDITIONS)]
            axis.errorbar(x, group[f"{metric}_mean"] * 100, yerr=group[f"{metric}_sd"].fillna(0) * 100, label=label, color=color, marker="o", capsize=4)
        axis.set(xticks=x, xticklabels=["Clean", "Light typo", "Medium typo"], ylabel="Percent", title=title)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Thresholds fixed from clean validation at >=95% in-scope acceptance", fontsize=11)
    fig.tight_layout()
    fig.savefig(output / "rejection_robustness.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    colors = ("#286694", "#d17642")
    for axis, metric, title in zip(axes, ("oos_f1", "oos_auroc"), ("Clean OOS F1", "Clean OOS AUROC")):
        for offset, calibration, color in zip((-.18, .18), ("raw", "temperature_scaled"), colors):
            values, errors = [], []
            for score in ("msp", "entropy", "margin"):
                group = clean_oos[(clean_oos["score"] == score) & (clean_oos["policy"] == "max_oos_f1") & (clean_oos["calibration"] == calibration)]
                values.append(group[f"{metric}_mean"].iloc[0] * 100)
                errors.append(group[f"{metric}_sd"].iloc[0] * 100)
            axis.bar(x + offset, values, width=.34, yerr=errors, capsize=3, color=color, label=calibration.replace("_", " "))
        axis.set(xticks=x, xticklabels=["MSP", "Entropy", "Margin"], ylabel="Percent", title=title)
        axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "multiseed_calibration_oos.png", dpi=180)
    plt.close(fig)


def conditional_bootstrap(root: Path, seeds: list[int], repeats: int = 2000) -> pd.DataFrame:
    records = []
    for model, seed_list in (("bert", seeds), ("tfidf_lr", [42])):
        groups = []
        for seed in seed_list:
            path = root / (f"seed{seed}" if model == "bert" else "tfidf_lr")
            groups.append({condition: pd.read_csv(path / f"{condition}_predictions.csv", keep_default_na=False) for condition in CONDITIONS})
        is_mask = ~groups[0]["clean"]["is_oos"].to_numpy(dtype=bool)
        for condition in CONDITIONS[1:]:
            vectors = {"accuracy_drop": [], "false_rejection_increase_raw_msp_95": [], "false_rejection_increase_raw_entropy_95": []}
            for group in groups:
                clean, noisy = group["clean"], group[condition]
                vectors["accuracy_drop"].append(clean.loc[is_mask, "intent_prediction_correct"].to_numpy(dtype=float) - noisy.loc[is_mask, "intent_prediction_correct"].to_numpy(dtype=float))
                for score in ("msp", "entropy"):
                    column = f"raw_{score}_is_acceptance_95_reject"
                    vectors[f"false_rejection_increase_raw_{score}_95"].append(noisy.loc[is_mask, column].to_numpy(dtype=float) - clean.loc[is_mask, column].to_numpy(dtype=float))
            for metric, arrays in vectors.items():
                per_row_mean = np.mean(arrays, axis=0)
                rng = np.random.default_rng(2026)
                estimates = [per_row_mean[rng.integers(0, len(per_row_mean), len(per_row_mean))].mean() * 100 for _ in range(repeats)]
                lower, upper = np.quantile(estimates, [.025, .975])
                records.append({"model": model, "n_fixed_models": len(seed_list), "condition": condition, "metric": metric, "point_change_pp": float(per_row_mean.mean() * 100), "ci95_low_pp": float(lower), "ci95_high_pp": float(upper), "bootstrap_repeats": repeats, "seed": 2026})
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=".")
    parser.add_argument("--robustness", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    project, robustness, output = Path(args.project), Path(args.robustness), Path(args.output)
    if (output / "summary.json").exists():
        raise FileExistsError("Summary already exists; choose a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    training, oos, calibration = [], [], []
    for seed in args.seeds:
        result = json.loads((project / f"outputs/bert/seed{seed}/run_summary.json").read_text(encoding="utf-8"))
        training.append({"seed": seed, "best_epoch": result["best_epoch"], "val_macro_f1": result["best_validation_macro_f1"], "test_accuracy": result["test_in_scope_accuracy"], "test_macro_f1": result["test_in_scope_macro_f1"], "runtime_seconds": result["runtime_seconds"]})
        for collection, filename in ((oos, "oos_results.csv"), (calibration, "calibration_metrics.csv")):
            frame = pd.read_csv(project / f"outputs/oos/seed{seed}/{filename}")
            frame.insert(0, "seed", seed)
            collection.append(frame)
    train_frame = pd.DataFrame(training)
    oos_frame = pd.concat(oos, ignore_index=True)
    calibration_frame = pd.concat(calibration, ignore_index=True)
    train_frame.to_csv(output / "training_by_seed.csv", index=False)
    oos_frame.to_csv(output / "clean_oos_by_seed.csv", index=False)
    calibration_frame.to_csv(output / "clean_calibration_by_seed.csv", index=False)
    oos_aggregate = aggregate(oos_frame, ["calibration", "score", "policy"], ["oos_precision", "oos_recall", "oos_f1", "oos_auroc", "oos_aupr_average_precision", "in_scope_acceptance"])
    calibration_aggregate = aggregate(calibration_frame, ["split", "calibration"], ["accuracy", "macro_f1", "nll", "ece_15_bins", "mean_confidence", "brier_sum_over_classes"])
    oos_aggregate.to_csv(output / "clean_oos_mean_sd.csv", index=False)
    calibration_aggregate.to_csv(output / "clean_calibration_mean_sd.csv", index=False)
    paired = []
    for (seed, score, policy), group in oos_frame.groupby(["seed", "score", "policy"]):
        group = group.set_index("calibration")
        paired.append({"seed": seed, "score": score, "policy": policy, **{f"{metric}_delta_pp": float(100 * (group.loc["temperature_scaled", metric] - group.loc["raw", metric])) for metric in ("oos_f1", "oos_auroc", "oos_recall", "in_scope_acceptance")}})
    paired_frame = pd.DataFrame(paired)
    paired_frame.to_csv(output / "temperature_paired_deltas.csv", index=False)

    cls = pd.read_csv(robustness / "classification_by_seed.csv")
    reject = pd.read_csv(robustness / "rejection_by_seed.csv")
    cls_aggregate = aggregate(cls, ["model", "calibration", "condition"], ["accuracy", "macro_f1", "mean_confidence", "eligible_only_accuracy", "accuracy_change_pp", "macro_f1_change_pp"])
    reject_aggregate = aggregate(reject, ["model", "calibration", "score", "policy", "condition"], ["oos_f1", "oos_recall", "in_scope_acceptance", "in_scope_false_rejection_rate", "in_scope_correct_and_accepted_rate", "in_scope_false_rejection_rate_change_pp"])
    cls_aggregate.to_csv(output / "typo_classification_mean_sd.csv", index=False)
    reject_aggregate.to_csv(output / "typo_rejection_mean_sd.csv", index=False)
    bootstrap = conditional_bootstrap(robustness, args.seeds)
    bootstrap.to_csv(output / "typo_conditional_bootstrap.csv", index=False)
    write_json(output / "bootstrap_scope.json", {"description": "Paired bootstrap of 4500 in-scope test rows, averaging over the same fixed trained seeds. Resampling does not include model training or threshold-selection uncertainty.", "repeats": 2000, "seed": 2026})
    plots(cls_aggregate, reject_aggregate, oos_aggregate, output)
    write_json(output / "summary.json", {
        "seeds": args.seeds, "n_independent_training_runs": len(args.seeds),
        "test_accuracy_mean": float(train_frame["test_accuracy"].mean()),
        "test_accuracy_sample_sd": float(train_frame["test_accuracy"].std(ddof=1)),
        "test_macro_f1_mean": float(train_frame["test_macro_f1"].mean()),
        "test_macro_f1_sample_sd": float(train_frame["test_macro_f1"].std(ddof=1)),
        "clean_oos_rows": len(oos_frame), "typo_classification_rows": len(cls), "typo_rejection_rows": len(reject),
        "sd_definition": "Sample SD across independent training seeds, ddof=1; baseline has one deterministic fit and no across-seed SD.",
        "raw_training_rows": train_frame.to_dict("records"),
    })
    print(train_frame.to_string(index=False))
    print(cls_aggregate[["model", "calibration", "condition", "accuracy_mean", "accuracy_sd", "macro_f1_mean", "macro_f1_sd", "macro_f1_change_pp_mean"]].to_string(index=False))
    print(oos_aggregate[["calibration", "score", "policy", "oos_f1_mean", "oos_f1_sd", "oos_auroc_mean", "in_scope_acceptance_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
