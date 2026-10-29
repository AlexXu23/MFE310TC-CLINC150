"""Audit the existing single-parameter modification without selecting new models.

The original primary endpoint remains MSP OOS F1. Calibration improvements
are reported separately, not substituted retrospectively for that endpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from evaluate_oos import (anomaly_scores, calibration_metrics, choose_threshold,
                          evaluate, file_hash, fit_temperature, load_split,
                          log_probabilities, threshold_curve, write_json)


def per_row_calibration(logits: np.ndarray, labels: np.ndarray, temperature: float) -> dict:
    """Return losses and binning inputs for paired, conditional bootstrap."""
    log_p = log_probabilities(logits, temperature)
    p = np.exp(log_p)
    return {"nll": -log_p[np.arange(len(labels)), labels],
            "brier": (p * p).sum(axis=1) - 2 * p[np.arange(len(labels)), labels] + 1,
            "confidence": p.max(axis=1), "correct": p.argmax(axis=1) == labels}


def ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Compute equal-width ECE; empty bins contribute zero."""
    ids = np.minimum((confidence * bins).astype(int), bins - 1)
    difference = np.bincount(ids, weights=correct.astype(float) - confidence, minlength=bins)
    return float(np.abs(difference).sum() / len(confidence))


def paired_loss_intervals(raw: dict, scaled: dict, repeats: int = 2000, seed: int = 20261019) -> dict:
    """Resample identical in-scope rows for both arms; no seed-level inference."""
    rng = np.random.default_rng(seed)
    result = np.empty((repeats, 3))
    n = len(raw["nll"])
    for repeat in range(repeats):
        indices = rng.integers(0, n, size=n)
        result[repeat] = [
            (scaled["nll"][indices] - raw["nll"][indices]).mean(),
            (scaled["brier"][indices] - raw["brier"][indices]).mean(),
            ece(scaled["confidence"][indices], scaled["correct"][indices]) - ece(raw["confidence"][indices], raw["correct"][indices]),
        ]
    limits = np.quantile(result, [.025, .975], axis=0)
    return {f"{metric}_delta_ci_{side}": float(limits[row, column])
            for column, metric in enumerate(("nll", "brier", "ece"))
            for row, side in enumerate(("low", "high"))}


def audit(project: Path, output: Path) -> dict:
    """Reconstruct every frozen comparison and export paired effect sizes."""
    output.mkdir(parents=True, exist_ok=True)
    rows, controls, shifts, effects, inputs = [], [], [], [], {}
    for seed in (42, 43, 44):
        run = project / f"outputs/bert/seed{seed}"
        directory = project / f"outputs/oos/seed{seed}"
        mapping = json.loads((run / "label_mapping.json").read_text(encoding="utf-8"))["label_to_id"]
        val_z, val_y, val_oos, _ = load_split(run, "validation", mapping)
        refit = fit_temperature(val_z[~val_oos], val_y[~val_oos])
        saved_fit = json.loads((directory / "temperature_fit.json").read_text(encoding="utf-8"))
        assert np.isclose(refit["temperature"], saved_fit["temperature"], atol=1e-10, rtol=0)
        temperature = saved_fit["temperature"]
        frozen = json.loads((directory / "frozen_validation_decisions.json").read_text(encoding="utf-8"))["decisions"]
        for decision in frozen:
            t = 1 if decision["calibration"] == "raw" else temperature
            selected = choose_threshold(threshold_curve(anomaly_scores(val_z, t)[decision["score"]], val_oos), decision["policy"])
            assert np.isclose(selected["threshold"], decision["threshold"], atol=1e-12, rtol=0)
        z, y, oos, _ = load_split(run, "test", mapping)
        assert np.array_equal(z.argmax(axis=1), (z / temperature).argmax(axis=1))
        raw_cal, _ = calibration_metrics(z[~oos], y[~oos], 1)
        scaled_cal, _ = calibration_metrics(z[~oos], y[~oos], temperature)
        row = {"seed": seed, "temperature": temperature}
        for metric in ("accuracy", "macro_f1", "nll", "ece_15_bins", "brier_sum_over_classes", "mean_confidence"):
            row[f"raw_{metric}"] = raw_cal[metric]
            row[f"scaled_{metric}"] = scaled_cal[metric]
            row[f"delta_{metric}"] = scaled_cal[metric] - raw_cal[metric]
        row.update(paired_loss_intervals(per_row_calibration(z[~oos], y[~oos], 1), per_row_calibration(z[~oos], y[~oos], temperature)))
        rows.append(row)
        original = pd.read_csv(directory / "oos_results.csv")
        for decision in frozen:
            values = anomaly_scores(z, decision["temperature"])[decision["score"]]
            result = evaluate(values, oos, y, z.argmax(axis=1), decision["threshold"])
            historical = original[(original["calibration"] == decision["calibration"]) & (original["score"] == decision["score"]) & (original["policy"] == decision["policy"])].iloc[0]
            for metric, value in result.items():
                if value is not None:
                    assert np.isclose(value, historical[metric], atol=1e-12, rtol=0), metric
        for score in ("msp", "entropy", "margin"):
            raw_values = anomaly_scores(z, 1)[score]
            scaled_values = anomaly_scores(z, temperature)[score]
            for policy in ("max_oos_f1", "is_acceptance_95"):
                arms = {cal: next(d for d in frozen if (d["calibration"], d["score"], d["policy"]) == (cal, score, policy)) for cal in ("raw", "temperature_scaled")}
                arm_metrics = {cal: evaluate(raw_values if cal == "raw" else scaled_values, oos, y, z.argmax(axis=1), d["threshold"]) for cal, d in arms.items()}
                raw_reject = raw_values >= arms["raw"]["threshold"]
                scaled_reject = scaled_values >= arms["temperature_scaled"]["threshold"]
                effect = {"seed": seed, "score": score, "policy": policy,
                          "raw_threshold": arms["raw"]["threshold"], "scaled_threshold": arms["temperature_scaled"]["threshold"],
                          "score_rank_spearman": float(spearmanr(raw_values, scaled_values).statistic),
                          "oos_newly_missed": int((oos & raw_reject & ~scaled_reject).sum()),
                          "oos_newly_detected": int((oos & ~raw_reject & scaled_reject).sum()),
                          "is_newly_rejected": int((~oos & ~raw_reject & scaled_reject).sum()),
                          "is_rejection_removed": int((~oos & raw_reject & ~scaled_reject).sum())}
                for metric in ("oos_f1", "oos_recall", "in_scope_acceptance", "oos_auroc"):
                    effect[f"raw_{metric}"] = arm_metrics["raw"][metric]
                    effect[f"scaled_{metric}"] = arm_metrics["temperature_scaled"][metric]
                    effect[f"delta_{metric}"] = arm_metrics["temperature_scaled"][metric] - arm_metrics["raw"][metric]
                effects.append(effect)
        for group, mask in (("correct_in_scope", ~oos & (z.argmax(axis=1) == y)), ("wrong_in_scope", ~oos & (z.argmax(axis=1) != y)), ("oos", oos)):
            shifts.append({"seed": seed, "group": group, "n": int(mask.sum()),
                           "raw_mean_confidence": float(np.exp(log_probabilities(z[mask])).max(axis=1).mean()),
                           "scaled_mean_confidence": float(np.exp(log_probabilities(z[mask], temperature)).max(axis=1).mean())})
        controls.append({"seed": seed, "single_scalar_refit_matches": True, "argmax_unchanged_all_5500": True,
                         "all_12_validation_thresholds_reproduced": True, "all_12_test_comparisons_reproduced": True})
        for path in (run / "validation_logits.npy", run / "test_logits.npy", run / "run_summary.json", directory / "frozen_validation_decisions.json"):
            inputs[str(path.relative_to(project))] = file_hash(path)
    pd.DataFrame(rows).to_csv(output / "calibration_paired_effects.csv", index=False)
    pd.DataFrame(effects).to_csv(output / "oos_paired_effects.csv", index=False)
    pd.DataFrame(shifts).to_csv(output / "confidence_by_correctness.csv", index=False)
    summary = {"verified_controls": controls, "input_sha256": inputs,
               "intervention": "One positive scalar T fitted by in-scope validation NLL; classifier weights, data and class predictions fixed.",
               "primary_endpoint_unchanged": "MSP OOS F1 with validation max-OOS-F1 thresholds; negative result retained.",
               "calibration_endpoints": "In-scope test NLL, ECE (15 equal-width bins), multiclass Brier (sum over classes).",
               "threshold_control": "Same selection algorithm and validation data in both arms; numeric thresholds differ because scores change units.",
               "conditional_bootstrap": {"repeats": 2000, "seed": 20261019, "unit": "same test row in both arms", "scope": "Each fixed seed separately; excludes training and validation-selection uncertainty."},
               "historical_limitations": ["BERT checkpoint selection and temperature fitting both use in-scope validation: validation-fit calibration is not independent validation.", "Temperature fitting and threshold selection reuse validation rows; test labels do not enter either optimization.", "This audit follows existing test exposure and is not a new preregistration or unseen confirmation set.", "Only three training seeds; no population-level significance claim from n=3.", "One-dimensional calibration is not an OOS training objective and need not preserve across-example uncertainty rankings."]}
    write_json(output / "verification.json", summary)
    return {"seeds": len(rows), "oos_pairs": len(effects), "all_controls_pass": True}


def main() -> None:
    """Verify saved runs without training or threshold tuning on test data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("outputs/controlled_audit"))
    args = parser.parse_args()
    print(json.dumps(audit(args.project, args.output), indent=2))


if __name__ == "__main__":
    main()
