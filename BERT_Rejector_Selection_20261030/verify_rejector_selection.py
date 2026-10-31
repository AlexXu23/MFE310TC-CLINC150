"""Independently verify rejector-analysis artifacts without importing the analysis code.

Reads upstream predictions and current derived outputs. Writes only
outputs/verification.json. Missing or inconsistent artifacts produce FAIL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUT = ROOT / "outputs"
SEEDS = (42, 43, 44)
SCORES = ("msp", "entropy", "mahalanobis")
sys.dont_write_bytecode = True


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    return pd.read_csv(path, keep_default_na=False, float_precision="round_trip", encoding="utf-8-sig")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def boolean(values):
    return values.astype(str).str.lower().eq("true").to_numpy(dtype=bool)


def aligned_threshold(is_scores, target=0.95):
    values = np.sort(np.asarray(is_scores, dtype=np.float64))
    thresholds = np.r_[np.unique(values), np.nextafter(values[-1], np.inf)]
    accepted = np.searchsorted(values, thresholds, side="left")
    needed = math.ceil(target * len(values) - 1e-9)
    return float(thresholds[int(np.flatnonzero(accepted >= needed)[0])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT / "verification.json",
                        help="Write audit JSON here; use .downloads/rejector_verification.json to retain the frozen snapshot.")
    args = parser.parse_args()
    checks = []

    def check(name, ok, **detail):
        checks.append({"check": name, "status": "PASS" if bool(ok) else "FAIL", **detail})

    def close(name, actual, expected, atol=1e-10, **detail):
        check(name, np.isclose(float(actual), float(expected), atol=atol, rtol=1e-9),
              actual=float(actual), expected=float(expected), **detail)

    frozen = read_json(ROOT / "INPUTS_FROZEN.json")
    check("input_manifest_nonempty", len(frozen) == 27, files=len(frozen))
    for relative, expected in frozen.items():
        source = (PROJECT / relative).resolve()
        check("upstream_input_hash", source.is_relative_to(PROJECT.resolve()) and digest(source) == expected,
              file=relative)

    gate = read_json(OUT / "inputs_verification.json")
    check("input_gate_pass", gate["status"] == "PASS" and gate["failed"] == 0,
          checks_run=gate["checks_run"])

    auroc = {score: [] for score in SCORES}
    for seed in SEEDS:
        frame = read_csv(PROJECT / f"BERT_Rejection_Study_20261019/outputs/seed{seed}/test_predictions.csv")
        positive = boolean(frame.is_oos)
        for score in SCORES:
            value = roc_auc_score(positive, frame[f"{score}_score"])
            auroc[score].append(value)
            saved = read_csv(OUT / "separation_clean.csv")
            row = saved[(saved.model == "original") & (saved.seed == seed) &
                        (saved.split == "test") & (saved.score == score)].iloc[0]
            close("auroc_recomputed", value, row.auroc_oos, seed=seed, score=score)
    summary = read_csv(OUT / "separation_clean_summary.csv")
    for score in SCORES:
        row = summary[(summary.model == "original") & (summary.split == "test") &
                      (summary.score == score)].iloc[0]
        close("auroc_mean_recomputed", np.mean(auroc[score]), row.auroc_oos_mean, score=score)
        close("auroc_sd_recomputed", np.std(auroc[score], ddof=1), row.auroc_oos_sd, score=score)

    calibration = read_csv(OUT / "calibration_vs_separation.csv")
    deltas = {score: [] for score in ("msp", "entropy")}
    fa_deltas = {score: [] for score in ("msp", "entropy")}
    for seed in SEEDS:
        test = read_csv(PROJECT / f"CLINC150_Team_20261019/outputs/oos/seed{seed}/test_scores.csv")
        validation = read_csv(PROJECT / f"CLINC150_Team_20261019/outputs/oos/seed{seed}/validation_scores.csv")
        positive = boolean(test.is_oos)
        for score in SCORES[:2]:
            values = {}
            for calibration_name, prefix in (("raw", "raw"), ("temperature_scaled", "temperature_scaled")):
                column = f"{prefix}_{score}_anomaly"
                value = roc_auc_score(positive, test[column])
                row = calibration[(calibration.seed == seed) & (calibration.calibration == calibration_name)].iloc[0]
                close("calibration_auroc_recomputed", value, row[f"test_auroc_{score}"],
                      seed=seed, score=score, calibration=calibration_name)
                threshold = aligned_threshold(validation.loc[~boolean(validation.is_oos), column])
                false_acceptance = float((test.loc[positive, column] < threshold).mean())
                close("validation_95_false_acceptance_recomputed", false_acceptance,
                      row[f"val95_test_oos_false_acceptance_{score}"],
                      seed=seed, score=score, calibration=calibration_name)
                values[calibration_name] = (value, false_acceptance)
            deltas[score].append(values["temperature_scaled"][0] - values["raw"][0])
            fa_deltas[score].append(values["temperature_scaled"][1] - values["raw"][1])
    calibration_summary = read_json(OUT / "calibration_summary.json")["scaled_minus_raw_mean"]
    for score in SCORES[:2]:
        close("calibration_delta_mean_recomputed", np.mean(deltas[score]),
              calibration_summary[f"test_auroc_{score}"], score=score)
        close("false_acceptance_delta_mean_recomputed", np.mean(fa_deltas[score]),
              calibration_summary[f"val95_test_oos_false_acceptance_{score}"], score=score)

    distribution = read_csv(OUT / "threshold_bootstrap_distribution.csv")
    stability = read_json(OUT / "threshold_stability_summary.json")["summary"]
    for row in stability:
        subset = distribution[(distribution.model == row["model"]) & (distribution.score == row["score"]) &
                              (distribution.policy == row["policy"])]
        check("bootstrap_draw_count", len(subset) == 6000, model=row["model"], score=row["score"],
              policy=row["policy"], rows=len(subset))
        for field in ("original_val_is_acceptance", "test_is_acceptance", "test_oos_recall"):
            close("bootstrap_p2p5_recomputed", subset[field].quantile(0.025), row[f"{field}_p2.5"], atol=1e-8,
                  model=row["model"], score=row["score"], policy=row["policy"], field=field)
            close("bootstrap_p97p5_recomputed", subset[field].quantile(0.975), row[f"{field}_p97.5"], atol=1e-8,
                  model=row["model"], score=row["score"], policy=row["policy"], field=field)

    choices = read_csv(OUT / "method_choice_by_cost.csv")
    primary = choices[choices.cost == "fr_plus_fa"]
    check("cost_grid_counts", len(choices) == 48 and len(primary) == 24, all=len(choices), primary=len(primary))
    check("primary_cost_choice_reproduced", primary.chosen_on_validation.str.startswith("mahalanobis|").all())
    close("primary_regret_recomputed", primary.regret.max(), 0.006966222222222207)

    cases = read_csv(OUT / "disagreement_cases.csv")
    check("case_count_and_interpretations", len(cases) == 30 and cases.interpretation.astype(str).str.strip().ne("").all(),
          rows=len(cases))
    covariance = read_json(OUT / "covariance_diagnostics.json")
    check("covariance_limitation_preserved", "NOT run" in covariance["mode"] and len(covariance["per_seed"]) == 3)
    close("covariance_shrinkage_mean_recomputed",
          np.mean([row["lw_shrinkage"] for row in covariance["per_seed"]]), covariance["lw_shrinkage_mean"])

    evidence = read_json(OUT / "evidence_map.json")
    expected_stages = {"verify", "mechanism", "calibration", "covariance", "thresholds", "cost"}
    check("evidence_stage_inventory", set(evidence) == expected_stages, stages=sorted(evidence))
    for stage, record in evidence.items():
        for relative, expected in record["files"].items():
            artifact = (PROJECT / relative).resolve()
            check("derived_artifact_hash", artifact.is_relative_to(PROJECT.resolve()) and digest(artifact) == expected,
                  stage=stage, file=relative)

    failed = [row for row in checks if row["status"] == "FAIL"]
    result = {
        "status": "PASS" if not failed else "FAIL",
        "checks_run": len(checks),
        "failed": len(failed),
        "independence": "No import from run_rejector_selection.py or rejector_lib.py",
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Independent verification: {result['status']} ({len(checks)} checks, {len(failed)} failed)")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
