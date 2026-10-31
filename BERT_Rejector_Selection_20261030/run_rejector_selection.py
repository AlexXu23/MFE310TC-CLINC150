"""Rejector design and selection using saved predictions and frozen thresholds.

No model is trained and no upstream experiment directory is modified. Local CLS
features and FP32 logits are used only when present; otherwise the documented
historical fallbacks record the limitation without inventing missing results.

Stages (see docs/PLAN.md):
  verify       reproduce frozen thresholds, operating points and available scores
  mechanism    compare MSP, entropy and Mahalanobis mechanisms and complementarity
  calibration  compare temperature scaling with OOS separation
  covariance   run the covariance ablation, or record why it is unavailable
  thresholds   bootstrap validation-selected threshold stability
  cost         select scores and operating points using validation error costs

Usage (repository root):
  python BERT_Rejector_Selection_20261030/run_rejector_selection.py --stage all
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "CLINC150_Team_20261019" / "src"), str(HERE)]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_oos import anomaly_scores, fit_temperature
import rejector_lib as lib

SEEDS = (42, 43, 44)
RS = "BERT_Rejection_Study_20261019/outputs"
TA = "BERT_Typo_Augmentation_20261019/outputs"
OP = "BERT_Rejection_Operating_Points_20261019/outputs"
MODELS = {
    "original": {"validation": RS + "/seed{seed}/validation_predictions.csv",
                 "test": RS + "/seed{seed}/test_predictions.csv",
                 "thresholds": RS + "/seed{seed}/frozen_thresholds.json"},
    "typo_mix": {"validation": TA + "/typo_mix/seed{seed}/validation_predictions.csv",
                 "test": TA + "/typo_mix/seed{seed}/clean_predictions.csv",
                 "thresholds": TA + "/typo_mix/seed{seed}/frozen_thresholds.json"},
    "clean_continue": {"validation": TA + "/clean_continue/seed{seed}/validation_predictions.csv",
                       "test": TA + "/clean_continue/seed{seed}/clean_predictions.csv",
                       "thresholds": TA + "/clean_continue/seed{seed}/frozen_thresholds.json"},
}
LOCAL = {"train_cls": RS + "/seed{seed}/train_cls.npy", "validation_cls": RS + "/seed{seed}/validation_cls.npy",
         "test_cls": RS + "/seed{seed}/test_cls.npy", "train_metadata": RS + "/seed{seed}/train_metadata.csv",
         "validation_logits": RS + "/seed{seed}/validation_logits_fp32.npy",
         "test_logits": RS + "/seed{seed}/test_logits_fp32.npy"}
LAMBDAS = (1, 2, 5, 10)
PIS = (0.05, round(1000 / 5500, 4), 0.30)
LABEL = {"msp": "MSP", "entropy": "Entropy", "mahalanobis": "Mahalanobis"}
COLOR = {"msp": "#2a78d6", "entropy": "#eb6834", "mahalanobis": "#1baf7a"}  # fixed categorical order
GROUP_COLOR = {"IS correct": "#2a78d6", "IS wrong": "#eb6834", "OOS": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})


# ----------------------------------------------------------------------------- I/O helpers

def path(template, **values):
    return ROOT / template.format(**values)


def sha256(file):
    digest = hashlib.sha256()
    with Path(file).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(file):
    return json.loads(Path(file).read_text(encoding="utf-8"))


def write_json(file, value):
    Path(file).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")


def read_csv(file):
    return pd.read_csv(file, float_precision="round_trip", keep_default_na=False, encoding="utf-8-sig")


def predictions(model, seed, split):
    """Saved per-row predictions with a boolean ``correct`` column (IS rows only can be correct)."""
    frame = read_csv(path(MODELS[model][split], seed=seed))
    frame["is_oos"] = frame["is_oos"].astype(str).str.lower().eq("true")
    expected = (3000, 100) if split == "validation" else (4500, 1000)
    if (int((~frame.is_oos).sum()), int(frame.is_oos.sum())) != expected:
        raise ValueError(f"Unexpected IS/OOS counts in {MODELS[model][split].format(seed=seed)}")
    frame["correct"] = ~frame.is_oos & (frame.predicted_label_id == frame.label_id)
    return frame


def frozen(model, seed):
    return read_json(path(MODELS[model]["thresholds"], seed=seed))


def local_available(seed, keys=tuple(LOCAL)):
    return all(path(LOCAL[key], seed=seed).exists() for key in keys)


def save_table(rows, name):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT / name, index=False)
    return OUT / name


def register(step, files):
    """Record produced files and their hashes in outputs/evidence_map.json."""
    evidence_path = OUT / "evidence_map.json"
    evidence = read_json(evidence_path) if evidence_path.exists() else {}
    evidence[step] = {"created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
                      "files": {Path(f).relative_to(ROOT).as_posix(): sha256(f) for f in files}}
    write_json(evidence_path, evidence)


def summarise(frame, keys, values):
    grouped = frame.groupby(list(keys))[list(values)]
    out = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=1).add_suffix("_sd"))
    return out.reset_index()


# ----------------------------------------------------------------------------- verify

def step_verify(args):
    checks, hashes = [], {}

    def check(name, ok, **detail):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", **detail})

    for model, seed in itertools.product(MODELS, SEEDS):
        for key in ("validation", "test", "thresholds"):
            hashes[MODELS[model][key].format(seed=seed)] = sha256(path(MODELS[model][key], seed=seed))
        val, test, saved = predictions(model, seed, "validation"), predictions(model, seed, "test"), frozen(model, seed)
        for score, policy in itertools.product(lib.SCORES, lib.POLICIES):
            mine = lib.policy_threshold(val[f"{score}_score"], val.is_oos, policy)
            theirs = saved[f"{score}/{policy}"]["threshold"]
            check("frozen_threshold_recomputed", math.isclose(mine, theirs, rel_tol=1e-12, abs_tol=0.0),
                  model=model, seed=seed, score=score, policy=policy, recomputed=mine, frozen=theirs)
            decisions = test[f"{score}_score"].to_numpy() >= theirs
            check("saved_test_decisions_match_threshold",
                  bool(np.array_equal(decisions, test[f"{score}_{policy}_reject"].astype(str).str.lower().eq("true"))),
                  model=model, seed=seed, score=score, policy=policy)

    for arm_dir in sorted((ROOT / OP).iterdir()):
        for seed in SEEDS:
            threshold_file, val_file = arm_dir / f"seed{seed}/frozen_thresholds.json", arm_dir / f"seed{seed}/validation_predictions.csv"
            if not (threshold_file.exists() and val_file.exists()):
                continue
            saved, val = read_json(threshold_file), read_csv(val_file)
            is_scores = val.loc[~val.is_oos.astype(str).str.lower().eq("true"), "mahalanobis_score"].to_numpy()
            for target, name in lib.ALIGNED_TARGETS:
                if f"mahalanobis/{name}" in saved:
                    theirs = saved[f"mahalanobis/{name}"]["threshold"]
                    mine = lib.aligned_threshold(is_scores, target)[0]
                    check("operating_point_rule_reproduced", math.isclose(mine, theirs, rel_tol=1e-12, abs_tol=0.0),
                          arm=arm_dir.name, seed=seed, policy=name, recomputed=mine, frozen=theirs)

    features = {}
    for seed in SEEDS:
        present = {key: path(LOCAL[key], seed=seed).exists() for key in LOCAL}
        features[f"seed{seed}"] = present
        if not all(present.values()):
            continue
        for key in LOCAL:
            hashes[LOCAL[key].format(seed=seed)] = sha256(path(LOCAL[key], seed=seed))
        train_labels = read_csv(path(LOCAL["train_metadata"], seed=seed))["label_id"].to_numpy()
        origin, offsets, residuals = lib.class_residuals(np.load(path(LOCAL["train_cls"], seed=seed)), train_labels)
        precision, _ = lib.fit_precision(residuals, "lw")
        for split in ("validation", "test"):
            frame = predictions("original", seed, split)
            mine = lib.min_mahalanobis(np.load(path(LOCAL[f"{split}_cls"], seed=seed)), origin, offsets, precision)
            saved = frame["mahalanobis_score"].to_numpy()
            check("mahalanobis_scores_refit", bool(np.allclose(mine, saved, rtol=1e-6, atol=0.0)), seed=seed, split=split,
                  max_relative_difference=float(np.max(np.abs(mine - saved) / np.abs(saved))))
            recomputed = anomaly_scores(np.load(path(LOCAL[f"{split}_logits"], seed=seed)))
            for score in ("msp", "entropy"):
                check("softmax_scores_from_logits",
                      bool(np.allclose(recomputed[score], frame[f"{score}_score"], rtol=1e-9, atol=1e-12)),
                      seed=seed, split=split, score=score)

    failed = [c for c in checks if c["status"] == "FAIL"]
    result = {"status": "PASS" if not failed else "FAIL", "checks_run": len(checks), "failed": len(failed),
              "local_feature_files": features,
              "covariance_calibration_possible": all(all(v.values()) for v in features.values()), "checks": checks}
    write_json(OUT / "inputs_verification.json", result)
    write_json(HERE / "INPUTS_FROZEN.json", hashes)
    register("verify", [OUT / "inputs_verification.json", HERE / "INPUTS_FROZEN.json"])
    print(f"verify: {result['status']} ({len(checks)} checks, {len(failed)} failed); local features: {result['covariance_calibration_possible']}")
    if failed:
        raise SystemExit("verify failed: stop and report outputs/inputs_verification.json; do not continue")


# ----------------------------------------------------------------------------- mechanism

def step_mechanism(args):
    separation, correlations, overlaps, decisions = [], [], [], []
    for model, seed in itertools.product(MODELS, SEEDS):
        saved = frozen(model, seed)
        for split in ("validation", "test"):
            frame = predictions(model, seed, split)
            is_oos, correct = frame.is_oos.to_numpy(), frame.correct.to_numpy()
            for score in lib.SCORES:
                values = frame[f"{score}_score"].to_numpy()
                separation.append({"model": model, "seed": seed, "split": split, "score": score,
                                   "auroc_oos": lib.auroc(values, is_oos),
                                   "fpr_at_95_is_tpr": lib.fpr_at_95_tpr(values, is_oos),
                                   "auroc_is_error_vs_correct": lib.auroc(values[~is_oos], ~correct[~is_oos])})
            if split != "test":
                continue
            for a, b in itertools.combinations(lib.SCORES, 2):
                for group, mask in (("IS", ~is_oos), ("OOS", is_oos)):
                    correlations.append({"model": model, "seed": seed, "pair": f"{a}~{b}", "group": group,
                                         "spearman": lib.spearman(frame.loc[mask, f"{a}_score"], frame.loc[mask, f"{b}_score"])})
            reject = {s: frame[f"{s}_score"].to_numpy() >= saved[f"{s}/is_acceptance_95"]["threshold"] for s in lib.SCORES}
            groups = {"IS correct": ~is_oos & correct, "IS wrong": ~is_oos & ~correct, "OOS": is_oos}
            for a, b in (("msp", "mahalanobis"), ("entropy", "mahalanobis"), ("msp", "entropy")):
                for row in lib.overlap_table(reject[a], reject[b], groups):
                    overlaps.append({"model": model, "seed": seed, "a": a, "b": b, "policy": "is_acceptance_95", **row})
                oos_only = int((reject[a] ^ reject[b])[is_oos].sum())
                decisions.append({"model": model, "seed": seed, "pair": f"{a}~{b}",
                                  "oos_rejected_by_exactly_one_share": oos_only / int(is_oos.sum())})
            union = reject["msp"] | reject["entropy"] | reject["mahalanobis"]
            overlaps.append({"model": model, "seed": seed, "a": "any_of_three", "b": "", "policy": "is_acceptance_95",
                             "group": "summary", "rows": len(frame),
                             "union_oos_recall": float(union[is_oos].mean()),
                             "union_is_false_rejection": float(union[~is_oos].mean())})
    separation = pd.DataFrame(separation)
    files = [save_table(separation, "separation_clean.csv"),
             save_table(summarise(separation, ("model", "split", "score"),
                                  ("auroc_oos", "fpr_at_95_is_tpr", "auroc_is_error_vs_correct")), "separation_clean_summary.csv"),
             save_table(correlations, "score_correlations.csv"), save_table(overlaps, "rejection_overlap.csv")]

    decisions = pd.DataFrame(decisions)
    rule = {}
    for pair in ("entropy~mahalanobis", "msp~mahalanobis"):
        shares = decisions[(decisions.model == "original") & (decisions.pair == pair)]["oos_rejected_by_exactly_one_share"]
        rule[pair] = {"per_seed": shares.round(6).tolist(), "all_seeds_at_least_5pct": bool((shares >= 0.05).all())}
    write_json(OUT / "complementarity_decision.json", {
        "rule": "Only-one-rejects OOS share >= 5% on clean test at is_acceptance_95 in all three seeds (original BERT)",
        "pairs": rule, "run_combination_extension": any(v["all_seeds_at_least_5pct"] for v in rule.values()),
        "all_models": decisions.to_dict(orient="records")})
    files.append(OUT / "complementarity_decision.json")
    files.append(disagreement_cases())
    files.append(plot_distributions())
    register("mechanism", files)
    print("mechanism: separation, correlations, overlap and complementarity decision written")


def disagreement_cases(per_category=5):
    """Rule-based (not hand-picked) cases: first rows by row_id per category, original BERT seed 42, clean test.

    Categories contrast Mahalanobis with entropy (the stronger softmax score); the MSP decision is
    reported in ``msp_rejects`` but does not define the category. Hand-written ``interpretation``
    text already in the file is kept (matched by row_id) when the step is re-run.
    """
    frame = predictions("original", 42, "test")
    saved = frozen("original", 42)
    reject = {s: frame[f"{s}_score"] >= saved[f"{s}/is_acceptance_95"]["threshold"] for s in lib.SCORES}
    maha, ent, msp = reject["mahalanobis"], reject["entropy"], reject["msp"]
    categories = {
        "oos_mahalanobis_rejects_entropy_accepts": frame.is_oos & maha & ~ent,
        "oos_entropy_rejects_mahalanobis_accepts": frame.is_oos & ~maha & ent,
        "oos_all_three_accept": frame.is_oos & ~maha & ~ent & ~msp,
        "is_correct_mahalanobis_rejects_entropy_accepts": frame.correct & maha & ~ent,
        "is_correct_entropy_rejects_mahalanobis_accepts": frame.correct & ~maha & ent,
        "is_wrong_all_three_accept": ~frame.is_oos & ~frame.correct & ~maha & ~ent & ~msp,
    }
    previous = OUT / "disagreement_cases.csv"
    written = {}
    if previous.exists():
        old = read_csv(previous)
        written = {int(r): str(t) for r, t in zip(old.row_id, old.interpretation) if str(t).strip()}
    rows = []
    for name, mask in categories.items():
        for _, row in frame[mask].sort_values("row_id").head(per_category).iterrows():
            rows.append({"category": name, "row_id": int(row.row_id), "text": row.text, "true_label": row.true_label,
                         "predicted_label": row.predicted_label, **{f"{s}_score": row[f"{s}_score"] for s in lib.SCORES},
                         **{f"{s}_rejects": bool(reject[s][row.name]) for s in lib.SCORES},
                         "category_size_seed42": int(mask.sum()),
                         "interpretation": written.get(int(row.row_id), "")})
    return save_table(rows, "disagreement_cases.csv")


def plot_distributions():
    frames = []
    for seed in SEEDS:
        frame = predictions("original", seed, "test")
        frame["group"] = np.where(frame.is_oos, "OOS", np.where(frame.correct, "IS correct", "IS wrong"))
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    figure, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    for axis, score in zip(axes, lib.SCORES):
        values = frame[f"{score}_score"].to_numpy()
        log = score == "mahalanobis"
        edges = np.geomspace(values.min(), values.max(), 60) if log else np.linspace(values.min(), values.max(), 60)
        for group, color in GROUP_COLOR.items():
            subset = values[frame.group.to_numpy() == group]
            axis.hist(subset, bins=edges, density=True, histtype="step", linewidth=2, color=color,
                      label=f"{group} (n={len(subset)})")
        thresholds = [frozen("original", s)[f"{score}/is_acceptance_95"]["threshold"] for s in SEEDS]
        axis.axvline(np.mean(thresholds), color=INK, linestyle="--", linewidth=1)
        axis.text(np.mean(thresholds), axis.get_ylim()[1] * 0.95, " reject →", color=INK, va="top", fontsize=8)
        if log:
            axis.set_xscale("log")
        axis.set_title(LABEL[score], color=INK)
        axis.set_xlabel("score (larger = more OOS)")
        axis.set_yticks([])
    axes[0].set_ylabel("density")
    axes[0].legend(loc="upper left", fontsize=7)
    figure.suptitle("Original BERT, clean test, seeds 42-44 pooled; dashed: 95% validation-acceptance threshold (seed mean)",
                    fontsize=8, color=MUTED)
    figure.savefig(OUT / "score_distributions.png")
    plt.close(figure)
    return OUT / "score_distributions.png"


# ----------------------------------------------------------------------------- covariance

def step_covariance_without_features():
    """Fallback when the CLS features are unavailable: record the fitted LW estimator, run no ablation.

    Writes covariance_diagnostics.json only; covariance_selection.json is not written, so cost keeps LW.
    """
    rows = []
    for seed in SEEDS:
        summary = read_json(path(RS + "/seed{seed}/development_summary.json", seed=seed))
        classes = len(summary["class_counts"])
        rows.append({"seed": seed, "lw_shrinkage": summary["shrinkage"], "feature_dim": summary["feature_dim"],
                     "training_rows": summary["training_rows"], "classes": classes,
                     "residual_degrees_of_freedom": summary["training_rows"] - classes,
                     "residual_rows_per_dimension": (summary["training_rows"] - classes) / summary["feature_dim"]})
    shrinkage = [row["lw_shrinkage"] for row in rows]
    write_json(OUT / "covariance_diagnostics.json", {
        "mode": "no CLS features available: covariance ablation NOT run (report as a limitation)",
        "estimator": "Ledoit-Wolf shared covariance of class-centred train residuals (assume_centered), "
                     "covariance = (1 - shrinkage) * sample covariance + shrinkage * mu * I",
        "source": RS + "/seed{seed}/development_summary.json (saved when the detector was fitted)",
        "per_seed": rows, "lw_shrinkage_mean": float(np.mean(shrinkage)), "lw_shrinkage_sd": float(np.std(shrinkage, ddof=1)),
        "not_available": ["identity / diagonal / OAS / empirical variants", "condition numbers", "validation AUROC per variant"]})
    register("covariance", [OUT / "covariance_diagnostics.json"])
    print(f"covariance: no CLS features; ablation not run. LW shrinkage per seed {[round(s, 4) for s in shrinkage]} "
          "written to covariance_diagnostics.json")


def step_covariance(args):
    seeds = [seed for seed in SEEDS if local_available(seed, ("train_cls", "validation_cls", "test_cls", "train_metadata"))]
    if not seeds:
        return step_covariance_without_features()
    (OUT / "cache").mkdir(exist_ok=True)
    rows = []
    for seed in seeds:
        val, test = predictions("original", seed, "validation"), predictions("original", seed, "test")
        labels = read_csv(path(LOCAL["train_metadata"], seed=seed))["label_id"].to_numpy()
        train = np.load(path(LOCAL["train_cls"], seed=seed))
        if len(train) != len(labels) or len(labels) != 15000:
            raise ValueError("Train features and metadata do not align")
        origin, offsets, residuals = lib.class_residuals(train, labels)
        val_features, test_features = np.load(path(LOCAL["validation_cls"], seed=seed)), np.load(path(LOCAL["test_cls"], seed=seed))
        cache = {}
        for variant in lib.COVARIANCE_VARIANTS:
            started = time.monotonic()
            precision, info = lib.fit_precision(residuals, variant)
            val_scores = lib.min_mahalanobis(val_features, origin, offsets, precision)
            test_scores = lib.min_mahalanobis(test_features, origin, offsets, precision)
            cache[f"{variant}_validation"], cache[f"{variant}_test"] = val_scores, test_scores
            row = {"seed": seed, "variant": variant, **info, "seconds": round(time.monotonic() - started, 2),
                   "val_auroc": lib.auroc(val_scores, val.is_oos), "val_fpr_at_95_is_tpr": lib.fpr_at_95_tpr(val_scores, val.is_oos),
                   "test_auroc": lib.auroc(test_scores, test.is_oos), "test_fpr_at_95_is_tpr": lib.fpr_at_95_tpr(test_scores, test.is_oos)}
            for policy in lib.POLICIES:
                threshold = lib.policy_threshold(val_scores, val.is_oos, policy)
                rates = lib.decision_rates(test_scores, threshold, test.is_oos, test.correct)
                row.update({f"test_{policy}_{k}": rates[k] for k in ("oos_recall", "is_acceptance", "oos_f1")})
            rows.append(row)
            print(f"covariance: seed {seed} {variant}: val AUROC {row['val_auroc']:.4f}, test AUROC {row['test_auroc']:.4f}")
        np.savez_compressed(OUT / "cache" / f"covariance_scores_seed{seed}.npz", **cache)
    frame = pd.DataFrame(rows)
    summary = summarise(frame, ("variant",), [c for c in frame.columns if c.startswith(("val_", "test_"))])
    means = frame.groupby("variant")["val_auroc"].mean()
    best = float(means.max())
    selected = next(v for v in lib.COVARIANCE_VARIANTS if means[v] >= best - 0.002)
    write_json(OUT / "covariance_selection.json", {
        "rule": "Highest mean validation AUROC; within 0.002 of the best choose the simpler (identity < diagonal < lw < oas < empirical)",
        "seeds": seeds, "validation_auroc_mean": means.round(6).to_dict(), "selected": selected,
        "current_method": "lw", "note": "Test columns are reported only; they did not influence the selection."})
    files = [save_table(frame, "covariance_ablation.csv"), save_table(summary, "covariance_ablation_summary.csv"),
             OUT / "covariance_selection.json", plot_covariance(frame)]
    register("covariance", files)
    print(f"covariance: selected covariance variant by validation AUROC: {selected}")


def plot_covariance(frame):
    figure, axes = plt.subplots(1, 2, figsize=(8, 2.8))
    order = list(lib.COVARIANCE_VARIANTS)
    for axis, metric, title in ((axes[0], "auroc", "AUROC (higher is better)"),
                                (axes[1], "fpr_at_95_is_tpr", "FPR@95% IS-TPR (lower is better)")):
        for offset, split, color in ((-0.12, "val", "#2a78d6"), (0.12, "test", "#eb6834")):
            column = f"{split}_{metric}"
            for i, variant in enumerate(order):
                values = frame.loc[frame.variant == variant, column]
                axis.scatter(np.full(len(values), i + offset), values, s=12, color=color, alpha=0.5, linewidths=0)
                axis.scatter(i + offset, values.mean(), s=60, color=color, edgecolors="#fcfcfb", linewidths=2,
                             label=("validation" if split == "val" else "clean test") if i == 0 else None, zorder=3)
        axis.set_xticks(range(len(order)), ["Identity", "Diagonal", "Ledoit-Wolf", "OAS", "Empirical"])
        axis.set_title(title, color=INK)
    axes[0].legend(loc="lower right", fontsize=7)
    figure.suptitle("Mahalanobis covariance ablation, original BERT (small dots: seeds; large: mean)", fontsize=8, color=MUTED)
    figure.savefig(OUT / "covariance_ablation.png")
    plt.close(figure)
    return OUT / "covariance_ablation.png"


# ----------------------------------------------------------------------------- thresholds

def step_thresholds(args):
    rows, actual = [], []
    for m_index, model in enumerate(("original", "typo_mix")):
        for seed in SEEDS:
            val, test, saved = predictions(model, seed, "validation"), predictions(model, seed, "test"), frozen(model, seed)
            is_mask = ~val.is_oos.to_numpy()
            is_labels = val.label_id.to_numpy()[is_mask]
            for s_index, score in enumerate(lib.SCORES):
                values = val[f"{score}_score"].to_numpy()
                is_scores, oos_scores = values[is_mask], values[~is_mask]
                sorted_val_is = np.sort(is_scores)
                test_values = test[f"{score}_score"].to_numpy()
                test_is, test_oos = np.sort(test_values[~test.is_oos]), np.sort(test_values[test.is_oos])
                for policy in lib.POLICIES:
                    threshold = saved[f"{score}/{policy}"]["threshold"]
                    actual.append({"model": model, "seed": seed, "score": score, "policy": policy, "threshold": threshold,
                                   "test_is_acceptance": np.searchsorted(test_is, threshold, "left") / len(test_is),
                                   "test_oos_recall": 1 - np.searchsorted(test_oos, threshold, "left") / len(test_oos)})
                rng = np.random.default_rng([20261030, m_index, seed, s_index])
                labels = np.r_[np.zeros(len(is_scores), bool), np.ones(len(oos_scores), bool)]
                for draw in range(args.bootstrap):
                    is_idx, oos_idx = lib.validation_resample(is_labels, len(oos_scores), rng)
                    curve = lib.threshold_curve(np.r_[is_scores[is_idx], oos_scores[oos_idx]], labels)
                    for policy in lib.POLICIES:
                        threshold = float(lib.choose_threshold(curve, policy)["threshold"])
                        rows.append((model, seed, score, policy, draw, threshold,
                                     np.searchsorted(sorted_val_is, threshold, "left") / len(sorted_val_is),
                                     np.searchsorted(test_is, threshold, "left") / len(test_is),
                                     1 - np.searchsorted(test_oos, threshold, "left") / len(test_oos)))
                print(f"thresholds: {model} seed {seed} {score} done")
    frame = pd.DataFrame(rows, columns=["model", "seed", "score", "policy", "draw", "threshold",
                                        "original_val_is_acceptance", "test_is_acceptance", "test_oos_recall"])
    actual = pd.DataFrame(actual)
    summary = []
    for (model, score, policy), group in frame.groupby(["model", "score", "policy"]):
        entry = {"model": model, "score": score, "policy": policy, "draws_per_seed": args.bootstrap}
        for column in ("original_val_is_acceptance", "test_is_acceptance", "test_oos_recall"):
            low, mid, high = np.percentile(group[column], [2.5, 50, 97.5])
            entry.update({f"{column}_p2.5": low, f"{column}_p50": mid, f"{column}_p97.5": high,
                          f"{column}_width95": high - low})
        frozen_rows = actual[(actual.model == model) & (actual.score == score) & (actual.policy == policy)]
        entry.update({"frozen_test_is_acceptance_by_seed": frozen_rows.test_is_acceptance.round(6).tolist(),
                      "frozen_test_oos_recall_by_seed": frozen_rows.test_oos_recall.round(6).tolist()})
        summary.append(entry)
    write_json(OUT / "threshold_stability_summary.json", {
        "method": "IS validation resampled within intent (20 rows each), OOS validation (100 rows) resampled as one group; "
                  "each draw reselects the threshold with the original rule; models and test rows fixed",
        "note": "is_acceptance_95 depends only on IS validation rows; max_oos_f1 also depends on the 100 OOS rows",
        "summary": summary})
    files = [save_table(frame.round(8), "threshold_bootstrap_distribution.csv"), save_table(actual, "threshold_frozen_test.csv"),
             OUT / "threshold_stability_summary.json", plot_threshold_stability(frame, actual)]
    register("thresholds", files)
    print("thresholds: threshold bootstrap written")


def plot_threshold_stability(frame, actual, model="original"):
    figure, axes = plt.subplots(1, 2, figsize=(9, 3))
    positions = [(score, policy) for score in lib.SCORES for policy in lib.POLICIES]
    ticks = [f"{LABEL[s]}\n{'F1-opt' if p == 'max_oos_f1' else '≥95% IS'}" for s, p in positions]
    for axis, column, title in ((axes[0], "test_is_acceptance", "Clean-test IS acceptance"),
                                (axes[1], "test_oos_recall", "Clean-test OOS recall")):
        for i, (score, policy) in enumerate(positions):
            values = frame.loc[(frame.model == model) & (frame.score == score) & (frame.policy == policy), column]
            low, q1, mid, q3, high = np.percentile(values, [2.5, 25, 50, 75, 97.5])
            axis.vlines(i, low, high, color=COLOR[score], linewidth=2)
            axis.add_patch(plt.Rectangle((i - 0.18, q1), 0.36, max(q3 - q1, 1e-4), facecolor=COLOR[score], alpha=0.35,
                                         edgecolor=COLOR[score], linewidth=1))
            axis.hlines(mid, i - 0.18, i + 0.18, color=INK, linewidth=1)
            points = actual.loc[(actual.model == model) & (actual.score == score) & (actual.policy == policy), column]
            axis.scatter(np.full(len(points), i + 0.3), points, s=14, color=INK, zorder=3,
                         label="frozen threshold, per seed" if i == 0 else None)
        if column == "test_is_acceptance":
            axis.axhline(0.95, color=MUTED, linestyle=":", linewidth=1)
        axis.set_xticks(range(len(positions)), ticks, fontsize=7)
        axis.set_title(title, color=INK)
    axes[0].legend(loc="lower left", fontsize=7)
    figure.suptitle("Bootstrap of the validation threshold (original BERT; line: 2.5-97.5%, box: IQR, 3 seeds pooled)",
                    fontsize=8, color=MUTED)
    figure.savefig(OUT / "threshold_stability.png")
    plt.close(figure)
    return OUT / "threshold_stability.png"


# ----------------------------------------------------------------------------- calibration

def step_calibration(args):
    seeds = [seed for seed in SEEDS if local_available(seed, ("validation_logits", "test_logits"))]
    if not seeds:
        return step_calibration_historical()
    sweep, fitted_rows, bins = [], [], []
    for seed in seeds:
        val, test = predictions("original", seed, "validation"), predictions("original", seed, "test")
        val_logits, test_logits = np.load(path(LOCAL["validation_logits"], seed=seed)), np.load(path(LOCAL["test_logits"], seed=seed))
        val_is = ~val.is_oos.to_numpy()
        fit = fit_temperature(val_logits[val_is], val.label_id.to_numpy()[val_is])
        temperatures = sorted(set(np.round(np.geomspace(0.05, 20, 41), 6)) | {1.0, fit["temperature"]})
        test_is = ~test.is_oos.to_numpy()
        test_correct = test.correct.to_numpy()[test_is]
        mahalanobis_auroc = lib.auroc(test["mahalanobis_score"], test.is_oos)
        for temperature in temperatures:
            scores = anomaly_scores(test_logits, temperature)
            confidence = 1.0 - scores["msp"][test_is]
            sweep.append({"seed": seed, "temperature": temperature, "is_fitted": temperature == fit["temperature"],
                          "test_ece_15": lib.ece(confidence, test_correct), "test_mean_confidence": float(confidence.mean()),
                          "test_auroc_msp": lib.auroc(scores["msp"], test.is_oos),
                          "test_auroc_entropy": lib.auroc(scores["entropy"], test.is_oos),
                          "test_auroc_mahalanobis": mahalanobis_auroc})
        for name, temperature in (("raw", 1.0), ("temperature_scaled", fit["temperature"])):
            confidence = 1.0 - anomaly_scores(test_logits, temperature)["msp"][test_is]
            edges = np.minimum((confidence * 15).astype(int), 14)
            for b in range(15):
                mask = edges == b
                bins.append({"seed": seed, "calibration": name, "bin": b, "count": int(mask.sum()),
                             "mean_confidence": float(confidence[mask].mean()) if mask.any() else np.nan,
                             "accuracy": float(test_correct[mask].mean()) if mask.any() else np.nan})
        fitted_rows.append({"seed": seed, **{k: fit[k] for k in ("temperature", "raw_fit_nll", "fitted_nll")}})
    sweep, bins = pd.DataFrame(sweep), pd.DataFrame(bins)
    deltas = []
    for seed in seeds:
        rows = sweep[sweep.seed == seed]
        raw, scaled = rows[rows.temperature == 1.0].iloc[0], rows[rows.is_fitted].iloc[0]
        deltas.append({"seed": seed, "temperature": scaled.temperature,
                       **{f"delta_{k}": scaled[k] - raw[k] for k in ("test_ece_15", "test_auroc_msp", "test_auroc_entropy")},
                       "raw_test_ece_15": raw.test_ece_15, "scaled_test_ece_15": scaled.test_ece_15})
    history = ROOT / "CLINC150_Team_20261019/outputs/multiseed"
    reference = {}
    if (history / "temperature_paired_deltas.csv").exists():
        old = read_csv(history / "temperature_paired_deltas.csv")
        reference["temperature_paired_deltas_mean_pp"] = old.groupby(["score", "policy"])[
            ["oos_f1_delta_pp", "oos_auroc_delta_pp"]].mean().round(4).reset_index().to_dict(orient="records")
    write_json(OUT / "calibration_summary.json", {
        "inputs": "Unified FP32 logits of BERT_Rejection_Study (the report's 35.59% -> 0.85% ECE came from the historical FP16 forward; "
                  "small differences are expected and are not errors)",
        "temperature_fit": "evaluate_oos.fit_temperature on IS validation NLL, per seed", "fits": fitted_rows,
        "fitted_vs_raw": deltas, "historical_fp16_reference": reference})
    files = [save_table(sweep, "calibration_vs_separation.csv"), save_table(bins, "calibration_bins.csv"),
             OUT / "calibration_summary.json", plot_calibration(sweep, bins)]
    register("calibration", files)
    print("calibration: temperature sweep written; fitted-vs-raw deltas in calibration_summary.json")


HIST = "CLINC150_Team_20261019/outputs/oos/seed{seed}"
CALIBRATIONS = (("raw", "raw_"), ("temperature_scaled", "temperature_scaled_"))


def historical_scores(seed, split):
    """Per-row raw and temperature-scaled MSP/entropy of the historical (FP16) temperature-scaling run."""
    frame = read_csv(path(HIST + f"/{split}_scores.csv", seed=seed))
    frame["is_oos"] = frame["is_oos"].astype(str).str.lower().eq("true")
    expected = (3000, 100) if split == "validation" else (4500, 1000)
    if (int((~frame.is_oos).sum()), int(frame.is_oos.sum())) != expected:
        raise ValueError(f"Unexpected IS/OOS counts in {HIST.format(seed=seed)}/{split}_scores.csv")
    frame["correct"] = ~frame.is_oos & (frame.predicted_label_id == frame.label_id)
    return frame


def calibration_bin_rows(seed, name, confidence, correct):
    edges = np.minimum((confidence * 15).astype(int), 14)
    rows = []
    for b in range(15):
        mask = edges == b
        rows.append({"seed": seed, "calibration": name, "bin": b, "count": int(mask.sum()),
                     "mean_confidence": float(confidence[mask].mean()) if mask.any() else np.nan,
                     "accuracy": float(correct[mask].mean()) if mask.any() else np.nan})
    return rows


def step_calibration_historical():
    """Fallback without FP32 logits: raw (T = 1) versus fitted T only, from the saved historical scores.

    The historical run used the same checkpoints with an FP16 forward; the consistency rows compare its
    raw MSP with this study's FP32 MSP row by row. Mahalanobis does not depend on T (FP32 scores used).
    """
    results, bins, flips, consistency = [], [], [], []
    for seed in SEEDS:
        temperature = read_json(path(HIST + "/temperature_fit.json", seed=seed))["temperature"]
        hv, ht = historical_scores(seed, "validation"), historical_scores(seed, "test")
        current = predictions("original", seed, "test")
        merged = ht.merge(current, on="row_id", suffixes=("_hist", "_fp32"))
        consistency.append({"seed": seed, "rows_matched": len(merged), "text_identical": bool((merged.text_hist == merged.text_fp32).all()),
                            "top1_agreement": float((merged.predicted_label_id_hist == merged.predicted_label_id_fp32).mean()),
                            "msp_max_abs_difference": float((merged.raw_msp_anomaly - merged.msp_score).abs().max()),
                            "msp_spearman": lib.spearman(merged.raw_msp_anomaly, merged.msp_score)})
        test_oos, val_oos = ht.is_oos.to_numpy(), hv.is_oos.to_numpy()
        correct_is = ht.correct.to_numpy()[~test_oos]
        for name, prefix in CALIBRATIONS:
            confidence = 1.0 - ht[f"{prefix}msp_anomaly"].to_numpy()[~test_oos]
            row = {"seed": seed, "calibration": name, "temperature": 1.0 if name == "raw" else temperature,
                   "test_ece_15": lib.ece(confidence, correct_is), "test_mean_confidence": float(confidence.mean()),
                   "test_accuracy_is": float(correct_is.mean())}
            for score in ("msp", "entropy"):
                values = ht[f"{prefix}{score}_anomaly"].to_numpy()
                threshold, _ = lib.aligned_threshold(hv[f"{prefix}{score}_anomaly"].to_numpy()[~val_oos], 0.95)
                rates = lib.decision_rates(values, threshold, test_oos, ht.correct.to_numpy())
                row.update({f"test_auroc_{score}": lib.auroc(values, test_oos),
                            f"test_fpr95_{score}": lib.fpr_at_95_tpr(values, test_oos),
                            f"val95_test_false_rejection_{score}": rates["false_rejection"],
                            f"val95_test_oos_false_acceptance_{score}": rates["oos_false_acceptance"]})
            row["test_auroc_mahalanobis"] = lib.auroc(current.mahalanobis_score, current.is_oos)
            results.append(row)
            bins += calibration_bin_rows(seed, name, confidence, correct_is)
        for score in ("msp", "entropy"):
            raw, scaled = ht[f"raw_{score}_anomaly"].to_numpy(), ht[f"temperature_scaled_{score}_anomaly"].to_numpy()
            lost, gained = lib.pair_flips(raw, scaled, test_oos)
            flips.append({"seed": seed, "score": score, "temperature": temperature,
                          "is_oos_pairs_reordered_worse": lost, "is_oos_pairs_reordered_better": gained,
                          "auroc_change": gained - lost,
                          "spearman_raw_vs_scaled_is": lib.spearman(raw[~test_oos], scaled[~test_oos]),
                          "spearman_raw_vs_scaled_oos": lib.spearman(raw[test_oos], scaled[test_oos])})
    results, bins, flips = pd.DataFrame(results), pd.DataFrame(bins), pd.DataFrame(flips)
    value_columns = [c for c in results.columns if c.startswith(("test_", "val95_"))]
    paired = results.pivot(index="seed", columns="calibration", values=value_columns)
    deltas = pd.DataFrame({c: paired[(c, "temperature_scaled")] - paired[(c, "raw")] for c in value_columns})
    failed = [r for r in consistency if not (r["text_identical"] and r["msp_spearman"] > 0.9999 and r["msp_max_abs_difference"] < 0.01)]
    write_json(OUT / "calibration_summary.json", {
        "mode": "historical two-point comparison (raw T = 1 versus fitted T); FP32 logits unavailable, so no temperature sweep",
        "inputs": HIST + "/{validation,test}_scores.csv and temperature_fit.json (same checkpoints, FP16 forward)",
        "temperature_fit": "evaluate_oos.fit_temperature on IS validation NLL, per seed (historical run)",
        "consistency_with_fp32": {"status": "PASS" if not failed else "FAIL", "per_seed": consistency},
        "scaled_minus_raw_mean": deltas.mean().to_dict(), "scaled_minus_raw_sd": deltas.std(ddof=1).to_dict(),
        "pair_reordering_mean": flips.groupby("score")[["is_oos_pairs_reordered_worse", "is_oos_pairs_reordered_better",
                                                        "auroc_change"]].mean().to_dict(orient="index"),
        "reading": "val95_* columns: threshold set at 95% validation IS acceptance under each calibration, then applied to "
                   "clean test; equal IS acceptance isolates the effect of re-ordering from that of re-scaling"})
    files = [save_table(results, "calibration_vs_separation.csv"), save_table(bins, "calibration_bins.csv"),
             save_table(flips, "calibration_rank_flips.csv"), OUT / "calibration_summary.json",
             plot_calibration_two_point(results, bins)]
    register("calibration", files)
    print(f"calibration: historical two-point comparison written (consistency with FP32 scores: "
          f"{'PASS' if not failed else 'FAIL'})")
    if failed:
        raise SystemExit(1)


def plot_reliability(axis, bins):
    pooled = bins.dropna().assign(weighted=lambda d: d.accuracy * d["count"], conf=lambda d: d.mean_confidence * d["count"])
    pooled = pooled.groupby(["calibration", "bin"])[["weighted", "conf", "count"]].sum().reset_index()
    axis.plot([0, 1], [0, 1], color=MUTED, linestyle=":", linewidth=1)
    for name, color, label in (("raw", "#2a78d6", "raw (T = 1)"), ("temperature_scaled", "#eb6834", "temperature scaled")):
        part = pooled[(pooled.calibration == name) & (pooled["count"] > 0)]
        axis.plot(part.conf / part["count"], part.weighted / part["count"], marker="o", markersize=4, linewidth=2,
                  color=color, label=label)
    axis.set(xlabel="mean confidence", ylabel="accuracy", xlim=(0, 1), ylim=(0, 1))
    axis.set_title("Reliability, clean-test IS (seeds pooled)", color=INK)
    axis.legend(loc="upper left", fontsize=7)


def plot_calibration_two_point(results, bins):
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    plot_reliability(axes[0], bins)
    mean = results.groupby("calibration")[[f"test_auroc_{s}" for s in lib.SCORES]].mean()
    for position, score in enumerate(lib.SCORES):
        raw, scaled = mean.loc["raw", f"test_auroc_{score}"], mean.loc["temperature_scaled", f"test_auroc_{score}"]
        axes[1].plot([raw, scaled], [position, position], color=COLOR[score], linewidth=2)
        axes[1].plot(raw, position, "o", markersize=8, markerfacecolor="#fcfcfb", markeredgecolor=COLOR[score], markeredgewidth=2)
        axes[1].plot(scaled, position, "o", markersize=8, color=COLOR[score])
        note = "unchanged by T" if score == "mahalanobis" else f"{100 * (scaled - raw):+.2f} pp"
        axes[1].text(max(raw, scaled) + 0.002, position, note, va="center", fontsize=7, color=INK)
    axes[1].set_yticks(range(len(lib.SCORES)), [LABEL[s] for s in lib.SCORES])
    axes[1].set_ylim(-0.6, len(lib.SCORES) - 0.4)
    axes[1].set_xlim(axes[1].get_xlim()[0], axes[1].get_xlim()[1] + 0.012)
    axes[1].set(xlabel="OOS AUROC (clean test, seed mean)")
    axes[1].set_title("Separation: hollow = raw, filled = temperature scaled", color=INK)
    figure.suptitle("Calibration versus separation, raw versus fitted temperature "
                    "(historical FP16 scores, T fitted per seed; Mahalanobis from FP32 features)", fontsize=8, color=MUTED)
    figure.savefig(OUT / "calibration_vs_separation.png")
    plt.close(figure)
    return OUT / "calibration_vs_separation.png"


def plot_calibration(sweep, bins):
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    plot_reliability(axes[0], bins)
    mean = sweep.groupby("temperature")[["test_auroc_msp", "test_auroc_entropy", "test_auroc_mahalanobis"]].mean()
    mean = mean[~mean.index.isin(sweep.loc[sweep.is_fitted, "temperature"]) | mean.index.isin([1.0])]
    for score in lib.SCORES:
        axes[1].plot(mean.index, mean[f"test_auroc_{score}"], linewidth=2, color=COLOR[score], label=LABEL[score])
    fitted = sweep.loc[sweep.is_fitted, "temperature"].mean()
    for value, text in ((1.0, "T = 1"), (fitted, "fitted T")):
        axes[1].axvline(value, color=MUTED, linestyle="--", linewidth=1)
        axes[1].text(value, axes[1].get_ylim()[0], f" {text}", color=MUTED, fontsize=7, va="bottom")
    axes[1].set_xscale("log")
    axes[1].set(xlabel="temperature T", ylabel="OOS AUROC (clean test)")
    axes[1].set_title("Separation versus temperature (seed mean)", color=INK)
    axes[1].legend(loc="lower right", fontsize=7)
    figure.savefig(OUT / "calibration_vs_separation.png")
    plt.close(figure)
    return OUT / "calibration_vs_separation.png"


# ----------------------------------------------------------------------------- cost

def combination_due():
    """True when mechanism's pre-registered complementarity rule asks for the combination extension."""
    decision = OUT / "complementarity_decision.json"
    return decision.exists() and bool(read_json(decision)["run_combination_extension"])


def score_sets(model, seed):
    """{score name: (validation scores, test scores)}.

    Adds the covariance-selected covariance variant if it is not LW, and the ECDF combinations of entropy and
    Mahalanobis (reference: clean IS validation scores of the same model and seed) if mechanism asked for them.
    """
    val, test = predictions(model, seed, "validation"), predictions(model, seed, "test")
    sets = {score: (val[f"{score}_score"].to_numpy(), test[f"{score}_score"].to_numpy()) for score in lib.SCORES}
    if combination_due():
        reference = [val.loc[~val.is_oos, f"{s}_score"].to_numpy() for s in ("entropy", "mahalanobis")]
        on_val = lib.combined_scores(*reference, *(val[f"{s}_score"].to_numpy() for s in ("entropy", "mahalanobis")))
        on_test = lib.combined_scores(*reference, *(test[f"{s}_score"].to_numpy() for s in ("entropy", "mahalanobis")))
        sets.update({name: (on_val[name], on_test[name]) for name in on_val})
    selection = OUT / "covariance_selection.json"
    cache = OUT / "cache" / f"covariance_scores_seed{seed}.npz"
    if model == "original" and selection.exists() and cache.exists():
        variant = read_json(selection)["selected"]
        if variant != "lw":
            with np.load(cache) as stored:
                sets[f"mahalanobis_{variant}"] = (stored[f"{variant}_validation"], stored[f"{variant}_test"])
    return val, test, sets


def step_cost(args):
    rows, handoff, choices, separation = [], {}, [], []
    for m_index, model in enumerate(("original", "typo_mix")):
        boot_rates = {}  # (draw, candidate) -> list of per-seed rate dicts
        for seed in SEEDS:
            val, test, sets = score_sets(model, seed)
            is_oos_val, correct_val = val.is_oos.to_numpy(), val.correct.to_numpy()
            is_mask = ~is_oos_val
            is_labels = val.label_id.to_numpy()[is_mask]
            handoff.setdefault(model, {})[f"seed{seed}"] = {}
            for score, (val_scores, test_scores) in sets.items():
                separation.append({"model": model, "seed": seed, "score": score,
                                   "val_auroc": lib.auroc(val_scores, is_oos_val), "test_auroc": lib.auroc(test_scores, test.is_oos),
                                   "test_fpr_at_95_is_tpr": lib.fpr_at_95_tpr(test_scores, test.is_oos)})
                thresholds = {p: lib.policy_threshold(val_scores, is_oos_val, p) for p in lib.COST_CANDIDATE_POLICIES}
                handoff[model][f"seed{seed}"][score] = thresholds
                for policy, threshold in thresholds.items():
                    rv = lib.decision_rates(val_scores, threshold, is_oos_val, correct_val)
                    rt = lib.decision_rates(test_scores, threshold, test.is_oos, test.correct)
                    rows.append({"model": model, "seed": seed, "candidate": f"{score}|{policy}", "score": score,
                                 "policy": policy, "threshold": threshold, **{f"val_{k}": v for k, v in rv.items()},
                                 **{f"test_{k}": v for k, v in rt.items()}})
            # Paired bootstrap: one validation resample per draw, shared by every score of this seed.
            rng = np.random.default_rng([20261031, m_index, seed])
            n_oos = int(is_oos_val.sum())
            is_correct = correct_val[is_mask]
            labels = np.r_[np.zeros(int(is_mask.sum()), bool), np.ones(n_oos, bool)]
            for draw in range(args.bootstrap):
                is_idx, oos_idx = lib.validation_resample(is_labels, n_oos, rng)
                correct = np.r_[is_correct[is_idx], np.zeros(n_oos, bool)]
                for score, (val_scores, _) in sets.items():
                    scores = np.r_[val_scores[is_mask][is_idx], val_scores[~is_mask][oos_idx]]
                    for policy in lib.COST_CANDIDATE_POLICIES:
                        threshold = lib.policy_threshold(scores, labels, policy)
                        boot_rates.setdefault((draw, f"{score}|{policy}"), []).append(
                            lib.decision_rates(scores, threshold, labels, correct))
            print(f"cost: {model} seed {seed} done")
        frame = pd.DataFrame([r for r in rows if r["model"] == model])
        means = frame.groupby("candidate")[[c for c in frame.columns if c.startswith(("val_", "test_"))]].mean()
        candidates = list(dict.fromkeys(frame.candidate))
        boot = {}
        for (draw, candidate), per_seed in boot_rates.items():
            boot.setdefault(draw, {})[candidate] = {k: np.mean([r[k] for r in per_seed])
                                                    for k in ("false_rejection", "oos_false_acceptance", "wrong_acceptance")}
        for include_wa, lam, pi in itertools.product((False, True), LAMBDAS, PIS):
            def total(prefix, table):
                return {c: lib.cost(table.loc[c, f"{prefix}_false_rejection"], table.loc[c, f"{prefix}_oos_false_acceptance"],
                                    table.loc[c, f"{prefix}_wrong_acceptance"], lam, pi, include_wa) for c in candidates}
            val_cost, test_cost = total("val", means), total("test", means)
            chosen = min(candidates, key=lambda c: (val_cost[c], candidates.index(c)))
            oracle = min(candidates, key=lambda c: (test_cost[c], candidates.index(c)))
            picks = [min(candidates, key=lambda c: (lib.cost(rates[c]["false_rejection"], rates[c]["oos_false_acceptance"],
                                                             rates[c]["wrong_acceptance"], lam, pi, include_wa), candidates.index(c)))
                     for rates in boot.values()]
            choices.append({"model": model, "cost": "with_wrong_acceptance" if include_wa else "fr_plus_fa", "lambda": lam,
                            "pi": pi, "chosen_on_validation": chosen, "validation_cost": val_cost[chosen],
                            "test_cost": test_cost[chosen], "test_oracle": oracle, "test_oracle_cost": test_cost[oracle],
                            "regret": test_cost[chosen] - test_cost[oracle],
                            "bootstrap_choice_frequency": picks.count(chosen) / len(picks) if picks else np.nan,
                            "bootstrap_most_frequent": max(set(picks), key=picks.count) if picks else ""})
    choices = pd.DataFrame(choices)
    write_json(OUT / "rejector_candidates.json", {
        "for": "Controlled noise and reference-mixture evaluation; thresholds chosen on clean validation only",
        "decision_rule": "reject iff score >= threshold",
        "score_definitions": {
            "msp / entropy / mahalanobis": "the *_score columns of the model's prediction CSVs",
            "combo_max / combo_mean": "F_ent = share of the same model+seed clean IS validation entropy scores <= x, F_maha likewise; "
                                      "combo_max = max(F_ent, F_maha), combo_mean = (F_ent + F_maha) / 2 (rejector_lib.combined_scores)"},
        "lambdas": LAMBDAS, "pis": PIS, "thresholds": handoff,
        "choices": choices.to_dict(orient="records")})
    separation = pd.DataFrame(separation)
    files = [save_table(pd.DataFrame(rows), "cost_candidates.csv"), save_table(choices, "method_choice_by_cost.csv"),
             save_table(separation, "candidate_separation.csv"),
             save_table(summarise(separation, ("model", "score"), ("val_auroc", "test_auroc", "test_fpr_at_95_is_tpr")),
                        "candidate_separation_summary.csv"),
             OUT / "rejector_candidates.json", plot_choices(choices)]
    register("cost", files)
    print("cost: cost-driven choices written")


def short(candidate):
    score, policy = candidate.split("|")
    name = {"msp": "MSP", "entropy": "Ent", "mahalanobis": "Maha", "combo_max": "Max(E,M)",
            "combo_mean": "Mean(E,M)"}.get(score, score.replace("mahalanobis_", "Maha-"))
    point = "F1" if policy == "max_oos_f1" else policy.replace("aligned_is_acceptance_", "").replace("p", ".") + "%"
    return f"{name} {point}"


def plot_choices(choices):
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for axis, model in zip(axes, ("original", "typo_mix")):
        part = choices[(choices.model == model) & (choices.cost == "fr_plus_fa")]
        grid = part.pivot(index="lambda", columns="pi", values="bootstrap_choice_frequency").loc[list(LAMBDAS), list(PIS)]
        image = axis.imshow(grid.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i, lam in enumerate(LAMBDAS):
            for j, pi in enumerate(PIS):
                row = part[(part["lambda"] == lam) & (part.pi == pi)].iloc[0]
                value = grid.iloc[i, j]
                axis.text(j, i, f"{short(row.chosen_on_validation)}\nregret {100 * row.regret:.2f} pp",
                          ha="center", va="center", fontsize=7, color="#fcfcfb" if value > 0.6 else INK)
        axis.set_xticks(range(len(PIS)), [f"π = {p:g}" for p in PIS])
        axis.set_yticks(range(len(LAMBDAS)), [f"λ = {lam}" for lam in LAMBDAS])
        axis.grid(False)
        axis.set_title({"original": "Original BERT", "typo_mix": "Spelling-augmented BERT"}[model], color=INK)
    colorbar = figure.colorbar(image, ax=axes, fraction=0.03)
    colorbar.set_label("bootstrap choice frequency", color=MUTED)
    figure.suptitle("Validation-chosen rejector per cost setting (cost = (1-π)FR + λπFA; regret vs clean-test oracle)",
                    fontsize=8, color=MUTED)
    figure.savefig(OUT / "method_choice_heatmap.png")
    plt.close(figure)
    return OUT / "method_choice_heatmap.png"


# ----------------------------------------------------------------------------- main

STEPS = {
    "verify": step_verify,
    "mechanism": step_mechanism,
    "calibration": step_calibration,
    "covariance": step_covariance,
    "thresholds": step_thresholds,
    "cost": step_cost,
}


def protocol():
    return {
        "date": "2026-10-30",
        "kind": "post_hoc_exploratory_analysis",
        "question": "Why does the system reject a request, and how should a rejection method be selected?",
        "seeds": list(SEEDS),
        "models": list(MODELS),
        "scores": list(lib.SCORES),
        "decision_rule": "reject iff score >= threshold; larger scores mean more OOS",
        "selection_data": "validation only; test oracle values are descriptive references",
        "threshold_policies": list(lib.COST_CANDIDATE_POLICIES),
        "bootstrap_draws": 2000,
        "cost": {
            "lambdas": list(LAMBDAS),
            "pis": list(PIS),
            "primary": "(1-pi)*FR + lambda*pi*FA",
            "secondary": "primary + lambda*(1-pi)*WA",
        },
        "training": False,
        "new_test_inference": False,
        "missing_assets": "Covariance ablation and full temperature sweep require unavailable CLS/FP32 arrays",
        "test_exposure": "Earlier test results were inspected before this analysis; findings are exploratory",
        "organisation": "Retrospective layout and naming change only; numerical methods are unchanged",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=(*STEPS, "all"))
    parser.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap draws for thresholds and cost (default 2000)")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    write_json(HERE / "PROTOCOL.json", protocol())
    for name in (STEPS if args.stage == "all" else (args.stage,)):
        started = time.monotonic()
        STEPS[name](args)
        print(f"{name} finished in {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
