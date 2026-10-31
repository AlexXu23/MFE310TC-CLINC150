"""Pure functions for rejector design and selection (no file I/O).

Score convention everywhere: larger means more OOS, and a row is rejected iff
``score >= threshold``. Thresholds for the two original policies come from the
project's own ``evaluate_oos.threshold_curve``/``choose_threshold``; the aligned
operating points follow the rule of BERT_Rejection_Operating_Points_20261019.
The caller must put ``CLINC150_Team_20261019/src`` on ``sys.path`` first.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.covariance import OAS, EmpiricalCovariance, LedoitWolf
from sklearn.metrics import roc_auc_score

from evaluate_oos import choose_threshold, threshold_curve

SCORES = ("msp", "entropy", "mahalanobis")
POLICIES = ("max_oos_f1", "is_acceptance_95")
ALIGNED_TARGETS = ((0.90, "aligned_is_acceptance_90"), (0.925, "aligned_is_acceptance_92p5"),
                   (0.95, "aligned_is_acceptance_95"), (0.975, "aligned_is_acceptance_97p5"),
                   (0.99, "aligned_is_acceptance_99"))
COST_CANDIDATE_POLICIES = ("max_oos_f1",) + tuple(name for _, name in ALIGNED_TARGETS)
COVARIANCE_VARIANTS = ("identity", "diagonal", "lw", "oas", "empirical")  # simplest first


# ----------------------------------------------------------------------------- thresholds

def aligned_threshold(is_scores, target):
    """Smallest cutoff whose accepted IS count is the smallest attainable count >= ceil(target*n).

    Candidates are the sorted unique IS scores plus nextafter(max, +inf); a row is accepted iff
    score < cutoff. Returns (threshold, accepted_count).
    """
    values = np.sort(np.asarray(is_scores, dtype=np.float64))
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("IS scores must be a non-empty finite array")
    thresholds = np.r_[np.unique(values), np.nextafter(values[-1], np.inf)]
    accepted = np.searchsorted(values, thresholds, side="left")
    needed = math.ceil(target * len(values) - 1e-9)
    index = int(np.flatnonzero(accepted >= needed)[0])  # accepted counts strictly increase
    return float(thresholds[index]), int(accepted[index])


def policy_threshold(scores, is_oos, policy):
    """Threshold for one policy chosen on (validation) scores; same rules as the existing studies."""
    scores = np.asarray(scores, dtype=np.float64)
    is_oos = np.asarray(is_oos, dtype=bool)
    if policy in POLICIES:
        return float(choose_threshold(threshold_curve(scores, is_oos), policy)["threshold"])
    for target, name in ALIGNED_TARGETS:
        if policy == name:
            return aligned_threshold(scores[~is_oos], target)[0]
    raise ValueError(f"Unknown policy {policy}")


# ----------------------------------------------------------------------------- metrics

def auroc(scores, positive):
    """AUROC with ``positive`` as the class expected to have larger scores."""
    positive = np.asarray(positive, dtype=bool)
    if positive.all() or not positive.any():
        return float("nan")
    return float(roc_auc_score(positive, np.asarray(scores, dtype=np.float64)))


def fpr_at_95_tpr(scores, is_oos):
    """IS-positive FPR@95%TPR: share of OOS accepted at the cutoff accepting 95% of IS rows."""
    scores = np.asarray(scores, dtype=np.float64)
    is_oos = np.asarray(is_oos, dtype=bool)
    threshold, _ = aligned_threshold(scores[~is_oos], 0.95)
    return float((scores[is_oos] < threshold).mean())


def decision_rates(scores, threshold, is_oos, correct):
    """Pipeline rates; IS denominators are all IS rows, OOS denominators all OOS rows."""
    rejected = np.asarray(scores, dtype=np.float64) >= threshold
    is_oos = np.asarray(is_oos, dtype=bool)
    correct = np.asarray(correct, dtype=bool)
    is_rows, oos_rows = ~is_oos, is_oos
    tp = int((rejected & oos_rows).sum())
    fp = int((rejected & is_rows).sum())
    fn = int((~rejected & oos_rows).sum())
    return {
        "is_acceptance": float((~rejected[is_rows]).mean()),
        "false_rejection": float(rejected[is_rows].mean()),
        "correct_acceptance": float((~rejected & correct)[is_rows].mean()),
        "wrong_acceptance": float((~rejected & ~correct)[is_rows].mean()),
        "oos_recall": float(rejected[oos_rows].mean()),
        "oos_false_acceptance": float((~rejected[oos_rows]).mean()),
        "oos_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def cost(false_rejection, oos_false_acceptance, wrong_acceptance, lam, pi, include_wrong_acceptance=False):
    """Rate-based cost: (1-pi)*FR + lam*pi*FA [+ lam*(1-pi)*WA]."""
    value = (1 - pi) * false_rejection + lam * pi * oos_false_acceptance
    if include_wrong_acceptance:
        value += lam * (1 - pi) * wrong_acceptance
    return value


def ece(confidence, correct, n_bins=15):
    """Equal-width expected calibration error, same binning as evaluate_oos.calibration_metrics."""
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


# ----------------------------------------------------------------------------- resampling

def stratified_indices(labels, rng):
    """Resample row positions with replacement inside each label group (group sizes preserved)."""
    labels = np.asarray(labels)
    order = np.argsort(labels, kind="stable")
    _, starts, counts = np.unique(labels[order], return_index=True, return_counts=True)
    offsets = np.floor(rng.random(len(labels)) * np.repeat(counts, counts)).astype(np.int64)
    return order[np.repeat(starts, counts) + offsets]


def validation_resample(is_labels, n_oos, rng):
    """(IS positions stratified by intent, OOS positions resampled as one group)."""
    return stratified_indices(is_labels, rng), rng.integers(0, n_oos, n_oos)


# ----------------------------------------------------------------------------- Mahalanobis variants

def class_residuals(features, labels):
    """Origin, class offsets (from origin) and class-centred residuals, as in distance_model.py."""
    features = np.asarray(features, dtype=np.float64)
    classes, inverse, counts = np.unique(np.asarray(labels), return_inverse=True, return_counts=True)
    origin = features.mean(axis=0)
    centered = features - origin
    offsets = np.zeros((len(classes), features.shape[1]))
    np.add.at(offsets, inverse, centered)
    offsets /= counts[:, None]
    return origin, offsets, centered - offsets[inverse]


def fit_precision(residuals, variant):
    """Shared precision matrix for one covariance variant plus diagnostics."""
    dim = residuals.shape[1]
    shrinkage = None
    if variant == "identity":
        covariance = np.eye(dim)
        precision = np.eye(dim)
    elif variant == "diagonal":
        variance = np.mean(residuals ** 2, axis=0)
        covariance = np.diag(variance)
        precision = np.diag(1.0 / variance)
    elif variant in ("lw", "oas", "empirical"):
        estimator = {"lw": LedoitWolf, "oas": OAS, "empirical": EmpiricalCovariance}[variant]
        fitted = estimator(assume_centered=True).fit(residuals)
        covariance = np.asarray(fitted.covariance_, dtype=np.float64)
        precision = np.asarray(fitted.get_precision(), dtype=np.float64)
        shrinkage = float(getattr(fitted, "shrinkage_", 0.0))
    else:
        raise ValueError(f"Unknown covariance variant {variant}")
    eigenvalues = np.linalg.eigvalsh(covariance)
    return precision, {"shrinkage": shrinkage, "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
                       "min_eigenvalue": float(eigenvalues[0])}


def min_mahalanobis(features, origin, offsets, precision, batch_size=512):
    """Minimum squared distance to any class mean under one shared precision; larger is more OOS."""
    features = np.asarray(features, dtype=np.float64)
    mean_precision = offsets @ precision
    mean_norm = np.einsum("ij,ij->i", mean_precision, offsets)
    scores = np.empty(len(features))
    for start in range(0, len(features), batch_size):
        centered = features[start:start + batch_size] - origin
        norm = np.einsum("ij,ij->i", centered @ precision, centered)
        distances = norm[:, None] - 2.0 * centered @ mean_precision.T + mean_norm[None, :]
        scores[start:start + batch_size] = np.maximum(distances.min(axis=1), 0.0)
    return scores


# ----------------------------------------------------------------------------- comparisons

def spearman(a, b):
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def ecdf(reference, values):
    """Empirical CDF of ``reference`` evaluated at ``values``: share of reference scores <= value."""
    reference = np.sort(np.asarray(reference, dtype=np.float64))
    return np.searchsorted(reference, np.asarray(values, dtype=np.float64), side="right") / len(reference)


def combined_scores(entropy_reference, mahalanobis_reference, entropy, mahalanobis):
    """ECDF-normalised combinations of entropy and Mahalanobis (reference = clean IS validation scores).

    Returns {"combo_max": max(F_ent, F_maha), "combo_mean": (F_ent + F_maha) / 2}; larger is more OOS.
    """
    f_ent, f_maha = ecdf(entropy_reference, entropy), ecdf(mahalanobis_reference, mahalanobis)
    return {"combo_max": np.maximum(f_ent, f_maha), "combo_mean": (f_ent + f_maha) / 2}


def pair_flips(scores_a, scores_b, positive, chunk=256):
    """How score b re-orders (negative, positive) pairs relative to score a.

    Each pair is credited 1 if the positive scores higher, 0.5 on a tie, 0 otherwise (the AUROC
    convention). Returns (lost, gained): shares of all pairs whose credit falls / rises from a to b,
    weighted by the change, so that AUROC(b) - AUROC(a) == gained - lost.
    """
    a, b = np.asarray(scores_a, dtype=np.float64), np.asarray(scores_b, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    neg_a, neg_b = a[~positive], b[~positive]
    pos_a, pos_b = a[positive], b[positive]
    lost = gained = 0.0
    for start in range(0, len(pos_a), chunk):
        credit_a = np.sign(pos_a[start:start + chunk, None] - neg_a[None, :]) / 2 + 0.5
        credit_b = np.sign(pos_b[start:start + chunk, None] - neg_b[None, :]) / 2 + 0.5
        change = credit_b - credit_a
        lost += float(-change[change < 0].sum())
        gained += float(change[change > 0].sum())
    pairs = len(pos_a) * len(neg_a)
    return lost / pairs, gained / pairs


def overlap_table(reject_a, reject_b, groups):
    """2x2 rejection counts per group plus Jaccard of the two rejection sets."""
    rows = []
    for name, mask in groups.items():
        a, b = reject_a[mask], reject_b[mask]
        union = int((a | b).sum())
        rows.append({"group": name, "rows": int(mask.sum()), "both_reject": int((a & b).sum()),
                     "only_a_rejects": int((a & ~b).sum()), "only_b_rejects": int((~a & b).sum()),
                     "neither_rejects": int((~a & ~b).sum()),
                     "jaccard": float((a & b).sum() / union) if union else float("nan")})
    return rows
