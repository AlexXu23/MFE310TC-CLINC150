"""Unit tests for rejector_lib using synthetic data only.

Run from the repository root:
  python -m unittest BERT_Rejector_Selection_20261030/test_rejector_lib.py -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "CLINC150_Team_20261019" / "src"), str(HERE),
                str(ROOT / "BERT_Rejection_Study_20261019")]

import numpy as np

import rejector_lib as lib
from distance_model import SharedMahalanobisDetector


class Thresholds(unittest.TestCase):
    def test_aligned_threshold_accepts_exact_count(self):
        scores = np.arange(100, dtype=float)
        threshold, accepted = lib.aligned_threshold(scores, 0.95)
        self.assertEqual(accepted, 95)
        self.assertEqual(threshold, 95.0)
        self.assertEqual(int((scores < threshold).sum()), 95)

    def test_aligned_threshold_with_ties_takes_next_attainable_count(self):
        scores = np.array([0, 1, 2, 2, 2, 3, 4, 5, 6, 7], dtype=float)
        threshold, accepted = lib.aligned_threshold(scores, 0.3)  # 3 not attainable -> 5
        self.assertEqual((threshold, accepted), (3.0, 5))

    def test_aligned_threshold_all_accepted(self):
        threshold, accepted = lib.aligned_threshold(np.array([1.0, 2.0]), 1.0)
        self.assertEqual(accepted, 2)
        self.assertGreater(threshold, 2.0)

    def test_policy_threshold_aligned_matches_function(self):
        rng = np.random.default_rng(0)
        scores = rng.normal(size=300)
        is_oos = np.r_[np.zeros(270, bool), np.ones(30, bool)]
        self.assertEqual(lib.policy_threshold(scores, is_oos, "aligned_is_acceptance_90"),
                         lib.aligned_threshold(scores[~is_oos], 0.90)[0])

    def test_is_acceptance_95_respects_constraint(self):
        rng = np.random.default_rng(1)
        scores = np.r_[rng.normal(0, 1, 400), rng.normal(2, 1, 40)]
        is_oos = np.r_[np.zeros(400, bool), np.ones(40, bool)]
        threshold = lib.policy_threshold(scores, is_oos, "is_acceptance_95")
        self.assertGreaterEqual((scores[~is_oos] < threshold).mean(), 0.95 - 1e-12)


class Metrics(unittest.TestCase):
    def test_decision_rates_direction_and_partition(self):
        scores = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
        is_oos = np.array([False, False, False, True, True])
        correct = np.array([True, False, True, False, False])
        rates = lib.decision_rates(scores, 0.5, is_oos, correct)  # rejects 0.5, 0.9, 0.8 (>=)
        self.assertAlmostEqual(rates["false_rejection"], 2 / 3)
        self.assertAlmostEqual(rates["correct_acceptance"], 1 / 3)
        self.assertAlmostEqual(rates["wrong_acceptance"], 0.0)
        self.assertAlmostEqual(rates["oos_recall"], 0.5)
        self.assertAlmostEqual(rates["false_rejection"] + rates["correct_acceptance"] + rates["wrong_acceptance"], 1.0)
        self.assertAlmostEqual(rates["oos_f1"], 2 * 1 / (2 * 1 + 2 + 1))

    def test_cost_hand_example(self):
        self.assertAlmostEqual(lib.cost(0.05, 0.2, 0.03, lam=5, pi=0.2), 0.8 * 0.05 + 5 * 0.2 * 0.2)
        self.assertAlmostEqual(lib.cost(0.05, 0.2, 0.03, lam=5, pi=0.2, include_wrong_acceptance=True),
                               0.8 * 0.05 + 5 * 0.2 * 0.2 + 5 * 0.8 * 0.03)

    def test_fpr_at_95(self):
        is_scores = np.arange(100, dtype=float)              # accept 0..94 -> cutoff 95
        oos_scores = np.array([10.0, 94.0, 95.0, 200.0])     # 10, 94 accepted
        scores = np.r_[is_scores, oos_scores]
        is_oos = np.r_[np.zeros(100, bool), np.ones(4, bool)]
        self.assertAlmostEqual(lib.fpr_at_95_tpr(scores, is_oos), 0.5)

    def test_ecdf_and_combinations(self):
        reference = np.array([1.0, 2.0, 2.0, 4.0])
        np.testing.assert_allclose(lib.ecdf(reference, [0.5, 2.0, 3.0, 9.0]), [0.0, 0.75, 0.75, 1.0])
        combo = lib.combined_scores(reference, 10 * reference, np.array([1.0, 4.0]), np.array([40.0, 10.0]))
        np.testing.assert_allclose(combo["combo_max"], [1.0, 1.0])
        np.testing.assert_allclose(combo["combo_mean"], [0.625, 0.625])

    def test_pair_flips_explain_auroc_change(self):
        rng = np.random.default_rng(5)
        positive = np.r_[np.zeros(300, bool), np.ones(70, bool)]
        a = rng.normal(size=370) + positive
        b = np.round(a + rng.normal(scale=0.5, size=370), 1)  # re-ordered, with ties
        lost, gained = lib.pair_flips(a, b, positive, chunk=16)
        self.assertGreater(lost, 0.0)
        self.assertGreater(gained, 0.0)
        self.assertAlmostEqual(lib.auroc(b, positive) - lib.auroc(a, positive), gained - lost, places=12)
        self.assertEqual(lib.pair_flips(a, 2 * a + 1, positive), (0.0, 0.0))  # monotone map: no flips

    def test_auroc_direction(self):
        self.assertEqual(lib.auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([False, False, True, True])), 1.0)
        self.assertTrue(np.isnan(lib.auroc(np.array([0.1, 0.2]), np.array([True, True]))))

    def test_ece_perfect_and_off(self):
        self.assertAlmostEqual(lib.ece(np.full(10, 1.0), np.ones(10, bool)), 0.0)
        self.assertAlmostEqual(lib.ece(np.full(10, 0.5), np.ones(10, bool)), 0.5)


class Resampling(unittest.TestCase):
    def test_stratified_indices_keep_group_sizes(self):
        labels = np.repeat(np.arange(5), [3, 4, 5, 6, 7])
        rng = np.random.default_rng(3)
        for _ in range(20):
            drawn = labels[lib.stratified_indices(labels, rng)]
            self.assertEqual(np.bincount(drawn).tolist(), [3, 4, 5, 6, 7])

    def test_resample_reproducible(self):
        labels = np.repeat(np.arange(3), 4)
        a = lib.validation_resample(labels, 10, np.random.default_rng([1, 2]))
        b = lib.validation_resample(labels, 10, np.random.default_rng([1, 2]))
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, b)))


class Mahalanobis(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.labels = np.repeat(np.arange(4), 60)
        means = rng.normal(0, 3, size=(4, 6))
        self.train = means[self.labels] + rng.normal(0, 1, size=(240, 6)) * np.array([1, 2, 0.5, 1, 3, 1])
        self.query = rng.normal(0, 3, size=(25, 6))

    def test_lw_matches_project_detector(self):
        origin, offsets, residuals = lib.class_residuals(self.train, self.labels)
        precision, info = lib.fit_precision(residuals, "lw")
        mine = lib.min_mahalanobis(self.query, origin, offsets, precision)
        theirs = SharedMahalanobisDetector().fit(self.train, self.labels).score_samples(self.query)
        np.testing.assert_allclose(mine, theirs, rtol=1e-9, atol=1e-9)
        self.assertGreaterEqual(info["shrinkage"], 0.0)

    def test_identity_is_squared_euclidean_to_nearest_mean(self):
        origin, offsets, residuals = lib.class_residuals(self.train, self.labels)
        precision, info = lib.fit_precision(residuals, "identity")
        means = np.stack([self.train[self.labels == c].mean(axis=0) for c in range(4)])
        expected = ((self.query[:, None, :] - means[None]) ** 2).sum(axis=2).min(axis=1)
        np.testing.assert_allclose(lib.min_mahalanobis(self.query, origin, offsets, precision), expected, rtol=1e-9)
        self.assertEqual(info["condition_number"], 1.0)

    def test_all_variants_positive_definite(self):
        _, _, residuals = lib.class_residuals(self.train, self.labels)
        for variant in lib.COVARIANCE_VARIANTS:
            _, info = lib.fit_precision(residuals, variant)
            self.assertGreater(info["min_eigenvalue"], 0.0, variant)

    def test_overlap_table_counts(self):
        a = np.array([True, True, False, False])
        b = np.array([True, False, True, False])
        row = lib.overlap_table(a, b, {"all": np.ones(4, bool)})[0]
        self.assertEqual((row["both_reject"], row["only_a_rejects"], row["only_b_rejects"], row["neither_rejects"]),
                         (1, 1, 1, 1))
        self.assertAlmostEqual(row["jaccard"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
