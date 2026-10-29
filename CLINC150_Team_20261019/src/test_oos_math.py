"""Boundary and independent metric checks for the OOS research pipeline."""

import unittest

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from evaluate_oos import (
    anomaly_scores, choose_threshold, fit_temperature, log_probabilities,
    nll, threshold_curve,
)


class OOSMathTests(unittest.TestCase):
    def test_tied_scores_match_sklearn_at_every_threshold(self):
        scores = np.array([.2, .2, .8, .8, .9, .1])
        y = np.array([False, True, True, False, True, False])
        curve = threshold_curve(scores, y)
        for row in curve.to_dict("records"):
            rejected = scores >= row["threshold"]
            self.assertAlmostEqual(row["oos_precision"], precision_score(y, rejected, zero_division=0))
            self.assertAlmostEqual(row["oos_recall"], recall_score(y, rejected, zero_division=0))
            self.assertAlmostEqual(row["oos_f1"], f1_score(y, rejected, zero_division=0))
            self.assertAlmostEqual(row["in_scope_acceptance"], (~rejected[~y]).mean())
        self.assertEqual(curve.iloc[-1]["in_scope_rejected"], 0)
        self.assertEqual(curve.iloc[0]["oos_recall"], 1)

    def test_95_acceptance_respects_ties(self):
        scores = np.array([.1] * 95 + [.9] * 5 + [.8] * 10)
        y = np.array([False] * 100 + [True] * 10)
        decision = choose_threshold(threshold_curve(scores, y), "is_acceptance_95")
        self.assertGreaterEqual(decision["in_scope_acceptance"], .95)
        self.assertEqual(decision["oos_recall"], 1)
        self.assertEqual(decision["in_scope_rejected"], 5)

    def test_equal_scores_do_not_permit_partial_tie_rejection(self):
        curve = threshold_curve(np.ones(10), np.array([False] * 5 + [True] * 5))
        self.assertEqual(len(curve), 2)
        decision = choose_threshold(curve, "is_acceptance_95")
        self.assertEqual(decision["in_scope_acceptance"], 1)
        self.assertEqual(decision["oos_recall"], 0)

    def test_temperature_recovers_known_synthetic_scale(self):
        rng = np.random.default_rng(7)
        true_logits = rng.normal(size=(15000, 5)) * 1.5
        p = np.exp(log_probabilities(true_logits))
        labels = (rng.random((len(p), 1)) > np.cumsum(p, axis=1)).sum(axis=1)
        distorted_logits = true_logits * 2.5
        result = fit_temperature(distorted_logits, labels)
        self.assertLess(abs(result["temperature"] - 2.5), .15)
        self.assertLess(result["fitted_nll"], nll(distorted_logits, labels))

    def test_stable_softmax_and_argmax_invariance(self):
        logits = np.array([[10000., 10002., -10000.], [1., 1., 1.]])
        for t in (.05, .7, 20):
            log_p = log_probabilities(logits, t)
            self.assertTrue(np.isfinite(log_p).all())
            np.testing.assert_allclose(np.exp(log_p).sum(axis=1), 1)
            np.testing.assert_array_equal(logits.argmax(axis=1), log_p.argmax(axis=1))
            scores = anomaly_scores(logits, t)
            self.assertTrue(all(np.isfinite(value).all() for value in scores.values()))

    def test_temperature_fit_rejects_oos_labels(self):
        with self.assertRaises(ValueError):
            fit_temperature(np.zeros((2, 3)), np.array([0, -1]))


if __name__ == "__main__":
    unittest.main()
