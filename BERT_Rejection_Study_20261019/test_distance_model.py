"""Mathematical and persistence checks; run: python -m unittest -v test_distance_model."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.covariance import LedoitWolf

from distance_model import SharedMahalanobisDetector


class SharedMahalanobisTests(unittest.TestCase):
    def setUp(self):
        generator = np.random.default_rng(1979)
        self.labels = np.repeat([2, 7, 19], [17, 23, 19])
        means = np.array([[-2.0, 1.0, 0.5], [0.0, -3.0, 1.0], [4.0, 0.5, -1.0]])
        self.features = generator.normal(size=(59, 3)) @ np.array(
            [[0.7, 0.3, 0.0], [0.0, 1.2, 0.5], [0.2, 0.0, 0.8]]
        ) + np.repeat(means, [17, 23, 19], axis=0)
        self.queries = generator.normal(size=(11, 3)) * 3.0
        self.detector = SharedMahalanobisDetector().fit(self.features, self.labels)

    def test_direct_covariance_solve_matches_blocked_quadratic(self):
        expected = np.empty((len(self.queries), 3))
        for row, query in enumerate(self.queries):
            for column, mean in enumerate(self.detector.class_means_):
                delta = query - mean
                expected[row, column] = delta @ np.linalg.solve(self.detector.covariance_, delta)
        scores, nearest = self.detector.score_with_nearest(self.queries, batch_size=4)
        assert_allclose(scores, expected.min(axis=1), rtol=1e-12, atol=1e-12)
        assert_array_equal(nearest, self.detector.classes_[expected.argmin(axis=1)])
        assert_allclose(scores, self.detector.score_samples(self.queries, batch_size=100), rtol=1e-13)
        self.assertEqual(scores.dtype, np.dtype("float64"))

    def test_means_counts_and_ledoit_wolf_residual_protocol(self):
        expected_means = np.array([self.features[self.labels == label].mean(axis=0) for label in [2, 7, 19]])
        assert_allclose(self.detector.class_means_, expected_means, rtol=1e-14, atol=1e-14)
        residuals = self.features - expected_means[np.searchsorted([2, 7, 19], self.labels)]
        expected = LedoitWolf(assume_centered=True).fit(residuals)
        assert_allclose(self.detector.covariance_, expected.covariance_, rtol=1e-13, atol=1e-14)
        assert_allclose(self.detector.precision_, expected.precision_, rtol=1e-13, atol=1e-14)
        self.assertAlmostEqual(self.detector.shrinkage_, expected.shrinkage_, places=13)
        assert_array_equal(self.detector.class_counts_, [17, 23, 19])
        self.assertEqual(self.detector.n_features_in_, 3)
        self.assertEqual(self.detector.n_samples_seen_, 59)

    def test_translation_invariance(self):
        translation = np.array([1e6, -2e6, 3e6])
        translated = SharedMahalanobisDetector().fit(self.features + translation, self.labels)
        scores, nearest = self.detector.score_with_nearest(self.queries)
        translated_scores, translated_nearest = translated.score_with_nearest(self.queries + translation)
        assert_allclose(translated_scores, scores, rtol=2e-9, atol=2e-9)
        assert_array_equal(translated_nearest, nearest)

    def test_centroid_distances_and_empty_queries(self):
        scores, nearest = self.detector.score_with_nearest(self.detector.class_means_)
        assert_allclose(scores, 0.0, atol=1e-12)
        assert_array_equal(nearest, self.detector.classes_)
        empty_scores, empty_nearest = self.detector.score_with_nearest(np.empty((0, 3)))
        self.assertEqual(empty_scores.shape, (0,))
        self.assertEqual(empty_nearest.shape, (0,))

    def test_save_load_round_trip_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detector.npz"
            self.detector.save(path)
            loaded = SharedMahalanobisDetector.load(path)
            assert_array_equal(loaded.class_means_, self.detector.class_means_)
            assert_array_equal(loaded.covariance_, self.detector.covariance_)
            assert_array_equal(loaded.score_samples(self.queries), self.detector.score_samples(self.queries))
            with self.assertRaises(FileExistsError):
                self.detector.save(path)
            self.detector.save(path, overwrite=True)

    def test_invalid_training_inputs(self):
        invalid = [
            (np.array([1.0, 2.0]), [0, 1]), (np.empty((0, 3)), []),
            (np.empty((2, 0)), [0, 1]), (np.ones((1, 3)), [0]),
            (self.features, self.labels[:-1]), (self.features, self.labels[:, None]),
            (self.features, self.labels.astype(float)), (self.features, self.labels - 3),
            (self.features.astype(complex), self.labels),
            (self.features.astype(str), self.labels),
        ]
        for features, labels in invalid:
            with self.subTest(shape=np.shape(features), labels_shape=np.shape(labels)):
                with self.assertRaises(ValueError):
                    SharedMahalanobisDetector().fit(features, labels)
        for value in [np.nan, np.inf, -np.inf]:
            features = self.features.copy()
            features[0, 0] = value
            with self.assertRaises(ValueError):
                SharedMahalanobisDetector().fit(features, self.labels)
        with self.assertRaises(ValueError):
            SharedMahalanobisDetector().fit(np.ones((4, 3)), [0, 0, 1, 1])

    def test_invalid_scoring_inputs(self):
        with self.assertRaises(RuntimeError):
            SharedMahalanobisDetector().score_samples(self.queries)
        for values in [np.ones((2, 4)), np.ones(3), np.array([[1.0, 2.0, np.inf]])]:
            with self.assertRaises(ValueError):
                self.detector.score_samples(values)
        for batch_size in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                self.detector.score_samples(self.queries, batch_size=batch_size)

    def test_rejects_malformed_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detector.npz"
            self.detector.save(path)
            with np.load(path, allow_pickle=False) as archive:
                contents = {name: archive[name].copy() for name in archive.files}
            contents["precision"] = np.zeros((3, 3))
            np.savez(path, **contents)
            with self.assertRaises(ValueError):
                SharedMahalanobisDetector.load(path)
            contents["precision"] = 2.0 * self.detector.precision_
            np.savez(path, **contents)
            with self.assertRaises(ValueError):
                SharedMahalanobisDetector.load(path)


if __name__ == "__main__":
    unittest.main()
