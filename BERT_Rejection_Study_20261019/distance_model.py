"""Training-only shared-covariance Mahalanobis OOS scoring.

Features are the frozen BERT final-hidden-state CLS vectors. This module does
not normalize or project features, choose rejection thresholds, or replace the
original classifier prediction. Nearest-centroid IDs are diagnostics only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf


class SharedMahalanobisDetector:
    """Minimum squared distance to training class means, with one covariance.

    ``fit`` estimates float64 class means and fits LedoitWolf(assume_centered=True)
    to the training residuals X - mean[y]. Scoring uses that shared precision
    matrix and a centered, blocked quadratic expansion. Centering is a change
    of origin only; it is not feature normalization or dimensionality reduction.
    """

    FORMAT_VERSION = 1

    @staticmethod
    def _features(features, *, expected_dim=None, allow_empty=False):
        values = np.asarray(features)
        if values.ndim != 2:
            raise ValueError("Features must be a two-dimensional (rows, features) array")
        if values.shape[1] == 0 or (not allow_empty and values.shape[0] == 0):
            raise ValueError("Features must contain at least one row and one feature")
        if expected_dim is not None and values.shape[1] != expected_dim:
            raise ValueError(f"Expected {expected_dim} features, received {values.shape[1]}")
        if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
            raise ValueError("Features must contain real numeric values")
        values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Features must contain only finite values")
        return values

    @staticmethod
    def _labels(labels, n_rows):
        values = np.asarray(labels)
        if values.ndim != 1 or len(values) != n_rows:
            raise ValueError("Labels must be a one-dimensional array with one ID per row")
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError("Labels must be non-negative integer class IDs")
        if np.any(values < 0) or np.any(values > np.iinfo(np.int64).max):
            raise ValueError("Labels must fit non-negative int64 class IDs")
        return values.astype(np.int64, copy=False)

    def _require_fitted(self):
        if not hasattr(self, "precision_"):
            raise RuntimeError("Call fit or load before scoring or saving")

    def fit(self, train_features, train_label_ids):
        """Fit from in-scope training rows only and return this detector."""
        features = self._features(train_features)
        if features.shape[0] < 2:
            raise ValueError("At least two training rows are required")
        labels = self._labels(train_label_ids, len(features))
        classes, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)

        origin = features.mean(axis=0)
        centered = features - origin
        offsets = np.zeros((len(classes), features.shape[1]), dtype=np.float64)
        np.add.at(offsets, inverse, centered)
        offsets /= counts[:, None]
        residuals = centered - offsets[inverse]
        estimator = LedoitWolf(assume_centered=True).fit(residuals)
        covariance = np.asarray(estimator.covariance_, dtype=np.float64)
        precision = np.asarray(estimator.precision_, dtype=np.float64)
        if not np.isfinite(covariance).all() or not np.isfinite(precision).all():
            raise ValueError("Training residuals produced non-finite covariance statistics")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "Ledoit-Wolf residual covariance is not positive definite; "
                "no extra regularization is silently added"
            ) from error

        self.classes_ = classes.astype(np.int64)
        self.class_counts_ = counts.astype(np.int64)
        self.origin_ = origin
        self.class_offsets_ = offsets
        self.class_means_ = offsets + origin
        self.covariance_ = covariance
        self.precision_ = precision
        self.shrinkage_ = float(estimator.shrinkage_)
        self.n_features_in_ = int(features.shape[1])
        self.n_samples_seen_ = int(features.shape[0])
        self._prepare_scoring()
        return self

    def _prepare_scoring(self):
        self._mean_precision = self.class_offsets_ @ self.precision_
        self._mean_squared_norm = np.einsum(
            "ij,ij->i", self._mean_precision, self.class_offsets_
        )

    def score_with_nearest(self, features, *, batch_size=256):
        """Return (squared_distances, nearest_class_ids); larger is more OOS.

        Equal distances resolve to the smallest class ID. Memory scales with a
        feature batch and its class-distance matrix, never N x classes x features.
        """
        self._require_fitted()
        values = self._features(features, expected_dim=self.n_features_in_, allow_empty=True)
        if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer)):
            raise ValueError("batch_size must be a positive integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        scores = np.empty(len(values), dtype=np.float64)
        nearest = np.empty(len(values), dtype=np.int64)
        epsilon = np.finfo(np.float64).eps
        for start in range(0, len(values), int(batch_size)):
            stop = min(start + int(batch_size), len(values))
            centered = values[start:stop] - self.origin_
            projected = centered @ self.precision_
            squared_norm = np.einsum("ij,ij->i", projected, centered)
            cross = centered @ self._mean_precision.T
            distances = squared_norm[:, None] - 2.0 * cross + self._mean_squared_norm[None, :]
            scale = (
                np.abs(squared_norm[:, None])
                + 2.0 * np.abs(cross)
                + np.abs(self._mean_squared_norm[None, :])
            )
            tolerance = 64.0 * epsilon * np.maximum(scale, 1.0)
            if not np.isfinite(distances).all():
                raise FloatingPointError("Non-finite Mahalanobis distance; check feature magnitudes")
            if np.any(distances < -tolerance):
                raise FloatingPointError("Negative squared distance exceeds floating-point tolerance")
            np.maximum(distances, 0.0, out=distances)
            indices = distances.argmin(axis=1)
            scores[start:stop] = distances[np.arange(stop - start), indices]
            nearest[start:stop] = self.classes_[indices]
        return scores, nearest

    def score_samples(self, features, *, batch_size=256):
        """Return one float64 minimum squared Mahalanobis distance per row."""
        return self.score_with_nearest(features, batch_size=batch_size)[0]

    def save(self, path, *, overwrite=False):
        """Save numeric arrays to an NPZ archive; refuse overwrites by default."""
        self._require_fitted()
        destination = Path(path)
        if destination.suffix.lower() != ".npz":
            raise ValueError("Detector archive path must end in .npz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb" if overwrite else "xb") as handle:
            np.savez_compressed(
                handle,
                format_version=np.array(self.FORMAT_VERSION, dtype=np.int64),
                classes=self.classes_,
                class_counts=self.class_counts_,
                origin=self.origin_,
                class_offsets=self.class_offsets_,
                class_means=self.class_means_,
                covariance=self.covariance_,
                precision=self.precision_,
                shrinkage=np.array(self.shrinkage_, dtype=np.float64),
                n_features=np.array(self.n_features_in_, dtype=np.int64),
                n_samples=np.array(self.n_samples_seen_, dtype=np.int64),
            )
        return destination

    @classmethod
    def load(cls, path):
        """Load and validate a numeric NPZ archive without pickle execution."""
        required = {
            "format_version", "classes", "class_counts", "origin", "class_offsets",
            "class_means", "covariance", "precision", "shrinkage", "n_features", "n_samples",
        }
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != required:
                raise ValueError("Detector archive has missing or unexpected fields")
            arrays = {name: np.array(archive[name], copy=True) for name in required}
        for name in ("format_version", "n_features", "n_samples"):
            if arrays[name].shape != () or not np.issubdtype(arrays[name].dtype, np.integer):
                raise ValueError(f"Invalid integer scalar {name} in detector archive")
        if int(arrays["format_version"]) != cls.FORMAT_VERSION:
            raise ValueError("Unsupported detector archive version")
        dimensions = int(arrays["n_features"])
        samples = int(arrays["n_samples"])
        if dimensions < 1 or samples < 2:
            raise ValueError("Invalid dimensions or sample count in detector archive")
        class_ids = arrays["classes"]
        if class_ids.ndim != 1 or len(class_ids) == 0:
            raise ValueError("Invalid class IDs in detector archive")
        class_ids = cls._labels(class_ids, len(class_ids))
        if not np.array_equal(class_ids, np.unique(class_ids)):
            raise ValueError("Archived class IDs must be sorted and unique")
        counts = cls._labels(arrays["class_counts"], len(class_ids))
        if np.any(counts == 0) or sum(int(count) for count in counts) != samples:
            raise ValueError("Invalid class counts in detector archive")
        shapes = {
            "origin": (dimensions,), "class_offsets": (len(class_ids), dimensions),
            "class_means": (len(class_ids), dimensions), "covariance": (dimensions, dimensions),
            "precision": (dimensions, dimensions), "shrinkage": (),
        }
        for name, shape in shapes.items():
            array = arrays[name]
            if (array.shape != shape or not np.issubdtype(array.dtype, np.floating)
                    or not np.isfinite(array).all()):
                raise ValueError(f"Invalid {name} in detector archive")
        shrinkage = float(arrays["shrinkage"])
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("Archived shrinkage must lie in [0, 1]")
        for name in ("covariance", "precision"):
            if not np.allclose(arrays[name], arrays[name].T, rtol=1e-10, atol=1e-12):
                raise ValueError(f"Archived {name} must be symmetric")
            try:
                np.linalg.cholesky(arrays[name])
            except np.linalg.LinAlgError as error:
                raise ValueError(f"Archived {name} must be positive definite") from error
        if not np.allclose(
            arrays["covariance"] @ arrays["precision"], np.eye(dimensions), rtol=1e-5, atol=1e-5
        ):
            raise ValueError("Archived covariance and precision are inconsistent")
        if not np.allclose(
            arrays["class_means"], arrays["origin"] + arrays["class_offsets"], rtol=1e-12, atol=1e-12
        ):
            raise ValueError("Archived class means and centered offsets disagree")
        result = cls()
        result.classes_ = class_ids
        result.class_counts_ = counts
        result.origin_ = arrays["origin"].astype(np.float64)
        result.class_offsets_ = arrays["class_offsets"].astype(np.float64)
        result.class_means_ = arrays["class_means"].astype(np.float64)
        result.covariance_ = arrays["covariance"].astype(np.float64)
        result.precision_ = arrays["precision"].astype(np.float64)
        result.shrinkage_ = shrinkage
        result.n_features_in_ = dimensions
        result.n_samples_seen_ = samples
        result._prepare_scoring()
        return result
