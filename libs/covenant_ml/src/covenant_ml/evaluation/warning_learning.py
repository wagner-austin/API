"""Focused warning learner factories with explicit partitions and no hidden splits."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ..backends.logreg.backend import _create_logreg_model
from ..backends.protocol import PreparedClassifier
from .warning_panel import WarningRow, warning_label
from .warning_records import WarningDisclosure


class WarningLearnerFactory(Protocol):
    """Create a fresh learner using only the supplied earlier training arrays."""

    def __call__(
        self, features: NDArray[np.float64], labels: NDArray[np.int64]
    ) -> PreparedClassifier:
        """Fit a classifier; positive-class probability must occupy column one."""
        ...


def warning_features(disclosures: Sequence[WarningDisclosure]) -> NDArray[np.float64]:
    """Build the declared three ratios without outcome or identity features.

    Args:
        disclosures: Decoder-validated, available-at-cutoff disclosures.

    Returns:
        NDArray[np.float64]: Debt/EBITDA, interest cover and current ratio columns.
    """
    values: list[tuple[float, float, float]] = [
        (row["debt_to_ebitda"], row["interest_cover"], row["current_ratio"]) for row in disclosures
    ]
    return np.asarray(values, dtype=np.float64)


def warning_labels(rows: Sequence[WarningRow]) -> NDArray[np.int64]:
    """Require observed labels rather than converting censoring to a class.

    Args:
        rows: Validated prediction opportunities.

    Returns:
        NDArray[np.int64]: Explicit binary outcome array.

    Raises:
        ValueError: WARN-016 if a row's future label is censored.
    """
    labels = [warning_label(row) for row in rows]
    if None in labels:
        raise ValueError("[WARN-016] learning requires observed future labels")
    return np.asarray(labels, dtype=np.int64)


def require_warning_probabilities(values: NDArray[np.float64], count: int) -> NDArray[np.float64]:
    """Reject malformed probability output instead of clipping or broadcasting it.

    Args:
        values: Positive-class probabilities supplied by a learner or calibrator.
        count: Exact number of requested prediction rows.

    Returns:
        NDArray[np.float64]: Original valid one-dimensional probability vector.

    Raises:
        ValueError: WARN-017 for shape, nonfinite or out-of-range output.
    """
    finite: NDArray[np.bool_] = np.isfinite(values)
    if values.shape != (count,) or not bool(np.all(finite)):
        raise ValueError("[WARN-017] predictions must be a finite vector matching the cohort")
    if bool(np.any((values < 0) | (values > 1))):
        raise ValueError("[WARN-017] prediction probabilities must be in [0, 1]")
    return values


def predict_warning_probabilities(
    model: PreparedClassifier, disclosures: Sequence[WarningDisclosure]
) -> NDArray[np.float64]:
    """Predict without giving the model future follow-up evidence.

    Args:
        model: Fitted binary classifier with columns zero and one.
        disclosures: Available financial inputs only.

    Returns:
        NDArray[np.float64]: Validated positive-class probabilities.

    Raises:
        ValueError: WARN-017 for invalid binary distributions or shape.
    """
    raw = model.predict_proba(warning_features(disclosures))
    return require_warning_distribution(raw, len(disclosures))


def require_warning_distribution(raw: NDArray[np.float64], count: int) -> NDArray[np.float64]:
    """Validate both columns before extracting positive-class probabilities.

    Args:
        raw: Learner binary distributions.
        count: Requested number of rows.

    Returns:
        NDArray[np.float64]: Validated positive-class probabilities.

    Raises:
        ValueError: WARN-017 for shape, range or normalization violations.
    """
    if raw.shape != (count, 2):
        raise ValueError("[WARN-017] binary learner must return exactly two probability columns")
    negative = require_warning_probabilities(raw[:, 0], count)
    positive = require_warning_probabilities(raw[:, 1], count)
    if not bool(np.allclose(negative + positive, 1.0, rtol=0, atol=1e-12)):
        raise ValueError("[WARN-017] binary probability columns must sum to one")
    return positive


def create_warning_logistic(
    inverse_regularization: float, max_iter: int = 2000
) -> WarningLearnerFactory:
    """Create a real logistic learner with fixed regularization and no internal split.

    Args:
        inverse_regularization: Positive finite sklearn C, frozen candidate setting.
        max_iter: Positive frozen solver iteration limit.

    Returns:
        WarningLearnerFactory: Fresh deterministic fit for each explicit fold.

    Raises:
        ValueError: WARN-018 for invalid regularization, one-class fitting or
            failure to converge within the iteration budget.
    """
    if not math.isfinite(inverse_regularization) or inverse_regularization <= 0:
        raise ValueError("[WARN-018] inverse regularization must be positive and finite")
    if isinstance(max_iter, bool) or max_iter <= 0:
        raise ValueError("[WARN-018] solver iteration limit must be a positive integer")

    def fit(features: NDArray[np.float64], labels: NDArray[np.int64]) -> PreparedClassifier:
        """Fit the existing typed logistic implementation to the supplied partition.

        Args:
            features: Earlier training features only.
            labels: Earlier binary labels only.

        Returns:
            PreparedClassifier: Converged fitted classifier.

        Raises:
            ValueError: WARN-018 for one-class data or exhausted iterations.
        """
        classes: NDArray[np.int64] = np.unique(labels)
        expected: NDArray[np.int64] = np.asarray((0, 1), dtype=np.int64)
        if not bool(np.array_equal(classes, expected)):
            raise ValueError("[WARN-018] logistic fitting requires both binary classes")
        model = _create_logreg_model(
            penalty="l2",
            inverse_reg_strength=inverse_regularization,
            solver="lbfgs",
            max_iter=max_iter,
            tol=1e-8,
            random_state=0,
            class_weight=None,
            l1_ratio=None,
        )
        model.fit(features, labels)
        if int(np.max(model.n_iter_)) >= max_iter:
            raise ValueError("[WARN-018] logistic fitting exhausted its iteration limit")
        return model

    return fit


__all__ = [
    "WarningLearnerFactory",
    "create_warning_logistic",
    "predict_warning_probabilities",
    "require_warning_distribution",
    "require_warning_probabilities",
    "warning_features",
    "warning_labels",
]
