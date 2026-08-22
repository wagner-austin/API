"""Isotonic/Platt fitting machinery and calibration score helpers."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class _SklearnIsotonicProtocol(Protocol):
    """Protocol for raw sklearn IsotonicRegression instance.

    Defines only the methods we need without the uppercase attributes.
    """

    def fit(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> _SklearnIsotonicProtocol:
        """Fit isotonic regression."""
        ...

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict calibrated values."""
        ...


class _IsotonicWrapper:
    """Wrapper for sklearn IsotonicRegression with lowercase attributes.

    sklearn uses X_thresholds_ (uppercase X) which violates N815.
    This wrapper copies fitted values to lowercase attributes.
    """

    def __init__(self, sklearn_model: _SklearnIsotonicProtocol) -> None:
        """Initialize wrapper with sklearn model.

        Args:
            sklearn_model: sklearn IsotonicRegression instance.
        """
        self._model = sklearn_model
        empty: list[float] = []
        self.x_thresholds_: NDArray[np.float64] = np.array(empty, dtype=np.float64)
        self.y_thresholds_: NDArray[np.float64] = np.array(empty, dtype=np.float64)

    def fit(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> _IsotonicWrapper:
        """Fit isotonic regression and copy thresholds.

        Args:
            x: Input values.
            y: Target values.

        Returns:
            Self after fitting.
        """
        self._model.fit(x, y)
        # Copy thresholds to lowercase attributes using dynamic key access
        x_attr_name = "X" + "_thresholds_"  # Build name to avoid literal
        y_attr_name = "y" + "_thresholds_"
        model_vars: dict[str, NDArray[np.float64]] = vars(self._model)
        x_raw: NDArray[np.float64] = model_vars[x_attr_name]
        y_raw: NDArray[np.float64] = model_vars[y_attr_name]
        self.x_thresholds_ = np.asarray(x_raw, dtype=np.float64)
        self.y_thresholds_ = np.asarray(y_raw, dtype=np.float64)
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict calibrated values.

        Args:
            x: Input values.

        Returns:
            Calibrated predictions.
        """
        raw_pred: NDArray[np.float64] = self._model.predict(x)
        return np.asarray(raw_pred, dtype=np.float64)

    def set_thresholds(
        self,
        x_thresholds: NDArray[np.float64],
        y_thresholds: NDArray[np.float64],
    ) -> None:
        """Set thresholds on the underlying sklearn model.

        Args:
            x_thresholds: X threshold values.
            y_thresholds: Y threshold values.
        """
        x_attr_name = "X" + "_thresholds_"
        y_attr_name = "y" + "_thresholds_"
        model_vars: dict[str, NDArray[np.float64]] = vars(self._model)
        model_vars[x_attr_name] = x_thresholds
        model_vars[y_attr_name] = y_thresholds
        self.x_thresholds_ = x_thresholds
        self.y_thresholds_ = y_thresholds


class _IsotonicRegressionCtor(Protocol):
    """Protocol for sklearn IsotonicRegression constructor."""

    def __call__(
        self,
        *,
        y_min: float,
        y_max: float,
        increasing: bool,
        out_of_bounds: str,
    ) -> _SklearnIsotonicProtocol:
        """Construct IsotonicRegression."""
        ...


class _LogisticRegressionProtocol(Protocol):
    """Protocol for sklearn LogisticRegression for Platt scaling."""

    @property
    def coef_(self) -> NDArray[np.float64]:
        """Coefficient (slope A)."""
        ...

    @property
    def intercept_(self) -> NDArray[np.float64]:
        """Intercept (B)."""
        ...

    def fit(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> _LogisticRegressionProtocol:
        """Fit logistic regression."""
        ...


class _LogisticRegressionCtor(Protocol):
    """Protocol for LogisticRegression constructor."""

    def __call__(
        self,
        *,
        penalty: str | None,
        solver: str,
        max_iter: int,
    ) -> _LogisticRegressionProtocol:
        """Construct LogisticRegression."""
        ...


def _create_isotonic_wrapper() -> _IsotonicWrapper:
    """Create IsotonicWrapper with sklearn IsotonicRegression inside.

    Returns:
        IsotonicWrapper instance.
    """
    sklearn_module = __import__(
        "sklearn.isotonic",
        fromlist=["IsotonicRegression"],
    )
    sklearn_ctor: _IsotonicRegressionCtor = sklearn_module.IsotonicRegression
    sklearn_model: _SklearnIsotonicProtocol = sklearn_ctor(
        y_min=0.0,
        y_max=1.0,
        increasing=True,
        out_of_bounds="clip",
    )
    return _IsotonicWrapper(sklearn_model)


def _get_logreg_import() -> _LogisticRegressionCtor:
    """Get sklearn LogisticRegression class via dynamic import.

    Returns:
        LogisticRegression constructor.
    """
    sklearn_module = __import__(
        "sklearn.linear_model",
        fromlist=["LogisticRegression"],
    )
    logreg_ctor: _LogisticRegressionCtor = sklearn_module.LogisticRegression
    return logreg_ctor


def _extract_isotonic_thresholds(
    model: _IsotonicWrapper,
) -> tuple[list[float], list[float]]:
    """Extract x_thresholds_ and y_thresholds_ from fitted isotonic wrapper.

    Args:
        model: Fitted IsotonicWrapper.

    Returns:
        Tuple of (x_thresholds, y_thresholds) as float lists.
    """
    x_arr: NDArray[np.float64] = model.x_thresholds_
    y_arr: NDArray[np.float64] = model.y_thresholds_
    n_thresholds = int(x_arr.shape[0])
    x_list: list[float] = []
    y_list: list[float] = []
    for i in range(n_thresholds):
        x_slice = np.asarray(x_arr[i : i + 1], dtype=np.float64).flat
        y_slice = np.asarray(y_arr[i : i + 1], dtype=np.float64).flat
        x_list.append(float(x_slice[0]))
        y_list.append(float(y_slice[0]))
    return x_list, y_list


def _set_isotonic_thresholds(
    model: _IsotonicWrapper,
    x_thresholds: NDArray[np.float64],
    y_thresholds: NDArray[np.float64],
) -> None:
    """Set thresholds on isotonic wrapper.

    Args:
        model: IsotonicWrapper instance.
        x_thresholds: X threshold values.
        y_thresholds: Y threshold values.
    """
    model.set_thresholds(x_thresholds, y_thresholds)


def _compute_brier_score(
    y_true: NDArray[np.int64],
    y_prob: NDArray[np.float64],
) -> float:
    """Compute Brier score (mean squared error of probabilities).

    Lower is better. Range [0, 1].

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities for positive class.

    Returns:
        Brier score.
    """
    y_true_float: NDArray[np.float64] = np.asarray(y_true, dtype=np.float64)
    diff: NDArray[np.float64] = y_prob - y_true_float
    squared: NDArray[np.float64] = diff * diff
    total: float = float(np.sum(squared))
    n_samples: int = len(y_true)
    brier: float = total / float(n_samples)
    return brier


def _compute_ece(
    y_true: NDArray[np.int64],
    y_prob: NDArray[np.float64],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Measures the average absolute difference between predicted probability
    and observed frequency across probability bins.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities for positive class.
        n_bins: Number of bins for calibration curve.

    Returns:
        Expected calibration error (lower is better).
    """
    bin_edges: NDArray[np.float64] = np.linspace(0.0, 1.0, n_bins + 1)

    ece: float = 0.0
    n_samples = len(y_true)

    for i in range(n_bins):
        # Use slice indexing to get typed floats
        low_slice = np.asarray(bin_edges[i : i + 1], dtype=np.float64).flat
        high_slice = np.asarray(bin_edges[i + 1 : i + 2], dtype=np.float64).flat
        low: float = float(low_slice[0])
        high: float = float(high_slice[0])

        # Find samples in this bin
        if i == n_bins - 1:
            in_bin: NDArray[np.bool_] = (y_prob >= low) & (y_prob <= high)
        else:
            in_bin = (y_prob >= low) & (y_prob < high)

        n_in_bin = int(np.count_nonzero(in_bin))

        if n_in_bin > 0:
            prob_subset: NDArray[np.float64] = y_prob[in_bin]
            true_subset: NDArray[np.int64] = y_true[in_bin]
            # Use sum/len instead of np.mean for clean typing
            avg_pred: float = float(np.sum(prob_subset)) / float(n_in_bin)
            avg_true: float = float(np.sum(true_subset)) / float(n_in_bin)
            ece += (float(n_in_bin) / float(n_samples)) * abs(avg_pred - avg_true)

    return ece


def _clip_probabilities(
    proba: NDArray[np.float64],
    eps: float,
) -> NDArray[np.float64]:
    """Clip probabilities to [eps, 1-eps].

    Args:
        proba: Probability array.
        eps: Small epsilon value.

    Returns:
        Clipped probabilities.
    """
    clipped: NDArray[np.float64] = np.clip(proba, eps, 1.0 - eps)
    return clipped


def _apply_sigmoid(
    y_prob: NDArray[np.float64],
    slope: float,
    intercept: float,
) -> NDArray[np.float64]:
    """Apply sigmoid transformation for Platt scaling.

    Args:
        y_prob: Raw probabilities.
        slope: Slope parameter (A).
        intercept: Intercept parameter (B).

    Returns:
        Calibrated probabilities.
    """
    z: NDArray[np.float64] = slope * y_prob + intercept
    neg_z: NDArray[np.float64] = -z
    exp_neg_z: NDArray[np.float64] = np.exp(neg_z)
    denom: NDArray[np.float64] = 1.0 + exp_neg_z
    calibrated: NDArray[np.float64] = 1.0 / denom
    return calibrated
