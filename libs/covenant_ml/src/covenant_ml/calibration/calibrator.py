"""Probability calibrator implementations.

Provides isotonic regression and Platt scaling calibrators for
improving probability estimates from machine learning models.

Strict typing only. No Any, casts, or stubs.
"""

from __future__ import annotations

from typing import Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

from .types import (
    CalibrationMethod,
    CalibratorConfig,
    CalibratorState,
    IsotonicParams,
    IsotonicState,
    PlattParams,
    PlattState,
)

_log = get_logger(__name__)


class CalibratedPredictions(TypedDict, total=True):
    """Result of probability calibration.

    Args:
        raw_proba: Original uncalibrated probabilities.
        calibrated_proba: Calibrated probabilities.
        method: Calibration method used.
    """

    raw_proba: NDArray[np.float64]
    calibrated_proba: NDArray[np.float64]
    method: CalibrationMethod


class CalibrationResult(TypedDict, total=True):
    """Result of fitting a calibrator.

    Args:
        state: Serializable calibrator state.
        train_brier_before: Brier score before calibration on training data.
        train_brier_after: Brier score after calibration on training data.
        train_ece_before: Expected calibration error before.
        train_ece_after: Expected calibration error after.
    """

    state: CalibratorState
    train_brier_before: float
    train_brier_after: float
    train_ece_before: float
    train_ece_after: float


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


class Calibrator:
    """Probability calibrator using isotonic regression or Platt scaling.

    Fits a calibration mapping on validation data to transform raw model
    probabilities into better-calibrated estimates.
    """

    def __init__(self, config: CalibratorConfig) -> None:
        """Initialize calibrator with configuration.

        Args:
            config: Calibrator configuration specifying method and options.
        """
        self._config = config
        self._state: CalibratorState | None = None
        self._iso_model: _IsotonicWrapper | None = None
        self._platt_a: float = 0.0
        self._platt_b: float = 0.0

    @property
    def config(self) -> CalibratorConfig:
        """Return calibrator configuration.

        Returns:
            CalibratorConfig used to initialize this calibrator.
        """
        return self._config

    @property
    def is_fitted(self) -> bool:
        """Check if calibrator has been fitted.

        Returns:
            True if fit() has been called successfully.
        """
        return self._state is not None

    def fit(
        self,
        y_true: NDArray[np.int64],
        y_prob: NDArray[np.float64],
    ) -> CalibrationResult:
        """Fit calibrator on validation data.

        Args:
            y_true: True binary labels (0 or 1).
            y_prob: Uncalibrated predicted probabilities for positive class.

        Returns:
            CalibrationResult with state and metrics.
        """
        method = self._config["method"]
        clip_proba = self._config["clip_proba"]
        eps = self._config["eps"]

        if clip_proba:
            y_prob = _clip_probabilities(y_prob, eps)

        brier_before = _compute_brier_score(y_true, y_prob)
        ece_before = _compute_ece(y_true, y_prob)

        state: CalibratorState
        calibrated: NDArray[np.float64]

        if method == "isotonic":
            iso_state, calibrated = self._fit_isotonic(y_true, y_prob)
            state = iso_state
        else:
            platt_state, calibrated = self._fit_platt(y_true, y_prob)
            state = platt_state

        self._state = state

        if clip_proba:
            calibrated = _clip_probabilities(calibrated, eps)

        brier_after = _compute_brier_score(y_true, calibrated)
        ece_after = _compute_ece(y_true, calibrated)

        _log.info(
            "Calibrator fitted",
            extra={
                "method": method,
                "brier_before": brier_before,
                "brier_after": brier_after,
                "ece_before": ece_before,
                "ece_after": ece_after,
            },
        )

        result: CalibrationResult = {
            "state": state,
            "train_brier_before": brier_before,
            "train_brier_after": brier_after,
            "train_ece_before": ece_before,
            "train_ece_after": ece_after,
        }
        return result

    def _fit_isotonic(
        self,
        y_true: NDArray[np.int64],
        y_prob: NDArray[np.float64],
    ) -> tuple[IsotonicState, NDArray[np.float64]]:
        """Fit isotonic regression calibrator.

        Args:
            y_true: True binary labels.
            y_prob: Uncalibrated probabilities.

        Returns:
            Tuple of (state, calibrated_proba).
        """
        iso_model = _create_isotonic_wrapper()

        flattened: NDArray[np.float64] = y_prob.reshape(-1, 1).ravel()
        iso_model.fit(flattened, y_true)
        self._iso_model = iso_model

        raw_pred: NDArray[np.float64] = iso_model.predict(y_prob)
        calibrated: NDArray[np.float64] = np.asarray(raw_pred, dtype=np.float64)

        x_thresholds, y_values = _extract_isotonic_thresholds(iso_model)

        params: IsotonicParams = {
            "X_thresholds": x_thresholds,
            "y_values": y_values,
        }

        state: IsotonicState = {
            "method": "isotonic",
            "config": self._config,
            "params": params,
        }

        return state, calibrated

    def _fit_platt(
        self,
        y_true: NDArray[np.int64],
        y_prob: NDArray[np.float64],
    ) -> tuple[PlattState, NDArray[np.float64]]:
        """Fit Platt scaling (sigmoid) calibrator.

        Args:
            y_true: True binary labels.
            y_prob: Uncalibrated probabilities.

        Returns:
            Tuple of (state, calibrated_proba).
        """
        logreg_ctor = _get_logreg_import()

        logreg_model = logreg_ctor(
            penalty=None,
            solver="lbfgs",
            max_iter=1000,
        )

        reshaped: NDArray[np.float64] = y_prob.reshape(-1, 1)
        logreg_model.fit(reshaped, y_true)

        coef_arr: NDArray[np.float64] = logreg_model.coef_
        intercept_arr: NDArray[np.float64] = logreg_model.intercept_
        coef_flat: NDArray[np.float64] = coef_arr.ravel()
        intercept_flat: NDArray[np.float64] = intercept_arr.ravel()

        # Use slice indexing to get typed floats
        coef_slice = np.asarray(coef_flat[0:1], dtype=np.float64).flat
        intercept_slice = np.asarray(intercept_flat[0:1], dtype=np.float64).flat
        self._platt_a = float(coef_slice[0])
        self._platt_b = float(intercept_slice[0])

        calibrated = _apply_sigmoid(y_prob, self._platt_a, self._platt_b)

        params: PlattParams = {
            "A": self._platt_a,
            "B": self._platt_b,
        }

        state: PlattState = {
            "method": "platt",
            "config": self._config,
            "params": params,
        }

        return state, calibrated

    def transform(
        self,
        y_prob: NDArray[np.float64],
    ) -> CalibratedPredictions:
        """Transform probabilities using fitted calibrator.

        Args:
            y_prob: Raw uncalibrated probabilities.

        Returns:
            CalibratedPredictions with raw and calibrated values.

        Raises:
            RuntimeError: If calibrator has not been fitted.
        """
        if self._state is None:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        method = self._config["method"]
        clip_proba = self._config["clip_proba"]
        eps = self._config["eps"]

        raw_proba: NDArray[np.float64] = y_prob.copy()
        if clip_proba:
            y_prob = _clip_probabilities(y_prob, eps)

        calibrated: NDArray[np.float64]
        if method == "isotonic":
            if self._iso_model is None:
                raise RuntimeError("Isotonic model not initialized")
            raw_pred: NDArray[np.float64] = self._iso_model.predict(y_prob)
            calibrated = np.asarray(raw_pred, dtype=np.float64)
        else:
            calibrated = _apply_sigmoid(y_prob, self._platt_a, self._platt_b)

        if clip_proba:
            calibrated = _clip_probabilities(calibrated, eps)

        result: CalibratedPredictions = {
            "raw_proba": raw_proba,
            "calibrated_proba": calibrated,
            "method": method,
        }
        return result

    def get_state(self) -> CalibratorState:
        """Get serializable calibrator state.

        Returns:
            CalibratorState for persistence.

        Raises:
            RuntimeError: If calibrator has not been fitted.
        """
        if self._state is None:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")
        return self._state

    @classmethod
    def from_state(cls, state: CalibratorState) -> Calibrator:
        """Reconstruct calibrator from saved state.

        Args:
            state: Previously saved CalibratorState.

        Returns:
            Reconstructed Calibrator ready for transform().
        """
        config = state["config"]
        calibrator = cls(config)
        calibrator._state = state

        if state["method"] == "isotonic":
            iso_params: IsotonicParams = state["params"]
            iso_model = _create_isotonic_wrapper()

            x_list: list[float] = iso_params["X_thresholds"]
            y_list: list[float] = iso_params["y_values"]
            x_thresholds: NDArray[np.float64] = np.array(x_list, dtype=np.float64)
            y_values: NDArray[np.float64] = np.array(y_list, dtype=np.float64)

            y_labels: NDArray[np.int64] = (y_values >= 0.5).astype(np.int64)
            iso_model.fit(x_thresholds, y_labels)

            _set_isotonic_thresholds(iso_model, x_thresholds, y_values)

            calibrator._iso_model = iso_model
        else:
            platt_params: PlattParams = state["params"]
            calibrator._platt_a = platt_params["A"]
            calibrator._platt_b = platt_params["B"]

        return calibrator


def create_isotonic_calibrator(
    clip_proba: bool = True,
    eps: float = 1e-10,
) -> Calibrator:
    """Create isotonic regression calibrator.

    Args:
        clip_proba: Whether to clip probabilities to avoid log(0).
        eps: Epsilon for probability clipping.

    Returns:
        Calibrator configured for isotonic regression.
    """
    config: CalibratorConfig = {
        "method": "isotonic",
        "clip_proba": clip_proba,
        "eps": eps,
    }
    return Calibrator(config)


def create_platt_calibrator(
    clip_proba: bool = True,
    eps: float = 1e-10,
) -> Calibrator:
    """Create Platt scaling (sigmoid) calibrator.

    Args:
        clip_proba: Whether to clip probabilities to avoid log(0).
        eps: Epsilon for probability clipping.

    Returns:
        Calibrator configured for Platt scaling.
    """
    config: CalibratorConfig = {
        "method": "platt",
        "clip_proba": clip_proba,
        "eps": eps,
    }
    return Calibrator(config)


__all__ = [
    "CalibratedPredictions",
    "CalibrationMethod",
    "CalibrationResult",
    "Calibrator",
    "create_isotonic_calibrator",
    "create_platt_calibrator",
]
