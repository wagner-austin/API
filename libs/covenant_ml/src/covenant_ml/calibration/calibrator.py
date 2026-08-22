"""Probability calibrator implementations.

Provides isotonic regression and Platt scaling calibrators for
improving probability estimates from machine learning models.

Strict typing only. No Any, casts, or stubs.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

from covenant_ml.calibration._calibrator_math import (
    _apply_sigmoid,
    _clip_probabilities,
    _compute_brier_score,
    _compute_ece,
    _create_isotonic_wrapper,
    _extract_isotonic_thresholds,
    _get_logreg_import,
    _IsotonicWrapper,
    _set_isotonic_thresholds,
)

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
    "CalibrationResult",
    "Calibrator",
    "create_isotonic_calibrator",
    "create_platt_calibrator",
]
