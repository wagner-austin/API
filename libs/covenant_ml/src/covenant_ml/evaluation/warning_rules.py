"""A predeclared ratio-rule baseline with no fitted or outcome-dependent thresholds."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ..backends.protocol import PreparedClassifier


class WarningRatioRule:
    """Stateless baseline and learner factory for the same chronological workflow.

    Its zero/one output encodes a rule decision, not an estimated probability.
    The workflow fits its probability calibration only on the calibration partition.
    """

    def __init__(self, max_leverage: float, min_cover: float, min_current: float) -> None:
        """Freeze the three ratio thresholds.

        Args:
            max_leverage: Alert when debt/EBITDA exceeds this finite value.
            min_cover: Alert when interest cover falls below this finite value.
            min_current: Alert when current ratio falls below this finite value.

        Raises:
            ValueError: WARN-028 if a threshold is nonfinite.
        """
        if not all(math.isfinite(v) for v in (max_leverage, min_cover, min_current)):
            raise ValueError("[WARN-028] ratio rule thresholds must be finite")
        self._max_leverage = max_leverage
        self._min_cover = min_cover
        self._min_current = min_current

    def __call__(
        self, features: NDArray[np.float64], labels: NDArray[np.int64]
    ) -> PreparedClassifier:
        """Supply the fixed rule without fitting thresholds to the passed outcomes.

        Args:
            features: Training array, deliberately not used for threshold selection.
            labels: Training labels, deliberately not used for threshold selection.

        Returns:
            PreparedClassifier: This stateless, predeclared ratio rule.
        """
        return self

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Encode the fixed ratio decision in two columns for common calibration.

        Args:
            x: Debt/EBITDA, interest cover and current ratio matrix.

        Returns:
            NDArray[np.float64]: Zero/one rule decisions as binary distributions.

        Raises:
            ValueError: WARN-028 for a matrix with the wrong number of columns.
        """
        shape: tuple[int, ...] = x.shape
        if len(shape) != 2 or shape[1] != 3:
            raise ValueError("[WARN-028] ratio rules require exactly three feature columns")
        alert: NDArray[np.bool_] = (
            (x[:, 0] > self._max_leverage)
            | (x[:, 1] < self._min_cover)
            | (x[:, 2] < self._min_current)
        )
        probabilities: NDArray[np.float64] = alert.astype(np.float64)
        return np.column_stack((1 - probabilities, probabilities))


__all__ = ["WarningRatioRule"]
