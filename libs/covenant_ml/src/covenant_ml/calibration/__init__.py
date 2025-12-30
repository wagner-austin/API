"""Probability calibration module for covenant_ml.

Provides post-hoc calibration methods to improve probability estimates:
- Isotonic regression (non-parametric, monotonic)
- Platt scaling (parametric, sigmoid)

Both methods fit a calibration mapping on held-out validation data
and transform raw model outputs to better-calibrated probabilities.
"""

from __future__ import annotations

from .calibrator import (
    CalibratedPredictions,
    CalibrationMethod,
    CalibrationResult,
    Calibrator,
    create_isotonic_calibrator,
    create_platt_calibrator,
)
from .types import (
    CalibratorConfig,
    CalibratorState,
    decode_calibrator_state,
    encode_calibrator_state,
)

__all__ = [
    "CalibratedPredictions",
    "CalibrationMethod",
    "CalibrationResult",
    "Calibrator",
    "CalibratorConfig",
    "CalibratorState",
    "create_isotonic_calibrator",
    "create_platt_calibrator",
    "decode_calibrator_state",
    "encode_calibrator_state",
]
