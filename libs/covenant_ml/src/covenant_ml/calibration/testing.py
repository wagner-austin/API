"""Testing utilities for calibration module.

Public test utilities exported for consumers to use.
"""

from __future__ import annotations

from .types import (
    CalibratorConfig,
    IsotonicParams,
    IsotonicState,
    PlattParams,
    PlattState,
)


def make_isotonic_config(
    clip_proba: bool = True,
    eps: float = 1e-10,
) -> CalibratorConfig:
    """Create isotonic calibrator config for tests.

    Args:
        clip_proba: Whether to clip probabilities.
        eps: Epsilon for clipping.

    Returns:
        CalibratorConfig for isotonic regression.
    """
    return {
        "method": "isotonic",
        "clip_proba": clip_proba,
        "eps": eps,
    }


def make_platt_config(
    clip_proba: bool = True,
    eps: float = 1e-10,
) -> CalibratorConfig:
    """Create Platt scaling calibrator config for tests.

    Args:
        clip_proba: Whether to clip probabilities.
        eps: Epsilon for clipping.

    Returns:
        CalibratorConfig for Platt scaling.
    """
    return {
        "method": "platt",
        "clip_proba": clip_proba,
        "eps": eps,
    }


def make_isotonic_state(
    x_thresholds: list[float] | None = None,
    y_values: list[float] | None = None,
    clip_proba: bool = True,
    eps: float = 1e-10,
) -> IsotonicState:
    """Create isotonic calibrator state for tests.

    Args:
        x_thresholds: Input probability thresholds.
        y_values: Corresponding calibrated values.
        clip_proba: Whether to clip probabilities.
        eps: Epsilon for clipping.

    Returns:
        IsotonicState for testing.
    """
    config: CalibratorConfig = {
        "method": "isotonic",
        "clip_proba": clip_proba,
        "eps": eps,
    }

    params: IsotonicParams = {
        "X_thresholds": x_thresholds if x_thresholds is not None else [0.0, 0.5, 1.0],
        "y_values": y_values if y_values is not None else [0.1, 0.5, 0.9],
    }

    return {
        "method": "isotonic",
        "config": config,
        "params": params,
    }


def make_platt_state(
    slope: float = 1.0,
    intercept: float = 0.0,
    clip_proba: bool = True,
    eps: float = 1e-10,
) -> PlattState:
    """Create Platt scaling calibrator state for tests.

    Args:
        slope: Slope parameter (A in sigmoid formula).
        intercept: Intercept parameter (B in sigmoid formula).
        clip_proba: Whether to clip probabilities.
        eps: Epsilon for clipping.

    Returns:
        PlattState for testing.
    """
    config: CalibratorConfig = {
        "method": "platt",
        "clip_proba": clip_proba,
        "eps": eps,
    }

    params: PlattParams = {
        "A": slope,
        "B": intercept,
    }

    return {
        "method": "platt",
        "config": config,
        "params": params,
    }


__all__ = [
    "make_isotonic_config",
    "make_isotonic_state",
    "make_platt_config",
    "make_platt_state",
]
