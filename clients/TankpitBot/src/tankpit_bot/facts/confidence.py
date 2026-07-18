"""Confidence arithmetic over the [0.0, 1.0] interval.

Three operations per the Phase 1 spec: combine independent evidence,
weighted combine, and exponential decay by age. Every operation
enforces the confidence-in-bounds contract on its inputs and its
output -- an out-of-range value anywhere raises
:class:`~tankpit_bot.contracts.base.ConfidenceOutOfBoundsError` at the
operation, not downstream.
"""

from __future__ import annotations

import math

from tankpit_bot.contracts.base import ConfidenceOutOfBoundsError
from tankpit_bot.contracts.enforcement import require

CONFIDENCE_MIN = 0.0
CONFIDENCE_MAX = 1.0


def require_confidence_in_bounds(value: float) -> float:
    """Validate a confidence value against the [0.0, 1.0] contract.

    Args:
        value: Confidence value to validate.

    Returns:
        The validated value, unchanged.

    Raises:
        ConfidenceOutOfBoundsError: If the value is out of range.
    """
    require(
        CONFIDENCE_MIN <= value <= CONFIDENCE_MAX,
        ConfidenceOutOfBoundsError,
        value=repr(value),
    )
    return value


def combine_independent(first: float, second: float) -> float:
    """Combine two independent confirmations of the same belief.

    Noisy-OR: two independent sources each failing to be wrong.
    ``combine_independent(0.8, 0.5) == 0.9``.

    Args:
        first: Confidence from the first source.
        second: Confidence from the second source.

    Returns:
        Combined confidence, always >= max(first, second).

    Raises:
        ConfidenceOutOfBoundsError: If any input is out of range.
    """
    require_confidence_in_bounds(first)
    require_confidence_in_bounds(second)
    return require_confidence_in_bounds(1.0 - (1.0 - first) * (1.0 - second))


def combine_weighted(
    first: float,
    first_weight: float,
    second: float,
    second_weight: float,
) -> float:
    """Combine two confidences as a weighted average.

    Args:
        first: Confidence from the first source.
        first_weight: Non-negative weight of the first source.
        second: Confidence from the second source.
        second_weight: Non-negative weight of the second source.

    Returns:
        Weighted-average confidence.

    Raises:
        ConfidenceOutOfBoundsError: If any confidence is out of range,
            if a weight is negative, or if both weights are zero.
    """
    require_confidence_in_bounds(first)
    require_confidence_in_bounds(second)
    require(
        first_weight >= 0.0 and second_weight >= 0.0,
        ConfidenceOutOfBoundsError,
        first_weight=repr(first_weight),
        second_weight=repr(second_weight),
    )
    total = first_weight + second_weight
    require(total > 0.0, ConfidenceOutOfBoundsError, total_weight=repr(total))
    combined = (first * first_weight + second * second_weight) / total
    return require_confidence_in_bounds(combined)


def decay_by_age(confidence: float, age_ms: int, half_life_ms: int) -> float:
    """Decay a confidence exponentially by the age of its observation.

    Halves every ``half_life_ms``: ``decay_by_age(0.8, h, h) == 0.4``.

    Args:
        confidence: Confidence at observation time.
        age_ms: Milliseconds elapsed since the observation.
        half_life_ms: Half-life of trust in this kind of observation.

    Returns:
        Decayed confidence.

    Raises:
        ConfidenceOutOfBoundsError: If the confidence is out of range,
            the age is negative, or the half-life is not positive.
    """
    require_confidence_in_bounds(confidence)
    require(age_ms >= 0, ConfidenceOutOfBoundsError, age_ms=repr(age_ms))
    require(half_life_ms > 0, ConfidenceOutOfBoundsError, half_life_ms=repr(half_life_ms))
    return require_confidence_in_bounds(confidence * math.pow(0.5, age_ms / half_life_ms))


__all__ = [
    "CONFIDENCE_MAX",
    "CONFIDENCE_MIN",
    "combine_independent",
    "combine_weighted",
    "decay_by_age",
    "require_confidence_in_bounds",
]
