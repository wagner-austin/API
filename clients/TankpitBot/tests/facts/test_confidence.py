"""Tests for confidence arithmetic and its bounds contract."""

from __future__ import annotations

import pytest

from tankpit_bot.contracts.base import ConfidenceOutOfBoundsError
from tankpit_bot.facts.confidence import (
    combine_independent,
    combine_weighted,
    decay_by_age,
    require_confidence_in_bounds,
)


def test_require_confidence_accepts_bounds_inclusive() -> None:
    """0.0 and 1.0 are both valid confidences."""
    assert require_confidence_in_bounds(0.0) == 0.0
    assert require_confidence_in_bounds(1.0) == 1.0


@pytest.mark.parametrize("value", [-0.001, 1.001, 2.0, -5.0])
def test_require_confidence_rejects_out_of_range(value: float) -> None:
    """Out-of-range values raise ConfidenceOutOfBoundsError."""
    with pytest.raises(ConfidenceOutOfBoundsError) as exc:
        require_confidence_in_bounds(value)
    assert exc.value.details == {"value": repr(value)}


def test_combine_independent_is_noisy_or() -> None:
    """Two independent confirmations compose as noisy-OR."""
    assert combine_independent(0.8, 0.5) == pytest.approx(0.9)
    assert combine_independent(0.0, 0.3) == pytest.approx(0.3)
    assert combine_independent(1.0, 0.2) == pytest.approx(1.0)


def test_combine_independent_rejects_bad_input() -> None:
    """An out-of-range input raises before combining."""
    with pytest.raises(ConfidenceOutOfBoundsError):
        combine_independent(1.5, 0.5)
    with pytest.raises(ConfidenceOutOfBoundsError):
        combine_independent(0.5, -0.1)


def test_combine_weighted_averages_by_weight() -> None:
    """Weighted combine is the weighted average of the inputs."""
    assert combine_weighted(1.0, 3.0, 0.0, 1.0) == pytest.approx(0.75)
    assert combine_weighted(0.4, 1.0, 0.8, 1.0) == pytest.approx(0.6)


def test_combine_weighted_rejects_bad_confidence() -> None:
    """An out-of-range confidence raises before weighting."""
    with pytest.raises(ConfidenceOutOfBoundsError):
        combine_weighted(1.2, 1.0, 0.5, 1.0)
    with pytest.raises(ConfidenceOutOfBoundsError):
        combine_weighted(0.5, 1.0, -0.2, 1.0)


def test_combine_weighted_rejects_negative_weight() -> None:
    """A negative weight raises with the weights in the details."""
    with pytest.raises(ConfidenceOutOfBoundsError) as exc:
        combine_weighted(0.5, -1.0, 0.5, 1.0)
    assert exc.value.details == {"first_weight": "-1.0", "second_weight": "1.0"}


def test_combine_weighted_rejects_zero_total_weight() -> None:
    """Both weights zero raises: no average is defined."""
    with pytest.raises(ConfidenceOutOfBoundsError) as exc:
        combine_weighted(0.5, 0.0, 0.5, 0.0)
    assert exc.value.details == {"total_weight": "0.0"}


def test_decay_by_age_halves_at_half_life() -> None:
    """Confidence halves exactly at one half-life."""
    assert decay_by_age(0.8, 1000, 1000) == pytest.approx(0.4)
    assert decay_by_age(0.8, 0, 1000) == pytest.approx(0.8)
    assert decay_by_age(0.8, 2000, 1000) == pytest.approx(0.2)


def test_decay_by_age_rejects_negative_age() -> None:
    """A negative age raises with the age in the details."""
    with pytest.raises(ConfidenceOutOfBoundsError) as exc:
        decay_by_age(0.5, -1, 1000)
    assert exc.value.details == {"age_ms": "-1"}


def test_decay_by_age_rejects_non_positive_half_life() -> None:
    """A zero half-life raises with the half-life in the details."""
    with pytest.raises(ConfidenceOutOfBoundsError) as exc:
        decay_by_age(0.5, 100, 0)
    assert exc.value.details == {"half_life_ms": "0"}


def test_decay_by_age_rejects_bad_confidence() -> None:
    """An out-of-range confidence raises before decaying."""
    with pytest.raises(ConfidenceOutOfBoundsError):
        decay_by_age(1.5, 100, 1000)
