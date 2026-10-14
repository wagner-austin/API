"""Frozen empirical inflation probabilities fitted only to earlier releases."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from math import isfinite

from .inflation_records import InflationMonth, forecast_before
from .validation import require_day


def rounded_inflation(value: float, error: float = 0.0) -> Decimal:
    """Represent the reported one-decimal inflation settlement boundary.

    Args:
        value: Finite inflation in percentage points.
        error: Finite forecast error added using decimal arithmetic before rounding.

    Returns:
        Decimal rounded to one place, with ties away from zero.

    Raises:
        ValueError: INFLATION-009 for a nonfinite value.
    """
    if not isfinite(value) or not isfinite(error):
        raise ValueError("[INFLATION-009] rounding requires finite inflation")
    return (Decimal(str(value)) + Decimal(str(error))).quantize(
        Decimal("0.1"), rounding=ROUND_HALF_UP
    )


def earlier_errors(
    history: tuple[InflationMonth, ...], target_month: str, cutoff_day: str
) -> tuple[float, ...]:
    """Collect the last sixty available errors without using later releases.

    Args:
        history: Validated chronological monthly observations.
        target_month: Canonical target month whose prediction is being made.
        cutoff_day: Midnight decision date.

    Returns:
        At most sixty earlier first-release minus prior-forecast errors.

    Raises:
        ValueError: Invalid calendar identities propagate from shared validators.
    """
    require_day({"day": target_month + "-01"}, "day")
    require_day({"day": cutoff_day}, "day")
    errors: list[float] = []
    for month in history:
        actual = month["actual"]
        if month["month"] >= target_month or actual is None or actual["day"] >= cutoff_day:
            continue
        forecast = forecast_before(month, actual["day"], 4)
        if forecast is not None:
            errors.append(actual["value"] - forecast["value"])
    return tuple(errors[-60:])


def inflation_probability(forecast: float, errors: tuple[float, ...], strike: float) -> float:
    """Apply the frozen smoothed empirical distribution at a rounded strike.

    Args:
        forecast: Earlier published point forecast in percentage points.
        errors: Thirty-six to sixty earlier released forecast errors.
        strike: Finite one-decimal strictly-greater settlement threshold.

    Returns:
        Probability from rounded synthetic releases with half-count smoothing.

    Raises:
        ValueError: INFLATION-010 for sample size, nonfinite inputs or strike precision.
    """
    if (
        not 36 <= len(errors) <= 60
        or not isfinite(forecast)
        or not isfinite(strike)
        or not all(isfinite(error) for error in errors)
    ):
        raise ValueError("[INFLATION-010] require finite inputs and 36 to 60 earlier errors")
    boundary = Decimal(str(strike))
    if rounded_inflation(strike) != boundary:
        raise ValueError("[INFLATION-010] strike must have one-decimal precision")
    successes = sum(rounded_inflation(forecast, error) > boundary for error in errors)
    return (successes + 0.5) / (len(errors) + 1)
