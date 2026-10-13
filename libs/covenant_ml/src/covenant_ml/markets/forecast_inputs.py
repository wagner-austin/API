"""Native strike semantics and explicit-run hourly forecast validation."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

from platform_core.json_utils import (
    JSONObject,
    require_dict,
    require_float,
    require_int,
    require_list,
    require_str,
)

from .validation import parse_timestamp


def strike_bounds(document: JSONObject) -> tuple[float | None, float | None] | None:
    """Convert supported integer-Fahrenheit strikes to continuous rounding bounds.

    Args:
        document: Native market metadata.

    Returns:
        Lower/upper bounds, or explicit unsupported metadata exclusion.

    Raises:
        ValueError: FORECAST-003 for invalid supported strike boundaries.
        TypeError: A declared numeric boundary has an incorrect type.
    """
    kind = document.get("strike_type")
    if kind not in ("between", "greater", "less"):
        return None
    lower: float | None = None
    upper: float | None = None
    if kind in ("between", "greater"):
        lower = require_float(document, "floor_strike")
        if not math.isfinite(lower) or not lower.is_integer():
            raise ValueError("[FORECAST-003] require finite integer Fahrenheit strikes")
        lower += -0.5 if kind == "between" else 0.5
    if kind in ("between", "less"):
        upper = require_float(document, "cap_strike")
        if not math.isfinite(upper) or not upper.is_integer():
            raise ValueError("[FORECAST-003] require finite integer Fahrenheit strikes")
        upper += 0.5 if kind == "between" else -0.5
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError("[FORECAST-003] strike interval must have positive width")
    return lower, upper


def forecast_maximum(document: JSONObject, day: date) -> float:
    """Read exactly 24 UTC hours spanning the target Eastern standard-time day.

    Args:
        document: Native explicit-run Open-Meteo response.
        day: Target event date, whose window begins at 05:00 UTC.

    Returns:
        Maximum hourly forecast in Fahrenheit.

    Raises:
        ValueError: FORECAST-004 for units, missing/duplicate hours or invalid values.
        TypeError: Incorrect native field types; null is never imputed.
    """
    units = require_dict(document, "hourly_units")
    if (
        require_int(document, "utc_offset_seconds") != 0
        or require_str(units, "temperature_2m") != "°C"
    ):
        raise ValueError("[FORECAST-004] require UTC timestamps and Celsius temperatures")
    hourly = require_dict(document, "hourly")
    times = require_list(hourly, "time")
    temperatures = require_list(hourly, "temperature_2m")
    if len(times) != len(temperatures):
        raise ValueError("[FORECAST-004] forecast time and temperature lengths differ")
    start = datetime(day.year, day.month, day.day, 5, tzinfo=UTC)
    expected = {int((start + timedelta(hours=i)).timestamp() * 1000) for i in range(24)}
    selected: dict[int, float] = {}
    seen: set[int] = set()
    for raw_time, raw_value in zip(times, temperatures, strict=True):
        timestamp = parse_timestamp(require_str({"time": raw_time}, "time") + "Z")
        if timestamp in seen:
            raise ValueError("[FORECAST-004] duplicate forecast timestamp")
        seen.add(timestamp)
        if timestamp in expected:
            value = require_float({"temperature": raw_value}, "temperature")
            if not math.isfinite(value):
                raise ValueError("[FORECAST-004] nonfinite target-window temperature")
            selected[timestamp] = value
    if set(selected) != expected:
        raise ValueError("[FORECAST-004] target standard-time day needs all 24 hours")
    return max(selected.values()) * 1.8 + 32


def bracket_probability(
    mean: float, bounds: tuple[float | None, float | None], sigma: float
) -> float:
    """Compute a normal forecast's bracket probability with explicit uncertainty.

    Args:
        mean: Forecast maximum in Fahrenheit.
        bounds: Validated continuous lower and upper boundaries.
        sigma: Positive finite standard deviation in Fahrenheit.

    Returns:
        Probability in the closed unit interval, before logit clipping.

    Raises:
        ValueError: CALIBRATION-001 for nonfinite mean or invalid uncertainty.
    """
    if not math.isfinite(mean) or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("[CALIBRATION-001] require finite mean and positive finite uncertainty")
    lower, upper = bounds
    low = 0.0 if lower is None else 0.5 * (1 + math.erf((lower - mean) / (sigma * math.sqrt(2))))
    high = 1.0 if upper is None else 0.5 * (1 + math.erf((upper - mean) / (sigma * math.sqrt(2))))
    return high - low


__all__ = ["bracket_probability", "forecast_maximum", "strike_bounds"]
