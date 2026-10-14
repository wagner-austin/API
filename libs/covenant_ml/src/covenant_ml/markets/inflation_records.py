"""Immutable dated inflation observations and release-aware forecast selection."""

from __future__ import annotations

from datetime import date
from math import isfinite

from platform_core.json_utils import (
    JSONObject,
    narrow_json_to_dict,
    require_float,
    require_list,
    require_str,
)
from typing_extensions import ReadOnly, TypedDict

from .validation import require_day


class InflationPoint(TypedDict):
    """Calendar observation date and inflation in percentage points."""

    day: ReadOnly[str]
    value: ReadOnly[float]


class InflationMonth(TypedDict):
    """Target month with ordered forecasts and an explicitly optional release."""

    month: ReadOnly[str]
    forecasts: ReadOnly[tuple[InflationPoint, ...]]
    actual: ReadOnly[InflationPoint | None]


def encode_inflation_point(point: InflationPoint) -> JSONObject:
    """Encode a dated observation.

    Args:
        point: Validated inflation observation.

    Returns:
        JSON date and percentage-point value.
    """
    return {"day": point["day"], "value": point["value"]}


def decode_inflation_point(document: JSONObject) -> InflationPoint:
    """Require a canonical date and finite numeric observation.

    Args:
        document: Untrusted observation object.

    Returns:
        Immutable dated percentage-point value.

    Raises:
        ValueError: INFLATION-001 for a nonfinite value, or invalid calendar date.
        TypeError: Missing or incorrectly typed fields.
    """
    day = require_day(document, "day")
    value = require_float(document, "value")
    if not isfinite(value):
        raise ValueError("[INFLATION-001] inflation observation must be finite")
    return InflationPoint(day=day, value=value)


def encode_inflation_month(month: InflationMonth) -> JSONObject:
    """Encode a complete target month without replacing missing observations.

    Args:
        month: Validated target and observations.

    Returns:
        JSON record with an explicit nullable actual.
    """
    actual = month["actual"]
    return {
        "month": month["month"],
        "forecasts": [encode_inflation_point(point) for point in month["forecasts"]],
        "actual": None if actual is None else encode_inflation_point(actual),
    }


def decode_inflation_month(document: JSONObject) -> InflationMonth:
    """Validate target identity, ordered forecasts and release chronology.

    Args:
        document: Untrusted normalized month.

    Returns:
        Immutable month with forecasts strictly earlier than its release.

    Raises:
        ValueError: INFLATION-002 for target or observation chronology;
            INFLATION-003 for an absent actual field.
        TypeError: Missing or incorrectly typed fields.
    """
    month = require_str(document, "month")
    first = require_day({"day": month + "-01"}, "day")
    forecasts = tuple(
        decode_inflation_point(narrow_json_to_dict(value))
        for value in require_list(document, "forecasts")
    )
    if "actual" not in document:
        raise ValueError("[INFLATION-003] actual must be explicitly present or null")
    raw_actual = document["actual"]
    actual = None if raw_actual is None else decode_inflation_point(narrow_json_to_dict(raw_actual))
    previous = ""
    for point in forecasts:
        day = point["day"]
        if day < first or day <= previous or (actual is not None and day >= actual["day"]):
            raise ValueError("[INFLATION-002] forecasts must increase and precede release")
        previous = day
    if actual is not None and actual["day"][:7] <= month:
        raise ValueError("[INFLATION-002] actual release must follow its target month")
    return InflationMonth(month=month, forecasts=forecasts, actual=actual)


def forecast_before(
    month: InflationMonth, cutoff_day: str, max_age_days: int
) -> InflationPoint | None:
    """Select a forecast strictly before a midnight cutoff with bounded age.

    Args:
        month: Validated monthly history.
        cutoff_day: Canonical calendar day whose midnight is the cutoff.
        max_age_days: Positive maximum calendar age, including weekends.

    Returns:
        Last earlier forecast, or None when no sufficiently recent one exists.

    Raises:
        ValueError: INFLATION-004 for invalid age or a cutoff after the release
            day; invalid calendar dates propagate from the shared validator.
    """
    cutoff = date.fromisoformat(require_day({"day": cutoff_day}, "day"))
    actual = month["actual"]
    if max_age_days < 1 or (actual is not None and cutoff_day > actual["day"]):
        raise ValueError("[INFLATION-004] cutoff must precede release with positive age")
    earlier = tuple(point for point in month["forecasts"] if point["day"] < cutoff_day)
    if not earlier:
        return None
    last = earlier[-1]
    if (cutoff - date.fromisoformat(last["day"])).days > max_age_days:
        return None
    return last
