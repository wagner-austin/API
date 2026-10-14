"""Strict decoding of Cleveland Fed yearly headline inflation chart histories."""

from __future__ import annotations

import re
from datetime import date
from itertools import pairwise

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    narrow_json_to_list,
    require_dict,
    require_list,
    require_str,
)

from .inflation_records import InflationMonth, decode_inflation_month


def _chart_dates(document: JSONObject, target: str) -> tuple[tuple[str, str], ...]:
    """Resolve ordered month/day labels against a target-year anchor.

    Args:
        document: Native chart object.
        target: Canonical target month.

    Returns:
        Native label and canonical calendar date pairs, excluding markers.

    Raises:
        ValueError: INFLATION-005 for category shape, date order or marker drift.
        TypeError: Missing or incorrectly typed native fields.
    """
    groups = require_list(document, "categories")
    if len(groups) != 1:
        raise ValueError("[INFLATION-005] exactly one chart date axis is required")
    entries = require_list(narrow_json_to_dict(groups[0]), "category")
    anchor = date.fromisoformat(target + "-01")
    dates: list[tuple[str, str]] = []
    for value in entries:
        entry = narrow_json_to_dict(value)
        label = require_str(entry, "label")
        if "vline" in entry:
            if (
                require_str(entry, "vline") != "true"
                or re.fullmatch(r"(?:CPI|PCE) [A-Z][a-z]{2}", label) is None
            ):
                raise ValueError("[INFLATION-005] unsupported release marker")
            continue
        if re.fullmatch(r"[0-9]{2}/[0-9]{2}", label) is None:
            raise ValueError("[INFLATION-005] chart date must be MM/DD")
        month, day = (int(part) for part in label.split("/"))
        year = anchor.year + int(month < anchor.month)
        resolved = date(year, month, day).isoformat()
        if dates and resolved <= dates[-1][1]:
            raise ValueError("[INFLATION-005] chart dates must strictly increase")
        dates.append((label, resolved))
    if not dates:
        raise ValueError("[INFLATION-005] chart requires dated observations")
    return tuple(dates)


def _series_points(
    series: JSONObject, dates: tuple[tuple[str, str], ...], name: str
) -> list[JSONValue]:
    """Validate native tooltip alignment and preserve only explicit numeric cells.

    Args:
        series: Named native observation series.
        dates: Validated native and canonical dates.
        name: Required series and tooltip identity.

    Returns:
        Normalized observation objects, without blank cells.

    Raises:
        ValueError: INFLATION-006 for row counts, tooltip or decimal spelling.
        TypeError: Missing or incorrectly typed native fields.
    """
    values = require_list(series, "data")
    if len(values) != len(dates):
        raise ValueError("[INFLATION-006] series rows must align with chart dates")
    points: list[JSONValue] = []
    for value, (label, day) in zip(values, dates, strict=True):
        point = narrow_json_to_dict(value)
        number = require_str(point, "value")
        tooltip = require_str(point, "tooltext").split("{br}")
        if len(tooltip) != 4 or tooltip[:3] != [name, label, number]:
            raise ValueError("[INFLATION-006] tooltip identity, date or value mismatch")
        if number == "":
            continue
        if re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", number) is None:
            raise ValueError("[INFLATION-006] inflation value must be a decimal string")
        points.append({"day": day, "value": float(number)})
    return points


def decode_inflation_chart(document: JSONObject) -> InflationMonth:
    """Decode the native yearly headline forecast and first-release observation.

    Args:
        document: Untrusted chart object from the yearly feed.

    Returns:
        Normalized target month with validated observations.

    Raises:
        ValueError: INFLATION-007 for chart identity, duplicate or absent series,
            or multiple actual observations; observation validation propagates.
        TypeError: Missing or incorrectly typed native fields.
    """
    chart = require_dict(document, "chart")
    caption = require_str(chart, "subcaption")
    if (
        require_str(chart, "caption") != "Inflation Nowcasting"
        or require_str(chart, "yaxisname") != "Year-over-year percent change"
        or re.fullmatch(r"[0-9]{4}-[1-9][0-9]?", caption) is None
    ):
        raise ValueError("[INFLATION-007] unsupported inflation chart identity")
    year, month_number = (int(part) for part in caption.split("-"))
    target = date(year, month_number, 1).isoformat()[:7]
    dates = _chart_dates(document, target)
    selected: dict[str, list[JSONValue]] = {}
    seen: set[str] = set()
    for value in require_list(document, "dataset"):
        series = narrow_json_to_dict(value)
        name = require_str(series, "seriesname")
        if name in seen:
            raise ValueError("[INFLATION-007] duplicate native series identity")
        seen.add(name)
        if name in ("CPI Inflation", "Actual CPI Inflation"):
            selected[name] = _series_points(series, dates, name)
    if set(selected) != {"CPI Inflation", "Actual CPI Inflation"}:
        raise ValueError("[INFLATION-007] headline forecast and actual series required")
    actuals = selected["Actual CPI Inflation"]
    if len(actuals) > 1:
        raise ValueError("[INFLATION-007] at most one first-release actual is allowed")
    return decode_inflation_month(
        {
            "month": target,
            "forecasts": selected["CPI Inflation"],
            "actual": actuals[0] if actuals else None,
        }
    )


def decode_inflation_history(value: JSONValue) -> tuple[InflationMonth, ...]:
    """Require a nonempty chronological feed with exactly one chart per month.

    Args:
        value: Parsed native JSON root.

    Returns:
        Immutable ordered monthly histories.

    Raises:
        ValueError: INFLATION-008 for empty, duplicate or reversed target months;
            chart validation failures propagate without substitution.
        TypeError: The root or native fields have incorrect types.
    """
    months = tuple(
        decode_inflation_chart(narrow_json_to_dict(item)) for item in narrow_json_to_list(value)
    )
    if not months:
        raise ValueError("[INFLATION-008] inflation history cannot be empty")
    for previous, current in pairwise(months):
        if current["month"] <= previous["month"]:
            raise ValueError("[INFLATION-008] target months must strictly increase")
    return months
