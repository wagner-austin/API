"""Forecast contract refusals, exact hourly windows and native strike semantics."""

from __future__ import annotations

from datetime import date

import pytest
from platform_core.json_utils import JSONObject, JSONValue, require_dict, require_list

from covenant_ml.markets.forecast_collection import select_quote
from covenant_ml.markets.forecast_inputs import bracket_probability, forecast_maximum, strike_bounds
from covenant_ml.markets.forecast_records import (
    decode_comparison_plan,
    decode_forecast_row,
    encode_comparison_plan,
    encode_forecast_row,
)
from covenant_ml.markets.types import Quote

from ._forecast_fixtures import comparison_plan, forecast_document, forecast_row, native_market


def test_record_roundtrips() -> None:
    plan, row = comparison_plan(), forecast_row()
    assert decode_comparison_plan(encode_comparison_plan(plan)) == plan
    assert decode_forecast_row(encode_forecast_row(row)) == row


@pytest.mark.parametrize(
    "key,value",
    [
        ("train_end", "2026-04-19"),
        ("start", "2026-04-09"),
        ("protocol_digest", "a"),
        ("protocol_digest", "Z" * 64),
        ("excluded_days", ["2026-04-03", "2026-04-03"]),
        ("excluded_days", ["2026-04-02"]),
        ("forecast_hour", 1),
    ],
)
def test_invalid_plan(key: str, value: JSONValue) -> None:
    document = encode_comparison_plan(comparison_plan())
    document[key] = value
    with pytest.raises(ValueError, match="FORECAST-001"):
        decode_comparison_plan(document)


@pytest.mark.parametrize(
    "key,value",
    [
        ("series", "OTHER"),
        ("ticker", "OTHER-26APR03"),
        ("quote_ms", 0),
        ("forecast_init_ms", 2**60),
        ("settled_ms", 0),
        ("bid", 9000),
        ("forecast_f", float("inf")),
        ("weather_p", -0.1),
        ("weather_p", float("nan")),
    ],
)
def test_invalid_row(key: str, value: JSONValue) -> None:
    document = encode_forecast_row(forecast_row())
    document[key] = value
    with pytest.raises(ValueError, match="FORECAST-002"):
        decode_forecast_row(document)


def test_strikes_and_probability_mass_partition() -> None:
    middle: JSONObject = {"strike_type": "between", "floor_strike": 60, "cap_strike": 61}
    below: JSONObject = {"strike_type": "less", "cap_strike": 60}
    above: JSONObject = {"strike_type": "greater", "floor_strike": 61}
    bounds = [strike_bounds(document) for document in (middle, below, above)]
    assert bounds == [(59.5, 61.5), (None, 59.5), (61.5, None)]
    mass = [bracket_probability(60.5, bound) for bound in bounds if bound is not None]
    assert sum(mass) == pytest.approx(1.0, abs=1e-15)
    assert mass[1] == pytest.approx(mass[2], abs=1e-15)
    assert strike_bounds({"strike_type": None}) is None
    assert strike_bounds({}) is None


@pytest.mark.parametrize(
    "key,value",
    [
        ("floor_strike", 60.25),
        ("cap_strike", 61.25),
        ("floor_strike", float("inf")),
        ("cap_strike", float("nan")),
        ("cap_strike", 50),
    ],
)
def test_invalid_strikes(key: str, value: JSONValue) -> None:
    document = native_market(date(2026, 4, 3))
    document[key] = value
    with pytest.raises(ValueError, match="FORECAST-003"):
        strike_bounds(document)


def test_forecast_uses_all_target_hours_and_correct_units() -> None:
    document = forecast_document(date(2026, 4, 3))
    assert forecast_maximum(document, date(2026, 4, 3)) == pytest.approx(62.87)
    values = require_list(require_dict(document, "hourly"), "temperature_2m")
    values[0] = 1000.0
    assert forecast_maximum(document, date(2026, 4, 3)) == pytest.approx(62.87)


@pytest.mark.parametrize(
    "mutation", ["offset", "unit", "length", "duplicate", "missing", "nonfinite", "null"]
)
def test_forecast_refuses_bad_evidence(mutation: str) -> None:
    document = forecast_document(date(2026, 4, 3))
    hourly = require_dict(document, "hourly")
    times = require_list(hourly, "time")
    values = require_list(hourly, "temperature_2m")
    if mutation == "offset":
        document["utc_offset_seconds"] = 3600
    elif mutation == "unit":
        require_dict(document, "hourly_units")["temperature_2m"] = "°F"
    elif mutation == "length":
        values.pop()
    elif mutation == "duplicate":
        times[30] = times[29]
    elif mutation == "missing":
        times.pop(30)
        values.pop(30)
    elif mutation == "nonfinite":
        values[30] = float("inf")
    else:
        values[30] = None
    if mutation == "null":
        with pytest.raises(TypeError):
            forecast_maximum(document, date(2026, 4, 3))
    else:
        with pytest.raises(ValueError, match="FORECAST-004"):
            forecast_maximum(document, date(2026, 4, 3))


@pytest.mark.parametrize(
    "quotes,reason",
    [
        ((), "no_quote"),
        ((Quote(timestamp_ms=10, bid=1, ask=2),), "stale_quote"),
        ((Quote(timestamp_ms=5000000, bid=None, ask=2),), "missing_quote_side"),
        ((Quote(timestamp_ms=5000000, bid=1, ask=None),), "missing_quote_side"),
        ((Quote(timestamp_ms=5000000, bid=3, ask=2),), "crossed_quote"),
        ((Quote(timestamp_ms=5000001, bid=1, ask=2),), "no_quote"),
    ],
)
def test_quote_exclusions(quotes: tuple[Quote, ...], reason: str) -> None:
    assert select_quote(quotes, 5000000) == (None, reason)
