"""Real inflation codecs preserve dates and enforce release-safe selection."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONValue

from covenant_ml.markets.inflation_history import decode_inflation_chart
from covenant_ml.markets.inflation_records import (
    InflationMonth,
    decode_inflation_month,
    decode_inflation_point,
    encode_inflation_month,
    encode_inflation_point,
    forecast_before,
)

from ._inflation_fixtures import inflation_chart


def test_roundtrip_and_cutoff_selection() -> None:
    month = decode_inflation_chart(inflation_chart())
    expected: InflationMonth = {
        "month": "2023-12",
        "forecasts": (
            {"day": "2023-12-29", "value": 2.6},
            {"day": "2024-01-02", "value": 2.65},
        ),
        "actual": {"day": "2024-01-03", "value": 2.7},
    }
    assert month == expected
    assert decode_inflation_month(encode_inflation_month(month)) == expected
    assert forecast_before(month, "2024-01-03", 1) == expected["forecasts"][1]
    assert forecast_before(month, "2024-01-02", 4) == expected["forecasts"][0]
    assert forecast_before(month, "2024-01-02", 3) is None
    assert forecast_before(month, "2023-12-29", 4) is None
    missing: JSONObject = {"month": "2023-12", "forecasts": [], "actual": None}
    assert encode_inflation_month(decode_inflation_month(missing)) == missing
    assert forecast_before(decode_inflation_month(missing), "2024-01-03", 1) is None
    negative: JSONObject = {"day": "2024-02-29", "value": -0.1}
    assert encode_inflation_point(decode_inflation_point(negative)) == negative


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_observations(value: float) -> None:
    with pytest.raises(ValueError, match="INFLATION-001"):
        decode_inflation_point({"day": "2024-01-01", "value": value})


@pytest.mark.parametrize("value", [True, "2.7", None])
def test_observation_types(value: JSONValue) -> None:
    with pytest.raises(TypeError, match="value"):
        decode_inflation_point({"day": "2024-01-01", "value": value})


@pytest.mark.parametrize("kind", ["before_month", "duplicate", "release", "early_actual"])
def test_normalized_chronology(kind: str) -> None:
    document = encode_inflation_month(decode_inflation_chart(inflation_chart()))
    if kind == "before_month":
        document["forecasts"] = [{"day": "2023-11-30", "value": 2.0}]
    elif kind == "duplicate":
        document["forecasts"] = [{"day": "2024-01-02", "value": 2.0}] * 2
    elif kind == "release":
        document["forecasts"] = [{"day": "2024-01-03", "value": 2.0}]
    else:
        document["forecasts"] = []
        document["actual"] = {"day": "2023-12-31", "value": 2.0}
    with pytest.raises(ValueError, match="INFLATION-002"):
        decode_inflation_month(document)


def test_missing_actual_and_invalid_dates() -> None:
    with pytest.raises(ValueError, match="INFLATION-003"):
        decode_inflation_month({"month": "2023-12", "forecasts": []})
    with pytest.raises(ValueError, match="Invalid isoformat"):
        decode_inflation_month({"month": "2023-1", "forecasts": [], "actual": None})
    with pytest.raises(ValueError, match="day is out of range"):
        decode_inflation_point({"day": "2023-02-29", "value": 0.0})


@pytest.mark.parametrize("cutoff,age", [("2024-01-04", 4), ("2024-01-03", 0)])
def test_invalid_cutoffs(cutoff: str, age: int) -> None:
    month = decode_inflation_chart(inflation_chart())
    with pytest.raises(ValueError, match="INFLATION-004"):
        forecast_before(month, cutoff, age)
