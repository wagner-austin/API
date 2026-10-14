"""Paper outcome codecs reject invalid costs and inconsistent reported arithmetic."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONValue

from covenant_ml.markets.equity_outcome_records import (
    EquityWindowResult,
    decode_equity_window_result,
    encode_equity_window_result,
)


def document() -> JSONObject:
    """Provide explicit gross, cost, net and benchmark-relative arithmetic."""
    return {
        "selected_return": 0.1,
        "benchmark_return": 0.02,
        "round_trip_cost": 0.002,
        "net_selected_return": 0.1 - 0.002,
        "net_excess_return": 0.1 - 0.002 - 0.02,
    }


def test_arithmetic_roundtrip() -> None:
    expected: EquityWindowResult = {
        "selected_return": 0.1,
        "benchmark_return": 0.02,
        "round_trip_cost": 0.002,
        "net_selected_return": 0.1 - 0.002,
        "net_excess_return": 0.1 - 0.002 - 0.02,
    }
    assert decode_equity_window_result(document()) == expected
    assert decode_equity_window_result(encode_equity_window_result(expected)) == expected


@pytest.mark.parametrize(
    "key,value",
    [
        ("selected_return", -1.1),
        ("benchmark_return", -1.1),
        ("round_trip_cost", -0.001),
        ("round_trip_cost", 1.001),
        ("net_selected_return", 0.1),
        ("net_excess_return", 0.01),
        ("selected_return", float("nan")),
        ("benchmark_return", float("inf")),
    ],
)
def test_invalid_reported_arithmetic(key: str, value: float) -> None:
    malformed = document()
    malformed[key] = value
    with pytest.raises(ValueError, match="EQUITY-013"):
        decode_equity_window_result(malformed)


@pytest.mark.parametrize("value", [None, True, "0.1"])
def test_wrong_outcome_types(value: JSONValue) -> None:
    malformed = document()
    malformed["selected_return"] = value
    with pytest.raises(TypeError, match="selected_return"):
        decode_equity_window_result(malformed)
