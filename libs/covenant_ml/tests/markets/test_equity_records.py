"""Daily price codecs retain semantics and refuse unusable observations."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONValue

from covenant_ml.markets.equity_records import (
    EquityPrice,
    decode_equity_price,
    encode_equity_price,
)


def price_document() -> JSONObject:
    """Provide one actual archived Apple close for codec tests."""
    return {
        "symbol": "AAPL",
        "currency": "USD",
        "session_start_s": 1256736600,
        "split_adjusted_close": 6.871428966522217,
        "distribution_adjusted_close": 5.754613399505615,
    }


def test_price_roundtrip() -> None:
    expected: EquityPrice = {
        "symbol": "AAPL",
        "currency": "USD",
        "session_start_s": 1256736600,
        "split_adjusted_close": 6.871428966522217,
        "distribution_adjusted_close": 5.754613399505615,
    }
    assert decode_equity_price(price_document()) == expected
    assert decode_equity_price(encode_equity_price(expected)) == expected


@pytest.mark.parametrize("key", ["split_adjusted_close", "distribution_adjusted_close"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_unusable_prices(key: str, value: float) -> None:
    document = price_document()
    document[key] = value
    with pytest.raises(ValueError, match="EQUITY-001"):
        decode_equity_price(document)


@pytest.mark.parametrize(
    "key,value",
    [
        ("split_adjusted_close", None),
        ("distribution_adjusted_close", True),
        ("session_start_s", 1.5),
        ("currency", 1),
        ("symbol", False),
    ],
)
def test_malformed_price_fields(key: str, value: JSONValue) -> None:
    document = price_document()
    document[key] = value
    with pytest.raises(TypeError, match=key):
        decode_equity_price(document)


def test_invalid_identity_timestamp_and_absent_price() -> None:
    for key, value, code in (("symbol", " ", "MARKET-003"), ("session_start_s", -1, "MARKET-001")):
        document = price_document()
        document[key] = value
        with pytest.raises(ValueError, match=code):
            decode_equity_price(document)
    document = price_document()
    del document["distribution_adjusted_close"]
    with pytest.raises(TypeError, match="distribution_adjusted_close"):
        decode_equity_price(document)
