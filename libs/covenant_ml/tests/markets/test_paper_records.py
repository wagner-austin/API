"""Exact liquidity decoding, outcome-free feature contracts and window refusals."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONValue, require_dict

from covenant_ml.markets.forecast_model import fit_logit, predict_logit
from covenant_ml.markets.forecast_records import decode_forecast_features, encode_forecast_features
from covenant_ml.markets.paper_records import (
    BookTop,
    decode_book_top,
    decode_native_book,
    decode_paper_plan,
    encode_book_top,
    encode_paper_plan,
)

from ._forecast_fixtures import comparison_rows
from ._paper_fixtures import native_book, paper_plan


def test_outcome_free_inputs_match_settled_prediction_exactly() -> None:
    rows = comparison_rows()
    inputs = tuple(decode_forecast_features(encode_forecast_features(row)) for row in rows)
    for weather in (False, True):
        model = fit_logit(rows, weather)
        assert predict_logit(model, inputs) == predict_logit(model, rows)
    assert set(inputs[0]) == {"series", "bid", "ask", "weather_p"}


def test_plan_and_book_roundtrips() -> None:
    plan = paper_plan()
    assert decode_paper_plan(encode_paper_plan(plan)) == plan
    book = decode_native_book(native_book())
    expected = BookTop(bid=4000, ask=6000, bid_size=750, ask_size=900)
    assert book == expected
    assert decode_book_top(encode_book_top(expected)) == book


@pytest.mark.parametrize(
    "key,value",
    [
        ("series", "OTHER"),
        ("forecast_hour", 1),
        ("model_digest", "x"),
        ("protocol_digest", "G" * 64),
        ("window_start", "2026-01-01T00:00:00Z"),
        ("window_end", "2100-01-01T00:00:00Z"),
    ],
)
def test_invalid_plan(key: str, value: JSONValue) -> None:
    document = encode_paper_plan(paper_plan())
    document[key] = value
    with pytest.raises(ValueError, match="PAPER-001"):
        decode_paper_plan(document)


def test_window_length_limit() -> None:
    document: JSONObject = {
        "name": "long",
        "series": "KXHIGHTBOS",
        "event_day": "2026-10-13",
        "forecast_hour": 0,
        "window_start": "2026-10-12T12:00:00Z",
        "window_end": "2026-10-12T17:00:00Z",
        "model_digest": "a" * 64,
        "protocol_digest": "b" * 64,
    }
    with pytest.raises(ValueError, match="PAPER-001"):
        decode_paper_plan(document)


@pytest.mark.parametrize(
    "levels",
    [
        [["0.1"]],
        [["1.0001", "1"]],
        [["0.1", "0"]],
        [["0.1", "92233720368547758.08"]],
        [["0.1", "1"], ["0.10", "2"]],
    ],
)
def test_invalid_levels(levels: JSONValue) -> None:
    document = native_book()
    require_dict(document, "orderbook_fp")["yes_dollars"] = levels
    with pytest.raises(ValueError, match="PAPER-003"):
        decode_native_book(document)


@pytest.mark.parametrize("side", ["yes_dollars", "no_dollars"])
def test_missing_side_is_explicit_exclusion(side: str) -> None:
    document = native_book()
    require_dict(document, "orderbook_fp")[side] = []
    assert decode_native_book(document) is None
    del require_dict(document, "orderbook_fp")[side]
    with pytest.raises(TypeError):
        decode_native_book(document)


def test_crossed_book_rejected() -> None:
    document = native_book()
    require_dict(document, "orderbook_fp")["yes_dollars"] = [["0.7000", "1.00"]]
    with pytest.raises(ValueError, match="PAPER-002"):
        decode_native_book(document)
