"""Deterministic ranking treats ties, losses and malformed cohorts explicitly."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONValue

from covenant_ml.markets.equity_signals import (
    StrengthSignal,
    decode_strength_signal,
    encode_strength_signal,
    rank_financial_strength,
)


def test_roundtrip_and_frozen_rank_rule() -> None:
    first: StrengthSignal = {"symbol": "AAA", "cik": 1, "current_ratio": 2.0, "annual_roa": -0.1}
    second: StrengthSignal = {"symbol": "BBB", "cik": 2, "current_ratio": 1.0, "annual_roa": 0.2}
    third: StrengthSignal = {"symbol": "CCC", "cik": 3, "current_ratio": 2.0, "annual_roa": 0.2}
    assert decode_strength_signal(encode_strength_signal(first)) == first
    expected = ((third, 1, 1), (first, 1, 0), (second, 0, 1))
    assert rank_financial_strength((first, second, third)) == expected
    assert rank_financial_strength((third, second, first)) == expected
    assert rank_financial_strength((first,)) == ((first, 0, 0),)


@pytest.mark.parametrize(
    "field,value",
    [
        ("current_ratio", -1.0),
        ("current_ratio", float("nan")),
        ("current_ratio", float("inf")),
        ("annual_roa", float("-inf")),
    ],
)
def test_invalid_ratios(field: str, value: float) -> None:
    document: JSONObject = {"symbol": "AAA", "cik": 1, "current_ratio": 1.0, "annual_roa": 0.0}
    document[field] = value
    with pytest.raises(ValueError, match="EQUITY-011"):
        decode_strength_signal(document)


@pytest.mark.parametrize(
    "field,value", [("current_ratio", None), ("annual_roa", "1.2"), ("cik", True)]
)
def test_signal_wrong_types(field: str, value: JSONValue) -> None:
    document: JSONObject = {"symbol": "AAA", "cik": 1, "current_ratio": 1.0, "annual_roa": 0.0}
    document[field] = value
    with pytest.raises(TypeError, match=field):
        decode_strength_signal(document)


def test_empty_or_duplicate_cohort() -> None:
    first: StrengthSignal = {"symbol": "AAA", "cik": 1, "current_ratio": 0.0, "annual_roa": 0.0}
    duplicate_ticker: StrengthSignal = {
        "symbol": "AAA",
        "cik": 2,
        "current_ratio": 1.0,
        "annual_roa": 0.0,
    }
    duplicate_issuer: StrengthSignal = {
        "symbol": "BBB",
        "cik": 1,
        "current_ratio": 1.0,
        "annual_roa": 0.0,
    }
    for cohort in ((), (first, duplicate_ticker), (first, duplicate_issuer)):
        with pytest.raises(ValueError, match="EQUITY-012"):
            rank_financial_strength(cohort)
