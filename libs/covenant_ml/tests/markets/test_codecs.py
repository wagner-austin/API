"""Saved market evidence roundtrips and explicit refusals."""

from __future__ import annotations

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_list,
)

from covenant_ml.markets.codecs import decode_history, decode_rule, encode_history, encode_rule
from covenant_ml.markets.replay import replay_maker
from covenant_ml.markets.replay_codecs import decode_result, encode_result
from covenant_ml.markets.validation import parse_fixed, parse_timestamp

from ._fixtures import history, history_document, rule


def test_evidence_rule_and_actual_replay_result_roundtrip_through_json() -> None:
    item = history()
    assert (
        decode_history(narrow_json_to_dict(load_json_str(dump_json_str(encode_history(item)))))
        == item
    )
    assert decode_rule(encode_rule(rule())) == rule()
    for record in (
        replay_maker(item, rule()),
        replay_maker(history(trades=[]), rule()),
        replay_maker(history(quotes=[]), rule()),
    ):
        assert (
            decode_result(narrow_json_to_dict(load_json_str(dump_json_str(encode_result(record)))))
            == record
        )


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("ticker", " ", "MARKET-003"),
        ("close_ms", 0, "MARKET-007"),
        ("event_day", "20260101", "MARKET-004"),
        ("quotes", [{"timestamp_ms": 1, "ask": 1000}], "MARKET-002"),
        ("quotes", [{"timestamp_ms": 1, "ask": 10001, "bid": 0}], "MARKET-001"),
    ],
)
def test_invalid_history_fields_are_refused(key: str, value: JSONValue, match: str) -> None:
    document = history_document()
    document[key] = value
    with pytest.raises(ValueError, match=match):
        decode_history(document)


@pytest.mark.parametrize("field", ["quotes", "trades"])
def test_duplicate_evidence_is_refused(field: str) -> None:
    document = encode_history(history())
    value = require_list(document, field)
    document[field] = value + value
    with pytest.raises(ValueError, match="MARKET-008"):
        decode_history(document)


def test_optional_prices_keep_explicit_nulls_and_required_types() -> None:
    record = history(quotes=[{"timestamp_ms": 1, "bid": None, "ask": None}])
    assert decode_history(encode_history(record))["quotes"] == record["quotes"]
    document = history_document()
    document["decision_ms"] = True
    with pytest.raises(TypeError, match="decision_ms"):
        decode_history(document)


@pytest.mark.parametrize("changes", [{"min_ask": 2000}, {"improve": 200}, {"through": -1}])
def test_inconsistent_rule_is_refused(changes: dict[str, int]) -> None:
    with pytest.raises(ValueError, match=r"MARKET-00[19]"):
        rule(**changes)


@pytest.mark.parametrize(
    "text,precision,expected", [("0.5600", 4, 5600), ("1", 2, 100), ("1.25", 2, 125), ("12", 0, 12)]
)
def test_fixed_point_values_are_exact(text: str, precision: int, expected: int) -> None:
    assert parse_fixed(text, precision) == expected


@pytest.mark.parametrize(
    "text,precision",
    [("NaN", 4), ("inf", 4), ("-1", 4), (".1", 4), ("1.001", 2), ("1", 7), ("1", -1)],
)
def test_invalid_fixed_point_values_are_refused(text: str, precision: int) -> None:
    with pytest.raises(ValueError, match="MARKET-005"):
        parse_fixed(text, precision)


def test_exchange_timestamps_preserve_milliseconds_and_require_timezone() -> None:
    assert parse_timestamp("1970-01-01T00:00:01.123Z") == 1123
    assert parse_timestamp("1970-01-01T01:00:01.123+01:00") == 1123
    with pytest.raises(ValueError, match="MARKET-006"):
        parse_timestamp("2026-01-01T00:00:00")


@pytest.mark.parametrize(
    "changes",
    [
        {"offer": None},
        {"contracts": 0},
        {"matching_quantity": 99},
        {"capital": 1},
        {"pnl": 1},
    ],
)
def test_impossible_filled_accounting_is_refused(changes: JSONObject) -> None:
    document = encode_result(replay_maker(history(), rule()))
    document.update(changes)
    with pytest.raises(ValueError, match="MARKET-012"):
        decode_result(document)


@pytest.mark.parametrize("field", ["contracts", "fee", "pnl", "capital"])
def test_unfilled_results_cannot_claim_money(field: str) -> None:
    document = encode_result(replay_maker(history(trades=[]), rule()))
    document[field] = 1
    with pytest.raises(ValueError, match="MARKET-012"):
        decode_result(document)


__all__: list[str] = []
