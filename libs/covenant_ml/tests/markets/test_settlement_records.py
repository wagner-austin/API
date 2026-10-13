"""Finalized versus provisional outcomes, exact scalar payouts and timing refusals."""

from __future__ import annotations

from datetime import date

import pytest
from platform_core.json_utils import JSONObject, JSONValue

from covenant_ml.markets.settlement_records import (
    PENDING_STATUSES,
    SavedPrediction,
    SettlementState,
    decode_native_settlement,
    decode_saved_prediction,
    decode_settlement,
    encode_saved_prediction,
    encode_settlement,
)
from covenant_ml.markets.validation import parse_timestamp

from ._forecast_fixtures import native_market


def saved_prediction(yes: int = 1) -> SavedPrediction:
    native = native_market(date(2026, 4, 3), yes=yes)
    return decode_saved_prediction(
        {
            "ticker": native["ticker"],
            "series": "KXHIGHTBOS",
            "event_day": "2026-04-03",
            "recorded_ms": parse_timestamp("2026-04-02T19:00:00Z"),
            "rules_primary": native["rules_primary"],
            "probabilities": [0.5, 0.8, 0.6, 0.7],
        }
    )


def final_market(yes: int = 1) -> JSONObject:
    document = native_market(date(2026, 4, 3), yes=yes)
    document["status"] = "finalized"
    return document


def test_binary_roundtrips() -> None:
    row = saved_prediction()
    assert decode_saved_prediction(encode_saved_prediction(row)) == row
    settlement = decode_native_settlement(
        final_market(), row, parse_timestamp("2026-04-05T00:00:00Z")
    )
    assert settlement["state"] is SettlementState.BINARY
    assert settlement["payout"] == 10000
    assert settlement["settled_ms"] == parse_timestamp("2026-04-04T12:00:00Z")
    assert settlement["rules_changed"] is False
    assert decode_settlement(encode_settlement(settlement)) == settlement


@pytest.mark.parametrize("status", PENDING_STATUSES)
def test_provisional_results_never_become_labels(status: str) -> None:
    document = final_market()
    document["status"] = status
    settlement = decode_native_settlement(
        document, saved_prediction(), parse_timestamp("2026-04-05T00:00:00Z")
    )
    assert settlement["state"] is SettlementState.PENDING
    assert settlement["payout"] is None
    assert settlement["settled_ms"] is None
    assert decode_settlement(encode_settlement(settlement)) == settlement


@pytest.mark.parametrize("payout", ["0.0100", "0.0000", "1.0000"])
def test_scalar_result_remains_scalar_at_endpoints(payout: str) -> None:
    document = final_market()
    document.update(
        result="scalar", settlement_value_dollars=payout, rules_primary="Changed source"
    )
    settlement = decode_native_settlement(
        document, saved_prediction(), parse_timestamp("2026-04-05T00:00:00Z")
    )
    assert settlement["state"] is SettlementState.SCALAR
    assert settlement["payout"] == int(float(payout) * 10000)
    assert settlement["rules_changed"] is True


@pytest.mark.parametrize(
    "key,value",
    [
        ("series", "OTHER"),
        ("ticker", "OTHER-T60"),
        ("probabilities", [0.5]),
        ("probabilities", [0.5, 0.5, 0.5, -0.1]),
        ("probabilities", [0.5, 0.5, 0.5, float("nan")]),
    ],
)
def test_invalid_saved_prediction(key: str, value: JSONValue) -> None:
    document = encode_saved_prediction(saved_prediction())
    document[key] = value
    with pytest.raises(ValueError, match="SETTLEMENT-001"):
        decode_saved_prediction(document)


@pytest.mark.parametrize(
    "key,value",
    [
        ("native_status", "active"),
        ("payout", None),
        ("settled_ms", None),
        ("settled_ms", 2**60),
        ("payout", 5000),
        ("state", "pending"),
    ],
)
def test_inconsistent_normalized_settlement(key: str, value: JSONValue) -> None:
    settlement = decode_native_settlement(
        final_market(), saved_prediction(), parse_timestamp("2026-04-05T00:00:00Z")
    )
    document = encode_settlement(settlement)
    document[key] = value
    with pytest.raises(ValueError, match="SETTLEMENT-002"):
        decode_settlement(document)


@pytest.mark.parametrize("change", ["identity", "receipt", "status", "date", "late"])
def test_native_identity_and_lookahead_fail(change: str) -> None:
    document, row = final_market(), saved_prediction()
    received = parse_timestamp("2026-04-05T00:00:00Z")
    if change == "identity":
        document["ticker"] = "OTHER"
    elif change == "receipt":
        received = 0
    elif change == "status":
        document["status"] = "new_unknown_state"
    else:
        values = encode_saved_prediction(row)
        if change == "date":
            values["event_day"] = "2026-04-04"
        else:
            values["recorded_ms"] = received
        row = decode_saved_prediction(values)
    with pytest.raises(ValueError, match="SETTLEMENT-003"):
        decode_native_settlement(document, row, received)


__all__ = ["final_market", "saved_prediction"]
