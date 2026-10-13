"""Strict manifest, acknowledgement and independent fill accounting contracts."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONValue, require_dict

from covenant_ml.markets.demo_trial_records import (
    decode_trial,
    encode_trial,
    validate_acknowledgement,
    verify_fills,
)
from covenant_ml.markets.execution_records import (
    decode_native_order,
    encode_intent,
    require_order_money,
)

from ._execution_fixtures import intent, native_order
from .test_kalshi_demo import preflight_document


def manifest() -> JSONObject:
    return {
        "schema": 1,
        "mode": "demo_single_order",
        "intent": encode_intent(intent(created=120000)),
        "initial": preflight_document(),
    }


def test_manifest_and_exact_native_precision() -> None:
    document = manifest()
    assert encode_trial(decode_trial(document)) == document
    assert require_order_money({"price": "0.980000"}, "price") == 9800
    with pytest.raises(ValueError, match="EXECUTION-010"):
        require_order_money({"price": "0.980001"}, "price")


@pytest.mark.parametrize("failure", ["mode", "contracts", "reserve", "funds", "future", "stale"])
def test_invalid_manifest(failure: str) -> None:
    document = manifest()
    row = require_dict(document, "intent")
    initial = require_dict(document, "initial")
    if failure == "mode":
        document["mode"] = "production"
    elif failure == "contracts":
        row.update(contracts=2, limit_price=4000, fee_reserve=2000)
    elif failure == "reserve":
        row["fee_reserve"] = 100
    elif failure == "funds":
        initial["balance_microdollars"] = 999999
    elif failure == "future":
        initial["started_ms"] = 120001
        initial["completed_ms"] = 120001
    else:
        row["created_ms"] = 150001
        row["expires_ms"] = 450001
    with pytest.raises(ValueError, match="DEMO-TRIAL-001"):
        decode_trial(document)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("client_order_id", "wrong"),
        ("fill_count", "1.01"),
        ("remaining_count", "0.01"),
        ("ts_ms", 301000),
    ],
)
def test_acknowledgement_does_not_infer_execution(field: str, value: JSONValue) -> None:
    row = intent()
    ack: JSONObject = {
        "client_order_id": row["client_order_id"],
        "order_id": "order",
        "fill_count": "1.00",
        "remaining_count": "0.00",
        "ts_ms": 1000,
    }
    assert validate_acknowledgement(ack, row, 400000) == "order"
    ack[field] = value
    with pytest.raises(ValueError, match="DEMO-TRIAL-002"):
        validate_acknowledgement(ack, row, 400000)


def fill() -> JSONObject:
    row = intent()
    receipt = native_order(row)
    return {
        "fill_id": "fill",
        "order_id": receipt["order_id"],
        "ticker": row["ticker"],
        "outcome_side": "yes",
        "book_side": "bid",
        "count_fp": "1.00",
        "yes_price_dollars": "0.980000",
        "fee_cost": "0.020000",
        "created_time": receipt["last_update_time"],
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("order_id", "other"),
        ("ticker", "other"),
        ("outcome_side", "no"),
        ("book_side", "ask"),
        ("count_fp", "0.00"),
        ("count_fp", "1.01"),
        ("yes_price_dollars", "0.9900"),
        ("created_time", "1970-01-01T00:00:00Z"),
        ("fee_cost", "0.019900"),
    ],
)
def test_fill_identity_and_totals(field: str, value: JSONValue) -> None:
    row = intent()
    receipt = decode_native_order(native_order(row))
    document = fill()
    verify_fills((document,), receipt, row)
    document[field] = value
    with pytest.raises(ValueError, match="DEMO-TRIAL-003"):
        verify_fills((document,), receipt, row)


def test_partial_fills_and_duplicates_are_accounted_exactly() -> None:
    row = intent()
    receipt = decode_native_order(native_order(row))
    first, second = fill(), fill()
    first.update(count_fp="0.50", fee_cost="0.0100")
    second.update(fill_id="second", count_fp="0.50", fee_cost="0.0100")
    verify_fills((first, second), receipt, row)
    with pytest.raises(ValueError, match="DEMO-TRIAL-003"):
        verify_fills((first, first), receipt, row)
    with pytest.raises(ValueError, match="DEMO-TRIAL-003"):
        verify_fills((), receipt, row)
