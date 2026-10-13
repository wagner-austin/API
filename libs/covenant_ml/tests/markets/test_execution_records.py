"""Exact-money codec, native schema and conservative accounting checks."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONValue

from covenant_ml.markets.execution_records import (
    decode_intent,
    decode_native_order,
    decode_receipt,
    encode_intent,
    encode_receipt,
    order_payload,
)
from covenant_ml.markets.execution_state import (
    ExecutionState,
    TrackedOrder,
    decode_execution_state,
    decode_tracked_order,
    encode_execution_state,
    encode_tracked_order,
    execution_report,
    order_accounting,
)
from covenant_ml.markets.settlement_records import decode_settlement

from ._execution_fixtures import empty_state, intent, native_order, settlement


def test_intent_and_current_v2_payload() -> None:
    row = intent()
    assert decode_intent(encode_intent(row)) == row
    assert order_payload(row) == {
        "ticker": "MARKET-1",
        "client_order_id": row["client_order_id"],
        "side": "bid",
        "count": "1.00",
        "price": "0.9800",
        "time_in_force": "immediate_or_cancel",
        "self_trade_prevention_type": "taker_at_cross",
        "cancel_order_on_pause": True,
        "post_only": False,
        "reduce_only": False,
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("client_order_id", "00000000000000000000000000000001"),
        ("expires_ms", 1000),
        ("expires_ms", 301001),
        ("limit_price", 9900),
        ("contracts", 2),
    ],
)
def test_invalid_intents(key: str, value: JSONValue) -> None:
    row = encode_intent(intent())
    row[key] = value
    with pytest.raises(ValueError, match="EXECUTION-001"):
        decode_intent(row)


@pytest.mark.parametrize(
    "key,value",
    [
        ("contracts", True),
        ("limit_price", "0.98"),
        ("fee_reserve", None),
    ],
)
def test_untyped_money_is_refused(key: str, value: JSONValue) -> None:
    row = encode_intent(intent())
    row[key] = value
    with pytest.raises(TypeError):
        decode_intent(row)


@pytest.mark.parametrize(
    "key,value",
    [
        ("outcome_side", "no"),
        ("book_side", "ask"),
        ("type", "market"),
    ],
)
def test_unsupported_native_direction(key: str, value: JSONValue) -> None:
    row = native_order(intent())
    row[key] = value
    with pytest.raises(ValueError, match="EXECUTION-003"):
        decode_native_order(row)


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "resting"},
        {"status": "resting", "remaining_count": 50, "filled_count": 0, "fill_cost": 0, "fees": 0},
        {"status": "canceled", "filled_count": 50, "remaining_count": 50},
        {"filled_count": 50},
        {"status": "canceled", "filled_count": 0},
    ],
)
def test_inconsistent_native_counts(changes: JSONObject) -> None:
    row = encode_receipt(decode_native_order(native_order(intent())))
    row.update(changes)
    with pytest.raises(ValueError, match="EXECUTION-002"):
        decode_receipt(row)


def test_full_roundtrip_and_empty_report() -> None:
    row = intent()
    filled = decode_native_order(native_order(row))
    assert decode_receipt(encode_receipt(filled)) == filled
    tracked = TrackedOrder(
        intent=row, receipt=filled, settlement=decode_settlement(settlement(row))
    )
    assert decode_tracked_order(encode_tracked_order(tracked)) == tracked
    state = ExecutionState(orders=(tracked,), halt_reason="finished", observed_ms=2000)
    assert decode_execution_state(encode_execution_state(state)) == state
    empty = empty_state()
    assert decode_execution_state(encode_execution_state(empty)) == empty
    reserved = TrackedOrder(intent=row, receipt=None, settlement=None)
    assert decode_tracked_order(encode_tracked_order(reserved)) == reserved
    assert execution_report(empty) == {
        "mode": "local_rehearsal",
        "live_ready": False,
        "orders_sent": 0,
        "money_units_per_dollar": 10000,
        "capital": 200000,
        "order_limit": 10000,
        "exposure_limit": 30000,
        "loss_limit": 50000,
        "exposure": 0,
        "gross_realized_loss": 0,
        "net_realized_profit": 0,
        "remaining_loss_allowance": 50000,
        "can_reserve": True,
        "issues": [],
        "state": {"orders": [], "halt_reason": None, "observed_ms": 0},
    }


def test_partial_fill_cancellation_and_fractional_settlement() -> None:
    row = intent()
    document = native_order(row)
    document.update(
        {
            "status": "resting",
            "fill_count_fp": "0.25",
            "remaining_count_fp": "0.75",
            "taker_fill_cost_dollars": "0.2450",
            "taker_fees_dollars": "0.0100",
        }
    )
    receipt = decode_native_order(document)
    tracked = TrackedOrder(intent=row, receipt=receipt, settlement=None)
    assert order_accounting(tracked) == (10000, 0, 0)
    document.update({"status": "canceled", "remaining_count_fp": "0.00"})
    canceled = decode_native_order(document)
    assert order_accounting(TrackedOrder(intent=row, receipt=canceled, settlement=None)) == (
        2550,
        0,
        0,
    )
    final = settlement(row)
    final.update({"state": "scalar", "payout": 3333})
    assert order_accounting(
        TrackedOrder(intent=row, receipt=canceled, settlement=decode_settlement(final))
    ) == (0, 1717, -1717)


def test_actual_cost_breach_increases_risk_and_blocks_reservations() -> None:
    row = intent()
    document = native_order(row)
    document.update(
        {
            "status": "resting",
            "fill_count_fp": "0.50",
            "remaining_count_fp": "0.50",
            "taker_fill_cost_dollars": "0.5100",
            "taker_fees_dollars": "0.0300",
        }
    )
    tracked = TrackedOrder(intent=row, receipt=decode_native_order(document), settlement=None)
    state = ExecutionState(orders=(tracked,), halt_reason=None, observed_ms=2000)
    report = execution_report(state)
    assert report["exposure"] == 10300
    assert report["issues"] == ["EXECUTION-FEE-BREACH", "EXECUTION-PRICE-BREACH"]
    assert report["can_reserve"] is False


def test_pending_settlement_shape_does_not_release_exposure() -> None:
    row = intent()
    pending = settlement(row)
    pending.update(
        {"state": "pending", "native_status": "active", "payout": None, "settled_ms": None}
    )
    tracked = TrackedOrder(
        intent=row,
        receipt=decode_native_order(native_order(row)),
        settlement=decode_settlement(pending),
    )
    assert order_accounting(tracked) == (10000, 0, 0)
