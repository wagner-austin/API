"""State-machine boundary tests with hand-calculated loss and exposure outcomes."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, require_dict

from covenant_ml.markets.execution_policy import apply_execution_event
from covenant_ml.markets.execution_records import (
    OrderIntent,
    decode_native_order,
    encode_intent,
    encode_receipt,
)
from covenant_ml.markets.execution_state import ExecutionState, execution_report

from ._execution_fixtures import empty_state, intent, native_order, settlement


def reserve(state: ExecutionState, row: OrderIntent) -> ExecutionState:
    """Apply the actual reservation transition at a declared observation time."""
    return apply_execution_event(
        state,
        {
            "action": "reserve",
            "observed_ms": 2000,
            "data": encode_intent(row),
        },
    )


def receipt_event(row: OrderIntent) -> JSONObject:
    """Encode a complete native fill through the production adapter."""
    return {
        "action": "receipt",
        "client_order_id": row["client_order_id"],
        "observed_ms": 2000,
        "data": encode_receipt(decode_native_order(native_order(row))),
    }


def settle_event(row: OrderIntent, payout: int = 0) -> JSONObject:
    """Create a final settlement journal input."""
    return {
        "action": "settlement",
        "client_order_id": row["client_order_id"],
        "observed_ms": 2000,
        "data": settlement(row, payout),
    }


def test_unknown_submission_blocks_new_orders_without_releasing_money() -> None:
    row = intent()
    state = reserve(empty_state(), row)
    report = execution_report(state)
    assert report["exposure"] == 10000
    assert report["issues"] == ["EXECUTION-UNRESOLVED"]
    with pytest.raises(ValueError, match=r"EXECUTION-004.*unresolved"):
        reserve(state, intent(2))
    assert execution_report(state) == report


def test_three_dollars_open_and_five_dollars_lifetime_loss() -> None:
    state = empty_state()
    for number in (1, 2, 3):
        row = intent(number)
        state = apply_execution_event(reserve(state, row), receipt_event(row))
    assert execution_report(state)["exposure"] == 30000
    with pytest.raises(ValueError, match=r"EXECUTION-004.*exceeds"):
        reserve(state, intent(4))
    for number in (1, 2, 3):
        state = apply_execution_event(state, settle_event(intent(number)))
    assert execution_report(state)["gross_realized_loss"] == 30000
    for number in (4, 5):
        row = intent(number)
        state = apply_execution_event(reserve(state, row), receipt_event(row))
    report = execution_report(state)
    assert (report["exposure"], report["remaining_loss_allowance"], report["can_reserve"]) == (
        20000,
        0,
        False,
    )
    with pytest.raises(ValueError, match=r"EXECUTION-004.*exceeds"):
        reserve(state, intent(6))
    for number in (4, 5):
        state = apply_execution_event(state, settle_event(intent(number)))
    assert execution_report(state)["net_realized_profit"] == -50000
    with pytest.raises(ValueError, match=r"EXECUTION-004.*exceeds"):
        reserve(state, intent(7))


def test_profits_do_not_replenish_loss_allowance_and_duplicate_ids_never_reuse() -> None:
    row = intent()
    cheap = encode_intent(row)
    cheap["limit_price"] = 1000
    state = apply_execution_event(
        empty_state(), {"action": "reserve", "observed_ms": 2000, "data": cheap}
    )
    receipt = receipt_event(row)
    data = require_dict(receipt, "data")
    data.update({"limit_price": 1000, "fill_cost": 1000})
    state = apply_execution_event(state, receipt)
    state = apply_execution_event(state, settle_event(row, 10000))
    report = execution_report(state)
    assert (report["net_realized_profit"], report["remaining_loss_allowance"]) == (8800, 50000)
    with pytest.raises(ValueError, match=r"EXECUTION-004.*already been"):
        reserve(state, row)


def test_same_contract_cannot_net_away_unsettled_risk() -> None:
    row = intent()
    state = apply_execution_event(reserve(empty_state(), row), receipt_event(row))
    other = encode_intent(intent(2))
    other["ticker"] = row["ticker"]
    with pytest.raises(ValueError, match=r"EXECUTION-004.*unsettled"):
        apply_execution_event(state, {"action": "reserve", "observed_ms": 2000, "data": other})


@pytest.mark.parametrize("observed", [999, 301000])
def test_reservation_time_window(observed: int) -> None:
    with pytest.raises(ValueError, match=r"EXECUTION-004.*expired"):
        apply_execution_event(
            empty_state(),
            {
                "action": "reserve",
                "observed_ms": observed,
                "data": encode_intent(intent()),
            },
        )


def test_cancel_receipt_releases_only_unfilled_reservation_and_is_idempotent() -> None:
    row = intent()
    state = reserve(empty_state(), row)
    partial = receipt_event(row)
    data = require_dict(partial, "data")
    data.update(
        {
            "status": "resting",
            "filled_count": 25,
            "remaining_count": 75,
            "fill_cost": 2450,
            "fees": 100,
        }
    )
    state = apply_execution_event(state, partial)
    assert execution_report(state)["exposure"] == 10000
    data.update({"status": "canceled", "remaining_count": 0, "updated_ms": 1001})
    state = apply_execution_event(state, partial)
    assert execution_report(state)["exposure"] == 2550
    assert apply_execution_event(state, partial) == state
    final = settle_event(row)
    final_data = require_dict(final, "data")
    final_data.update({"state": "scalar", "payout": 5000})
    state = apply_execution_event(state, final)
    assert execution_report(state)["gross_realized_loss"] == 1300
    assert apply_execution_event(state, final) == state
    assert apply_execution_event(state, partial) == state


@pytest.mark.parametrize(
    "changes",
    [
        {"ticker": "WRONG"},
        {"initial_count": 200, "filled_count": 200},
        {"limit_price": 9700},
        {"updated_ms": 999},
        {"updated_ms": 2001},
    ],
)
def test_receipt_identity_and_time(changes: JSONObject) -> None:
    row = intent()
    state = reserve(empty_state(), row)
    event = receipt_event(row)
    require_dict(event, "data").update(changes)
    with pytest.raises(ValueError, match="EXECUTION-005"):
        apply_execution_event(state, event)


@pytest.mark.parametrize(
    "changes",
    [
        {"order_id": "different", "updated_ms": 1001},
        {"fill_cost": 2500},
        {"filled_count": 0, "remaining_count": 100, "fill_cost": 0, "fees": 0, "updated_ms": 1001},
        {"fill_cost": 2449, "updated_ms": 1001},
        {"fees": 99, "updated_ms": 1001},
    ],
)
def test_cumulative_evidence_cannot_regress(changes: JSONObject) -> None:
    row = intent()
    event = receipt_event(row)
    require_dict(event, "data").update(
        {
            "status": "resting",
            "filled_count": 25,
            "remaining_count": 75,
            "fill_cost": 2450,
            "fees": 100,
        }
    )
    state = apply_execution_event(reserve(empty_state(), row), event)
    require_dict(event, "data").update(changes)
    with pytest.raises(ValueError, match=r"EXECUTION-005.*regressing"):
        apply_execution_event(state, event)


def test_terminal_receipt_cannot_be_replaced() -> None:
    row = intent()
    state = apply_execution_event(reserve(empty_state(), row), receipt_event(row))
    changed = receipt_event(row)
    require_dict(changed, "data")["updated_ms"] = 1001
    with pytest.raises(ValueError, match=r"EXECUTION-005.*regressing"):
        apply_execution_event(state, changed)


def test_unknown_and_reused_exchange_order_identity() -> None:
    row, second = intent(), intent(2)
    with pytest.raises(ValueError, match=r"EXECUTION-006.*exactly one"):
        apply_execution_event(empty_state(), receipt_event(row))
    state = reserve(empty_state(), row)
    mismatched = receipt_event(row)
    require_dict(mismatched, "data")["client_order_id"] = second["client_order_id"]
    with pytest.raises(ValueError, match=r"EXECUTION-006.*client ID"):
        apply_execution_event(state, mismatched)
    state = apply_execution_event(state, receipt_event(row))
    state = reserve(state, second)
    reused = receipt_event(second)
    require_dict(reused, "data")["order_id"] = "exchange-" + row["client_order_id"]
    with pytest.raises(ValueError, match=r"EXECUTION-006.*reused"):
        apply_execution_event(state, reused)


@pytest.mark.parametrize(
    "changes",
    [
        {"ticker": "WRONG"},
        {"state": "pending", "native_status": "active", "payout": None, "settled_ms": None},
        {"settled_ms": 999},
        {"received_ms": 2001},
    ],
)
def test_invalid_settlement_evidence(changes: JSONObject) -> None:
    row = intent()
    state = apply_execution_event(reserve(empty_state(), row), receipt_event(row))
    event = settle_event(row)
    require_dict(event, "data").update(changes)
    with pytest.raises(ValueError, match=r"EXECUTION-006.*terminal"):
        apply_execution_event(state, event)


def test_settlement_requires_terminal_order_and_cannot_change_afterward() -> None:
    row = intent()
    state = reserve(empty_state(), row)
    with pytest.raises(ValueError, match="EXECUTION-006"):
        apply_execution_event(state, settle_event(row))
    event = receipt_event(row)
    require_dict(event, "data").update(
        {"status": "resting", "filled_count": 0, "remaining_count": 100, "fill_cost": 0, "fees": 0}
    )
    resting = apply_execution_event(state, event)
    with pytest.raises(ValueError, match="EXECUTION-006"):
        apply_execution_event(resting, settle_event(row))
    state = apply_execution_event(state, receipt_event(row))
    state = apply_execution_event(state, settle_event(row))
    with pytest.raises(ValueError, match="EXECUTION-006"):
        apply_execution_event(state, settle_event(row, 10000))


def test_breaches_record_facts_and_prevent_new_risk_even_after_settlement() -> None:
    row = intent()
    event = receipt_event(row)
    require_dict(event, "data").update({"fees": 300, "fill_cost": 9900})
    state = apply_execution_event(reserve(empty_state(), row), event)
    report = execution_report(state)
    assert report["exposure"] == 10200
    assert report["issues"] == [
        "EXECUTION-HALTED",
        "EXECUTION-FEE-BREACH",
        "EXECUTION-PRICE-BREACH",
    ]
    final = settle_event(row)
    require_dict(final, "data")["rules_changed"] = True
    state = apply_execution_event(state, final)
    assert execution_report(state)["issues"] == [
        "EXECUTION-HALTED",
        "EXECUTION-FEE-BREACH",
        "EXECUTION-PRICE-BREACH",
        "EXECUTION-RULES-CHANGED",
    ]
    with pytest.raises(ValueError, match=r"EXECUTION-004.*halt"):
        reserve(state, intent(2))


def test_later_better_fills_cannot_erase_an_earlier_price_breach() -> None:
    row = intent()
    event = receipt_event(row)
    require_dict(event, "data").update(
        {
            "status": "resting",
            "filled_count": 50,
            "remaining_count": 50,
            "fill_cost": 5000,
            "fees": 100,
        }
    )
    state = apply_execution_event(reserve(empty_state(), row), event)
    assert state["halt_reason"] == "Evidence breach: EXECUTION-PRICE-BREACH"
    final = receipt_event(row)
    require_dict(final, "data")["updated_ms"] = 1001
    state = apply_execution_event(state, final)
    assert execution_report(state)["issues"] == ["EXECUTION-HALTED"]
    with pytest.raises(ValueError, match="EXECUTION-004"):
        reserve(state, intent(2))


def test_manual_halt_allows_reconciliation_but_never_more_orders() -> None:
    row = intent()
    state = reserve(empty_state(), row)
    state = apply_execution_event(
        state, {"action": "halt", "reason": "operator stopped", "observed_ms": 2000}
    )
    state = apply_execution_event(state, receipt_event(row))
    assert execution_report(state)["issues"] == ["EXECUTION-HALTED"]
    assert state["halt_reason"] == "operator stopped"
    with pytest.raises(ValueError, match="EXECUTION-004"):
        reserve(state, intent(2))
    with pytest.raises(ValueError, match="EXECUTION-007"):
        apply_execution_event(state, {"action": "resume", "observed_ms": 2000})
    with pytest.raises(ValueError, match="MARKET-001"):
        apply_execution_event(
            state, {"action": "halt", "reason": "clock rollback", "observed_ms": 1999}
        )
