"""Immutable execution rehearsal state and conservative loss accounting."""

from __future__ import annotations

from platform_core.json_utils import JSONObject, narrow_json_to_dict, require_dict, require_list
from typing_extensions import ReadOnly, TypedDict

from .account_settlement import (
    AccountSettlement,
    decode_account_settlement,
    encode_account_settlement,
)
from .execution_records import (
    CAPITAL,
    EXPOSURE_LIMIT,
    LOSS_LIMIT,
    ORDER_LIMIT,
    OrderIntent,
    OrderReceipt,
    OrderStatus,
    decode_intent,
    decode_receipt,
    encode_intent,
    encode_receipt,
)
from .settlement_records import Settlement, decode_settlement, encode_settlement
from .validation import require_bounded_int, require_nonempty


class TrackedOrder(TypedDict):
    """Reservation plus cumulative receipt and eventual final payout evidence."""

    intent: ReadOnly[OrderIntent]
    receipt: ReadOnly[OrderReceipt | None]
    settlement: ReadOnly[Settlement | None]
    account_settlement: ReadOnly[AccountSettlement | None]


class ExecutionState(TypedDict):
    """Replayed journal state; a manual halt has no reset operation."""

    orders: ReadOnly[tuple[TrackedOrder, ...]]
    halt_reason: ReadOnly[str | None]
    observed_ms: ReadOnly[int]


def encode_tracked_order(row: TrackedOrder) -> JSONObject:
    """Encode a reservation and its nullable evidence.

    Args:
        row: TrackedOrder from the journal reducer.

    Returns:
        JSONObject with explicit receipt and settlement nulls.
    """
    receipt, settlement = row["receipt"], row["settlement"]
    return {
        "intent": encode_intent(row["intent"]),
        "receipt": None if receipt is None else encode_receipt(receipt),
        "settlement": None if settlement is None else encode_settlement(settlement),
        "account_settlement": None
        if row["account_settlement"] is None
        else encode_account_settlement(row["account_settlement"]),
    }


def decode_tracked_order(document: JSONObject) -> TrackedOrder:
    """Decode record shapes; the journal reducer enforces cross-record invariants.

    Args:
        document: JSONObject with three required fields.

    Returns:
        TrackedOrder with typed evidence.

    Raises:
        KeyError: A nullable field was omitted.
        ValueError: A contained record is invalid.
        TypeError: A required field has an incorrect type.
    """
    return TrackedOrder(
        intent=decode_intent(require_dict(document, "intent")),
        receipt=None
        if document["receipt"] is None
        else decode_receipt(require_dict(document, "receipt")),
        settlement=None
        if document["settlement"] is None
        else decode_settlement(require_dict(document, "settlement")),
        account_settlement=None
        if document["account_settlement"] is None
        else decode_account_settlement(require_dict(document, "account_settlement")),
    )


def encode_execution_state(state: ExecutionState) -> JSONObject:
    """Encode the complete local rehearsal state.

    Args:
        state: Replayed ExecutionState.

    Returns:
        JSONObject for inspection and codec verification.
    """
    return {
        "orders": [encode_tracked_order(order) for order in state["orders"]],
        "halt_reason": state["halt_reason"],
        "observed_ms": state["observed_ms"],
    }


def decode_execution_state(document: JSONObject) -> ExecutionState:
    """Decode state shapes; persisted state is always rebuilt from journal events.

    Args:
        document: JSONObject with orders, halt reason and timestamp.

    Returns:
        ExecutionState with immutable tuples.

    Raises:
        KeyError: Required nullable halt reason is absent.
        TypeError: State fields have incorrect types.
        ValueError: A contained record or timestamp is invalid.
    """
    return ExecutionState(
        orders=tuple(
            decode_tracked_order(narrow_json_to_dict(value))
            for value in require_list(document, "orders")
        ),
        halt_reason=None
        if document["halt_reason"] is None
        else require_nonempty(document, "halt_reason"),
        observed_ms=require_bounded_int(document, "observed_ms", 0, 2**63 - 1),
    )


def order_accounting(order: TrackedOrder) -> tuple[int, int, int]:
    """Calculate reserved exposure, gross realized loss and net realized profit.

    Args:
        order: TrackedOrder from validated journal history.

    Returns:
        Integer money tuple (exposure, gross loss, net profit); payout rounds down.
    """
    intent, receipt, settlement = order["intent"], order["receipt"], order["settlement"]
    reserved = intent["contracts"] * intent["limit_price"] + intent["fee_reserve"]
    if receipt is None:
        return reserved, 0, 0
    spent = receipt["fill_cost"] + receipt["fees"]
    credited = order["account_settlement"]
    if credited is not None:
        profit = credited["revenue"] - spent
        return 0, max(-profit, 0), profit
    if settlement is not None and settlement["payout"] is not None:
        proceeds = settlement["payout"] * receipt["filled_count"] // 100
        profit = proceeds - spent
        return 0, max(-profit, 0), profit
    if receipt["status"] is OrderStatus.RESTING:
        remaining_cost = (receipt["remaining_count"] * intent["limit_price"] + 99) // 100
        remaining_fee = max(intent["fee_reserve"] - receipt["fees"], 0)
        return max(reserved, spent + remaining_cost + remaining_fee), 0, 0
    return spent, 0, 0


def policy_issues(state: ExecutionState) -> tuple[str, ...]:
    """Identify unresolved submissions and permanent accounting breaches.

    Args:
        state: Validated ExecutionState.

    Returns:
        Stable issue codes; an empty tuple does not assert profitable trading.
    """
    issues: list[str] = []
    if state["halt_reason"] is not None:
        issues.append("EXECUTION-HALTED")
    for order in state["orders"]:
        intent, receipt, settlement = order["intent"], order["receipt"], order["settlement"]
        if receipt is None:
            issues.append("EXECUTION-UNRESOLVED")
            continue
        if receipt["fees"] > intent["fee_reserve"]:
            issues.append("EXECUTION-FEE-BREACH")
        if receipt["fill_cost"] * 100 > intent["limit_price"] * receipt["filled_count"]:
            issues.append("EXECUTION-PRICE-BREACH")
        if settlement is not None and settlement["rules_changed"]:
            issues.append("EXECUTION-RULES-CHANGED")
    return tuple(dict.fromkeys(issues, True))


def execution_report(state: ExecutionState) -> JSONObject:
    """Report exact risk totals, fixed ceilings and explicit non-live readiness.

    Args:
        state: Replayed ExecutionState.

    Returns:
        JSONObject with amounts in 1/10000 dollars and orders_sent equal to zero.
    """
    totals = tuple(order_accounting(order) for order in state["orders"])
    exposure = sum(row[0] for row in totals)
    losses = sum(row[1] for row in totals)
    profit = sum(row[2] for row in totals)
    issues = policy_issues(state)
    return {
        "mode": "local_rehearsal",
        "live_ready": False,
        "orders_sent": 0,
        "money_units_per_dollar": 10000,
        "capital": CAPITAL,
        "order_limit": ORDER_LIMIT,
        "exposure_limit": EXPOSURE_LIMIT,
        "loss_limit": LOSS_LIMIT,
        "exposure": exposure,
        "gross_realized_loss": losses,
        "net_realized_profit": profit,
        "remaining_loss_allowance": max(LOSS_LIMIT - losses - exposure, 0),
        "can_reserve": not issues and exposure < EXPOSURE_LIMIT and losses + exposure < LOSS_LIMIT,
        "issues": list(issues),
        "state": encode_execution_state(state),
    }
