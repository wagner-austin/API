"""Pure journal transitions for bounded local order rehearsal."""

from __future__ import annotations

from platform_core.json_utils import JSONObject, require_dict, require_str

from .account_settlement import decode_account_settlement
from .execution_records import (
    EXPOSURE_LIMIT,
    LOSS_LIMIT,
    OrderReceipt,
    OrderStatus,
    decode_intent,
    decode_receipt,
)
from .execution_state import ExecutionState, TrackedOrder, order_accounting, policy_issues
from .settlement_records import SettlementState, decode_settlement
from .validation import require_bounded_int, require_nonempty


def _reserve(
    state: ExecutionState, document: JSONObject, observed: int
) -> tuple[TrackedOrder, ...]:
    """Validate a new reservation against existing worst-case exposure.

    Args:
        state: Prior ExecutionState.
        document: OrderIntent JSON.
        observed: Local decision timestamp in milliseconds.

    Returns:
        Tuple containing the newly reserved order.

    Raises:
        ValueError: EXECUTION-004 for blocked, duplicate, stale or excessive reservations.
    """
    intent = decode_intent(document)
    if policy_issues(state):
        raise ValueError("[EXECUTION-004] new orders blocked by unresolved evidence or policy halt")
    if not intent["created_ms"] <= observed < intent["expires_ms"]:
        raise ValueError("[EXECUTION-004] reservation intent is future-dated or expired")
    for order in state["orders"]:
        old = order["intent"]
        if old["client_order_id"] == intent["client_order_id"]:
            raise ValueError("[EXECUTION-004] client order ID has already been reserved")
        if old["ticker"] == intent["ticker"] and order_accounting(order)[0] > 0:
            raise ValueError("[EXECUTION-004] contract already has unsettled exposure")
    totals = tuple(order_accounting(order) for order in state["orders"])
    risk = intent["contracts"] * intent["limit_price"] + intent["fee_reserve"]
    exposure = sum(row[0] for row in totals) + risk
    losses = sum(row[1] for row in totals)
    if exposure > EXPOSURE_LIMIT or losses + exposure > LOSS_LIMIT:
        raise ValueError("[EXECUTION-004] reservation exceeds exposure or cumulative loss limit")
    return (
        *state["orders"],
        TrackedOrder(intent=intent, receipt=None, settlement=None, account_settlement=None),
    )


def _receipt(order: TrackedOrder, receipt: OrderReceipt, observed: int) -> TrackedOrder:
    """Require matching identity and monotonic cumulative execution evidence.

    Args:
        order: Existing TrackedOrder.
        receipt: New cumulative OrderReceipt.
        observed: Local receipt time in milliseconds.

    Returns:
        Updated TrackedOrder; actual fee/price breaches remain recorded.

    Raises:
        ValueError: EXECUTION-005 for changed identity, stale or contradictory evidence.
    """
    intent, previous = order["intent"], order["receipt"]
    if (
        receipt["ticker"] != intent["ticker"]
        or receipt["initial_count"] != intent["contracts"] * 100
        or receipt["limit_price"] != intent["limit_price"]
        or not intent["created_ms"] <= receipt["updated_ms"] <= observed
    ):
        raise ValueError("[EXECUTION-005] order receipt identity, quantity, price or time mismatch")
    if previous is not None:
        if previous == receipt:
            return order
        if (
            previous["status"] is not OrderStatus.RESTING
            or previous["order_id"] != receipt["order_id"]
            or receipt["updated_ms"] <= previous["updated_ms"]
            or receipt["filled_count"] < previous["filled_count"]
            or receipt["remaining_count"] > previous["remaining_count"]
            or receipt["fill_cost"] < previous["fill_cost"]
            or receipt["fees"] < previous["fees"]
        ):
            raise ValueError("[EXECUTION-005] contradictory or regressing cumulative receipt")
    return TrackedOrder(intent=intent, receipt=receipt, settlement=None, account_settlement=None)


def _reconcile(
    state: ExecutionState, action: str, document: JSONObject, observed: int
) -> tuple[TrackedOrder, ...]:
    """Reconcile one known reservation while preserving every unrelated order.

    Args:
        state: Prior ExecutionState.
        action: Receipt or settlement operation.
        document: Event JSON with client_order_id and data.
        observed: Local receipt time in milliseconds.

    Returns:
        Updated immutable order tuple.

    Raises:
        ValueError: EXECUTION-006 for unknown order or inadmissible settlement.
    """
    identity = require_nonempty(document, "client_order_id")
    matches = tuple(
        order for order in state["orders"] if order["intent"]["client_order_id"] == identity
    )
    if len(matches) != 1:
        raise ValueError("[EXECUTION-006] reconciliation requires exactly one reserved order")
    order = matches[0]
    data = require_dict(document, "data")
    if action == "receipt":
        receipt = decode_receipt(data)
        if receipt["client_order_id"] != identity or any(
            other["receipt"] is not None
            and other["receipt"]["order_id"] == receipt["order_id"]
            and other["intent"]["client_order_id"] != identity
            for other in state["orders"]
        ):
            raise ValueError("[EXECUTION-006] receipt client ID mismatch or reused exchange ID")
        updated = _receipt(order, receipt, observed)
    elif action == "account_settlement":
        credit = decode_account_settlement(data)
        filled = order["receipt"]
        previous = order["account_settlement"]
        if (
            filled is None
            or filled["status"] is OrderStatus.RESTING
            or order["settlement"] is not None
            or (credit["ticker"], credit["quantity"], credit["cost"], credit["fees"])
            != (filled["ticker"], filled["filled_count"], filled["fill_cost"], filled["fees"])
            or not filled["updated_ms"] <= credit["settled_ms"] <= observed
            or (previous is not None and previous != credit)
        ):
            raise ValueError("[EXECUTION-011] require matching immutable account settlement credit")
        updated = TrackedOrder(
            intent=order["intent"], receipt=filled, settlement=None, account_settlement=credit
        )
    else:
        settlement = decode_settlement(data)
        settled_receipt = order["receipt"]
        if (
            settled_receipt is None
            or settled_receipt["status"] is OrderStatus.RESTING
            or settlement["ticker"] != order["intent"]["ticker"]
            or settlement["state"] is SettlementState.PENDING
            or order["account_settlement"] is not None
            or settlement["settled_ms"] is None
            or settlement["settled_ms"] < settled_receipt["updated_ms"]
            or not settlement["settled_ms"] <= settlement["received_ms"] <= observed
            or (order["settlement"] is not None and order["settlement"] != settlement)
        ):
            raise ValueError("[EXECUTION-006] require terminal order and matching final settlement")
        updated = TrackedOrder(
            intent=order["intent"],
            receipt=settled_receipt,
            settlement=settlement,
            account_settlement=None,
        )
    return tuple(
        updated if item["intent"]["client_order_id"] == identity else item
        for item in state["orders"]
    )


def apply_execution_event(state: ExecutionState, event: JSONObject) -> ExecutionState:
    """Apply one validated event or propagate its specific failure.

    Args:
        state: Prior immutable ExecutionState.
        event: JSONObject from the CLI or persisted journal.

    Returns:
        New ExecutionState; failed transitions leave prior state untouched.

    Raises:
        ValueError: EXECUTION-007 for unknown operation; MARKET-001 for a backward clock.
        TypeError: Event fields are absent or malformed.
    """
    action = require_str(event, "action")
    observed = require_bounded_int(event, "observed_ms", state["observed_ms"], 2**63 - 1)
    orders, reason = state["orders"], state["halt_reason"]
    if action == "reserve":
        orders = _reserve(state, require_dict(event, "data"), observed)
    elif action in ("receipt", "settlement", "account_settlement"):
        orders = _reconcile(state, action, event, observed)
    elif action == "halt":
        reason = require_nonempty(event, "reason")
    else:
        raise ValueError("[EXECUTION-007] unknown execution journal operation")
    candidate = ExecutionState(orders=orders, halt_reason=reason, observed_ms=observed)
    breaches = tuple(
        issue
        for issue in policy_issues(candidate)
        if issue in ("EXECUTION-FEE-BREACH", "EXECUTION-PRICE-BREACH", "EXECUTION-RULES-CHANGED")
    )
    if breaches and reason is None:
        return ExecutionState(
            orders=orders,
            halt_reason="Evidence breach: " + ", ".join(breaches),
            observed_ms=observed,
        )
    return candidate
