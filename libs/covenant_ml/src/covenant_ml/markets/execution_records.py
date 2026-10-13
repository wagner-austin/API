"""Exact-money records for the local, long-YES execution rehearsal."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from platform_core.json_utils import JSONObject, require_str
from typing_extensions import ReadOnly, TypedDict

from .types import DOLLAR
from .validation import parse_fixed, parse_timestamp, require_bounded_int, require_nonempty

CAPITAL = 20 * DOLLAR
ORDER_LIMIT = DOLLAR
EXPOSURE_LIMIT = 3 * DOLLAR
LOSS_LIMIT = 5 * DOLLAR


class OrderStatus(StrEnum):
    """Native order states accepted by the rehearsal."""

    RESTING = "resting"
    CANCELED = "canceled"
    EXECUTED = "executed"


class OrderIntent(TypedDict):
    """Whole-contract YES purchase; prices and fee reserve use 1/10000 dollars."""

    client_order_id: ReadOnly[str]
    ticker: ReadOnly[str]
    contracts: ReadOnly[int]
    limit_price: ReadOnly[int]
    fee_reserve: ReadOnly[int]
    created_ms: ReadOnly[int]
    expires_ms: ReadOnly[int]


class OrderReceipt(TypedDict):
    """Cumulative native order accounting; counts use 1/100 contracts."""

    client_order_id: ReadOnly[str]
    order_id: ReadOnly[str]
    ticker: ReadOnly[str]
    status: ReadOnly[OrderStatus]
    initial_count: ReadOnly[int]
    filled_count: ReadOnly[int]
    remaining_count: ReadOnly[int]
    limit_price: ReadOnly[int]
    fill_cost: ReadOnly[int]
    fees: ReadOnly[int]
    updated_ms: ReadOnly[int]


def encode_intent(row: OrderIntent) -> JSONObject:
    """Encode an immutable order intent.

    Args:
        row: Validated OrderIntent.

    Returns:
        JSONObject with integer amounts and timestamps.
    """
    return {
        "client_order_id": row["client_order_id"],
        "ticker": row["ticker"],
        "contracts": row["contracts"],
        "limit_price": row["limit_price"],
        "fee_reserve": row["fee_reserve"],
        "created_ms": row["created_ms"],
        "expires_ms": row["expires_ms"],
    }


def decode_intent(document: JSONObject) -> OrderIntent:
    """Validate a bounded, expiring order without guessing a fee schedule.

    Args:
        document: JSONObject supplied at the boundary.

    Returns:
        OrderIntent with a canonical UUID and at most one dollar reserved.

    Raises:
        ValueError: EXECUTION-001 for identity, timing or per-order budget errors.
        TypeError: Required fields are missing or incorrectly typed.
    """
    identity = require_nonempty(document, "client_order_id")
    contracts = require_bounded_int(document, "contracts", 1, 100)
    price = require_bounded_int(document, "limit_price", 1, DOLLAR - 1)
    fee = require_bounded_int(document, "fee_reserve", 0, ORDER_LIMIT)
    created = require_bounded_int(document, "created_ms", 0, 2**63 - 1)
    expires = require_bounded_int(document, "expires_ms", 0, 2**63 - 1)
    if (
        str(UUID(identity)) != identity
        or not created < expires <= created + 300_000
        or price * contracts + fee > ORDER_LIMIT
    ):
        raise ValueError("[EXECUTION-001] require canonical UUID, <=5 minute intent and <=$1 risk")
    return OrderIntent(
        client_order_id=identity,
        ticker=require_nonempty(document, "ticker"),
        contracts=contracts,
        limit_price=price,
        fee_reserve=fee,
        created_ms=created,
        expires_ms=expires,
    )


def encode_receipt(row: OrderReceipt) -> JSONObject:
    """Encode cumulative order evidence without losing fractional fills.

    Args:
        row: Validated OrderReceipt.

    Returns:
        JSONObject in the normalized accounting schema.
    """
    return {
        "client_order_id": row["client_order_id"],
        "order_id": row["order_id"],
        "ticker": row["ticker"],
        "status": row["status"].value,
        "initial_count": row["initial_count"],
        "filled_count": row["filled_count"],
        "remaining_count": row["remaining_count"],
        "limit_price": row["limit_price"],
        "fill_cost": row["fill_cost"],
        "fees": row["fees"],
        "updated_ms": row["updated_ms"],
    }


def decode_receipt(document: JSONObject) -> OrderReceipt:
    """Require consistent cumulative counts and terminal order states.

    Args:
        document: JSONObject in the normalized schema.

    Returns:
        OrderReceipt; excessive costs remain evidence for a policy halt.

    Raises:
        ValueError: EXECUTION-002 for contradictory counts or empty-fill costs.
        TypeError: Required fields are missing or incorrectly typed.
    """
    status = OrderStatus(require_str(document, "status"))
    initial = require_bounded_int(document, "initial_count", 1, 10000)
    filled = require_bounded_int(document, "filled_count", 0, initial)
    remaining = require_bounded_int(document, "remaining_count", 0, initial - filled)
    cost = require_bounded_int(document, "fill_cost", 0, CAPITAL)
    fees = require_bounded_int(document, "fees", 0, CAPITAL)
    if (
        (status is OrderStatus.RESTING and (remaining == 0 or filled + remaining != initial))
        or (status is not OrderStatus.RESTING and remaining != 0)
        or (status is OrderStatus.EXECUTED and filled != initial)
        or (filled == 0 and cost + fees != 0)
    ):
        raise ValueError("[EXECUTION-002] inconsistent order quantities, status or costs")
    return OrderReceipt(
        client_order_id=require_nonempty(document, "client_order_id"),
        order_id=require_nonempty(document, "order_id"),
        ticker=require_nonempty(document, "ticker"),
        status=status,
        initial_count=initial,
        filled_count=filled,
        remaining_count=remaining,
        limit_price=require_bounded_int(document, "limit_price", 1, DOLLAR - 1),
        fill_cost=cost,
        fees=fees,
        updated_ms=require_bounded_int(document, "updated_ms", 0, 2**63 - 1),
    )


def decode_native_order(document: JSONObject) -> OrderReceipt:
    """Decode the current Get Order object's long-YES accounting fields.

    Args:
        document: Native order object, without its response envelope.

    Returns:
        OrderReceipt retaining actual fees and cumulative acquisition cost.

    Raises:
        ValueError: EXECUTION-003 for unsupported order direction or type.
        TypeError: Required native fields are absent or malformed.
    """
    if (
        require_str(document, "outcome_side") != "yes"
        or require_str(document, "book_side") != "bid"
        or require_str(document, "type") != "limit"
    ):
        raise ValueError("[EXECUTION-003] rehearsal supports only long YES limit orders")
    return decode_receipt(
        {
            "client_order_id": require_nonempty(document, "client_order_id"),
            "order_id": require_nonempty(document, "order_id"),
            "ticker": require_nonempty(document, "ticker"),
            "status": require_str(document, "status"),
            "initial_count": parse_fixed(require_str(document, "initial_count_fp"), 2),
            "filled_count": parse_fixed(require_str(document, "fill_count_fp"), 2),
            "remaining_count": parse_fixed(require_str(document, "remaining_count_fp"), 2),
            "limit_price": parse_fixed(require_str(document, "yes_price_dollars"), 4),
            "fill_cost": (
                parse_fixed(require_str(document, "taker_fill_cost_dollars"), 4)
                + parse_fixed(require_str(document, "maker_fill_cost_dollars"), 4)
            ),
            "fees": (
                parse_fixed(require_str(document, "taker_fees_dollars"), 4)
                + parse_fixed(require_str(document, "maker_fees_dollars"), 4)
            ),
            "updated_ms": parse_timestamp(require_str(document, "last_update_time")),
        }
    )


def order_payload(intent: OrderIntent) -> JSONObject:
    """Render a V2 IOC request for inspection, without sending it.

    Args:
        intent: Validated long-YES OrderIntent.

    Returns:
        JSONObject matching the current event-order request shape.
    """
    price = intent["limit_price"]
    return {
        "ticker": intent["ticker"],
        "client_order_id": intent["client_order_id"],
        "side": "bid",
        "count": f"{intent['contracts']}.00",
        "price": f"{price // DOLLAR}.{price % DOLLAR:04d}",
        "time_in_force": "immediate_or_cancel",
        "self_trade_prevention_type": "taker_at_cross",
        "cancel_order_on_pause": True,
        "post_only": False,
        "reduce_only": False,
    }
