"""Immutable credential-bound demo trial records and exact fill reconciliation."""

from __future__ import annotations

from platform_core.json_utils import JSONObject, require_dict, require_str
from typing_extensions import ReadOnly, TypedDict

from .execution_records import (
    ORDER_LIMIT,
    OrderIntent,
    OrderReceipt,
    decode_intent,
    encode_intent,
    require_order_money,
)
from .kalshi_demo import DemoPreflight, decode_preflight, encode_preflight
from .validation import parse_fixed, parse_timestamp, require_bounded_int, require_nonempty


class DemoTrial(TypedDict):
    """One immutable intent bound to the initial authenticated account observation."""

    intent: ReadOnly[OrderIntent]
    initial: ReadOnly[DemoPreflight]


def encode_trial(trial: DemoTrial) -> JSONObject:
    """Encode a credential-bound trial manifest.

    Args:
        trial: Validated DemoTrial.

    Returns:
        Versioned demo manifest with integer monetary units.
    """
    return {
        "schema": 1,
        "mode": "demo_single_order",
        "intent": encode_intent(trial["intent"]),
        "initial": encode_preflight(trial["initial"]),
    }


def decode_trial(document: JSONObject) -> DemoTrial:
    """Require a one-contract trial whose full one-dollar budget is reserved.

    Args:
        document: Manifest JSON object.

    Returns:
        DemoTrial with sufficient initial mock balance.

    Raises:
        ValueError: DEMO-TRIAL-001 for mode, sizing, timing or balance mismatch.
        TypeError: A required manifest field is missing or malformed.
    """
    require_bounded_int(document, "schema", 1, 1)
    intent = decode_intent(require_dict(document, "intent"))
    initial = decode_preflight(require_dict(document, "initial"))
    if (
        require_str(document, "mode") != "demo_single_order"
        or intent["contracts"] != 1
        or intent["limit_price"] + intent["fee_reserve"] != ORDER_LIMIT
        or initial["balance_microdollars"] < ORDER_LIMIT * 100
        or not initial["completed_ms"] <= intent["created_ms"] <= initial["completed_ms"] + 30000
    ):
        raise ValueError("[DEMO-TRIAL-001] require fresh demo observation and one funded $1 trial")
    return DemoTrial(intent=intent, initial=initial)


def validate_acknowledgement(document: JSONObject, intent: OrderIntent, observed: int) -> str:
    """Validate an IOC acknowledgement without converting averages into accounting.

    Args:
        document: Native Create Order V2 response.
        intent: Persisted intent submitted once.
        observed: Local receipt timestamp in milliseconds.

    Returns:
        Native order identifier for later comparison with authenticated evidence.

    Raises:
        ValueError: DEMO-TRIAL-002 for mismatched identity, counts or processing time.
    """
    filled = parse_fixed(require_str(document, "fill_count"), 2)
    remaining = parse_fixed(require_str(document, "remaining_count"), 2)
    processed = require_bounded_int(document, "ts_ms", intent["created_ms"], observed)
    if (
        require_str(document, "client_order_id") != intent["client_order_id"]
        or filled > 100
        or remaining != 0
        or processed >= intent["expires_ms"]
    ):
        raise ValueError("[DEMO-TRIAL-002] inconsistent IOC acknowledgement")
    return require_nonempty(document, "order_id")


def verify_fills(
    documents: tuple[JSONObject, ...], receipt: OrderReceipt, intent: OrderIntent
) -> None:
    """Match independently paginated fills against cumulative native order totals.

    Args:
        documents: Complete native fill objects, scoped to primary subaccount.
        receipt: Validated cumulative order receipt.
        intent: Persisted trial intent.

    Raises:
        ValueError: DEMO-TRIAL-003 for duplicate, foreign or inconsistent fills.
    """
    identities: set[str] = set()
    quantity = cost = fees = 0
    for document in documents:
        identity = require_nonempty(document, "fill_id")
        count = parse_fixed(require_str(document, "count_fp"), 2)
        price = require_order_money(document, "yes_price_dollars")
        timestamp = parse_timestamp(require_str(document, "created_time"))
        if (
            identity in identities
            or require_str(document, "order_id") != receipt["order_id"]
            or require_str(document, "ticker") != intent["ticker"]
            or require_str(document, "outcome_side") != "yes"
            or require_str(document, "book_side") != "bid"
            or not 0 < count <= 100
            or not 0 < price <= intent["limit_price"]
            or not intent["created_ms"] <= timestamp <= receipt["updated_ms"]
        ):
            raise ValueError("[DEMO-TRIAL-003] duplicate, foreign or inconsistent fill evidence")
        identities.add(identity)
        quantity += count
        cost += count * price
        fees += parse_fixed(require_str(document, "fee_cost"), 6)
    if (quantity, cost, fees) != (
        receipt["filled_count"],
        receipt["fill_cost"] * 100,
        receipt["fees"] * 100,
    ):
        raise ValueError("[DEMO-TRIAL-003] independent fills disagree with cumulative order totals")
