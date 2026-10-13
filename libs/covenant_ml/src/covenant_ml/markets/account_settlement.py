"""Exact account settlement credits, distinct from public market outcome evidence."""

from __future__ import annotations

from platform_core.json_utils import JSONObject, require_str
from typing_extensions import ReadOnly, TypedDict

from .execution_records import CAPITAL, OrderReceipt, require_order_money
from .types import DOLLAR
from .validation import parse_fixed, parse_timestamp, require_bounded_int, require_nonempty


class AccountSettlement(TypedDict):
    """Credited long-YES position; all amounts use the journal's exact money units."""

    ticker: ReadOnly[str]
    quantity: ReadOnly[int]
    cost: ReadOnly[int]
    fees: ReadOnly[int]
    revenue: ReadOnly[int]
    payout: ReadOnly[int]
    settled_ms: ReadOnly[int]


def encode_account_settlement(row: AccountSettlement) -> JSONObject:
    """Encode validated account credit evidence.

    Args:
        row: AccountSettlement with exact integer amounts.

    Returns:
        JSONObject for journal persistence.
    """
    return {
        "ticker": row["ticker"],
        "quantity": row["quantity"],
        "cost": row["cost"],
        "fees": row["fees"],
        "revenue": row["revenue"],
        "payout": row["payout"],
        "settled_ms": row["settled_ms"],
    }


def decode_account_settlement(document: JSONObject) -> AccountSettlement:
    """Require exactly representable payout arithmetic without rounding.

    Args:
        document: Normalized account credit object.

    Returns:
        Immutable AccountSettlement.

    Raises:
        ValueError: ACCOUNT-SETTLEMENT-001 for inconsistent revenue.
        TypeError: Required fields are missing or incorrectly typed.
    """
    quantity = require_bounded_int(document, "quantity", 1, 10000)
    payout = require_bounded_int(document, "payout", 0, DOLLAR)
    revenue = require_bounded_int(document, "revenue", 0, 100 * DOLLAR)
    if revenue * 100 != quantity * payout:
        raise ValueError("[ACCOUNT-SETTLEMENT-001] credited revenue disagrees with exact payout")
    return AccountSettlement(
        ticker=require_nonempty(document, "ticker"),
        quantity=quantity,
        cost=require_bounded_int(document, "cost", 0, CAPITAL),
        fees=require_bounded_int(document, "fees", 0, CAPITAL),
        revenue=revenue,
        payout=payout,
        settled_ms=require_bounded_int(document, "settled_ms", 0, 2**63 - 1),
    )


def decode_native_account_settlement(
    document: JSONObject, market: JSONObject, receipt: OrderReceipt, observed: int
) -> AccountSettlement:
    """Cross-check authenticated settlement against finalized public market and order.

    Args:
        document: Native primary-subaccount settlement row.
        market: Native market object from the same reconciliation pass.
        receipt: Independently verified terminal order receipt.
        observed: Response observation timestamp in milliseconds.

    Returns:
        AccountSettlement whose credit matches the position and native payout.

    Raises:
        ValueError: ACCOUNT-SETTLEMENT-002 for identity, lifecycle or result mismatch;
            ACCOUNT-SETTLEMENT-003 for a position or acquisition accounting mismatch.
        TypeError: Required native fields are missing or malformed.
    """
    result = require_str(document, "market_result")
    payout = require_order_money(market, "settlement_value_dollars")
    settled = parse_timestamp(require_str(document, "settled_time"))
    market_settled = parse_timestamp(require_str(market, "settlement_ts"))
    if (
        require_str(document, "ticker") != receipt["ticker"]
        or require_str(market, "ticker") != receipt["ticker"]
        or require_str(document, "event_ticker") != require_nonempty(market, "event_ticker")
        or require_str(market, "status") != "finalized"
        or result != require_str(market, "result")
        or result not in ("yes", "no", "scalar")
        or (result == "yes" and payout != DOLLAR)
        or (result == "no" and payout != 0)
        or not receipt["updated_ms"] <= market_settled <= settled <= observed
    ):
        raise ValueError("[ACCOUNT-SETTLEMENT-002] inconsistent final market and account outcome")
    row = decode_account_settlement(
        {
            "ticker": receipt["ticker"],
            "quantity": parse_fixed(require_str(document, "yes_count_fp"), 2),
            "cost": require_order_money(document, "yes_total_cost_dollars"),
            "fees": require_order_money(document, "fee_cost"),
            "revenue": require_bounded_int(document, "revenue", 0, 10000) * 100,
            "payout": payout,
            "settled_ms": settled,
        }
    )
    if (
        parse_fixed(require_str(document, "no_count_fp"), 2) != 0
        or require_order_money(document, "no_total_cost_dollars") != 0
        or (row["quantity"], row["cost"], row["fees"])
        != (receipt["filled_count"], receipt["fill_cost"], receipt["fees"])
    ):
        raise ValueError(
            "[ACCOUNT-SETTLEMENT-003] account settlement disagrees with acquired position"
        )
    return row
