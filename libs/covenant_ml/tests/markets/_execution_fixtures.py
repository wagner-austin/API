"""Deterministic input records passed through the real execution codecs."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from platform_core.json_utils import JSONObject

from covenant_ml.markets.execution_records import OrderIntent, decode_intent
from covenant_ml.markets.execution_state import ExecutionState


def intent(number: int = 1, created: int = 1000) -> OrderIntent:
    """Return a one-contract intent reserving exactly one dollar."""
    return decode_intent(
        {
            "client_order_id": str(UUID(int=number)),
            "ticker": f"MARKET-{number}",
            "contracts": 1,
            "limit_price": 9800,
            "fee_reserve": 200,
            "created_ms": created,
            "expires_ms": created + 300000,
        }
    )


def empty_state() -> ExecutionState:
    """Return the initial immutable state."""
    return ExecutionState(orders=(), halt_reason=None, observed_ms=0)


def native_order(row: OrderIntent) -> JSONObject:
    """Return a native terminal fill at its limit with the reserved fee."""
    return {
        "client_order_id": row["client_order_id"],
        "order_id": "exchange-" + row["client_order_id"],
        "ticker": row["ticker"],
        "outcome_side": "yes",
        "book_side": "bid",
        "type": "limit",
        "status": "executed",
        "initial_count_fp": "1.00",
        "fill_count_fp": "1.00",
        "remaining_count_fp": "0.00",
        "yes_price_dollars": "0.9800",
        "maker_fill_cost_dollars": "0.0000",
        "taker_fill_cost_dollars": "0.9800",
        "maker_fees_dollars": "0.0000",
        "taker_fees_dollars": "0.0200",
        "last_update_time": datetime.fromtimestamp(row["created_ms"] / 1000, UTC).isoformat(),
    }


def settlement(row: OrderIntent, payout: int = 0) -> JSONObject:
    """Return finalized normalized settlement evidence for one contract."""
    return {
        "ticker": row["ticker"],
        "state": "binary",
        "native_status": "finalized",
        "received_ms": row["created_ms"] + 1,
        "settled_ms": row["created_ms"] + 1,
        "payout": payout,
        "rules_changed": False,
    }
