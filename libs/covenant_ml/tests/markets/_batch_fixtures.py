"""Native multi-order account evidence consumed through real HTTP and journal code."""

from __future__ import annotations

from pathlib import Path

from platform_core.http_client import HttpxClient
from platform_core.json_utils import JSONObject, JSONValue, require_dict, require_list, require_str

from covenant_ml.markets.batch_account import account_snapshot_event
from covenant_ml.markets.batch_records import DemoBatchPlan, baseline_digest, decode_batch_plan
from covenant_ml.markets.demo_trial import build_demo_trial
from covenant_ml.markets.demo_trial_records import decode_trial
from covenant_ml.markets.execution_policy import apply_execution_event
from covenant_ml.markets.execution_records import OrderIntent, decode_intent, encode_intent
from covenant_ml.markets.execution_state import ExecutionState
from covenant_ml.markets.kalshi_auth import KalshiSigner
from covenant_ml.markets.validation import read_document

from ._account_settlement_fixtures import final_market, native_credit
from ._demo_matrix_fixtures import fixed
from ._demo_trial_fixtures import serve_submission
from ._execution_fixtures import empty_state, intent, native_order
from ._kalshi_fixtures import account_clock, balance, serve_account
from ._server import ApiServer
from .test_demo_settlement import serve_credit

PREFIX = "/trade-api/v2/portfolio/"


def baseline_state() -> ExecutionState:
    """Replay real reservation and native settlement codecs for a settled predecessor."""
    row = intent(1, 120000)
    state = apply_execution_event(
        empty_state(), {"action": "reserve", "observed_ms": 120000, "data": encode_intent(row)}
    )
    return apply_execution_event(
        state, account_snapshot_event(account_document((row,), (100,)), state, 20000099)
    )


def batch_plan(state: ExecutionState, count: int = 20) -> DemoBatchPlan:
    """Decode a fixed account-bound plan for pure validation and real storage tests."""
    return decode_batch_plan(
        {
            "schema": 1,
            "mode": "demo_execution_batch",
            "credential_fingerprint": "a" * 64,
            "baseline_digest": baseline_digest(state),
            "initial_cash": 20000099,
            "tickers": [f"MARKET-{index}" for index in range(2, count + 2)],
            "limit_price": 500,
            "created_ms": 120000,
            "expires_ms": 86520000,
        }
    )


def batch_intent(number: int, price: int = 500) -> OrderIntent:
    """Construct a deterministic one-contract attempt through the actual codec."""
    document = encode_intent(intent(number, 120000))
    document.update(limit_price=price, fee_reserve=10000 - price)
    return decode_intent(document)


def settled_predecessor(
    server: ApiServer, client: HttpxClient, signer: KalshiSigner, root: Path
) -> OrderIntent:
    """Execute and settle the actual single-trial service over loopback sockets."""
    service = build_demo_trial(client, signer, account_clock, origin=server.url)
    serve_account(server)
    service.prepare(root, "MARKET-1", 9800)
    serve_submission(server, root)
    service.submit(root)
    serve_credit(server, root)
    service.reconcile(root)
    return decode_trial(read_document(root / "trial.json"))["intent"]


def account_document(
    rows: tuple[OrderIntent, ...],
    quantities: tuple[int, ...],
    *,
    settled: bool = False,
    payout: int = 100,
) -> JSONObject:
    """Build exact native account evidence; the predecessor is always credited.

    Args:
        rows: Predecessor followed by batch intents.
        quantities: Filled hundredths of contracts, including predecessor.
        settled: Whether every filled batch order has also settled.
        payout: Batch settlement value in cents; predecessor remains settled YES.

    Returns:
        Complete independent native balances, orders, fills, positions and credits.
    """
    orders: list[JSONValue] = []
    fills: list[JSONValue] = []
    credits: list[JSONValue] = []
    markets: list[JSONValue] = []
    positions: list[JSONValue] = []
    cash = 20000099
    for index, (row, quantity) in enumerate(zip(rows, quantities, strict=True)):
        price = row["limit_price"]
        fee = (200 if index == 0 else 100) if quantity else 0
        cost = price * quantity // 100
        order = native_order(row)
        order.update(
            status="executed" if quantity == 100 else "canceled",
            fill_count_fp=fixed(quantity, 2),
            yes_price_dollars=fixed(price, 4),
            taker_fill_cost_dollars=fixed(cost, 4),
            taker_fees_dollars=fixed(fee, 4),
        )
        orders.append(order)
        cash -= (cost + fee) * 100
        if quantity:
            fills.append(
                {
                    "fill_id": "fill-" + row["client_order_id"],
                    "order_id": order["order_id"],
                    "ticker": row["ticker"],
                    "outcome_side": "yes",
                    "book_side": "bid",
                    "count_fp": fixed(quantity, 2),
                    "yes_price_dollars": fixed(price, 4),
                    "fee_cost": fixed(fee * 100, 6),
                    "created_time": order["last_update_time"],
                }
            )
            if index == 0 or settled:
                settlement_value = 100 if index == 0 else payout
                credit = native_credit(row["ticker"], settlement_value)
                credit.update(
                    yes_count_fp=fixed(quantity, 2),
                    yes_total_cost_dollars=fixed(cost, 4),
                    fee_cost=fixed(fee, 4),
                    revenue=quantity * settlement_value // 100,
                )
                credits.append(credit)
                markets.append(final_market(row["ticker"], settlement_value))
                cash += quantity * settlement_value * 100
            else:
                positions.append({"ticker": row["ticker"], "position_fp": fixed(quantity, 2)})
    first, last = balance(), balance()
    for value in (first, last):
        value.update(balance=cash // 10000, balance_dollars=fixed(cash, 6))
    return {
        "started_ms": 120000,
        "completed_ms": 120000,
        "first_balance": first,
        "last_balance": last,
        "orders": orders,
        "fills": fills,
        "settlements": credits,
        "markets": markets,
        "positions": positions,
    }


def serve_snapshot(server: ApiServer, document: JSONObject) -> None:
    """Queue all native pages, deliberately splitting order and fill pagination."""
    for key in ("first_balance", "last_balance"):
        server.documents.setdefault(PREFIX + "balance", []).append(require_dict(document, key))
    for key, endpoint in (
        ("orders", "orders"),
        ("fills", "fills"),
        ("settlements", "settlements"),
        ("positions", "positions"),
    ):
        rows = require_list(document, key)
        field = "market_positions" if key == "positions" else key
        server.documents.setdefault(PREFIX + endpoint, []).extend(
            [{field: rows[:1], "cursor": "next"}, {field: rows[1:], "cursor": ""}]
        )
    for value in require_list(document, "markets"):
        market = require_dict({"market": value}, "market")
        server.documents.setdefault(
            "/trade-api/v2/markets/" + require_str(market, "ticker"), []
        ).append({"market": market})


def serve_batch_quote(server: ApiServer, row: OrderIntent, quantity: int) -> None:
    """Queue actual quote capture and the matching native order acknowledgement."""
    path = "/trade-api/v2/markets/" + row["ticker"]
    server.documents.setdefault(path, []).append(
        {
            "market": {
                "ticker": row["ticker"],
                "status": "active",
                "close_time": "2099-01-01T00:00:00Z",
            }
        }
    )
    server.documents[path + "/orderbook"] = [
        {
            "orderbook_fp": {
                "yes_dollars": [["0.0100", "1.00"]],
                "no_dollars": [[fixed(10000 - row["limit_price"], 4), "1.00"]],
            }
        }
    ]
    server.statuses[PREFIX + "events/orders"] = 201
    server.documents[PREFIX + "events/orders"] = [
        {
            "client_order_id": row["client_order_id"],
            "order_id": "exchange-" + row["client_order_id"],
            "fill_count": fixed(quantity, 2),
            "remaining_count": "0.00",
            "ts_ms": row["created_ms"],
        }
    ]
