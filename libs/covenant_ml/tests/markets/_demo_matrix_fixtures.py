"""Variable native exchange evidence served by the existing real socket server."""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import JSONObject, require_dict, require_list

from covenant_ml.markets.demo_trial_records import decode_trial
from covenant_ml.markets.validation import read_document

from ._account_settlement_fixtures import final_market, native_credit
from ._demo_trial_fixtures import serve_reconciliation, serve_submission
from ._server import ApiServer

PREFIX = "/trade-api/v2/portfolio/"


def fixed(value: int, digits: int) -> str:
    """Render nonnegative integer test evidence without floating-point rounding."""
    scale = {2: 100, 4: 10000, 6: 1000000}[digits]
    return f"{value // scale}.{value % scale:0{digits}d}"


def serve_variable_account(
    server: ApiServer, root: Path, price: int, quantity: int, *, submitting: bool
) -> None:
    """Serve two paginated fills and exact cash for a bounded one-contract order.

    Args:
        server: Real loopback HTTP server.
        root: Prepared trial containing the generated order identity.
        price: Purchase price in ten-thousandths of a dollar.
        quantity: Actual fill in hundredths of a contract, from 25 through 100.
        submitting: Whether preflight, quote and acknowledgement are also needed.
    """
    if submitting:
        serve_submission(server, root)
        book = require_dict(
            server.documents["/trade-api/v2/markets/MARKET-1/orderbook"][0], "orderbook_fp"
        )
        book["yes_dollars"] = [[fixed(price // 2, 4), "1.00"]]
        book["no_dollars"] = [[fixed(10000 - price, 4), "1.00"]]
        server.documents[PREFIX + "events/orders"][0]["fill_count"] = fixed(quantity, 2)
    else:
        serve_reconciliation(server, root)
    intent = decode_trial(read_document(root / "trial.json"))["intent"]
    order = require_dict(
        {"row": require_list(server.documents[PREFIX + "orders"][-1], "orders")[0]}, "row"
    )
    order.update(
        status="executed" if quantity == 100 else "canceled",
        fill_count_fp=fixed(quantity, 2),
        yes_price_dollars=fixed(price, 4),
        taker_fill_cost_dollars=fixed(price * quantity // 100, 4),
        taker_fees_dollars="0.0147",
    )
    fills: list[JSONObject] = [
        {
            "fill_id": "fill-" + str(index),
            "order_id": order["order_id"],
            "ticker": intent["ticker"],
            "outcome_side": "yes",
            "book_side": "bid",
            "count_fp": fixed(count, 2),
            "yes_price_dollars": fixed(price, 4),
            "fee_cost": fixed(fee, 6),
            "created_time": order["last_update_time"],
        }
        for index, (count, fee) in enumerate(((1, 7300), (quantity - 1, 7400)))
    ]
    server.documents[PREFIX + "fills"] = [
        {"fills": [fills[0]], "cursor": "second-fill"},
        {"fills": [fills[1]], "cursor": ""},
    ]
    server.documents[PREFIX + "positions"][-1]["market_positions"] = [
        {"ticker": intent["ticker"], "position_fp": fixed(quantity, 2)}
    ]
    cash = 20000099 - price * quantity - 14700
    server.documents[PREFIX + "balance"][-1].update(
        balance=cash // 10000, balance_dollars=fixed(cash, 6)
    )


def serve_variable_credit(
    server: ApiServer, root: Path, price: int, quantity: int, payout: int
) -> None:
    """Serve one exact settled credit and cleared position for the supplied fill.

    Args:
        server: Real loopback HTTP server.
        root: Prepared trial directory.
        price: Purchase price in ten-thousandths of a dollar.
        quantity: Filled hundredths of a contract.
        payout: Settlement value in cents per whole contract.
    """
    serve_variable_account(server, root, price, quantity, submitting=False)
    credit = native_credit(payout=payout)
    credit.update(
        yes_count_fp=fixed(quantity, 2),
        yes_total_cost_dollars=fixed(price * quantity // 100, 4),
        fee_cost="0.0147",
        revenue=payout * quantity // 100,
    )
    server.documents[PREFIX + "settlements"][-1]["settlements"] = [credit]
    server.documents[PREFIX + "positions"][-1]["market_positions"] = []
    cash = 20000099 - price * quantity - 14700 + payout * quantity * 100
    server.documents[PREFIX + "balance"][-1].update(
        balance=cash // 10000, balance_dollars=fixed(cash, 6)
    )
    server.documents["/trade-api/v2/markets/MARKET-1"] = [{"market": final_market(payout=payout)}]
