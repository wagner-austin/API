"""Native one-order account fixtures served over actual sockets."""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import JSONObject

from covenant_ml.markets.demo_trial_records import decode_trial
from covenant_ml.markets.validation import read_document

from ._execution_fixtures import native_order
from ._kalshi_fixtures import balance, serve_account
from ._server import ApiServer


def serve_submission(server: ApiServer, root: Path) -> None:
    trial = decode_trial(read_document(root / "trial.json"))
    intent = trial["intent"]
    serve_account(server)
    server.documents["/trade-api/v2/markets/" + intent["ticker"]] = [
        {
            "market": {
                "ticker": intent["ticker"],
                "status": "active",
                "close_time": "2099-01-01T00:00:00Z",
            }
        }
    ]
    server.documents["/trade-api/v2/markets/" + intent["ticker"] + "/orderbook"] = [
        {"orderbook_fp": {"yes_dollars": [["0.9000", "2.00"]], "no_dollars": [["0.0200", "1.00"]]}}
    ]
    endpoint = "/trade-api/v2/portfolio/events/orders"
    server.statuses[endpoint] = 201
    server.documents[endpoint] = [
        {
            "client_order_id": intent["client_order_id"],
            "order_id": "exchange-" + intent["client_order_id"],
            "fill_count": "1.00",
            "remaining_count": "0.00",
            "ts_ms": intent["created_ms"],
        }
    ]
    serve_reconciliation(server, root)


def serve_reconciliation(server: ApiServer, root: Path) -> None:
    trial = decode_trial(read_document(root / "trial.json"))
    intent = trial["intent"]
    receipt = native_order(intent)
    fills: JSONObject = {
        "fills": [
            {
                "fill_id": "actual-fill",
                "order_id": receipt["order_id"],
                "ticker": intent["ticker"],
                "outcome_side": "yes",
                "book_side": "bid",
                "count_fp": "1.00",
                "yes_price_dollars": "0.9800",
                "fee_cost": "0.020000",
                "created_time": receipt["last_update_time"],
            }
        ],
        "cursor": "",
    }
    positions: JSONObject = {
        "market_positions": [{"ticker": intent["ticker"], "position_fp": "1.00"}],
        "cursor": "",
    }
    cash = balance()
    cash["balance"] = 1900
    cash["balance_dollars"] = "19.000099"
    documents: tuple[tuple[str, JSONObject], ...] = (
        ("orders", {"orders": [receipt], "cursor": ""}),
        ("fills", fills),
        ("positions", positions),
        ("balance", cash),
    )
    for endpoint, document in documents:
        server.documents.setdefault("/trade-api/v2/portfolio/" + endpoint, []).append(document)
