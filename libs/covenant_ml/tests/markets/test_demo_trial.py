"""Real demo service, SQLite journal and HTTP socket lifecycle tests."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

import pytest
from httpx import RemoteProtocolError
from platform_core.http_client import build_client
from platform_core.json_utils import dump_json_str, require_dict, require_list

from covenant_ml.markets.demo_trial import build_demo_trial
from covenant_ml.markets.demo_trial_records import decode_trial
from covenant_ml.markets.execution_journal import execution_journal
from covenant_ml.markets.kalshi_transport import DemoTransport
from covenant_ml.markets.validation import read_document

from ._demo_trial_fixtures import serve_reconciliation, serve_submission
from ._execution_fixtures import intent
from ._kalshi_fixtures import account_clock, balance, serve_account, signing_key
from ._server import ApiServer, api_server

__all__ = ["api_server"]


def test_expiration_checked_at_http_boundary(api_server: ApiServer) -> None:
    _, signer = signing_key()
    with closing(build_client(5)) as client:
        transport = DemoTransport(client, signer, account_clock, origin=api_server.url)
        with pytest.raises(ValueError, match="DEMO-TRIAL-008"):
            transport.submit(intent(created=200000), valid_from_ms=200000, valid_until_ms=300000)
    assert api_server.requests == []


def test_actual_order_lifecycle_and_restart(api_server: ApiServer, tmp_path: Path) -> None:
    _, signer = signing_key()
    root = tmp_path / "trial"
    with closing(build_client(5)) as client:
        service = build_demo_trial(client, signer, account_clock, origin=api_server.url)
        serve_account(api_server)
        manifest = service.prepare(root, "MARKET-1", 9800)
        assert read_document(root / "trial.json") == manifest
        assert service.status(root)["submission_attempt_reserved"] is False
        serve_submission(api_server, root)
        report = service.submit(root)
        assert report["mode"] == "demo_single_order"
        assert report["confirmed_orders"] == 1
        assert report["exposure"] == 10000
        assert report["live_ready"] is False
        assert report["can_reserve"] is False
        assert "orders_sent" not in report
        trial = decode_trial(manifest)
        assert api_server.bodies == [
            {
                "ticker": "MARKET-1",
                "client_order_id": trial["intent"]["client_order_id"],
                "side": "bid",
                "count": "1.00",
                "price": "0.9800",
                "time_in_force": "immediate_or_cancel",
                "self_trade_prevention_type": "taker_at_cross",
                "cancel_order_on_pause": True,
                "post_only": False,
                "reduce_only": False,
                "subaccount": 0,
            }
        ]
        assert len(list(root.glob("reconciliation-*.json"))) == 1
        restarted = build_demo_trial(client, signer, account_clock, origin=api_server.url)
        assert restarted.status(root) == report
        with pytest.raises(ValueError, match="DEMO-TRIAL-005"):
            restarted.submit(root)
        serve_reconciliation(api_server, root)
        assert restarted.reconcile(root) == report
        assert len(api_server.bodies) == 1
        assert len(list(root.glob("reconciliation-*.json"))) == 2


def test_real_disconnect_retains_risk_and_reconciles_without_resending(
    api_server: ApiServer, tmp_path: Path
) -> None:
    _, signer = signing_key()
    root = tmp_path / "ambiguous"
    with closing(build_client(5)) as client:
        service = build_demo_trial(client, signer, account_clock, origin=api_server.url)
        serve_account(api_server)
        service.prepare(root, "MARKET-1", 9800)
        serve_submission(api_server, root)
        api_server.disconnect_paths.add("/trade-api/v2/portfolio/events/orders")
        with pytest.raises(RemoteProtocolError):
            service.submit(root)
        state = execution_journal().read(root / "execution.sqlite")
        assert len(state["orders"]) == 1
        assert state["orders"][0]["receipt"] is None
        assert not (root / "acknowledgement.json").exists()
        assert service.status(root)["issues"] == ["EXECUTION-UNRESOLVED"]
        assert service.status(root)["exposure"] == 10000
        with pytest.raises(ValueError, match="DEMO-TRIAL-005"):
            service.submit(root)
        assert service.reconcile(root)["confirmed_orders"] == 1
        assert len(api_server.bodies) == 1


def test_delayed_visibility_and_fractional_cent_fee_recover_by_get_only(
    api_server: ApiServer, tmp_path: Path
) -> None:
    """Reproduce the actual demo's delayed list response and $0.0147 fee."""
    _, signer = signing_key()
    root = tmp_path / "trial"
    with closing(build_client(5)) as client:
        service = build_demo_trial(client, signer, account_clock, origin=api_server.url)
        serve_account(api_server)
        service.prepare(root, "MARKET-1", 9800)
        serve_submission(api_server, root)
        prefix = "/trade-api/v2/portfolio/"
        api_server.documents[prefix + "orders"].insert(1, {"orders": [], "cursor": ""})
        order = require_dict(
            {"row": require_list(api_server.documents[prefix + "orders"][2], "orders")[0]},
            "row",
        )
        order["taker_fees_dollars"] = "0.014700"
        fill = require_dict(
            {"row": require_list(api_server.documents[prefix + "fills"][0], "fills")[0]}, "row"
        )
        fill["fee_cost"] = "0.014700"
        cash = api_server.documents[prefix + "balance"][2]
        cash["balance_dollars"] = "19.005399"
        with pytest.raises(ValueError, match="DEMO-TRIAL-007"):
            service.submit(root)
        assert (root / "acknowledgement.json").exists()
        assert service.status(root)["issues"] == ["EXECUTION-UNRESOLVED"]
        report = service.reconcile(root)
        assert report["confirmed_orders"] == 1
        assert report["exposure"] == 9947
        assert report["issues"] == []
        assert len(api_server.bodies) == 1


def test_ioc_zero_fill_cancellation_releases_cost_but_never_rearms_trial(
    api_server: ApiServer, tmp_path: Path
) -> None:
    _, signer = signing_key()
    root = tmp_path / "trial"
    with closing(build_client(5)) as client:
        service = build_demo_trial(client, signer, account_clock, origin=api_server.url)
        serve_account(api_server)
        service.prepare(root, "MARKET-1", 9800)
        serve_submission(api_server, root)
        prefix = "/trade-api/v2/portfolio/"
        api_server.documents[prefix + "events/orders"][0]["fill_count"] = "0.00"
        order = require_dict(
            {"row": require_list(api_server.documents[prefix + "orders"][1], "orders")[0]}, "row"
        )
        order.update(
            status="canceled",
            fill_count_fp="0.00",
            taker_fill_cost_dollars="0.0000",
            taker_fees_dollars="0.0000",
        )
        api_server.documents[prefix + "fills"][0]["fills"] = []
        api_server.documents[prefix + "positions"][1]["market_positions"] = []
        api_server.documents[prefix + "balance"][2] = balance()
        report = service.submit(root)
        assert report["confirmed_orders"] == 1
        assert report["exposure"] == 0
        assert report["issues"] == []
        assert report["can_reserve"] is False
        with pytest.raises(ValueError, match="DEMO-TRIAL-005"):
            service.submit(root)
        assert len(api_server.bodies) == 1


@pytest.mark.parametrize("failure", ["balance", "market", "book", "size", "closed", "ticker"])
def test_admission_failures_never_reserve_or_send(
    api_server: ApiServer, tmp_path: Path, failure: str
) -> None:
    _, signer = signing_key()
    with closing(build_client(5)) as client:
        service = build_demo_trial(client, signer, account_clock, origin=api_server.url)
        serve_account(api_server)
        service.prepare(tmp_path / "trial", "MARKET-1", 9800)
        root = tmp_path / "trial"
        serve_submission(api_server, root)
        code = "DEMO-TRIAL-006"
        if failure == "balance":
            for document in api_server.documents["/trade-api/v2/portfolio/balance"][:2]:
                document["balance"] = 2100
                document["balance_dollars"] = "21.000099"
            code = "DEMO-TRIAL-005"
        elif failure in ("market", "closed", "ticker"):
            market = require_dict(
                api_server.documents["/trade-api/v2/markets/MARKET-1"][0], "market"
            )
            if failure == "market":
                market["status"] = "closed"
            elif failure == "ticker":
                market["ticker"] = "OTHER"
            else:
                market["close_time"] = "1970-01-01T00:00:01Z"
        else:
            book = require_dict(
                api_server.documents["/trade-api/v2/markets/MARKET-1/orderbook"][0], "orderbook_fp"
            )
            book["no_dollars"] = [] if failure == "book" else [["0.0200", "0.50"]]
        with pytest.raises(ValueError, match=code):
            service.submit(root)
        assert api_server.bodies == []
        assert execution_journal().read(root / "execution.sqlite")["orders"] == ()


def test_credentials_and_missing_reservation_refused(api_server: ApiServer, tmp_path: Path) -> None:
    _, signer = signing_key()
    with closing(build_client(5)) as client:
        service = build_demo_trial(client, signer, account_clock, origin=api_server.url)
        serve_account(api_server)
        service.prepare(tmp_path / "trial", "MARKET-1", 9800)
        with pytest.raises(ValueError, match="DEMO-TRIAL-007"):
            service.reconcile(tmp_path / "trial")
        _, other = signing_key()
        wrong = build_demo_trial(client, other, account_clock, origin=api_server.url)
        with pytest.raises(ValueError, match="DEMO-TRIAL-004"):
            wrong.status(tmp_path / "trial")
        serve_account(api_server)
        with pytest.raises(FileExistsError):
            service.prepare(tmp_path / "trial", "MARKET-1", 9800)


@pytest.mark.parametrize(
    "failure", ["no_order", "foreign", "resting", "ack", "position", "cash", "count", "duplicate"]
)
def test_reconciliation_failures_keep_reservation(
    api_server: ApiServer, tmp_path: Path, failure: str
) -> None:
    _, signer = signing_key()
    root = tmp_path / "trial"
    with closing(build_client(5)) as client:
        service = build_demo_trial(client, signer, account_clock, origin=api_server.url)
        serve_account(api_server)
        service.prepare(root, "MARKET-1", 9800)
        serve_submission(api_server, root)
        api_server.statuses["/trade-api/v2/portfolio/events/orders"] = 500
        with pytest.raises(ValueError, match="KALSHI-DEMO-003"):
            service.submit(root)
        orders = api_server.documents["/trade-api/v2/portfolio/orders"][0]
        row = require_dict({"row": require_list(orders, "orders")[0]}, "row")
        positions = api_server.documents["/trade-api/v2/portfolio/positions"][0]
        if failure == "no_order":
            orders["orders"] = []
        elif failure == "foreign":
            row["client_order_id"] = "another-order"
        elif failure == "resting":
            row.update(
                status="resting",
                fill_count_fp="0.00",
                remaining_count_fp="1.00",
                taker_fill_cost_dollars="0.0000",
                taker_fees_dollars="0.0000",
            )
        elif failure == "ack":
            ack = {
                "client_order_id": row["client_order_id"],
                "order_id": "different",
                "fill_count": "1.00",
                "remaining_count": "0.00",
                "ts_ms": 120000,
            }
            (root / "acknowledgement.json").write_text(dump_json_str(ack), encoding="utf-8")
        elif failure == "position":
            positions["market_positions"] = [{"ticker": "OTHER", "position_fp": "1.00"}]
        elif failure == "count":
            positions["market_positions"] = []
        elif failure == "duplicate":
            positions["market_positions"] = [
                {"ticker": "MARKET-1", "position_fp": "0.50"},
                {"ticker": "MARKET-1", "position_fp": "0.50"},
            ]
        else:
            api_server.documents["/trade-api/v2/portfolio/balance"][0]["balance_dollars"] = (
                "19.010099"
            )
        with pytest.raises(ValueError, match="DEMO-TRIAL-007"):
            service.reconcile(root)
        assert service.status(root)["issues"] == ["EXECUTION-UNRESOLVED"]
        assert service.status(root)["exposure"] == 10000
        assert len(api_server.bodies) == 1
