"""Real HTTP, SQLite and account cash transitions through eventual settlement."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

import pytest
from platform_core.http_client import build_client
from platform_core.json_utils import require_dict

from covenant_ml.markets.demo_trial import build_demo_trial
from covenant_ml.markets.validation import read_document

from ._account_settlement_fixtures import final_market, native_credit
from ._demo_trial_fixtures import serve_reconciliation, serve_submission
from ._kalshi_fixtures import account_clock, serve_account, signing_key
from ._server import ApiServer, api_server

__all__ = ["api_server"]


def serve_credit(server: ApiServer, root: Path, payout: int = 100) -> None:
    serve_reconciliation(server, root)
    prefix = "/trade-api/v2/portfolio/"
    server.documents[prefix + "settlements"][-1]["settlements"] = [native_credit(payout=payout)]
    server.documents[prefix + "positions"][-1]["market_positions"] = []
    cash = server.documents[prefix + "balance"][-1]
    cash["balance"] = 1900 + payout
    cash["balance_dollars"] = f"{19 + payout // 100}.{payout % 100:02d}0099"
    server.documents["/trade-api/v2/markets/MARKET-1"] = [{"market": final_market(payout=payout)}]


@pytest.mark.parametrize("payout", [0, 37, 100])
def test_actual_socket_credit_cash_and_restart(
    api_server: ApiServer, tmp_path: Path, payout: int
) -> None:
    _, signer = signing_key()
    root = tmp_path / "trial"
    with closing(build_client(5)) as client:
        service = build_demo_trial(client, signer, account_clock, origin=api_server.url)
        serve_account(api_server)
        service.prepare(root, "MARKET-1", 9800)
        serve_submission(api_server, root)
        pending = service.submit(root)
        assert pending["confirmed_settlements"] == 0
        assert pending["exposure"] == 10000
        serve_credit(api_server, root, payout)
        report = service.reconcile(root)
        assert report["confirmed_settlements"] == 1
        assert report["exposure"] == 0
        assert report["net_realized_profit"] == payout * 100 - 10000
        assert report["gross_realized_loss"] == 10000 - payout * 100
        assert report["live_ready"] is False
        assert report["can_reserve"] is False
        restarted = build_demo_trial(client, signer, account_clock, origin=api_server.url)
        assert restarted.status(root) == report
        serve_credit(api_server, root, payout)
        assert restarted.reconcile(root) == report
        with pytest.raises(ValueError, match="DEMO-TRIAL-005"):
            restarted.submit(root)
        assert len(api_server.bodies) == 1
        captures = [read_document(path) for path in root.glob("reconciliation-*.json")]
        settled = [row for row in captures if row["market"] is not None]
        assert len(settled) == 2
        assert all(row["settlements"] == [native_credit(payout=payout)] for row in settled)


@pytest.mark.parametrize(
    "failure", ["cash", "position", "zero_position", "multiple", "unfinalized", "missing_credit"]
)
def test_settlement_disagreement_preserves_filled_exposure(
    api_server: ApiServer, tmp_path: Path, failure: str
) -> None:
    _, signer = signing_key()
    root = tmp_path / "trial"
    with closing(build_client(5)) as client:
        service = build_demo_trial(client, signer, account_clock, origin=api_server.url)
        serve_account(api_server)
        service.prepare(root, "MARKET-1", 9800)
        serve_submission(api_server, root)
        before = service.submit(root)
        serve_credit(api_server, root)
        prefix = "/trade-api/v2/portfolio/"
        code = "DEMO-TRIAL-007"
        if failure == "cash":
            api_server.documents[prefix + "balance"][0]["balance_dollars"] = "19.000099"
        elif failure in ("position", "zero_position"):
            api_server.documents[prefix + "positions"][0]["market_positions"] = [
                {"ticker": "MARKET-1", "position_fp": "1.00" if failure == "position" else "0.00"}
            ]
        elif failure == "multiple":
            api_server.documents[prefix + "settlements"][0]["settlements"] = [
                native_credit(),
                native_credit(),
            ]
            code = "DEMO-TRIAL-009"
        elif failure == "unfinalized":
            require_dict(api_server.documents["/trade-api/v2/markets/MARKET-1"][0], "market")[
                "status"
            ] = "determined"
            code = "ACCOUNT-SETTLEMENT-002"
        else:
            api_server.documents[prefix + "settlements"][0]["settlements"] = []
        with pytest.raises(ValueError, match=code):
            service.reconcile(root)
        assert service.status(root) == before
        assert len(api_server.bodies) == 1


def test_disappeared_settlement_cannot_erase_credited_history(
    api_server: ApiServer, tmp_path: Path
) -> None:
    _, signer = signing_key()
    root = tmp_path / "trial"
    with closing(build_client(5)) as client:
        service = build_demo_trial(client, signer, account_clock, origin=api_server.url)
        serve_account(api_server)
        service.prepare(root, "MARKET-1", 9800)
        serve_submission(api_server, root)
        service.submit(root)
        serve_credit(api_server, root)
        credited = service.reconcile(root)
        serve_reconciliation(api_server, root)
        with pytest.raises(ValueError, match="DEMO-TRIAL-009"):
            service.reconcile(root)
        assert service.status(root) == credited
        assert len(api_server.bodies) == 1
