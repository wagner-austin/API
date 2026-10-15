"""Frozen attempt counts never override account exposure, cumulative loss or expiry."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
from uuid import UUID

from platform_core.http_client import build_client

from covenant_ml.markets.demo_batch import build_demo_batch
from covenant_ml.markets.execution_records import OrderIntent, encode_intent
from covenant_ml.markets.execution_storage import persist_evidence

from ._batch_fixtures import (
    account_document,
    batch_intent,
    serve_batch_quote,
    serve_snapshot,
    settled_predecessor,
)
from ._kalshi_fixtures import account_clock, signing_key
from ._server import ApiServer, api_server
from .test_demo_settlement import serve_credit

__all__ = ["api_server"]


def test_larger_price_never_weakens_exposure_ceiling_or_expiry(
    api_server: ApiServer, tmp_path: Path
) -> None:
    _, signer = signing_key()
    root, previous, guards = tmp_path / "batch", tmp_path / "trial", tmp_path / "owners"
    identities = iter(UUID(int=index) for index in range(2, 22))
    with closing(build_client(5)) as client:
        baseline = settled_predecessor(api_server, client, signer, previous)
        service = build_demo_batch(
            client,
            signer,
            account_clock,
            guards,
            origin=api_server.url,
            identity=identities.__next__,
        )
        serve_credit(api_server, previous)
        service.prepare(root, previous, tuple(f"MARKET-{index}" for index in range(2, 22)), 5500)
        rows: tuple[OrderIntent, ...] = (baseline,)
        for index in range(2, 6):
            serve_snapshot(api_server, account_document(rows, (100,) * len(rows)))
            row = batch_intent(index, 5500)
            serve_batch_quote(api_server, row, 100)
            rows = (*rows, row)
            serve_snapshot(api_server, account_document(rows, (100,) * len(rows)))
            report = service.advance(root)
            assert report["exposure"] == (index - 1) * 5600
        assert report["attempts_started"] == 4
        assert report["can_reserve"] is False
        serve_snapshot(api_server, account_document(rows, (100,) * len(rows)))
        assert service.advance(root) == report
        assert len(api_server.bodies) == 5
        credited = account_document(rows, (100,) * len(rows), settled=True)
        serve_snapshot(api_server, credited)
        settled = service.reconcile(root)
        assert settled["exposure"] == 0
        assert settled["can_reserve"] is True
        assert settled["attempts_started"] == 4
        assert settled["batch_net_realized_profit"] == 17600
        expired = build_demo_batch(
            client, signer, lambda: "1970-01-03T00:02:00Z", guards, origin=api_server.url
        )
        serve_snapshot(api_server, credited)
        final = expired.advance(root)
        assert final["expired"] is True
        assert final["can_reserve"] is False
        assert final["attempts_started"] == 4
        assert len(api_server.bodies) == 5


def test_hundred_consumed_attempts_cannot_be_reset_by_restart(
    api_server: ApiServer, tmp_path: Path
) -> None:
    _, signer = signing_key()
    root, previous, guards = tmp_path / "batch", tmp_path / "trial", tmp_path / "owners"
    with closing(build_client(5)) as client:
        baseline = settled_predecessor(api_server, client, signer, previous)
        service = build_demo_batch(client, signer, account_clock, guards, origin=api_server.url)
        serve_credit(api_server, previous)
        service.prepare(root, previous, tuple(f"MARKET-{index}" for index in range(2, 102)), 500)
        for index in range(1, 101):
            path = root / "attempts" / f"{index:03d}"
            path.mkdir()
            persist_evidence(path / "intent.json", encode_intent(batch_intent(index + 1)))
        serve_snapshot(api_server, account_document((baseline,), (100,)))
        restarted = build_demo_batch(client, signer, account_clock, guards, origin=api_server.url)
        report = restarted.advance(root)
        assert report["attempt_limit"] == report["attempts_started"] == 100
        assert report["reserved_batch_orders"] == report["confirmed_batch_orders"] == 0
        assert report["can_reserve"] is False
        assert len(api_server.bodies) == 1


def test_settled_losses_prevent_new_orders_even_with_zero_current_exposure(
    api_server: ApiServer, tmp_path: Path
) -> None:
    _, signer = signing_key()
    root, previous = tmp_path / "batch", tmp_path / "trial"
    identities = iter(UUID(int=index) for index in range(2, 22))
    with closing(build_client(5)) as client:
        baseline = settled_predecessor(api_server, client, signer, previous)
        service = build_demo_batch(
            client,
            signer,
            account_clock,
            tmp_path / "owners",
            origin=api_server.url,
            identity=identities.__next__,
        )
        serve_credit(api_server, previous)
        service.prepare(root, previous, tuple(f"MARKET-{index}" for index in range(2, 22)), 5500)
        rows: tuple[OrderIntent, ...] = (baseline,)
        for index in range(2, 10):
            serve_snapshot(
                api_server, account_document(rows, (100,) * len(rows), settled=True, payout=0)
            )
            row = batch_intent(index, 5500)
            serve_batch_quote(api_server, row, 100)
            rows = (*rows, row)
            serve_snapshot(
                api_server, account_document(rows, (100,) * len(rows), settled=True, payout=0)
            )
            report = service.advance(root)
            assert report["exposure"] == 0
            assert report["gross_realized_loss"] == (index - 1) * 5600
        assert report["attempts_started"] == 8
        assert report["can_reserve"] is False
        assert report["gross_realized_loss"] == 44800
        assert report["remaining_loss_allowance"] == 5200
        serve_snapshot(
            api_server, account_document(rows, (100,) * len(rows), settled=True, payout=0)
        )
        assert service.advance(root) == report
        assert len(api_server.bodies) == 9
