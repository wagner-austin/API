"""Real socket execution, account-wide accounting and restart behavior for bounded batches."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
from uuid import UUID

import pytest
from platform_core.http_client import build_client
from platform_core.json_utils import dump_json_str, require_dict

from covenant_ml.markets.demo_batch import build_demo_batch
from covenant_ml.markets.execution_journal import read_execution_journal
from covenant_ml.markets.execution_records import OrderIntent
from covenant_ml.markets.submission import read_submission_evidence
from covenant_ml.markets.validation import read_document

from ._batch_fixtures import (
    PREFIX,
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


def test_bounded_socket_orders_partial_fills_cancellations_and_restart(
    api_server: ApiServer, tmp_path: Path
) -> None:
    count = 20
    _, signer = signing_key()
    root, previous, guards = tmp_path / "batch", tmp_path / "trial", tmp_path / "owners"
    identities = iter(UUID(int=index) for index in range(2, count + 2))
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
        plan = service.prepare(
            root, previous, tuple(f"MARKET-{index}" for index in range(2, count + 2)), 500
        )
        assert plan["tickers"] == [f"MARKET-{index}" for index in range(2, count + 2)]
        rows: tuple[OrderIntent, ...] = (baseline,)
        quantities: tuple[int, ...] = (100,)
        for index in range(2, count + 2):
            row, quantity = batch_intent(index), (100, 50, 0)[index % 3]
            serve_snapshot(api_server, account_document(rows, quantities))
            serve_batch_quote(api_server, row, quantity)
            rows, quantities = (*rows, row), (*quantities, quantity)
            serve_snapshot(api_server, account_document(rows, quantities))
            report = service.advance(root)
            assert report["attempts_started"] == index - 1
            assert report["confirmed_batch_orders"] == index - 1
            assert report["filled_batch_orders"] == sum(value > 0 for value in quantities[1:])
            assert report["canceled_batch_orders"] == sum(value < 100 for value in quantities[1:])
            assert report["exposure"] == sum(
                5 * value + (100 if value else 0) for value in quantities[1:]
            )
            assert report["batch_net_realized_profit"] == 0
            assert report["live_ready"] is False
            evidence = read_submission_evidence(root / "attempts" / f"{index - 1:03d}", row)
            assert (
                require_dict(evidence, "acknowledgement")["order_id"]
                == "exchange-" + row["client_order_id"]
            )
        assert report["can_reserve"] is False
        restarted = build_demo_batch(client, signer, account_clock, guards, origin=api_server.url)
        assert restarted.status(root) == report
        serve_snapshot(api_server, account_document(rows, quantities, settled=True))
        settled = restarted.reconcile(root)
        assert settled["settled_batch_orders"] == sum(value > 0 for value in quantities[1:])
        assert settled["exposure"] == 0
        assert settled["batch_net_realized_profit"] == sum(
            95 * value - 100 for value in quantities[1:] if value
        )
        serve_snapshot(api_server, account_document(rows, quantities, settled=True))
        assert restarted.advance(root) == settled
        assert len(api_server.bodies) == count + 1
        assert len(tuple(root.glob("account-*.json"))) == 2 * count + 2


def test_disconnect_cannot_resend_and_later_account_visibility_resolves_it(
    api_server: ApiServer, tmp_path: Path
) -> None:
    _, signer = signing_key()
    root, previous, guards = tmp_path / "batch", tmp_path / "trial", tmp_path / "owners"
    row = batch_intent(2)
    with closing(build_client(5)) as client:
        baseline = settled_predecessor(api_server, client, signer, previous)
        service = build_demo_batch(
            client,
            signer,
            account_clock,
            guards,
            origin=api_server.url,
            identity=lambda: UUID(int=2),
        )
        serve_credit(api_server, previous)
        service.prepare(root, previous, ("MARKET-2",), 500)
        serve_snapshot(api_server, account_document((baseline,), (100,)))
        serve_batch_quote(api_server, row, 100)
        api_server.disconnect_paths.add(PREFIX + "events/orders")
        from httpx import RemoteProtocolError

        with pytest.raises(RemoteProtocolError):
            service.advance(root)
        before = read_execution_journal(root / "execution.sqlite")
        assert before["orders"][1]["receipt"] is None
        assert service.status(root)["issues"] == ["EXECUTION-UNRESOLVED"]
        serve_snapshot(api_server, account_document((baseline,), (100,)))
        with pytest.raises(ValueError, match="DEMO-BATCH-001"):
            service.advance(root)
        assert read_execution_journal(root / "execution.sqlite") == before
        assert len(api_server.bodies) == 2
        api_server.disconnect_paths.clear()
        serve_snapshot(api_server, account_document((baseline, row), (100, 100)))
        recovered = service.advance(root)
        assert recovered["confirmed_batch_orders"] == 1
        assert recovered["attempts_started"] == 1
        assert recovered["can_reserve"] is False
        assert recovered["exposure"] == 600
        assert len(api_server.bodies) == 2


def test_refused_quote_consumes_attempt_without_post_or_reservation(
    api_server: ApiServer, tmp_path: Path
) -> None:
    _, signer = signing_key()
    root, previous = tmp_path / "batch", tmp_path / "trial"
    with closing(build_client(5)) as client:
        baseline = settled_predecessor(api_server, client, signer, previous)
        service = build_demo_batch(
            client, signer, account_clock, tmp_path / "owners", origin=api_server.url
        )
        serve_credit(api_server, previous)
        service.prepare(root, previous, ("MARKET-2",), 500)
        serve_snapshot(api_server, account_document((baseline,), (100,)))
        serve_batch_quote(api_server, batch_intent(2), 100)
        require_dict(
            api_server.documents["/trade-api/v2/markets/MARKET-2/orderbook"][0], "orderbook_fp"
        )["no_dollars"] = []
        with pytest.raises(ValueError, match="DEMO-TRIAL-006"):
            service.advance(root)
        report = service.status(root)
        assert report["attempts_started"] == 1
        assert report["reserved_batch_orders"] == 0
        assert report["can_reserve"] is False
        assert len(api_server.bodies) == 1


@pytest.mark.parametrize("changed_id", [False, True])
def test_partial_acknowledgement_persistence_is_verified_before_journal_update(
    api_server: ApiServer, tmp_path: Path, changed_id: bool
) -> None:
    _, signer = signing_key()
    root, previous = tmp_path / "batch", tmp_path / "trial"
    row = batch_intent(2)
    with closing(build_client(5)) as client:
        baseline = settled_predecessor(api_server, client, signer, previous)
        service = build_demo_batch(
            client,
            signer,
            account_clock,
            tmp_path / "owners",
            origin=api_server.url,
            identity=lambda: UUID(int=2),
        )
        serve_credit(api_server, previous)
        service.prepare(root, previous, ("MARKET-2",), 500)
        serve_snapshot(api_server, account_document((baseline,), (100,)))
        serve_batch_quote(api_server, row, 100)
        serve_snapshot(api_server, account_document((baseline, row), (100, 100)))
        service.advance(root)
        path = root / "attempts" / "001"
        (path / "submission-timing.json").unlink()
        if changed_id:
            document = read_document(path / "acknowledgement.json")
            document["order_id"] = "wrong-exchange-order"
            (path / "acknowledgement.json").write_text(dump_json_str(document), encoding="utf-8")
        before = read_execution_journal(root / "execution.sqlite")
        serve_snapshot(api_server, account_document((baseline, row), (100, 100), settled=True))
        if changed_id:
            with pytest.raises(ValueError, match="acknowledgement and account order disagree"):
                service.reconcile(root)
            assert read_execution_journal(root / "execution.sqlite") == before
        else:
            report = service.reconcile(root)
            assert report["settled_batch_orders"] == 1
            assert report["batch_net_realized_profit"] == 9400
        assert len(api_server.bodies) == 2
