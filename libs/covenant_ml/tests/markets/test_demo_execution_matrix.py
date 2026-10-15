"""One hundred distinct socket lifecycles, including rejected accounting corruption."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

import pytest
from platform_core.http_client import build_client
from platform_core.json_utils import require_dict, require_list, require_str

from covenant_ml.markets.demo_trial import build_demo_trial
from covenant_ml.markets.execution_journal import read_execution_journal
from covenant_ml.markets.execution_records import OrderReceipt, OrderStatus

from ._demo_matrix_fixtures import PREFIX, serve_variable_account, serve_variable_credit
from ._kalshi_fixtures import account_clock, serve_account, signing_key
from ._server import ApiServer, api_server

__all__ = ["api_server"]


@pytest.mark.parametrize("price", [100, 1000, 3000, 5000, 9800])
@pytest.mark.parametrize("quantity", [25, 50, 75, 100])
@pytest.mark.parametrize("payout", [0, 20, 40, 60, 100])
def test_execution_matrix_rejects_corruption_and_replays_exactly(
    api_server: ApiServer, tmp_path: Path, price: int, quantity: int, payout: int
) -> None:
    """Check independent integer expectations through real HTTP and persisted state."""
    _, signer = signing_key()
    root = tmp_path / "trial"
    with closing(build_client(5)) as client:
        service = build_demo_trial(client, signer, account_clock, origin=api_server.url)
        serve_account(api_server)
        service.prepare(root, "MARKET-1", price)
        serve_variable_account(api_server, root, price, quantity, submitting=True)
        filled = service.submit(root)
        spent = price * quantity // 100 + 147
        assert filled["exposure"] == spent
        assert filled["confirmed_orders"] == 1
        assert filled["confirmed_settlements"] == 0
        assert filled["issues"] == []
        assert len(api_server.bodies) == 1
        assert api_server.bodies[0]["count"] == "1.00"
        assert api_server.bodies[0]["time_in_force"] == "immediate_or_cancel"
        assert (
            api_server.requests.count(PREFIX + "fills?subaccount=0&limit=1000&cursor=second-fill")
            == 1
        )
        state = read_execution_journal(root / "execution.sqlite")
        evidence = service.evidence(root)
        assert require_dict(evidence, "acknowledgement")["remaining_count"] == "0.00"
        assert state["orders"][0]["receipt"] == OrderReceipt(
            client_order_id=require_str(api_server.bodies[0], "client_order_id"),
            order_id=require_str(require_dict(evidence, "acknowledgement"), "order_id"),
            ticker="MARKET-1",
            status=OrderStatus.EXECUTED if quantity == 100 else OrderStatus.CANCELED,
            initial_count=100,
            filled_count=quantity,
            remaining_count=0,
            limit_price=price,
            fill_cost=price * quantity // 100,
            fees=147,
            updated_ms=120000,
        )

        # A one-microdollar independent fill error must not change persisted state.
        serve_variable_credit(api_server, root, price, quantity, payout)
        fill = require_dict(
            {"row": require_list(api_server.documents[PREFIX + "fills"][1], "fills")[0]},
            "row",
        )
        fill["fee_cost"] = "0.007401"
        with pytest.raises(ValueError, match="DEMO-TRIAL-003"):
            service.reconcile(root)
        assert read_execution_journal(root / "execution.sqlite") == state
        api_server.documents.clear()

        # Settlement cannot release exposure if the credited account cash disagrees.
        serve_variable_credit(api_server, root, price, quantity, payout)
        api_server.documents[PREFIX + "balance"][-1]["balance_dollars"] = "99.000000"
        with pytest.raises(ValueError, match="DEMO-TRIAL-007"):
            service.reconcile(root)
        assert read_execution_journal(root / "execution.sqlite") == state
        api_server.documents.clear()

        serve_variable_credit(api_server, root, price, quantity, payout)
        settled = service.reconcile(root)
        expected_profit = payout * quantity - spent
        assert settled["exposure"] == 0
        assert settled["net_realized_profit"] == expected_profit
        assert settled["gross_realized_loss"] == max(0, -expected_profit)
        assert settled["confirmed_settlements"] == 1
        assert settled["issues"] == []
        assert settled["live_ready"] is False
        assert settled["can_reserve"] is False
        restarted = build_demo_trial(client, signer, account_clock, origin=api_server.url)
        assert restarted.status(root) == settled
        assert restarted.evidence(root) == evidence
        serve_variable_credit(api_server, root, price, quantity, payout)
        assert restarted.reconcile(root) == settled
        with pytest.raises(ValueError, match="DEMO-TRIAL-005"):
            restarted.submit(root)
        assert len(api_server.bodies) == 1
        assert len(list(root.glob("reconciliation-*.json"))) == 3
