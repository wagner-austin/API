"""Complete demo account evidence reconciled against every known reservation."""

from __future__ import annotations

from collections.abc import Callable
from urllib.parse import quote

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    require_dict,
    require_list,
    require_str,
)

from .account_settlement import decode_native_account_settlement, encode_account_settlement
from .collection import MarketCollector
from .demo_trial_records import verify_fills
from .execution_policy import apply_execution_event
from .execution_records import OrderStatus, decode_native_order, encode_receipt
from .execution_state import ExecutionState
from .kalshi_demo import decode_native_balance
from .kalshi_transport import DemoTransport
from .validation import parse_fixed, parse_timestamp, require_bounded_int


def _require_balances(document: JSONObject, minimum_ms: int) -> tuple[int, int, JSONObject]:
    """Require stable exact cash and a bounded chronological account collection.

    Args:
        document: Native balances and local collection timestamps.
        minimum_ms: Last accepted journal observation time.

    Returns:
        Completion time, cash microdollars and final native balance.

    Raises:
        ValueError: DEMO-BATCH-002 for changing cash; KALSHI-DEMO-006 for
            inconsistent monetary representations. Time validators propagate.
    """
    started = require_bounded_int(document, "started_ms", minimum_ms, 2**63 - 1)
    observed = require_bounded_int(document, "completed_ms", started, started + 30000)
    balances = tuple(require_dict(document, key) for key in ("first_balance", "last_balance"))
    money = tuple(decode_native_balance(row)[0] for row in balances)
    for balance in balances:
        require_bounded_int(balance, "updated_ts", 0, observed // 1000)
    if money[0] != money[1]:
        raise ValueError("[DEMO-BATCH-002] cash changed during account collection")
    return observed, money[1], balances[1]


def account_snapshot_event(
    document: JSONObject, state: ExecutionState, initial_cash: int
) -> JSONObject:
    """Validate a complete account snapshot before producing an atomic journal event.

    Args:
        document: Native orders, fills, positions, settlements, markets and balances.
        state: Replayed journal containing every allowed account order.
        initial_cash: Account microdollars before the earliest journaled order.

    Returns:
        Atomic account_snapshot event containing verified receipts and credits.

    Raises:
        ValueError: DEMO-BATCH-001 for incomplete identity sets; DEMO-BATCH-002
            for changing cash or collection times; DEMO-BATCH-003 for accounting
            disagreement. Native validators retain their specific failure codes.
    """
    observed, cash, last_balance = _require_balances(document, state["observed_ms"])
    orders = tuple(
        decode_native_order(narrow_json_to_dict(value))
        for value in require_list(document, "orders")
    )
    by_client = {order["client_order_id"]: order for order in orders}
    exchange_ids = {order["order_id"] for order in orders}
    expected_ids = {order["intent"]["client_order_id"] for order in state["orders"]}
    if (
        not expected_ids
        or set(by_client) != expected_ids
        or len(by_client) != len(orders)
        or len(exchange_ids) != len(orders)
        or len({order["ticker"] for order in orders}) != len(orders)
    ):
        raise ValueError("[DEMO-BATCH-001] require exactly the known distinct account orders")
    fills = tuple(narrow_json_to_dict(value) for value in require_list(document, "fills"))
    if len({require_str(fill, "fill_id") for fill in fills}) != len(fills) or any(
        require_str(fill, "order_id") not in exchange_ids for fill in fills
    ):
        raise ValueError("[DEMO-BATCH-001] duplicate or unknown account fills")
    credits = tuple(narrow_json_to_dict(value) for value in require_list(document, "settlements"))
    settlements = {require_str(row, "ticker"): row for row in credits}
    native_markets = tuple(
        narrow_json_to_dict(value) for value in require_list(document, "markets")
    )
    markets = {require_str(row, "ticker"): row for row in native_markets}
    if (
        len(settlements) != len(credits)
        or len(markets) != len(native_markets)
        or set(markets) != set(settlements)
        or not set(settlements) <= {order["ticker"] for order in orders}
    ):
        raise ValueError("[DEMO-BATCH-001] duplicate or unknown settlement markets")
    events: list[JSONValue] = []
    expected_positions: dict[str, int] = {}
    spent = revenue = 0
    candidate = state
    for tracked in state["orders"]:
        intent = tracked["intent"]
        receipt = by_client[intent["client_order_id"]]
        if receipt["status"] is OrderStatus.RESTING or tracked["settlement"] is not None:
            raise ValueError("[DEMO-BATCH-003] require terminal orders and authenticated credits")
        verify_fills(
            tuple(fill for fill in fills if require_str(fill, "order_id") == receipt["order_id"]),
            receipt,
            intent,
        )
        event: JSONObject = {
            "action": "receipt",
            "observed_ms": observed,
            "client_order_id": intent["client_order_id"],
            "data": encode_receipt(receipt),
        }
        candidate = apply_execution_event(candidate, event)
        events.append(event)
        spent += receipt["fill_cost"] + receipt["fees"]
        ticker = intent["ticker"]
        if ticker in settlements:
            credit = decode_native_account_settlement(
                settlements[ticker], markets[ticker], receipt, observed
            )
            credited: JSONObject = {
                "action": "account_settlement",
                "observed_ms": observed,
                "client_order_id": intent["client_order_id"],
                "data": encode_account_settlement(credit),
            }
            candidate = apply_execution_event(candidate, credited)
            events.append(credited)
            revenue += credit["revenue"]
        elif tracked["account_settlement"] is not None:
            raise ValueError("[DEMO-BATCH-003] a previously credited settlement disappeared")
        elif receipt["filled_count"]:
            expected_positions[ticker] = receipt["filled_count"]
    native_positions = tuple(
        narrow_json_to_dict(value) for value in require_list(document, "positions")
    )
    positions = {
        require_str(row, "ticker"): parse_fixed(require_str(row, "position_fp"), 2)
        for row in native_positions
    }
    if (
        len(positions) != len(native_positions)
        or positions != expected_positions
        or cash != initial_cash - spent * 100 + revenue * 100
        or (
            not positions
            and require_bounded_int(last_balance, "portfolio_value", 0, 2**63 - 1) != 0
        )
    ):
        raise ValueError("[DEMO-BATCH-003] account cash or positions disagree with all orders")
    return {"action": "account_snapshot", "observed_ms": observed, "events": events}


class DemoAccountReconciler:
    """Collect every account page through the real demo transport and validate at the edge."""

    def __init__(self, transport: DemoTransport, clock: Callable[[], str]) -> None:
        """Bind the demo boundary and explicit aware clock.

        Args:
            transport: Signed demo-only HTTP service.
            clock: Aware local observation clock.
        """
        self._transport, self._clock = transport, clock

    def collect(self, state: ExecutionState, initial_cash: int) -> tuple[JSONObject, JSONObject]:
        """Read and verify all account evidence without posting an order.

        Args:
            state: Complete known account journal.
            initial_cash: Exact cash before the earliest journaled order.

        Returns:
            Native evidence document and validated atomic reconciliation event.

        Raises:
            ValueError: Incomplete, stale, foreign or contradictory evidence.
            HTTPError: Required evidence cannot be read; no retry is attempted.
        """
        started = parse_timestamp(self._clock())
        first = self._transport.get("/portfolio/balance", {"subaccount": 0})
        collector = MarketCollector(self._transport.get)
        orders = tuple(collector.pages("/portfolio/orders", {"subaccount": 0}, "orders"))
        fills = tuple(collector.pages("/portfolio/fills", {"subaccount": 0}, "fills"))
        settlements = tuple(
            collector.pages("/portfolio/settlements", {"subaccount": 0}, "settlements")
        )
        markets = tuple(
            require_dict(
                self._transport.get("/markets/" + quote(require_str(row, "ticker"), safe=""), {}),
                "market",
            )
            for row in settlements
        )
        positions = tuple(
            collector.pages(
                "/portfolio/positions",
                {"subaccount": 0, "settlement_status": "unsettled"},
                "market_positions",
            )
        )
        last = self._transport.get("/portfolio/balance", {"subaccount": 0})
        document: JSONObject = {
            "started_ms": started,
            "completed_ms": parse_timestamp(self._clock()),
            "first_balance": first,
            "last_balance": last,
            "orders": list(orders),
            "fills": list(fills),
            "positions": list(positions),
            "settlements": list(settlements),
            "markets": list(markets),
        }
        return document, account_snapshot_event(document, state, initial_cash)
