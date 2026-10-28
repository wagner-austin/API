"""Durable single-order demo execution, with no retry or second-order admission."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

from platform_core.http_client import HttpxClient
from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    require_dict,
    require_str,
)

from .account_settlement import (
    AccountSettlement,
    decode_native_account_settlement,
    encode_account_settlement,
)
from .collection import MarketCollector
from .demo_trial_records import (
    DemoTrial,
    decode_trial,
    encode_trial,
    validate_acknowledgement,
    verify_fills,
)
from .execution_journal import execution_journal
from .execution_records import (
    ORDER_LIMIT,
    OrderReceipt,
    OrderStatus,
    decode_intent,
    decode_native_order,
    encode_receipt,
)
from .execution_state import ExecutionState, execution_report
from .kalshi_auth import KalshiSigner
from .kalshi_demo import DemoAccountClient
from .kalshi_transport import DEMO_ORIGIN, DemoTransport
from .submission import read_submission_evidence, submit_reserved_order
from .validation import parse_fixed, parse_timestamp, read_document, require_bounded_int


class DemoTrialService:
    """Compose authenticated account reads, demo transport and the existing risk journal."""

    def __init__(
        self,
        account: DemoAccountClient,
        transport: DemoTransport,
        fingerprint: str,
        clock: Callable[[], str],
    ) -> None:
        """Bind focused services and an explicit credential identity.

        Args:
            account: Empty-account preflight service.
            transport: Demo-only authenticated HTTP boundary.
            fingerprint: Signer credential fingerprint.
            clock: Aware ISO wall-clock reader.
        """
        self._account, self._transport = account, transport
        self._fingerprint, self._clock = fingerprint, clock

    def prepare(self, root: Path, ticker: str, limit_price: int) -> JSONObject:
        """Exclusively create one funded demo trial without sending an order.

        Args:
            root: New trial directory whose parent must exist.
            ticker: Demo contract selected for execution mechanics testing.
            limit_price: Maximum YES price in 1/10000 dollars.

        Returns:
            Validated immutable manifest saved beside a new journal.

        Raises:
            FileExistsError: This trial directory already exists.
            ValueError: Account, intent or mock balance validation fails.
        """
        initial = self._account.preflight()
        created = parse_timestamp(self._clock())
        intent = decode_intent(
            {
                "client_order_id": str(uuid4()),
                "ticker": ticker,
                "contracts": 1,
                "limit_price": limit_price,
                "fee_reserve": ORDER_LIMIT - limit_price,
                "created_ms": created,
                "expires_ms": created + 300000,
            }
        )
        document = encode_trial(DemoTrial(intent=intent, initial=initial))
        decode_trial(document)
        root.mkdir()
        execution_journal().initialize(root / "execution.sqlite")
        with (root / "trial.json").open("x", encoding="utf-8") as stream:
            stream.write(dump_json_str(document))
        return encode_trial(self._load(root))

    def _load(self, root: Path) -> DemoTrial:
        """Read and bind the immutable manifest to the current credentials.

        Args:
            root: Existing demo trial directory.

        Returns:
            Validated DemoTrial.

        Raises:
            ValueError: DEMO-TRIAL-004 for changed credentials.
            OSError: Required manifest is unavailable.
        """
        trial = decode_trial(read_document(root / "trial.json"))
        if trial["initial"]["credential_fingerprint"] != self._fingerprint:
            raise ValueError("[DEMO-TRIAL-004] trial belongs to different demo credentials")
        return trial

    def submit(self, root: Path) -> JSONObject:
        """Reserve risk durably, send once, then reconcile actual account evidence.

        A timeout or malformed response leaves the reservation unresolved. Calling
        this method again is always refused after that reservation, even when no
        order is visible yet. Use reconcile to investigate without resending.

        Args:
            root: Prepared trial directory.

        Returns:
            Reconciled execution report with live readiness false.

        Raises:
            ValueError: DEMO-TRIAL-005 for repeat submission or changed balance;
                DEMO-TRIAL-006 for unavailable, stale or insufficient liquidity.
            HTTPError: Transport failure leaves reserved risk in the journal.
            OSError: Evidence cannot be persisted; failures propagate.
        """
        trial = self._load(root)
        ledger = root / "execution.sqlite"
        if execution_journal().read(ledger)["orders"]:
            raise ValueError("[DEMO-TRIAL-005] trial already reserved; reconcile without resending")
        current = self._account.preflight()
        if current["balance_microdollars"] != trial["initial"]["balance_microdollars"]:
            raise ValueError("[DEMO-TRIAL-005] account cash changed after trial preparation")
        submit_reserved_order(
            self._transport, self._clock, root, ledger, trial["intent"], current["completed_ms"]
        )
        return self.reconcile(root)

    def reconcile(self, root: Path) -> JSONObject:
        """Resolve the reserved UUID through authenticated orders, fills, position and cash.

        Args:
            root: Trial whose order was reserved before an attempted submission.

        Returns:
            Execution report with independently verified account credits when settled.

        Raises:
            ValueError: DEMO-TRIAL-007 for missing, foreign or inconsistent account evidence.
            HTTPError: Account evidence could not be collected.
        """
        trial = self._load(root)
        ledger = root / "execution.sqlite"
        state = execution_journal().read(ledger)
        intent = trial["intent"]
        if len(state["orders"]) != 1 or state["orders"][0]["intent"] != intent:
            raise ValueError("[DEMO-TRIAL-007] require exactly the manifest's reserved intent")
        collector = MarketCollector(self._transport.get)
        orders = tuple(collector.pages("/portfolio/orders", {"subaccount": 0}, "orders"))
        if len(orders) != 1:
            raise ValueError("[DEMO-TRIAL-007] require exactly one visible demo account order")
        receipt = decode_native_order(orders[0])
        if (
            receipt["client_order_id"] != intent["client_order_id"]
            or receipt["status"] is OrderStatus.RESTING
        ):
            raise ValueError("[DEMO-TRIAL-007] require matching terminal IOC order")
        acknowledgement = root / "acknowledgement.json"
        if acknowledgement.exists():
            acknowledged_id = validate_acknowledgement(
                read_document(acknowledgement), intent, parse_timestamp(self._clock())
            )
            if acknowledged_id != receipt["order_id"]:
                raise ValueError("[DEMO-TRIAL-007] acknowledgement and reconciled order disagree")
        fills = tuple(collector.pages("/portfolio/fills", {"subaccount": 0}, "fills"))
        verify_fills(fills, receipt, intent)
        credit, market, settlements = self._read_settlement(
            collector, receipt, state["orders"][0]["account_settlement"]
        )
        positions = tuple(
            collector.pages(
                "/portfolio/positions",
                {"subaccount": 0, "settlement_status": "unsettled"},
                "market_positions",
            )
        )
        quantity = 0
        for position in positions:
            if require_str(position, "ticker") != intent["ticker"]:
                raise ValueError("[DEMO-TRIAL-007] unexpected account position")
            quantity += parse_fixed(require_str(position, "position_fp"), 2)
        balance = self._transport.get("/portfolio/balance", {"subaccount": 0})
        cash = parse_fixed(require_str(balance, "balance_dollars"), 6)
        require_bounded_int(balance, "updated_ts", 0, parse_timestamp(self._clock()) // 1000)
        if (
            quantity != (receipt["filled_count"] if credit is None else 0)
            or len(positions) > 1
            or (credit is not None and bool(positions))
            or cash
            != trial["initial"]["balance_microdollars"]
            - (receipt["fill_cost"] + receipt["fees"]) * 100
            + (0 if credit is None else credit["revenue"] * 100)
        ):
            raise ValueError("[DEMO-TRIAL-007] account position or cash disagrees with order fills")
        observed = parse_timestamp(self._clock())
        event: JSONObject = {
            "action": "receipt",
            "observed_ms": observed,
            "client_order_id": intent["client_order_id"],
            "data": encode_receipt(receipt),
        }
        state = execution_journal().transact(ledger, lambda _: event)
        if credit is not None:
            state = execution_journal().transact(
                ledger,
                lambda _: {
                    "action": "account_settlement",
                    "observed_ms": observed,
                    "client_order_id": intent["client_order_id"],
                    "data": encode_account_settlement(credit),
                },
            )
        evidence: JSONObject = {
            "observed_ms": observed,
            "orders": list(orders),
            "fills": list(fills),
            "positions": list(positions),
            "balance": balance,
            "settlements": list(settlements),
            "market": market,
        }
        with (root / ("reconciliation-" + str(uuid4()) + ".json")).open(
            "x", encoding="utf-8"
        ) as stream:
            stream.write(dump_json_str(evidence))
        return self._report(state)

    def _read_settlement(
        self,
        collector: MarketCollector,
        receipt: OrderReceipt,
        previous: AccountSettlement | None,
    ) -> tuple[AccountSettlement | None, JSONObject | None, tuple[JSONObject, ...]]:
        """Collect complete account credits and cross-check the final market outcome.

        Args:
            collector: Authenticated paginator sharing this trial's transport.
            receipt: Independently verified order receipt.
            previous: Previously journaled credit, if present.

        Returns:
            Validated credit, native market and complete native settlement rows.

        Raises:
            ValueError: DEMO-TRIAL-009 for extra or disappearing account settlements.
            HTTPError: Required evidence is unavailable; no retry is attempted.
        """
        rows = tuple(collector.pages("/portfolio/settlements", {"subaccount": 0}, "settlements"))
        if not rows:
            if previous is not None:
                raise ValueError("[DEMO-TRIAL-009] credited settlement is no longer visible")
            return None, None, rows
        if len(rows) != 1:
            raise ValueError("[DEMO-TRIAL-009] require exactly one account settlement")
        market = require_dict(
            self._transport.get("/markets/" + quote(receipt["ticker"], safe=""), {}), "market"
        )
        credit = decode_native_account_settlement(
            rows[0], market, receipt, parse_timestamp(self._clock())
        )
        return credit, market, rows

    def status(self, root: Path) -> JSONObject:
        """Read persisted trial accounting without network requests.

        Args:
            root: Existing trial directory.

        Returns:
            Validated local journal report with submission permanently disabled.

        Raises:
            ValueError: Changed credentials or invalid persisted evidence.
        """
        self._load(root)
        return self._report(execution_journal().read(root / "execution.sqlite"))

    def evidence(self, root: Path) -> JSONObject:
        """Verify a completed submission capture without contacting the exchange.

        Args:
            root: Trial with capture, native acknowledgement and local timing files.

        Returns:
            Verified evidence and local workflow duration in milliseconds. This
            includes persistence, signing and response parsing, not just network time.

        Raises:
            ValueError: DEMO-EVIDENCE-004 for mismatched records; endpoint validators
                propagate malformed evidence failures without inferring missing data.
            OSError: A required evidence file is absent or unreadable.
        """
        trial = self._load(root)
        return read_submission_evidence(root, trial["intent"])

    @staticmethod
    def _report(state: ExecutionState) -> JSONObject:
        """Describe demo accounting without inheriting the rehearsal's zero-send claim.

        Args:
            state: Validated execution state.

        Returns:
            Demo report distinguishing reserved attempts from confirmed native orders.
        """
        report = execution_report(state)
        del report["orders_sent"]
        report["mode"] = "demo_single_order"
        report["submission_attempt_reserved"] = bool(state["orders"])
        report["confirmed_orders"] = sum(order["receipt"] is not None for order in state["orders"])
        report["confirmed_settlements"] = sum(
            order["account_settlement"] is not None for order in state["orders"]
        )
        report["can_reserve"] = False
        return report


def build_demo_trial(
    client: HttpxClient,
    signer: KalshiSigner,
    clock: Callable[[], str],
    *,
    origin: str = DEMO_ORIGIN,
) -> DemoTrialService:
    """Create focused account and transport services for one demo executor.

    Args:
        client: Caller-owned HTTP client with redirects disabled.
        signer: Validated demo credential signer.
        clock: Aware ISO wall-clock reader.
        origin: Demo origin or loopback integration-test server.

    Returns:
        DemoTrialService with real implementations injected directly.

    Raises:
        ValueError: An unsupported origin is supplied.
    """
    return DemoTrialService(
        DemoAccountClient(client, signer, clock, origin=origin),
        DemoTransport(client, signer, clock, origin=origin),
        signer.fingerprint,
        clock,
    )
