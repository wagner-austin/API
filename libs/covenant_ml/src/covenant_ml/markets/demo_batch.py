"""Bounded demo execution with account-wide accounting and restart-safe admission."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from uuid import UUID, uuid4

from platform_core.http_client import HttpxClient
from platform_core.json_utils import JSONObject

from .batch_account import DemoAccountReconciler
from .batch_attempts import read_batch_attempts
from .batch_records import (
    DemoBatchPlan,
    baseline_digest,
    decode_batch_plan,
    encode_batch_plan,
    require_batch_history,
)
from .demo_trial import DemoTrialService, build_demo_trial
from .demo_trial_records import decode_trial, validate_acknowledgement
from .execution_journal import append_execution_event, read_execution_journal
from .execution_policy import apply_execution_event
from .execution_records import (
    EXPOSURE_LIMIT,
    LOSS_LIMIT,
    ORDER_LIMIT,
    OrderIntent,
    OrderStatus,
    decode_intent,
    encode_intent,
)
from .execution_state import ExecutionState, execution_report, order_accounting, policy_issues
from .execution_storage import (
    batch_operation,
    claim_batch_owner,
    initialize_batch_storage,
    persist_evidence,
    require_batch_owner,
)
from .kalshi_auth import KalshiSigner
from .kalshi_transport import DEMO_ORIGIN, DemoTransport
from .submission import read_submission_evidence, submit_reserved_order
from .validation import parse_timestamp, read_document


class DemoBatchService:
    """Admit at most one new demo POST per call, after full-account reconciliation."""

    def __init__(
        self,
        trial: DemoTrialService,
        transport: DemoTransport,
        reconciler: DemoAccountReconciler,
        fingerprint: str,
        guard_root: Path,
        clock: Callable[[], str],
        identity: Callable[[], UUID],
    ) -> None:
        """Inject focused real services and explicit identity, storage and time effects.

        Args:
            trial: Existing single-order predecessor verifier.
            transport: Demo-only order boundary.
            reconciler: Complete authenticated account collector.
            fingerprint: Loaded credential identity.
            guard_root: Shared ownership directory, separate from batch directories.
            clock: Aware observation clock.
            identity: Unique attempt identity generator.
        """
        self._trial, self._transport, self._reconciler = trial, transport, reconciler
        self._fingerprint, self._guard_root = fingerprint, guard_root
        self._clock, self._identity = clock, identity

    def prepare(
        self, root: Path, predecessor: Path, tickers: tuple[str, ...], price: int
    ) -> JSONObject:
        """Freeze an account-bound batch after verifying its single settled predecessor.

        Args:
            root: New exclusive batch directory.
            predecessor: Completed single-order trial directory.
            tickers: Distinct sequence of one through one hundred demo contracts.
            price: YES limit price in ten-thousandths of a dollar, within ORDER_LIMIT.

        Returns:
            Persisted immutable plan; this action sends no orders.

        Raises:
            ValueError: Invalid plan, account evidence or predecessor identity.
            FileExistsError: Existing batch root or account ownership prevents a new batch.
            HTTPError: Required predecessor account evidence cannot be read.
        """
        self._trial.reconcile(predecessor)
        source = predecessor / "execution.sqlite"
        state = read_execution_journal(source)
        previous = decode_trial(read_document(predecessor / "trial.json"))
        created = parse_timestamp(self._clock())
        plan = decode_batch_plan(
            {
                "schema": 1,
                "mode": "demo_execution_batch",
                "credential_fingerprint": self._fingerprint,
                "initial_cash": previous["initial"]["balance_microdollars"],
                "baseline_digest": baseline_digest(state),
                "tickers": list(tickers),
                "limit_price": price,
                "created_ms": created,
                "expires_ms": created + 86400000,
            }
        )
        require_batch_history(plan, state, self._fingerprint)
        initialize_batch_storage(root, source)
        copied = read_execution_journal(root / "execution.sqlite")
        require_batch_history(plan, copied, self._fingerprint)
        read_batch_attempts(root, plan, copied)
        persist_evidence(root / "plan.json", encode_batch_plan(plan))
        claim_batch_owner(self._guard_root, root, self._fingerprint)
        return encode_batch_plan(plan)

    def _load(self, root: Path) -> tuple[DemoBatchPlan, ExecutionState, tuple[OrderIntent, ...]]:
        """Bind the plan, ownership record, journal and persisted attempt sequence.

        Args:
            root: Existing batch directory.

        Returns:
            Validated plan, current state and every consumed attempt.

        Raises:
            ValueError: Identity, plan, history or attempt evidence disagrees.
            OSError: Required evidence is unavailable.
        """
        require_batch_owner(self._guard_root, root, self._fingerprint)
        plan = decode_batch_plan(read_document(root / "plan.json"))
        state = read_execution_journal(root / "execution.sqlite")
        require_batch_history(plan, state, self._fingerprint)
        return plan, state, read_batch_attempts(root, plan, state)

    def _reconcile(self, root: Path, plan: DemoBatchPlan, state: ExecutionState) -> ExecutionState:
        """Persist complete native evidence before atomically accepting its journal events.

        Args:
            root: Locked batch directory.
            plan: Validated plan supplying initial cash.
            state: Previously accepted journal state.

        Returns:
            New state only after every account record agrees.

        Raises:
            ValueError: Account, acknowledgement or cash evidence disagrees.
            OSError: Native evidence cannot be persisted.
            HTTPError: Account collection fails; no order is sent.
        """
        document, event = self._reconciler.collect(state, plan["initial_cash"])
        updated = apply_execution_event(state, event)
        attempts = read_batch_attempts(root, plan, updated)
        for index, intent in enumerate(attempts, 1):
            path = root / "attempts" / f"{index:03d}"
            if (path / "acknowledgement.json").exists():
                acknowledged = validate_acknowledgement(
                    read_document(path / "acknowledgement.json"), intent, updated["observed_ms"]
                )
                if (path / "submission-timing.json").exists():
                    read_submission_evidence(path, intent)
                receipts = tuple(
                    order["receipt"] for order in updated["orders"] if order["intent"] == intent
                )
                if (
                    len(receipts) != 1
                    or receipts[0] is None
                    or receipts[0]["order_id"] != acknowledged
                ):
                    raise ValueError("[DEMO-BATCH-003] acknowledgement and account order disagree")
        persist_evidence(root / ("account-" + str(uuid4()) + ".json"), document)
        return append_execution_event(root / "execution.sqlite", event)

    def advance(self, root: Path) -> JSONObject:
        """Reconcile first, consume one frozen attempt, reserve once and submit once.

        Args:
            root: Prepared batch root.

        Returns:
            Current report after one attempt, or a completed/expired/risk-blocked report.

        Raises:
            ValueError: Inconsistent evidence or refused quote; consumed attempts stay consumed.
            HTTPError: A failed POST stays reserved; subsequent calls only reconcile until visible.
            OperationalError: Another process owns the operation lock.
        """
        with batch_operation(root):
            plan, state, attempts = self._load(root)
            state = self._reconcile(root, plan, state)
            observed = parse_timestamp(self._clock())
            report = self._report(plan, state, attempts, observed)
            if not report["can_reserve"]:
                return report
            intent = decode_intent(
                {
                    "client_order_id": str(self._identity()),
                    "ticker": plan["tickers"][len(attempts)],
                    "contracts": 1,
                    "limit_price": plan["limit_price"],
                    "fee_reserve": ORDER_LIMIT - plan["limit_price"],
                    "created_ms": observed,
                    "expires_ms": min(observed + 300000, plan["expires_ms"]),
                }
            )
            path = root / "attempts" / f"{len(attempts) + 1:03d}"
            path.mkdir()
            persist_evidence(path / "intent.json", encode_intent(intent))
            submit_reserved_order(
                self._transport,
                self._clock,
                path,
                root / "execution.sqlite",
                intent,
                state["observed_ms"],
            )
            state = self._reconcile(root, plan, read_execution_journal(root / "execution.sqlite"))
            return self._report(plan, state, (*attempts, intent), parse_timestamp(self._clock()))

    def reconcile(self, root: Path) -> JSONObject:
        """Collect account evidence without submitting, including after cap or expiry.

        Args:
            root: Existing batch directory.

        Returns:
            Reconciled report retaining all attempt and risk limits.

        Raises:
            ValueError: Identity or complete account evidence is inconsistent.
            HTTPError: Required account evidence is unavailable.
        """
        with batch_operation(root):
            plan, state, attempts = self._load(root)
            state = self._reconcile(root, plan, state)
            return self._report(plan, state, attempts, parse_timestamp(self._clock()))

    def status(self, root: Path) -> JSONObject:
        """Read saved batch evidence locally; no HTTP request or submission occurs.

        Args:
            root: Existing batch directory.

        Returns:
            Local status with accounting and consumed attempts.

        Raises:
            ValueError: Saved plan, ownership or journal evidence disagrees.
        """
        with batch_operation(root):
            plan, state, attempts = self._load(root)
            return self._report(plan, state, attempts, parse_timestamp(self._clock()))

    @staticmethod
    def _report(
        plan: DemoBatchPlan, state: ExecutionState, attempts: tuple[OrderIntent, ...], now: int
    ) -> JSONObject:
        """Separate account totals from batch observations and avoid claiming predictive skill.

        Args:
            plan: Frozen batch limits.
            state: Replayed account state.
            attempts: Every consumed attempt, whether posted or refused.
            now: Current timestamp for admission expiry.

        Returns:
            Exact account accounting, separate batch counts and conservative admission state.
        """
        report = execution_report(state)
        del report["orders_sent"]
        totals = tuple(order_accounting(order) for order in state["orders"])
        exposure, losses = sum(row[0] for row in totals), sum(row[1] for row in totals)
        receipts = tuple(
            order["receipt"] for order in state["orders"][1:] if order["receipt"] is not None
        )
        report.update(
            mode="demo_execution_batch",
            attempt_limit=len(plan["tickers"]),
            attempts_started=len(attempts),
            reserved_batch_orders=len(state["orders"]) - 1,
            confirmed_batch_orders=len(receipts),
            filled_batch_orders=sum(row["filled_count"] > 0 for row in receipts),
            canceled_batch_orders=sum(row["status"] is OrderStatus.CANCELED for row in receipts),
            settled_batch_orders=sum(
                row["account_settlement"] is not None for row in state["orders"][1:]
            ),
            batch_net_realized_profit=sum(order_accounting(row)[2] for row in state["orders"][1:]),
            expired=now >= plan["expires_ms"],
            can_reserve=(
                not policy_issues(state)
                and len(attempts) < len(plan["tickers"])
                and plan["created_ms"] <= now < plan["expires_ms"]
                and exposure + ORDER_LIMIT <= EXPOSURE_LIMIT
                and losses + exposure + ORDER_LIMIT <= LOSS_LIMIT
            ),
        )
        return report


def build_demo_batch(
    client: HttpxClient,
    signer: KalshiSigner,
    clock: Callable[[], str],
    guard_root: Path,
    *,
    origin: str = DEMO_ORIGIN,
    identity: Callable[[], UUID] = uuid4,
) -> DemoBatchService:
    """Assemble the demo-only batch using focused production services.

    Args:
        client: Caller-owned real HTTP client.
        signer: Validated credential signer.
        clock: Aware observation clock.
        guard_root: Shared credential ownership directory.
        origin: Demo or loopback integration-test origin.
        identity: UUID generator, injected for deterministic real-transport tests.

    Returns:
        Fully wired batch service with no production exchange endpoint.

    Raises:
        ValueError: An unsupported origin is supplied.
    """
    transport = DemoTransport(client, signer, clock, origin=origin)
    return DemoBatchService(
        build_demo_trial(client, signer, clock, origin=origin),
        transport,
        DemoAccountReconciler(transport, clock),
        signer.fingerprint,
        guard_root,
        clock,
        identity,
    )
