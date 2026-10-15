"""Immutable bounded demo batch plans and account-history identity checks."""

from __future__ import annotations

from hashlib import sha256

from platform_core.json_utils import JSONObject, dump_json_str, require_list, require_str
from typing_extensions import ReadOnly, TypedDict

from .campaign_records import require_digest
from .execution_records import ORDER_LIMIT
from .execution_state import ExecutionState, encode_tracked_order, policy_issues
from .validation import require_bounded_int, require_nonempty


class DemoBatchPlan(TypedDict):
    """Frozen contract sequence, account identity and deadline; real trading is absent."""

    credential_fingerprint: ReadOnly[str]
    initial_cash: ReadOnly[int]
    baseline_digest: ReadOnly[str]
    tickers: ReadOnly[tuple[str, ...]]
    limit_price: ReadOnly[int]
    created_ms: ReadOnly[int]
    expires_ms: ReadOnly[int]


def encode_batch_plan(plan: DemoBatchPlan) -> JSONObject:
    """Encode every declared demo constraint.

    Args:
        plan: Validated immutable batch plan.

    Returns:
        Versioned plan with exact monetary units and no live endpoint.
    """
    return {
        "schema": 1,
        "mode": "demo_execution_batch",
        "credential_fingerprint": plan["credential_fingerprint"],
        "initial_cash": plan["initial_cash"],
        "baseline_digest": plan["baseline_digest"],
        "tickers": list(plan["tickers"]),
        "limit_price": plan["limit_price"],
        "created_ms": plan["created_ms"],
        "expires_ms": plan["expires_ms"],
    }


def decode_batch_plan(document: JSONObject) -> DemoBatchPlan:
    """Require a finite distinct-contract plan within the existing one-dollar order limit.

    Args:
        document: Untrusted saved plan or command input.

    Returns:
        Immutable plan containing at most one hundred one-contract attempts.

    Raises:
        ValueError: DEMO-BATCH-004 for mode, duplicate contracts or invalid count.
            Field validators propagate invalid amounts, identities and deadlines.
    """
    require_bounded_int(document, "schema", 1, 1)
    tickers = tuple(
        require_nonempty({"ticker": value}, "ticker") for value in require_list(document, "tickers")
    )
    created = require_bounded_int(document, "created_ms", 0, 2**63 - 1)
    if (
        require_str(document, "mode") != "demo_execution_batch"
        or not 1 <= len(tickers) <= 100
        or len(set(tickers)) != len(tickers)
    ):
        raise ValueError("[DEMO-BATCH-004] require a distinct bounded demo contract sequence")
    return DemoBatchPlan(
        credential_fingerprint=require_digest(document, "credential_fingerprint"),
        initial_cash=require_bounded_int(document, "initial_cash", 1000000, 2**63 - 1),
        baseline_digest=require_digest(document, "baseline_digest"),
        tickers=tickers,
        limit_price=require_bounded_int(document, "limit_price", 1, ORDER_LIMIT - 1),
        created_ms=created,
        expires_ms=require_bounded_int(document, "expires_ms", created + 1, created + 86400000),
    )


def baseline_digest(state: ExecutionState) -> str:
    """Bind a batch to one fully settled and reconciled predecessor order.

    Args:
        state: Actual predecessor journal or batch journal containing its first order.

    Returns:
        SHA-256 of the complete normalized predecessor order and credited settlement.

    Raises:
        ValueError: DEMO-BATCH-005 for absent or uncredited baseline history.
    """
    if not state["orders"] or state["orders"][0]["account_settlement"] is None:
        raise ValueError("[DEMO-BATCH-005] baseline requires an authenticated settled order")
    return sha256(dump_json_str(encode_tracked_order(state["orders"][0])).encode()).hexdigest()


def require_batch_history(plan: DemoBatchPlan, state: ExecutionState, fingerprint: str) -> None:
    """Bind the replayed journal to this account, baseline and frozen ticker sequence.

    Args:
        plan: Decoded immutable batch plan.
        state: Complete current journal.
        fingerprint: Credential identity used by the actual demo signer.

    Raises:
        ValueError: DEMO-BATCH-005 for changed identity or out-of-plan reservations.
    """
    baseline = baseline_digest(state)
    reserved = tuple(order["intent"]["ticker"] for order in state["orders"][1:])
    if (
        plan["credential_fingerprint"] != fingerprint
        or baseline != plan["baseline_digest"]
        or len(reserved) > len(plan["tickers"])
        or any(ticker not in plan["tickers"] for ticker in reserved)
        or len(set(reserved)) != len(reserved)
        or state["orders"][0]["intent"]["ticker"] in plan["tickers"]
        or any(
            order["intent"]["contracts"] != 1
            or order["intent"]["limit_price"] != plan["limit_price"]
            for order in state["orders"][1:]
        )
    ):
        raise ValueError("[DEMO-BATCH-005] account identity or reserved history differs from plan")
    if any(issue != "EXECUTION-UNRESOLVED" for issue in policy_issues(state)):
        raise ValueError("[DEMO-BATCH-005] batch journal has a permanent policy breach")
