"""Durable attempt admission and identity checks for a fixed demo contract sequence."""

from __future__ import annotations

from pathlib import Path

from .batch_records import DemoBatchPlan
from .execution_records import ORDER_LIMIT, OrderIntent, decode_intent
from .execution_state import ExecutionState
from .validation import read_document


def read_batch_attempts(
    root: Path, plan: DemoBatchPlan, state: ExecutionState
) -> tuple[OrderIntent, ...]:
    """Require contiguous immutable attempts containing every journal reservation.

    Args:
        root: Existing batch directory.
        plan: Validated frozen plan.
        state: Replayed account journal.

    Returns:
        Ordered attempt intents, including attempts refused before reservation.

    Raises:
        ValueError: DEMO-BATCH-006 for missing, duplicate or out-of-plan attempts.
        OSError: An attempt manifest cannot be read; partial creation fails closed.
    """
    paths = tuple(sorted((root / "attempts").iterdir()))
    if len(paths) > len(plan["tickers"]) or tuple(path.name for path in paths) != tuple(
        f"{index:03d}" for index in range(1, len(paths) + 1)
    ):
        raise ValueError("[DEMO-BATCH-006] attempt directories exceed or break the frozen sequence")
    attempts = tuple(decode_intent(read_document(path / "intent.json")) for path in paths)
    for index, intent in enumerate(attempts):
        if (
            intent["ticker"] != plan["tickers"][index]
            or intent["contracts"] != 1
            or intent["limit_price"] != plan["limit_price"]
            or intent["fee_reserve"] != ORDER_LIMIT - plan["limit_price"]
            or not plan["created_ms"] <= intent["created_ms"] < plan["expires_ms"]
            or intent["expires_ms"] != min(intent["created_ms"] + 300000, plan["expires_ms"])
        ):
            raise ValueError("[DEMO-BATCH-006] attempt differs from the frozen contract or limits")
    by_id = {intent["client_order_id"]: intent for intent in attempts}
    if len(by_id) != len(attempts) or any(
        by_id.get(order["intent"]["client_order_id"]) != order["intent"]
        for order in state["orders"][1:]
    ):
        raise ValueError("[DEMO-BATCH-006] reserved intents must match distinct persisted attempts")
    return attempts
