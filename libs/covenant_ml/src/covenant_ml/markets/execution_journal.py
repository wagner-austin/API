"""Fixed execution policy bound to the shared transactional event journal."""

from platform_core.event_journal import EventJournal

from .execution_policy import apply_execution_event
from .execution_records import CAPITAL, EXPOSURE_LIMIT, LOSS_LIMIT, ORDER_LIMIT
from .execution_state import ExecutionState


def execution_journal() -> EventJournal[ExecutionState]:
    """Build the rehearsal journal with its immutable limits and reducer.

    Returns:
        Journal preserving the execution format and transition rules.
    """
    return EventJournal(
        {
            "schema": 1,
            "mode": "local_rehearsal",
            "capital": CAPITAL,
            "order_limit": ORDER_LIMIT,
            "exposure_limit": EXPOSURE_LIMIT,
            "loss_limit": LOSS_LIMIT,
        },
        lambda: ExecutionState(orders=(), halt_reason=None, observed_ms=0),
        apply_execution_event,
    )


__all__ = ["execution_journal"]
