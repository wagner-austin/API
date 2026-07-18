"""Unified action-outcome fabric: one recorded event per attempt.

Phase 2 of the self-observing architecture. Replaces the three
parallel diagnostic mechanisms (``emit_wire_complete``,
``teleport_attempt``, ``combat_feedback``) and the invisible fourth
(executor ``emit_ai`` discards) with one ``action_outcome`` diagnostic
kind + per-kind ring records.
"""

from tankpit_bot.ledger.outcome._emit import (
    emit_action_outcome,
    reset_action_outcome_tracking,
)
from tankpit_bot.ledger.outcomes import (
    ActionOutcome,
    CollectOutcome,
    MapOpenOutcome,
    MoveOutcome,
    ScanOutcome,
    ShootOutcome,
    TeleportOutcome,
)

__all__ = [
    "ActionOutcome",
    "CollectOutcome",
    "MapOpenOutcome",
    "MoveOutcome",
    "ScanOutcome",
    "ShootOutcome",
    "TeleportOutcome",
    "emit_action_outcome",
    "reset_action_outcome_tracking",
]
