"""Event identity for the ledger: monotonic event ids and action kinds.

Phase 2 of the self-observing architecture. Every ledger record
(decision, outcome, mode transition) carries a process-wide monotonic
``event_id`` so causal references (``caused_by``) are unambiguous.
"""

from __future__ import annotations

from typing import Literal

ActionKind = Literal["scan", "move", "teleport", "collect", "map_open", "shoot"]
"""The six bot action kinds the ledger records.

Deliberately narrower than :data:`tankpit_bot.bot.states.ActionKind`,
which adds the ``"none"`` in-flight sentinel. The ledger records what
the bot DID; "none" is a lifecycle placeholder, not an action.
"""

ACTION_KINDS: tuple[ActionKind, ...] = (
    "scan",
    "move",
    "teleport",
    "collect",
    "map_open",
    "shoot",
)
"""All action kinds, for iteration and validation messages."""

_event_counter = 0


def next_event_id() -> int:
    """Return the next process-wide monotonic event id.

    Returns:
        Strictly increasing integer, starting at 1.
    """
    global _event_counter
    _event_counter += 1
    return _event_counter


def reset_event_ids() -> None:
    """Reset the event counter. Called from test-isolation fixtures."""
    global _event_counter
    _event_counter = 0


__all__ = [
    "ACTION_KINDS",
    "ActionKind",
    "next_event_id",
    "reset_event_ids",
]
