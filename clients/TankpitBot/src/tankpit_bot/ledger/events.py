"""The ledger's action-kind vocabulary.

Every ledger record (decision, outcome, mode transition) carries a
monotonic ``event_id`` so causal references (``caused_by``) are
unambiguous. That counter is session state and lives on
:class:`tankpit_bot.ledger.service.LedgerService`; what remains here is
the vocabulary, which is constant and belongs to no session.
"""

from __future__ import annotations

from typing import Literal

ActionKind = Literal["scan", "move", "teleport", "collect", "map_open", "shoot", "scope"]
"""The seven bot action kinds the ledger records.

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
    "scope",
)
"""All action kinds, for iteration and validation messages."""


__all__ = [
    "ACTION_KINDS",
    "ActionKind",
]
