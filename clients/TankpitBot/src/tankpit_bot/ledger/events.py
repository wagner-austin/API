"""The ledger's action-kind vocabulary.

Every ledger record (decision, outcome, mode transition) carries a
monotonic ``event_id`` so causal references (``caused_by``) are
unambiguous. That counter is session state and lives on
:class:`tankpit_bot.ledger.service.LedgerService`; what remains here is
the vocabulary, which is constant and belongs to no session.
"""

from __future__ import annotations

from enum import StrEnum


class ActionKind(StrEnum):
    """The seven bot action kinds the ledger records.

    Deliberately narrower than :class:`tankpit_bot.bot.states.ActionKind`,
    which adds the ``NONE`` in-flight sentinel. The ledger records what
    the bot DID; "none" is a lifecycle placeholder, not an action.
    Declaration order is the order the ledger's per-kind tables iterate.
    """

    SCAN = "scan"
    MOVE = "move"
    TELEPORT = "teleport"
    COLLECT = "collect"
    MAP_OPEN = "map_open"
    SHOOT = "shoot"
    SCOPE = "scope"


__all__ = [
    "ActionKind",
]
