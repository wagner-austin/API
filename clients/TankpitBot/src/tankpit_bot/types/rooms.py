"""Lobby room vocabulary — the selectors an operator picks between.

Tankpit's lobby lists exactly two rooms: the practice room and the
live world. Ground truth from a captured ROOM_LIST frame
(``runs/bot/arterial/bot-20260813-212329.log``)::

    b'+1|Practice|1|0,0,0,0,0,0,0|-1|p|field01.gif|2026'
    ... 5=World (Desert)

The world's DISPLAY name carries the current map in parentheses and
rotates with it ("World (Desert)" one week, another field the next),
so what the operator selects is the stable PREFIX, not the live name:
:func:`tankpit_bot.browser.room_join._resolve_room_entry` matches a
selector exactly, or as a prefix followed by a space or ``(``. That
is what makes ``"World"`` a durable choice and why this tuple does
not try to enumerate map names.

The selectors are a suggestion surface, not a closed set the spawn
path enforces: the live list is the server's to change, so an exact
room name ("World (Desert)") stays a legal selector over the API even
though the control page only offers the two durable ones.
"""

from __future__ import annotations

DEFAULT_LOBBY_ROOM = "Practice"
"""The room a bot joins when ``TANKPIT_ROOM`` is unset or empty."""

LOBBY_ROOMS: tuple[str, ...] = (
    DEFAULT_LOBBY_ROOM,
    "World",
)
"""Durable room selectors, in lobby order; the first is the default."""


__all__ = [
    "DEFAULT_LOBBY_ROOM",
    "LOBBY_ROOMS",
]
