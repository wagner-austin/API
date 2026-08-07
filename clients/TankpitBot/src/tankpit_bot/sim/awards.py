"""Law 11 — decorations: the awards a tank carries, and 0x4E.

The sim shipped every tank with four zero bytes of ``decoration_state``
and never emitted a 0x4E, so the award layer existed on the wire and
nowhere else ([[session-state-deglobalisation]]).

The encoding is mined ([[decoration-encoding]], JS ``yg``): four bytes
pack into a 32-bit little-endian integer holding NINE two-bit slots,
level 0-3 each. The earning table comes from the in-client guide, and
the archive confirms it end to end — Artax joins every session up to
2026-07-29 carrying ``Stars=2, Tank=2, Honor=1``, that session's lone
0x4E fires ``Tank -> 3``, and every session after it joins at
``Tank=3``. One frame, and it is the 500th kill landing.

Four slots are wire-observable and modelled here:

* 0 Stars — reaching Major, Colonel, General (ranks 6, 7, 8).
* 1 Tank — deactivating 100, 200, 500 enemies.
* 2 Honor — being deactivated 20, 50, 100 times.
* 3 Sword — 100, 200, 500 hours played.

The other five (Shield, Cup, Purple Heart, War Correspondent,
Lightbulb) are earned by bug reports, tournament placings and
community contributions. Nothing on this wire can observe them, so the
sim leaves them at zero rather than inventing a rate.

The counters are SESSION-scoped, as everything else in the sim is: a
sim account is a fresh one, and it earns what a fresh one earns.
"""

from __future__ import annotations

from tankpit_bot.protocol.types import BinaryMessage, DecorationDict

SLOT_STARS = 0
SLOT_TANK = 1
SLOT_HONOR = 2
SLOT_SWORD = 3

DECORATION_SLOTS = 9
"""Nine two-bit slots pack into the four ``decoration_state`` bytes."""

MAX_LEVEL = 3
"""Bronze/silver/gold — two bits, so 3 is the cap."""

STAR_RANKS: tuple[int, int, int] = (6, 7, 8)
"""Major, Colonel, General — the ranks that earn each star level."""

TANK_KILLS: tuple[int, int, int] = (100, 200, 500)
"""Bronze, Silver, Golden Tank Award thresholds."""

HONOR_DEATHS: tuple[int, int, int] = (20, 50, 100)
"""Combat, Battle, Heroic Honor Medal thresholds."""

SWORD_HOURS: tuple[int, int, int] = (100, 200, 500)
"""Shining, Battered, Rusty Sword thresholds, in hours played."""

_SECONDS_PER_HOUR = 3600


def _level_for(value: int, thresholds: tuple[int, int, int]) -> int:
    """Return the award level a counter has reached.

    Args:
        value: The observed counter.
        thresholds: Level 1, 2 and 3 cut-offs, ascending.

    Returns:
        0 when nothing is earned, else the highest level reached.
    """
    level = 0
    for index, threshold in enumerate(thresholds, start=1):
        if value >= threshold:
            level = index
    return level


def pack_decorations(levels: tuple[int, ...]) -> bytes:
    """Pack nine award levels into the wire's four bytes.

    The inverse of the JS ``yg``: slot 0 occupies bits 0-1 of a 32-bit
    little-endian integer, each further slot the next two bits.

    Args:
        levels: Nine levels, 0-3 each.

    Returns:
        The four ``decoration_state`` bytes.

    Raises:
        ValueError: If the shape or any level is out of range — a
            packing that silently truncated would put an award on the
            wire that nobody earned.
    """
    if len(levels) != DECORATION_SLOTS:
        raise ValueError(f"decorations need {DECORATION_SLOTS} slots, got {len(levels)}")
    packed = 0
    for slot, level in enumerate(levels):
        if not 0 <= level <= MAX_LEVEL:
            raise ValueError(f"decoration slot {slot} level {level} outside 0-{MAX_LEVEL}")
        packed |= level << (slot * 2)
    return packed.to_bytes(4, "little")


class AwardLedger:
    """The client's earned awards, and the 0x4E frames that grant them.

    Holds the levels rather than deriving them each tick, because a
    0x4E fires on the TRANSITION and a level that is merely recomputed
    has no transition to notice.
    """

    def __init__(self, client_id: int) -> None:
        """Start a fresh account: nothing earned.

        Args:
            client_id: The tank the granted awards name.
        """
        self._client_id = client_id
        self.levels: list[int] = [0] * DECORATION_SLOTS

    @property
    def decoration_state(self) -> bytes:
        """The four bytes every 0x21, 0x28 and 0x3E carries."""
        return pack_decorations(tuple(self.levels))

    def advance(
        self,
        rank: int,
        destroyed: int,
        deactivated: int,
        played_seconds: int,
        messages: list[BinaryMessage],
    ) -> None:
        """Grant any award the counters have just earned.

        Args:
            rank: The client's current rank.
            destroyed: Enemies the client has deactivated.
            deactivated: Times the client has been deactivated.
            played_seconds: Session playtime.
            messages: This tick's outgoing batch (appended).
        """
        earned = {
            SLOT_STARS: _level_for(rank, STAR_RANKS),
            SLOT_TANK: _level_for(destroyed, TANK_KILLS),
            SLOT_HONOR: _level_for(deactivated, HONOR_DEATHS),
            SLOT_SWORD: _level_for(played_seconds // _SECONDS_PER_HOUR, SWORD_HOURS),
        }
        for slot, level in earned.items():
            # Awards only ever go UP: a demotion does not take a star
            # back, and the archive shows Artax keeping Tank=3 in every
            # session after the one that granted it.
            if level <= self.levels[slot]:
                continue
            self.levels[slot] = level
            messages.append(
                DecorationDict(msg_type=0x4E, tank_id=self._client_id, slot=slot, level=level)
            )


__all__ = [
    "DECORATION_SLOTS",
    "HONOR_DEATHS",
    "MAX_LEVEL",
    "SLOT_HONOR",
    "SLOT_STARS",
    "SLOT_SWORD",
    "SLOT_TANK",
    "STAR_RANKS",
    "SWORD_HOURS",
    "TANK_KILLS",
    "AwardLedger",
    "pack_decorations",
]
