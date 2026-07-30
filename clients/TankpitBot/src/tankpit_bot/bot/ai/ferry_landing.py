"""Ferry boarding as the teleport landing for water-locked containers.

User doctrine ([[flag-triage-20260729]] F5, verbatim): "ferries are
actually the best way to get fuel and equipment, since you can use
them to access many equipment and fuel canisters you otherwise
couldn't. you generally will need to teleport to the ferry since many
times it will be on its own area in the water."

The mechanics are all pre-proven ([[ferry-mechanics]]): a ferry tile
is always boardable, water is passable while riding, and a floating
container is served while riding exactly like a land container. The
one missing link was the LANDING: a container in open water has no
passable tile on or beside it, so ``find_teleport_landing_tile``
returns ``None`` and the larder tallied it ``no_landing`` forever
(15/15 in run bot-20260730-000038). This module finds the boarding
tile instead — the freshest-known ferry near the goal — so the hop
becomes teleport-to-ferry + ride-to-container under the existing
lock-continuation machinery.

Ferries move on their own, so a remembered ferry tile rots like any
other dynamic belief: candidates are gated on
:data:`FERRY_BELIEF_TTL_MS` and callers tally stale/absent ferries as
``no_landing`` still — an honest decline, never a blind teleport onto
open water (the server refuses water landings, [[flag-triage-20260729]]
F4 fix notes).
"""

from __future__ import annotations

from tankpit_bot.state.types import WorldStateDict
from tankpit_bot.state.types.constants import TERRAIN_FERRY

FERRY_BELIEF_TTL_MS = 60000
"""How long a believed ferry tile stays a trusted boarding target.

Ferries drift freely on water; a minute-old sighting is a guess, and
a teleport aimed at vacated water is a refused command. Unbracketed —
no wire measurement of ferry drift speed yet.
"""

FERRY_SEARCH_RADIUS = 12
"""Chebyshev radius around the goal container searched for a ferry.

Within one viewport-ish of the container the ride is short and the
same-water-body assumption is usually safe; beyond it the odds that
the ferry can actually reach the goal drop and the ride cost erodes
the hop's value.
"""


def find_ferry_boarding_tile(
    world: WorldStateDict,
    goal_x: int,
    goal_y: int,
    now_ms: int,
) -> tuple[int, int] | None:
    """Return the best believed ferry tile to board for a water goal.

    Candidates are wire-terrain beliefs of type ``TERRAIN_FERRY``
    observed within :data:`FERRY_BELIEF_TTL_MS`, at most
    :data:`FERRY_SEARCH_RADIUS` from the goal; the one nearest the
    goal wins (shortest ride).

    Args:
        world: Current world state with wire-terrain beliefs.
        goal_x: Water-locked container X.
        goal_y: Water-locked container Y.
        now_ms: Current tick timestamp for the freshness gate.

    Returns:
        ``(x, y)`` of the boarding tile, or ``None`` when no fresh
        ferry is known near the goal.
    """
    best: tuple[int, int] | None = None
    best_dist = 0
    for tile in world["terrain"].values():
        if tile["terrain_type"] != TERRAIN_FERRY:
            continue
        if now_ms - tile["observed_ms"] > FERRY_BELIEF_TTL_MS:
            continue
        dist = max(abs(tile["x"] - goal_x), abs(tile["y"] - goal_y))
        if dist > FERRY_SEARCH_RADIUS:
            continue
        if best is None or dist < best_dist:
            best = (tile["x"], tile["y"])
            best_dist = dist
    return best


__all__ = [
    "FERRY_BELIEF_TTL_MS",
    "FERRY_SEARCH_RADIUS",
    "find_ferry_boarding_tile",
]
