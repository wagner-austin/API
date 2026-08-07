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

Ferry memory is POSITIONALLY invalidated, not clocked (user ruling
2026-08-05). The measured movement law ([[ferry-mechanics]]) says
ferries NEVER drift — they move only when a rider drives them (136/148
wire movements rider-attributed, zero spontaneous) — and practice
bots never ride, so a sighted ferry stays where it was until someone
visibly moves it. Three channels already keep the belief honest:

1. Wire moves: every 0x4A terrain update overwrites the belief at its
   tile, and ferry moves arrive as atomic old→water / new→ferry pairs
   (``update_terrain_tiles``).
2. Re-observation: viewport patches rewrite the tile's current truth.
3. Contact disproof: a boarding teleport displaced off a believed
   ferry deletes the belief on the spot
   (``_expire_disproven_ferry_belief``, world_state_dispatch).

The residual risk — a human rides it away while unobserved — costs
one displaced hop, which channel 3 turns into a deletion and a replan
the same tick: the same accepted stale-belief economics as container
hops. The old 60 s ``FERRY_BELIEF_TTL_MS`` clock sat on top of this
and only FORGOT true ferries (its "ferries drift freely" premise was
falsified by the movement mining); it forced rediscovery pans and the
release→re-lock churn logged in the 2026-08-05 chain.
"""

from __future__ import annotations

from collections import deque

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.state.types import WorldStateDict
from tankpit_bot.types.constants import ASCII_FERRY, TERRAIN_FERRY

FERRY_SEARCH_RADIUS = 12
"""Chebyshev radius around the goal container searched for a ferry.

Within one viewport-ish of the container the ride is short; beyond it
the ride cost erodes the hop's value. Radius alone does NOT establish
that the ride exists — see the pond gate below.
"""


def _goal_water_pond(
    terrain: TerrainMapProtocol,
    goal_x: int,
    goal_y: int,
) -> set[tuple[int, int]]:
    """Flood-fill the connected static-water body holding the goal.

    Args:
        terrain: Static terrain of the current field.
        goal_x: Water-locked container X.
        goal_y: Water-locked container Y.

    Returns:
        Every water tile 4-connected to the goal (empty when the goal
        itself is not afloat). A live ferry tile counts as water for
        connectivity: on the composed decision view a ferry renders
        as ``~`` OVER the lake it floats on, and treating it as a
        wall would both split the pond and exclude the very boarding
        tile the search is validating.
    """
    afloat = {terrain.WATER, ASCII_FERRY}
    if terrain.get_terrain(goal_x, goal_y) not in afloat:
        return set()
    pond = {(goal_x, goal_y)}
    queue = deque([(goal_x, goal_y)])
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx <= 255 and 0 <= ny <= 255):
                continue
            if (nx, ny) in pond or terrain.get_terrain(nx, ny) not in afloat:
                continue
            pond.add((nx, ny))
            queue.append((nx, ny))
    return pond


def find_ferry_boarding_tile(
    world: WorldStateDict,
    terrain: TerrainMapProtocol,
    goal_x: int,
    goal_y: int,
) -> tuple[int, int] | None:
    """Return the best believed ferry tile to board for a water goal.

    Candidates are wire-terrain beliefs of type ``TERRAIN_FERRY``
    (positionally invalidated — see the module docstring; no clock),
    at most :data:`FERRY_SEARCH_RADIUS` from the goal, AND floating on
    the goal's own water body — the ride must EXIST. The two live
    deadlocks of 2026-08-04/05 (runs bot-20260804-234008 and
    bot-20260805-070006) were both this gate's absence: a ferry
    docked on a separate pool one land ridge away from the
    container's pond was served as the boarding tile, the ride could
    never reach the pickup, and the hop + lock + disembark contract
    cycled for 11 and 3 minutes respectively (field01 truth: the
    (106,11-12) pond holds 4,456 water tiles and does not contain
    (112,15)). Among qualifying candidates the one nearest the goal
    wins (shortest ride).

    Args:
        world: Current world state with wire-terrain beliefs.
        terrain: Static terrain of the current field (pond gate).
        goal_x: Water-locked container X.
        goal_y: Water-locked container Y.

    Returns:
        ``(x, y)`` of the boarding tile, or ``None`` when no believed
        ferry floats on the goal's water body nearby.
    """
    pond: set[tuple[int, int]] | None = None
    best: tuple[int, int] | None = None
    best_dist = 0
    for tile in world["terrain"].values():
        if tile["terrain_type"] != TERRAIN_FERRY:
            continue
        dist = max(abs(tile["x"] - goal_x), abs(tile["y"] - goal_y))
        if dist > FERRY_SEARCH_RADIUS:
            continue
        if pond is None:
            pond = _goal_water_pond(terrain, goal_x, goal_y)
        if (tile["x"], tile["y"]) not in pond:
            continue
        if best is None or dist < best_dist:
            best = (tile["x"], tile["y"])
            best_dist = dist
    return best


__all__ = [
    "FERRY_SEARCH_RADIUS",
    "find_ferry_boarding_tile",
]
