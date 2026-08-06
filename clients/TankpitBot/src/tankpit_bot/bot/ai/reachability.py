"""Reachability helpers for movement and collection.

These helpers answer the questions the live bot actually needs: can
the current visible viewport execute this command right now, and can
a teleport actually ARRIVE where the plan needs the tank? Walkability
— including dynamic obstacles like hostile mines — is owned entirely
by the terrain view (see ``compose_decision_terrain``); the viewport
helpers only add bounding and pickup adjacency service. Landing
ATTAINABILITY is the teleport-side twin: legality says the server will
accept the aim, attainability says the tank will stand on that tile
afterwards (the displacement law bounces landings off mines,
[[teleport-mechanics]] / [[mine-mechanics]]).
"""

from __future__ import annotations

from collections.abc import Mapping

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.pathfinding import path_exists
from tankpit_bot.state.types import MineStateDict, WorldStateDict
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds

_ADJACENT_DIRECTIONS: tuple[tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
_MAP_MIN = 0
_MAP_MAX = 255


def is_move_reachable_in_viewport(
    world: WorldStateDict,
    terrain: TerrainMapProtocol,
    start_x: int,
    start_y: int,
    goal_x: int,
    goal_y: int,
) -> bool:
    """Return True when the viewport contains a walk path to the exact tile.

    Args:
        world: Current world state with visible viewport bounds.
        terrain: Terrain map for passability checks.
        start_x: Starting X coordinate.
        start_y: Starting Y coordinate.
        goal_x: Goal X coordinate.
        goal_y: Goal Y coordinate.

    Returns:
        True if a path exists entirely inside the current visible viewport.
    """
    left, top, right, bottom = viewport_visible_bounds(world["viewport"])
    return path_exists(
        terrain,
        start_x,
        start_y,
        goal_x,
        goal_y,
        min_x=left,
        min_y=top,
        max_x=right,
        max_y=bottom,
    )


def is_collection_reachable_in_viewport(
    world: WorldStateDict,
    terrain: TerrainMapProtocol,
    start_x: int,
    start_y: int,
    goal_x: int,
    goal_y: int,
) -> bool:
    """Return True when a pickup can be completed from the current viewport.

    Collection commands can complete either by reaching the target tile itself
    or by reaching a safe cardinally adjacent tile when the target tile cannot
    be occupied directly (impassable terrain or a hostile mine on the
    container's tile — both are impassable in the composed terrain view, and
    both are serviced from an adjacent tile the same way).

    Args:
        world: Current world state with visible viewport bounds.
        terrain: Terrain map for passability checks.
        start_x: Starting X coordinate.
        start_y: Starting Y coordinate.
        goal_x: Pickup target X coordinate.
        goal_y: Pickup target Y coordinate.

    Returns:
        True if a collection path exists entirely inside the current viewport.
    """
    left, top, right, bottom = viewport_visible_bounds(world["viewport"])
    if terrain.is_passable(goal_x, goal_y) and _viewport_path_exists(
        terrain,
        start_x,
        start_y,
        goal_x,
        goal_y,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
    ):
        return True

    for landing_x, landing_y in _collection_landing_tiles(goal_x, goal_y):
        if not (left <= landing_x <= right and top <= landing_y <= bottom):
            continue
        if not terrain.is_passable(landing_x, landing_y):
            continue
        if _viewport_path_exists(
            terrain,
            start_x,
            start_y,
            landing_x,
            landing_y,
            left=left,
            top=top,
            right=right,
            bottom=bottom,
        ):
            return True
    return False


def find_attainable_landing_tile(
    terrain: TerrainMapProtocol,
    mines: Mapping[str, MineStateDict],
    goal_x: int,
    goal_y: int,
) -> tuple[int, int] | None:
    """Find a teleport landing the tank will actually END UP on.

    Legality (``is_landing_legal``) only says the server accepts the
    aim; the displacement law then bounces the landing off any mine on
    the tile ([[mine-mechanics]] § teleport landings displace, probe
    3/3 2026-07-28). For a pickup that is fatal: the measured transfer
    choreography needs the tank ON the container or cardinally
    adjacent, so a landing that displaces never completes the mission.
    Session bot-20260805-173034 spent 43 minutes re-aiming at a known
    mine — 534 displaced teleports, zero pickups — because the
    selector answered legality when the plan needed attainability.

    Every known mine displaces regardless of team (user law
    2026-06-16, verbatim: "you get moved off if there are mines") —
    pass the full ``world["mines"]`` layer, not the hostile filter.

    Args:
        terrain: Terrain view answering landing legality.
        mines: Known mines indexed by ``"x,y"`` key (all teams).
        goal_x: Target container X coordinate.
        goal_y: Target container Y coordinate.

    Returns:
        The first service tile (target, then cardinal neighbors) that
        is terrain-legal AND mine-free, or ``None`` when every service
        tile would refuse or displace the landing.
    """
    if not (_MAP_MIN <= goal_x <= _MAP_MAX and _MAP_MIN <= goal_y <= _MAP_MAX):
        return None
    if terrain.is_landing_legal(goal_x, goal_y) and f"{goal_x},{goal_y}" not in mines:
        return (goal_x, goal_y)
    for landing_x, landing_y in _collection_landing_tiles(goal_x, goal_y):
        if not terrain.is_landing_legal(landing_x, landing_y):
            continue
        if f"{landing_x},{landing_y}" in mines:
            continue
        return (landing_x, landing_y)
    return None


def _collection_landing_tiles(goal_x: int, goal_y: int) -> list[tuple[int, int]]:
    """Return cardinal landing tiles that could service a pickup target.

    Args:
        goal_x: Pickup target X coordinate.
        goal_y: Pickup target Y coordinate.

    Returns:
        Cardinally adjacent in-bounds coordinates.
    """
    result: list[tuple[int, int]] = []
    for dx, dy in _ADJACENT_DIRECTIONS:
        landing_x = goal_x + dx
        landing_y = goal_y + dy
        if not (_MAP_MIN <= landing_x <= _MAP_MAX and _MAP_MIN <= landing_y <= _MAP_MAX):
            continue
        result.append((landing_x, landing_y))
    return result


def _viewport_path_exists(
    terrain: TerrainMapProtocol,
    start_x: int,
    start_y: int,
    goal_x: int,
    goal_y: int,
    *,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> bool:
    """Return True when a path exists entirely inside supplied viewport bounds.

    Args:
        terrain: Terrain map for passability checks.
        start_x: Starting X coordinate.
        start_y: Starting Y coordinate.
        goal_x: Goal X coordinate.
        goal_y: Goal Y coordinate.
        left: Inclusive viewport minimum X bound.
        top: Inclusive viewport minimum Y bound.
        right: Inclusive viewport maximum X bound.
        bottom: Inclusive viewport maximum Y bound.

    Returns:
        True if a bounded path exists inside the visible viewport.
    """
    return path_exists(
        terrain,
        start_x,
        start_y,
        goal_x,
        goal_y,
        min_x=left,
        min_y=top,
        max_x=right,
        max_y=bottom,
    )


__all__ = [
    "find_attainable_landing_tile",
    "is_collection_reachable_in_viewport",
    "is_move_reachable_in_viewport",
]
