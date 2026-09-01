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

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.pathfinding import path_exists
from tankpit_bot.state.types import WorldStateDict
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
    return is_collection_reachable_within_bounds(
        terrain,
        start_x,
        start_y,
        goal_x,
        goal_y,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
    )


def is_collection_reachable_within_bounds(
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
    """Return True when a pickup can be completed inside supplied bounds.

    The bounds-parameterized core of
    :func:`is_collection_reachable_in_viewport`: the same
    goal-or-cardinal-service rule, evaluated against any rectangle.
    The quad-sweep harvest ([[quad-sweep-doctrine]]) calls it with the
    tank->target bounding box (plus margin) instead of the current
    window -- window anchoring lets a leg-by-leg walk follow any path
    such a rectangle contains, so this answers "can the block serve
    this container at all", not "can the current window".

    Args:
        terrain: Terrain map for passability checks.
        start_x: Starting X coordinate.
        start_y: Starting Y coordinate.
        goal_x: Pickup target X coordinate.
        goal_y: Pickup target Y coordinate.
        left: Inclusive minimum X bound.
        top: Inclusive minimum Y bound.
        right: Inclusive maximum X bound.
        bottom: Inclusive maximum Y bound.

    Returns:
        True if a collection path exists entirely inside the bounds.
    """
    if not (left <= goal_x <= right and top <= goal_y <= bottom):
        # The pickup click itself must target a tile inside the bounds:
        # the server refuses a pickup at an out-of-window container
        # with 0x52 code 0 even when an in-bounds cardinal neighbour
        # is walkable. Run bot-20260901-024845 drew all three of its
        # collect cant_do receipts exactly this way — (79,92) against
        # window left 80, (138,144) against top 145, (100,136) against
        # bottom 135 — each dispatched because the adjacent-service
        # branch below accepted the in-window neighbour. An
        # out-of-bounds goal is not collectable from these bounds,
        # full stop; the caller's hold law leaves it to the hop lane.
        return False
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
    goal_x: int,
    goal_y: int,
) -> tuple[int, int] | None:
    """Find a teleport landing the tank will actually END UP on.

    Legality (``is_landing_legal``) only says the server accepts the
    aim; the displacement law then bounces the landing off any HOSTILE
    mine on the tile ([[mine-mechanics]] § team scope, archive
    2026-08-06: 1,227 enemy vs 2 friendly displacements — own-color
    mines never displace). For a pickup that is fatal: the measured
    transfer choreography needs the tank ON the container or cardinally
    adjacent, so a landing that displaces never completes the mission.
    Session bot-20260805-173034 spent 43 minutes re-aiming at a known
    mine — 534 displaced teleports, zero pickups — because the
    selector answered legality when the plan needed attainability.

    The team scoping lives in the terrain view, not here: the composed
    decision view (``FerryAwareTerrain``) answers
    ``is_landing_attainable`` from its per-tick hostile-mine set, so
    no call site ever selects a mine layer (the 2026-08-05
    all-team-mines over-reach was exactly a call site choosing the
    wrong layer).

    Args:
        terrain: Terrain view answering landing attainability.
        goal_x: Target container X coordinate.
        goal_y: Target container Y coordinate.

    Returns:
        The first service tile (target, then cardinal neighbors) the
        tank would actually stand on, or ``None`` when every service
        tile would refuse or displace the landing.
    """
    if not (_MAP_MIN <= goal_x <= _MAP_MAX and _MAP_MIN <= goal_y <= _MAP_MAX):
        return None
    if terrain.is_landing_attainable(goal_x, goal_y):
        return (goal_x, goal_y)
    for landing_x, landing_y in _collection_landing_tiles(goal_x, goal_y):
        if terrain.is_landing_attainable(landing_x, landing_y):
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
    "is_collection_reachable_within_bounds",
    "is_move_reachable_in_viewport",
]
