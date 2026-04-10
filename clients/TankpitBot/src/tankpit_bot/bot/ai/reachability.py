"""Viewport-bounded reachability helpers for movement and collection.

These helpers answer the question the live bot actually needs:
can the current visible viewport execute this command right now?
"""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.pathfinding import path_exists
from tankpit_bot.state.types import MineStateDict, WorldStateDict, coord_key
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
    blocked_mines: dict[str, MineStateDict] | None = None,
) -> bool:
    """Return True when the viewport contains a walk path to the exact tile.

    Args:
        world: Current world state with visible viewport bounds.
        terrain: Terrain map for passability checks.
        start_x: Starting X coordinate.
        start_y: Starting Y coordinate.
        goal_x: Goal X coordinate.
        goal_y: Goal Y coordinate.
        blocked_mines: Optional known mines indexed by coordinate.

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
        blocked_mines.keys() if blocked_mines is not None else None,
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
    blocked_mines: dict[str, MineStateDict] | None = None,
) -> bool:
    """Return True when a pickup can be completed from the current viewport.

    Collection commands can complete either by reaching the target tile itself
    or by reaching a safe cardinally adjacent tile when the target tile cannot
    be occupied directly.

    Args:
        world: Current world state with visible viewport bounds.
        terrain: Terrain map for passability checks.
        start_x: Starting X coordinate.
        start_y: Starting Y coordinate.
        goal_x: Pickup target X coordinate.
        goal_y: Pickup target Y coordinate.
        blocked_mines: Optional known mines indexed by coordinate.

    Returns:
        True if a collection path exists entirely inside the current viewport.
    """
    left, top, right, bottom = viewport_visible_bounds(world["viewport"])
    target_key = coord_key(goal_x, goal_y)
    if (
        terrain.is_passable(goal_x, goal_y)
        and (blocked_mines is None or target_key not in blocked_mines)
        and _viewport_path_exists(
            terrain,
            start_x,
            start_y,
            goal_x,
            goal_y,
            blocked_mines,
            left=left,
            top=top,
            right=right,
            bottom=bottom,
        )
    ):
        return True

    for landing_x, landing_y in _collection_landing_tiles(goal_x, goal_y):
        if not (left <= landing_x <= right and top <= landing_y <= bottom):
            continue
        if not terrain.is_passable(landing_x, landing_y):
            continue
        if blocked_mines is not None and coord_key(landing_x, landing_y) in blocked_mines:
            continue
        if _viewport_path_exists(
            terrain,
            start_x,
            start_y,
            landing_x,
            landing_y,
            blocked_mines,
            left=left,
            top=top,
            right=right,
            bottom=bottom,
        ):
            return True
    return False


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
    blocked_mines: dict[str, MineStateDict] | None,
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
        blocked_mines: Optional known mines indexed by coordinate.
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
        blocked_mines.keys() if blocked_mines is not None else None,
        min_x=left,
        min_y=top,
        max_x=right,
        max_y=bottom,
    )


__all__ = [
    "is_collection_reachable_in_viewport",
    "is_move_reachable_in_viewport",
]
