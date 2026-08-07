"""Terrain-aware pathfinding helpers for the AI system.

Computes paths that avoid water and rock tiles, uses Manhattan distance
as the heuristic, and exposes helper predicates for deciding whether the
server can walk directly to a target or needs waypointed path-following.
"""

from __future__ import annotations

import heapq

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.world_types import (
    PathStepDict,
    make_path_step,
)

# 4-directional movement: right, down, left, up
_DIRECTIONS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))

# Maximum A* iterations before giving up.
# Map is 256x256 = 65536 tiles. Must be large enough to find winding paths
# through narrow corridors surrounded by water/rocks.
_MAX_ITERATIONS = 65536

# Game map bounds
_MAP_MIN = 0
_MAP_MAX = 255


def find_path(
    terrain: TerrainMapProtocol,
    start_x: int,
    start_y: int,
    goal_x: int,
    goal_y: int,
    *,
    min_x: int | None = None,
    min_y: int | None = None,
    max_x: int | None = None,
    max_y: int | None = None,
) -> list[PathStepDict]:
    """Find a terrain-aware path using A* search.

    Uses Manhattan distance as heuristic and 4-directional movement.
    Only traverses tiles where terrain.is_passable() returns True --
    the terrain view is the single owner of walkability, including
    dynamic obstacles like hostile mines (composed in
    ``compose_decision_terrain``).

    Args:
        terrain: Terrain map for passability checks.
        start_x: Starting X coordinate (0-255).
        start_y: Starting Y coordinate (0-255).
        goal_x: Goal X coordinate (0-255).
        goal_y: Goal Y coordinate (0-255).
        min_x: Optional inclusive minimum X bound for traversable tiles.
        min_y: Optional inclusive minimum Y bound for traversable tiles.
        max_x: Optional inclusive maximum X bound for traversable tiles.
        max_y: Optional inclusive maximum Y bound for traversable tiles.

    Returns:
        List of PathStepDict from start to goal (inclusive of both).
        Empty list if no path found.
    """
    if not _endpoints_within_bounds(
        start_x,
        start_y,
        goal_x,
        goal_y,
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
    ):
        return []
    if start_x == goal_x and start_y == goal_y:
        return [make_path_step(start_x, start_y)]

    start = (start_x, start_y)
    goal = (goal_x, goal_y)

    # Priority queue: (f_score, tie_breaker, (x, y))
    # Tie breaker ensures deterministic ordering for equal f_scores
    open_set: list[tuple[int, int, tuple[int, int]]] = []
    counter = 0
    heapq.heappush(open_set, (_heuristic(start_x, start_y, goal_x, goal_y), counter, start))
    counter += 1

    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], int] = {start: 0}

    iterations = 0
    while open_set and iterations < _MAX_ITERATIONS:
        iterations += 1
        _, _, current = heapq.heappop(open_set)

        if current == goal:
            return _reconstruct_path(came_from, current)

        cx, cy = current
        current_g = g_score[current]

        for neighbor in _iter_reachable_neighbors(
            terrain,
            cx,
            cy,
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
        ):
            nx, ny = neighbor
            tentative_g = current_g + 1

            if tentative_g < g_score.get(neighbor, _MAX_ITERATIONS + 1):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + _heuristic(nx, ny, goal_x, goal_y)
                heapq.heappush(open_set, (f, counter, neighbor))
                counter += 1

    return []


def path_exists(
    terrain: TerrainMapProtocol,
    start_x: int,
    start_y: int,
    goal_x: int,
    goal_y: int,
    *,
    min_x: int | None = None,
    min_y: int | None = None,
    max_x: int | None = None,
    max_y: int | None = None,
) -> bool:
    """Return True when A* can find a path within optional bounds.

    Args:
        terrain: Terrain map for passability checks.
        start_x: Starting X coordinate.
        start_y: Starting Y coordinate.
        goal_x: Goal X coordinate.
        goal_y: Goal Y coordinate.
        min_x: Optional inclusive minimum X bound for traversable tiles.
        min_y: Optional inclusive minimum Y bound for traversable tiles.
        max_x: Optional inclusive maximum X bound for traversable tiles.
        max_y: Optional inclusive maximum Y bound for traversable tiles.

    Returns:
        True if a path exists, False otherwise.
    """
    path = find_path(
        terrain,
        start_x,
        start_y,
        goal_x,
        goal_y,
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
    )
    return len(path) > 0


def is_direct_path_clear(
    terrain: TerrainMapProtocol,
    start_x: int,
    start_y: int,
    goal_x: int,
    goal_y: int,
) -> bool:
    """Check whether a straight server-side walk can reach the goal.

    The game can move directly toward a target across open ground, but it does
    not reliably path around obstacles. This helper traces the straight line
    between the endpoints and requires every traversed tile except the start to
    be passable.

    Args:
        terrain: Terrain map for passability checks.
        start_x: Starting X coordinate.
        start_y: Starting Y coordinate.
        goal_x: Goal X coordinate.
        goal_y: Goal Y coordinate.

    Returns:
        True if the direct line is fully passable, False otherwise.
    """
    for x, y in _bresenham_line(start_x, start_y, goal_x, goal_y):
        if x == start_x and y == start_y:
            continue
        if not terrain.is_passable(x, y):
            return False
    return True


def find_path_segment_target(
    terrain: TerrainMapProtocol,
    start_x: int,
    start_y: int,
    goal_x: int,
    goal_y: int,
    *,
    min_x: int | None = None,
    min_y: int | None = None,
    max_x: int | None = None,
    max_y: int | None = None,
) -> tuple[int, int] | None:
    """Return the farthest directly walkable waypoint from an A* path.

    When a direct server-side walk is blocked by terrain but an A* path exists,
    the planner should walk as far along that path as the server can still
    execute as one direct movement command. This avoids tiny first-turn
    segments that force a full replan every 1-2 tiles while still avoiding
    impossible final targets hidden behind terrain.

    Args:
        terrain: Terrain map for passability checks.
        start_x: Starting X coordinate.
        start_y: Starting Y coordinate.
        goal_x: Goal X coordinate.
        goal_y: Goal Y coordinate.
        min_x: Optional inclusive minimum X bound for a usable waypoint.
        min_y: Optional inclusive minimum Y bound for a usable waypoint.
        max_x: Optional inclusive maximum X bound for a usable waypoint.
        max_y: Optional inclusive maximum Y bound for a usable waypoint.

    Returns:
        A waypoint tuple for the farthest directly walkable chunk, or None if
        no path exists or the path has no progress beyond the start tile.
    """
    path = find_path(
        terrain,
        start_x,
        start_y,
        goal_x,
        goal_y,
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
    )
    if len(path) <= 1:
        return None

    best_step: tuple[int, int] | None = None
    for step in path[1:]:
        candidate_x = step["x"]
        candidate_y = step["y"]
        if not is_direct_path_clear(
            terrain,
            start_x,
            start_y,
            candidate_x,
            candidate_y,
        ):
            break
        best_step = (candidate_x, candidate_y)

    return best_step


def _endpoints_within_bounds(
    start_x: int,
    start_y: int,
    goal_x: int,
    goal_y: int,
    *,
    min_x: int | None = None,
    min_y: int | None = None,
    max_x: int | None = None,
    max_y: int | None = None,
) -> bool:
    """Return True when both endpoints lie inside the optional bounds.

    Args:
        start_x: Starting X coordinate.
        start_y: Starting Y coordinate.
        goal_x: Goal X coordinate.
        goal_y: Goal Y coordinate.
        min_x: Optional inclusive minimum X bound.
        min_y: Optional inclusive minimum Y bound.
        max_x: Optional inclusive maximum X bound.
        max_y: Optional inclusive maximum Y bound.

    Returns:
        True if both endpoints are within every provided bound.
    """
    return _is_candidate_within_bounds(
        start_x,
        start_y,
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
    ) and _is_candidate_within_bounds(
        goal_x,
        goal_y,
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
    )


def _iter_reachable_neighbors(
    terrain: TerrainMapProtocol,
    current_x: int,
    current_y: int,
    *,
    min_x: int | None = None,
    min_y: int | None = None,
    max_x: int | None = None,
    max_y: int | None = None,
) -> list[tuple[int, int]]:
    """Return walkable 4-connected neighbors for an A* expansion step.

    Args:
        terrain: Terrain map for passability checks.
        current_x: Current X coordinate.
        current_y: Current Y coordinate.
        min_x: Optional inclusive minimum X bound.
        min_y: Optional inclusive minimum Y bound.
        max_x: Optional inclusive maximum X bound.
        max_y: Optional inclusive maximum Y bound.

    Returns:
        Reachable neighbor coordinates in deterministic iteration order.
    """
    neighbors: list[tuple[int, int]] = []
    for dx, dy in _DIRECTIONS:
        nx = current_x + dx
        ny = current_y + dy
        if not (_MAP_MIN <= nx <= _MAP_MAX and _MAP_MIN <= ny <= _MAP_MAX):
            continue
        if not _is_candidate_within_bounds(
            nx,
            ny,
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
        ):
            continue
        if not terrain.is_passable(nx, ny):
            continue
        neighbors.append((nx, ny))
    return neighbors


def _heuristic(x: int, y: int, goal_x: int, goal_y: int) -> int:
    """Manhattan distance heuristic for A*.

    Args:
        x: Current X coordinate.
        y: Current Y coordinate.
        goal_x: Goal X coordinate.
        goal_y: Goal Y coordinate.

    Returns:
        Manhattan distance to goal.
    """
    return abs(x - goal_x) + abs(y - goal_y)


def _is_candidate_within_bounds(
    x: int,
    y: int,
    *,
    min_x: int | None,
    min_y: int | None,
    max_x: int | None,
    max_y: int | None,
) -> bool:
    """Return True when a candidate satisfies optional inclusive bounds.

    Args:
        x: Candidate X coordinate.
        y: Candidate Y coordinate.
        min_x: Optional inclusive minimum X bound.
        min_y: Optional inclusive minimum Y bound.
        max_x: Optional inclusive maximum X bound.
        max_y: Optional inclusive maximum Y bound.

    Returns:
        True if the candidate satisfies every provided bound.
    """
    return (
        (min_x is None or x >= min_x)
        and (min_y is None or y >= min_y)
        and (max_x is None or x <= max_x)
        and (max_y is None or y <= max_y)
    )


def _reconstruct_path(
    came_from: dict[tuple[int, int], tuple[int, int]],
    current: tuple[int, int],
) -> list[PathStepDict]:
    """Reconstruct path from A* came_from map.

    Args:
        came_from: Map of node to its predecessor.
        current: Goal node to trace back from.

    Returns:
        List of PathStepDict from start to goal (inclusive).
    """
    path: list[PathStepDict] = []
    while current in came_from:
        path.append(make_path_step(current[0], current[1]))
        current = came_from[current]
    path.append(make_path_step(current[0], current[1]))
    path.reverse()
    return path


def path_length(path: list[PathStepDict]) -> int:
    """Get the number of steps in a path.

    Args:
        path: Path from find_path.

    Returns:
        Number of steps (0 if empty).
    """
    return len(path)


def _bresenham_line(
    start_x: int,
    start_y: int,
    goal_x: int,
    goal_y: int,
) -> list[tuple[int, int]]:
    """Compute the integer line between two coordinates.

    Args:
        start_x: Starting X coordinate.
        start_y: Starting Y coordinate.
        goal_x: Goal X coordinate.
        goal_y: Goal Y coordinate.

    Returns:
        Inclusive list of integer coordinates along the line.
    """
    x = start_x
    y = start_y
    dx = abs(goal_x - start_x)
    dy = abs(goal_y - start_y)
    step_x = 1 if start_x < goal_x else -1
    step_y = 1 if start_y < goal_y else -1
    err = dx - dy

    points: list[tuple[int, int]] = []
    while True:
        points.append((x, y))
        if x == goal_x and y == goal_y:
            return points
        doubled = err * 2
        if doubled > -dy:
            err -= dy
            x += step_x
        if doubled < dx:
            err += dx
            y += step_y


__all__ = [
    "find_path",
    "find_path_segment_target",
    "is_direct_path_clear",
    "path_exists",
    "path_length",
]
