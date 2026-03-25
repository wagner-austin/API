"""Terrain-aware A* pathfinding for the AI system.

Computes paths that avoid water and rock tiles, using Manhattan distance
as the heuristic. Paths are returned as lists of PathStepDict.
"""

from __future__ import annotations

import heapq

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.types import PathStepDict, make_path_step

# 4-directional movement: right, down, left, up
_DIRECTIONS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))

# Maximum A* iterations before giving up
_MAX_ITERATIONS = 10000

# Game map bounds
_MAP_MIN = 0
_MAP_MAX = 255


def find_path(
    terrain: TerrainMapProtocol,
    start_x: int,
    start_y: int,
    goal_x: int,
    goal_y: int,
) -> list[PathStepDict]:
    """Find a terrain-aware path using A* search.

    Uses Manhattan distance as heuristic and 4-directional movement.
    Only traverses tiles where terrain.is_passable() returns True.
    Returns empty list if no path exists within iteration limit.

    Args:
        terrain: Terrain map for passability checks.
        start_x: Starting X coordinate (0-255).
        start_y: Starting Y coordinate (0-255).
        goal_x: Goal X coordinate (0-255).
        goal_y: Goal Y coordinate (0-255).

    Returns:
        List of PathStepDict from start to goal (inclusive of both).
        Empty list if no path found.
    """
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

        for dx, dy in _DIRECTIONS:
            nx, ny = cx + dx, cy + dy

            if not (_MAP_MIN <= nx <= _MAP_MAX and _MAP_MIN <= ny <= _MAP_MAX):
                continue

            if not terrain.is_passable(nx, ny):
                continue

            neighbor = (nx, ny)
            tentative_g = current_g + 1

            if tentative_g < g_score.get(neighbor, _MAX_ITERATIONS + 1):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + _heuristic(nx, ny, goal_x, goal_y)
                heapq.heappush(open_set, (f, counter, neighbor))
                counter += 1

    return []


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


__all__ = [
    "find_path",
    "path_length",
]
