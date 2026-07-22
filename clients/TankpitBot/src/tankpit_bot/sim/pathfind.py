"""The simulator's server router — deterministic and quadrant-keyed.

Implements the measured routing law (wiki [[walk-mechanics]] and the
2026-07-21 pathfinder log entries): shortest path, preferring the
single-turn L whose first leg follows the quadrant rule — VERTICAL
first except toward the northeast quadrant, which goes HORIZONTAL
first (byte-identical repeated routes in the manual determinism
session). When both Ls are blocked the router falls back to a
breadth-first search whose fixed neighbor order keeps the result
deterministic, mirroring the server's terrain- and mine-forced
staircases and detours.
"""

from __future__ import annotations

from collections import deque
from typing import Protocol

_STEPS: dict[str, tuple[int, int]] = {
    "n": (0, -1),
    "s": (0, 1),
    "e": (1, 0),
    "w": (-1, 0),
}
_OPPOSITE: dict[str, str] = {"n": "s", "s": "n", "e": "w", "w": "e"}

# The game map is a bordered 256x256 grid; the router never leaves it
# regardless of what the injected predicate would say off-map.
_MAP_MAX = 255


def _on_map(x: int, y: int) -> bool:
    """Report whether a tile lies on the 256x256 game map.

    Args:
        x: Tile X.
        y: Tile Y.

    Returns:
        True when both coordinates are in [0, 255].
    """
    return 0 <= x <= _MAP_MAX and 0 <= y <= _MAP_MAX


class PassableFn(Protocol):
    """Predicate deciding whether the router may step onto a tile."""

    def __call__(self, x: int, y: int) -> bool:
        """Report whether the tile at (x, y) is enterable."""
        ...


def _axis_order(start_x: int, start_y: int, dest_x: int, dest_y: int) -> list[str]:
    """Return the four step directions in quadrant-keyed priority order.

    Args:
        start_x: Route start X.
        start_y: Route start Y.
        dest_x: Route destination X.
        dest_y: Route destination Y.

    Returns:
        Directions ordered: toward-primary, toward-secondary, then
        their opposites — vertical primary except toward the NE
        quadrant (east AND north), which is horizontal primary.
    """
    dx = dest_x - start_x
    dy = dest_y - start_y
    step_x = "e" if dx >= 0 else "w"
    step_y = "s" if dy >= 0 else "n"
    horizontal_first = dx > 0 and dy < 0
    toward = [step_x, step_y] if horizontal_first else [step_y, step_x]
    return [toward[0], toward[1], _OPPOSITE[toward[1]], _OPPOSITE[toward[0]]]


def _l_path(
    passable: PassableFn,
    start_x: int,
    start_y: int,
    dest_x: int,
    dest_y: int,
    first: str,
    second: str,
) -> str | None:
    """Try the single-turn L walking ``first``-axis steps then ``second``.

    Args:
        passable: Tile predicate.
        start_x: Route start X.
        start_y: Route start Y.
        dest_x: Route destination X.
        dest_y: Route destination Y.
        first: Direction of the first leg.
        second: Direction of the second leg.

    Returns:
        The nsew path when every stepped tile is passable, else None.
    """
    legs = (
        (first, abs(dest_x - start_x) if first in "ew" else abs(dest_y - start_y)),
        (second, abs(dest_x - start_x) if second in "ew" else abs(dest_y - start_y)),
    )
    x, y = start_x, start_y
    path = ""
    for direction, count in legs:
        step_x, step_y = _STEPS[direction]
        for _ in range(count):
            x += step_x
            y += step_y
            if not _on_map(x, y) or not passable(x, y):
                return None
            path += direction
    return path


def _bfs(
    passable: PassableFn,
    start_x: int,
    start_y: int,
    dest_x: int,
    dest_y: int,
    order: list[str],
) -> str | None:
    """Deterministic breadth-first search with a fixed neighbor order.

    Args:
        passable: Tile predicate.
        start_x: Route start X.
        start_y: Route start Y.
        dest_x: Route destination X.
        dest_y: Route destination Y.
        order: Neighbor expansion order (quadrant-keyed).

    Returns:
        A shortest nsew path, or None when the destination is
        unreachable.
    """
    came_from: dict[tuple[int, int], tuple[int, int, str]] = {}
    seen = {(start_x, start_y)}
    queue: deque[tuple[int, int]] = deque([(start_x, start_y)])
    while queue:
        x, y = queue.popleft()
        if (x, y) == (dest_x, dest_y):
            path = ""
            while (x, y) != (start_x, start_y):
                px, py, direction = came_from[(x, y)]
                path = direction + path
                x, y = px, py
            return path
        for direction in order:
            step_x, step_y = _STEPS[direction]
            nx, ny = x + step_x, y + step_y
            if (nx, ny) in seen or not _on_map(nx, ny) or not passable(nx, ny):
                continue
            seen.add((nx, ny))
            came_from[(nx, ny)] = (x, y, direction)
            queue.append((nx, ny))
    return None


def route(
    passable: PassableFn,
    start_x: int,
    start_y: int,
    dest_x: int,
    dest_y: int,
) -> str | None:
    """Route from start to destination under the quadrant-keyed law.

    Args:
        passable: Predicate over every stepped tile INCLUDING the
            destination — the caller composes terrain, mines, and
            tank occupancy into it.
        start_x: Route start X.
        start_y: Route start Y.
        dest_x: Route destination X.
        dest_y: Route destination Y.

    Returns:
        The nsew path ("" when already at the destination), or None
        when no route exists.
    """
    if (start_x, start_y) == (dest_x, dest_y):
        return ""
    order = _axis_order(start_x, start_y, dest_x, dest_y)
    primary = _l_path(passable, start_x, start_y, dest_x, dest_y, order[0], order[1])
    if primary is not None:
        return primary
    secondary = _l_path(passable, start_x, start_y, dest_x, dest_y, order[1], order[0])
    if secondary is not None:
        return secondary
    return _bfs(passable, start_x, start_y, dest_x, dest_y, order)


__all__ = [
    "PassableFn",
    "route",
]
