"""Tile-level scan coverage tracking for equipment foraging.

The bot tracks which TILES it has scanned, keyed by ``"x,y"`` with the
scan timestamp. Both the built-in 5x5 radar and the inventory-consuming
extra radar reveal tiles only inside the visible viewport (the radar
never sees past viewport edges -- user-confirmed game mechanic
2026-06-21). Each scan therefore marks the **intersection** of the
radar's footprint with the viewport bounds:

* Free 5x5 radar at tank ``(X, Y)``: marks tiles
  ``(X-2..X+2, Y-2..Y+2) intersected with the viewport``.
  At a viewport corner this can be as few as ~3 tiles, not 25.
* Extra radar: marks every tile inside the viewport bounds.

The forager uses this map to:

* Decide whether the current viewport is worth another scan
  (``is_viewport_fully_covered`` returns True → no scan, just move
  toward unscanned tiles or teleport away).
* Pick a walking destination -- the nearest unscanned tile inside the
  current viewport, since walking off the viewport requires a
  teleport in this game configuration.

Game-config note: viewport shifting is OFF in this session, so once
the bot teleports the viewport is fixed until the next teleport.
Tiles outside the current viewport remain in the map until the TTL
expires; they never become reachable by walking.

This module is a pure leaf -- no AI or world-state imports.
"""

from __future__ import annotations

# A swept tile is re-foraged after this interval -- long enough to push
# the sweep across the viewport before doubling back, short enough that
# equipment that respawns later is eventually re-discovered.
FORAGE_COVERAGE_TTL_MS = 180000

# Built-in radar reveals a 5x5 around the tank (chebyshev radius 2,
# wire-verified 2026-06-12). Bounds are inclusive.
FREE_RADAR_RADIUS = 2


def tile_key(x: int, y: int) -> str:
    """Return the dict key for a tile.

    Args:
        x: Tile X.
        y: Tile Y.

    Returns:
        ``"x,y"`` string used as the dict key.
    """
    return f"{x},{y}"


def is_tile_covered(
    local_scan_tiles: dict[str, int],
    x: int,
    y: int,
    now_ms: int,
) -> bool:
    """Return True when ``(x, y)`` carries a live scan mark.

    Args:
        local_scan_tiles: Coverage map keyed by ``"x,y"`` -> scan ms.
        x: Tile X.
        y: Tile Y.
        now_ms: Current timestamp for TTL evaluation.

    Returns:
        True if the tile was scanned within :data:`FORAGE_COVERAGE_TTL_MS`.
    """
    scanned_ms = local_scan_tiles.get(tile_key(x, y))
    return scanned_ms is not None and now_ms - scanned_ms <= FORAGE_COVERAGE_TTL_MS


def record_tile_scan(
    local_scan_tiles: dict[str, int],
    scanned_tiles: list[tuple[int, int]],
    now_ms: int,
) -> dict[str, int]:
    """Return the coverage map with ``scanned_tiles`` marked + pruned.

    Adds every tile in ``scanned_tiles`` at ``now_ms`` and drops any
    existing entry that has aged past the coverage TTL so the map
    cannot grow without bound.

    Args:
        local_scan_tiles: Existing coverage map.
        scanned_tiles: Tiles that were just scanned.
        now_ms: Scan timestamp recorded for each tile.

    Returns:
        New coverage map.
    """
    pruned = {
        key: scanned_ms
        for key, scanned_ms in local_scan_tiles.items()
        if now_ms - scanned_ms <= FORAGE_COVERAGE_TTL_MS
    }
    for tx, ty in scanned_tiles:
        pruned[tile_key(tx, ty)] = now_ms
    return pruned


def viewport_tiles(
    viewport_left: int,
    viewport_top: int,
    viewport_right: int,
    viewport_bottom: int,
) -> list[tuple[int, int]]:
    """Return every tile inside the viewport bounds (inclusive).

    Args:
        viewport_left: Viewport left X (inclusive).
        viewport_top: Viewport top Y (inclusive).
        viewport_right: Viewport right X (inclusive).
        viewport_bottom: Viewport bottom Y (inclusive).

    Returns:
        List of ``(x, y)`` tile tuples in row-major order.
    """
    return [
        (x, y)
        for y in range(viewport_top, viewport_bottom + 1)
        for x in range(viewport_left, viewport_right + 1)
    ]


def free_radar_revealed_tiles(
    tank_x: int,
    tank_y: int,
    viewport_left: int,
    viewport_top: int,
    viewport_right: int,
    viewport_bottom: int,
) -> list[tuple[int, int]]:
    """Return tiles a built-in 5x5 radar reveals from ``(tank_x, tank_y)``.

    The radar footprint is the 5x5 block centered on the tank, then
    intersected with the viewport -- the radar does not see past
    viewport edges.

    Args:
        tank_x: Tank X tile.
        tank_y: Tank Y tile.
        viewport_left: Viewport left X (inclusive).
        viewport_top: Viewport top Y (inclusive).
        viewport_right: Viewport right X (inclusive).
        viewport_bottom: Viewport bottom Y (inclusive).

    Returns:
        Tiles the free radar reveals from the tank's position.
    """
    left = max(tank_x - FREE_RADAR_RADIUS, viewport_left)
    right = min(tank_x + FREE_RADAR_RADIUS, viewport_right)
    top = max(tank_y - FREE_RADAR_RADIUS, viewport_top)
    bottom = min(tank_y + FREE_RADAR_RADIUS, viewport_bottom)
    if left > right or top > bottom:
        return []
    return [(x, y) for y in range(top, bottom + 1) for x in range(left, right + 1)]


def is_viewport_fully_covered(
    local_scan_tiles: dict[str, int],
    viewport_left: int,
    viewport_top: int,
    viewport_right: int,
    viewport_bottom: int,
    now_ms: int,
) -> bool:
    """Return True when every tile in the viewport carries a live scan mark.

    Args:
        local_scan_tiles: Coverage map.
        viewport_left: Viewport left X (inclusive).
        viewport_top: Viewport top Y (inclusive).
        viewport_right: Viewport right X (inclusive).
        viewport_bottom: Viewport bottom Y (inclusive).
        now_ms: Current timestamp for TTL evaluation.

    Returns:
        True when no viewport tile is unscanned.
    """
    for y in range(viewport_top, viewport_bottom + 1):
        for x in range(viewport_left, viewport_right + 1):
            if not is_tile_covered(local_scan_tiles, x, y, now_ms):
                return False
    return True


def _free_radar_new_coverage(
    local_scan_tiles: dict[str, int],
    tile_x: int,
    tile_y: int,
    viewport_left: int,
    viewport_top: int,
    viewport_right: int,
    viewport_bottom: int,
    now_ms: int,
) -> int:
    """Return how many uncovered tiles a free radar at ``(tile_x, tile_y)`` would reveal.

    The free radar footprint is the 5x5 block centered on the tank,
    intersected with the viewport (see :func:`free_radar_revealed_tiles`).
    This helper counts the subset of that footprint that is *not* in
    fresh scan coverage -- the actual coverage gain from walking the
    tank to ``(tile_x, tile_y)`` and firing the free radar next tick.

    Args:
        local_scan_tiles: Current tile coverage map.
        tile_x: Candidate destination X.
        tile_y: Candidate destination Y.
        viewport_left: Viewport left X (inclusive).
        viewport_top: Viewport top Y (inclusive).
        viewport_right: Viewport right X (inclusive).
        viewport_bottom: Viewport bottom Y (inclusive).
        now_ms: Current timestamp for TTL evaluation.

    Returns:
        Count of uncovered tiles inside the 5x5 footprint clipped to
        the viewport.
    """
    x_lo = max(tile_x - FREE_RADAR_RADIUS, viewport_left)
    x_hi = min(tile_x + FREE_RADAR_RADIUS, viewport_right)
    y_lo = max(tile_y - FREE_RADAR_RADIUS, viewport_top)
    y_hi = min(tile_y + FREE_RADAR_RADIUS, viewport_bottom)
    count = 0
    for y in range(y_lo, y_hi + 1):
        for x in range(x_lo, x_hi + 1):
            if not is_tile_covered(local_scan_tiles, x, y, now_ms):
                count += 1
    return count


def select_best_free_radar_position(
    local_scan_tiles: dict[str, int],
    tank_x: int,
    tank_y: int,
    viewport_left: int,
    viewport_top: int,
    viewport_right: int,
    viewport_bottom: int,
    now_ms: int,
) -> tuple[int, int] | None:
    """Return the viewport tile whose next free radar would reveal the most uncovered ground.

    Picks the destination that maximises the next-radar coverage gain
    rather than the nearest unscanned tile -- the optimal walk step for
    the free-radar tile-expansion strategy is ~5 tiles (matching the
    5x5 radar diameter), not 1. Ties are broken by Manhattan distance
    from the tank (closer wins, to avoid pointless long walks).

    Returns ``None`` when no destination in the viewport would reveal
    any new ground -- the caller should treat the viewport as scanned
    and fall through to a teleport hop.

    Args:
        local_scan_tiles: Current tile coverage map.
        tank_x: Tank X.
        tank_y: Tank Y.
        viewport_left: Viewport left X (inclusive).
        viewport_top: Viewport top Y (inclusive).
        viewport_right: Viewport right X (inclusive).
        viewport_bottom: Viewport bottom Y (inclusive).
        now_ms: Current timestamp for TTL evaluation.

    Returns:
        ``(x, y)`` of the highest-scoring destination, or ``None`` when
        every viewport position would reveal zero new tiles.
    """
    best: tuple[int, int] | None = None
    best_score = 0
    best_dist = 0
    for y in range(viewport_top, viewport_bottom + 1):
        for x in range(viewport_left, viewport_right + 1):
            score = _free_radar_new_coverage(
                local_scan_tiles,
                x,
                y,
                viewport_left,
                viewport_top,
                viewport_right,
                viewport_bottom,
                now_ms,
            )
            if score == 0:
                continue
            dist = abs(x - tank_x) + abs(y - tank_y)
            if best is None or score > best_score or (score == best_score and dist < best_dist):
                best = (x, y)
                best_score = score
                best_dist = dist
    return best


__all__ = [
    "FORAGE_COVERAGE_TTL_MS",
    "FREE_RADAR_RADIUS",
    "free_radar_revealed_tiles",
    "is_tile_covered",
    "is_viewport_fully_covered",
    "record_tile_scan",
    "select_best_free_radar_position",
    "tile_key",
    "viewport_tiles",
]
