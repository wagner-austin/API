"""Per-tile radar coverage primitives and the world-state mutator.

The bot tracks which TILES have been radared, keyed by ``"x,y"`` with
the scan timestamp. Both the built-in radar and the inventory-consuming
extra radar reveal tiles only inside the visible viewport (the radar
never sees past viewport edges -- user-confirmed game mechanic
2026-06-21). Each scan therefore marks the **intersection** of the
radar's footprint with the viewport bounds:

* Free radar at tank ``(X, Y)``: marks tiles
  ``(X-radius..X+radius, Y-radius..Y+radius) intersected with the
  viewport``, where ``radius = 2 + rank // 3`` (see
  :func:`tankpit_bot.physics.capacity.free_radar_radius`). At a
  viewport corner this can be as few as ~3 tiles.
* Extra radar: marks every tile inside the viewport bounds regardless
  of rank.

Coverage is written into ``WorldStateDict["scanned_tiles"]`` by the
wire-side radar handler when the server confirms a scan. The forager
reads it to:

* Decide whether the current viewport is worth another scan
  (``is_viewport_fully_covered`` returns True → no scan, just move
  toward unscanned tiles or teleport away).
* Pick a walking destination -- the viewport tile whose next free
  radar would reveal the most uncovered ground.

Game-config note: autoscroll is OFF in this session, so walking
never moves the window — it moves on teleport landings and on the
bot's own Rb scope pans ([[viewport-shift-protocol]], 2026-08-01),
both of which arrive as 0x5A and update the origin this module
reads. Tiles outside the current viewport remain in the map until
the TTL expires; they never become reachable by walking.
"""

from __future__ import annotations

from tankpit_bot.physics.capacity import free_radar_radius
from tankpit_bot.state.types import WorldStateDict

# A swept tile is re-foraged after this interval -- long enough to push
# the sweep across the viewport before doubling back, short enough that
# equipment that respawns later is eventually re-discovered.
FORAGE_COVERAGE_TTL_MS = 180000

HARVEST_MEMORY_TTL_MS = 600000
"""How long harvested/barren ground stays vetoed for collect dot hops.

The forage TTL above answers "is this ground worth another RADAR" and
deliberately ages out fast. Harvest memory answers "did we already
LEARN this ground is worthless" and must outlive it: run
bot-20260729-232252 re-hopped picked-clean viewports the moment the
180 s coverage expired (63% zero-yield hops,
[[flag-triage-20260729]] F2). Two vetoes share this window: drained
container beliefs (`_landing_viewport_known_empty`) and barren scan
memory (`is_viewport_scanned_within` — scanned recently, revealed
nothing). Container respawn cadence is unmeasured — this bound is a
working assumption, not a wire-derived law.
"""


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
    scanned_tiles: dict[str, int],
    x: int,
    y: int,
    now_ms: int,
) -> bool:
    """Return True when ``(x, y)`` carries a live scan mark.

    Args:
        scanned_tiles: Coverage map keyed by ``"x,y"`` -> scan ms.
        x: Tile X.
        y: Tile Y.
        now_ms: Current timestamp for TTL evaluation.

    Returns:
        True if the tile was scanned within :data:`FORAGE_COVERAGE_TTL_MS`.
    """
    scanned_ms = scanned_tiles.get(tile_key(x, y))
    return scanned_ms is not None and now_ms - scanned_ms <= FORAGE_COVERAGE_TTL_MS


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
    rank: int,
) -> list[tuple[int, int]]:
    """Return tiles the built-in radar reveals from ``(tank_x, tank_y)``.

    The radar footprint is a ``(2r+1)x(2r+1)`` block centered on the
    tank, where ``r = free_radar_radius(rank)``, then intersected with
    the viewport -- the radar does not see past viewport edges.

    Args:
        tank_x: Tank X tile.
        tank_y: Tank Y tile.
        viewport_left: Viewport left X (inclusive).
        viewport_top: Viewport top Y (inclusive).
        viewport_right: Viewport right X (inclusive).
        viewport_bottom: Viewport bottom Y (inclusive).
        rank: Controlled tank rank (``self_state["rank"]``, 0..8).

    Returns:
        Tiles the free radar reveals from the tank's position.
    """
    radius = free_radar_radius(rank)
    left = max(tank_x - radius, viewport_left)
    right = min(tank_x + radius, viewport_right)
    top = max(tank_y - radius, viewport_top)
    bottom = min(tank_y + radius, viewport_bottom)
    if left > right or top > bottom:
        return []
    return [(x, y) for y in range(top, bottom + 1) for x in range(left, right + 1)]


def is_viewport_untouched(
    scanned_tiles: dict[str, int],
    viewport_left: int,
    viewport_top: int,
    viewport_right: int,
    viewport_bottom: int,
    now_ms: int,
) -> bool:
    """Return True when NO tile in the viewport carries a live scan mark.

    The collect hop's "clean viewport" test (user ruling, verbatim,
    2026-07-26: "when i say it should collect on clean viewports,
    that means zero overlap"). A single live-scanned tile inside the
    bounds makes the viewport dirty — the 2026-07-18 hop gate had the
    polarity inverted (any unscanned tile counted as fresh), which
    made consecutive hops overlap ~35% of their scans (run
    bot-20260725-235637: mean 89/256 shared tiles).

    Args:
        scanned_tiles: Coverage map.
        viewport_left: Viewport left X (inclusive).
        viewport_top: Viewport top Y (inclusive).
        viewport_right: Viewport right X (inclusive).
        viewport_bottom: Viewport bottom Y (inclusive).
        now_ms: Current timestamp for TTL evaluation.

    Returns:
        True when zero viewport tiles carry a live scan mark.
    """
    for y in range(viewport_top, viewport_bottom + 1):
        for x in range(viewport_left, viewport_right + 1):
            if is_tile_covered(scanned_tiles, x, y, now_ms):
                return False
    return True


def is_viewport_fully_covered(
    scanned_tiles: dict[str, int],
    viewport_left: int,
    viewport_top: int,
    viewport_right: int,
    viewport_bottom: int,
    now_ms: int,
) -> bool:
    """Return True when every tile in the viewport carries a live scan mark.

    Args:
        scanned_tiles: Coverage map.
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
            if not is_tile_covered(scanned_tiles, x, y, now_ms):
                return False
    return True


def viewport_uncovered_count(
    scanned_tiles: dict[str, int],
    viewport_left: int,
    viewport_top: int,
    viewport_right: int,
    viewport_bottom: int,
    now_ms: int,
) -> int:
    """Return how many viewport tiles carry no live scan mark.

    The expected-reveal input for the radar-spend economics
    ([[flag-triage-20260729]] s9-2/4/5): an extra radar reveals
    exactly the uncovered tiles of the visible viewport, so this
    count IS the spend's yield.

    Args:
        scanned_tiles: Coverage map.
        viewport_left: Viewport left X (inclusive).
        viewport_top: Viewport top Y (inclusive).
        viewport_right: Viewport right X (inclusive).
        viewport_bottom: Viewport bottom Y (inclusive).
        now_ms: Current timestamp for TTL evaluation.

    Returns:
        Count of viewport tiles without live coverage.
    """
    uncovered = 0
    for y in range(viewport_top, viewport_bottom + 1):
        for x in range(viewport_left, viewport_right + 1):
            if not is_tile_covered(scanned_tiles, x, y, now_ms):
                uncovered += 1
    return uncovered


def free_radar_new_coverage(
    scanned_tiles: dict[str, int],
    tile_x: int,
    tile_y: int,
    viewport_left: int,
    viewport_top: int,
    viewport_right: int,
    viewport_bottom: int,
    now_ms: int,
    rank: int,
) -> int:
    """Return how many uncovered tiles a free radar at ``(tile_x, tile_y)`` would reveal.

    The free radar footprint is the ``(2r+1)x(2r+1)`` block centered
    on the tank (``r = free_radar_radius(rank)``), intersected with
    the viewport (see :func:`free_radar_revealed_tiles`). This helper
    counts the subset of that footprint that is *not* in fresh scan
    coverage -- the actual coverage gain from walking the tank to
    ``(tile_x, tile_y)`` and firing the free radar next tick.

    Args:
        scanned_tiles: Current tile coverage map.
        tile_x: Candidate destination X.
        tile_y: Candidate destination Y.
        viewport_left: Viewport left X (inclusive).
        viewport_top: Viewport top Y (inclusive).
        viewport_right: Viewport right X (inclusive).
        viewport_bottom: Viewport bottom Y (inclusive).
        now_ms: Current timestamp for TTL evaluation.
        rank: Controlled tank rank (``self_state["rank"]``, 0..8).

    Returns:
        Count of uncovered tiles inside the rank-scaled footprint
        clipped to the viewport.
    """
    radius = free_radar_radius(rank)
    x_lo = max(tile_x - radius, viewport_left)
    x_hi = min(tile_x + radius, viewport_right)
    y_lo = max(tile_y - radius, viewport_top)
    y_hi = min(tile_y + radius, viewport_bottom)
    count = 0
    for y in range(y_lo, y_hi + 1):
        for x in range(x_lo, x_hi + 1):
            if not is_tile_covered(scanned_tiles, x, y, now_ms):
                count += 1
    return count


def select_best_free_radar_position(
    scanned_tiles: dict[str, int],
    tank_x: int,
    tank_y: int,
    viewport_left: int,
    viewport_top: int,
    viewport_right: int,
    viewport_bottom: int,
    now_ms: int,
    rank: int,
) -> tuple[int, int] | None:
    """Return the viewport tile whose next free radar would reveal the most uncovered ground.

    Picks the destination that maximises the next-radar coverage gain
    rather than the nearest unscanned tile -- the optimal walk step
    for the free-radar tile-expansion strategy matches the radar
    diameter ``2r+1`` (5 at ranks 0-2, 7 at 3-5, 9 at 6-8), not 1.
    Ties are broken by Manhattan distance from the tank (closer wins,
    to avoid pointless long walks).

    Returns ``None`` when no destination in the viewport would reveal
    any new ground -- the caller should treat the viewport as scanned
    and fall through to a teleport hop.

    Args:
        scanned_tiles: Current tile coverage map.
        tank_x: Tank X.
        tank_y: Tank Y.
        viewport_left: Viewport left X (inclusive).
        viewport_top: Viewport top Y (inclusive).
        viewport_right: Viewport right X (inclusive).
        viewport_bottom: Viewport bottom Y (inclusive).
        now_ms: Current timestamp for TTL evaluation.
        rank: Controlled tank rank (``self_state["rank"]``, 0..8).

    Returns:
        ``(x, y)`` of the highest-scoring destination, or ``None`` when
        every viewport position would reveal zero new tiles.
    """
    best: tuple[int, int] | None = None
    best_score = 0
    best_dist = 0
    for y in range(viewport_top, viewport_bottom + 1):
        for x in range(viewport_left, viewport_right + 1):
            # The tank's own tile is never a productive walk destination:
            # the next free radar's footprint is the same whether the
            # tank moves 0 tiles or stays put, so walking here extends
            # coverage by exactly nothing. Excluding it prevents the
            # tie-break from collapsing on distance=0 when every other
            # interior tile has the same uncovered-tiles score.
            if (x, y) == (tank_x, tank_y):
                continue
            score = free_radar_new_coverage(
                scanned_tiles,
                x,
                y,
                viewport_left,
                viewport_top,
                viewport_right,
                viewport_bottom,
                now_ms,
                rank,
            )
            if score == 0:
                continue
            dist = abs(x - tank_x) + abs(y - tank_y)
            if best is None or score > best_score or (score == best_score and dist < best_dist):
                best = (x, y)
                best_score = score
                best_dist = dist
    return best


def is_viewport_scanned_within(
    scanned_tiles: dict[str, int],
    viewport_left: int,
    viewport_top: int,
    viewport_right: int,
    viewport_bottom: int,
    now_ms: int,
    *,
    ttl_ms: int,
) -> bool:
    """Return True when every on-map viewport tile was scanned within ``ttl_ms``.

    The barren-memory predicate ([[flag-triage-20260729]] F2): a
    viewport the radar fully swept within the window is KNOWN ground —
    if the sweep revealed no containers, hopping back is a guaranteed
    zero-delta scan. Distinct from :func:`is_viewport_fully_covered`,
    which asks the forage question ("worth another radar?") on the
    short :data:`FORAGE_COVERAGE_TTL_MS`; this predicate takes its
    window explicitly so harvest memory can outlive forage coverage.

    Bounds are clamped to the 0..255 map — off-map tiles can never
    carry a mark and must not make an edge viewport read unscanned. A
    viewport with no on-map tiles has no scan knowledge and returns
    False.

    Args:
        scanned_tiles: Coverage map keyed by ``"x,y"`` -> scan ms.
        viewport_left: Viewport left X (inclusive).
        viewport_top: Viewport top Y (inclusive).
        viewport_right: Viewport right X (inclusive).
        viewport_bottom: Viewport bottom Y (inclusive).
        now_ms: Current timestamp for TTL evaluation.
        ttl_ms: Scan-mark lifetime for this question.

    Returns:
        True when the whole on-map viewport was scanned within
        ``ttl_ms``.
    """
    left = max(viewport_left, 0)
    top = max(viewport_top, 0)
    right = min(viewport_right, 255)
    bottom = min(viewport_bottom, 255)
    if left > right or top > bottom:
        return False
    for y in range(top, bottom + 1):
        for x in range(left, right + 1):
            scanned_ms = scanned_tiles.get(tile_key(x, y))
            if scanned_ms is None or now_ms - scanned_ms > ttl_ms:
                return False
    return True


def record_scanned_tiles(
    state: WorldStateDict,
    scanned: list[tuple[int, int]],
    timestamp_ms: int,
) -> WorldStateDict:
    """Return state with ``scanned`` tiles marked at ``timestamp_ms`` and stale entries pruned.

    Marks are retained for :data:`HARVEST_MEMORY_TTL_MS` — longer than
    the forage TTL they serve for coverage checks — because the
    barren-memory veto must still know "we swept this ground 4 minutes
    ago" after the 180 s forage window has aged out
    ([[flag-triage-20260729]] F2). Coverage predicates keep their own
    windows; retention only bounds how long the raw marks exist.

    Args:
        state: Current world state.
        scanned: Tiles the server radar just revealed.
        timestamp_ms: Scan completion timestamp.

    Returns:
        New WorldStateDict with the coverage map updated.
    """
    pruned = {
        key: scanned_ms
        for key, scanned_ms in state["scanned_tiles"].items()
        if timestamp_ms - scanned_ms <= HARVEST_MEMORY_TTL_MS
    }
    for tx, ty in scanned:
        pruned[tile_key(tx, ty)] = timestamp_ms
    return WorldStateDict(
        self_state=state["self_state"],
        tanks=state["tanks"],
        containers=state["containers"],
        mines=state["mines"],
        terrain=state["terrain"],
        viewport=state["viewport"],
        scanned_tiles=pruned,
        timestamp_ms=timestamp_ms,
    )


__all__ = [
    "FORAGE_COVERAGE_TTL_MS",
    "HARVEST_MEMORY_TTL_MS",
    "free_radar_new_coverage",
    "free_radar_revealed_tiles",
    "is_tile_covered",
    "is_viewport_fully_covered",
    "is_viewport_scanned_within",
    "is_viewport_untouched",
    "record_scanned_tiles",
    "select_best_free_radar_position",
    "tile_key",
    "viewport_tiles",
    "viewport_uncovered_count",
]
