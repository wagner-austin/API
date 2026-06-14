"""Equipment and container targeting for the AI system.

Pure functions that find the nearest reachable fuel and equipment containers
from world state, with optional terrain-aware pathfinding for reachability.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.pathfinding import find_path
from tankpit_bot.bot.ai.reachability import is_collection_reachable_in_viewport
from tankpit_bot.bot.ai.threats import manhattan_distance
from tankpit_bot.state.types import (
    ContainerStateDict,
    MineStateDict,
    SelfStateDict,
    WorldStateDict,
    coord_key,
)
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds

log = get_logger(__name__)

_MAP_MIN = 0
_MAP_MAX = 255
_ADJACENT_DIRECTIONS: tuple[tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Containers older than this are considered stale and skipped.
# Radar/viewport updates refresh the timestamp. 30 seconds is generous
# enough for stable containers but short enough to reject ghost targets
# that were picked up by other players.
_CONTAINER_FRESHNESS_TTL_MS = 30000

# A locked resource target is released only when a same-kind candidate
# is at most HALF the locked distance AND at least this many tiles
# closer. The minimum gap keeps near-equal candidates from oscillating
# the lock (the churn the lock exists to prevent); the halving rule
# stops the bot from crossing the map past abundant nearby resources
# (observed in live run 20260610-011x).
_LOCK_RELEASE_MIN_GAP = 10

# Known containers beyond this Manhattan distance are not worth walking
# to: three viewport widths is ~96 seconds of travel at one tile per
# tick, by which point the 30s freshness TTL guarantees the belief is
# stale on arrival. Live run 20260610 walked across the map to a
# container drained long before ("Empty container" x8); local search is
# strictly better past this radius.
_KNOWN_PURSUIT_MAX_DIST = 48

# Radar coverage is keyed by exact viewport origin, but the viewport
# shifts a tile or two with every walk. A scan whose origin is within
# this offset still covers nearly the whole current viewport, so it
# counts -- otherwise corner-hopping re-radars 95%-overlapping ground
# (live run 20260610: same corner scanned repeatedly).
_SCAN_COVERAGE_OVERLAP_TILES = 4

# Coverage older than this no longer vetoes a rescan: containers spawn
# and drain continuously, so a scan is only authoritative briefly.
_SCAN_COVERAGE_TTL_MS = 45000

#: Public alias for cross-module consumers (recover_fuel_mode).
SCAN_COVERAGE_TTL_MS = _SCAN_COVERAGE_TTL_MS


def is_lock_release_warranted(
    self_state: SelfStateDict,
    locked_x: int,
    locked_y: int,
    candidate_x: int,
    candidate_y: int,
) -> bool:
    """Return True when a candidate is enough closer to drop a locked target.

    Args:
        self_state: Player state for the distance origin.
        locked_x: Locked target X coordinate.
        locked_y: Locked target Y coordinate.
        candidate_x: Fresh candidate X coordinate.
        candidate_y: Fresh candidate Y coordinate.

    Returns:
        True when the candidate is at most half the locked distance and
        at least ``_LOCK_RELEASE_MIN_GAP`` tiles closer.
    """
    sx, sy = self_state["x"], self_state["y"]
    locked_dist = manhattan_distance(sx, sy, locked_x, locked_y)
    candidate_dist = manhattan_distance(sx, sy, candidate_x, candidate_y)
    if candidate_dist * 2 > locked_dist:
        return False
    return locked_dist - candidate_dist >= _LOCK_RELEASE_MIN_GAP


def describe_container_search(
    world: WorldStateDict,
    self_state: SelfStateDict,
    terrain: TerrainMapProtocol | None = None,
    *,
    want_fuel: bool,
    allow_unreachable: bool,
    minimum_volume: int = 0,
) -> str:
    """Summarize why container targeting did or did not find an actionable target.

    Args:
        world: Current world state with container positions.
        self_state: Player's own state for position.
        terrain: Optional terrain map for reachability checks.
        want_fuel: True to inspect fuel containers, False for equipment.
        allow_unreachable: Whether teleport fallback is allowed.
        minimum_volume: Minimum fuel volume for a candidate to count as actionable.

    Returns:
        Compact diagnostic string suitable for bot logs.
    """
    sx, sy = self_state["x"], self_state["y"]
    total = 0
    nearby = 0
    actionable = 0
    low_volume = 0
    blocked = 0
    no_landing = 0
    nearest_desc = "none"
    nearest_dist = _MAX_DIST
    left, top, right, bottom = _viewport_bounds(world)
    for container in world["containers"].values():
        if container["is_fuel"] != want_fuel:
            continue
        total += 1
        cx, cy = container["x"], container["y"]
        if not (left <= cx <= right and top <= cy <= bottom):
            continue
        nearby += 1
        dist = manhattan_distance(sx, sy, cx, cy)
        reason, is_actionable, is_blocked, missing_landing, low_volume_target = (
            _describe_candidate_reason(
                world,
                container,
                sx,
                sy,
                terrain,
                want_fuel=want_fuel,
                allow_unreachable=allow_unreachable,
                minimum_volume=minimum_volume,
                blocked_mines=world["mines"],
            )
        )
        if is_blocked:
            blocked += 1
        if missing_landing:
            no_landing += 1
        if low_volume_target:
            low_volume += 1
        if is_actionable:
            actionable += 1
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_desc = f"({cx},{cy}) {reason}"

    target_kind = "fuel" if want_fuel else "equipment"
    return (
        f"{target_kind}: total={total} nearby={nearby} actionable={actionable} "
        f"blocked={blocked} no_landing={no_landing} low_volume={low_volume} "
        f"nearest={nearest_desc}"
    )


def is_reachable(
    terrain: TerrainMapProtocol,
    start_x: int,
    start_y: int,
    goal_x: int,
    goal_y: int,
    blocked_mines: dict[str, MineStateDict] | None = None,
) -> bool:
    """Check if a target is reachable via terrain-aware pathfinding.

    Uses A* pathfinding to determine if a walkable path exists between
    start and goal positions, accounting for rocks and water.

    Args:
        terrain: Terrain map for passability checks.
        start_x: Starting X coordinate.
        start_y: Starting Y coordinate.
        goal_x: Goal X coordinate.
        goal_y: Goal Y coordinate.
        blocked_mines: Optional known mines indexed by coordinate.

    Returns:
        True if a path exists, False if blocked by terrain.
    """
    path = find_path(
        terrain,
        start_x,
        start_y,
        goal_x,
        goal_y,
        blocked_mines.keys() if blocked_mines is not None else None,
    )
    return len(path) > 0


def find_teleport_landing_tile(
    terrain: TerrainMapProtocol,
    start_x: int,
    start_y: int,
    goal_x: int,
    goal_y: int,
    blocked_mines: dict[str, MineStateDict] | None = None,
) -> tuple[int, int] | None:
    """Find a legal teleport landing point for a container target.

    Containers may be teleported onto directly when the container tile itself
    is passable land and not mined. When the target tile is not directly
    landable, fall back to the nearest passable cardinally adjacent tile.

    Args:
        terrain: Terrain map for passability checks.
        start_x: Bot X coordinate before teleporting.
        start_y: Bot Y coordinate before teleporting.
        goal_x: Target container X coordinate.
        goal_y: Target container Y coordinate.
        blocked_mines: Optional known mines indexed by coordinate.

    Returns:
        Tuple of landing coordinates, or None if no safe landing tile exists.
    """
    if terrain.is_passable(goal_x, goal_y):
        target_key = coord_key(goal_x, goal_y)
        if blocked_mines is None or target_key not in blocked_mines:
            return (goal_x, goal_y)

    best_tile: tuple[int, int] | None = None
    best_dist = _MAX_DIST

    for dx, dy in _ADJACENT_DIRECTIONS:
        nx = goal_x + dx
        ny = goal_y + dy
        if not (_MAP_MIN <= nx <= _MAP_MAX and _MAP_MIN <= ny <= _MAP_MAX):
            continue
        if not terrain.is_passable(nx, ny):
            continue
        if blocked_mines is not None and coord_key(nx, ny) in blocked_mines:
            continue
        dist = manhattan_distance(start_x, start_y, nx, ny)
        if dist < best_dist:
            best_dist = dist
            best_tile = (nx, ny)

    return best_tile


def find_nearest_fuel(
    world: WorldStateDict,
    self_state: SelfStateDict,
    terrain: TerrainMapProtocol | None = None,
    *,
    allow_unreachable: bool = False,
    now_ms: int = 0,
) -> ContainerStateDict | None:
    """Find the nearest reachable fuel container.

    When terrain is provided, skips containers that are unreachable
    due to terrain obstacles (rocks, water). Without terrain, falls
    back to Manhattan distance only.

    Args:
        world: Current world state with container positions.
        self_state: Player's own state for position.
        terrain: Optional terrain map for reachability checks.
        allow_unreachable: Whether teleport fallback is allowed.
        now_ms: Current timestamp for freshness filtering. 0 disables.

    Returns:
        Nearest reachable fuel ContainerStateDict, or None if none visible.
    """
    best: ContainerStateDict | None = None
    best_dist = _MAX_DIST

    sx, sy = self_state["x"], self_state["y"]
    for container in world["containers"].values():
        if not _is_visible_candidate(
            container,
            world,
            want_fuel=True,
            now_ms=now_ms,
        ):
            continue
        cx, cy = container["x"], container["y"]
        dist = manhattan_distance(sx, sy, cx, cy)
        if dist < best_dist:
            if not _is_actionable_with_terrain(
                world,
                terrain,
                sx,
                sy,
                cx,
                cy,
                allow_unreachable=allow_unreachable,
                blocked_mines=world["mines"],
            ):
                continue
            best_dist = dist
            best = container

    return best


def find_nearest_equipment(
    world: WorldStateDict,
    self_state: SelfStateDict,
    terrain: TerrainMapProtocol | None = None,
    *,
    allow_unreachable: bool = False,
    now_ms: int = 0,
) -> ContainerStateDict | None:
    """Find the nearest reachable equipment container.

    When terrain is provided, skips containers that are unreachable
    due to terrain obstacles (rocks, water). Without terrain, falls
    back to Manhattan distance only.

    Args:
        world: Current world state with container positions.
        self_state: Player's own state for position.
        terrain: Optional terrain map for reachability checks.
        allow_unreachable: Whether teleport fallback is allowed.
        now_ms: Current timestamp for freshness filtering. 0 disables.

    Returns:
        Nearest reachable equipment ContainerStateDict, or None if none visible.
    """
    candidates = find_equipment_candidates(
        world,
        self_state,
        terrain,
        allow_unreachable=allow_unreachable,
        now_ms=now_ms,
    )
    if not candidates:
        return None
    return candidates[0]


def find_equipment_candidates(
    world: WorldStateDict,
    self_state: SelfStateDict,
    terrain: TerrainMapProtocol | None = None,
    *,
    allow_unreachable: bool = False,
    now_ms: int = 0,
) -> list[ContainerStateDict]:
    """Return visible equipment candidates ordered nearest-first.

    Args:
        world: Current world state with container positions.
        self_state: Player's own state for position.
        terrain: Optional terrain map for reachability checks.
        allow_unreachable: Whether teleport fallback is allowed.
        now_ms: Current timestamp for freshness filtering. 0 disables.

    Returns:
        List of visible equipment containers ordered by Manhattan distance.
    """
    candidates: list[tuple[int, ContainerStateDict]] = []
    sx, sy = self_state["x"], self_state["y"]
    for container in world["containers"].values():
        if not _is_visible_candidate(
            container,
            world,
            want_fuel=False,
            now_ms=now_ms,
        ):
            continue
        cx, cy = container["x"], container["y"]
        if not _is_actionable_with_terrain(
            world,
            terrain,
            sx,
            sy,
            cx,
            cy,
            allow_unreachable=allow_unreachable,
            blocked_mines=world["mines"],
        ):
            continue
        candidates.append((manhattan_distance(sx, sy, cx, cy), container))
    candidates.sort(key=_equipment_candidate_distance)
    return [container for _, container in candidates]


def find_best_fuel(
    world: WorldStateDict,
    self_state: SelfStateDict,
    terrain: TerrainMapProtocol | None = None,
    *,
    allow_unreachable: bool = False,
    now_ms: int = 0,
    minimum_volume: int = 100,
) -> ContainerStateDict | None:
    """Find the best fuel container, prioritizing volume over distance.

    When fuel is critical, a nearby low-volume container (200 fuel) may not
    be enough to recover. This function scores containers by volume first,
    then proximity, so the bot prefers a 1000-fuel container slightly
    farther away over a 200-fuel container nearby.

    Score = volume - distance, so high-volume nearby containers win.

    Args:
        world: Current world state with container positions.
        self_state: Player's own state for position.
        terrain: Optional terrain map for reachability checks.
        allow_unreachable: Whether teleport fallback is allowed.
        now_ms: Current timestamp for freshness filtering. 0 disables.
        minimum_volume: Minimum fuel volume that counts as actionable.

    Returns:
        Best fuel ContainerStateDict, or None if none visible.
    """
    best: ContainerStateDict | None = None
    best_score = -_MAX_DIST
    sx, sy = self_state["x"], self_state["y"]

    for container in world["containers"].values():
        if not _is_visible_candidate(
            container,
            world,
            want_fuel=True,
            now_ms=now_ms,
        ):
            continue
        if container["volume"] < minimum_volume:
            continue
        cx, cy = container["x"], container["y"]
        dist = manhattan_distance(sx, sy, cx, cy)
        if not _is_actionable_with_terrain(
            world,
            terrain,
            sx,
            sy,
            cx,
            cy,
            allow_unreachable=allow_unreachable,
            blocked_mines=world["mines"],
        ):
            continue
        # Score: volume is more important than proximity (teleport handles terrain)
        score = container["volume"] - dist
        if score > best_score:
            best_score = score
            best = container

    return best


def find_known_fuel_candidates(
    world: WorldStateDict,
    self_state: SelfStateDict,
    *,
    now_ms: int = 0,
    minimum_volume: int = 100,
) -> list[ContainerStateDict]:
    """Return known fuel containers ordered best-first across the full registry.

    Args:
        world: Current world state with tracked containers.
        self_state: Player state used for distance scoring.
        now_ms: Current timestamp for freshness filtering. ``0`` disables TTL.
        minimum_volume: Minimum fuel volume worth pursuing.

    Returns:
        Known fuel containers ordered by ``volume - distance`` descending.
    """
    sx, sy = self_state["x"], self_state["y"]
    scored: list[tuple[int, int, ContainerStateDict]] = []
    for container in world["containers"].values():
        if not is_container_pursuable(container, want_fuel=True, now_ms=now_ms):
            continue
        if container["volume"] < minimum_volume:
            continue
        dist = manhattan_distance(sx, sy, container["x"], container["y"])
        if dist > _KNOWN_PURSUIT_MAX_DIST:
            continue
        scored.append((container["volume"] - dist, dist, container))
    scored.sort(key=_known_fuel_candidate_key)
    return [container for _, _, container in scored]


def find_known_equipment_candidates(
    world: WorldStateDict,
    self_state: SelfStateDict,
    *,
    now_ms: int = 0,
) -> list[ContainerStateDict]:
    """Return known equipment containers ordered nearest-first.

    Args:
        world: Current world state with tracked containers.
        self_state: Player state used for distance ordering.
        now_ms: Current timestamp for freshness filtering. ``0`` disables TTL.

    Returns:
        Known equipment containers ordered by Manhattan distance.
    """
    sx, sy = self_state["x"], self_state["y"]
    candidates: list[tuple[int, ContainerStateDict]] = []
    for container in world["containers"].values():
        if not is_container_pursuable(container, want_fuel=False, now_ms=now_ms):
            continue
        dist = manhattan_distance(sx, sy, container["x"], container["y"])
        if dist > _KNOWN_PURSUIT_MAX_DIST:
            continue
        candidates.append((dist, container))
    candidates.sort(key=_equipment_candidate_distance)
    return [container for _, container in candidates]


def find_adjacent_container(
    world: WorldStateDict,
    self_state: SelfStateDict,
    terrain: TerrainMapProtocol | None,
    *,
    want_fuel: bool,
    now_ms: int,
) -> ContainerStateDict | None:
    """Return a fresh, reachable container of the requested kind within one tile.

    Used for opportunistic cross-kind pickups: a recovery mode hunting
    one resource kind should not walk straight past the other kind when
    it is standing next to it (live run 20260610-011x ignored adjacent
    equipment during fuel search). Adjacency alone is NOT pickability:
    a diagonal neighbor across a water gap is one tile away and
    unreachable (live run 20260611-000x dispatched a pickup at
    (129,152) from (128,153) that the server rejected -- the bot's own
    A* already knew it), so the SAME terrain-reachability predicate as
    candidate selection applies here.

    Args:
        world: Current world state with container positions.
        self_state: Player's own state for adjacency.
        terrain: Terrain map for reachability; ``None`` skips the check.
        want_fuel: True to look for fuel, False for equipment.
        now_ms: Current timestamp for freshness filtering.

    Returns:
        An adjacent fresh reachable container of the requested kind, or
        ``None``.
    """
    sx, sy = self_state["x"], self_state["y"]
    for container in world["containers"].values():
        if not is_container_pursuable(container, want_fuel=want_fuel, now_ms=now_ms):
            continue
        if abs(container["x"] - sx) > 1 or abs(container["y"] - sy) > 1:
            continue
        if not _is_actionable_with_terrain(
            world,
            terrain,
            sx,
            sy,
            container["x"],
            container["y"],
            allow_unreachable=False,
            blocked_mines=world["mines"],
        ):
            continue
        return container
    return None


def find_nearest_deposit(
    world: WorldStateDict,
    self_state: SelfStateDict,
    terrain: TerrainMapProtocol | None = None,
    *,
    allow_unreachable: bool = False,
) -> ContainerStateDict | None:
    """Find the nearest fuel deposit target (fuel container for depositing).

    In TankPit, fuel is deposited into fuel containers. This finds the
    nearest reachable fuel container to deposit surplus fuel.

    Args:
        world: Current world state with container positions.
        self_state: Player's own state for position.
        terrain: Optional terrain map for reachability checks.

    Returns:
        Nearest reachable fuel ContainerStateDict for depositing, or None.
    """
    return find_nearest_fuel(
        world,
        self_state,
        terrain,
        allow_unreachable=allow_unreachable,
    )


# Sentinel distance larger than any possible Manhattan distance on 256x256 map
_MAX_DIST = 512


def _is_stale(container: ContainerStateDict, now_ms: int) -> bool:
    """Return True when a container's timestamp is older than the freshness TTL.

    Args:
        container: Container to check.
        now_ms: Current timestamp in milliseconds.

    Returns:
        True if the container is stale and should be skipped.
    """
    age = now_ms - container["timestamp_ms"]
    return age > _CONTAINER_FRESHNESS_TTL_MS


def _equipment_candidate_distance(item: tuple[int, ContainerStateDict]) -> int:
    """Return the distance component of an equipment candidate tuple.

    Args:
        item: ``(distance, container)`` tuple.

    Returns:
        The distance value used for nearest-first ordering.
    """
    return item[0]


def _known_fuel_candidate_key(item: tuple[int, int, ContainerStateDict]) -> tuple[int, int]:
    """Return a stable best-first sort key for known fuel candidates."""
    return (-item[0], item[1])


def _describe_candidate_reason(
    world: WorldStateDict,
    container: ContainerStateDict,
    start_x: int,
    start_y: int,
    terrain: TerrainMapProtocol | None,
    *,
    want_fuel: bool,
    allow_unreachable: bool,
    minimum_volume: int,
    blocked_mines: dict[str, MineStateDict],
) -> tuple[str, bool, bool, bool, bool]:
    """Describe whether a visible candidate is actionable for diagnostics."""
    if container["failed_pickups"] > 0:
        return ("failed_pickup", False, False, False, False)
    if want_fuel and container["volume"] < minimum_volume:
        return ("low_volume", False, False, False, True)
    if terrain is None or is_collection_reachable_in_viewport(
        world,
        terrain,
        start_x,
        start_y,
        container["x"],
        container["y"],
        blocked_mines,
    ):
        return ("actionable", True, False, False, False)
    if not allow_unreachable:
        return ("blocked_walk", False, True, False, False)
    landing = find_teleport_landing_tile(
        terrain,
        start_x,
        start_y,
        container["x"],
        container["y"],
        blocked_mines,
    )
    if landing is None:
        return ("blocked_no_landing", False, True, True, False)
    return ("actionable", True, True, False, False)


def _is_visible_candidate(
    container: ContainerStateDict,
    world: WorldStateDict,
    *,
    want_fuel: bool,
    now_ms: int,
) -> bool:
    """Return True when a container passes type, freshness, and viewport checks."""
    if not is_container_pursuable(container, want_fuel=want_fuel, now_ms=now_ms):
        return False
    cx, cy = container["x"], container["y"]
    left, top, right, bottom = _viewport_bounds(world)
    return left <= cx <= right and top <= cy <= bottom


def is_container_pursuable(
    container: ContainerStateDict,
    *,
    want_fuel: bool,
    now_ms: int,
) -> bool:
    """Return True when a tracked container is worth pursuing at all.

    This is the SINGLE definition of pursuability: candidate selection,
    opportunistic pickups, and lock continuation must all apply it. The
    lock path previously skipped the freshness check and kept the bot
    walking to containers whose belief had long expired.

    Args:
        container: Tracked container to check.
        want_fuel: True to require fuel, False to require equipment.
        now_ms: Current timestamp for freshness filtering. ``0`` disables
            the TTL.

    Returns:
        True when the container matches the kind, has no failed pickup,
        and is within the freshness TTL.
    """
    if container["is_fuel"] != want_fuel:
        return False
    if container["failed_pickups"] > 0:
        return False
    return not (now_ms > 0 and _is_stale(container, now_ms))


def is_area_scanned(world: WorldStateDict, left: int, top: int, now_ms: int) -> bool:
    """Return True when a viewport origin has fresh overlapping scan coverage.

    Coverage is keyed by exact scan-time viewport origins, but the
    viewport shifts with every walk; a scan within
    :data:`_SCAN_COVERAGE_OVERLAP_TILES` of the queried origin still
    covers nearly the whole area and counts. Entries older than
    :data:`_SCAN_COVERAGE_TTL_MS` no longer veto a rescan.

    Args:
        world: Current world state with scan coverage records.
        left: Queried viewport left X coordinate.
        top: Queried viewport top Y coordinate.
        now_ms: Current timestamp for coverage freshness.

    Returns:
        True when a fresh, mostly-overlapping scan covers the origin.
    """
    for key, scanned_ms in world["scanned_viewports"].items():
        if now_ms - scanned_ms > _SCAN_COVERAGE_TTL_MS:
            continue
        key_left_text, _, key_top_text = key.partition(",")
        if (
            abs(int(key_left_text) - left) <= _SCAN_COVERAGE_OVERLAP_TILES
            and abs(int(key_top_text) - top) <= _SCAN_COVERAGE_OVERLAP_TILES
        ):
            return True
    return False


def is_tile_scanned(world: WorldStateDict, x: int, y: int, now_ms: int) -> bool:
    """Return True when a world tile sits inside fresh scan coverage.

    Unlike :func:`is_area_scanned`, which asks whether a viewport
    ORIGIN is close enough to a past scan to skip re-scanning, this
    asks whether a specific TILE was inside any freshly scanned
    viewport. A tile that was scanned and produced no container is
    refuted ground truth: live run 20260611-155750 oscillated between
    two stale fuel dots 7-9 tiles away for ~35s (607->414 fuel)
    because the origin-proximity check let already-seen dots count as
    fresh leads.

    Args:
        world: Current world state with scan coverage records.
        x: World tile X coordinate.
        y: World tile Y coordinate.
        now_ms: Current timestamp for coverage freshness.

    Returns:
        True when a fresh scan's viewport contained the tile.
    """
    width = world["viewport"]["width"]
    height = world["viewport"]["height"]
    for key, scanned_ms in world["scanned_viewports"].items():
        if now_ms - scanned_ms > _SCAN_COVERAGE_TTL_MS:
            continue
        key_left_text, _, key_top_text = key.partition(",")
        left = int(key_left_text)
        top = int(key_top_text)
        if left <= x < left + width and top <= y < top + height:
            return True
    return False


def is_current_viewport_scanned(world: WorldStateDict) -> bool:
    """Return True when the current viewport has authoritative local coverage.

    Args:
        world: Current world state.

    Returns:
        True if the current viewport area is covered by a fresh radar
        scan whose origin overlaps the current one.
    """
    viewport = world["viewport"]
    return is_area_scanned(world, viewport["left"], viewport["top"], world["timestamp_ms"])


def _viewport_bounds(world: WorldStateDict) -> tuple[int, int, int, int]:
    """Return inclusive observable viewport bounds from world state."""
    return viewport_visible_bounds(world["viewport"])


def _is_actionable_with_terrain(
    world: WorldStateDict,
    terrain: TerrainMapProtocol | None,
    start_x: int,
    start_y: int,
    goal_x: int,
    goal_y: int,
    *,
    allow_unreachable: bool,
    blocked_mines: dict[str, MineStateDict],
) -> bool:
    """Return True when walkable directly or via teleport fallback."""
    if terrain is None:
        return True
    if is_collection_reachable_in_viewport(
        world,
        terrain,
        start_x,
        start_y,
        goal_x,
        goal_y,
        blocked_mines,
    ):
        return True
    if not allow_unreachable:
        return False
    return (
        find_teleport_landing_tile(
            terrain,
            start_x,
            start_y,
            goal_x,
            goal_y,
            blocked_mines,
        )
        is not None
    )


__all__ = [
    "SCAN_COVERAGE_TTL_MS",
    "describe_container_search",
    "find_adjacent_container",
    "find_best_fuel",
    "find_equipment_candidates",
    "find_known_equipment_candidates",
    "find_known_fuel_candidates",
    "find_nearest_deposit",
    "find_nearest_equipment",
    "find_nearest_fuel",
    "find_teleport_landing_tile",
    "is_area_scanned",
    "is_container_pursuable",
    "is_current_viewport_scanned",
    "is_lock_release_warranted",
    "is_reachable",
    "is_tile_scanned",
]
