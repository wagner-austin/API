"""Equipment and container targeting for the AI system.

Pure functions that find the nearest reachable fuel and equipment containers
from world state, with optional terrain-aware pathfinding for reachability.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.pathfinding import find_path
from tankpit_bot.bot.ai.threats import manhattan_distance
from tankpit_bot.state.types import (
    ContainerStateDict,
    MineStateDict,
    SelfStateDict,
    WorldStateDict,
    coord_key,
    viewport_scan_key,
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
    viewport_scanned = is_current_viewport_scanned(world)

    for container in world["containers"].values():
        if container["is_fuel"] != want_fuel:
            continue
        total += 1
        cx, cy = container["x"], container["y"]
        if not (left <= cx <= right and top <= cy <= bottom):
            continue
        nearby += 1
        dist = manhattan_distance(sx, sy, cx, cy)
        if not viewport_scanned:
            reason = "unconfirmed_viewport"
            is_actionable = False
            is_blocked = False
            missing_landing = False
            low_volume_target = False
        else:
            reason, is_actionable, is_blocked, missing_landing, low_volume_target = (
                _describe_candidate_reason(
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
        if container["volume"] < 100:
            continue
        cx, cy = container["x"], container["y"]
        dist = manhattan_distance(sx, sy, cx, cy)
        if not _is_actionable_with_terrain(
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


def _describe_candidate_reason(
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
    if terrain is None or is_reachable(
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
    if container["is_fuel"] != want_fuel:
        return False
    if container["failed_pickups"] > 0:
        return False
    if now_ms > 0 and _is_stale(container, now_ms):
        return False
    cx, cy = container["x"], container["y"]
    left, top, right, bottom = _viewport_bounds(world)
    return left <= cx <= right and top <= cy <= bottom


def is_current_viewport_scanned(world: WorldStateDict) -> bool:
    """Return True when the current viewport has authoritative local coverage.

    Args:
        world: Current world state.

    Returns:
        True if the current viewport origin has been confirmed by fresh radar
        data or a fresh visible viewport update.
    """
    viewport = world["viewport"]
    key = viewport_scan_key(viewport["left"], viewport["top"])
    return key in world["scanned_viewports"]


def _viewport_bounds(world: WorldStateDict) -> tuple[int, int, int, int]:
    """Return inclusive observable viewport bounds from world state."""
    return viewport_visible_bounds(world["viewport"])


def _is_actionable_with_terrain(
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
    if is_reachable(terrain, start_x, start_y, goal_x, goal_y, blocked_mines):
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
    "describe_container_search",
    "find_best_fuel",
    "find_equipment_candidates",
    "find_nearest_deposit",
    "find_nearest_equipment",
    "find_nearest_fuel",
    "find_teleport_landing_tile",
    "is_current_viewport_scanned",
    "is_reachable",
]
