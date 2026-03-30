"""Equipment and container targeting for the AI system.

Pure functions that find the nearest reachable fuel and equipment containers
from world state, with optional terrain-aware pathfinding for reachability.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.pathfinding import find_path
from tankpit_bot.bot.ai.threats import manhattan_distance
from tankpit_bot.state.types import ContainerStateDict, SelfStateDict, WorldStateDict

log = get_logger(__name__)

_VIEWPORT_RADIUS = 8
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

    for container in world["containers"].values():
        if container["is_fuel"] != want_fuel:
            continue
        total += 1
        cx, cy = container["x"], container["y"]
        if abs(cx - sx) > _VIEWPORT_RADIUS or abs(cy - sy) > _VIEWPORT_RADIUS:
            continue
        nearby += 1
        dist = manhattan_distance(sx, sy, cx, cy)
        reason = "actionable"
        is_actionable = True
        if want_fuel and container["volume"] < minimum_volume:
            low_volume += 1
            reason = "low_volume"
            is_actionable = False
        elif terrain is not None and not is_reachable(terrain, sx, sy, cx, cy):
            blocked += 1
            if not allow_unreachable:
                reason = "blocked_walk"
                is_actionable = False
            elif find_teleport_landing_tile(terrain, sx, sy, cx, cy) is None:
                no_landing += 1
                reason = "blocked_no_landing"
                is_actionable = False
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

    Returns:
        True if a path exists, False if blocked by terrain.
    """
    path = find_path(terrain, start_x, start_y, goal_x, goal_y)
    return len(path) > 0


def find_teleport_landing_tile(
    terrain: TerrainMapProtocol,
    start_x: int,
    start_y: int,
    goal_x: int,
    goal_y: int,
) -> tuple[int, int] | None:
    """Find a passable adjacent tile to use as a teleport landing point.

    For terrain-locked fuel and equipment containers, the bot must teleport
    beside the target rather than onto the blocked container tile itself.
    This helper inspects the four cardinally adjacent tiles and returns the
    closest passable landing tile relative to the current position.

    Args:
        terrain: Terrain map for passability checks.
        start_x: Bot X coordinate before teleporting.
        start_y: Bot Y coordinate before teleporting.
        goal_x: Target container X coordinate.
        goal_y: Target container Y coordinate.

    Returns:
        Tuple of landing coordinates, or None if no passable adjacent tile exists.
    """
    best_tile: tuple[int, int] | None = None
    best_dist = _MAX_DIST

    for dx, dy in _ADJACENT_DIRECTIONS:
        nx = goal_x + dx
        ny = goal_y + dy
        if not (_MAP_MIN <= nx <= _MAP_MAX and _MAP_MIN <= ny <= _MAP_MAX):
            continue
        if not terrain.is_passable(nx, ny):
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
            sx,
            sy,
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
    best: ContainerStateDict | None = None
    best_dist = _MAX_DIST

    sx, sy = self_state["x"], self_state["y"]
    for container in world["containers"].values():
        if not _is_visible_candidate(
            container,
            sx,
            sy,
            want_fuel=False,
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
            ):
                continue
            best_dist = dist
            best = container

    return best


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
            sx,
            sy,
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


def _is_visible_candidate(
    container: ContainerStateDict,
    self_x: int,
    self_y: int,
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
    return abs(cx - self_x) <= _VIEWPORT_RADIUS and abs(cy - self_y) <= _VIEWPORT_RADIUS


def _is_actionable_with_terrain(
    terrain: TerrainMapProtocol | None,
    start_x: int,
    start_y: int,
    goal_x: int,
    goal_y: int,
    *,
    allow_unreachable: bool,
) -> bool:
    """Return True when walkable directly or via teleport fallback."""
    if terrain is None:
        return True
    if is_reachable(terrain, start_x, start_y, goal_x, goal_y):
        return True
    if not allow_unreachable:
        return False
    return find_teleport_landing_tile(terrain, start_x, start_y, goal_x, goal_y) is not None


__all__ = [
    "describe_container_search",
    "find_best_fuel",
    "find_nearest_deposit",
    "find_nearest_equipment",
    "find_nearest_fuel",
    "find_teleport_landing_tile",
    "is_reachable",
]
