"""Container search — find the best fuel, equipment, or deposit target.

All ``find_*`` functions search world state for reachable containers,
optionally filtering by terrain passability. Split from ``equipment.py``
which keeps predicates and scan-coverage checks.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.equipment import (
    _KNOWN_PURSUIT_MAX_DIST,
    _viewport_bounds,
    is_container_pursuable,
)
from tankpit_bot.bot.ai.reachability import is_collection_reachable_in_viewport
from tankpit_bot.bot.ai.threats import manhattan_distance
from tankpit_bot.state.types import (
    ContainerStateDict,
    MineStateDict,
    SelfStateDict,
    WorldStateDict,
)

log = get_logger(__name__)

_MAP_MIN = 0
_MAP_MAX = 255
_ADJACENT_DIRECTIONS: tuple[tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
_MAX_DIST = 512


def find_teleport_landing_tile(
    terrain: TerrainMapProtocol,
    start_x: int,
    start_y: int,
    goal_x: int,
    goal_y: int,
    blocked_mines: dict[str, MineStateDict] | None = None,
) -> tuple[int, int] | None:
    """Find a legal teleport landing point for a container target.

    Teleports directly to the target when it is on passable ground.
    When the target is impassable (water, rock), checks cardinal
    neighbors for a passable tile. Returns None when the target and
    all neighbors are impassable (e.g. container in the middle of a
    lake) — the caller should skip this container.

    Args:
        terrain: Terrain map for passability checks.
        start_x: Bot X coordinate before teleporting.
        start_y: Bot Y coordinate before teleporting.
        goal_x: Target container X coordinate.
        goal_y: Target container Y coordinate.
        blocked_mines: Optional known mines indexed by coordinate.

    Returns:
        Tuple of landing coordinates, or None when unreachable.
    """
    del start_x, start_y, blocked_mines
    if not (_MAP_MIN <= goal_x <= _MAP_MAX and _MAP_MIN <= goal_y <= _MAP_MAX):
        return None
    if terrain.is_passable(goal_x, goal_y):
        return (goal_x, goal_y)
    for dx, dy in _ADJACENT_DIRECTIONS:
        nx, ny = goal_x + dx, goal_y + dy
        if not (_MAP_MIN <= nx <= _MAP_MAX and _MAP_MIN <= ny <= _MAP_MAX):
            continue
        if terrain.is_passable(nx, ny):
            return (nx, ny)
    return None


def is_reachable(
    terrain: TerrainMapProtocol,
    start_x: int,
    start_y: int,
    goal_x: int,
    goal_y: int,
    blocked_mines: dict[str, MineStateDict] | None = None,
) -> bool:
    """Check if a target is reachable via terrain-aware pathfinding.

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
    from tankpit_bot.bot.ai.pathfinding import find_path

    path = find_path(
        terrain,
        start_x,
        start_y,
        goal_x,
        goal_y,
        blocked_mines.keys() if blocked_mines is not None else None,
    )
    return len(path) > 0


def find_nearest_fuel(
    world: WorldStateDict,
    self_state: SelfStateDict,
    terrain: TerrainMapProtocol | None = None,
    *,
    allow_unreachable: bool = False,
    now_ms: int = 0,
) -> ContainerStateDict | None:
    """Find the nearest reachable fuel container.

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

    Args:
        world: Current world state with container positions.
        self_state: Player's own state for adjacency.
        terrain: Terrain map for reachability; ``None`` skips the check.
        want_fuel: True to look for fuel, False for equipment.
        now_ms: Current timestamp for freshness filtering.

    Returns:
        An adjacent fresh reachable container, or ``None``.
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
    """Find the nearest fuel container for depositing surplus fuel.

    Args:
        world: Current world state with container positions.
        self_state: Player's own state for position.
        terrain: Optional terrain map for reachability checks.
        allow_unreachable: Whether teleport fallback is allowed.

    Returns:
        Nearest reachable fuel ContainerStateDict for depositing, or None.
    """
    return find_nearest_fuel(
        world,
        self_state,
        terrain,
        allow_unreachable=allow_unreachable,
    )


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
        reason, is_actionable, is_blocked, _missing_landing, low_volume_target = (
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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


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
    # Server handles displacement on teleport landing, so in-bounds
    # containers always have a valid landing tile.
    return ("actionable", True, True, False, False)


def _equipment_candidate_distance(item: tuple[int, ContainerStateDict]) -> int:
    """Sort key for equipment candidates — nearest first."""
    return item[0]


def _known_fuel_candidate_key(item: tuple[int, int, ContainerStateDict]) -> tuple[int, int]:
    """Sort key for known fuel candidates — best score first, then nearest."""
    return (-item[0], item[1])


__all__ = [
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
    "is_reachable",
]
