"""Container search — find the best fuel, equipment, or deposit target.

All ``find_*`` functions search world state for reachable containers,
optionally filtering by terrain passability. Split from ``equipment.py``
which keeps predicates and scan-coverage checks.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.equipment import (
    _viewport_bounds,
    is_container_pursuable,
)
from tankpit_bot.bot.ai.reachability import is_collection_reachable_in_viewport
from tankpit_bot.bot.ai.threat_primitives import manhattan_distance
from tankpit_bot.state.types import (
    ContainerStateDict,
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
    goal_x: int,
    goal_y: int,
) -> tuple[int, int] | None:
    """Find a legal teleport landing point for a container target.

    Teleports directly to the target when the tile is terrain-legal.
    When it is not (water, rock), checks cardinal neighbors. Returns
    None when the target and all neighbors are illegal (e.g. container
    in the middle of a lake) — the caller should skip this container.

    Asks ``is_landing_legal``, never ``is_passable``: the server
    displaces a landing off mines and off occupied tiles rather than
    refusing it. That makes this the TRANSPORT answer — arriving NEAR
    the aim is acceptable (combat aims, scouting, mine-flip escapes;
    using the walk question would forbid aiming at any enemy at all,
    since an enemy always occupies its own tile). It is NOT the pickup
    answer: a transfer needs the tank ON or cardinally adjacent to the
    container, and a mined landing displaces outside that reach every
    time — pickup selectors must use ``find_attainable_landing_tile``
    (session bot-20260805-173034: 534 displaced teleports re-aimed at
    one known mine, zero pickups, 43 minutes).

    Args:
        terrain: Terrain view answering landing legality.
        goal_x: Target container X coordinate.
        goal_y: Target container Y coordinate.

    Returns:
        Tuple of landing coordinates, or None when no legal tile
        exists at the target or its cardinal neighbors.
    """
    if not (_MAP_MIN <= goal_x <= _MAP_MAX and _MAP_MIN <= goal_y <= _MAP_MAX):
        return None
    if terrain.is_landing_legal(goal_x, goal_y):
        return (goal_x, goal_y)
    for dx, dy in _ADJACENT_DIRECTIONS:
        nx, ny = goal_x + dx, goal_y + dy
        if not (_MAP_MIN <= nx <= _MAP_MAX and _MAP_MIN <= ny <= _MAP_MAX):
            continue
        if terrain.is_landing_legal(nx, ny):
            return (nx, ny)
    return None


def is_reachable(
    terrain: TerrainMapProtocol,
    start_x: int,
    start_y: int,
    goal_x: int,
    goal_y: int,
) -> bool:
    """Check if a target is reachable via terrain-aware pathfinding.

    Args:
        terrain: Terrain map for passability checks.
        start_x: Starting X coordinate.
        start_y: Starting Y coordinate.
        goal_x: Goal X coordinate.
        goal_y: Goal Y coordinate.

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
    )
    return len(path) > 0


def find_nearest_fuel(
    world: WorldStateDict,
    self_state: SelfStateDict,
    terrain: TerrainMapProtocol | None = None,
) -> ContainerStateDict | None:
    """Find the nearest walk-reachable fuel container.

    Args:
        world: Current world state with container positions.
        self_state: Player's own state for position.
        terrain: Optional terrain map for reachability checks.

    Returns:
        Nearest walk-reachable fuel ContainerStateDict, or None if none visible.
    """
    best: ContainerStateDict | None = None
    best_dist = _MAX_DIST

    sx, sy = self_state["x"], self_state["y"]
    for container in world["containers"].values():
        if not _is_visible_candidate(
            container,
            world,
            want_fuel=True,
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
            ):
                continue
            best_dist = dist
            best = container

    return best


def find_nearest_equipment(
    world: WorldStateDict,
    self_state: SelfStateDict,
    terrain: TerrainMapProtocol | None = None,
) -> ContainerStateDict | None:
    """Find the nearest walk-reachable equipment container.

    Args:
        world: Current world state with container positions.
        self_state: Player's own state for position.
        terrain: Optional terrain map for reachability checks.

    Returns:
        Nearest walk-reachable equipment ContainerStateDict, or None if none visible.
    """
    candidates = find_equipment_candidates(
        world,
        self_state,
        terrain,
    )
    if not candidates:
        return None
    return candidates[0]


def find_equipment_candidates(
    world: WorldStateDict,
    self_state: SelfStateDict,
    terrain: TerrainMapProtocol | None = None,
) -> list[ContainerStateDict]:
    """Return visible walk-reachable equipment candidates ordered nearest-first.

    Args:
        world: Current world state with container positions.
        self_state: Player's own state for position.
        terrain: Optional terrain map for reachability checks.

    Returns:
        List of visible walk-reachable equipment containers ordered by Manhattan distance.
    """
    candidates: list[tuple[int, ContainerStateDict]] = []
    sx, sy = self_state["x"], self_state["y"]
    for container in world["containers"].values():
        if not _is_visible_candidate(
            container,
            world,
            want_fuel=False,
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
    minimum_volume: int = 100,
) -> ContainerStateDict | None:
    """Find the best walk-reachable fuel container, prioritizing volume over distance.

    Score = volume - distance, so high-volume nearby containers win.

    Args:
        world: Current world state with container positions.
        self_state: Player's own state for position.
        terrain: Optional terrain map for reachability checks.
        minimum_volume: Minimum fuel volume that counts as actionable.

    Returns:
        Best walk-reachable fuel ContainerStateDict, or None if none visible.
    """
    best: ContainerStateDict | None = None
    best_score = -_MAX_DIST
    sx, sy = self_state["x"], self_state["y"]

    for container in world["containers"].values():
        if not _is_visible_candidate(
            container,
            world,
            want_fuel=True,
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
        ):
            continue
        score = container["volume"] - dist
        if score > best_score:
            best_score = score
            best = container

    return best


def find_fuel_candidates(
    world: WorldStateDict,
    self_state: SelfStateDict,
    terrain: TerrainMapProtocol | None = None,
    *,
    minimum_volume: int = 100,
) -> list[ContainerStateDict]:
    """Return walk-reachable fuel candidates ordered best-first.

    The same eligibility gates as :func:`find_best_fuel`, returning
    the FULL ranked list (``volume - distance`` descending) instead of
    only the winner -- flag s9-2/3 (2026-07-30): the walk step's
    single-best pick was vetoed as not worth its 13-tile walk while a
    762-volume container sat 3 tiles away, and with no second look
    the cascade fell through to an in-viewport larder TELEPORT (map
    open + displaced landing + spent radar). The walk step now
    iterates this list and takes the first walk-worthy candidate.

    Args:
        world: Current world state with container positions.
        self_state: Player's own state for position.
        terrain: Optional terrain map for reachability checks.
        minimum_volume: Minimum fuel volume that counts as actionable.

    Returns:
        Ranked walk-reachable fuel containers, best score first.
    """
    sx, sy = self_state["x"], self_state["y"]
    scored: list[tuple[int, ContainerStateDict]] = []
    for container in world["containers"].values():
        if not _is_visible_candidate(container, world, want_fuel=True):
            continue
        if container["volume"] < minimum_volume:
            continue
        cx, cy = container["x"], container["y"]
        dist = manhattan_distance(sx, sy, cx, cy)
        if not _is_actionable_with_terrain(world, terrain, sx, sy, cx, cy):
            continue
        scored.append((container["volume"] - dist, container))
    scored.sort(key=_fuel_candidate_score, reverse=True)
    return [container for _, container in scored]


def _fuel_candidate_score(entry: tuple[int, ContainerStateDict]) -> int:
    """Sort key: the candidate's volume-minus-distance score.

    Args:
        entry: ``(score, container)`` pair.

    Returns:
        The score component.
    """
    return entry[0]


def find_adjacent_container(
    world: WorldStateDict,
    self_state: SelfStateDict,
    terrain: TerrainMapProtocol | None,
    *,
    want_fuel: bool,
) -> ContainerStateDict | None:
    """Return a walk-reachable container of the requested kind within one tile.

    Args:
        world: Current world state with container positions.
        self_state: Player's own state for adjacency.
        terrain: Terrain map for reachability; ``None`` skips the check.
        want_fuel: True to look for fuel, False for equipment.

    Returns:
        An adjacent walk-reachable container, or ``None``.
    """
    sx, sy = self_state["x"], self_state["y"]
    for container in world["containers"].values():
        if not is_container_pursuable(container, want_fuel=want_fuel):
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
        ):
            continue
        return container
    return None


def describe_container_search(
    world: WorldStateDict,
    self_state: SelfStateDict,
    terrain: TerrainMapProtocol | None = None,
    *,
    want_fuel: bool,
    minimum_volume: int = 0,
) -> str:
    """Summarize why container targeting did or did not find an actionable target.

    Args:
        world: Current world state with container positions.
        self_state: Player's own state for position.
        terrain: Optional terrain map for reachability checks.
        want_fuel: True to inspect fuel containers, False for equipment.
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
                minimum_volume=minimum_volume,
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
        f"blocked={blocked} low_volume={low_volume} "
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
) -> bool:
    """Return True when a container passes type, pursuability, and viewport checks."""
    if not is_container_pursuable(container, want_fuel=want_fuel):
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
) -> bool:
    """Return True when a walk path to the container exists in the viewport.

    User contract (2026-06-26): the bot collects containers by walking
    to them — one ``pickup_*`` command, server routes the tank — just
    like a human clicking a container. Teleport-to-container is gone;
    containers without a viewport walk path are skipped and the
    search-hop relocates the bot to a fresh viewport.
    """
    if terrain is None:
        return True
    return is_collection_reachable_in_viewport(
        world,
        terrain,
        start_x,
        start_y,
        goal_x,
        goal_y,
    )


def _describe_candidate_reason(
    world: WorldStateDict,
    container: ContainerStateDict,
    start_x: int,
    start_y: int,
    terrain: TerrainMapProtocol | None,
    *,
    want_fuel: bool,
    minimum_volume: int,
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
    ):
        return ("actionable", True, False, False, False)
    return ("blocked_walk", False, True, False, False)


def _equipment_candidate_distance(item: tuple[int, ContainerStateDict]) -> int:
    """Sort key for equipment candidates — nearest first."""
    return item[0]


def find_all_tracked_equipment(world: WorldStateDict) -> list[ContainerStateDict]:
    """Return every tracked equipment container across the whole map.

    ``find_nearest_equipment`` filters to the current viewport bounds
    -- what the bot can walk to right now -- but ``world.containers``
    accumulates every equipment container revealed by radar or 0x5A
    patch since the session began. When the current viewport is dry
    the bot still needs a target to hop to, and Bug 0.7 introduces
    the equipment-hop cascade step consuming this atlas. The filter
    is minimal:

    * ``is_fuel = False`` -- fuel dots have their own hop path in
      :func:`~tankpit_bot.bot.ai.resource_search.make_resource_search_hop`.
    * ``failed_pickups == 0`` -- containers the server has already
      refused (empty, geometry, inventory-full) stay refused; the
      hop path does not retry them.

    Stale beliefs are accepted: a container revealed 5 minutes ago
    may have been picked up by another player and no wire signal
    confirms distant consumption. The pragmatic Phase 0 hop pays the
    wasted-teleport cost when the container is gone.

    Args:
        world: Current world state.

    Returns:
        Every equipment container in ``world["containers"]`` that has
        not been blacklisted by prior failed pickups.
    """
    return [
        container
        for container in world["containers"].values()
        if not container["is_fuel"] and container["failed_pickups"] == 0
    ]


__all__ = [
    "describe_container_search",
    "find_adjacent_container",
    "find_all_tracked_equipment",
    "find_best_fuel",
    "find_equipment_candidates",
    "find_fuel_candidates",
    "find_nearest_equipment",
    "find_nearest_fuel",
    "find_teleport_landing_tile",
    "is_reachable",
]
