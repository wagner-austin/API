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


def find_nearest_fuel(
    world: WorldStateDict,
    self_state: SelfStateDict,
    terrain: TerrainMapProtocol | None = None,
) -> ContainerStateDict | None:
    """Find the nearest reachable fuel container.

    When terrain is provided, skips containers that are unreachable
    due to terrain obstacles (rocks, water). Without terrain, falls
    back to Manhattan distance only.

    Args:
        world: Current world state with container positions.
        self_state: Player's own state for position.
        terrain: Optional terrain map for reachability checks.

    Returns:
        Nearest reachable fuel ContainerStateDict, or None if none visible.
    """
    best: ContainerStateDict | None = None
    best_dist = _MAX_DIST

    for container in world["containers"].values():
        if not container["is_fuel"]:
            continue
        dist = manhattan_distance(
            self_state["x"],
            self_state["y"],
            container["x"],
            container["y"],
        )
        if dist < best_dist:
            if terrain is not None and not is_reachable(
                terrain,
                self_state["x"],
                self_state["y"],
                container["x"],
                container["y"],
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
    """Find the nearest reachable equipment container.

    When terrain is provided, skips containers that are unreachable
    due to terrain obstacles (rocks, water). Without terrain, falls
    back to Manhattan distance only.

    Args:
        world: Current world state with container positions.
        self_state: Player's own state for position.
        terrain: Optional terrain map for reachability checks.

    Returns:
        Nearest reachable equipment ContainerStateDict, or None if none visible.
    """
    best: ContainerStateDict | None = None
    best_dist = _MAX_DIST

    for container in world["containers"].values():
        if container["is_fuel"]:
            continue
        dist = manhattan_distance(
            self_state["x"],
            self_state["y"],
            container["x"],
            container["y"],
        )
        if dist < best_dist:
            if terrain is not None and not is_reachable(
                terrain,
                self_state["x"],
                self_state["y"],
                container["x"],
                container["y"],
            ):
                continue
            best_dist = dist
            best = container

    return best


def find_best_fuel(
    world: WorldStateDict,
    self_state: SelfStateDict,
    terrain: TerrainMapProtocol | None = None,
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

    Returns:
        Best fuel ContainerStateDict, or None if none visible.
    """
    best: ContainerStateDict | None = None
    best_score = -_MAX_DIST

    for container in world["containers"].values():
        if not container["is_fuel"]:
            continue
        cx, cy = container["x"], container["y"]
        dist = manhattan_distance(self_state["x"], self_state["y"], cx, cy)
        passable = terrain.is_passable(cx, cy) if terrain is not None else True
        reachable = is_reachable(
            terrain, self_state["x"], self_state["y"], cx, cy,
        ) if terrain is not None else True
        log.info(
            "FUEL_CHECK: (%d,%d) vol=%d dist=%d passable=%s reachable=%s terrain=%s",
            cx, cy, container["volume"], dist, passable, reachable,
            terrain is not None,
        )
        if terrain is not None and not reachable:
            continue
        # Score: volume is more important than proximity
        score = container["volume"] - dist
        if score > best_score:
            best_score = score
            best = container

    return best


def find_nearest_deposit(
    world: WorldStateDict,
    self_state: SelfStateDict,
    terrain: TerrainMapProtocol | None = None,
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
    return find_nearest_fuel(world, self_state, terrain)


# Sentinel distance larger than any possible Manhattan distance on 256x256 map
_MAX_DIST = 512


__all__ = [
    "find_best_fuel",
    "find_nearest_deposit",
    "find_nearest_equipment",
    "find_nearest_fuel",
    "is_reachable",
]
