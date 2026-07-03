"""Shared combat landing helpers for enemy-directed teleports."""

from __future__ import annotations

from collections.abc import Callable

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.equipment import hostile_mines
from tankpit_bot.bot.ai.threats import manhattan_distance
from tankpit_bot.bot.ai.types import EnemyThreatDict
from tankpit_bot.state import SelfStateDict, WorldStateDict


def combat_landing_candidates(
    world: WorldStateDict,
    self_state: SelfStateDict,
    target: EnemyThreatDict,
) -> list[tuple[int, int]]:
    """Return usable adjacent landing tiles ordered by self distance.

    Args:
        world: Current world state.
        self_state: Player state.
        target: Enemy threat to approach.

    Returns:
        Ordered usable landing tiles adjacent to the target.
    """
    candidates = [
        (target["x"] + 1, target["y"]),
        (target["x"] - 1, target["y"]),
        (target["x"], target["y"] + 1),
        (target["x"], target["y"] - 1),
    ]
    usable: list[tuple[int, int]] = []
    for candidate_x, candidate_y in candidates:
        if not (0 <= candidate_x <= 255 and 0 <= candidate_y <= 255):
            continue
        if _is_dynamically_occupied(world, candidate_x, candidate_y):
            continue
        usable.append((candidate_x, candidate_y))
    usable.sort(key=_distance_key(self_state["x"], self_state["y"]))
    return usable


def choose_combat_landing_tile(
    world: WorldStateDict,
    self_state: SelfStateDict,
    target: EnemyThreatDict,
    terrain: TerrainMapProtocol | None,
) -> tuple[int, int]:
    """Choose the tile to teleport to for combat.

    Teleports directly to the enemy's coordinates. The server handles
    displacement — when a tank occupies the tile, it places us on the
    nearest open tile (typically cardinal adjacent). This is how human
    players teleport: click on the enemy, let the server place you.

    Args:
        world: Current world state (unused — server is authoritative).
        self_state: Player state (unused).
        target: Enemy threat currently being engaged.
        terrain: Terrain map (unused — server handles placement).

    Returns:
        Tuple of landing coordinates (the target's position).
    """
    del world, self_state, terrain
    return (target["x"], target["y"])


def has_cardinal_enemy_adjacency(
    self_state: SelfStateDict,
    target: EnemyThreatDict,
) -> bool:
    """Return True when self is cardinally adjacent to the target.

    Args:
        self_state: Player state.
        target: Enemy threat.

    Returns:
        True if Manhattan distance is exactly one.
    """
    return (
        manhattan_distance(
            self_state["x"],
            self_state["y"],
            target["x"],
            target["y"],
        )
        == 1
    )


def _is_dynamically_occupied(world: WorldStateDict, x: int, y: int) -> bool:
    """Return True when a tile is occupied by a tank, container, or mine.

    Args:
        world: Current world state.
        x: Candidate X coordinate.
        y: Candidate Y coordinate.

    Returns:
        True if the tile is blocked by a dynamic entity.
    """
    if any(tank["x"] == x and tank["y"] == y for tank in world["tanks"].values()):
        return True
    if f"{x},{y}" in world["containers"]:
        return True
    return f"{x},{y}" in hostile_mines(world)


def _distance_key(self_x: int, self_y: int) -> Callable[[tuple[int, int]], int]:
    """Return a typed Manhattan-distance sort key.

    Args:
        self_x: Player X coordinate.
        self_y: Player Y coordinate.

    Returns:
        Sort key closure for candidate positions.
    """

    def key(position: tuple[int, int]) -> int:
        return manhattan_distance(self_x, self_y, position[0], position[1])

    return key


__all__ = [
    "choose_combat_landing_tile",
    "combat_landing_candidates",
    "has_cardinal_enemy_adjacency",
]
