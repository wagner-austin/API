"""Shared combat landing helpers for enemy-directed teleports."""

from __future__ import annotations

from collections.abc import Callable

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.equipment import hostile_mines
from tankpit_bot.bot.ai.threats import manhattan_distance
from tankpit_bot.bot.ai.types import EnemyThreatDict
from tankpit_bot.state import SelfStateDict, WorldStateDict

# The server's effective shot range. User ruling 2026-07-29: "i dont
# think i would teleport to the target if im like a few tiles away
# right? as long as theyre on the viewport and its a clear dual shot
# then id just hit them from my new location" -- in-view stationary
# targets take duals at range per [[weapon-selection]] (water never
# blocks; rock clips to a billed single that resolves as a miss). 8
# keeps the Manhattan bound inside the 18x18 viewport after a centered
# landing. The 2026-06-11 counter-measurement (distance 4+ hit ~0%: 45
# misses at 4, 35 at 12) predates id-targeted shots and the freshness
# model -- the shot billing analyzer re-prices this live; if ranged
# duals miss, the ledger will say so in one session and this constant
# reverts. Lives here (not combat_strategy) because landing choice and
# acquisition viability both key off it and combat_strategy already
# imports this module.
SHOT_RANGE_TILES = 8


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

    When the enemy stands on passable ground, teleports directly to
    their coordinates: the server handles displacement — a tank on the
    tile gets us placed on the nearest open tile (typically cardinal
    adjacent). This is how human players teleport: click on the enemy,
    let the server place you.

    When the enemy's own tile is impassable — a ferry rider on open
    water (live 2026-07-29: Yuppler rode a ferry at (128,102) and
    every acquisition pass rejected him) — the server will not land us
    there, so aim instead at the passable, unoccupied tile inside the
    ``SHOT_RANGE_TILES`` diamond nearest the target (tie-broken toward
    self for the cheaper teleport). Water never blocks shots
    ([[weapon-selection]]), so a shore tile within range is a firing
    position.

    Args:
        world: Current world state (dynamic occupancy for stand-off tiles).
        self_state: Player state (stand-off tie-break).
        target: Enemy threat currently being engaged.
        terrain: Terrain map; ``None`` trusts the server entirely.

    Returns:
        Tuple of landing coordinates.
    """
    target_x, target_y = target["x"], target["y"]
    if terrain is None or terrain.is_passable(target_x, target_y):
        return (target_x, target_y)
    best: tuple[int, int] | None = None
    best_key: tuple[int, int] | None = None
    for dx in range(-SHOT_RANGE_TILES, SHOT_RANGE_TILES + 1):
        remaining = SHOT_RANGE_TILES - abs(dx)
        for dy in range(-remaining, remaining + 1):
            tile_x, tile_y = target_x + dx, target_y + dy
            if not (0 <= tile_x <= 255 and 0 <= tile_y <= 255):
                continue
            if not terrain.is_passable(tile_x, tile_y):
                continue
            if _is_dynamically_occupied(world, tile_x, tile_y):
                continue
            key = (
                abs(dx) + abs(dy),
                manhattan_distance(self_state["x"], self_state["y"], tile_x, tile_y),
            )
            if best_key is None or key < best_key:
                best, best_key = (tile_x, tile_y), key
    if best is not None:
        return best
    return (target_x, target_y)


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
    "SHOT_RANGE_TILES",
    "choose_combat_landing_tile",
    "combat_landing_candidates",
    "has_cardinal_enemy_adjacency",
]
