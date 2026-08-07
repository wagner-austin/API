"""Shared combat landing helpers for enemy-directed teleports."""

from __future__ import annotations

from collections.abc import Callable

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.equipment import hostile_mines
from tankpit_bot.bot.ai.threat_primitives import manhattan_distance
from tankpit_bot.bot.ai.world_types import EnemyThreatDict
from tankpit_bot.sniffer.world_state import is_move_target_failed
from tankpit_bot.state import SelfStateDict, WorldStateDict
from tankpit_bot.state.occupancy import is_tank_body_present

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
    terrain: TerrainMapProtocol | None,
    now_ms: int,
) -> list[tuple[int, int]]:
    """Return usable adjacent landing tiles ordered by self distance.

    Usable means inside the map, not dynamically occupied, passable on
    the composed terrain, and not carrying a live failed-move mark.
    The last two filters are the F20 fix (run bot-20260730-110x ticks
    904-949: the walk-close dispatched a move to (240,46) forty-plus
    consecutive ticks — the server rejected every one and marked the
    tile failed every tick, but neither terrain nor the mark was ever
    consulted, so the identical move re-derived forever).

    Args:
        world: Current world state.
        self_state: Player state.
        target: Enemy threat to approach.
        terrain: Composed decision terrain; ``None`` trusts the server.
        now_ms: Current timestamp for the failed-move TTL check.

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
        if _is_dynamically_occupied(world, candidate_x, candidate_y, now_ms):
            continue
        if terrain is not None and not terrain.is_passable(candidate_x, candidate_y):
            continue
        if is_move_target_failed(candidate_x, candidate_y, now_ms):
            continue
        usable.append((candidate_x, candidate_y))
    usable.sort(key=_distance_key(self_state["x"], self_state["y"]))
    return usable


def choose_combat_landing_tile(
    world: WorldStateDict,
    self_state: SelfStateDict,
    target: EnemyThreatDict,
    terrain: TerrainMapProtocol | None,
    now_ms: int,
) -> tuple[int, int]:
    """Choose the tile to teleport to for combat.

    When the enemy stands on terrain-legal ground, teleports directly
    to their coordinates: the server handles displacement — a tank on
    the tile gets us placed on the nearest open tile (typically
    cardinal adjacent). This is how human players teleport: click on
    the enemy, let the server place you. The question asked is
    therefore ``is_landing_legal``, never ``is_passable`` — an enemy
    always occupies its own tile, so the walk question would reject
    every direct approach and silently downgrade it to a stand-off.

    When the enemy's own tile is terrain-illegal — a ferry rider on open
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
        now_ms: Current tick timestamp for body freshness in the
            stand-off occupancy check.

    Returns:
        Tuple of landing coordinates.
    """
    target_x, target_y = target["x"], target["y"]
    if terrain is None or terrain.is_landing_legal(target_x, target_y):
        return (target_x, target_y)
    best: tuple[int, int] | None = None
    best_key: tuple[int, int] | None = None
    for dx in range(-SHOT_RANGE_TILES, SHOT_RANGE_TILES + 1):
        remaining = SHOT_RANGE_TILES - abs(dx)
        for dy in range(-remaining, remaining + 1):
            tile_x, tile_y = target_x + dx, target_y + dy
            if not (0 <= tile_x <= 255 and 0 <= tile_y <= 255):
                continue
            if not terrain.is_landing_legal(tile_x, tile_y):
                continue
            if _is_dynamically_occupied(world, tile_x, tile_y, now_ms):
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


GREETING_STANDOFF_TILES = 6
"""Manhattan distance band center for a greeting approach landing.

User ruling 2026-07-30: before any combat with a human the bot
teleports to them so both sides can SEE each other, says HELLO, and
waits for consent -- "not an adjacent teleport. a few tiles off." Six
tiles is unmistakably visible inside the 16x16 viewport and equally
unmistakably non-hostile (outside auto-engage adjacency); the chooser
accepts one tile of slack either side before giving up.
"""

_GREETING_BAND_SLACK = 1


def choose_greeting_landing_tile(
    world: WorldStateDict,
    self_state: SelfStateDict,
    target: EnemyThreatDict,
    terrain: TerrainMapProtocol | None,
    now_ms: int,
) -> tuple[int, int] | None:
    """Choose a visible, non-adjacent landing near a human to greet.

    Scans the Manhattan ring band ``GREETING_STANDOFF_TILES ±
    _GREETING_BAND_SLACK`` around the target for a passable,
    dynamically unoccupied tile, preferring the band center and then
    the tile nearest self (cheapest teleport). Unlike the combat
    landing chooser this never falls back to the target's own tile --
    landing on or beside an unconsented human reads as an attack.

    Args:
        world: Current world state (dynamic occupancy).
        self_state: Player state (tie-break toward self).
        target: The human to greet.
        terrain: Terrain map; ``None`` means passability is unknown
            and no greeting landing can be vouched for.
        now_ms: Current tick timestamp for body freshness in the band
            occupancy check.

    Returns:
        Landing coordinates a few tiles off the human, or ``None``
        when terrain is unknown or no band tile qualifies.
    """
    if terrain is None:
        return None
    target_x, target_y = target["x"], target["y"]
    best: tuple[int, int] | None = None
    best_key: tuple[int, int, int] | None = None
    max_d = GREETING_STANDOFF_TILES + _GREETING_BAND_SLACK
    for dx in range(-max_d, max_d + 1):
        for dy in range(-(max_d - abs(dx)), max_d - abs(dx) + 1):
            ring = abs(dx) + abs(dy)
            if ring < GREETING_STANDOFF_TILES - _GREETING_BAND_SLACK:
                continue
            tile_x, tile_y = target_x + dx, target_y + dy
            if not (0 <= tile_x <= 255 and 0 <= tile_y <= 255):
                continue
            if not terrain.is_landing_legal(tile_x, tile_y):
                continue
            if _is_dynamically_occupied(world, tile_x, tile_y, now_ms):
                continue
            key = (
                abs(ring - GREETING_STANDOFF_TILES),
                manhattan_distance(self_state["x"], self_state["y"], tile_x, tile_y),
                ring,
            )
            if best_key is None or key < best_key:
                best, best_key = (tile_x, tile_y), key
    return best


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


def _is_dynamically_occupied(world: WorldStateDict, x: int, y: int, now_ms: int) -> bool:
    """Return True when a tile is occupied by a tank body, container, or mine.

    The tank half is the occupancy law
    (:func:`~tankpit_bot.state.occupancy.is_tank_body_present`: not
    self, position ever observed, viewport-fresh). Before the lift this
    counted every registry entry -- including the login-roster (0, 0)
    phantoms, tanks long gone from the viewport, and the bot's own
    body. Containers and hostile mines remain this chooser's own
    displacement-avoidance policy, not part of the body law.

    Args:
        world: Current world state.
        x: Candidate X coordinate.
        y: Candidate Y coordinate.
        now_ms: Current tick timestamp for body freshness.

    Returns:
        True if the tile is blocked by a dynamic entity.
    """
    if any(
        tank["x"] == x and tank["y"] == y and is_tank_body_present(tank, now_ms)
        for tank in world["tanks"].values()
    ):
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
    "GREETING_STANDOFF_TILES",
    "SHOT_RANGE_TILES",
    "choose_combat_landing_tile",
    "choose_greeting_landing_tile",
    "combat_landing_candidates",
    "has_cardinal_enemy_adjacency",
]
