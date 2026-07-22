"""Container respawn — the archive-mined world-replenishment law.

Mined from 212 sessions with 2+ map snapshots (2026-07-22, 0x4C
global fuel-dot atlas diffs): the world holds a steady-state dot
population (569-656 observed; per-session spawns ~ consumption), new
dots appear at ~1/minute while the population is below equilibrium
(605 spawns over 605.7 observed minutes; a 12-minute idle session at
high population spawned ZERO), and a spawn NEVER reuses a consumed
position (0/605 exact reuses; 587/605 were entirely fresh map
locations). No wire message announces a spawn — the client discovers
new dots on the next map open or radar reveal.

Sim model (deterministic, documented assumptions): the target
population is the world's SEEDED container count; one fuel container
respawns per minute (every 30 of the 2 s ticks) while below target,
at a tick-derived fresh passable location. Equipment containers are
invisible to the 0x4C atlas, so their dynamics are unmeasured — the
sim mirrors the fuel law on the offset beat as an assumption.
Spawn volume is a constant 300 (map dots carry no volume; assumption).
"""

from __future__ import annotations

from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.sim.world import SimContainerDict, SimEquipmentDict, SimWorldDict

RESPAWN_INTERVAL_TICKS = 30
"""One spawn per minute at the 2 s tick — the measured ~1.00 dots/min."""

SPAWN_VOLUME = 300
"""Fuel volume of a respawned container (assumption — dots carry none)."""

_MAP_SPAN = 256
_TILE_COUNT = _MAP_SPAN * _MAP_SPAN
_SEED_STRIDE = 97
_PROBE_STRIDE = 251


def _tile_occupied(world: SimWorldDict, x: int, y: int) -> bool:
    """Report whether a tile already holds an entity a spawn must avoid.

    Args:
        world: Simulated world.
        x: Tile X.
        y: Tile Y.

    Returns:
        True when a living tank, a stocked container, equipment, or a
        mine sits on the tile.
    """
    for tank in world["tanks"].values():
        if tank["alive"] and (tank["x"], tank["y"]) == (x, y):
            return True
    for container in world["containers"]:
        if container["volume"] > 0 and (container["x"], container["y"]) == (x, y):
            return True
    for equipment in world["equipment"]:
        if (equipment["x"], equipment["y"]) == (x, y):
            return True
    return any((mine["x"], mine["y"]) == (x, y) for mine in world["mines"])


def find_open_tile(
    world: SimWorldDict, terrain: TerrainMapProtocol, tick: int
) -> tuple[int, int] | None:
    """Pick the deterministic fresh location for one spawn.

    A tick-derived start index walks the map in a full-cycle stride
    (251 is coprime with 65 536) until it finds a passable,
    unoccupied tile — spawns scatter across the whole map, matching
    the measured fresh-position law.

    Args:
        world: Simulated world.
        terrain: Static terrain of the world's field.
        tick: Current world tick (the determinism seed).

    Returns:
        The chosen tile, or None on a map with no open tile.
    """
    seed = (tick * _SEED_STRIDE) % _TILE_COUNT
    for step in range(_TILE_COUNT):
        index = (seed + step * _PROBE_STRIDE) % _TILE_COUNT
        x, y = index % _MAP_SPAN, index // _MAP_SPAN
        if not terrain.is_passable(x, y):
            continue
        if _tile_occupied(world, x, y):
            continue
        return x, y
    return None


def find_open_tile_near(
    world: SimWorldDict,
    terrain: TerrainMapProtocol,
    center_x: int,
    center_y: int,
    tick: int,
    min_radius: int,
    max_radius: int,
) -> tuple[int, int] | None:
    """Pick a deterministic open tile in a Chebyshev ring band.

    Walks rings outward from ``min_radius``, rotating each ring's
    start by the tick, and returns the first passable unoccupied tile
    — used for placements that must stay within reach (the harness
    opponent's revival: a corner-of-the-map respawn fails the HUNT
    owner's affordability gates and ends every session at first kill).

    Args:
        world: Simulated world.
        terrain: Static terrain of the world's field.
        center_x: Ring-band center X.
        center_y: Ring-band center Y.
        tick: Current world tick (the determinism seed).
        min_radius: Innermost ring (keeps placements off the center).
        max_radius: Outermost ring.

    Returns:
        The chosen tile, or None when the whole band is closed.
    """
    for radius in range(min_radius, max_radius + 1):
        ring = [
            (dx, dy)
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
            if max(abs(dx), abs(dy)) == radius
        ]
        start = tick % len(ring)
        for step in range(len(ring)):
            dx, dy = ring[(start + step) % len(ring)]
            x, y = center_x + dx, center_y + dy
            if not (0 <= x < _MAP_SPAN and 0 <= y < _MAP_SPAN):
                continue
            if not terrain.is_passable(x, y):
                continue
            if _tile_occupied(world, x, y):
                continue
            return x, y
    return None


def respawn_containers(
    world: SimWorldDict,
    terrain: TerrainMapProtocol,
    fuel_target: int,
    equipment_target: int,
) -> None:
    """Apply the replenishment law for one tick (wire-silent).

    Fuel respawns on the whole-minute beat, equipment on the offset
    half-minute beat; each spawns at most ONE per beat and only while
    its population is below target.

    Args:
        world: Simulated world (mutated on spawn).
        terrain: Static terrain of the world's field.
        fuel_target: Steady-state fuel-container population.
        equipment_target: Steady-state equipment population.
    """
    tick = world["tick"]
    beat = tick % RESPAWN_INTERVAL_TICKS
    if beat == 0:
        stocked = sum(1 for c in world["containers"] if c["volume"] > 0)
        if stocked < fuel_target:
            position = find_open_tile(world, terrain, tick)
            if position is not None:
                world["containers"].append(
                    SimContainerDict(x=position[0], y=position[1], volume=SPAWN_VOLUME)
                )
        return
    if beat == RESPAWN_INTERVAL_TICKS // 2 and len(world["equipment"]) < equipment_target:
        position = find_open_tile(world, terrain, tick)
        if position is not None:
            world["equipment"].append(SimEquipmentDict(x=position[0], y=position[1]))


__all__ = [
    "RESPAWN_INTERVAL_TICKS",
    "SPAWN_VOLUME",
    "find_open_tile",
    "find_open_tile_near",
    "respawn_containers",
]
