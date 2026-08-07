"""Deterministic open-tile placement for sim world seeding.

This module once implemented a "container respawn law" (~1 dot/min,
population-seeking) mined 2026-07-22 from 0x4C atlas diffs. That law
was FALSIFIED 2026-07-25: all 605 "spawns" were our own exposures of
pre-existing >=500-volume containers ([[game-economy]],
[[map-data-decode]] — `analysis_scripts/mine_dot_appearances.py`
found 605/605 appearances exposure-preceded). No true container
spawn has ever been witnessed, so the sim spawns nothing at runtime;
the world is a static population seeded by ``sim.world_seed``. The
deterministic tile pickers below serve seeding and bot reactivation.
"""

from __future__ import annotations

from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.sim.world import SimWorldDict
from tankpit_bot.state.scan_coverage import tile_key

_MAP_SPAN = 256
_TILE_COUNT = _MAP_SPAN * _MAP_SPAN
_SEED_STRIDE = 97
_PROBE_STRIDE = 251


def _tile_occupied(world: SimWorldDict, x: int, y: int) -> bool:
    """Report whether a tile already holds an entity a placement must avoid.

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
    if any((block["x"], block["y"]) == (x, y) for block in world["blocks"]):
        return True
    return tile_key(x, y) in world["mines"]


def find_open_tile(
    world: SimWorldDict, terrain: TerrainMapProtocol, tick: int
) -> tuple[int, int] | None:
    """Pick a deterministic fresh location scattered across the map.

    A tick-derived start index walks the map in a full-cycle stride
    (251 is coprime with 65 536) until it finds a passable,
    unoccupied tile.

    Args:
        world: Simulated world.
        terrain: Static terrain of the world's field.
        tick: Determinism seed (world tick or a seeding index).

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


__all__ = [
    "find_open_tile",
    "find_open_tile_near",
]
