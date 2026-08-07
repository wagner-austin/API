"""Mine and ferry seeding for a sim room.

The standing minefield laid at the archive's measured density, and the
ferries that float on the room's water. Both run AFTER the containers
are placed so they can share tiles, which is what the live game does.
"""

from __future__ import annotations

from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.sim.world import (
    SimFerryDict,
    SimWorldDict,
    place_mine,
)

MINE_DENSITY = 0.14

MINE_SHAPE_CYCLE: tuple[tuple[int, int], ...] = (
    (1, 1),
    (1, 2),
    (1, 1),
    (3, 1),
    (1, 1),
    (2, 3),
    (1, 1),
    (2, 1),
    (1, 1),
    (1, 3),
    (3, 3),
    (1, 1),
    (3, 2),
    (1, 1),
    (2, 2),
    (4, 3),
    (1, 1),
    (1, 2),
    (3, 4),
    (1, 3),
    (2, 3),
)

MINE_TEAM_CYCLE: tuple[int, ...] = (3, 3, 0, 3, 3, 0, 3, 0, 3, 3, 0)

_MAP_SPAN = 256

_TILE_COUNT = _MAP_SPAN * _MAP_SPAN

#: One ferry per this many water tiles.
#:
#: MEASURED 2026-08-06: a session's view holds a median 3 live ferry
#: tiles (p90 16, max 32) across 179 archived sessions, read off the
#: 0x5A patch's terrain nibble (wire value 5) and the 0x4A move pairs.
#: field01 carries 9,925 water tiles in 150 bodies, so one per 300
#: floats ~33 across the room — the density that puts the measured
#: handful inside a session's travel. The 391 distinct ferry tiles the
#: archive shows are drift PATHS over months, not a ferry count.
_FERRY_WATER_SPACING = 300

#: Components land on every third row, so a shape up to four tiles tall
#: cannot fuse with the row band above it into one blob — the merged
#: result would be the wrong SHAPE even at the right count.
_MINE_ROW_STRIDE = 3

#: One component per this many passable tiles within a seeding row.
#: With the row stride above, this lands :data:`MINE_DENSITY`.
_MINE_COMPONENT_SPACING = 9


def seed_minefield(world: SimWorldDict, terrain: TerrainMapProtocol) -> int:
    """Lay the room's standing minefield across the passable map.

    Mines arrive as separate solid COMPONENTS in the measured mix of
    :data:`MINE_SHAPE_CYCLE` — mostly single mines and press-sized
    blobs — because that is what the archive shows, and because the
    gaps between them are what a route needs.

    Mines are seeded INDEPENDENTLY of containers, because the game
    lets them share a tile ([[mine-mechanics]]: "Containers can
    coexist with mines on the same tile") and those shared tiles are
    exactly the ones the bot's clearance and landing-displacement
    machinery exists for. Living tanks are skipped — a placement never
    lands under one.

    Deterministic: component origins come from a fixed raster walk, so
    the same field yields the same minefield on every run.

    Args:
        world: Simulated world (mutated: ``mines`` filled).
        terrain: Static terrain of the world's field.

    Returns:
        How many mines were laid.
    """
    occupied = {(tank["x"], tank["y"]) for tank in world["tanks"].values() if tank["alive"]}
    passable = 0
    components = 0
    for linear in range(_TILE_COUNT):
        x, y = linear % _MAP_SPAN, linear // _MAP_SPAN
        if not terrain.is_passable(x, y):
            continue
        passable += 1
        if y % _MINE_ROW_STRIDE != 0 or passable % _MINE_COMPONENT_SPACING != 0:
            continue
        width, height = MINE_SHAPE_CYCLE[components % len(MINE_SHAPE_CYCLE)]
        team = MINE_TEAM_CYCLE[components % len(MINE_TEAM_CYCLE)]
        components += 1
        for dy in range(height):
            for dx in range(width):
                tile_x, tile_y = x + dx, y + dy
                if not (0 <= tile_x < _MAP_SPAN and 0 <= tile_y < _MAP_SPAN):
                    continue
                if not terrain.is_passable(tile_x, tile_y) or (tile_x, tile_y) in occupied:
                    continue
                place_mine(world, tile_x, tile_y, team)
    return len(world["mines"])


def seed_ferries(world: SimWorldDict, terrain: TerrainMapProtocol) -> int:
    """Float the room's ferries across its water.

    The sim seeded ferries in exactly one scenario, at one hardcoded
    tile, so the 205 archived 0x4A drift frames had no counterpart
    anywhere else and every ferry-doctrine conclusion a practice or
    atlas run produced was drawn on a room with no ferries in it.

    Args:
        world: Simulated world (mutated: ``ferries`` filled).
        terrain: Static terrain of the world's field — ferries float,
            so only WATER tiles are eligible.

    Returns:
        How many ferries were floated.
    """
    occupied = {(ferry["x"], ferry["y"]) for ferry in world["ferries"]}
    water = 0
    for linear in range(_TILE_COUNT):
        x, y = linear % _MAP_SPAN, linear // _MAP_SPAN
        if terrain.get_terrain(x, y) != terrain.WATER:
            continue
        water += 1
        if water % _FERRY_WATER_SPACING != 0 or (x, y) in occupied:
            continue
        world["ferries"].append(SimFerryDict(x=x, y=y))
        occupied.add((x, y))
    return len(world["ferries"])


__all__ = [
    "MINE_DENSITY",
    "MINE_SHAPE_CYCLE",
    "MINE_TEAM_CYCLE",
    "seed_ferries",
    "seed_minefield",
]
