"""Lifted shot-clearance (line-of-sight) primitives.

Lives in ``state`` rather than ``physics`` because it is a QUERY over
terrain state, not a rule constant: it reads ``TerrainTileDict``
patches and a ``TerrainMapProtocol``. The ``physics`` package holds
the pure rule layer (capacities, costs, damage) that ``state`` reads
in the other direction, so keeping this module there made the two
packages mutually dependent.

ONE line-of-sight answer shared by every shot consumer — combat fire
(prefer a clear-line dual over an over-terrain homing,
[[flag-triage-20260729]] F11) and mine clearance (mine shots NEVER
arc over terrain, [[mine-mechanics]] § shot clearance) — instead of
each path growing its own ad-hoc check. User directive 2026-07-29:
"i want to ensure that we have a proper integrated, and lifted
system that handles shot clearance reliably."

Blocking rules (user law, [[mine-mechanics]], re-confirmed verbatim
2026-07-30 flag s3-14: "we can shoot over other mines of course.
just not mountains or mobable blocks on land"):

- Rock/mountain terrain interrupts the line.
- Movable concrete blocks interrupt the line — bridge (1), land (2),
  and stacked (3) forms, plus ferry-rock (7); the wire vocabulary is
  shared by 0x42/0x4A/0x5A ([[movable-blocks]]).
- Water NEVER blocks a shot ([[weapon-selection]]).
- Mines, containers, and tanks on intermediate tiles never block.

The shot line is the Bresenham raster between shooter and target
with BOTH endpoints excluded — the shooter's own tile and the target
tile do not occlude their own shot. Wire terrain patches are
authoritative for the tiles they enumerate; the static field-image
map answers for everything else.
"""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.state.types import TerrainTileDict
from tankpit_bot.types.constants import (
    ASCII_ROCK,
    TERRAIN_BLOCK_BRIDGE,
    TERRAIN_BLOCK_LAND,
    TERRAIN_BLOCK_STACKED,
    TERRAIN_FERRY_ROCK,
)

_BLOCKING_WIRE_TYPES = frozenset(
    (
        TERRAIN_BLOCK_BRIDGE,
        TERRAIN_BLOCK_LAND,
        TERRAIN_BLOCK_STACKED,
        TERRAIN_FERRY_ROCK,
    )
)


def shot_line_tiles(
    from_x: int,
    from_y: int,
    to_x: int,
    to_y: int,
) -> list[tuple[int, int]]:
    """Return the intermediate tiles of the straight shot line.

    Bresenham raster from shooter to target, excluding both
    endpoints. Adjacent (or same-tile) shots have no intermediate
    tiles and return an empty list.

    Args:
        from_x: Shooter X coordinate.
        from_y: Shooter Y coordinate.
        to_x: Target X coordinate.
        to_y: Target Y coordinate.

    Returns:
        Intermediate ``(x, y)`` tiles in shooter-to-target order.
    """
    tiles: list[tuple[int, int]] = []
    dx = abs(to_x - from_x)
    dy = abs(to_y - from_y)
    step_x = 1 if to_x > from_x else -1
    step_y = 1 if to_y > from_y else -1
    error = dx - dy
    x, y = from_x, from_y
    while True:
        doubled = 2 * error
        if doubled > -dy:
            error -= dy
            x += step_x
        if doubled < dx:
            error += dx
            y += step_y
        if (x, y) == (to_x, to_y):
            return tiles
        tiles.append((x, y))


def is_shot_line_clear(
    from_x: int,
    from_y: int,
    to_x: int,
    to_y: int,
    terrain: TerrainMapProtocol | None,
    wire_terrain: dict[str, TerrainTileDict],
) -> bool:
    """Return True when the straight shot line has no occluder.

    Args:
        from_x: Shooter X coordinate.
        from_y: Shooter Y coordinate.
        to_x: Target X coordinate.
        to_y: Target Y coordinate.
        terrain: Static field-image map; ``None`` means the static
            layer cannot occlude (trust the wire layer alone).
        wire_terrain: The world state's ``terrain`` dict of wire
            patches keyed ``"x,y"`` — authoritative for their tiles
            (a block placed mid-session occludes even though the
            field image shows open ground; a cleared tile un-occludes
            even where the field image shows rock).

    Returns:
        True when no intermediate tile is rock or a movable block.
    """
    for tile_x, tile_y in shot_line_tiles(from_x, from_y, to_x, to_y):
        patch = wire_terrain.get(f"{tile_x},{tile_y}")
        if patch is not None:
            if patch["terrain_type"] in _BLOCKING_WIRE_TYPES:
                return False
            continue
        if terrain is not None and terrain.get_terrain(tile_x, tile_y) == ASCII_ROCK:
            return False
    return True


__all__ = [
    "is_shot_line_clear",
    "shot_line_tiles",
]
