"""Terrain tile TypedDict + factory + encode/decode.

A tile stores only its terrain type (ground/rock/water/ferry/etc).
Container and mine layers live in their own world-state registries
(``world.containers`` and ``world.mines``) populated by the per-tile
mutators in :mod:`tankpit_bot.state.container_mutations`.
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject, require_int
from typing_extensions import TypedDict


class TerrainTileDict(TypedDict):
    """State of a terrain tile.

    Attributes:
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        terrain_type: Terrain/structure type (0=ground, 1-3=rock variants, 5=ferry, 7=ferry+rock).
    """

    x: int
    y: int
    terrain_type: int


def make_terrain_tile(
    x: int,
    y: int,
    terrain_type: int,
) -> TerrainTileDict:
    """Create a terrain tile.

    Args:
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        terrain_type: Terrain type (0-7).

    Returns:
        TerrainTileDict with the provided values.
    """
    return TerrainTileDict(
        x=x,
        y=y,
        terrain_type=terrain_type,
    )


def encode_terrain_tile(tile: TerrainTileDict) -> JSONObject:
    """Encode TerrainTileDict to JSON-serializable dict.

    Args:
        tile: TerrainTileDict to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "x": tile["x"],
        "y": tile["y"],
        "terrain_type": tile["terrain_type"],
    }


def decode_terrain_tile(data: JSONObject) -> TerrainTileDict:
    """Decode TerrainTileDict from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated TerrainTileDict.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return TerrainTileDict(
        x=require_int(data, "x"),
        y=require_int(data, "y"),
        terrain_type=require_int(data, "terrain_type"),
    )


__all__ = [
    "TerrainTileDict",
    "decode_terrain_tile",
    "encode_terrain_tile",
    "make_terrain_tile",
]
