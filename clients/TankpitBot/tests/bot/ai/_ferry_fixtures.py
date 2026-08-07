"""Shared world builders for the ferry tests."""

from __future__ import annotations

from tankpit_bot.state.types import (
    TerrainTileDict,
    make_terrain_tile,
)
from tankpit_bot.types.constants import (
    TERRAIN_FERRY,
)

_NOW_MS = 1_000_000


def _ferry_tile(x: int, y: int) -> dict[str, TerrainTileDict]:
    """Build the wire terrain entry for a ferry at a coordinate.

    Args:
        x: Tile X coordinate.
        y: Tile Y coordinate.

    Returns:
        Single-entry wire-terrain mapping fragment.
    """
    return {
        f"{x},{y}": make_terrain_tile(
            x=x,
            y=y,
            terrain_type=TERRAIN_FERRY,
        )
    }
