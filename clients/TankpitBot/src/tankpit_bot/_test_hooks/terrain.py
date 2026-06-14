"""Terrain map loading hook and TerrainMap interface protocol.

The bot's pathfinding and viewport rendering consume a
``TerrainMapProtocol``-typed object; production loads it from a
``field##_r.gif`` minimap via :mod:`tankpit_bot.terrain`. Tests inject
hand-rolled terrain that satisfies the protocol so movement and
pickup logic stay deterministic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class TerrainMapProtocol(Protocol):
    """Protocol for TerrainMap interface."""

    ROCK: str
    GROUND: str
    WATER: str

    def get_terrain(self, x: int, y: int) -> str:
        """Get terrain type at game coordinates.

        Args:
            x: X coordinate (0-255).
            y: Y coordinate (0-255).

        Returns:
            Terrain character: '#' for rock, '.' for ground, 'W' for water.
        """
        ...

    def is_passable(self, x: int, y: int) -> bool:
        """Check if tile is passable (not rock or water).

        Args:
            x: X coordinate (0-255).
            y: Y coordinate (0-255).

        Returns:
            True if passable, False if rock or water.
        """
        ...

    def render_viewport(
        self,
        center_x: int,
        center_y: int,
        width: int = 16,
        height: int = 16,
    ) -> list[list[str]]:
        """Render a viewport grid centered on position.

        Args:
            center_x: Center X coordinate.
            center_y: Center Y coordinate.
            width: Viewport width.
            height: Viewport height.

        Returns:
            2D list of terrain characters.
        """
        ...


class LoadTerrainMapProtocol(Protocol):
    """Protocol for loading a terrain map."""

    def __call__(self, gif_path: Path) -> TerrainMapProtocol:
        """Load terrain map from GIF file.

        Args:
            gif_path: Path to field##_r.gif minimap file.

        Returns:
            TerrainMap instance.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If image is not 256x256.
        """
        ...


def _real_load_terrain_map(gif_path: Path) -> TerrainMapProtocol:
    """Real implementation - loads TerrainMap from GIF.

    Args:
        gif_path: Path to field##_r.gif minimap file.

    Returns:
        TerrainMap instance.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If image is not 256x256.
    """
    terrain_mod = __import__("tankpit_bot.terrain", fromlist=["TerrainMap"])
    terrain_map: TerrainMapProtocol = terrain_mod.TerrainMap(gif_path)
    return terrain_map


load_terrain_map: LoadTerrainMapProtocol = _real_load_terrain_map


__all__ = [
    "LoadTerrainMapProtocol",
    "TerrainMapProtocol",
    "load_terrain_map",
]
