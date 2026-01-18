"""Terrain lookup from minimap GIF files.

The minimap GIF files (field##_r.gif) contain terrain data encoded by exact RGB colors.

Field42 (Meltdown) - desert theme:
- ROCK: (214, 140, 57) - mountains, border, impassable
- WATER: (213, 164, 65) - water tiles
- GROUND: (239, 189, 90) or (239, 188, 90) - walkable terrain

Field01 (Practice) - forest theme:
- ROCK: (102, 51, 0) - brown mountains/border
- WATER: (51, 153, 255) - blue water
- GROUND: (51, 102, 51) - green walkable terrain

Game coordinates map directly to image pixel coordinates:
- X: 0-255 left to right
- Y: 0-255 top to bottom
"""

from pathlib import Path
from typing import ClassVar

from PIL import Image


class TerrainMap:
    """Terrain lookup from minimap GIF."""

    ROCK = "#"
    GROUND = "."
    WATER = "W"

    # Field42 (Meltdown) colors - desert theme
    COLOR_ROCK_42 = (214, 140, 57)
    COLOR_WATER_42 = (213, 164, 65)
    COLOR_GROUND_42A = (239, 189, 90)
    COLOR_GROUND_42B = (239, 188, 90)

    # Field01 (Practice) colors - forest theme
    COLOR_ROCK_01 = (102, 51, 0)
    COLOR_WATER_01 = (51, 153, 255)
    COLOR_GROUND_01 = (51, 102, 51)

    # Combined color sets for matching
    ROCK_COLORS: ClassVar[set[tuple[int, int, int]]] = {COLOR_ROCK_42, COLOR_ROCK_01}
    WATER_COLORS: ClassVar[set[tuple[int, int, int]]] = {COLOR_WATER_42, COLOR_WATER_01}
    GROUND_COLORS: ClassVar[set[tuple[int, int, int]]] = {
        COLOR_GROUND_42A,
        COLOR_GROUND_42B,
        COLOR_GROUND_01,
    }

    def __init__(self, gif_path: str | Path) -> None:
        """Load terrain from GIF file.

        Args:
            gif_path: Path to field##_r.gif minimap file.
        """
        img = Image.open(gif_path).convert("RGB")
        if img.size != (256, 256):
            raise ValueError(f"Expected 256x256 image, got {img.size}")
        # Pre-load all pixel data for type-safe access
        # Use tobytes() and manual parsing for strict typing and deprecation-safety
        raw = img.tobytes()
        self._pixels: list[tuple[int, int, int]] = [
            (raw[i], raw[i + 1], raw[i + 2]) for i in range(0, len(raw), 3)
        ]

    def get_terrain(self, x: int, y: int) -> str:
        """Get terrain type at game coordinates.

        Args:
            x: X coordinate (0-255).
            y: Y coordinate (0-255).

        Returns:
            Terrain character: '#' for rock, '.' for ground, 'W' for water.
        """
        if not (0 <= x < 256 and 0 <= y < 256):
            return " "

        pixel = self._pixels[y * 256 + x]

        # Exact color matching for terrain types (supports both field01 and field42)
        if pixel in self.WATER_COLORS:
            return self.WATER

        if pixel in self.GROUND_COLORS:
            return self.GROUND

        if pixel in self.ROCK_COLORS:
            return self.ROCK

        # Unknown color - treat as impassable for safety
        return self.ROCK

    def is_passable(self, x: int, y: int) -> bool:
        """Check if tile is passable (not rock or water).

        Args:
            x: X coordinate (0-255).
            y: Y coordinate (0-255).

        Returns:
            True if passable, False if rock or water.
        """
        return self.get_terrain(x, y) == self.GROUND

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
            width: Viewport width (default 16).
            height: Viewport height (default 16).

        Returns:
            2D list of terrain characters.
        """
        left = center_x - width // 2
        top = center_y - height // 2

        grid = []
        for vy in range(height):
            row = []
            for vx in range(width):
                x = left + vx
                y = top + vy
                row.append(self.get_terrain(x, y))
            grid.append(row)

        return grid

    def render_full_map(self) -> list[list[str]]:
        """Render the entire 256x256 map.

        Returns:
            2D list of terrain characters.
        """
        return self.render_viewport(128, 128, 256, 256)


def format_viewport(
    grid: list[list[str]],
    player_x: int | None = None,
    player_y: int | None = None,
    viewport_left: int = 0,
    viewport_top: int = 0,
    entities: dict[tuple[int, int], str] | None = None,
) -> str:
    """Format a viewport grid with optional entities overlay.

    Args:
        grid: 2D terrain grid from render_viewport.
        player_x: Player X coordinate (absolute).
        player_y: Player Y coordinate (absolute).
        viewport_left: Left edge X coordinate.
        viewport_top: Top edge Y coordinate.
        entities: Dict mapping (x, y) to entity character.

    Returns:
        Formatted string representation of the viewport.
    """
    if not grid:
        return ""

    height = len(grid)
    width = len(grid[0])
    lines: list[str] = []

    # Header
    header = "    "
    for vx in range(width):
        header += f"{(viewport_left + vx) % 10} "
    header += f"  (X: {viewport_left}-{viewport_left + width - 1})"
    lines.append(header)

    # Rows
    for vy in range(height):
        y = viewport_top + vy
        row = f"{y:3d} "

        for vx in range(width):
            x = viewport_left + vx

            # Player position
            if player_x is not None and x == player_x and y == player_y:
                row += "@ "
            # Entity overlay
            elif entities and (x, y) in entities:
                row += f"{entities[(x, y)]} "
            # Terrain
            else:
                row += f"{grid[vy][vx]} "

        lines.append(row)

    return "\n".join(lines)
