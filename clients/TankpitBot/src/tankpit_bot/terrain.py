"""Terrain lookup from minimap GIF files.

The minimap GIF files (field##_r.gif) contain terrain data encoded by exact
RGB colors.  Each field uses a different color palette, but the structure is
always the same: exactly three terrain types (ground, rock, water) whose
pixel counts follow a consistent pattern.

Terrain classification uses histogram-based auto-detection: the most
frequent color is ground (the walkable majority of every map), the second
most frequent is rock/border, and the third (if present) is water.  This
works for all fields without hardcoding per-field palettes.

Game coordinates map directly to image pixel coordinates:
- X: 0-255 left to right
- Y: 0-255 top to bottom
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from tankpit_bot._pillow import load_pillow_image_module

_IMAGE = load_pillow_image_module()


_RGB_NEAR_THRESHOLD = 15


def _rgb_near(a: tuple[int, int, int], b: tuple[int, int, int]) -> bool:
    """Return True when two RGB colors are within Manhattan distance threshold.

    Args:
        a: First RGB tuple.
        b: Second RGB tuple.

    Returns:
        True when all channels differ by at most ``_RGB_NEAR_THRESHOLD``.
    """
    return abs(a[0] - b[0]) <= _RGB_NEAR_THRESHOLD and (
        abs(a[1] - b[1]) <= _RGB_NEAR_THRESHOLD and abs(a[2] - b[2]) <= _RGB_NEAR_THRESHOLD
    )


def _classify_colors_by_border(
    pixels: list[tuple[int, int, int]],
) -> dict[tuple[int, int, int], str]:
    """Build a color-to-terrain lookup using border detection.

    Every field GIF has an impassable rock border on all four edges
    (row 0, row 255, col 0, col 255).  The border color identifies
    rock.  The most common interior color identifies ground.  GIF
    compression produces near-identical RGB variants of the dominant
    colors; these are grouped with their nearest canonical color.
    Any remaining color that is not near ground or rock is water.

    Args:
        pixels: All 65536 RGB pixel tuples from a 256x256 image.

    Returns:
        Dict mapping each observed RGB color to its terrain character.
    """
    border_colors: set[tuple[int, int, int]] = set()
    for x in range(256):
        border_colors.add(pixels[x])
        border_colors.add(pixels[255 * 256 + x])
    for y in range(256):
        border_colors.add(pixels[y * 256])
        border_colors.add(pixels[y * 256 + 255])

    counts: Counter[tuple[int, int, int]] = Counter(pixels)
    ranked = counts.most_common()

    ground_color = ranked[0][0]
    rock_color = next(
        (color for color, _count in ranked if color in border_colors),
        ranked[1][0] if len(ranked) > 1 else ground_color,
    )

    lookup: dict[tuple[int, int, int], str] = {}
    for color, _count in ranked:
        if color == ground_color or _rgb_near(color, ground_color):
            lookup[color] = "."
        elif color in border_colors or _rgb_near(color, rock_color):
            lookup[color] = "#"
        else:
            lookup[color] = "W"
    return lookup


class TerrainMap:
    """Terrain lookup from minimap GIF."""

    ROCK = "#"
    GROUND = "."
    WATER = "W"

    def __init__(self, gif_path: str | Path) -> None:
        """Load terrain from GIF file.

        Args:
            gif_path: Path to field##_r.gif minimap file.

        Raises:
            ValueError: If image is not 256x256.
        """
        # ``Image.open`` is lazy and holds the file handle open until the
        # image is closed; ``convert`` loads the data into an independent
        # image, so the source can be released immediately. Without this
        # the loader leaked one descriptor per call (ResourceWarning).
        with _IMAGE.open(gif_path) as source:
            img = source.convert("RGB")
        if img.size != (256, 256):
            raise ValueError(f"Expected 256x256 image, got {img.size}")
        raw = img.tobytes()
        self._pixels: list[tuple[int, int, int]] = [
            (raw[i], raw[i + 1], raw[i + 2]) for i in range(0, len(raw), 3)
        ]
        self._color_lookup: dict[tuple[int, int, int], str] = _classify_colors_by_border(
            self._pixels
        )

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
        return self._color_lookup.get(pixel, self.ROCK)

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
