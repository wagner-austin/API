"""Tests for tankpit_bot.terrain module."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from tankpit_bot.terrain import TerrainMap, format_viewport


def create_test_gif(path: Path, pixels: list[tuple[int, int, int]]) -> None:
    """Create a 256x256 test GIF with specified pixels.

    Args:
        path: Path to save the GIF.
        pixels: List of 65536 RGB tuples (row-major order).
    """
    img = Image.new("RGB", (256, 256))
    img.putdata(pixels)
    img.save(path)


def make_uniform_pixels(color: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    """Create 256x256 uniform pixels.

    Args:
        color: RGB tuple for all pixels.

    Returns:
        List of 65536 identical color tuples.
    """
    return [color] * (256 * 256)


# =============================================================================
# TerrainMap.__init__ Tests
# =============================================================================


def test_terrain_map_loads_valid_gif(tmp_path: Path) -> None:
    """Test TerrainMap loads a valid 256x256 GIF."""
    gif_path = tmp_path / "test.gif"
    create_test_gif(gif_path, make_uniform_pixels(TerrainMap.COLOR_GROUND_42A))

    terrain = TerrainMap(gif_path)
    assert terrain.get_terrain(0, 0) == TerrainMap.GROUND


def test_terrain_map_accepts_string_path(tmp_path: Path) -> None:
    """Test TerrainMap accepts string path."""
    gif_path = tmp_path / "test.gif"
    create_test_gif(gif_path, make_uniform_pixels(TerrainMap.COLOR_GROUND_42A))

    terrain = TerrainMap(str(gif_path))
    assert terrain.get_terrain(0, 0) == TerrainMap.GROUND


def test_terrain_map_raises_on_wrong_size(tmp_path: Path) -> None:
    """Test TerrainMap raises ValueError for non-256x256 images."""
    gif_path = tmp_path / "small.gif"
    img = Image.new("RGB", (128, 128), (60, 129, 85))
    img.save(gif_path)

    with pytest.raises(ValueError, match="Expected 256x256 image, got \\(128, 128\\)"):
        TerrainMap(gif_path)


# =============================================================================
# TerrainMap.get_terrain Tests
# =============================================================================


def test_get_terrain_rock_detection(tmp_path: Path) -> None:
    """Test get_terrain returns ROCK for exact rock color."""
    gif_path = tmp_path / "rocks.gif"
    create_test_gif(gif_path, make_uniform_pixels(TerrainMap.COLOR_ROCK_42))

    terrain = TerrainMap(gif_path)
    assert terrain.get_terrain(128, 128) == TerrainMap.ROCK


def test_get_terrain_ground_detection(tmp_path: Path) -> None:
    """Test get_terrain returns GROUND for exact ground colors."""
    gif_path = tmp_path / "ground.gif"
    create_test_gif(gif_path, make_uniform_pixels(TerrainMap.COLOR_GROUND_42A))

    terrain = TerrainMap(gif_path)
    assert terrain.get_terrain(128, 128) == TerrainMap.GROUND


def test_get_terrain_ground_variant_b(tmp_path: Path) -> None:
    """Test get_terrain returns GROUND for alternate ground color."""
    gif_path = tmp_path / "ground_b.gif"
    create_test_gif(gif_path, make_uniform_pixels(TerrainMap.COLOR_GROUND_42B))

    terrain = TerrainMap(gif_path)
    assert terrain.get_terrain(128, 128) == TerrainMap.GROUND


def test_get_terrain_water_detection(tmp_path: Path) -> None:
    """Test get_terrain returns WATER for exact water color."""
    gif_path = tmp_path / "water.gif"
    create_test_gif(gif_path, make_uniform_pixels(TerrainMap.COLOR_WATER_42))

    terrain = TerrainMap(gif_path)
    assert terrain.get_terrain(128, 128) == TerrainMap.WATER


def test_get_terrain_unknown_color_returns_rock(tmp_path: Path) -> None:
    """Test get_terrain returns ROCK for unknown colors (safety fallback)."""
    gif_path = tmp_path / "unknown.gif"
    # Random color not matching any known terrain
    unknown_color = (100, 100, 100)
    create_test_gif(gif_path, make_uniform_pixels(unknown_color))

    terrain = TerrainMap(gif_path)
    assert terrain.get_terrain(0, 0) == TerrainMap.ROCK


def test_get_terrain_out_of_bounds_returns_space(tmp_path: Path) -> None:
    """Test get_terrain returns space for out-of-bounds coordinates."""
    gif_path = tmp_path / "test.gif"
    create_test_gif(gif_path, make_uniform_pixels(TerrainMap.COLOR_GROUND_42A))

    terrain = TerrainMap(gif_path)
    assert terrain.get_terrain(-1, 0) == " "
    assert terrain.get_terrain(0, -1) == " "
    assert terrain.get_terrain(256, 0) == " "
    assert terrain.get_terrain(0, 256) == " "
    assert terrain.get_terrain(300, 300) == " "


def test_get_terrain_all_corners(tmp_path: Path) -> None:
    """Test get_terrain works for all four corners."""
    gif_path = tmp_path / "test.gif"
    create_test_gif(gif_path, make_uniform_pixels(TerrainMap.COLOR_GROUND_42A))

    terrain = TerrainMap(gif_path)
    assert terrain.get_terrain(0, 0) == TerrainMap.GROUND
    assert terrain.get_terrain(255, 0) == TerrainMap.GROUND
    assert terrain.get_terrain(0, 255) == TerrainMap.GROUND
    assert terrain.get_terrain(255, 255) == TerrainMap.GROUND


def test_get_terrain_pixel_mapping(tmp_path: Path) -> None:
    """Test that specific pixel positions are correctly mapped."""
    gif_path = tmp_path / "mixed.gif"
    pixels = make_uniform_pixels(TerrainMap.COLOR_GROUND_42A)

    # Place a rock at (10, 5)
    pixels[5 * 256 + 10] = TerrainMap.COLOR_ROCK_42
    # Place water at (100, 50)
    pixels[50 * 256 + 100] = TerrainMap.COLOR_WATER_42

    create_test_gif(gif_path, pixels)

    terrain = TerrainMap(gif_path)
    assert terrain.get_terrain(10, 5) == TerrainMap.ROCK
    assert terrain.get_terrain(100, 50) == TerrainMap.WATER
    assert terrain.get_terrain(0, 0) == TerrainMap.GROUND


# =============================================================================
# TerrainMap.is_passable Tests
# =============================================================================


def test_is_passable_ground(tmp_path: Path) -> None:
    """Test is_passable returns True for ground."""
    gif_path = tmp_path / "ground.gif"
    create_test_gif(gif_path, make_uniform_pixels(TerrainMap.COLOR_GROUND_42A))

    terrain = TerrainMap(gif_path)
    assert terrain.is_passable(128, 128) is True


def test_is_passable_rock(tmp_path: Path) -> None:
    """Test is_passable returns False for rock."""
    gif_path = tmp_path / "rocks.gif"
    create_test_gif(gif_path, make_uniform_pixels(TerrainMap.COLOR_ROCK_42))

    terrain = TerrainMap(gif_path)
    assert terrain.is_passable(128, 128) is False


def test_is_passable_water(tmp_path: Path) -> None:
    """Test is_passable returns False for water."""
    gif_path = tmp_path / "water.gif"
    create_test_gif(gif_path, make_uniform_pixels(TerrainMap.COLOR_WATER_42))

    terrain = TerrainMap(gif_path)
    assert terrain.is_passable(128, 128) is False


# =============================================================================
# TerrainMap.render_viewport Tests
# =============================================================================


def test_render_viewport_default_size(tmp_path: Path) -> None:
    """Test render_viewport returns 16x16 grid by default."""
    gif_path = tmp_path / "test.gif"
    create_test_gif(gif_path, make_uniform_pixels(TerrainMap.COLOR_GROUND_42A))

    terrain = TerrainMap(gif_path)
    grid = terrain.render_viewport(128, 128)

    assert len(grid) == 16
    assert all(len(row) == 16 for row in grid)


def test_render_viewport_custom_size(tmp_path: Path) -> None:
    """Test render_viewport with custom dimensions."""
    gif_path = tmp_path / "test.gif"
    create_test_gif(gif_path, make_uniform_pixels(TerrainMap.COLOR_GROUND_42A))

    terrain = TerrainMap(gif_path)
    grid = terrain.render_viewport(128, 128, width=8, height=12)

    assert len(grid) == 12
    assert all(len(row) == 8 for row in grid)


def test_render_viewport_centered_on_position(tmp_path: Path) -> None:
    """Test render_viewport is correctly centered."""
    gif_path = tmp_path / "mixed.gif"
    pixels = make_uniform_pixels(TerrainMap.COLOR_GROUND_42A)

    # Place a rock at (128, 128)
    pixels[128 * 256 + 128] = TerrainMap.COLOR_ROCK_42

    create_test_gif(gif_path, pixels)

    terrain = TerrainMap(gif_path)
    grid = terrain.render_viewport(128, 128, width=16, height=16)

    # Center of 16x16 viewport centered on (128, 128) should be at grid[8][8]
    # The viewport covers (120-135, 120-135)
    # Position (128, 128) maps to grid index (8, 8)
    assert grid[8][8] == TerrainMap.ROCK


def test_render_viewport_edge_positions(tmp_path: Path) -> None:
    """Test render_viewport handles edge-of-map positions."""
    gif_path = tmp_path / "test.gif"
    create_test_gif(gif_path, make_uniform_pixels(TerrainMap.COLOR_GROUND_42A))

    terrain = TerrainMap(gif_path)

    # Near corner - should include out-of-bounds spaces
    grid = terrain.render_viewport(5, 5, width=16, height=16)
    assert len(grid) == 16
    # Top-left cells should be out-of-bounds (spaces)
    assert grid[0][0] == " "


# =============================================================================
# TerrainMap.render_full_map Tests
# =============================================================================


def test_render_full_map_size(tmp_path: Path) -> None:
    """Test render_full_map returns 256x256 grid."""
    gif_path = tmp_path / "test.gif"
    create_test_gif(gif_path, make_uniform_pixels(TerrainMap.COLOR_GROUND_42A))

    terrain = TerrainMap(gif_path)
    grid = terrain.render_full_map()

    assert len(grid) == 256
    assert all(len(row) == 256 for row in grid)


def test_render_full_map_content(tmp_path: Path) -> None:
    """Test render_full_map correctly renders all terrain types."""
    gif_path = tmp_path / "mixed.gif"
    pixels = make_uniform_pixels(TerrainMap.COLOR_GROUND_42A)

    # Add rock and water
    pixels[0] = TerrainMap.COLOR_ROCK_42  # Rock at (0, 0)
    pixels[255] = TerrainMap.COLOR_WATER_42  # Water at (255, 0)

    create_test_gif(gif_path, pixels)

    terrain = TerrainMap(gif_path)
    grid = terrain.render_full_map()

    assert grid[0][0] == TerrainMap.ROCK
    assert grid[0][255] == TerrainMap.WATER
    assert grid[128][128] == TerrainMap.GROUND


# =============================================================================
# format_viewport Tests
# =============================================================================


def test_format_viewport_empty_grid() -> None:
    """Test format_viewport returns empty string for empty grid."""
    result = format_viewport([])
    assert result == ""


def test_format_viewport_basic_output() -> None:
    """Test format_viewport produces correctly formatted output."""
    grid = [[".", ".", "#"], [".", "#", "."]]

    result = format_viewport(grid, viewport_left=10, viewport_top=20)

    lines = result.split("\n")
    assert len(lines) == 3  # header + 2 rows
    assert "(X: 10-12)" in lines[0]
    assert "20 " in lines[1]
    assert "21 " in lines[2]


def test_format_viewport_with_player() -> None:
    """Test format_viewport shows player at correct position."""
    grid = [[".", ".", "."], [".", ".", "."]]

    result = format_viewport(
        grid,
        player_x=11,
        player_y=20,
        viewport_left=10,
        viewport_top=20,
    )

    lines = result.split("\n")
    # Player should be at (11, 20) which is column 1, row 0 in viewport
    assert "@ " in lines[1]


def test_format_viewport_with_entities() -> None:
    """Test format_viewport shows entities at correct positions."""
    grid = [[".", ".", "."], [".", ".", "."]]
    entities = {(11, 20): "F", (12, 21): "E"}

    result = format_viewport(
        grid,
        viewport_left=10,
        viewport_top=20,
        entities=entities,
    )

    lines = result.split("\n")
    assert "F " in lines[1]
    assert "E " in lines[2]


def test_format_viewport_player_overrides_entity() -> None:
    """Test format_viewport shows player even when entity exists at position."""
    grid = [[".", ".", "."]]
    entities = {(11, 20): "F"}

    result = format_viewport(
        grid,
        player_x=11,
        player_y=20,
        viewport_left=10,
        viewport_top=20,
        entities=entities,
    )

    lines = result.split("\n")
    # Player @ should appear, not the entity F
    assert "@ " in lines[1]
    assert "F " not in lines[1]


def test_format_viewport_column_headers() -> None:
    """Test format_viewport shows correct column headers."""
    grid = [[".", ".", ".", "."]]

    result = format_viewport(grid, viewport_left=8, viewport_top=0)

    lines = result.split("\n")
    # Headers should be 8 % 10, 9 % 10, 0, 1 = 8, 9, 0, 1
    assert "8 9 0 1" in lines[0]
