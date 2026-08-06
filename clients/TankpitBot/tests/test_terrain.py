"""Tests for tankpit_bot.terrain module."""

from __future__ import annotations

from pathlib import Path

import pytest

from tankpit_bot._pillow import load_pillow_image_module
from tankpit_bot.terrain import TerrainMap, format_viewport

_IMAGE = load_pillow_image_module()

# Arbitrary test colors — the histogram classifier assigns terrain by
# frequency, so these have no special meaning besides being distinct.
_GROUND_COLOR = (60, 129, 85)
_ROCK_COLOR = (140, 70, 30)
_WATER_COLOR = (30, 80, 200)


def create_test_gif(path: Path, pixels: list[tuple[int, int, int]]) -> None:
    """Create a 256x256 test GIF with specified pixels.

    Args:
        path: Path to save the GIF.
        pixels: List of 65536 RGB tuples (row-major order).
    """
    img = _IMAGE.new("RGB", (256, 256))
    img.putdata(pixels)
    img.save(path)


def _make_three_terrain_pixels() -> list[tuple[int, int, int]]:
    """Build a 256x256 pixel list with rock border, water pocket, ground interior.

    The border-based classifier identifies rock by checking the outermost
    rows/columns, so rock MUST appear on the border for correct classification.

    Returns:
        List of 65536 RGB tuples with rock on border, water in a pocket,
        and ground filling the rest.
    """
    pixels = [_GROUND_COLOR] * (256 * 256)
    for x in range(256):
        pixels[x] = _ROCK_COLOR
        pixels[255 * 256 + x] = _ROCK_COLOR
    for y in range(256):
        pixels[y * 256] = _ROCK_COLOR
        pixels[y * 256 + 255] = _ROCK_COLOR
    for y in range(100, 110):
        for x in range(100, 110):
            pixels[y * 256 + x] = _WATER_COLOR
    return pixels


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
    """Histogram assigns sole color as ground."""
    gif_path = tmp_path / "test.gif"
    create_test_gif(gif_path, make_uniform_pixels(_GROUND_COLOR))

    terrain = TerrainMap(gif_path)
    assert terrain.get_terrain(0, 0) == TerrainMap.GROUND


def test_terrain_map_accepts_string_path(tmp_path: Path) -> None:
    """TerrainMap accepts string path."""
    gif_path = tmp_path / "test.gif"
    create_test_gif(gif_path, make_uniform_pixels(_GROUND_COLOR))

    terrain = TerrainMap(str(gif_path))
    assert terrain.get_terrain(0, 0) == TerrainMap.GROUND


def test_terrain_map_raises_on_wrong_size(tmp_path: Path) -> None:
    """TerrainMap raises ValueError for non-256x256 images."""
    gif_path = tmp_path / "small.gif"
    img = _IMAGE.new("RGB", (128, 128), _GROUND_COLOR)
    img.save(gif_path)

    with pytest.raises(ValueError, match="Expected 256x256 image, got \\(128, 128\\)"):
        TerrainMap(gif_path)


# =============================================================================
# Border-based terrain classification
# =============================================================================


def test_border_classifies_three_colors(tmp_path: Path) -> None:
    """Rock on border, water in interior pocket, ground everywhere else."""
    gif_path = tmp_path / "three.gif"
    create_test_gif(gif_path, _make_three_terrain_pixels())

    terrain = TerrainMap(gif_path)
    assert terrain.get_terrain(0, 0) == TerrainMap.ROCK
    assert terrain.get_terrain(128, 128) == TerrainMap.GROUND
    assert terrain.get_terrain(105, 105) == TerrainMap.WATER


def test_border_classifies_two_colors(tmp_path: Path) -> None:
    """Rock on border, ground interior — no water present."""
    pixels = [_GROUND_COLOR] * (256 * 256)
    for x in range(256):
        pixels[x] = _ROCK_COLOR
        pixels[255 * 256 + x] = _ROCK_COLOR
    for y in range(256):
        pixels[y * 256] = _ROCK_COLOR
        pixels[y * 256 + 255] = _ROCK_COLOR
    gif_path = tmp_path / "two.gif"
    create_test_gif(gif_path, pixels)

    terrain = TerrainMap(gif_path)
    assert terrain.get_terrain(128, 128) == TerrainMap.GROUND
    assert terrain.get_terrain(0, 0) == TerrainMap.ROCK


def test_border_single_color(tmp_path: Path) -> None:
    """Single color: border and interior are the same, treated as ground."""
    gif_path = tmp_path / "one.gif"
    create_test_gif(gif_path, make_uniform_pixels((200, 150, 100)))

    terrain = TerrainMap(gif_path)
    assert terrain.get_terrain(128, 128) == TerrainMap.GROUND


def test_unknown_color_returns_rock(tmp_path: Path) -> None:
    """A color absent from the lookup defaults to rock."""
    gif_path = tmp_path / "mostly_ground.gif"
    create_test_gif(gif_path, make_uniform_pixels(_GROUND_COLOR))

    terrain = TerrainMap(gif_path)
    terrain._pixels[128 * 256 + 128] = (1, 2, 3)
    assert terrain.get_terrain(128, 128) == TerrainMap.ROCK


def test_gif_compression_variants_grouped_with_ground(tmp_path: Path) -> None:
    """Near-identical colors from GIF compression are grouped with ground."""
    pixels = [_GROUND_COLOR] * (256 * 256)
    for x in range(256):
        pixels[x] = _ROCK_COLOR
        pixels[255 * 256 + x] = _ROCK_COLOR
    for y in range(256):
        pixels[y * 256] = _ROCK_COLOR
        pixels[y * 256 + 255] = _ROCK_COLOR
    # Inject a near-ground variant (±1 per channel) at an interior pixel
    variant = (_GROUND_COLOR[0] + 1, _GROUND_COLOR[1] - 1, _GROUND_COLOR[2])
    pixels[50 * 256 + 50] = variant
    gif_path = tmp_path / "variant.gif"
    create_test_gif(gif_path, pixels)

    terrain = TerrainMap(gif_path)
    assert terrain.get_terrain(50, 50) == TerrainMap.GROUND


def test_real_field01_terrain_detection() -> None:
    """Real field01_r.gif classifies ground, rock, and water correctly."""
    terrain = TerrainMap(Path("field01_r.gif"))
    assert terrain.get_terrain(131, 110) == TerrainMap.GROUND
    assert terrain.get_terrain(125, 126) == TerrainMap.WATER
    assert terrain.get_terrain(0, 0) == TerrainMap.ROCK


def test_real_field42_terrain_detection() -> None:
    """Real field42-r.gif classifies ground, rock, and water correctly."""
    terrain = TerrainMap(Path("field42-r.gif"))
    assert terrain.get_terrain(128, 128) == TerrainMap.GROUND
    assert terrain.get_terrain(0, 0) == TerrainMap.ROCK
    assert terrain.is_passable(128, 128) is True
    assert terrain.is_passable(0, 0) is False


# =============================================================================
# TerrainMap.get_terrain Tests
# =============================================================================


def test_get_terrain_out_of_bounds_returns_space(tmp_path: Path) -> None:
    """get_terrain returns space for out-of-bounds coordinates."""
    gif_path = tmp_path / "test.gif"
    create_test_gif(gif_path, make_uniform_pixels(_GROUND_COLOR))

    terrain = TerrainMap(gif_path)
    assert terrain.get_terrain(-1, 0) == " "
    assert terrain.get_terrain(0, -1) == " "
    assert terrain.get_terrain(256, 0) == " "
    assert terrain.get_terrain(0, 256) == " "
    assert terrain.get_terrain(300, 300) == " "


def test_get_terrain_all_corners(tmp_path: Path) -> None:
    """get_terrain works for all four corners."""
    gif_path = tmp_path / "test.gif"
    create_test_gif(gif_path, make_uniform_pixels(_GROUND_COLOR))

    terrain = TerrainMap(gif_path)
    assert terrain.get_terrain(0, 0) == TerrainMap.GROUND
    assert terrain.get_terrain(255, 0) == TerrainMap.GROUND
    assert terrain.get_terrain(0, 255) == TerrainMap.GROUND
    assert terrain.get_terrain(255, 255) == TerrainMap.GROUND


def test_get_terrain_pixel_mapping(tmp_path: Path) -> None:
    """Specific pixel positions correctly mapped via border classification."""
    gif_path = tmp_path / "mixed.gif"
    pixels = _make_three_terrain_pixels()
    create_test_gif(gif_path, pixels)

    terrain = TerrainMap(gif_path)
    assert terrain.get_terrain(0, 0) == TerrainMap.ROCK
    assert terrain.get_terrain(128, 128) == TerrainMap.GROUND
    assert terrain.get_terrain(105, 105) == TerrainMap.WATER


# =============================================================================
# TerrainMap.is_passable Tests
# =============================================================================


def test_is_passable_ground(tmp_path: Path) -> None:
    """is_passable returns True for ground."""
    gif_path = tmp_path / "ground.gif"
    create_test_gif(gif_path, make_uniform_pixels(_GROUND_COLOR))

    terrain = TerrainMap(gif_path)
    assert terrain.is_passable(128, 128) is True


def test_is_passable_rock(tmp_path: Path) -> None:
    """is_passable returns False for rock (border tile)."""
    gif_path = tmp_path / "rocks.gif"
    create_test_gif(gif_path, _make_three_terrain_pixels())

    terrain = TerrainMap(gif_path)
    assert terrain.is_passable(0, 0) is False


def test_is_passable_water(tmp_path: Path) -> None:
    """is_passable returns False for water (interior pocket)."""
    gif_path = tmp_path / "water.gif"
    create_test_gif(gif_path, _make_three_terrain_pixels())

    terrain = TerrainMap(gif_path)
    assert terrain.is_passable(105, 105) is False


# =============================================================================
# TerrainMap.render_viewport Tests
# =============================================================================


def test_render_viewport_default_size(tmp_path: Path) -> None:
    """render_viewport returns 16x16 grid by default."""
    gif_path = tmp_path / "test.gif"
    create_test_gif(gif_path, make_uniform_pixels(_GROUND_COLOR))

    terrain = TerrainMap(gif_path)
    grid = terrain.render_viewport(128, 128)

    assert len(grid) == 16
    assert all(len(row) == 16 for row in grid)


def test_render_viewport_custom_size(tmp_path: Path) -> None:
    """render_viewport with custom dimensions."""
    gif_path = tmp_path / "test.gif"
    create_test_gif(gif_path, make_uniform_pixels(_GROUND_COLOR))

    terrain = TerrainMap(gif_path)
    grid = terrain.render_viewport(128, 128, width=8, height=12)

    assert len(grid) == 12
    assert all(len(row) == 8 for row in grid)


def test_render_viewport_centered_on_position(tmp_path: Path) -> None:
    """render_viewport is correctly centered."""
    gif_path = tmp_path / "mixed.gif"
    pixels = _make_three_terrain_pixels()
    create_test_gif(gif_path, pixels)

    terrain = TerrainMap(gif_path)
    # Water pocket at (100-109, 100-109) — center on (105,105) to see water at grid center
    grid = terrain.render_viewport(105, 105, width=16, height=16)
    assert grid[8][8] == TerrainMap.WATER


def test_render_viewport_edge_positions(tmp_path: Path) -> None:
    """render_viewport handles edge-of-map positions."""
    gif_path = tmp_path / "test.gif"
    create_test_gif(gif_path, make_uniform_pixels(_GROUND_COLOR))

    terrain = TerrainMap(gif_path)
    grid = terrain.render_viewport(5, 5, width=16, height=16)
    assert len(grid) == 16
    assert grid[0][0] == " "


# =============================================================================
# TerrainMap.render_full_map Tests
# =============================================================================


def test_render_full_map_size(tmp_path: Path) -> None:
    """render_full_map returns 256x256 grid."""
    gif_path = tmp_path / "test.gif"
    create_test_gif(gif_path, make_uniform_pixels(_GROUND_COLOR))

    terrain = TerrainMap(gif_path)
    grid = terrain.render_full_map()

    assert len(grid) == 256
    assert all(len(row) == 256 for row in grid)


def test_render_full_map_content(tmp_path: Path) -> None:
    """render_full_map correctly renders all terrain types."""
    gif_path = tmp_path / "mixed.gif"
    pixels = _make_three_terrain_pixels()
    create_test_gif(gif_path, pixels)

    terrain = TerrainMap(gif_path)
    grid = terrain.render_full_map()

    assert grid[0][0] == TerrainMap.ROCK
    assert grid[128][128] == TerrainMap.GROUND
    assert grid[105][105] == TerrainMap.WATER


# =============================================================================
# format_viewport Tests
# =============================================================================


def test_format_viewport_empty_grid() -> None:
    """format_viewport returns empty string for empty grid."""
    result = format_viewport([])
    assert result == ""


def test_format_viewport_basic_output() -> None:
    """format_viewport produces correctly formatted output."""
    grid = [[".", ".", "#"], [".", "#", "."]]

    result = format_viewport(grid, viewport_left=10, viewport_top=20)

    lines = result.split("\n")
    assert len(lines) == 3  # header + 2 rows
    assert "(X: 10-12)" in lines[0]
    assert "20 " in lines[1]
    assert "21 " in lines[2]


def test_format_viewport_with_player() -> None:
    """format_viewport shows player at correct position."""
    grid = [[".", ".", "."], [".", ".", "."]]

    result = format_viewport(
        grid,
        player_x=11,
        player_y=20,
        viewport_left=10,
        viewport_top=20,
    )

    lines = result.split("\n")
    assert "@ " in lines[1]


def test_format_viewport_with_entities() -> None:
    """format_viewport shows entities at correct positions."""
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
    """format_viewport shows player even when entity exists at position."""
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
    assert "@ " in lines[1]
    assert "F " not in lines[1]


def test_format_viewport_column_headers() -> None:
    """format_viewport shows correct column headers."""
    grid = [[".", ".", ".", "."]]

    result = format_viewport(grid, viewport_left=8, viewport_top=0)

    lines = result.split("\n")
    assert "8 9 0 1" in lines[0]


def test_static_map_attainability_collapses_to_legality(tmp_path: Path) -> None:
    """The static map carries no mine knowledge: attainable == legal."""
    gif_path = tmp_path / "test.gif"
    create_test_gif(gif_path, make_uniform_pixels(_GROUND_COLOR))
    terrain = TerrainMap(gif_path)

    assert terrain.is_landing_attainable(0, 0) is terrain.is_landing_legal(0, 0)
    assert terrain.is_landing_attainable(0, 0) is True
