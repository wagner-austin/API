"""Tests for the lifted shot-clearance (line-of-sight) primitives."""

from __future__ import annotations

from tankpit_bot.state.line_of_sight import is_shot_line_clear, shot_line_tiles
from tankpit_bot.state.types import TerrainTileDict, make_terrain_tile
from tankpit_bot.types.constants import (
    TERRAIN_BLOCK_BRIDGE,
    TERRAIN_BLOCK_LAND,
    TERRAIN_BLOCK_STACKED,
    TERRAIN_FERRY,
    TERRAIN_FERRY_ROCK,
    TERRAIN_GROUND,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _wire(entries: dict[tuple[int, int], int]) -> dict[str, TerrainTileDict]:
    """Build a wire-terrain dict from ``(x, y) -> terrain_type``.

    Args:
        entries: Patched tiles and their wire terrain types.

    Returns:
        World-state style terrain dict keyed ``"x,y"``.
    """
    return {
        f"{x},{y}": make_terrain_tile(x=x, y=y, terrain_type=terrain_type)
        for (x, y), terrain_type in entries.items()
    }


class TestShotLineTiles:
    """Tests for the Bresenham raster with excluded endpoints."""

    def test_cardinal_line_excludes_endpoints(self) -> None:
        """A straight east shot rasters only the tiles strictly between."""
        assert shot_line_tiles(10, 20, 14, 20) == [(11, 20), (12, 20), (13, 20)]

    def test_adjacent_target_has_no_intermediate_tiles(self) -> None:
        """Point-blank shots have nothing to occlude them."""
        assert shot_line_tiles(10, 20, 11, 20) == []
        assert shot_line_tiles(10, 20, 10, 21) == []

    def test_diagonal_line_rasters_the_diagonal(self) -> None:
        """A perfect diagonal steps both axes each tile."""
        assert shot_line_tiles(0, 0, 4, 4) == [(1, 1), (2, 2), (3, 3)]

    def test_shallow_line_stays_within_one_row_of_the_ideal(self) -> None:
        """A 2:1 slope rasters the mixed steps between the endpoints."""
        tiles = shot_line_tiles(0, 0, 6, 3)
        assert tiles[0] == (1, 1) or tiles[0] == (1, 0)
        assert len(tiles) == 5
        assert all(abs(y - x / 2) <= 1 for x, y in tiles)

    def test_westward_line_is_symmetric(self) -> None:
        """Negative-direction rasters mirror the positive ones."""
        assert shot_line_tiles(14, 20, 10, 20) == [(13, 20), (12, 20), (11, 20)]


class TestIsShotLineClear:
    """Tests for the occlusion rules on the shot line."""

    def test_open_ground_line_is_clear(self) -> None:
        """All-ground static terrain with no patches never occludes."""
        assert is_shot_line_clear(10, 20, 14, 20, InMemoryTerrainMap(), {}) is True

    def test_static_rock_blocks_the_line(self) -> None:
        """A mountain tile between shooter and target interrupts the shot."""
        terrain = InMemoryTerrainMap({(12, 20): InMemoryTerrainMap.ROCK})

        assert is_shot_line_clear(10, 20, 14, 20, terrain, {}) is False

    def test_water_never_blocks(self) -> None:
        """A water channel between shooter and target does not occlude."""
        terrain = InMemoryTerrainMap(
            {(12, 20): InMemoryTerrainMap.WATER, (13, 20): InMemoryTerrainMap.WATER}
        )

        assert is_shot_line_clear(10, 20, 14, 20, terrain, {}) is True

    def test_rock_on_the_target_tile_does_not_occlude_its_own_shot(self) -> None:
        """Endpoints are excluded: aiming AT a rocky tile is the server's call."""
        terrain = InMemoryTerrainMap({(14, 20): InMemoryTerrainMap.ROCK})

        assert is_shot_line_clear(10, 20, 14, 20, terrain, {}) is True

    def test_each_movable_block_form_blocks_the_line(self) -> None:
        """Bridge, land, stacked, and ferry-rock wire patches all occlude."""
        for blocking_type in (
            TERRAIN_BLOCK_BRIDGE,
            TERRAIN_BLOCK_LAND,
            TERRAIN_BLOCK_STACKED,
            TERRAIN_FERRY_ROCK,
        ):
            wire = _wire({(12, 20): blocking_type})
            assert is_shot_line_clear(10, 20, 14, 20, InMemoryTerrainMap(), wire) is False

    def test_ground_and_ferry_patches_do_not_block(self) -> None:
        """Non-block wire patches leave the line clear."""
        wire = _wire({(12, 20): TERRAIN_GROUND, (13, 20): TERRAIN_FERRY})

        assert is_shot_line_clear(10, 20, 14, 20, InMemoryTerrainMap(), wire) is True

    def test_wire_ground_patch_overrides_static_rock(self) -> None:
        """A cleared tile un-occludes even where the field image shows rock."""
        terrain = InMemoryTerrainMap({(12, 20): InMemoryTerrainMap.ROCK})
        wire = _wire({(12, 20): TERRAIN_GROUND})

        assert is_shot_line_clear(10, 20, 14, 20, terrain, wire) is True

    def test_none_static_terrain_trusts_the_wire_layer_alone(self) -> None:
        """With no field image loaded only wire blocks can occlude."""
        assert is_shot_line_clear(10, 20, 14, 20, None, {}) is True
        wire = _wire({(12, 20): TERRAIN_BLOCK_LAND})
        assert is_shot_line_clear(10, 20, 14, 20, None, wire) is False
