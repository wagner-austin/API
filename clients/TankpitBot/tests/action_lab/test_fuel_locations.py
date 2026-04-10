"""Tests for distinct fuel probe ground-target selection."""

from __future__ import annotations

import pytest

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.action_lab.fuel_locations import build_distinct_ground_targets


class _Terrain:
    """Minimal terrain map fake for ground-target tests."""

    ROCK = "#"
    GROUND = "."
    WATER = "W"

    def __init__(self, passable: set[tuple[int, int]]) -> None:
        """Store the passable tile set.

        Args:
            passable: Coordinates considered walkable ground.
        """
        self._passable = passable

    def get_terrain(self, x: int, y: int) -> str:
        """Return a terrain character for one tile.

        Args:
            x: Tile X coordinate.
            y: Tile Y coordinate.

        Returns:
            ``"."`` for passable tiles, otherwise ``"W"``.
        """
        return self.GROUND if (x, y) in self._passable else self.WATER

    def is_passable(self, x: int, y: int) -> bool:
        """Return whether a tile is walkable.

        Args:
            x: Tile X coordinate.
            y: Tile Y coordinate.

        Returns:
            True when the tile is in the passable set.
        """
        return (x, y) in self._passable

    def render_viewport(
        self,
        center_x: int,
        center_y: int,
        width: int = 16,
        height: int = 16,
    ) -> list[list[str]]:
        """Render a viewport-sized terrain grid.

        Args:
            center_x: Viewport center X coordinate.
            center_y: Viewport center Y coordinate.
            width: Viewport width in tiles.
            height: Viewport height in tiles.

        Returns:
            Terrain grid centered on ``center_x`` and ``center_y``.
        """
        rows: list[list[str]] = []
        left = center_x - (width // 2)
        top = center_y - (height // 2)
        for y in range(top, top + height):
            row: list[str] = []
            for x in range(left, left + width):
                row.append(self.get_terrain(x, y))
            rows.append(row)
        return rows


def _terrain(passable: set[tuple[int, int]]) -> TerrainMapProtocol:
    """Build a typed terrain fake.

    Args:
        passable: Passable coordinates.

    Returns:
        Typed terrain fake.
    """
    return _Terrain(passable)


def _fill_neighborhood(passable: set[tuple[int, int]], x: int, y: int) -> None:
    """Mark a full 3x3 neighborhood as passable.

    Args:
        passable: Mutable passable tile set.
        x: Neighborhood center X coordinate.
        y: Neighborhood center Y coordinate.
    """
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            passable.add((x + dx, y + dy))


def test_build_distinct_ground_targets_prefers_ground_heavy_unique_tiles() -> None:
    """Ground-target selection prefers fully passable nearby tiles."""
    passable: set[tuple[int, int]] = set()
    _fill_neighborhood(passable, 124, 100)
    _fill_neighborhood(passable, 100, 124)
    _fill_neighborhood(passable, 76, 100)
    passable.add((100, 76))

    targets = build_distinct_ground_targets(
        100,
        100,
        _terrain(passable),
        count=3,
        step=24,
        max_radius=48,
    )

    assert targets == [
        {"label": "fuel_ground_76_100", "x": 76, "y": 100},
        {"label": "fuel_ground_124_100", "x": 124, "y": 100},
        {"label": "fuel_ground_100_124", "x": 100, "y": 124},
    ]


def test_build_distinct_ground_targets_clamps_to_map_edges() -> None:
    """Ground-target selection clamps ring coordinates to valid map bounds."""
    passable = {
        (0, 24),
        (24, 0),
        (24, 24),
    }

    targets = build_distinct_ground_targets(
        0,
        0,
        _terrain(passable),
        count=3,
        step=24,
        max_radius=24,
    )

    assert [(target["x"], target["y"]) for target in targets] == [(24, 0), (0, 24), (24, 24)]


def test_build_distinct_ground_targets_clamps_upper_map_edges() -> None:
    """Ground-target selection clamps oversized ring coordinates to the map maximum."""
    passable = {
        (255, 250),
        (250, 255),
        (255, 255),
    }

    targets = build_distinct_ground_targets(
        250,
        250,
        _terrain(passable),
        count=3,
        step=24,
        max_radius=24,
    )

    assert [(target["x"], target["y"]) for target in targets] == [
        (255, 250),
        (250, 255),
        (255, 255),
    ]


def test_build_distinct_ground_targets_respects_excluded_coordinates() -> None:
    """Ground-target selection skips already-used coordinates."""
    passable: set[tuple[int, int]] = set()
    _fill_neighborhood(passable, 124, 100)
    _fill_neighborhood(passable, 100, 124)
    _fill_neighborhood(passable, 76, 100)

    targets = build_distinct_ground_targets(
        100,
        100,
        _terrain(passable),
        count=2,
        step=24,
        max_radius=48,
        excluded=frozenset({(76, 100)}),
    )

    assert targets == [
        {"label": "fuel_ground_124_100", "x": 124, "y": 100},
        {"label": "fuel_ground_100_124", "x": 100, "y": 124},
    ]


@pytest.mark.parametrize(
    ("count", "step", "max_radius", "message"),
    [
        (0, 24, 96, "count must be positive"),
        (1, 0, 96, "step must be positive"),
        (1, 24, 12, "max_radius must be at least step"),
    ],
)
def test_build_distinct_ground_targets_rejects_invalid_arguments(
    count: int,
    step: int,
    max_radius: int,
    message: str,
) -> None:
    """Ground-target selection validates configuration arguments."""
    with pytest.raises(ValueError, match=message):
        build_distinct_ground_targets(
            100,
            100,
            _terrain(set()),
            count=count,
            step=step,
            max_radius=max_radius,
        )


def test_build_distinct_ground_targets_requires_enough_passable_tiles() -> None:
    """Ground-target selection raises when too few distinct targets exist."""
    with pytest.raises(ValueError, match="not enough distinct passable targets"):
        build_distinct_ground_targets(
            100,
            100,
            _terrain({(124, 100)}),
            count=2,
            step=24,
            max_radius=24,
        )
