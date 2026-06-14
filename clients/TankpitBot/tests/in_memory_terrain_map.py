"""In-memory ``TerrainMapProtocol`` data builder for tests.

Not a fake of behavior; this is what a small terrain map would look like
in production if the data fit in memory. Lifted out of ``tests/fakes``
because it is a pure data builder and not Playwright-related.
"""

from __future__ import annotations


class InMemoryTerrainMap:
    """In-memory ``TerrainMapProtocol`` for tests.

    Real implementation backed by a ``dict[(x, y), str]`` plus a
    configurable default tile. Callers seed the coordinates they care
    about; every other tile resolves to ``default``.

    The ``default`` parameter lifts two common test shapes into one
    class: "all ground" (the default) and "passable set as ground,
    rest as water" (via :meth:`from_passable_set`).
    """

    ROCK: str = "#"
    GROUND: str = "."
    WATER: str = "W"

    def __init__(
        self,
        terrain_data: dict[tuple[int, int], str] | None = None,
        *,
        default: str = GROUND,
    ) -> None:
        """Initialize an in-memory terrain map.

        Args:
            terrain_data: Dict mapping ``(x, y)`` to terrain character.
                Unmapped tiles resolve to ``default``.
            default: Tile returned for any coordinate not present in
                ``terrain_data``. Defaults to ``GROUND``.
        """
        self._terrain_data = terrain_data or {}
        self._default = default

    @classmethod
    def from_passable_set(
        cls,
        passable: set[tuple[int, int]],
    ) -> InMemoryTerrainMap:
        """Build a map where ``passable`` tiles are ground, rest is water.

        Args:
            passable: Coordinates considered walkable ground.

        Returns:
            Terrain map whose ``passable`` tiles resolve to ``GROUND``
            and every other coordinate resolves to ``WATER``.
        """
        return cls(
            dict.fromkeys(passable, cls.GROUND),
            default=cls.WATER,
        )

    def get_terrain(self, x: int, y: int) -> str:
        """Get terrain at coordinates.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            Terrain character.
        """
        return self._terrain_data.get((x, y), self._default)

    def is_passable(self, x: int, y: int) -> bool:
        """Check if terrain is passable.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            True if passable.
        """
        terrain = self.get_terrain(x, y)
        return terrain not in (self.ROCK, self.WATER)

    def render_viewport(
        self,
        center_x: int,
        center_y: int,
        width: int = 16,
        height: int = 16,
    ) -> list[list[str]]:
        """Render a viewport grid centered on the given coordinates.

        Args:
            center_x: Viewport center X.
            center_y: Viewport center Y.
            width: Viewport width in tiles.
            height: Viewport height in tiles.

        Returns:
            ``height`` rows of ``width`` terrain characters each.
        """
        left = center_x - width // 2
        top = center_y - height // 2
        grid: list[list[str]] = []
        for row in range(height):
            row_data: list[str] = []
            for col in range(width):
                row_data.append(self.get_terrain(left + col, top + row))
            grid.append(row_data)
        return grid


__all__ = ["InMemoryTerrainMap"]
