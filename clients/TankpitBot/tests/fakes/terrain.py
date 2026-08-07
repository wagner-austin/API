"""In-memory terrain map fake.

Mirrors ``TerrainMapProtocol`` semantics exactly so reachability logic
exercised against it behaves as it does live.
"""

from __future__ import annotations


class InMemoryTerrainMap:
    """In-memory ``TerrainMapProtocol`` for tests.

    Real implementation backed by a ``dict[(x, y), str]`` plus a
    configurable default tile. Callers seed the coordinates they care
    about; every other tile resolves to ``default``. Not a fake of
    behavior; this is what a small terrain map would look like in
    production if the data fit in memory.

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

    def is_landing_legal(self, x: int, y: int) -> bool:
        """Check if the server may place the tank on the tile.

        Terrain-only data, so the walk and landing questions coincide.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            True if a teleport may be aimed at the tile.
        """
        return self.is_passable(x, y)

    def is_landing_attainable(self, x: int, y: int) -> bool:
        """Terrain-only fake: attainability collapses to legality.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            True if a teleport aimed here lands here.
        """
        return self.is_landing_legal(x, y)

    def render_viewport(
        self,
        center_x: int,
        center_y: int,
        width: int = 16,
        height: int = 16,
    ) -> list[list[str]]:
        """Render viewport grid.

        Args:
            center_x: Center X.
            center_y: Center Y.
            width: Viewport width.
            height: Viewport height.

        Returns:
            2D list of terrain characters.
        """
        left = center_x - width // 2
        top = center_y - height // 2
        grid: list[list[str]] = []
        for row in range(height):
            row_data: list[str] = []
            for col in range(width):
                x = left + col
                y = top + row
                row_data.append(self.get_terrain(x, y))
            grid.append(row_data)
        return grid
