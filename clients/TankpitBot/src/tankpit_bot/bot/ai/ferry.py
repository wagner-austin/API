"""Ferry-aware terrain composition and surface-transition move clamping.

The static minimap cannot know about ferries: they are dynamic wire
terrain (``TERRAIN_FERRY``) that arrives with viewport patches and
moves with the tank that rides them. This module owns the ferry
concern end to end:

- :class:`FerryAwareTerrain` composes the live wire ferry tiles over
  the static minimap and encodes the riding rule -- a ferry tile is
  always passable (boarding), and water is passable exactly while the
  tank's own tile is a ferry (ferries can go anywhere on water).
- :func:`clamp_move_target_at_surface_transition` bounds a planned
  move at the first queue-consuming surface transition. Boarding a
  ferry and stepping from ferry/water onto land each consume one
  action-queue slot: the server stops the tank on the transition tile,
  so a move planned past it would stall against its own target.
  Clamping makes the planned target and the server's stop tile the
  same tile, and the next tick replans from there.

Live origin: run 20260612-131003 sat "marooned" at 7 fuel for 28
minutes on what the model called a one-tile island -- the tank was
standing on a ferry the whole time, with known fuel two tiles away
across the water.
"""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.equipment import hostile_mines
from tankpit_bot.bot.ai.pathfinding import find_path
from tankpit_bot.state.occupancy import occupied_tank_keys
from tankpit_bot.state.types import (
    TerrainTileDict,
    WorldStateDict,
    coord_key,
)
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds
from tankpit_bot.types.constants import (
    TERRAIN_BLOCK_BRIDGE,
    TERRAIN_BLOCK_LAND,
    TERRAIN_BLOCK_STACKED,
    TERRAIN_FERRY,
)

_ASCII_FERRY = "~"


class FerryAwareTerrain:
    """Terrain view composing every dynamic blocker over the static map.

    Four blocker classes fold into one passability answer: live wire
    ferry tiles and movable blocks from the wire terrain overlay,
    hostile mines from the mine registry, and other tanks' bodies from
    the tank registry. Implements :class:`TerrainMapProtocol`, so every
    existing passability consumer (pathfinding, reachability, movement
    planning, exploration) inherits all four through composition
    without per-callsite changes.
    """

    ROCK = "#"
    GROUND = "."
    WATER = "W"

    def __init__(
        self,
        base: TerrainMapProtocol,
        wire_terrain: dict[str, TerrainTileDict],
        *,
        riding: bool,
        hostile_mine_keys: frozenset[str],
        occupied_tank_keys: frozenset[str],
    ) -> None:
        """Initialize the composed terrain view.

        Args:
            base: Static minimap terrain.
            wire_terrain: Live wire terrain tiles keyed by "x,y".
            riding: Whether the tank's own tile is currently a ferry.
            hostile_mine_keys: "x,y" keys of known hostile mines. A
                hostile-mine tile cannot be WALKED onto (detonation
                costs 45 fuel), so it is impassable in this view.
                Teleport LANDING legality is a different question --
                the server displaces off mines on landing -- and is
                deliberately not answered here (see
                ``find_teleport_landing_tile``).
            occupied_tank_keys: "x,y" keys of tiles holding another
                tank's body (:func:`~tankpit_bot.state.occupancy.
                occupied_tank_keys`). A body stops a walk at the tile
                before it, so it is impassable here for the same
                reason a mine is.
        """
        self._base = base
        self._wire_terrain = wire_terrain
        self._riding = riding
        self._hostile_mine_keys = hostile_mine_keys
        self._occupied_tank_keys = occupied_tank_keys

    def get_terrain(self, x: int, y: int) -> str:
        """Get terrain type at game coordinates.

        Live wire tiles override the static map: ferries render as
        ``~``; movable concrete blocks ([[movable-blocks]]) collapse
        to their walkability class -- a block in water (wire value 1)
        is a walkable bridge and reads as ground, a block on land
        (value 2) or a stacked block (value 3) is an obstacle and
        reads as rock. Archive evidence (228 room-1 sessions,
        2026-07-20): value 1 appears ONLY over static water, value 2
        ONLY over static ground, value 3 ONLY over static water --
        the wire value alone determines walkability.

        Args:
            x: X coordinate (0-255).
            y: Y coordinate (0-255).

        Returns:
            Terrain character: ``~`` for a live ferry tile, otherwise
            ``#``, ``.``, or ``W``.
        """
        tile = self._wire_terrain.get(coord_key(x, y))
        if tile is not None:
            terrain_type = tile["terrain_type"]
            if terrain_type == TERRAIN_FERRY:
                return _ASCII_FERRY
            if terrain_type == TERRAIN_BLOCK_BRIDGE:
                return self.GROUND
            if terrain_type in (TERRAIN_BLOCK_LAND, TERRAIN_BLOCK_STACKED):
                return self.ROCK
        return self._base.get_terrain(x, y)

    def is_passable(self, x: int, y: int) -> bool:
        """Check if a tile can be moved onto right now.

        Ground is always passable, a ferry tile is always passable
        (boarding), and water is passable exactly while riding a
        ferry -- ferries can go anywhere on water. A known hostile
        mine makes any tile impassable: stepping on it detonates for
        45 fuel. Another tank's body makes any tile impassable: the
        server walks us up to it and stops, then reports
        ``error_code=1`` ([[walk-mechanics]] user contract
        2026-08-04).

        Composing every dynamic blocker here (like ferries) means each
        passability consumer -- pathfinding, reachability, selectors,
        clamps -- shares ONE answer to "can I walk here", instead of
        each threading a separate parameter it can forget (run
        2026-07-20 17:16: the dot-hop selector consulted terrain but
        not mines and looped 23 ticks against the executor's mine
        veto; run 2026-08-03 18:22-18:40: ten code-1 stops on routes
        the mine-and-block-only view believed were open).

        Args:
            x: X coordinate (0-255).
            y: Y coordinate (0-255).

        Returns:
            True if the tank can enter the tile this tick.
        """
        key = coord_key(x, y)
        if key in self._hostile_mine_keys or key in self._occupied_tank_keys:
            return False
        return self.is_landing_legal(x, y)

    def is_landing_legal(self, x: int, y: int) -> bool:
        """Check if the server may place the tank on the tile.

        Terrain legality only: ground and ferry tiles always qualify,
        water qualifies exactly while riding. Mines and tank bodies are
        deliberately ignored -- the server displaces the landing off
        both and charges the plain teleport cost (``mine-mechanics``,
        live-proven 2026-07-28; occupied tiles share the displacement
        rule). Asking the walk question here would make an approach
        teleport at any enemy impossible, since an enemy always
        occupies its own tile.

        Args:
            x: X coordinate (0-255).
            y: Y coordinate (0-255).

        Returns:
            True if a teleport may be aimed at the tile.
        """
        cell = self.get_terrain(x, y)
        if cell == _ASCII_FERRY or cell == self.GROUND:
            return True
        if cell == self.WATER:
            return self._riding
        return False

    def is_landing_attainable(self, x: int, y: int) -> bool:
        """Check if a teleport aimed here will actually stand here.

        Landing legality intersected with the composed view's
        TEAM-SCOPED hostile-mine set — the same set the walk side
        already consumes, built once per tick from the self model's
        team. Own-color mines never displace ([[mine-mechanics]]
        § team scope, archive 2026-08-06: 1,227 enemy vs 2 friendly),
        so they are absent from this set by construction and never
        repel a landing.

        Args:
            x: X coordinate (0-255).
            y: Y coordinate (0-255).

        Returns:
            True if a teleport aimed here lands here.
        """
        if not self.is_landing_legal(x, y):
            return False
        return f"{x},{y}" not in self._hostile_mine_keys

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
            2D list of terrain characters including live ferry tiles.
        """
        left = center_x - width // 2
        top = center_y - height // 2
        return [
            [self.get_terrain(left + vx, top + vy) for vx in range(width)] for vy in range(height)
        ]


class SurfaceRouteTerrain:
    """Terrain view restricted to one routing surface for pickups.

    Encodes the single-action routing surface for server-routed
    pickups (user contract 2026-07-19/20): one command never chains
    surfaces -- the server routes each click on the tank's CURRENT
    surface only, and a click it cannot reach that way draws "You
    can't go there!". Standing on land the surface is plain ground
    (water and ferry tiles block: crossing onto a ferry is a
    queue-consuming boarding action, not a walk step). Riding a ferry
    the surface is water (water and ferry tiles pass, land blocks) --
    a container floating on water picks up normally while riding.

    Live origins: run 2026-07-19 18:20:33 -- riding at (167,40), the
    riding rule made ALL tiles passable, the gate approved a LAND
    container across a channel, and the server refused with code 1
    after the disembark stop. Run 2026-07-20 00:57 -- the ground-only
    overcorrection: the bot sailed onto a water container's own tile
    and sat there 78 ticks refusing to dispatch the pickup because a
    water tile is never "ground-reachable".
    """

    ROCK = "#"
    GROUND = "."
    WATER = "W"

    def __init__(self, base: TerrainMapProtocol, *, water: bool) -> None:
        """Wrap any terrain view with single-surface passability.

        Args:
            base: Terrain view to read cells from (static or
                ferry-aware; excluded cells stay visible, just never
                traversable).
            water: The routing surface -- True when the tank is riding
                a ferry (water/ferry tiles pass), False on land (plain
                ground passes).
        """
        self._base = base
        self._water = water

    def get_terrain(self, x: int, y: int) -> str:
        """Get terrain type at game coordinates.

        Args:
            x: X coordinate (0-255).
            y: Y coordinate (0-255).

        Returns:
            The wrapped view's terrain character.
        """
        return self._base.get_terrain(x, y)

    def is_passable(self, x: int, y: int) -> bool:
        """Check if a single-action server route may cross the tile.

        Intersects the wrapped view's passability (which composes
        hostile mines) with the surface class, so a mined tile is
        never routable regardless of surface.

        Args:
            x: X coordinate (0-255).
            y: Y coordinate (0-255).

        Returns:
            True when the tile is passable in the wrapped view AND
            lies on the routing surface: water or ferry while riding,
            plain ground on land.
        """
        if not self._base.is_passable(x, y):
            return False
        return self._is_on_routing_surface(x, y)

    def is_landing_legal(self, x: int, y: int) -> bool:
        """Check if the server may place the tank on the tile.

        Intersects the wrapped view's landing legality with the routing
        surface, so this view answers the landing question without the
        wrapped view's walk-only blockers (mines, tank bodies).

        Args:
            x: X coordinate (0-255).
            y: Y coordinate (0-255).

        Returns:
            True if a teleport may be aimed at the tile.
        """
        return self._is_on_routing_surface(x, y)

    def is_landing_attainable(self, x: int, y: int) -> bool:
        """Check if a teleport aimed here will actually stand here.

        Delegates to the wrapped view's attainability (which carries
        the team-scoped hostile-mine knowledge) intersected with the
        routing surface.

        Args:
            x: X coordinate (0-255).
            y: Y coordinate (0-255).

        Returns:
            True if a teleport aimed here lands here on this surface.
        """
        if not self._base.is_landing_attainable(x, y):
            return False
        return self._is_on_routing_surface(x, y)

    def _is_on_routing_surface(self, x: int, y: int) -> bool:
        """Return whether the tile lies on the current routing surface.

        Args:
            x: X coordinate (0-255).
            y: Y coordinate (0-255).

        Returns:
            True for water and ferry tiles while riding, plain ground
            on land.
        """
        cell = self.get_terrain(x, y)
        if self._water:
            return cell == _ASCII_FERRY or cell == self.WATER
        return cell == self.GROUND

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
            The wrapped view's rendering (display is unchanged; only
            passability differs).
        """
        return self._base.render_viewport(center_x, center_y, width, height)


def is_riding_ferry(world: WorldStateDict) -> bool:
    """Return True when the tank's own tile is a live ferry tile.

    Args:
        world: Current world state with wire terrain and self state.

    Returns:
        True if the tank is standing on a ferry.
    """
    self_state = world["self_state"]
    if self_state is None:
        return False
    tile = world["terrain"].get(coord_key(self_state["x"], self_state["y"]))
    return tile is not None and tile["terrain_type"] == TERRAIN_FERRY


def compose_decision_terrain(
    world: WorldStateDict,
    terrain: TerrainMapProtocol | None,
    now_ms: int,
) -> TerrainMapProtocol | None:
    """Compose the decision terrain view for one tick.

    Assembles all four blocker classes the server routes around into a
    single passability answer: static minimap terrain, movable blocks
    and ferries from the wire terrain overlay, hostile mines from the
    mine registry, and other tanks' bodies from the tank registry.

    Args:
        world: Current world state with wire terrain, mines, tanks and
            self state.
        terrain: Static minimap terrain, or None when unavailable.
        now_ms: Current wall-clock time in milliseconds, used to age
            tank observations out of the occupancy set. Callers pass
            the tick timestamp they already hold, or read the canonical
            clock (``_test_hooks.get_current_time_ms``) -- never a
            second clock source.

    Returns:
        Composed terrain view, or None when no static map exists.
    """
    if terrain is None:
        return None
    return FerryAwareTerrain(
        terrain,
        world["terrain"],
        riding=is_riding_ferry(world),
        hostile_mine_keys=frozenset(hostile_mines(world)),
        occupied_tank_keys=occupied_tank_keys(world, now_ms),
    )


def _is_water_class(terrain: TerrainMapProtocol, x: int, y: int) -> bool:
    """Return True for tiles the tank traverses by ferry.

    Args:
        terrain: Terrain view for cell lookups.
        x: X coordinate.
        y: Y coordinate.

    Returns:
        True for water and ferry tiles, False for land.
    """
    cell = terrain.get_terrain(x, y)
    return cell == _ASCII_FERRY or cell == FerryAwareTerrain.WATER


def clamp_move_target_at_surface_transition(
    world: WorldStateDict,
    terrain: TerrainMapProtocol,
    start_x: int,
    start_y: int,
    target_x: int,
    target_y: int,
) -> tuple[int, int]:
    """Bound a planned move at the first queue-consuming transition.

    Boarding a ferry (land to ferry) and disembarking (ferry/water to
    land) each consume one action-queue slot: the server stops the
    tank ON the transition tile regardless of the clicked target. The
    planned move target must therefore be that transition tile, so
    arrival matches the plan and the next tick replans the remainder.

    Args:
        world: Current world state with visible viewport bounds.
        terrain: Ferry-aware terrain view used for planning.
        start_x: Starting X coordinate.
        start_y: Starting Y coordinate.
        target_x: Requested move target X coordinate.
        target_y: Requested move target Y coordinate.

    Returns:
        ``(x, y)`` of the first surface-transition tile along the
        planned path, or the original target when the path never
        changes surface class.
    """
    left, top, right, bottom = viewport_visible_bounds(world["viewport"])
    path = find_path(
        terrain,
        start_x,
        start_y,
        target_x,
        target_y,
        min_x=left,
        min_y=top,
        max_x=right,
        max_y=bottom,
    )
    previous_water = _is_water_class(terrain, start_x, start_y)
    for step in path[1:]:
        step_water = _is_water_class(terrain, step["x"], step["y"])
        if step_water != previous_water:
            return (step["x"], step["y"])
        previous_water = step_water
    return (target_x, target_y)


__all__ = [
    "FerryAwareTerrain",
    "SurfaceRouteTerrain",
    "clamp_move_target_at_surface_transition",
    "compose_decision_terrain",
    "is_riding_ferry",
]
