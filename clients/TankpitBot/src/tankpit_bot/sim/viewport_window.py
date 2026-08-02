"""The client's viewport-window model — one concern, one owner.

The visible viewport is a 16x16 window and 0x5A ViewportUpdate is the
ONLY message that sets it ([[viewport-shift-protocol]]). Under the
bot's operating mode (autoscroll OFF) the window is STATIC between
teleports: a teleport landing recenters it at ``(x-8, y-8)``
(map-clamped) and walking never moves it — measured live 2026-07-25
as a controlled OFF/ON pair. The tracker owns the stored window, the
dynamic-terrain patch memory, and the viewport-membership set that
drives 0x58 exits / 0x3D entries.
"""

from __future__ import annotations

from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.protocol.types import (
    BinaryMessage,
    TankRemoveDict,
    ViewportEntityDict,
    ViewportUpdateDict,
)
from tankpit_bot.sim.actions import VIEWPORT_RADIUS
from tankpit_bot.sim.blocks import block_tile_value
from tankpit_bot.sim.wire_statements import position_statement
from tankpit_bot.sim.world import SimWorldDict

VIEWPORT_SPAN = 16
_MAP_SPAN = 256

# Wire terrain vocabulary shared by 0x42/0x4A/0x5A ([[movable-blocks]]):
# ferries patch as 5; a vacated ferry tile reverts by patching 0
# (ground), which the client's composition resolves back to the
# static map value (water under a ferry).
_WIRE_TERRAIN_FERRY = 5
_WIRE_TERRAIN_REVERT = 0
_OVERLAY_NO_MINE = 8


class ViewportTracker:
    """Owns the client's stored window, patch memory, and visibility.

    ``removed_at`` maps a tank id to the tick its living exit was
    announced (0x58) — the law-4 reroute clock the shot resolver
    reads. ``visible`` is the current viewport-membership set.
    """

    def __init__(
        self,
        world: SimWorldDict,
        terrain: TerrainMapProtocol,
        client_id: int,
    ) -> None:
        """Bind the tracker and compute the join-time window.

        Viewport membership is computed at construction: other living
        tanks inside the client's window are visible from the first
        tick; later transitions emit 0x58 TankRemove on exit and a
        0x3D position statement on re-entry.

        Args:
            world: Simulated world (read, never owned).
            terrain: Static terrain for the world's field.
            client_id: The connected client's tank id.
        """
        self._world = world
        self._terrain = terrain
        self._client_id = client_id
        self.removed_at: dict[int, int] = {}
        self._patched_dynamic_tiles: dict[tuple[int, int], int] = {}
        self.window: tuple[int, int] = self._centered_window()
        self.visible: set[int] = {
            tank_id
            for tank_id, tank in world["tanks"].items()
            if tank_id != client_id and tank["alive"] and self.in_viewport(tank_id)
        }

    def _centered_window(self) -> tuple[int, int]:
        """Compute the map-clamped window origin centered on the client."""
        client = self._world["tanks"][self._client_id]
        left = min(max(client["x"] - VIEWPORT_RADIUS, 0), _MAP_SPAN - VIEWPORT_SPAN)
        top = min(max(client["y"] - VIEWPORT_RADIUS, 0), _MAP_SPAN - VIEWPORT_SPAN)
        return left, top

    def recenter(self) -> None:
        """Recenter the stored window on the client.

        A teleport landing is the ONE window recenter under autoscroll
        OFF ([[viewport-shift-protocol]]).
        """
        self.window = self._centered_window()

    def apply_scope_shift(self, direction: int) -> None:
        """Shift the stored window per the measured Rb anchor law.

        Wire-measured 2026-08-01 (capture sniff-20260710-202821, all 8
        "Extend view" events fit exactly, zero free parameters): the
        server does NOT stride the window — it ANCHORS it to the tank
        so the full 16x16 extends in the requested compass direction.
        An eastward component pins ``left = tank_x`` (tank on the west
        edge), a westward one ``left = tank_x - 15``; south/north pin
        ``top`` the same way; an axis the direction does not name
        keeps its current origin (E kept top three times; N kept
        left). Map-clamped like every window origin.

        Direction 8 is Scope Center (user-confirmed 2026-08-01):
        recenter on the tank, the same window a teleport landing
        produces.

        Args:
            direction: Compass byte, clockwise from north (0=N..7=NW),
                or 8 for center. Unknown bytes shift nothing but still
                confirm with the 0x5A the server always answers.
        """
        if direction == 8:
            self.window = self._centered_window()
            return
        client = self._world["tanks"][self._client_id]
        left, top = self.window
        if direction in (1, 2, 3):  # NE, E, SE
            left = client["x"]
        elif direction in (5, 6, 7):  # SW, W, NW
            left = client["x"] - (VIEWPORT_SPAN - 1)
        if direction in (3, 4, 5):  # SE, S, SW
            top = client["y"]
        elif direction in (7, 0, 1):  # NW, N, NE
            top = client["y"] - (VIEWPORT_SPAN - 1)
        self.window = (
            min(max(left, 0), _MAP_SPAN - VIEWPORT_SPAN),
            min(max(top, 0), _MAP_SPAN - VIEWPORT_SPAN),
        )

    def in_window(self, x: int, y: int) -> bool:
        """Report whether a tile lies inside the client's current window."""
        left, top = self.window
        return left <= x < left + VIEWPORT_SPAN and top <= y < top + VIEWPORT_SPAN

    def in_viewport(self, tank_id: int) -> bool:
        """Report whether a tank sits inside the client's viewport.

        Args:
            tank_id: The tank to test.

        Returns:
            True when the tank is inside the client's stored 16x16
            window (the last 0x5A statement — static between
            teleports under autoscroll OFF).
        """
        tank = self._world["tanks"][tank_id]
        return self.in_window(tank["x"], tank["y"])

    def build_update(self) -> ViewportUpdateDict:
        """Build the client's 0x5A viewport statement.

        The origin is the client's STORED window — set at join and on
        every teleport landing (centered, map-clamped), and never by
        walking. Entities enumerate the VISIBLE dynamic-terrain layer
        only: in-window ferry tiles (wire terrain 5,
        [[ferry-mechanics]]) and resting movable blocks (1/2/3 by
        context, [[movable-blocks]]) plus explicit reverts (wire
        terrain 0) for previously patched tiles the entity has left.
        Hidden-layer entities (containers, mines) stay absent by
        design: they reveal only by radar, and the production
        reset-then-apply sweep explicitly SPARES radar-sourced
        entries on silent tiles.

        Returns:
            The viewport update for the client's stored window.
        """
        left, top = self.window

        def in_patch(x: int, y: int) -> bool:
            """Report whether a tile sits inside the 18x18 patch grid.

            The wire patch grid carries a one-tile border around the
            16x16 window: ``col = x - left + 1`` (production
            ``viewport_patch_world_coords`` subtracts the border).
            """
            return left - 1 <= x < left + VIEWPORT_SPAN + 1 and (
                top - 1 <= y < top + VIEWPORT_SPAN + 1
            )

        def entity(x: int, y: int, terrain_type: int) -> ViewportEntityDict:
            """Build one patch entity for an absolute tile."""
            return ViewportEntityDict(
                col=x - left + 1,
                row=y - top + 1,
                cache_value=0,
                overlay_value=_OVERLAY_NO_MINE,
                terrain_type=terrain_type,
            )

        current: dict[tuple[int, int], int] = {
            (ferry["x"], ferry["y"]): _WIRE_TERRAIN_FERRY for ferry in self._world["ferries"]
        }
        for block in self._world["blocks"]:
            tile = (block["x"], block["y"])
            current[tile] = block_tile_value(self._world, self._terrain, tile[0], tile[1])
        entities: list[ViewportEntityDict] = []
        for x, y in sorted(set(self._patched_dynamic_tiles) - set(current)):
            if in_patch(x, y):
                entities.append(entity(x, y, _WIRE_TERRAIN_REVERT))
                del self._patched_dynamic_tiles[(x, y)]
        for (x, y), value in sorted(current.items()):
            if in_patch(x, y) and self._patched_dynamic_tiles.get((x, y)) != value:
                entities.append(entity(x, y, value))
                self._patched_dynamic_tiles[(x, y)] = value
        # The 0x5A skip-RLE cursor only walks FORWARD: entities must
        # be in ascending patch-linear order or the encoder's delta
        # goes negative. A ferry riding WEST surfaces this — its
        # revert (the vacated tile) sits later in the walk than its
        # fresh patch one tile earlier.
        entities.sort(key=lambda item: (item["row"], item["col"]))
        return ViewportUpdateDict(
            msg_type=0x5A, viewport_left=left, viewport_top=top, entities=entities
        )

    def emit_transitions(self, messages: list[BinaryMessage]) -> None:
        """Diff viewport membership after this tick's relocations.

        A living tank leaving the client's viewport emits 0x58
        TankRemove and starts the law-4 reroute clock; one entering
        emits a 0x3D position statement (positions are
        viewport-scoped on the real wire). Deactivated tanks simply
        drop from the visible set — their exit is announced by 0x41,
        not 0x58.

        Args:
            messages: This tick's outgoing batch (appended).
        """
        for tank_id in sorted(self._world["tanks"]):
            tank = self._world["tanks"][tank_id]
            if tank_id == self._client_id:
                continue
            if not tank["alive"]:
                self.visible.discard(tank_id)
                continue
            inside = self.in_viewport(tank_id)
            if inside and tank_id not in self.visible:
                self.visible.add(tank_id)
                self.removed_at.pop(tank_id, None)
                messages.append(position_statement(self._world, tank_id))
            elif not inside and tank_id in self.visible:
                self.visible.discard(tank_id)
                self.removed_at[tank_id] = self._world["tick"]
                messages.append(TankRemoveDict(msg_type=0x58, tank_id=tank_id))


__all__ = [
    "VIEWPORT_SPAN",
    "ViewportTracker",
]
