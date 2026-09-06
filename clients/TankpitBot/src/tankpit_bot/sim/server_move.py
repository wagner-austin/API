"""Move and teleport command handling for the simulator.

The two position-changing commands and the pickup-stock check they
share. Mixed into :class:`~tankpit_bot.sim.server.SimServer`, which
owns the state these annotate.
"""

from __future__ import annotations

from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.protocol.constants import (
    SUPERVISOR_ERROR_CANT_DO,
    SUPERVISOR_ERROR_EMPTY_CONTAINER,
)
from tankpit_bot.protocol.types import (
    BinaryMessage,
    SupervisorDict,
    SyncDict,
)
from tankpit_bot.sim.actions import process_teleport
from tankpit_bot.sim.client_session import ClientSession
from tankpit_bot.sim.commands import ClientCommandDict
from tankpit_bot.sim.equipment import resolve_equipment_pickup
from tankpit_bot.sim.fuel_deposit import resolve_fuel_deposit
from tankpit_bot.sim.fuel_pickup import resolve_fuel_pickup
from tankpit_bot.sim.movement import process_move
from tankpit_bot.sim.narrate import (
    narrate_equipment_pickup,
    narrate_fuel_deposit,
    narrate_fuel_pickup,
    narrate_move,
    narrate_teleport,
)
from tankpit_bot.sim.server_sessions import SimServerSessionsMixin
from tankpit_bot.sim.world import SimWorldDict


def _fuel_container_volume(world: SimWorldDict, x: int, y: int) -> int:
    """The fuel volume recorded at a tile before a command resolves."""
    for container in world["containers"]:
        if (container["x"], container["y"]) == (x, y):
            return container["volume"]
    return 0


class SimServerMoveMixin(SimServerSessionsMixin):
    """Move and teleport command handling for the simulator.

    The attributes below are DECLARATIONS, not assignments: the
    server's ``__init__`` remains their single owner.
    """

    world: SimWorldDict
    terrain: TerrainMapProtocol
    session: ClientSession

    def _process_move_command(
        self,
        tank_id: int,
        kind: str,
        command: ClientCommandDict,
        messages: list[BinaryMessage],
        ammo_changed: set[int],
        moved: set[int],
    ) -> None:
        """Route one move-family command (move / pickup clicks).

        The client's destination must lie inside its stored 0x5A
        window — the real router rejects any target outside it with
        0x52 code 0, at exactly the boundary column (measured
        2026-07-25, [[viewport-shift-protocol]]); the check precedes
        the container validation because the server never answers for
        a coordinate it does not consider actionable. A fuel-pickup
        click at a KNOWN container executes its walk and answers with
        the measured pickup choreography even when the container is
        drained (archive windows 2026-08-01: the walk echo, the
        duplicate remaining-0 records, then the code-4 close — the
        old pre-move refusal was a sim invention; only a click at a
        tile with NO container record short-circuits). Equipment
        clicks keep the presence pre-check. A successful walk does
        NOT re-emit 0x5A: with autoscroll OFF the window is static
        between teleports.

        Args:
            tank_id: The commanding tank.
            kind: The command kind.
            command: The queued command.
            messages: This tick's outgoing batch (appended).
            ammo_changed: Accumulator of tanks whose counts moved.
            moved: Accumulator of tanks that relocated this tick.
        """
        if self.click_leaves_own_window(tank_id, command["x"], command["y"]):
            messages.append(
                # Measured fields, 2026-09-02: code 0 carries
                # (reset_action=0, close_map=1) in 83 of the archive's
                # 86 windows and in 10 of 10 move windows. The sim sent
                # (1, 0) — both halves inverted. Nothing in the BOT
                # reads either field, which is why it survived; the
                # real client does ([[decode-coverage]]: reset_action
                # "reset to idle", close_map "close map view"), and a
                # 1:1 server is for real clients.
                SupervisorDict(
                    msg_type=0x52,
                    reset_action=0,
                    close_map=1,
                    error_code=SUPERVISOR_ERROR_CANT_DO,
                )
            )
            return
        if not self._pickup_target_stocked(kind, command["x"], command["y"]):
            if tank_id == self.session.client_id:
                messages.append(
                    SupervisorDict(
                        msg_type=0x52,
                        reset_action=1,
                        close_map=0,
                        error_code=SUPERVISOR_ERROR_EMPTY_CONTAINER,
                    )
                )
            return
        fuel_before = _fuel_container_volume(self.world, command["x"], command["y"])
        outcome = process_move(self.world, self.terrain, tank_id, command["x"], command["y"])
        if outcome["kind"] == "moved":
            moved.add(tank_id)
        choreographed = kind == "pickup_fuel" and outcome["kind"] == "moved"
        messages.extend(
            narrate_move(
                self.world,
                outcome,
                self.session.client_id,
                include_pickups=not choreographed,
            )
        )
        if choreographed:
            pickup = resolve_fuel_pickup(
                self.world,
                tank_id,
                command["x"],
                command["y"],
                volume_before=fuel_before,
                walked=outcome["path"] != "",
            )
            messages.extend(narrate_fuel_pickup(pickup, self.session.client_id))
        if outcome["kind"] == "moved":
            self._resolve_arrival_equipment(tank_id, kind, messages)
            if kind == "deposit_fuel":
                # The deposit resolves on ARRIVAL, after the walk that
                # carried the tank to the tile, and it draws no 0x3F —
                # both of the archive's two walked deposits end at the
                # 0x43 with no sync behind it, where a plain move of
                # the same distance draws one ([[fuel-system]]).
                deposit = resolve_fuel_deposit(
                    self.world, tank_id, command["x"], command["y"], command["amount"]
                )
                messages.extend(narrate_fuel_deposit(self.world, deposit, self.session.client_id))
                return
            if tank_id == self.session.client_id and outcome["path"] != "":
                # The 0x3F Sync trails a walk that actually relocated
                # the client — an own-tile click resolves as a "moved"
                # outcome with an EMPTY path and draws none. Archive
                # 2026-08-06: 1,277 of the 1,528 syncs follow a move
                # command as the most recent thing the client sent,
                # against ZERO after any of the 13,698 shoots and 14
                # after 11,247 map opens — the association is specific,
                # not ambient, and 1,277 against 1,703 move commands is
                # the gap the empty-path clicks fill. The JS handler is
                # a view resync (``vg`` -> ``Q(a)``), which is what a
                # completed walk needs and a standing still does not.
                messages.append(SyncDict(msg_type=0x3F))

    def _process_teleport_command(
        self,
        tank_id: int,
        command: ClientCommandDict,
        messages: list[BinaryMessage],
        ammo_changed: set[int],
        moved: set[int],
    ) -> None:
        """Route one teleport command through the towing gate and law 5.

        Teleport while towing a block is refused with the measured
        0x52 code 0 ("You can't do this") — three-for-three in the
        2026-07-20 capture, and it keeps ``(reset_action=1,
        close_map=1)`` rather than code 0's usual ``(0, 1)``: the
        2026-09-02 field sweep found exactly THREE ``(1, 1)`` code-0
        frames in the whole archive, all in teleport windows, which is
        the same three. A landed client hop is the ONE window
        recenter under autoscroll OFF ([[viewport-shift-protocol]])
        and resolves equipment on arrival. Wire order of a landed
        client hop (archive-measured 2026-08-01, 38%+31% of 7,176
        live teleports fit ``5A -> 3D -> landed [-> pickup]``): the
        RECENTERED 0x5A leads the batch, then the position statement,
        then the landed confirm — the response-shape differ caught
        the sim emitting the 0x5A last.

        Args:
            tank_id: The hopping tank.
            command: The queued command.
            messages: This tick's outgoing batch (appended).
            ammo_changed: Accumulator of tanks whose counts moved.
            moved: Accumulator of tanks that relocated this tick.
        """
        if self.world["tanks"][tank_id]["carrying"]:
            if tank_id == self.session.client_id:
                messages.append(
                    SupervisorDict(
                        msg_type=0x52,
                        reset_action=1,
                        close_map=1,
                        error_code=SUPERVISOR_ERROR_CANT_DO,
                    )
                )
            return
        hop = process_teleport(self.world, self.terrain, tank_id, command["x"], command["y"])
        landing = narrate_teleport(self.world, hop, self.session.client_id)
        if hop["kind"] != "landed":
            messages.extend(landing)
            return
        moved.add(tank_id)
        if tank_id == self.session.client_id:
            self.session.viewport.recenter()
            messages.append(self.session.viewport.build_update())
        messages.extend(landing)
        # A landing auto-picks FUEL ONLY. Equipment needs the explicit
        # pickup command, and the sim granting it here was an invented
        # law: across the archive's 10,619 teleport windows NOT ONE
        # carries a 0x67 EquipmentGain, while 5,205 of the 5,409 gains
        # follow an explicit pickup_equipment ([[capture-differ]], the
        # suspected-invented-law row, settled 2026-09-01). The fuel
        # auto-pick is unaffected — it rides ``process_teleport``'s own
        # ``pickups`` and the duplicate-record law that narrates them.

    def _resolve_arrival_equipment(
        self,
        tank_id: int,
        kind: str,
        messages: list[BinaryMessage],
    ) -> None:
        """Resolve an equipment container under an arriving tank.

        Both arrival paths — a walk and a teleport landing — resolve
        the same way, so the resolve-then-narrate pair lives here once
        rather than at each call site.

        Args:
            tank_id: The arriving tank.
            kind: The command kind that caused the arrival.
            messages: This tick's outgoing batch (appended).
        """
        grant = resolve_equipment_pickup(self.world, tank_id)
        if grant is None:
            return
        messages.extend(
            narrate_equipment_pickup(self.world, grant, tank_id, kind, self.session.client_id)
        )

    def _pickup_target_stocked(self, kind: str, x: int, y: int) -> bool:
        """Validate a pickup click's destination before any movement.

        A fuel click needs a container RECORD at the tile — even a
        drained one: the archive shows the walk executing and the
        remaining-0 choreography answering for empty-but-known
        containers (2026-08-01); only a click at bare ground draws
        the moveless code-4 refusal the production belief-removal
        consumes. Equipment clicks keep the presence check. Plain
        moves are never validated this way.

        Args:
            kind: The command kind.
            x: Clicked tile X.
            y: Clicked tile Y.

        Returns:
            True when the command may proceed to the move law.
        """
        if kind == "pickup_fuel":
            return any((c["x"], c["y"]) == (x, y) for c in self.world["containers"])
        if kind == "pickup_equipment":
            return any((e["x"], e["y"]) == (x, y) for e in self.world["equipment"])
        return True
