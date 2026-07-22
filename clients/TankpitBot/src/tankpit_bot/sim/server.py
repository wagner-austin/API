"""Law 1 — the global command queue and the 2-second tick processor.

Commands queue in arrival order and process together at tick
boundaries; every wire message a tick produces flushes as one batch
(the measured sync cadence, wiki log 2026-07-21). Shooter firing
costs bill one tick AFTER the shot (measured charge latency); victim
damage bills instantly inside :mod:`tankpit_bot.sim.combat`.

The processor emits decoded ``BinaryMessage`` dicts; the transport
layer (build step c) turns them into wire bytes via
``protocol.encoders.encode_envelope_body``.
"""

from __future__ import annotations

from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.container.types import (
    ContainerPickupDict,
    ContainerPickupRecordDict,
    MineDetonationDict,
    MinePlacementDict,
    TeleportLandedDict,
)
from tankpit_bot.protocol.commands import TICK_RATE_MS
from tankpit_bot.protocol.constants import (
    SUPERVISOR_ERROR_CANT_GO,
    SUPERVISOR_ERROR_EMPTY_CONTAINER,
    SUPERVISOR_ERROR_INSUFFICIENT_FUEL,
    SUPERVISOR_ERROR_INVENTORY_FULL,
)
from tankpit_bot.protocol.types import (
    BinaryMessage,
    DeactivationDict,
    EquipmentGainDict,
    EquipmentToggleDict,
    FuelGainDict,
    InventoryDict,
    MovementDict,
    MovementResponseDict,
    RadarResultDict,
    RadarScanResultDict,
    ShootEventDict,
    SupervisorDict,
    TankInfoDict,
    TankRemoveDict,
    TankStatusSyncDict,
    ViewportUpdateDict,
)
from tankpit_bot.sim.actions import (
    MINE_PRESS_FUEL_COST,
    RADAR_FUEL_COST,
    VIEWPORT_RADIUS,
    build_map_data,
    process_mine_press,
    process_radar,
    process_teleport,
)
from tankpit_bot.sim.combat import process_shot
from tankpit_bot.sim.commands import ClientCommandDict, SimError
from tankpit_bot.sim.equipment import resolve_equipment_pickup
from tankpit_bot.sim.movement import MoveOutcomeDict, process_move
from tankpit_bot.sim.world import SimWorldDict

# TeleportLanded's 1-byte body observed in production captures.
_TELEPORT_LANDED_SUBTYPE = 0x0C

# The visible viewport is a 16x16 window (0x5A ViewportUpdate is the
# ONLY message that sets it — [[viewport-shift-protocol]]). The sim
# centers it on the client; the real window scrolls, so the center is
# a documented approximation.
_VIEWPORT_SPAN = 16
_MAP_SPAN = 256

_SUPPORTED_KINDS = frozenset(
    {
        "move",
        "shoot",
        "teleport",
        "radar",
        "mine",
        "map_open",
        "pickup_fuel",
        "pickup_equipment",
        "toggle_equipment",
    }
)
_MOVE_KINDS = frozenset({"move", "pickup_fuel", "pickup_equipment"})


def _movement_echo(world: SimWorldDict, outcome: MoveOutcomeDict) -> MovementDict:
    """Build the 0x47 echo for one processed move.

    Args:
        world: Simulated world (post-move).
        outcome: The move's outcome.

    Returns:
        The movement echo carrying the full routed path.
    """
    tank = world["tanks"][outcome["tank_id"]]
    path = outcome["path"]
    x, y = tank["x"], tank["y"]
    return MovementDict(
        msg_type=0x47,
        tank_id=tank["tank_id"],
        start_x=outcome["start_x"],
        start_y=outcome["start_y"],
        direction=0,
        damage_state=tank["damage_state"],
        lb_score=0,
        rank=tank["rank"],
        flag=1,
        is_carrying=False,
        waypoints=[(x, y)] if path else [],
        path_tiles=len(path),
        path=path,
    )


def _status_sync(tank_id: int, world: SimWorldDict, include_fuel: bool) -> TankStatusSyncDict:
    """Build a 0x2E status sync for one tank.

    The real wire carries the fuel field ONLY on the recipient's own
    tank (per-recipient long form); other tanks sync short-form. The
    production dispatcher treats any fuel-bearing 0x2E as self fuel,
    so emitting long-form for a victim would corrupt the client's own
    belief — caught by the step-(c) wire integration.

    Args:
        tank_id: The synced tank.
        world: Simulated world.
        include_fuel: True only for the connected client's tank.

    Returns:
        The status sync (long form with fuel, or short form).
    """
    tank = world["tanks"][tank_id]
    return TankStatusSyncDict(
        msg_type=0x2E,
        subtype=tank["team"],
        tank_id=tank_id,
        damage_state=tank["damage_state"],
        rank=tank["rank"],
        lb_score=0,
        promo_state=0,
        fuel=tank["fuel"] if include_fuel else None,
    )


class SimServer:
    """The fake server: one world, one command queue, one client.

    ``client_id`` is the tank whose connection this server speaks for
    — it receives its own fuel syncs and inventory snapshots; other
    tanks' commands are queued by the opponent policies directly.
    """

    def __init__(self, world: SimWorldDict, terrain: TerrainMapProtocol, client_id: int) -> None:
        """Bind the server to a world, its terrain, and the client tank.

        Viewport membership is computed at construction: other living
        tanks within :data:`VIEWPORT_RADIUS` of the client are visible
        from the first tick; later transitions emit 0x58 TankRemove on
        exit (starting the law-4 reroute clock) and a 0x3D position
        statement on re-entry.

        Args:
            world: Simulated world (owned and mutated by the server).
            terrain: Static terrain for the world's field.
            client_id: The connected client's tank id.
        """
        self.world = world
        self.terrain = terrain
        self.client_id = client_id
        self._queue: list[tuple[int, ClientCommandDict]] = []
        self._pending_debits: list[tuple[int, int]] = []
        self._removed_at: dict[int, int] = {}
        self._visible: set[int] = {
            tank_id
            for tank_id, tank in world["tanks"].items()
            if tank_id != client_id and tank["alive"] and self._in_viewport(tank_id)
        }

    def _viewport_update(self) -> ViewportUpdateDict:
        """Build the client's 0x5A viewport statement.

        The origin centers the 16x16 window on the client (clamped at
        the map edge). Entities are empty by design: every sim entity
        is hidden-layer and radar-owned (containers and mines reveal
        only by scan, [[radar-mechanics]]), and the production
        reset-then-apply sweep explicitly SPARES radar-sourced entries
        on silent tiles — an origin-only patch is exactly the truthful
        statement "the window moved; the visible layer shows nothing".

        Returns:
            The viewport update for the client's current position.
        """
        client = self.world["tanks"][self.client_id]
        left = min(max(client["x"] - VIEWPORT_RADIUS, 0), _MAP_SPAN - _VIEWPORT_SPAN)
        top = min(max(client["y"] - VIEWPORT_RADIUS, 0), _MAP_SPAN - _VIEWPORT_SPAN)
        return ViewportUpdateDict(msg_type=0x5A, viewport_left=left, viewport_top=top, entities=[])

    def _in_viewport(self, tank_id: int) -> bool:
        """Report whether a tank sits inside the client's viewport.

        Args:
            tank_id: The tank to test.

        Returns:
            True when the tank is within the Chebyshev viewport radius
            of the client's current position.
        """
        client = self.world["tanks"][self.client_id]
        tank = self.world["tanks"][tank_id]
        return max(abs(tank["x"] - client["x"]), abs(tank["y"] - client["y"])) <= VIEWPORT_RADIUS

    def handshake(self) -> list[BinaryMessage]:
        """Build the session-start burst the client receives on join.

        Mirrors the real server's join choreography (and the scenario
        harness's ``place_self``): the client's OWN identity (0x21)
        first — the archive convention the audit validators rely on is
        that the first TankInfo of a session names the player's own
        tank (``validate.wire_timeline``) — then own position (0x3D),
        absolute fuel (0x44), and inventory (0x49), then an identity
        (0x21) for every other living tank and a position statement
        (0x3D) only for those inside the client's viewport — tank
        positions are viewport-scoped on the real wire; out-of-view
        tanks surface as map blips ([[map-data-decode]]) until they
        enter the viewport.

        Returns:
            The decoded messages of the join burst, in order.
        """
        client = self.world["tanks"][self.client_id]
        messages: list[BinaryMessage] = [
            TankInfoDict(
                msg_type=0x21,
                tank_id=self.client_id,
                team=client["team"],
                decoration_state=bytes(4),
                persistent_tank_id=0,
                name=f"sim-{self.client_id}",
            ),
            self._position_statement(self.client_id),
            self._viewport_update(),
            FuelGainDict(msg_type=0x44, fuel_total=client["fuel"], is_free=False, flag=1),
            InventoryDict(
                msg_type=0x49,
                show=True,
                alternate=False,
                counts=list(client["counts"]),
                enabled=list(client["enabled"]),
            ),
        ]
        for tank_id in sorted(self.world["tanks"]):
            tank = self.world["tanks"][tank_id]
            if tank_id == self.client_id or not tank["alive"]:
                continue
            messages.append(
                TankInfoDict(
                    msg_type=0x21,
                    tank_id=tank_id,
                    team=tank["team"],
                    decoration_state=bytes(4),
                    persistent_tank_id=0,
                    name=f"sim-{tank_id}",
                )
            )
            if tank_id in self._visible:
                messages.append(self._position_statement(tank_id))
        return messages

    def _position_statement(self, tank_id: int) -> MovementResponseDict:
        """Build a 0x3D position statement for one tank.

        Args:
            tank_id: The positioned tank.

        Returns:
            The movement response carrying the tank's current tile.
        """
        tank = self.world["tanks"][tank_id]
        return MovementResponseDict(
            msg_type=0x3D,
            team=tank["team"],
            tank_id=tank_id,
            x=tank["x"],
            y=tank["y"],
            direction=0,
            damage_state=tank["damage_state"],
            rank=tank["rank"],
            lb_score=0,
            carrying=0,
        )

    def queue_command(self, tank_id: int, command: ClientCommandDict) -> None:
        """Queue one command for the next tick.

        Args:
            tank_id: The commanding tank.
            command: Decoded client command.

        A DEAD CLIENT's commands drop silently: the real connection
        survives deactivation and the server simply ignores a
        corpse's clicks (first real-terrain CLI run, 2026-07-22: the
        enemy killed the bot and the production loop kept clicking —
        real behavior, not a harness bug). Dead or unknown NON-client
        tanks still raise, because only the harness can queue those.

        Raises:
            SimError: For unsupported command kinds, unknown tanks, or
                dead harness-driven tanks.
        """
        if command["kind"] not in _SUPPORTED_KINDS:
            raise SimError(
                f"sim step b handles move/shoot only; got {command['kind']!r} "
                "(laws 4-8 land in build step d)"
            )
        tank = self.world["tanks"].get(tank_id)
        if tank is None:
            raise SimError(f"no tank {tank_id} to command")
        if not tank["alive"]:
            if tank_id == self.client_id:
                return
            raise SimError(f"no living tank {tank_id} to command")
        self._queue.append((tank_id, command))

    def _apply_pending_debits(self) -> None:
        """Bill last tick's firing costs (measured charge latency)."""
        for tank_id, debit in self._pending_debits:
            tank = self.world["tanks"][tank_id]
            tank["fuel"] = max(0, tank["fuel"] - debit)
        self._pending_debits = []

    def _emit_move(self, outcome: MoveOutcomeDict, messages: list[BinaryMessage]) -> None:
        """Emit the wire consequences of one processed move.

        Supervisor rejections (0x52) are PER-CONNECTION on the real
        wire — the client only ever sees its own. Another tank's
        rejected command must not leak into the client's stream
        (same per-recipient discipline as the fuel-sync fix).

        Args:
            outcome: The move's outcome.
            messages: This tick's outgoing batch (appended).
        """
        if outcome["kind"] == "cant_go":
            if outcome["tank_id"] == self.client_id:
                messages.append(
                    SupervisorDict(
                        msg_type=0x52,
                        reset_action=1,
                        close_map=0,
                        error_code=SUPERVISOR_ERROR_CANT_GO,
                    )
                )
            return
        if outcome["kind"] == "insufficient_fuel":
            if outcome["tank_id"] == self.client_id:
                messages.append(
                    SupervisorDict(
                        msg_type=0x52,
                        reset_action=1,
                        close_map=0,
                        error_code=SUPERVISOR_ERROR_INSUFFICIENT_FUEL,
                    )
                )
            return
        messages.append(_movement_echo(self.world, outcome))
        for x, y in outcome["mine_positions"]:
            messages.append(MineDetonationDict(msg_type=0x45, positions=[(x, y)]))
        if outcome["pickups"]:
            messages.append(
                ContainerPickupDict(
                    msg_type="container_pickup",
                    pickups=tuple(
                        ContainerPickupRecordDict(
                            x=p["x"], y=p["y"], remaining_volume=p["remaining_volume"]
                        )
                        for p in outcome["pickups"]
                    ),
                )
            )

    def _emit_shot(
        self,
        shooter_team: int,
        outcome_messages: list[BinaryMessage],
        ammo_changed: set[int],
        tank_id: int,
        command: ClientCommandDict,
        moved: frozenset[int],
    ) -> None:
        """Process one shoot command and emit its wire consequences.

        Args:
            shooter_team: The shooter's team (for the 0x53 echo).
            outcome_messages: This tick's outgoing batch (appended).
            ammo_changed: Accumulator of tanks whose counts moved.
            tank_id: The firing tank.
            command: The shoot command.
            moved: Tanks that moved earlier this tick.
        """
        target_id = command["target_id"]
        removed_tick = self._removed_at.get(target_id)
        departed_age_ms = (
            None if removed_tick is None else (self.world["tick"] - removed_tick) * TICK_RATE_MS
        )
        outcome = process_shot(
            self.world,
            self.terrain,
            tank_id,
            command["x"],
            command["y"],
            moved,
            target_id,
            departed_age_ms,
        )
        outcome_messages.append(
            ShootEventDict(
                msg_type=0x53,
                team=shooter_team,
                shooter_id=tank_id,
                source_x=outcome["source_x"],
                source_y=outcome["source_y"],
                target_x=outcome["impact_x"],
                target_y=outcome["impact_y"],
                aim_x=outcome["aim_x"],
                aim_y=outcome["aim_y"],
                weapon=outcome["weapon"],
            )
        )
        self._pending_debits.append((tank_id, outcome["shooter_debit"]))
        if outcome["ammo_slot"] is not None:
            ammo_changed.add(tank_id)
        if outcome["shields_consumed"] > 0 and outcome["victim_id"] is not None:
            ammo_changed.add(outcome["victim_id"])
        for packet in outcome["mine_cascade"]:
            outcome_messages.append(MineDetonationDict(msg_type=0x45, positions=packet))
        if outcome["victim_deactivated"] and outcome["victim_id"] is not None:
            outcome_messages.append(
                DeactivationDict(
                    msg_type=0x41,
                    status=1,
                    victim_id=outcome["victim_id"],
                    promo_eligible=False,
                    killer_id=tank_id,
                    is_mine_kill=False,
                )
            )

    def _emit_teleport(
        self,
        tank_id: int,
        command: ClientCommandDict,
        messages: list[BinaryMessage],
    ) -> bool:
        """Process one teleport command and emit its wire consequences.

        Args:
            tank_id: The teleporting tank.
            command: The teleport command.
            messages: This tick's outgoing batch (appended).

        Returns:
            True when the hop landed (the tank counts as a mover).
        """
        outcome = process_teleport(self.world, self.terrain, tank_id, command["x"], command["y"])
        if outcome["kind"] != "landed":
            code = (
                SUPERVISOR_ERROR_INSUFFICIENT_FUEL
                if outcome["kind"] == "insufficient_fuel"
                else SUPERVISOR_ERROR_CANT_GO
            )
            if tank_id == self.client_id:
                messages.append(
                    SupervisorDict(msg_type=0x52, reset_action=1, close_map=1, error_code=code)
                )
            return False
        messages.append(
            TeleportLandedDict(msg_type="teleport_landed", subtype=_TELEPORT_LANDED_SUBTYPE)
        )
        messages.append(self._position_statement(tank_id))
        if outcome["pickups"]:
            messages.append(
                ContainerPickupDict(
                    msg_type="container_pickup",
                    pickups=tuple(
                        ContainerPickupRecordDict(
                            x=p["x"], y=p["y"], remaining_volume=p["remaining_volume"]
                        )
                        for p in outcome["pickups"]
                    ),
                )
            )
        return True

    def _emit_radar(
        self,
        tank_id: int,
        messages: list[BinaryMessage],
        ammo_changed: set[int],
    ) -> None:
        """Process one radar command and emit its wire consequences.

        Args:
            tank_id: The scanning tank.
            messages: This tick's outgoing batch (appended).
            ammo_changed: Accumulator of tanks whose counts moved.
        """
        outcome = process_radar(self.world, tank_id)
        tank = self.world["tanks"][tank_id]
        tank["fuel"] = max(0, tank["fuel"] - RADAR_FUEL_COST)
        if outcome["consumed_extra"]:
            ammo_changed.add(tank_id)
        messages.append(
            RadarScanResultDict(
                msg_type=0x4F,
                containers=outcome["containers"],
                mines=outcome["mines"],
                mine_clears=[],
            )
        )
        messages.append(
            RadarResultDict(msg_type=0x46, detection_type=0, found=outcome["enemy_found"])
        )

    def _emit_mine_press(self, tank_id: int, messages: list[BinaryMessage]) -> None:
        """Process one mine press and emit its wire consequences.

        Args:
            tank_id: The placing tank.
            messages: This tick's outgoing batch (appended).
        """
        outcome = process_mine_press(self.world, self.terrain, tank_id)
        tank = self.world["tanks"][tank_id]
        tank["fuel"] = max(0, tank["fuel"] - MINE_PRESS_FUEL_COST)
        if outcome["placed"]:
            messages.append(
                MinePlacementDict(
                    msg_type=0x4B,
                    mine_type=outcome["mine_type"],
                    tank_id=tank_id,
                    positions=outcome["placed"],
                )
            )
        if outcome["detonated"]:
            messages.append(MineDetonationDict(msg_type=0x45, positions=outcome["detonated"]))

    def _process_command(
        self,
        tank_id: int,
        command: ClientCommandDict,
        messages: list[BinaryMessage],
        ammo_changed: set[int],
        moved: set[int],
    ) -> None:
        """Route one queued command to its law processor.

        Args:
            tank_id: The commanding tank.
            command: The queued command.
            messages: This tick's outgoing batch (appended).
            ammo_changed: Accumulator of tanks whose counts moved.
            moved: Accumulator of tanks that relocated this tick.
        """
        kind = command["kind"]
        if kind in _MOVE_KINDS:
            self._process_move_command(tank_id, kind, command, messages, ammo_changed, moved)
            return
        if kind == "shoot":
            self._emit_shot(
                self.world["tanks"][tank_id]["team"],
                messages,
                ammo_changed,
                tank_id,
                command,
                frozenset(moved),
            )
            return
        if kind == "teleport":
            if self._emit_teleport(tank_id, command, messages):
                moved.add(tank_id)
                if tank_id == self.client_id:
                    messages.append(self._viewport_update())
                self._emit_equipment_pickup(tank_id, kind, messages, ammo_changed)
            return
        if kind == "radar":
            self._emit_radar(tank_id, messages, ammo_changed)
            return
        if kind == "mine":
            self._emit_mine_press(tank_id, messages)
            return
        if kind == "toggle_equipment":
            self._emit_equipment_toggle(tank_id, command["slot"], messages)
            return
        messages.append(build_map_data(self.world))

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

        Pickup clicks validate their destination first (empty-container
        rejection); a successful relocation updates the client's
        viewport and resolves any equipment container on arrival.

        Args:
            tank_id: The commanding tank.
            kind: The command kind.
            command: The queued command.
            messages: This tick's outgoing batch (appended).
            ammo_changed: Accumulator of tanks whose counts moved.
            moved: Accumulator of tanks that relocated this tick.
        """
        if not self._pickup_target_stocked(kind, command["x"], command["y"]):
            if tank_id == self.client_id:
                messages.append(
                    SupervisorDict(
                        msg_type=0x52,
                        reset_action=1,
                        close_map=0,
                        error_code=SUPERVISOR_ERROR_EMPTY_CONTAINER,
                    )
                )
            return
        outcome = process_move(self.world, self.terrain, tank_id, command["x"], command["y"])
        if outcome["kind"] == "moved":
            moved.add(tank_id)
        self._emit_move(outcome, messages)
        if outcome["kind"] == "moved":
            if tank_id == self.client_id:
                messages.append(self._viewport_update())
            self._emit_equipment_pickup(tank_id, kind, messages, ammo_changed)

    def _pickup_target_stocked(self, kind: str, x: int, y: int) -> bool:
        """Validate a pickup click's destination before any movement.

        The real server answers a pickup at a consumed/absent
        container with 0x52 error 4 ("empty container") and no
        movement — the client removes its stale belief on that signal
        (``tick_loop_actions``: code=4 -> belief removed). Plain moves
        are never validated this way.

        Args:
            kind: The command kind.
            x: Clicked tile X.
            y: Clicked tile Y.

        Returns:
            True when the command may proceed to the move law.
        """
        if kind == "pickup_fuel":
            return any(
                (c["x"], c["y"]) == (x, y) and c["volume"] > 0 for c in self.world["containers"]
            )
        if kind == "pickup_equipment":
            return any((e["x"], e["y"]) == (x, y) for e in self.world["equipment"])
        return True

    def _emit_equipment_pickup(
        self,
        tank_id: int,
        kind: str,
        messages: list[BinaryMessage],
        ammo_changed: set[int],
    ) -> None:
        """Resolve an equipment container under an arriving tank.

        A grant emits the 0x67 gained array (the following 0x49 rides
        the ``ammo_changed`` snapshot — the archive shows every 0x67
        immediately followed by its inventory sync). A full-inventory
        attempt on an explicit ``pickup_equipment`` click answers with
        the measured 0x52 error 7 and leaves the container; incidental
        arrivals at full inventory are silent. Both 0x67 and the 0x52
        are PER-RECIPIENT: production treats any 0x67 as a SELF gain,
        so another tank's grant resolves silently server-side.

        Args:
            tank_id: The arriving tank.
            kind: The command kind that caused the arrival.
            messages: This tick's outgoing batch (appended).
            ammo_changed: Accumulator of tanks whose counts moved.
        """
        grant = resolve_equipment_pickup(self.world, tank_id)
        if grant is None:
            return
        if grant["kind"] == "granted":
            if tank_id == self.client_id:
                messages.append(
                    EquipmentGainDict(msg_type=0x67, show_message=True, gained=grant["gained"])
                )
            ammo_changed.add(tank_id)
            return
        if kind == "pickup_equipment" and tank_id == self.client_id:
            messages.append(
                SupervisorDict(
                    msg_type=0x52,
                    reset_action=1,
                    close_map=0,
                    error_code=SUPERVISOR_ERROR_INVENTORY_FULL,
                )
            )

    def _emit_equipment_toggle(
        self, tank_id: int, slot: int, messages: list[BinaryMessage]
    ) -> None:
        """Flip one equipment slot and answer with the 0x74 state.

        The toggle is free and server-authoritative: the response
        carries all five enabled flags (the wire's documented
        ``t + 5 bytes`` shape).

        Args:
            tank_id: The toggling tank.
            slot: Equipment slot, 1-5 (out-of-range presses are the
                client's problem and are ignored like the real UI).
            messages: This tick's outgoing batch (appended).
        """
        tank = self.world["tanks"][tank_id]
        if 1 <= slot <= len(tank["enabled"]):
            tank["enabled"][slot - 1] = not tank["enabled"][slot - 1]
        messages.append(EquipmentToggleDict(msg_type=0x74, enabled=list(tank["enabled"])))

    def _emit_viewport_transitions(self, messages: list[BinaryMessage]) -> None:
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
        for tank_id in sorted(self.world["tanks"]):
            tank = self.world["tanks"][tank_id]
            if tank_id == self.client_id:
                continue
            if not tank["alive"]:
                self._visible.discard(tank_id)
                continue
            inside = self._in_viewport(tank_id)
            if inside and tank_id not in self._visible:
                self._visible.add(tank_id)
                self._removed_at.pop(tank_id, None)
                messages.append(self._position_statement(tank_id))
            elif not inside and tank_id in self._visible:
                self._visible.discard(tank_id)
                self._removed_at[tank_id] = self.world["tick"]
                messages.append(TankRemoveDict(msg_type=0x58, tank_id=tank_id))

    def advance_tick(self) -> list[BinaryMessage]:
        """Process the queue and return this tick's outgoing batch.

        Returns:
            The decoded messages the client receives this tick, in
            emission order: last tick's deferred debits are billed
            first, then each queued command's consequences, then
            viewport transitions (0x58 exits that start the law-4
            reroute clock, 0x3D entries), then one status sync per
            LIVING tank — the measured broadcast cadence is every ~2 s
            for every active tank regardless of activity
            ([[tank-freshness-model]]), and the Phase 3 fuel book
            depends on exactly those quiet zero-delta readings to
            close its accounting blocks — and an inventory snapshot
            when the client's counts changed.
        """
        self.world["tick"] += 1
        messages: list[BinaryMessage] = []
        ammo_changed: set[int] = set()
        self._apply_pending_debits()
        moved: set[int] = set()
        for tank_id, command in self._queue:
            if not self.world["tanks"][tank_id]["alive"]:
                continue
            self._process_command(tank_id, command, messages, ammo_changed, moved)
        self._queue = []
        self._emit_viewport_transitions(messages)
        for tank_id in sorted(self.world["tanks"]):
            if self.world["tanks"][tank_id]["alive"]:
                messages.append(_status_sync(tank_id, self.world, tank_id == self.client_id))
        if self.client_id in ammo_changed:
            client = self.world["tanks"][self.client_id]
            messages.append(
                InventoryDict(
                    msg_type=0x49,
                    show=False,
                    alternate=False,
                    counts=list(client["counts"]),
                    enabled=list(client["enabled"]),
                )
            )
        return messages


TICK_MS = TICK_RATE_MS

__all__ = [
    "TICK_MS",
    "SimServer",
]
