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
    SUPERVISOR_ERROR_INSUFFICIENT_FUEL,
)
from tankpit_bot.protocol.types import (
    BinaryMessage,
    DeactivationDict,
    FuelGainDict,
    InventoryDict,
    MovementDict,
    MovementResponseDict,
    RadarResultDict,
    RadarScanResultDict,
    ShootEventDict,
    SupervisorDict,
    TankInfoDict,
    TankStatusSyncDict,
)
from tankpit_bot.sim.actions import (
    MINE_PRESS_FUEL_COST,
    RADAR_FUEL_COST,
    build_map_data,
    process_mine_press,
    process_radar,
    process_teleport,
)
from tankpit_bot.sim.combat import process_shot
from tankpit_bot.sim.commands import ClientCommandDict, SimError
from tankpit_bot.sim.movement import MoveOutcomeDict, process_move
from tankpit_bot.sim.world import SimWorldDict

# TeleportLanded's 1-byte body observed in production captures.
_TELEPORT_LANDED_SUBTYPE = 0x0C

_SUPPORTED_KINDS = frozenset(
    {"move", "shoot", "teleport", "radar", "mine", "map_open", "pickup_fuel", "pickup_equipment"}
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

    def handshake(self) -> list[BinaryMessage]:
        """Build the session-start burst the client receives on join.

        Mirrors the real server's join choreography (and the scenario
        harness's ``place_self``): the client's own position (0x3D),
        absolute fuel (0x44), and inventory (0x49), then an identity
        (0x21) and position statement (0x3D) for every other living
        tank.

        Returns:
            The decoded messages of the join burst, in order.
        """
        client = self.world["tanks"][self.client_id]
        messages: list[BinaryMessage] = [
            self._position_statement(self.client_id),
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

        Raises:
            SimError: For command kinds outside the current build stage
                (laws 4-8 land in step d) or unknown/dead tanks.
        """
        if command["kind"] not in _SUPPORTED_KINDS:
            raise SimError(
                f"sim step b handles move/shoot only; got {command['kind']!r} "
                "(laws 4-8 land in build step d)"
            )
        tank = self.world["tanks"].get(tank_id)
        if tank is None or not tank["alive"]:
            raise SimError(f"no living tank {tank_id} to command")
        self._queue.append((tank_id, command))

    def _apply_pending_debits(self, fuel_changed: set[int]) -> None:
        """Bill last tick's firing costs (measured charge latency).

        Args:
            fuel_changed: Accumulator of tanks whose fuel moved.
        """
        for tank_id, debit in self._pending_debits:
            tank = self.world["tanks"][tank_id]
            tank["fuel"] = max(0, tank["fuel"] - debit)
            fuel_changed.add(tank_id)
        self._pending_debits = []

    def _emit_move(
        self, outcome: MoveOutcomeDict, messages: list[BinaryMessage], fuel_changed: set[int]
    ) -> None:
        """Emit the wire consequences of one processed move.

        Args:
            outcome: The move's outcome.
            messages: This tick's outgoing batch (appended).
            fuel_changed: Accumulator of tanks whose fuel moved.
        """
        if outcome["kind"] == "cant_go":
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
        fuel_changed.add(outcome["tank_id"])
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
        fuel_changed: set[int],
        ammo_changed: set[int],
        tank_id: int,
        command: ClientCommandDict,
        moved: frozenset[int],
    ) -> None:
        """Process one shoot command and emit its wire consequences.

        Args:
            shooter_team: The shooter's team (for the 0x53 echo).
            outcome_messages: This tick's outgoing batch (appended).
            fuel_changed: Accumulator of tanks whose fuel moved.
            ammo_changed: Accumulator of tanks whose counts moved.
            tank_id: The firing tank.
            command: The shoot command.
            moved: Tanks that moved earlier this tick.
        """
        outcome = process_shot(self.world, self.terrain, tank_id, command["x"], command["y"], moved)
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
        if outcome["victim_id"] is not None and outcome["shields_consumed"] == 0:
            fuel_changed.add(outcome["victim_id"])
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
        fuel_changed: set[int],
    ) -> bool:
        """Process one teleport command and emit its wire consequences.

        Args:
            tank_id: The teleporting tank.
            command: The teleport command.
            messages: This tick's outgoing batch (appended).
            fuel_changed: Accumulator of tanks whose fuel moved.

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
            messages.append(
                SupervisorDict(msg_type=0x52, reset_action=1, close_map=1, error_code=code)
            )
            return False
        messages.append(
            TeleportLandedDict(msg_type="teleport_landed", subtype=_TELEPORT_LANDED_SUBTYPE)
        )
        messages.append(self._position_statement(tank_id))
        fuel_changed.add(tank_id)
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
        fuel_changed: set[int],
        ammo_changed: set[int],
    ) -> None:
        """Process one radar command and emit its wire consequences.

        Args:
            tank_id: The scanning tank.
            messages: This tick's outgoing batch (appended).
            fuel_changed: Accumulator of tanks whose fuel moved.
            ammo_changed: Accumulator of tanks whose counts moved.
        """
        outcome = process_radar(self.world, tank_id)
        tank = self.world["tanks"][tank_id]
        tank["fuel"] = max(0, tank["fuel"] - RADAR_FUEL_COST)
        fuel_changed.add(tank_id)
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

    def _emit_mine_press(
        self, tank_id: int, messages: list[BinaryMessage], fuel_changed: set[int]
    ) -> None:
        """Process one mine press and emit its wire consequences.

        Args:
            tank_id: The placing tank.
            messages: This tick's outgoing batch (appended).
            fuel_changed: Accumulator of tanks whose fuel moved.
        """
        outcome = process_mine_press(self.world, self.terrain, tank_id)
        tank = self.world["tanks"][tank_id]
        tank["fuel"] = max(0, tank["fuel"] - MINE_PRESS_FUEL_COST)
        fuel_changed.add(tank_id)
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
        fuel_changed: set[int],
        ammo_changed: set[int],
        moved: set[int],
    ) -> None:
        """Route one queued command to its law processor.

        Args:
            tank_id: The commanding tank.
            command: The queued command.
            messages: This tick's outgoing batch (appended).
            fuel_changed: Accumulator of tanks whose fuel moved.
            ammo_changed: Accumulator of tanks whose counts moved.
            moved: Accumulator of tanks that relocated this tick.
        """
        kind = command["kind"]
        if kind in _MOVE_KINDS:
            outcome = process_move(self.world, self.terrain, tank_id, command["x"], command["y"])
            if outcome["kind"] == "moved":
                moved.add(tank_id)
            self._emit_move(outcome, messages, fuel_changed)
            return
        if kind == "shoot":
            self._emit_shot(
                self.world["tanks"][tank_id]["team"],
                messages,
                fuel_changed,
                ammo_changed,
                tank_id,
                command,
                frozenset(moved),
            )
            return
        if kind == "teleport":
            if self._emit_teleport(tank_id, command, messages, fuel_changed):
                moved.add(tank_id)
            return
        if kind == "radar":
            self._emit_radar(tank_id, messages, fuel_changed, ammo_changed)
            return
        if kind == "mine":
            self._emit_mine_press(tank_id, messages, fuel_changed)
            return
        messages.append(build_map_data(self.world))

    def advance_tick(self) -> list[BinaryMessage]:
        """Process the queue and return this tick's outgoing batch.

        Returns:
            The decoded messages the client receives this tick, in
            emission order: last tick's deferred debits are billed
            first, then each queued command's consequences, then one
            fuel sync per fuel-changed tank and an inventory snapshot
            when the client's counts changed.
        """
        self.world["tick"] += 1
        messages: list[BinaryMessage] = []
        fuel_changed: set[int] = set()
        ammo_changed: set[int] = set()
        self._apply_pending_debits(fuel_changed)
        moved: set[int] = set()
        for tank_id, command in self._queue:
            if not self.world["tanks"][tank_id]["alive"]:
                continue
            self._process_command(tank_id, command, messages, fuel_changed, ammo_changed, moved)
        self._queue = []
        for tank_id in sorted(fuel_changed):
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
