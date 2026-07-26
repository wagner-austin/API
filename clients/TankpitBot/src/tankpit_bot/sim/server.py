"""Law 1 — the global command queue and the 2-second tick processor.

Commands queue in arrival order and process together at tick
boundaries; every wire message a tick produces flushes as one batch
(the measured sync cadence, wiki log 2026-07-21). Shooter firing
costs bill one tick AFTER the shot (measured charge latency); victim
damage bills instantly inside :mod:`tankpit_bot.sim.combat`.

The server here is routing and orchestration only. Each concern owns
its module: :mod:`tankpit_bot.sim.viewport_window` (the client's
stored 0x5A window, patch memory, and visibility diffs),
:mod:`tankpit_bot.sim.combat_emissions` (shots, kill rewards, the
deferred-debit and corpse-window clocks),
:mod:`tankpit_bot.sim.emissions` (per-command wire emission), and
:mod:`tankpit_bot.sim.wire_statements` (pure message builders).

The processor emits decoded ``BinaryMessage`` dicts; the transport
layer (build step c) turns them into wire bytes via
``protocol.encoders.encode_envelope_body``.
"""

from __future__ import annotations

from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.protocol.commands import TICK_RATE_MS
from tankpit_bot.protocol.constants import (
    SUPERVISOR_ERROR_CANT_DO,
    SUPERVISOR_ERROR_EMPTY_CONTAINER,
)
from tankpit_bot.protocol.types import (
    BinaryMessage,
    FuelGainDict,
    InventoryDict,
    SupervisorDict,
)
from tankpit_bot.sim.actions import build_map_data
from tankpit_bot.sim.bot_policy import reactivate_practice_bot
from tankpit_bot.sim.combat_emissions import CORPSE_WINDOW_TICKS, CombatLedger
from tankpit_bot.sim.commands import ClientCommandDict, SimError
from tankpit_bot.sim.emissions import (
    emit_block_action,
    emit_equipment_pickup,
    emit_equipment_toggle,
    emit_mine_press,
    emit_move,
    emit_radar,
    emit_teleport,
)
from tankpit_bot.sim.movement import process_move
from tankpit_bot.sim.viewport_window import ViewportTracker
from tankpit_bot.sim.wire_statements import (
    identity_statement,
    position_statement,
    queued_tank_id,
    status_sync,
)
from tankpit_bot.sim.world import SimWorldDict

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
        "block",
    }
)
_MOVE_KINDS = frozenset({"move", "pickup_fuel", "pickup_equipment"})


class SimServer:
    """The fake server: one world, one command queue, one client.

    ``client_id`` is the tank whose connection this server speaks for
    — it receives its own fuel syncs and inventory snapshots; other
    tanks' commands are queued by the opponent policies directly.
    """

    def __init__(
        self,
        world: SimWorldDict,
        terrain: TerrainMapProtocol,
        client_id: int,
        roster_ids: frozenset[int] = frozenset(),
    ) -> None:
        """Bind the server to a world, its terrain, and the client tank.

        Args:
            world: Simulated world (owned and mutated by the server).
            terrain: Static terrain for the world's field.
            client_id: The connected client's tank id.
            roster_ids: Practice-roster tanks that REACTIVATE in place
                with the same id at full fuel when their corpse clears
                (archive-mined 2026-07-24, [[enemy-bot-behavior]]).
                Empty for worlds without roster bots.
        """
        self.world = world
        self.terrain = terrain
        self.client_id = client_id
        self._roster_ids = roster_ids
        self._queue: list[tuple[int, ClientCommandDict]] = []
        self._pending_announcements: list[BinaryMessage] = []
        self._viewport = ViewportTracker(world, terrain, client_id)
        self._combat = CombatLedger(world, terrain, client_id)

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
            identity_statement(self.world, self.client_id),
            position_statement(self.world, self.client_id),
            self._viewport.build_update(),
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
            messages.append(identity_statement(self.world, tank_id))
            if tank_id in self._viewport.visible:
                messages.append(position_statement(self.world, tank_id))
        return messages

    def announce_tank(self, tank_id: int) -> None:
        """Queue a mid-session 0x21 identity broadcast (an activation).

        Real respawns join with a NEW wire tank id — that is what
        ``persistent_tank_id`` exists to bridge — and the room learns
        the identity from the activation's 0x21. The broadcast rides
        at the head of the next tick's batch.

        Args:
            tank_id: The newly activated tank.
        """
        self._pending_announcements.append(identity_statement(self.world, tank_id))

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
            self._combat.emit_shot(
                self.world["tanks"][tank_id]["team"],
                messages,
                ammo_changed,
                tank_id,
                command,
                frozenset(moved),
                self._viewport.removed_at.get(command["target_id"]),
            )
            return
        if kind == "teleport":
            self._process_teleport_command(tank_id, command, messages, ammo_changed, moved)
            return
        if kind == "radar":
            emit_radar(
                self.world, self.client_id, self._viewport.window, tank_id, messages, ammo_changed
            )
            return
        if kind == "mine":
            emit_mine_press(self.world, self.terrain, tank_id, messages)
            return
        if kind == "toggle_equipment":
            emit_equipment_toggle(self.world, tank_id, command["slot"], messages)
            return
        if kind == "block":
            if (
                emit_block_action(
                    self.world, self.terrain, self.client_id, tank_id, command, messages
                )
                and tank_id == self.client_id
            ):
                messages.append(self._viewport.build_update())
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

        The client's destination must lie inside its stored 0x5A
        window — the real router rejects any target outside it with
        0x52 code 0, at exactly the boundary column (measured
        2026-07-25, [[viewport-shift-protocol]]); the check precedes
        the container validation because the server never answers for
        a coordinate it does not consider actionable. Pickup clicks
        then validate their destination (empty-container rejection).
        A successful walk does NOT re-emit 0x5A: with autoscroll OFF
        the window is static between teleports.

        Args:
            tank_id: The commanding tank.
            kind: The command kind.
            command: The queued command.
            messages: This tick's outgoing batch (appended).
            ammo_changed: Accumulator of tanks whose counts moved.
            moved: Accumulator of tanks that relocated this tick.
        """
        if tank_id == self.client_id and not self._viewport.in_window(command["x"], command["y"]):
            messages.append(
                SupervisorDict(
                    msg_type=0x52,
                    reset_action=1,
                    close_map=0,
                    error_code=SUPERVISOR_ERROR_CANT_DO,
                )
            )
            return
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
        emit_move(self.world, self.client_id, outcome, messages)
        if outcome["kind"] == "moved":
            emit_equipment_pickup(self.world, self.client_id, tank_id, kind, messages, ammo_changed)

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
        2026-07-20 capture. A landed client hop is the ONE window
        recenter under autoscroll OFF ([[viewport-shift-protocol]])
        and resolves equipment on arrival.

        Args:
            tank_id: The hopping tank.
            command: The queued command.
            messages: This tick's outgoing batch (appended).
            ammo_changed: Accumulator of tanks whose counts moved.
            moved: Accumulator of tanks that relocated this tick.
        """
        if self.world["tanks"][tank_id]["carrying"]:
            if tank_id == self.client_id:
                messages.append(
                    SupervisorDict(
                        msg_type=0x52,
                        reset_action=1,
                        close_map=1,
                        error_code=SUPERVISOR_ERROR_CANT_DO,
                    )
                )
            return
        if emit_teleport(self.world, self.terrain, self.client_id, tank_id, command, messages):
            moved.add(tank_id)
            if tank_id == self.client_id:
                self._viewport.recenter()
                messages.append(self._viewport.build_update())
            emit_equipment_pickup(
                self.world, self.client_id, tank_id, "teleport", messages, ammo_changed
            )

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
        # No runtime container spawning: the 2026-07-22 "respawn law"
        # was falsified 2026-07-25 (every observed "spawn" was an
        # exposure of a pre-existing container — [[game-economy]]).
        # The world is a static population seeded by sim.world_seed.
        messages: list[BinaryMessage] = list(self._pending_announcements)
        self._pending_announcements = []
        ammo_changed: set[int] = set()
        self._combat.apply_pending_debits()
        moved: set[int] = set()
        # Within-round resolution order is ASCENDING TANK ID — the
        # measured law (2026-07-25, `analysis_scripts/mine_round_order.py`:
        # 1,820/1,825 archive multi-shooter bursts, the real server's
        # only ordering; bots 500-535 always resolve before players).
        # The sort is stable, so one tank's own commands keep arrival
        # order.
        for tank_id, command in sorted(self._queue, key=queued_tank_id):
            if not self.world["tanks"][tank_id]["alive"]:
                continue
            self._process_command(tank_id, command, messages, ammo_changed, moved)
        self._queue = []
        for tank_id in self._combat.expire_corpses(messages):
            if tank_id in self._roster_ids:
                # Roster bots come back the same tick their corpse
                # clears: same id, full fuel, respawned FAR from
                # the corpse — the viewport diff below announces
                # them if the landing is in view, and the sync
                # loop resumes their tier-3 cadence either way.
                reactivate_practice_bot(self.world, self.terrain, tank_id)
        self._viewport.emit_transitions(messages)
        # Dynamic-layer refresh is EVENT-driven, never walk-driven:
        # the client's window is static between teleports (autoscroll
        # OFF, [[viewport-shift-protocol]] — 16+ probed walks drew
        # zero 0x5A), but a ferry or block moving inside the patch
        # grid repaints it (the 2026-07-20 block captures show 0x5A
        # after block operations). An empty patch is not sent.
        refresh = self._viewport.build_update()
        if refresh["entities"]:
            messages.append(refresh)
        for tank_id in sorted(self.world["tanks"]):
            if self.world["tanks"][tank_id]["alive"]:
                messages.append(status_sync(tank_id, self.world, tank_id == self.client_id))
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
    "CORPSE_WINDOW_TICKS",
    "TICK_MS",
    "SimServer",
]
