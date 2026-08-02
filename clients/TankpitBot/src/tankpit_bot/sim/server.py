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
    emit_chat,
    emit_equipment_pickup,
    emit_equipment_toggle,
    emit_fuel_pickup_close,
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
        "chat",
        "scope",
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

    def relocate_tank(self, tank_id: int, x: int, y: int) -> None:
        """Place a tank at a tile by recorded authority (ghost replay).

        Ghost positions come from a capture's wire record, not from
        the sim's movement law — the recording IS the routing. When
        the tank sits inside the client's stored window after the
        placement, a 0x3D position statement rides the next batch head
        (positions are viewport-scoped on the real wire); out-of-view
        placements stay silent and the end-of-tick membership diff
        announces any enter/exit exactly as live.

        Args:
            tank_id: The tank to place (must exist and be alive).
            x: Destination tile X.
            y: Destination tile Y.

        Raises:
            SimError: For unknown or dead tanks — a ghost timeline
                referencing a corpse is skipped by the caller, so a
                reach here is a harness bug.
        """
        tank = self.world["tanks"].get(tank_id)
        if tank is None or not tank["alive"]:
            raise SimError(f"no living tank {tank_id} to relocate")
        tank["x"] = x
        tank["y"] = y
        if (
            tank_id != self.client_id
            and tank_id in self._viewport.visible
            and self._viewport.in_window(x, y)
        ):
            # In-window movement of an ALREADY-visible tank re-states
            # its position (0x3D, viewport-scoped like live); a tank
            # ENTERING the window gets its 0x3D from the end-of-tick
            # membership diff instead — appending here too would
            # double the statement.
            self._pending_announcements.append(position_statement(self.world, tank_id))

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
            self._process_block_command(tank_id, command, messages)
            return
        if kind == "chat":
            emit_chat(tank_id, command, messages)
            return
        if kind == "scope":
            self._process_scope_command(tank_id, command, messages)
            return
        messages.append(build_map_data(self.world))

    def _process_block_command(
        self,
        tank_id: int,
        command: ClientCommandDict,
        messages: list[BinaryMessage],
    ) -> None:
        """Route one block pick-up/drop press through the block law.

        A landed CLIENT block action repaints the dynamic layer, so
        the stored window's patch refresh rides the same batch (the
        2026-07-20 block captures show 0x5A after block operations).

        Args:
            tank_id: The commanding tank.
            command: The queued block command.
            messages: This tick's outgoing batch (appended).
        """
        landed = emit_block_action(
            self.world, self.terrain, self.client_id, tank_id, command, messages
        )
        if landed and tank_id == self.client_id:
            messages.append(self._viewport.build_update())

    def _process_scope_command(
        self,
        tank_id: int,
        command: ClientCommandDict,
        messages: list[BinaryMessage],
    ) -> None:
        """Route one scope-extend command (the Rb viewport pan).

        Scope-extend shifts only the CLIENT's stored window (the
        server keeps one per connection; other tanks' scopes are
        invisible to this client). The confirming 0x5A always comes —
        measured lag 50 ms-1.5 s, every Rb answered
        ([[viewport-shift-protocol]]) — and it is PAIRED with a self
        0x3D position statement (the corpus's 22:22 1:1 pairing; the
        archive's 27 scope commands all answered ``5A+3Dself`` —
        response-shape differ 2026-08-01). The end-of-tick membership
        diff announces any tanks the pan revealed.

        Args:
            tank_id: The commanding tank.
            command: The queued scope command.
            messages: This tick's outgoing batch (appended).
        """
        if tank_id != self.client_id:
            return
        self._viewport.apply_scope_shift(command["direction"])
        messages.append(self._viewport.build_update())
        messages.append(position_statement(self.world, tank_id))

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
        fuel_before = _fuel_container_volume(self.world, command["x"], command["y"])
        outcome = process_move(self.world, self.terrain, tank_id, command["x"], command["y"])
        if outcome["kind"] == "moved":
            moved.add(tank_id)
        choreographed = kind == "pickup_fuel" and outcome["kind"] == "moved"
        emit_move(
            self.world,
            self.client_id,
            outcome,
            messages,
            include_pickups=not choreographed,
        )
        if choreographed:
            emit_fuel_pickup_close(
                self.world,
                self.client_id,
                tank_id,
                command["x"],
                command["y"],
                volume_before=fuel_before,
                walked=outcome["path"] != "",
                messages=messages,
            )
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
        landing: list[BinaryMessage] = []
        if emit_teleport(self.world, self.terrain, self.client_id, tank_id, command, landing):
            moved.add(tank_id)
            if tank_id == self.client_id:
                self._viewport.recenter()
                messages.append(self._viewport.build_update())
            messages.extend(landing)
            emit_equipment_pickup(
                self.world, self.client_id, tank_id, "teleport", messages, ammo_changed
            )
        else:
            messages.extend(landing)

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


def _fuel_container_volume(world: SimWorldDict, x: int, y: int) -> int:
    """The fuel volume recorded at a tile before a command resolves."""
    for container in world["containers"]:
        if (container["x"], container["y"]) == (x, y):
            return container["volume"]
    return 0


TICK_MS = TICK_RATE_MS

__all__ = [
    "CORPSE_WINDOW_TICKS",
    "TICK_MS",
    "SimServer",
]
