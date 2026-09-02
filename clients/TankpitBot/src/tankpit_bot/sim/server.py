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
from tankpit_bot.protocol.types import (
    BinaryMessage,
    EquipmentToggleDict,
    InventoryDict,
    SyncDict,
)
from tankpit_bot.sim.actions import build_map_data
from tankpit_bot.sim.bot_policy import reactivate_practice_bot
from tankpit_bot.sim.client_session import ClientSession
from tankpit_bot.sim.combat_emissions import CORPSE_WINDOW_TICKS
from tankpit_bot.sim.commands import ClientCommandDict, SimError
from tankpit_bot.sim.emissions import (
    emit_block_action,
    emit_chat,
    emit_equipment_toggle,
    emit_mine_press,
    emit_radar,
)
from tankpit_bot.sim.ferries import drift_ferries
from tankpit_bot.sim.server_move import SimServerMoveMixin
from tankpit_bot.sim.visitors import RoomChurn
from tankpit_bot.sim.wire_statements import (
    full_status_statement,
    identity_statement,
    position_statement,
    queued_tank_id,
    statistics_statement,
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
        "statistics",
    }
)
_MOVE_KINDS = frozenset({"move", "pickup_fuel", "pickup_equipment"})


class SimServer(SimServerMoveMixin):
    """The fake server: one world, one command queue, one client.

    The server owns FIELD state directly — the world, its terrain, the
    command queue every tank feeds, and the room's churn — and holds
    the connected client's own state in a :class:`ClientSession`. That
    boundary is the point: the session's four holders are precisely
    what a second connection would need its own copy of, and nothing
    else here is.

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
        self.session = ClientSession(world, terrain, client_id)
        self._roster_ids = roster_ids
        self._queue: list[tuple[int, ClientCommandDict]] = []
        self._pending_announcements: list[BinaryMessage] = []
        self._churn = RoomChurn()

    def handshake(self) -> list[BinaryMessage]:
        """Build the session-start burst the client receives on join.

        Mirrors the real server's join choreography (and the scenario
        harness's ``place_self``). The whole burst was re-measured
        2026-09-01 across 341 archived sessions and is INVARIANT in
        every one of the 340 that carry it ([[recipient-policy]])::

            0x21  0x3E  0x5A  0x3D  0x2E  |  0x21 xN  |  0x49 x2  0x74  0x3F

        The client's OWN identity (0x21) leads — the archive convention
        the audit validators rely on is that the first TankInfo of a
        session names the player's own tank (``validate.wire_timeline``)
        — then its FULL status (0x3E), the viewport patch (0x5A), own
        position (0x3D) and own status sync (0x2E); then a PURE run of
        identities, one per other living tank; then the tail.

        Four corrections landed with that measurement, each of which
        the single-client sim had no way to falsify:

        * **No join-time 0x44.** 293 of 340 sessions carry no 0x44 in
          their first 90 received messages at all, and the rest carry
          it 11-36 messages past the sync — it answers fuel events, not
          joining. The burst's 0x2E already carries fuel
          ([[decode-coverage]]), which is why the real server needs no
          0x44 here. The sim emitted one and no 0x2E.
        * **The identity run is pure 0x21.** 340/340. With ~36 tanks on
          a 256x256 map a 16x16 window should hold one about 13% of the
          time, so zero of 340 is the law, not sampling: other tanks'
          positions arrive from the in-play membership diff, never from
          the burst. The sim rode a 0x3D on every visible tank.
        * **The inventory arrives TWICE**, not once.
        * **The 0x49 pair sits at the TAIL**, after the identity run —
          the sim had a single 0x49 in the self block, before it.

        The 0x74 equipment-enabled state closes the tail: 324 of 341
        sessions receive exactly one, 340/340 of them immediately after
        the 0x49 pair and immediately before the 0x3F. It is a JOIN
        message carrying the tank's persisted enabled flags, not an
        answer to a toggle ([[recipient-policy]]). The sim had none.

        285 of the 286 archived CMD_ENTER_GAME sends draw exactly one
        0x3F, and the median session carries exactly one sync in total
        — joining IS the common case
        ([[session-state-deglobalisation]]).

        Returns:
            The decoded messages of the join burst, in order.
        """
        client = self.world["tanks"][self.session.client_id]
        inventory = InventoryDict(
            msg_type=0x49,
            show=True,
            alternate=False,
            counts=list(client["counts"]),
            enabled=list(client["enabled"]),
        )
        messages: list[BinaryMessage] = [
            identity_statement(
                self.world, self.session.client_id, self.session.awards.decoration_state
            ),
            full_status_statement(
                self.world, self.session.client_id, self.session.awards.decoration_state
            ),
            self.session.viewport.build_update(),
            position_statement(self.world, self.session.client_id),
            status_sync(
                self.session.client_id, self.world, True, self.session.progression.promo_state
            ),
        ]
        # The identity run is PURE 0x21 — no position statements ride
        # it. Measured 340/340 (2026-09-01, [[recipient-policy]]): with
        # ~36 tanks on a 256x256 map a 16x16 window should hold one
        # about 13% of the time, so zero of 340 is not sampling, it is
        # the law. Other tanks' positions arrive from the in-play
        # membership diff, never from the join burst.
        for tank_id in sorted(self.world["tanks"]):
            tank = self.world["tanks"][tank_id]
            if tank_id == self.session.client_id or not tank["alive"]:
                continue
            messages.append(identity_statement(self.world, tank_id))
        # The burst TAIL, measured 340/340: the inventory arrives
        # TWICE, then the equipment-enabled state, then the sync — and
        # the pair sits AFTER the identity run, not in the self block
        # ([[recipient-policy]]). 285 of the 286 archived CMD_ENTER_GAME
        # sends draw exactly one 0x3F, and the median session carries
        # exactly one sync in total — joining IS the common case
        # ([[session-state-deglobalisation]]).
        messages.append(inventory)
        messages.append(inventory)
        messages.append(EquipmentToggleDict(msg_type=0x74, enabled=list(client["enabled"])))
        messages.append(SyncDict(msg_type=0x3F))
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
            tank_id != self.session.client_id
            and tank_id in self.session.viewport.visible
            and self.session.viewport.in_window(x, y)
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
            if tank_id == self.session.client_id:
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
        """Route one queued command that spends ammo or relocates.

        The kinds whose consequences feed this tick's accumulators are
        routed here; everything else goes to
        :meth:`_process_stateless_command`, which needs neither. The
        split is what keeps either router readable as the command
        vocabulary grows.

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
            self.session.combat.emit_shot(
                self.world["tanks"][tank_id]["team"],
                messages,
                ammo_changed,
                tank_id,
                command,
                frozenset(moved),
                self.session.viewport.removed_at.get(command["target_id"]),
            )
            return
        if kind == "teleport":
            self._process_teleport_command(tank_id, command, messages, ammo_changed, moved)
            return
        if kind == "radar":
            emit_radar(
                self.world,
                self.session.client_id,
                self.session.viewport.window,
                tank_id,
                messages,
                ammo_changed,
            )
            return
        self._process_stateless_command(tank_id, command, messages)

    def _process_stateless_command(
        self,
        tank_id: int,
        command: ClientCommandDict,
        messages: list[BinaryMessage],
    ) -> None:
        """Route one queued command with no ammo or movement effect.

        Args:
            tank_id: The commanding tank.
            command: The queued command.
            messages: This tick's outgoing batch (appended).
        """
        kind = command["kind"]
        if kind == "mine":
            emit_mine_press(self.world, self.terrain, self.session.client_id, tank_id, messages)
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
        if kind == "statistics":
            # Per-connection, like every other answer: the statistics
            # of the tank that asked, and only to that tank.
            if tank_id == self.session.client_id:
                messages.append(
                    statistics_statement(
                        self.world["tick"],
                        self.session.combat.client_destroyed,
                        self.session.combat.client_deactivated,
                    )
                )
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
            self.world, self.terrain, self.session.client_id, tank_id, command, messages
        )
        if landed and tank_id == self.session.client_id:
            messages.append(self.session.viewport.build_update())

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
        if tank_id != self.session.client_id:
            return
        self.session.viewport.apply_scope_shift(command["direction"])
        messages.append(self.session.viewport.build_update())
        messages.append(position_statement(self.world, tank_id))

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
        self.session.combat.apply_pending_debits()
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
        # A deactivation of the CLIENT demotes it to recruit, silently,
        # in the same batch as the 0x41 — three archived demotions,
        # every one at a zero-second gap. The batch is read rather than
        # the ledger being told about ranks: one 0x41 is the fact, and
        # the ledger's job is to produce it, not to interpret it
        # ([[session-state-deglobalisation]]).
        if any(
            message["msg_type"] == 0x41 and message["victim_id"] == self.session.client_id
            for message in messages
        ):
            self.session.progression.note_deactivation(self.world, messages)
        for tank_id in self.session.combat.expire_corpses(messages):
            if tank_id in self._roster_ids:
                # Roster bots come back the same tick their corpse
                # clears: same id, full fuel, respawned FAR from
                # the corpse — the viewport diff below announces
                # them if the landing is in view, and the sync
                # loop resumes their tier-3 cadence either way.
                reactivate_practice_bot(self.world, self.terrain, tank_id)
        # Ferries drift on the tick (measured median gap 2003 ms) and
        # BEFORE the viewport refresh below, so a ferry crossing the
        # patch grid repaints it in the same batch that announced the
        # move ([[ferry-mechanics]], [[session-state-deglobalisation]]).
        drift_ferries(self.world, self.terrain, messages)
        # Room churn runs BEFORE the viewport diff, so a visitor who
        # lands inside the client's window is announced by the same
        # membership pass that announces any other arrival
        # ([[session-state-deglobalisation]]).
        self._churn.advance(self.world, self.terrain, messages)
        self.session.viewport.emit_transitions(messages)
        # Dynamic-layer refresh is EVENT-driven, never walk-driven:
        # the client's window is static between teleports (autoscroll
        # OFF, [[viewport-shift-protocol]] — 16+ probed walks drew
        # zero 0x5A), but a ferry or block moving inside the patch
        # grid repaints it (the 2026-07-20 block captures show 0x5A
        # after block operations). An empty patch is not sent.
        refresh = self.session.viewport.build_update()
        if refresh["entities"]:
            messages.append(refresh)
        # The promotion that ends a recovery window, before the syncs
        # so this tick's bar already reads the restored steady state.
        self.session.progression.advance(self.world, messages)
        # Awards are granted from the same counters the 0x56 reports,
        # against the thresholds the in-client guide names
        # ([[decoration-encoding]]): 100/200/500 kills, 20/50/100
        # deaths, Major/Colonel/General, 100/200/500 hours. The archive
        # caught exactly one grant — Artax's 500th kill stepping the
        # Tank award to golden on 2026-07-29
        # ([[session-state-deglobalisation]]).
        self.session.awards.advance(
            self.world["tanks"][self.session.client_id]["rank"],
            self.session.combat.client_destroyed,
            self.session.combat.client_deactivated,
            self.world["tick"] * TICK_RATE_MS // 1000,
            messages,
        )
        for tank_id in sorted(self.world["tanks"]):
            if self.world["tanks"][tank_id]["alive"]:
                messages.append(
                    status_sync(
                        tank_id,
                        self.world,
                        tank_id == self.session.client_id,
                        self.session.progression.promo_state,
                    )
                )
        if self.session.client_id in ammo_changed:
            client = self.world["tanks"][self.session.client_id]
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


__all__ = [
    "CORPSE_WINDOW_TICKS",
    "SimServer",
]
