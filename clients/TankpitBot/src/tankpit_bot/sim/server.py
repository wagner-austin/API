"""Law 1 — the global command queue and the 2-second tick processor.

Commands queue in arrival order and process together at tick
boundaries; every wire message a tick produces flushes as one batch
(the measured sync cadence, wiki log 2026-07-21). Shooter firing
costs bill one tick AFTER the shot (measured charge latency); victim
damage bills instantly inside :mod:`tankpit_bot.sim.combat`.

The server here is routing and orchestration only. Each concern owns
its module: :mod:`tankpit_bot.sim.viewport_window` (the client's
stored 0x5A window, patch memory, and visibility diffs),
:mod:`tankpit_bot.sim.combat_clock` (the deferred-debit and
corpse-window clocks and the kill book),
:mod:`tankpit_bot.sim.narrate` (pure per-observer wire narration), and
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
    InventoryDict,
)
from tankpit_bot.sim.actions import build_map_data, process_mine_press, process_radar
from tankpit_bot.sim.blocks import process_block_press
from tankpit_bot.sim.client_session import ClientSession
from tankpit_bot.sim.combat_clock import CORPSE_WINDOW_TICKS, CombatClock
from tankpit_bot.sim.commands import ClientCommandDict, ClientCommandKind, SimError
from tankpit_bot.sim.equipment import toggle_equipment_slot
from tankpit_bot.sim.narrate import (
    narrate_block_action,
    narrate_chat,
    narrate_equipment_toggle,
    narrate_mine_press,
    narrate_radar,
)
from tankpit_bot.sim.outbox import TickOutbox
from tankpit_bot.sim.server_combat import SimServerCombatMixin
from tankpit_bot.sim.server_move import SimServerMoveMixin
from tankpit_bot.sim.server_queries import SimServerQueriesMixin
from tankpit_bot.sim.visitors import RoomChurn
from tankpit_bot.sim.wire_statements import (
    identity_statement,
    position_statement,
    queued_tank_id,
    status_sync,
)
from tankpit_bot.sim.world import SimWorldDict

#: Every client command kind the tick processor can route. PUBLIC
#: because it is a contract, not an implementation detail: the
#: command-coverage audit reads it to answer "does the sim survive
#: everything a real client sends", and a private copy of this set
#: living in the audit would drift the moment a command was added
#: ([[client-commands]]).
SUPPORTED_KINDS: frozenset[ClientCommandKind] = frozenset(
    kind for kind in ClientCommandKind if kind is not ClientCommandKind.OTHER
)
_MOVE_KINDS: frozenset[ClientCommandKind] = frozenset(
    {
        ClientCommandKind.MOVE,
        ClientCommandKind.PICKUP_FUEL,
        ClientCommandKind.PICKUP_EQUIPMENT,
        ClientCommandKind.DEPOSIT_FUEL,
    }
)


class SimServer(SimServerCombatMixin, SimServerMoveMixin, SimServerQueriesMixin):
    """The fake server: one field, one command queue, any connections.

    The server owns FIELD state directly — the world, its terrain, the
    command queue every tank feeds, the room's churn, and the combat
    clocks — and holds each connection's own state in a
    :class:`~tankpit_bot.sim.client_session.ClientSession`, admitted
    with :meth:`connect`. Every tank feeds the one queue whether a
    connection or a sim policy drives it; what a connection adds is a
    batch of its own each tick, holding the receipts it is owed and
    its view of everything else ([[recipient-policy]]).
    """

    def __init__(
        self,
        world: SimWorldDict,
        terrain: TerrainMapProtocol,
        roster_ids: frozenset[int] = frozenset(),
    ) -> None:
        """Bind the server to a world and its terrain, with no connections.

        Args:
            world: Simulated world (owned and mutated by the server).
            terrain: Static terrain for the world's field.
            roster_ids: Practice-roster tanks that REACTIVATE in place
                with the same id at full fuel when their corpse clears
                (archive-mined 2026-07-24, [[enemy-bot-behavior]]).
                Empty for worlds without roster bots.
        """
        self.world = world
        self.terrain = terrain
        self._sessions: dict[int, ClientSession] = {}
        # FIELD state, deliberately not on a session: a corpse clears
        # once for the room and a firing cost is billed once against
        # the shooter, however many connections watch.
        self.combat = CombatClock(world)
        self._roster_ids = roster_ids
        self._queue: list[tuple[int, ClientCommandDict]] = []
        # The NEXT tick's outbox, open between ticks: what happens to
        # the field outside the tick processor — an activation, a
        # ghost's recorded relocation, a connection joining — is owed
        # to the connections at the head of the next batch.
        self._outbox = TickOutbox(())
        self._churn = RoomChurn()

    def announce_tank(self, tank_id: int) -> None:
        """Queue a mid-session 0x21 identity broadcast (an activation).

        Real respawns join with a NEW wire tank id — that is what
        ``persistent_tank_id`` exists to bridge — and the room learns
        the identity from the activation's 0x21. The broadcast rides
        at the head of the next tick's batches.

        Args:
            tank_id: The newly activated tank.
        """
        self._outbox.broadcast([identity_statement(self.world, tank_id)])

    def relocate_tank(self, tank_id: int, x: int, y: int) -> None:
        """Place a tank at a tile by recorded authority (ghost replay).

        Ghost positions come from a capture's wire record, not from
        the sim's movement law — the recording IS the routing. Each
        connection whose stored window holds the tank after the
        placement gets a 0x3D position statement at its next batch
        head (positions are viewport-scoped on the real wire);
        out-of-view placements stay silent and the end-of-tick
        membership diff announces any enter/exit exactly as live.

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
        for session in self.sessions:
            if (
                tank_id != session.client_id
                and tank_id in session.viewport.visible
                and session.viewport.in_window(x, y)
            ):
                # In-window movement of an ALREADY-visible tank
                # re-states its position (0x3D, viewport-scoped like
                # live); a tank ENTERING the window gets its 0x3D from
                # the end-of-tick membership diff instead — sending it
                # here too would double the statement.
                self._outbox.to(session.client_id, position_statement(self.world, tank_id))

    def queue_command(self, tank_id: int, command: ClientCommandDict) -> None:
        """Queue one command for the next tick.

        Args:
            tank_id: The commanding tank.
            command: Decoded client command.

        A DEAD CONNECTED tank's commands drop silently: the real
        connection survives deactivation and the server simply ignores
        a corpse's clicks (first real-terrain CLI run, 2026-07-22: the
        enemy killed the bot and the production loop kept clicking —
        real behavior, not a harness bug). Dead or unknown tanks with
        no connection still raise, because only the harness can queue
        those.

        Raises:
            SimError: For unsupported command kinds, unknown tanks, or
                dead harness-driven tanks.
        """
        if command["kind"] not in SUPPORTED_KINDS:
            # The message names the gap rather than the build phase.
            # It used to read "sim step b handles move/shoot only ...
            # (laws 4-8 land in build step d)" — true in July 2026,
            # stale ever since, and actively misleading once the
            # server handled fifteen kinds. A refusal here is the sim
            # saying it has no MEASURED law for the command, which is
            # the correct answer for a fidelity harness and the wrong
            # one for a hosted server; that tension is real and
            # recorded, not resolved by guessing ([[client-commands]]).
            raise SimError(
                f"no modelled law for client command {command['kind'].value!r} "
                f"(byte 0x{command['command']:02X}); the sim refuses rather "
                "than inventing a response"
            )
        tank = self.world["tanks"].get(tank_id)
        if tank is None:
            raise SimError(f"no tank {tank_id} to command")
        if not tank["alive"]:
            if self.session_for(tank_id) is not None:
                return
            raise SimError(f"no living tank {tank_id} to command")
        self._queue.append((tank_id, command))

    def _process_command(
        self,
        tank_id: int,
        command: ClientCommandDict,
        outbox: TickOutbox,
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
            outbox: This tick's outgoing batches (appended).
            ammo_changed: Accumulator of tanks whose counts moved.
            moved: Accumulator of tanks that relocated this tick.
        """
        kind = command["kind"]
        if kind in _MOVE_KINDS:
            self._process_move_command(tank_id, kind, command, outbox, ammo_changed, moved)
            return
        if kind is ClientCommandKind.SHOOT:
            self._process_shoot_command(tank_id, command, outbox, ammo_changed, moved)
            return
        if kind is ClientCommandKind.TELEPORT:
            self._process_teleport_command(tank_id, command, outbox, ammo_changed, moved)
            return
        if kind is ClientCommandKind.RADAR:
            # The scan is bounded by the SCANNING tank's own window; a
            # tank with no connection has none to bound it.
            session = self.session_for(tank_id)
            window = None if session is None else session.viewport.window
            radar = process_radar(self.world, tank_id, window)
            outbox.narrate(lambda observer_id: narrate_radar(self.world, radar, observer_id))
            return
        self._process_stateless_command(tank_id, command, outbox)

    def _process_stateless_command(
        self,
        tank_id: int,
        command: ClientCommandDict,
        outbox: TickOutbox,
    ) -> None:
        """Route one queued command with no ammo or movement effect.

        Connection-scoped questions are answered first and elsewhere
        (:class:`SimServerQueriesMixin`); what reaches the chain below
        acts on the world. ``map_open`` is the fallthrough, and its
        0x4C dot atlas is an answer to the tank that opened the map.

        Args:
            tank_id: The commanding tank.
            command: The queued command.
            outbox: This tick's outgoing batches (appended).
        """
        kind = command["kind"]
        if self._answer_connection_query(tank_id, kind, outbox):
            return
        if kind is ClientCommandKind.MINE:
            press = process_mine_press(self.world, self.terrain, tank_id)
            outbox.narrate(lambda observer_id: narrate_mine_press(press, observer_id))
            return
        if kind is ClientCommandKind.TOGGLE_EQUIPMENT:
            toggle_equipment_slot(self.world, tank_id, command["slot"])
            outbox.narrate(
                lambda observer_id: narrate_equipment_toggle(self.world, tank_id, observer_id)
            )
            return
        if kind is ClientCommandKind.BLOCK:
            self._process_block_command(tank_id, command, outbox)
            return
        if kind is ClientCommandKind.CHAT:
            outbox.broadcast(narrate_chat(tank_id, command))
            return
        if kind is ClientCommandKind.SCOPE:
            self._process_scope_command(tank_id, command, outbox)
            return
        outbox.to(tank_id, build_map_data(self.world))

    def _process_block_command(
        self,
        tank_id: int,
        command: ClientCommandDict,
        outbox: TickOutbox,
    ) -> None:
        """Route one block pick-up/drop press through the block law.

        A landed block action by a CONNECTED tank repaints the dynamic
        layer, so its stored window's patch refresh rides the same
        batch (the 2026-07-20 block captures show 0x5A after block
        operations).

        Args:
            tank_id: The commanding tank.
            command: The queued block command.
            outbox: This tick's outgoing batches (appended).
        """
        outcome = process_block_press(self.world, self.terrain, tank_id, command["x"], command["y"])
        outbox.narrate(
            lambda observer_id: narrate_block_action(self.world, outcome, tank_id, observer_id)
        )
        session = self.session_for(tank_id)
        if session is not None and outcome["kind"] not in ("out_of_reach", "refused"):
            outbox.to(tank_id, session.viewport.build_update())

    def _process_scope_command(
        self,
        tank_id: int,
        command: ClientCommandDict,
        outbox: TickOutbox,
    ) -> None:
        """Route one scope-extend command (the Rb viewport pan).

        Scope-extend shifts only the commanding connection's stored
        window (the server keeps one per connection; another tank's
        scope is invisible to everyone else). The confirming 0x5A always comes —
        measured lag 50 ms-1.5 s, every Rb answered
        ([[viewport-shift-protocol]]) — and it is PAIRED with a self
        0x3D position statement (the corpus's 22:22 1:1 pairing; the
        archive's 27 scope commands all answered ``5A+3Dself`` —
        response-shape differ 2026-08-01). The end-of-tick membership
        diff announces any tanks the pan revealed.

        Args:
            tank_id: The commanding tank.
            command: The queued scope command.
            outbox: This tick's outgoing batches (appended).
        """
        session = self.session_for(tank_id)
        if session is None:
            return
        session.viewport.apply_scope_shift(command["direction"])
        outbox.to(tank_id, session.viewport.build_update())
        outbox.to(tank_id, position_statement(self.world, tank_id))

    def advance_tick(self) -> dict[int, list[BinaryMessage]]:
        """Process the queue and return this tick's outgoing batches.

        Returns:
            One batch per connected tank, keyed by its id in the order
            the connections joined. Each holds that connection's
            decoded messages in emission order: whatever reached the
            field between ticks, last tick's deferred debits billed,
            then each queued command's consequences as this connection
            is entitled to see them, then its viewport transitions
            (0x58 exits that start the law-4 reroute clock, 0x3D
            entries), then one status sync per LIVING tank — the
            measured broadcast cadence is every ~2 s for every active
            tank regardless of activity ([[tank-freshness-model]]), and
            the Phase 3 fuel book depends on exactly those quiet
            zero-delta readings to close its accounting blocks — and an
            inventory snapshot when its own tank's counts changed.
        """
        self.world["tick"] += 1
        # No runtime container spawning: the 2026-07-22 "respawn law"
        # was falsified 2026-07-25 (every observed "spawn" was an
        # exposure of a pre-existing container — [[game-economy]]).
        # The world is a static population seeded by sim.world_seed.
        outbox = self._outbox
        self._outbox = TickOutbox(tuple(self._sessions))
        ammo_changed: set[int] = set()
        self.combat.apply_pending_debits()
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
            self._process_command(tank_id, command, outbox, ammo_changed, moved)
        self._queue = []
        for session in self.sessions:
            batch = outbox.batch(session.client_id)
            # A deactivation of a connected tank demotes it to recruit,
            # silently, in the same batch as the 0x41 — three archived
            # demotions, every one at a zero-second gap. The batch is
            # read rather than the ledger being told about ranks: one
            # 0x41 is the fact, and the ledger's job is to produce it,
            # not to interpret it ([[session-state-deglobalisation]]).
            if any(
                message["msg_type"] == 0x41 and message["victim_id"] == session.client_id
                for message in batch
            ):
                session.progression.note_deactivation(self.world, batch)
        self._close_corpse_windows(outbox)
        # Room churn runs BEFORE the viewport diff, so a visitor who
        # lands inside a connection's window is announced by the same
        # membership pass that announces any other arrival
        # ([[session-state-deglobalisation]]).
        arrivals: list[BinaryMessage] = []
        self._churn.advance(self.world, self.terrain, arrivals)
        outbox.broadcast(arrivals)
        for session in self.sessions:
            self._close_session_view(session, outbox.batch(session.client_id))
        # The syncs run in a pass of their own, after every
        # connection's rank has settled: a promotion closed for one
        # connection above is part of the sync every OTHER connection
        # reads of that tank this tick.
        for session in self.sessions:
            self._emit_syncs(session, outbox.batch(session.client_id), ammo_changed)
        return outbox.batches()

    def _close_session_view(self, session: ClientSession, batch: list[BinaryMessage]) -> None:
        """Close one connection's own view of the tick.

        Args:
            session: The connection.
            batch: Its batch this tick (appended).
        """
        session.viewport.emit_transitions(batch)
        # Dynamic-layer refresh is EVENT-driven, never walk-driven:
        # the client's window is static between teleports (autoscroll
        # OFF, [[viewport-shift-protocol]] -- 16+ probed walks drew
        # zero 0x5A), but a ferry or block moving inside the patch
        # grid repaints it (the 2026-07-20 block captures show 0x5A
        # after block operations). An empty patch is not sent.
        #
        # This is NOT a duplicate of a repainting command's own 0x5A,
        # which is what it looks like from the response-shape differ.
        # ``build_update`` returns a DIFF against
        # ``_patched_dynamic_tiles`` and MUTATES it, so a second call
        # in the same tick carries what changed AFTER the first -- the
        # ferry drift and room churn just above. Suppressing it costs
        # the client that information: tried 2026-09-01, and the
        # wrong-pond ferry scenario stopped discovering its ferry.
        refresh = session.viewport.build_update()
        if refresh["entities"]:
            batch.append(refresh)
        # The promotion that ends a recovery window, before the syncs
        # so this tick's bar already reads the restored steady state.
        session.progression.advance(self.world, batch)
        # Awards are granted from the same counters the 0x56 reports,
        # against the thresholds the in-client guide names
        # ([[decoration-encoding]]): 100/200/500 kills, 20/50/100
        # deaths, Major/Colonel/General, 100/200/500 hours. The archive
        # caught exactly one grant — Artax's 500th kill stepping the
        # Tank award to golden on 2026-07-29
        # ([[session-state-deglobalisation]]).
        session.awards.advance(
            self.world["tanks"][session.client_id]["rank"],
            self.combat.destroyed_by(session.client_id),
            self.combat.deactivations_of(session.client_id),
            self.world["tick"] * TICK_RATE_MS // 1000,
            batch,
        )

    def _emit_syncs(
        self, session: ClientSession, batch: list[BinaryMessage], ammo_changed: set[int]
    ) -> None:
        """Close one connection's batch with the tick's syncs.

        Args:
            session: The connection.
            batch: Its batch this tick (appended).
            ammo_changed: Tanks whose counts moved this tick; the
                connection's own tank among them draws a 0x49.
        """
        for tank_id in sorted(self.world["tanks"]):
            if self.world["tanks"][tank_id]["alive"]:
                batch.append(
                    status_sync(
                        tank_id,
                        self.world,
                        tank_id == session.client_id,
                        session.progression.promo_state,
                    )
                )
        if session.client_id in ammo_changed:
            tank = self.world["tanks"][session.client_id]
            batch.append(
                InventoryDict(
                    msg_type=0x49,
                    show=False,
                    alternate=False,
                    counts=list(tank["counts"]),
                    enabled=list(tank["enabled"]),
                )
            )


__all__ = [
    "CORPSE_WINDOW_TICKS",
    "SUPPORTED_KINDS",
    "SimServer",
]
