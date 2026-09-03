"""The four commands that ask about the CONNECTION, not the world.

Fifteen client commands reach the tick processor. Four of them change
nothing and are addressed to the connection itself — "let me in",
"what am I carrying", "how have I done", and the keep-alive — and they
share the property that makes them different from every other command:
the reply is a function of ONE connection's own state, so a second
connection asking the same question gets a different answer and
neither touches the world.

Split from the world router 2026-09-03. Naming the two commands a real
browser sends and ours does not took that router past the branch
ceiling, which is what made the boundary visible; the boundary itself
was already there.
"""

from __future__ import annotations

from tankpit_bot.protocol.types import (
    BinaryMessage,
    EquipmentToggleDict,
    InventoryDict,
    SyncDict,
)
from tankpit_bot.sim.client_session import ClientSession
from tankpit_bot.sim.combat_clock import CombatClock
from tankpit_bot.sim.server_sessions import SimServerSessionsMixin
from tankpit_bot.sim.wire_statements import (
    full_status_statement,
    identity_statement,
    position_statement,
    statistics_statement,
    status_sync,
)
from tankpit_bot.sim.world import SimWorldDict


class SimServerQueriesMixin(SimServerSessionsMixin):
    """Connection-scoped question answering for the simulator.

    The attributes below are DECLARATIONS, not assignments: the
    server's ``__init__`` remains their single owner. ``handshake`` is
    declared too, because the join burst is the answer to one of these
    questions and the server owns the burst.
    """

    world: SimWorldDict
    session: ClientSession
    combat: CombatClock

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

    def _answer_connection_query(
        self,
        tank_id: int,
        kind: str,
        messages: list[BinaryMessage],
    ) -> bool:
        """Answer the commands that ask about the CONNECTION, not the world.

        Four of the fifteen client commands change nothing and are
        addressed to the connection itself — "let me in", "what am I
        carrying", "how have I done", and the keep-alive. They are
        answered here, together, because they share the property that
        makes them different from every other command: the reply is a
        function of one connection's own state, so a second connection
        asking the same question gets a different answer and neither
        touches the world.

        Split out of the world router 2026-09-03 when naming the two
        commands a real client sends took its branch count past the
        ceiling — the boundary was already there, the count only made
        it visible.

        Args:
            tank_id: The commanding tank.
            kind: The command kind.
            messages: This tick's outgoing batch (appended).

        Returns:
            True when ``kind`` was a connection query and is now
            answered; False to let the world router try it.
        """
        if kind == "statistics":
            # Per-connection, like every other answer: the statistics
            # of the tank that asked, and only to that tank.
            if tank_id == self.session.client_id:
                messages.append(
                    statistics_statement(
                        self.world["tick"],
                        self.combat.destroyed_by(self.session.client_id),
                        self.combat.deactivations_of(self.session.client_id),
                    )
                )
            return True
        if kind == "enter_game":
            # THE JOIN BURST IS AN ANSWER, NOT A PUSH. Measured over
            # 343 archived sends (2026-09-03): every one is answered,
            # and the self-caused tokens are 49 x2, 5A and 3Dself per
            # send — the tail of the burst ``handshake`` already
            # builds, whose full shape was measured 340/340
            # ([[recipient-policy]]).
            #
            # The sim pushed that burst unprompted at connect because
            # OUR bot never sends this command: ``enter_game()`` sat
            # in two production classes with zero callers while the
            # bot joined through the lobby's ``join_room`` instead. A
            # real client asks, so the server now answers.
            if tank_id == self.session.client_id:
                messages.extend(self.handshake())
            return True
        if kind == "inventory":
            # The 'i' key. Four archived sends, every one answered
            # with a 0x49 — thin, but the command's own name and its
            # answer agree, and the snapshot builder already exists.
            if tank_id == self.session.client_id:
                client = self.world["tanks"][tank_id]
                messages.append(
                    InventoryDict(
                        msg_type=0x49,
                        show=True,
                        alternate=False,
                        counts=list(client["counts"]),
                        enabled=list(client["enabled"]),
                    )
                )
            return True
        # THE HEARTBEAT DRAWS SILENCE, and this is the one query
        # answered by RETURNING rather than by appending. Measured over
        # 11,871 archived sends (2026-09-02): the real server answers
        # none of them — 9,746 windows are wholly silent and every
        # self-caused token in the rest belongs to another command
        # whose answer arrived late. So the keep-alive is not
        # "ignored"; it is answered correctly, with nothing, and
        # saying so as the return value keeps the law visible instead
        # of hiding it in an empty branch.
        #
        # Before it was handled at all, a heartbeat never reached this
        # router: ``queue_command`` raised ``SimError`` on any kind
        # outside ``SUPPORTED_KINDS``, so the first one from a REAL
        # client — one per tick, forever — took the server down. Our
        # bot never sends one, which is exactly why a sim built
        # against our bot never met it ([[client-commands]]).
        return kind == "keepalive"
