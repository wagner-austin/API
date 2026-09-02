"""Combat command routing and the corpse window, for the simulator.

The shoot family and the clock it drives, mixed into
:class:`~tankpit_bot.sim.server.SimServer` alongside
:class:`~tankpit_bot.sim.server_move.SimServerMoveMixin`. Both routers
have the same shape and it is the shape the whole emission side now
follows: RESOLVE the action once against the world, BOOK what the field
must remember, then NARRATE the outcome for one connection
([[recipient-policy]]).
"""

from __future__ import annotations

from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.protocol.commands import TICK_RATE_MS
from tankpit_bot.protocol.types import BinaryMessage
from tankpit_bot.sim.bot_policy import reactivate_practice_bot
from tankpit_bot.sim.client_session import ClientSession
from tankpit_bot.sim.combat import process_shot
from tankpit_bot.sim.combat_clock import CombatClock
from tankpit_bot.sim.commands import ClientCommandDict
from tankpit_bot.sim.narrate import narrate_corpse_removals, narrate_shot
from tankpit_bot.sim.world import SimWorldDict


class SimServerCombatMixin:
    """Shoot routing and corpse-window closure for the simulator.

    The attributes below are DECLARATIONS, not assignments: the
    server's ``__init__`` remains their single owner.
    """

    world: SimWorldDict
    terrain: TerrainMapProtocol
    session: ClientSession
    combat: CombatClock
    _roster_ids: frozenset[int]

    def _process_shoot_command(
        self,
        tank_id: int,
        command: ClientCommandDict,
        messages: list[BinaryMessage],
        ammo_changed: set[int],
        moved: set[int],
    ) -> None:
        """Route one shoot command: resolve once, then narrate.

        The shot's entire world effect — ballistics, damage, the ammo
        debit, the mine cascade and the radar-zero kill reward — lands
        in the single ``process_shot`` call. What happens afterwards
        only BOOKS or REPORTS it: the firing cost goes to the clock for
        next tick (measured charge latency), a deactivation opens the
        corpse window and scores the kill, and the narrator turns the
        outcome into this connection's wire. Nothing below mutates the
        world, which is what makes a second connection a second
        ``narrate_shot`` call rather than a second shot.

        A victim whose shields absorbed the hit has its counts moved,
        and so does a killer paid the mercy bundle — both feed the
        end-of-tick 0x49 for whichever of them is the client.

        Args:
            tank_id: The firing tank.
            command: The queued shoot command.
            messages: This tick's outgoing batch (appended).
            ammo_changed: Accumulator of tanks whose counts moved.
            moved: Tanks that relocated earlier this tick (drives the
                homing selection).
        """
        removed_tick = self.session.viewport.removed_at.get(command["target_id"])
        departed_age_ms = (
            None if removed_tick is None else (self.world["tick"] - removed_tick) * TICK_RATE_MS
        )
        outcome = process_shot(
            self.world,
            self.terrain,
            tank_id,
            command["x"],
            command["y"],
            frozenset(moved),
            command["target_id"],
            departed_age_ms,
        )
        self.combat.defer_debit(tank_id, outcome["shooter_debit"])
        if outcome["shields_consumed"] > 0 and outcome["victim_id"] is not None:
            ammo_changed.add(outcome["victim_id"])
        if outcome["victim_deactivated"] and outcome["victim_id"] is not None:
            self.combat.record_deactivation(tank_id, outcome["victim_id"])
        if outcome["mercy"] is not None:
            ammo_changed.add(tank_id)
        messages.extend(narrate_shot(outcome, self.session.client_id))

    def _close_corpse_windows(self, messages: list[BinaryMessage]) -> None:
        """Emit the 0x58 of every corpse whose 22 s window came due.

        Roster bots come back the same tick their corpse clears: same
        id, full fuel, respawned FAR from the corpse (archive-mined
        2026-07-24, [[enemy-bot-behavior]]). The reactivation draws no
        message of its own — the end-of-tick viewport diff announces it
        if the landing is in view, and the sync loop resumes the bot's
        tier-3 cadence either way — so the removals are narrated first
        and in one pass, exactly as the wire carried them before the
        clock and the narrator were separated.

        Args:
            messages: This tick's outgoing batch (appended).
        """
        cleared = self.combat.expire_corpses()
        messages.extend(narrate_corpse_removals(cleared))
        for tank_id in cleared:
            if tank_id in self._roster_ids:
                reactivate_practice_bot(self.world, self.terrain, tank_id)


__all__ = ["SimServerCombatMixin"]
