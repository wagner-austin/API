"""The two combat clocks the field runs across ticks, and the kill book.

Owns the charge-latency debit list (shooter firing costs bill one tick
AFTER the shot — measured) and the corpse window (a killed tank's 0x58
TankRemove arrives EXACTLY 22 s after its 0x41 — corpus-swept
2026-07-22: 37 kill->remove pairs, min = median = 22.0 s; 11 ticks at
the 2 s cadence).

This is FIELD state, not connection state, and that placement is the
correction this module was split out to make. Its predecessor lived on
:class:`~tankpit_bot.sim.client_session.ClientSession` because the sim
had exactly one connection, so "the client's corpse clocks" and "the
room's corpse clocks" were the same object. They are not the same fact:
a corpse clears once for the whole field, a firing cost is billed once
against the shooter's fuel, and a kill is scored once. Fanning a
per-connection copy out across N connections would clear each corpse N
times and bill each shot N times.

The kill counters live here for the same reason, keyed by tank rather
than fixed to one client: a deactivation is one event, and the 0x56
Statistics answer merely reports the asking tank's row of a book the
field keeps ([[session-state-deglobalisation]]).
"""

from __future__ import annotations

from collections import Counter

from tankpit_bot.sim.world import SimWorldDict

CORPSE_WINDOW_TICKS = 11


class CombatClock:
    """The field's deferred firing debits, corpse windows, and kill book.

    Nothing here emits. The clock records what happened and reports
    what has come due; :mod:`tankpit_bot.sim.narrate.combat` turns
    those facts into wire messages for a given connection.
    """

    def __init__(self, world: SimWorldDict) -> None:
        """Bind the clocks to a world.

        Args:
            world: Simulated world (read for the tick, mutated when
                deferred firing costs are billed).
        """
        self._world = world
        self._pending_debits: list[tuple[int, int]] = []
        self._died_at: dict[int, int] = {}
        self._destroyed: Counter[int] = Counter()
        self._deactivated: Counter[int] = Counter()

    def defer_debit(self, tank_id: int, debit: int) -> None:
        """Book one shot's firing cost against the NEXT tick.

        Args:
            tank_id: The firing tank.
            debit: The weapon's firing cost in fuel.
        """
        self._pending_debits.append((tank_id, debit))

    def apply_pending_debits(self) -> None:
        """Bill last tick's firing costs (measured charge latency)."""
        for tank_id, debit in self._pending_debits:
            tank = self._world["tanks"][tank_id]
            tank["fuel"] = max(0, tank["fuel"] - debit)
        self._pending_debits = []

    def record_deactivation(self, killer_id: int, victim_id: int) -> None:
        """Open a corpse window and score the kill in one entry.

        The two are one event and are recorded together so no caller
        can post half of it.

        Args:
            killer_id: The tank credited with the deactivation.
            victim_id: The tank deactivated.
        """
        self._died_at[victim_id] = self._world["tick"]
        self._destroyed[killer_id] += 1
        self._deactivated[victim_id] += 1

    def expire_corpses(self) -> list[int]:
        """Close every corpse window that has come due this tick.

        The window closes 22 s after the 0x41. NOT a departure — the
        law-4 reroute clock only runs for LIVING viewport exits, which
        the viewport tracker owns.

        Returns:
            The tank ids whose corpses cleared this tick, ascending.
            Each is returned exactly once; the window is forgotten.
        """
        expired: list[int] = []
        for tank_id in sorted(self._died_at):
            if self._world["tick"] - self._died_at[tank_id] >= CORPSE_WINDOW_TICKS:
                del self._died_at[tank_id]
                expired.append(tank_id)
        return expired

    def destroyed_by(self, tank_id: int) -> int:
        """Tanks this one has deactivated this session (0x56 ``destroyed``).

        Args:
            tank_id: The tank whose row is read.

        Returns:
            The kill count, zero for a tank that has scored none.
        """
        return self._destroyed[tank_id]

    def deactivations_of(self, tank_id: int) -> int:
        """Times this tank has been deactivated (0x56 ``deactivated``).

        Args:
            tank_id: The tank whose row is read.

        Returns:
            The death count, zero for a tank that has never died.
        """
        return self._deactivated[tank_id]


__all__ = [
    "CORPSE_WINDOW_TICKS",
    "CombatClock",
]
