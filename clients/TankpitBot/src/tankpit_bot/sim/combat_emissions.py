"""Combat-side wire emission: shots, kill rewards, and the fuel ledger.

Owns the two combat clocks the server defers across ticks: the
charge-latency debit list (shooter firing costs bill one tick AFTER
the shot — measured) and the corpse window (a killed tank's 0x58
TankRemove arrives EXACTLY 22 s after its 0x41 — corpus-swept
2026-07-22: 37 kill->remove pairs, min = median = 22.0 s; 11 ticks
at the 2 s cadence).
"""

from __future__ import annotations

from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.container.types import MineDetonationDict
from tankpit_bot.protocol.commands import TICK_RATE_MS
from tankpit_bot.protocol.types import (
    BinaryMessage,
    DeactivationDict,
    EquipmentGainDict,
    ShootEventDict,
    TankRemoveDict,
)
from tankpit_bot.sim.combat import process_shot
from tankpit_bot.sim.commands import ClientCommandDict
from tankpit_bot.sim.equipment import MERCY_BUNDLE, RADAR_SLOT, kill_grants_mercy
from tankpit_bot.sim.world import SimWorldDict

CORPSE_WINDOW_TICKS = 11


class CombatLedger:
    """Owns the deferred firing debits and the corpse-window clocks.

    It also keeps the two kill counters the 0x56 Statistics answer
    reports, because this is the one place a deactivation resolves —
    counting them anywhere else would mean a second reading of the
    same event ([[session-state-deglobalisation]]).
    """

    def __init__(
        self,
        world: SimWorldDict,
        terrain: TerrainMapProtocol,
        client_id: int,
    ) -> None:
        """Bind the ledger to a world and the client tank.

        Args:
            world: Simulated world (read and mutated on billing).
            terrain: Static terrain for shot resolution.
            client_id: The connected client's tank id.
        """
        self._world = world
        self._terrain = terrain
        self._client_id = client_id
        self._pending_debits: list[tuple[int, int]] = []
        self._died_at: dict[int, int] = {}
        self.client_destroyed = 0
        """Tanks the client has deactivated this session (0x56 ``destroyed``)."""
        self.client_deactivated = 0
        """Times the client has been deactivated this session (0x56 ``deactivated``)."""

    def apply_pending_debits(self) -> None:
        """Bill last tick's firing costs (measured charge latency)."""
        for tank_id, debit in self._pending_debits:
            tank = self._world["tanks"][tank_id]
            tank["fuel"] = max(0, tank["fuel"] - debit)
        self._pending_debits = []

    def emit_shot(
        self,
        shooter_team: int,
        outcome_messages: list[BinaryMessage],
        ammo_changed: set[int],
        tank_id: int,
        command: ClientCommandDict,
        moved: frozenset[int],
        removed_tick: int | None,
    ) -> None:
        """Process one shoot command and emit its wire consequences.

        Args:
            shooter_team: The shooter's team (for the 0x53 echo).
            outcome_messages: This tick's outgoing batch (appended).
            ammo_changed: Accumulator of tanks whose counts moved.
            tank_id: The firing tank.
            command: The shoot command.
            moved: Tanks that moved earlier this tick.
            removed_tick: Tick of the target's living viewport exit
                (the law-4 reroute clock), or None when never removed.
        """
        target_id = command["target_id"]
        departed_age_ms = (
            None if removed_tick is None else (self._world["tick"] - removed_tick) * TICK_RATE_MS
        )
        outcome = process_shot(
            self._world,
            self._terrain,
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
        # Firing-cost ammo consumption does NOT snapshot: the archive's
        # 11,051 shot windows are 92.4% a bare 0x53 echo — the real
        # server never answers a shot with 0x49 (response-shape differ
        # 2026-08-01). Counts re-sync on the next 0x49-bearing event
        # (radar extra, equipment gain), exactly as live.
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
            self._maybe_emit_kill_mercy_bundle(tank_id, outcome_messages, ammo_changed)
            self._died_at[outcome["victim_id"]] = self._world["tick"]
            if tank_id == self._client_id:
                self.client_destroyed += 1
            if outcome["victim_id"] == self._client_id:
                self.client_deactivated += 1

    def _maybe_emit_kill_mercy_bundle(
        self,
        killer_id: int,
        messages: list[BinaryMessage],
        ammo_changed: set[int],
    ) -> None:
        """Apply the radar-zero kill reward (archive-cracked 2026-07-22).

        A kill scored while the killer's extra-radar count is ZERO
        grants a silent (``show_message=False``) multi-slot bundle —
        deterministic in the corpus: 5/5 radar-zero kills granted,
        0/254 kills at radar > 0 granted, no exceptions. Measured
        amounts: dual +1..4, homing exactly +1, radar +1..2, and the
        bundle may OVERFILL past the 25 cap (one sample landed dual
        at 26). The sim grants the deterministic medians (+2/+1/+1).
        Per-recipient: only the client's own bundle rides the wire.

        Args:
            killer_id: The killing tank.
            messages: This tick's outgoing batch (appended).
            ammo_changed: Accumulator of tanks whose counts moved.
        """
        killer = self._world["tanks"][killer_id]
        if not kill_grants_mercy(killer["counts"][RADAR_SLOT]):
            return
        gained = list(MERCY_BUNDLE)
        for slot, amount in enumerate(gained):
            killer["counts"][slot] += amount
        ammo_changed.add(killer_id)
        if killer_id == self._client_id:
            messages.append(
                EquipmentGainDict(msg_type=0x67, show_message=False, gained=list(gained))
            )

    def expire_corpses(self, messages: list[BinaryMessage]) -> list[int]:
        """Close elapsed corpse windows and emit their 0x58 removals.

        The corpse window closes: 0x58 removes the corpse exactly
        22 s after the 0x41. NOT a departure — the law-4 reroute
        clock only runs for living exits.

        Args:
            messages: This tick's outgoing batch (appended).

        Returns:
            The tank ids whose corpses cleared this tick, ascending.
        """
        expired: list[int] = []
        for tank_id in sorted(self._died_at):
            if self._world["tick"] - self._died_at[tank_id] >= CORPSE_WINDOW_TICKS:
                messages.append(TankRemoveDict(msg_type=0x58, tank_id=tank_id))
                del self._died_at[tank_id]
                expired.append(tank_id)
        return expired


__all__ = [
    "CORPSE_WINDOW_TICKS",
    "CombatLedger",
]
