"""Disproven-belief and incoming-damage queries for one session.

The container-desync latch, the own-mine walk-over window, and the
fuel-confirmed incoming-damage rate instrument. Mixed into
:class:`~tankpit_bot.sniffer.world_service.WorldService`, which owns the
state these annotate.

These were free functions on the ``world_state`` module facade, reading
a process global. They are queries over one session's own belief, so
they belong on the session ([[session-state-deglobalisation]] step 8).
"""

from __future__ import annotations

from tankpit_bot.ledger.damage_book import DamageBookDict, incoming_damage_window
from tankpit_bot.state import WorldStateDict

#: The reactive walk->teleport flip stays live long enough for the next
#: decision to dispatch the teleport approach (a few server windows),
#: then expires so walking resumes per the doctrine.
_OWN_MINE_HIT_FLIP_MS = 6_000


class WorldServiceBeliefsMixin:
    """Belief-disproof and damage-rate queries for one session.

    The attributes below are DECLARATIONS, not assignments: the
    session's ``__init__`` remains their single owner, so this split
    does not move any per-session state.
    """

    world_state: WorldStateDict
    damage_book: DamageBookDict
    container_desync_ms: int
    last_own_mine_hit_ms: int

    def mark_container_desync(self, timestamp_ms: int) -> None:
        """Record a disproven remembered-container belief (code=4 pickup).

        Args:
            timestamp_ms: When the empty-container rejection arrived.
        """
        self.container_desync_ms = timestamp_ms

    def clear_container_desync(self) -> None:
        """Answer a container desync without a scan.

        Used when live coverage already tells the whole story (radar-spend
        economics, s9-4): rescanning ground scanned seconds earlier buys
        nothing, so the disproof is considered answered by the existing
        coverage.
        """
        self.container_desync_ms = 0

    def container_desync_pending(self) -> bool:
        """Check whether a container desync awaits its radar resync.

        Returns:
            True while a code=4 disproof has not yet been answered by a
            radar response (which reconciles the viewport and clears it).
        """
        return self.container_desync_ms > 0

    def recent_own_mine_hit(self, now_ms: int) -> bool:
        """Check whether a walk-over mine hit landed within the flip window.

        User doctrine 2026-07-30: "walk to targets or containers in
        viewport but if we hit a mine teleport to target or container.
        then resume walking within viewport." One window is enough for
        the flipped approach to dispatch; afterwards walking resumes.

        Args:
            now_ms: Current timestamp.

        Returns:
            True while the last own-tile detonation is fresh.
        """
        return now_ms - self.last_own_mine_hit_ms < _OWN_MINE_HIT_FLIP_MS

    def get_incoming_damage_window(self, now_ms: int, window_ms: int) -> tuple[int, int]:
        """Return fuel-confirmed incoming (hits, fuel) in the trailing window.

        The damage-aware engagement break's rate instrument -- reads the
        session damage book ([[bot-behavior-contract]] §3.3), excluding
        shooters the registry lists as DEACTIVATED: a dead attacker
        cannot keep firing, so their hits must not project into the next
        engagement (2026-07-31 arena soak -- a freshly killed enemy's
        rate blocked three healthy follow-up targets as "unwinnable").
        Unknown shooters still count -- a registry gap can never
        under-report live danger.

        Args:
            now_ms: Current wall-clock ms.
            window_ms: Trailing window length in ms.

        Returns:
            ``(hits, fuel)`` confirmed within the window from shooters
            not known to be dead.
        """
        dead_shooter_ids = frozenset(
            tank["tank_id"]
            for tank in self.world_state["tanks"].values()
            if tank["liveness"] == "deactivated"
        )
        return incoming_damage_window(self.damage_book, now_ms, window_ms, dead_shooter_ids)


__all__ = [
    "WorldServiceBeliefsMixin",
]
