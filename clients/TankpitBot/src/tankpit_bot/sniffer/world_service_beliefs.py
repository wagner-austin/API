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
    last_own_gain_ms: int
    mine_reveal_pending_ms: int

    def mark_mine_reveal_pending(self, timestamp_ms: int) -> None:
        """Record that an own-tile mine hit awaits its reveal scan.

        A walk-over detonation proves UNREVEALED hostile mines sit on
        the ground the bot is working (user ruling 2026-08-13, flag 2:
        "why aren't we using radar whenever we get hit by a mine?").
        One scan reveals the field team-scoped ([[walk-mechanics]]),
        turning invisible walk-over hazards into composed-terrain
        blockers the planner routes around and the clearance arms can
        shoot. Cleared by the radar response exactly like the
        container-desync latch — one scan per hit.

        Args:
            timestamp_ms: When the own-tile detonation arrived.
        """
        self.mine_reveal_pending_ms = timestamp_ms

    def mine_reveal_pending(self) -> bool:
        """Check whether an own-mine hit awaits its reveal scan.

        Returns:
            True while the hit has not been answered by a radar
            response.
        """
        return self.mine_reveal_pending_ms > 0

    def record_own_gain(self, timestamp_ms: int) -> None:
        """Record that the wire just announced a gain of OUR OWN.

        Stamped by the fuel-total dispatch when the announced delta is
        positive and by the inventory dispatch when any count rises.
        The code=4 discriminator reads it: a pickup click that GAINED
        something before its ``empty_container`` close drained the
        container itself (a receipt), while a click with no gain hit a
        container that was already empty (a stale belief -- the desync
        latch fires). The old tile-record discriminator could not tell
        the two apart in a co-farmed room: run arterial
        2026-08-13 22:23-27 classified 23 of 23 already-empty pickups
        as "own drain" and the rescan gate never fired all session.

        Args:
            timestamp_ms: When the gain announcement arrived.
        """
        self.last_own_gain_ms = timestamp_ms

    def own_gain_since(self, since_ms: int) -> bool:
        """Check whether an own-gain announcement arrived at/after a moment.

        Args:
            since_ms: Window start (the in-flight action's dispatch).

        Returns:
            True when the last recorded gain is inside the window.
        """
        return self.last_own_gain_ms >= since_ms > 0

    def own_mine_hit_since(self, since_ms: int) -> bool:
        """Check whether an own-tile detonation arrived at/after a moment.

        The collect cant_go discriminator reads it: a code=1 whose
        action window contains a walk-over detonation is the server
        refusing the REMAINDER of an interrupted walk (the partial-walk
        law, [[walk-mechanics]]), not a verdict on the container. Run
        arterial 2026-08-13 22:25:45 (flag 6): the detonation on
        (103,143) halted the walk to equipment at (103,147), the
        cant_go incremented the innocent container's failed_pickups,
        and the replan paid a 60-fuel teleport to different equipment
        four tiles further away.

        Args:
            since_ms: Window start (the in-flight action's dispatch).

        Returns:
            True when the last own-tile detonation is inside the window.
        """
        return self.last_own_mine_hit_ms >= since_ms > 0

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
