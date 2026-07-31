"""Keeping the match's figures, so the loop can keep the match.

Every figure the report carries used to be a local variable in the campaign
loop -- around forty of them, each needing an initialisation, an update and a
slot in the report literal, all threaded by hand through one function. Adding a
figure meant touching four places, and the loop's own logic was the thing
buried under them ([[policy-loop]]).

This class is those accumulators with a name. It decides nothing: it reads one
observation at a time, remembers the firsts, lasts and peaks the report is
defined in terms of, and assembles the :class:`~rw_bot.policy.match_report.MatchReport`
at the end. What each figure is *for* is documented on the report itself,
which stays the single home of that reasoning.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.mechanics.combat_profile import CombatProfile
from rw_bot.policy.combat import engageable
from rw_bot.policy.economy import count_extractors
from rw_bot.policy.ledger import Outlay, Reach
from rw_bot.policy.match_report import MatchReport
from rw_bot.policy.scoreboard import (
    best_rival,
    composition_of,
    deepest_dip,
    local_player,
    standing_of,
    worth_of,
)
from rw_bot.policy.verdict import GRADE_SURVIVED, eliminated, grade
from rw_bot.wire.state import Entity, Sample


class Scorekeeper:
    """Accumulates the report's figures across a match's observations.

    Attributes:
        samples_seen: World samples read so far, which is what the loop's
            budget counts.
        verdict: The engine's verdict as of the last observation, which is the
            only thing that ends a match early ([[policy-verdict]]).
        visible_now: Engine ids of the hostiles visible on the last
            observation, for the ``killed`` figure's fog caveat.
        army_end: Units able to fight at the last observation, as the recorder
            wants it per sample.
        targets_end: Hostile entities visible at the last observation.
        extractors_end: Finished extractors at the last observation.
        worth_end: Everything the player holds, at the last observation.
        rival_worth_end: The strongest hostile player's total, at the last
            observation.
    """

    def __init__(
        self,
        catalogue: Mapping[str, UnitStats],
        profiles: Mapping[str, CombatProfile],
    ) -> None:
        """Open a scorekeeper.

        Args:
            catalogue: Unit stats by type name, for telling a building from a
                unit.
            profiles: Combat profiles by type name, for the engageable count.
        """
        self._catalogue = catalogue
        self._profiles = profiles
        self.samples_seen = 0
        self.verdict = GRADE_SURVIVED
        self.visible_now: set[int] = set()
        self.army_end = 0
        self.targets_end = 0
        self.extractors_end = 0
        self.worth_end = 0
        self.rival_worth_end = 0
        self._first_frame = 0
        self._first_clock = 0
        self._frames_elapsed = 0
        self._clock_elapsed_ms = 0
        self._credits_at_end = 0
        self._army_start = 0
        self._targets_seen = 0
        self._engageable_end = 0
        self._extractors_start = 0
        self._army_value_start = 0
        self._army_value_end = 0
        self._worth_start = 0
        self._rival_worth_start = 0
        self._rival_worth_peak = 0
        self._rival_worth_drawdown = 0
        self._rival_peaks: dict[int, int] = {}
        self._workers_end = 0
        self._composition_end: tuple[tuple[str, int], ...] = ()
        self._standing_end: tuple[tuple[str, int], ...] = ()
        self._enemy_types_end: tuple[tuple[str, int], ...] = ()
        self._income_end = 0
        self._players_start = 0
        self._players_end = 0

    def observe(
        self,
        sample: Sample,
        army: Sequence[Entity],
        targets: Sequence[Entity],
        workers: int,
    ) -> None:
        """Read one observation's figures.

        The army and the targets are handed in rather than re-derived, because
        the loop already gathered both for the combat pass and two readings of
        one roster could disagree.

        Args:
            sample: One observation of the world.
            army: Units able to fight, as the combat pass found them.
            targets: The hostile entities visible.
            workers: Builders owned, as the workforce counts them.
        """
        self.samples_seen += 1
        if self.samples_seen == 1:
            self._first_frame = sample["frame"]
            self._first_clock = sample["clock_ms"]
        self._frames_elapsed = sample["frame"] - self._first_frame
        self._clock_elapsed_ms = sample["clock_ms"] - self._first_clock
        self._credits_at_end = sample["credits"]

        self.army_end = len(army)
        self.targets_end = len(targets)
        self._engageable_end = len(engageable(self._profiles, army, targets))
        self.visible_now = {entity["unit_id"] for entity in targets}
        self.extractors_end = count_extractors(sample)
        self._players_end = sample["players_left"]

        local = local_player(sample)
        self._army_value_end = 0 if local is None else local["army_value"]
        self.worth_end = worth_of(local)
        self._income_end = 0 if local is None else local["income"]
        self.rival_worth_end = best_rival(sample)
        self._rival_worth_peak = max(self._rival_worth_peak, self.rival_worth_end)
        self._rival_worth_drawdown = max(
            self._rival_worth_drawdown, deepest_dip(self._rival_peaks, sample)
        )
        self._workers_end = workers
        self._composition_end = composition_of(army)
        self._standing_end = composition_of(standing_of(sample, self._catalogue))
        self._enemy_types_end = composition_of(targets)
        self.verdict = grade(sample)

        if self.samples_seen == 1:
            self._army_start = self.army_end
            self._targets_seen = self.targets_end
            self._extractors_start = self.extractors_end
            self._players_start = self._players_end
            self._army_value_start = self._army_value_end
            self._worth_start = self.worth_end
            self._rival_worth_start = self.rival_worth_end

    def report(
        self,
        *,
        completed: int,
        planned: int,
        build_orders: int,
        build_outcome: str,
        build_reason: str,
        produced: int,
        expanded: int,
        expanded_factories: int,
        expand_reason: str,
        attack_orders: int,
        rallied: int,
        intercepts: int,
        sightings: int,
        raids: int,
        marches: int,
        killed: int,
        refused_claims: int,
        outlays: tuple[Outlay, ...],
        reaches: tuple[Reach, ...],
        outcome: str,
    ) -> MatchReport:
        """Assemble the match report from what was observed and what was done.

        The split between the two argument sources is the split this class
        makes: everything read from the world is already here, and everything
        the policies *did* -- orders sent, claims made, reasons given -- belongs
        to the objects that did it and arrives as arguments.

        Args:
            completed: Plan entries standing at the end.
            planned: Plan entries asked for.
            build_orders: Orders the plan issued.
            build_outcome: How the plan stands.
            build_reason: The plan's own words for its last decision.
            produced: Reinforcements ordered.
            expanded: Structures the economy ordered.
            expanded_factories: How many of those were producers.
            expand_reason: The economy's own words for its last decision.
            attack_orders: Attack orders issued.
            rallied: Move orders issued to gather the reserve.
            intercepts: Guard engagements issued against raiders.
            sightings: Hostile sightings the intel memory recorded.
            raids: Income objectives the raid party assaulted.
            marches: Outbound orders sent to raid party members.
            killed: Targets ordered against that are no longer visible.
            refused_claims: Credit claims the budget turned down.
            outlays: What each purpose asked for and got.
            reaches: How often each spender was arrived at.
            outcome: Why the loop stopped.

        Returns:
            The match report.
        """
        return MatchReport(
            grade=self.verdict,
            completed=completed,
            planned=planned,
            build_orders=build_orders,
            build_outcome=build_outcome,
            build_reason=build_reason,
            produced=produced,
            expanded=expanded,
            expanded_factories=expanded_factories,
            expand_reason=expand_reason,
            extractors_start=self._extractors_start,
            extractors_end=self.extractors_end,
            attack_orders=attack_orders,
            rallied=rallied,
            intercepts=intercepts,
            sightings=sightings,
            raids=raids,
            marches=marches,
            army_start=self._army_start,
            army_end=self.army_end,
            targets_seen=self._targets_seen,
            targets_end=self.targets_end,
            engageable_end=self._engageable_end,
            killed=killed,
            army_value_start=self._army_value_start,
            army_value_end=self._army_value_end,
            worth_start=self._worth_start,
            worth_end=self.worth_end,
            rival_worth_start=self._rival_worth_start,
            rival_worth_end=self.rival_worth_end,
            rival_worth_peak=self._rival_worth_peak,
            rival_worth_drawdown=self._rival_worth_drawdown,
            workers_end=self._workers_end,
            standing_end=self._standing_end,
            composition_end=self._composition_end,
            enemy_types_end=self._enemy_types_end,
            income_end=self._income_end,
            players_start=self._players_start,
            players_end=self._players_end,
            eliminated=eliminated(self._players_start, self._players_end),
            refused_claims=refused_claims,
            outlays=outlays,
            reaches=reaches,
            samples_seen=self.samples_seen,
            frames_elapsed=self._frames_elapsed,
            clock_elapsed_ms=self._clock_elapsed_ms,
            credits_at_end=self._credits_at_end,
            outcome=outcome,
        )


__all__ = ["Scorekeeper"]
