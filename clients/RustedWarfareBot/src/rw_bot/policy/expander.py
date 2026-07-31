"""Turning the economy's decisions into orders, in the order the policy says.

The decisions themselves are pure and live elsewhere -- income and throughput in
:mod:`rw_bot.policy.economy`, cover in :mod:`rw_bot.policy.defence`. What lives
here is the part that cannot be pure: which of them is even *asked* on this
observation, which worker each may use, and whether the credits are there.

**The order between them is the policy**, and it is expressed by call order
rather than by weights attached to each claim, because burying it in numbers
would make it unreadable ([[policy-budget]]). Reading down :meth:`Expander.step`
is reading the strategy.

Every stage records that it was reached whether or not it acted. Without that
the chain cannot be read from outside at all: a stage that declined three
thousand times and a stage nobody asked leave the same trace, which is none --
and four separate defence experiments were judged on exactly that ambiguity
before anyone noticed it had fired three times in twelve matches
([[policy-holding-ground]]).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.mechanics.combat_profile import CombatProfile
from rw_bot.policy.budget import Budget
from rw_bot.policy.defence import AA_TURRET_TYPE, TURRET_TYPE, expand_defence
from rw_bot.policy.economy import (
    FACTORY_TYPE,
    Expansion,
    expand_economy,
    expand_production,
    waiting,
)
from rw_bot.policy.ledger import Reaches
from rw_bot.policy.workforce import Workforce
from rw_bot.wire.command import BuildOrder, build_order
from rw_bot.wire.state import Entity, Sample

#: Free workers required before one may be diverted to throughput.
#:
#: Two, so expansion always keeps one. In the opening only the starting builder
#: exists, and withholding the diversion there is what the earlier attempt at
#: this lacked: with one builder every factory placed is an extractor not
#: placed, and that arm wiped three matches of six ([[policy-production]]).
_SPARE_WORKER_FLOOR = 2

#: Extractors below which expansion outranks replacing a loss.
#:
#: **Measured, not chosen.** Across 46 duels, final income at or above 50
#: credits a second won 36 matches of 36; at or below 38 it failed 6 of 7. Base
#: income is 18 and an extractor pays 8, so 50/s is exactly four extractors
#: ([[policy-holding-ground]]).
#:
#: Below that the player does not have an economy, and an army bought without
#: one is bought once: at Very Hard the bot spent roughly 65,000 credits of
#: which **2,800 reached the economy**, produced 129 units, and ended with two
#: alive and 26 credits a second. Above it the army resumes its priority,
#: because by then there is an income to defend.
ECONOMY_FLOOR = 4


class Expander:
    """Turns economy decisions into orders, and remembers what it already asked.

    The decisions are :func:`~rw_bot.policy.economy.expand_production` and
    :func:`~rw_bot.policy.economy.expand_economy` and both stay pure. What lives
    here is the part that cannot: whether an order already sent is worth sending
    again. Those are different questions, and keeping them apart is what lets the
    choice of pool be tested without a clock.


    **It keeps no memory of its own.** It used to hold the last site it asked
    for and suppress a repeat for :data:`EXPAND_RETRY_SAMPLES` observations,
    which was a second retry clock beside the workforce's -- two counters for one
    question, running in series, so a lost order took twice the window to be
    reissued. A worker it is handed is free by definition; a worker it has just
    ordered is marked busy by the workforce until the structure goes up or the
    order is presumed lost ([[policy-loop]]).

    Attributes:
        count: Expansion orders sent.
        reason: The economy's own words for its most recent decision.
    """

    def __init__(
        self,
        catalogue: Mapping[str, UnitStats],
        profiles: Mapping[str, CombatProfile],
        enabled: bool,
        aa_cover: bool = False,
        cover: bool = True,
    ) -> None:
        """Open an expander.

        Args:
            catalogue: Unit stats by type name, for prices and immobility.
            profiles: Combat profiles by type name, for the threat filter.
            enabled: Whether expansion is being played at all. False is the
                control arm of the A/B that measures whether it helps, and the
                mode a probe uses when it wants the economy held still.
            aa_cover: Whether an anti-air turret joins the cover once the
                opponent has shown aircraft. Off is the behaviour every
                measurement so far was taken under -- a defence that cannot
                touch an aircraft at all ([[policy-holding-ground]]).
            cover: Whether turrets are bought at all. The on-vs-off A/B
                became possible only when siting made them land
                ([[policy-holding-ground]]).
        """
        self.count = 0
        self.factories = 0
        self.enabled = enabled
        self.aa_cover = aa_cover
        self.cover = cover
        self.reason = "no sample seen yet" if enabled else "expansion disabled"
        # Which stage was even arrived at, per observation. The final `reason`
        # above is one sentence for a whole match and cannot tell "defence
        # declined three thousand times" from "defence was asked three times" --
        # opposite diagnoses that a refusal count reads identically
        # ([[policy-economy]]).
        self.reaches = Reaches()
        self._catalogue = catalogue
        self._profiles = profiles
        # Latched, not sampled: aircraft leave the viewport and come back, and
        # AA that stands down between sorties is AA that is never finished
        # when the sortie arrives.
        self._air_seen = False

    def step(
        self,
        sample: Sample,
        budget: Budget,
        free: Sequence[Entity],
        plan_holds_worker: int,
        wanted: Sequence[str],
        workforce: Workforce,
        plan_wants_worker: bool = False,
    ) -> tuple[BuildOrder, ...]:
        """Ask the economy what to do about this sample.

        **Income before throughput, because income compounds and throughput
        does not.** The order was the other way round first, on the reasoning
        that another extractor earns credits the player already cannot spend.
        Measured over one seed, that reasoning was wrong: buying factories first
        took the builder away from pools, and 4 extractors with 3 factories
        produced 62 units and an army worth 6,450, against 9 extractors with 1
        factory producing 28 units and an army worth 8,200. The bank is not the
        waste -- it is the symptom of an economy ahead of what the army can
        usefully spend, and spending it faster from a weaker economy just feeds
        more units in to die ([[policy-production]]).

        So the pool is taken while any pool is worth taking, and throughput is
        what the surplus buys once the map has nothing left to claim.

        Both are unprotected claims. Expansion is investment: it pays back over
        the rest of the match, and the reserve exists precisely so it cannot
        take the credits that replace a loss now.

        Args:
            sample: One observation of the world.
            budget: The tick's credits, already claimed against by the plan and
                by production.
            free: Workers not already carrying out an order, minus any the plan
                has taken this observation. Availability has one owner now, so
                this class no longer decides it ([[policy-loop]]).
            plan_holds_worker: The worker the opening plan took this
                observation, zero when it took none. The plan claims first, so
                what it took is not free -- but only *it* is not free.

                **This was a boolean and the boolean cost most of the economy.**
                Any worker the plan held stood the whole expander down, so with
                six workers a match spent 572 of 800 samples buying nothing at
                all: income, defence and throughput were not declining on those
                observations, they were never asked ([[policy-economy]]). Naming
                the one worker keeps the two spenders off each other -- which is
                the whole reason the gate exists ([[policy-loop]]) -- while the
                rest of the workforce goes on working.
            wanted: Type names production is trying to make, which is what
                decides whether throughput is actually short.
            plan_wants_worker: Whether the plan is waiting for the next free
                worker, in which case the expander stands down so that worker
                is the plan's rather than the economy's ([[policy-loop]]).
            workforce: Told what each worker was sent to build, so the next
                observation can see it working.

        Returns:
            The build order to send, or nothing when the economy declined to
            spend. At most one, because a worker carries out one order at a
            time and a second would replace the first.
        """
        if not self.enabled:
            self.reaches.reached("disabled", False, "expansion disabled")
            return ()
        if plan_wants_worker:
            # The plan's worker priority, matching the credit priority it has
            # always had. Every capable worker is busy and the plan is waiting
            # for whichever frees first; an expander that took that worker
            # would starve the plan forever -- measured at Hard as 126 turrets,
            # no factory, no army, and 1 win of 10 where the same doctrine had
            # won 10 of 12 (log: 2026-07-31). Standing down entirely is
            # correct here in the way it was wrong for plan-holds-only-worker:
            # that gate had five other workers to keep using; this one is, by
            # its own definition, the state where there is nothing free to
            # keep using.
            self.reaches.reached(
                "plan-first-in-line",
                False,
                "the plan is waiting for the next free worker",
            )
            self.reason = "the plan is waiting for the next free worker"
            return ()
        self._note_air(sample)
        # The plan's worker, and only the plan's worker. Filtering rather than
        # returning is the whole of the fix: the two spenders still never order
        # the same unit, and the other five workers stay available.
        any_free = bool(free)
        free = tuple(worker for worker in free if worker["unit_id"] != plan_holds_worker)
        if not free:
            # Two states that used to read alike and call for opposite work: an
            # opening with one builder, where standing down is correct and
            # temporary, against a workforce that is entirely committed. The
            # first is the position every match starts in ([[policy-economy]]).
            if any_free:
                self.reason = "the opening plan is using the only free worker"
                self.reaches.reached("plan-holds-only-worker", False, self.reason)
                return ()
            self.reason = "every worker is already building something"
            self.reaches.reached("no-free-worker", False, self.reason)
            return ()

        # **Throughput takes one worker, and only when income cannot be spent.**
        # Reordering the whole chain to put throughput first was tried and was
        # the worst arm measured: three wiped and three defeated, expansion
        # collapsing from 307-509 orders to 2-6. The defect was not the
        # priority but the arithmetic behind it -- there was **one** builder, so
        # every factory it placed was an extractor it did not
        # ([[policy-production]]).
        #
        # That objection is now answered by the roster rather than by argument.
        # A wanted builder joins the army composition, and duels run with seven
        # or eight of them, so diverting one leaves six claiming pools. The cost
        # of not diverting it is measured: matches finishing with a completed
        # plan, an army of 26, five extractors -- and **44,660 credits banked
        # against a single factory**, knocking an opponent from a peak of
        # 37,750 down to 6,650 without ever finishing it
        # ([[policy-holding-ground]]).
        #
        # Bounded twice over, which is what the earlier attempt lacked:
        # :func:`expand_production` self-gates on ``production_bound`` -- every
        # producer busy *and* surplus enough for a factory -- so it does nothing
        # while throughput is not the constraint; and it is offered a single
        # worker, so expansion always keeps the rest. In the opening, where only
        # one builder exists, the guard below withholds it entirely.
        capacity: tuple[BuildOrder, ...] = ()
        if len(free) >= _SPARE_WORKER_FLOOR:
            throughput = expand_production(
                sample,
                self._catalogue,
                available=budget.spendable(),
                wanted=wanted,
                free=free[-1:],
            )
            capacity = self._commit(throughput, budget, workforce)
            self.reaches.reached("spare-throughput", bool(capacity), throughput["reason"])
            if capacity:
                # The rest of the chain runs on the workers that are left, so a
                # factory this tick never costs the pool this tick.
                free = free[:-1]

        growth = expand_economy(
            sample,
            self._catalogue,
            self._profiles,
            reserve=0,
            free=free,
            # Pools already being walked to are not free pools. Without this
            # every worker freed by the gate above was offered the same nearest
            # one ([[policy-holding-ground]]).
            claimed=workforce.claims(),
        )
        # Kept because it is the informative one when nothing is bought at all.
        # "No pool was taken" has five distinct causes and :class:`Expansion`
        # enumerates them; whichever spender happens to run last would
        # otherwise report only that it ran last ([[policy-economy]]).
        income_reason = growth["reason"]
        # **Throughput was tried first and it lost decisively.** The reasoning
        # was that :func:`expand_production` self-gates on
        # :func:`~rw_bot.policy.production.production_bound` -- every producer
        # busy *and* surplus enough for a factory -- so leading with it would
        # only take the builder when income genuinely could not be spent. A
        # per-sample trace had just shown the bot banking 24,866 credits with
        # the producer count stuck at two.
        #
        # Measured over six seeds at full length it was the worst arm yet:
        # **three wiped and three defeated, against three survivals for the
        # same seeds without it.** The scorecards say why, and it is not
        # subtle. Expansion collapsed from 307-509 orders to **2-6**, every
        # one of them a factory; extractors finished at 0 or 1 and income at
        # 0/s. With one or two producers ``production_bound`` is satisfied on
        # nearly every observation, so the rule took the builder nearly every
        # time it came free.
        #
        # The docstring on ``production_bound`` had already said this, and it
        # was dismissed as a 1,500-sample artifact answering a question that
        # had since changed. It was not: **there is one builder, and every
        # factory it places is an extractor it does not**, and that holds at
        # full length exactly as it held at 1,500. The banked credits are real
        # and remain unexplained by this ordering -- what they are evidence
        # for is too few *builders*, which is what the engine's own AI answers
        # with two per base across several bases ([[ai-opponent-strategy]]),
        # not for re-spending the one builder's time ([[policy-production]]).
        #
        # Defence second: after income, before throughput.
        #
        # **Not first, and that was measured rather than argued.** Nothing asked
        # "can I keep what I bought", so defence was tried ahead of income on
        # the reasoning that a turret is cheaper than the extractor it covers
        # and 247 expansion orders were leaving one extractor standing
        # ([[policy-holding-ground]]). It
        # made every match worse -- four defeats out of four against two
        # survivals -- and the scorecards say why: there is *always* some
        # uncovered structure, so the rule took the builder nearly every tick.
        # Expansion collapsed from 275 orders to about 40 and income never
        # grew. Turrets were bought for an economy that no longer existed.
        #
        # Income still compounds and defence still does not, so income keeps
        # its place. What defence takes is the surplus that was previously
        # buying a twenty-second Land Factory, which is a trade between two
        # things that both fail to compound ([[policy-production]]).
        #
        # **Each stage records that it was reached, whether or not it acted.**
        # Without that the chain is unreadable from outside: a stage that
        # declined three thousand times and a stage that was never asked both
        # leave the same trace, which is none. Defence was judged and refuted on
        # exactly that ambiguity, and it turned out to have fired three times in
        # twelve full matches ([[policy-holding-ground]]).
        self.reaches.reached("income", growth["build"], growth["reason"])
        # **Nothing cheaper jumps the queue while income is merely short.**
        # Income needs the extractor's 700 *plus* the reserve; a turret needs
        # 500 of the spendable balance. So on every observation where the
        # economy was refused for credits, defence was offered the same money
        # and could afford it -- and did. Measured at Hard: 29 turrets bought
        # against 4 extractors, 43 of 47 extractor claims refused for credits,
        # income stuck at 34/s while the opponent compounded
        # ([[policy-holding-ground]]).
        #
        # A refusal for any *other* reason is a different matter and the surplus
        # is genuinely spare: every pool taken, every route exposed, no worker
        # able to place one. Those are what defence and throughput are for.
        if growth["priced_out"]:
            self.reason = income_reason
            return (*capacity, *self._commit(growth, budget, workforce))
        if not growth["build"]:
            growth = self._cover(sample, budget, free)
        if not growth["build"]:
            growth = expand_production(
                sample,
                self._catalogue,
                available=budget.spendable(),
                wanted=wanted,
                free=free,
            )
            self.reaches.reached("throughput", growth["build"], growth["reason"])
        self.reason = growth["reason"] if growth["build"] else income_reason
        return (*capacity, *self._commit(growth, budget, workforce))

    def _note_air(self, sample: Sample) -> None:
        """Latch the sighting that arms anti-air cover.

        Latched, not sampled: aircraft leave the viewport and come back, and
        AA that stands down between sorties is never finished when the
        sortie arrives.
        """
        if self.aa_cover and not self._air_seen:
            self._air_seen = any(
                entity["hostile"] and entity["flying"] for entity in sample["entities"]
            )

    def _cover(self, sample: Sample, budget: Budget, free: Sequence[Entity]) -> Expansion:
        """Cover a bare structure: anti-air first once aircraft have been shown.

        Skipped entirely when the doctrine turns cover off -- the arm that
        asks whether landing turrets was worth landing.

        **AA outranks ground cover on the latch, and the first arm measured
        why it must.** V1 put anti-air after ground defence in the chain, and
        its own reach line convicted it in one batch: ``aa-cover reached 50
        acted 0``, zero AA turrets standing across twelve matches -- reached
        only when everything above had declined, and never with 600 credits
        left by then (log: 2026-07-30). The old disease, "a policy that was
        reached, never one that ran", caught in one batch instead of four
        because the reach line now exists. The inversion is safe by the same
        gap that motivates it: ground raiders already have the guard, and
        nothing else in the bot touches an aircraft at all.

        Anti-air only after the opponent has **shown** aircraft -- it cannot
        touch the ground ([[mechanics-combat-profile]]), so before that
        sighting it is 600 credits pointed at a guess. The sighting is
        latched by :meth:`step`: sorties leave the viewport and come back,
        and AA that stands down between them is never finished when one
        arrives.

        Args:
            sample: One observation of the world.
            budget: The tick's credits.
            free: Workers not already carrying out an order.

        Returns:
            The first cover decision that builds, or the last that declined.
        """
        if self.aa_cover and self._air_seen:
            growth = expand_defence(
                sample,
                self._catalogue,
                self._profiles,
                available=budget.spendable(),
                free=free,
                turret_type=AA_TURRET_TYPE,
            )
            self.reaches.reached("aa-cover", growth["build"], growth["reason"])
            if growth["build"]:
                return growth
        if not self.cover:
            declined = waiting("cover disabled by doctrine", sample, TURRET_TYPE)
            self.reaches.reached("defence", False, declined["reason"])
            return declined
        growth = expand_defence(
            sample,
            self._catalogue,
            self._profiles,
            available=budget.spendable(),
            free=free,
        )
        self.reaches.reached("defence", growth["build"], growth["reason"])
        return growth

    def _commit(
        self, growth: Expansion, budget: Budget, workforce: Workforce
    ) -> tuple[BuildOrder, ...]:
        """Claim the credits for a decision and turn it into an order.

        **The first few extractors are protected; everything after is not.**
        The reserve exists to keep expansion from taking the credits that
        replace a loss now ([[policy-budget]]), and nothing did the reverse --
        so an army bleeding out consumed the whole income and the economy that
        funds it never started. Measured at Very Hard: **2,800 credits reached
        the economy out of roughly 65,000 spent**, 129 units produced and two
        alive, income ending at 26/s ([[policy-holding-ground]]).

        Args:
            growth: What the economy decided. Nothing is ordered when it
                declined to build.
            budget: The tick's credits.
            workforce: Told what the worker was sent to build, so the next
                observation can see it working.

        Returns:
            The order, or nothing when there was nothing to buy or the budget
            refused it.
        """
        if not growth["build"]:
            return ()
        stats = self._catalogue[growth["type_name"]]
        claim = budget.claim(
            f"expand:{growth['type_name']}",
            stats["price"],
            protected=growth["owned"] < ECONOMY_FLOOR,
        )
        if not claim["granted"]:
            self.reason = claim["reason"]
            return ()
        site = (growth["x"], growth["y"])
        workforce.assign(growth["unit_id"], growth["type_name"], site)
        self.count += 1
        if growth["type_name"] == FACTORY_TYPE:
            self.factories += 1
        return (
            build_order(
                unit_id=growth["unit_id"],
                type_name=growth["type_name"],
                x=growth["x"],
                y=growth["y"],
            ),
        )


__all__ = ["Expander"]
