"""Play a whole match in one tick: perceive, arbitrate, dispatch, acknowledge.

There used to be two loops. The build loop ran the opening plan to completion
and handed over to a fight loop, and the seam between them was the bot's largest
structural defect rather than a tidy separation of concerns:

* while building there was no army and no economy, so the opening was played
  defenceless and every credit above the plan's cost sat in the bank;
* once fighting there was no build policy at all, so ``extractorT1`` was the
  only structure that could ever be placed again and the factory count was
  frozen for the rest of the match — which is the arithmetic behind a run that
  banked 7,013 credits behind a single Land Factory ([[policy-production]]);
* and a plan that stalled meant a match that never fought, because the handover
  was conditional on the plan finishing.

So there is one loop, and everything runs on every observation. The plan keeps
building, the factories keep producing, the economy keeps expanding and the army
keeps fighting, for as long as the match lasts.

**Spending is arbitrated, not raced.** Every decision that costs credits claims
against one :class:`~rw_bot.policy.budget.Budget` per observation, in priority
order: the plan first because its prerequisites gate everything, then replacing
losses, then more income, then more throughput. Before this, the production pass
and the expansion pass each budgeted against ``sample["credits"]`` independently
and committed the same credit twice ([[policy-budget]]).

This module is orchestration only. What to build is
:mod:`rw_bot.policy.build_order`, what to attack is
:mod:`rw_bot.policy.combat`, what to claim is :mod:`rw_bot.policy.economy`, and
all of them are pure -- the channel is touched here and nowhere else.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from rw_bot.control.channel import AgentChannel
from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.mechanics.combat_profile import CombatProfile
from rw_bot.mechanics.placement import TypePlacement
from rw_bot.policy.budget import Budget
from rw_bot.policy.combat import WAVE_SIZES, find_army, find_targets
from rw_bot.policy.counter import counter_composition, mobile_threats
from rw_bot.policy.creep import Creeper
from rw_bot.policy.dispatch import WaveController
from rw_bot.policy.expander import Expander
from rw_bot.policy.intel import Intel
from rw_bot.policy.ledger import Outlays
from rw_bot.policy.match_report import MatchReport
from rw_bot.policy.production import wanted_producers
from rw_bot.policy.raid import Raider
from rw_bot.policy.recorder import Recorder
from rw_bot.policy.runner import AFFORD_STALL_SAMPLES, DEFAULT_STALL_SAMPLES, OrderTracker
from rw_bot.policy.rush import Rusher
from rw_bot.policy.scorekeeper import Scorekeeper
from rw_bot.policy.scouting import SCOUT_TYPE, ScoutRunner
from rw_bot.policy.spending import (
    PlanStep,
    build_plan,
    replace_losses,
    unlock_tech,
    upgrade_income,
    worker_need,
)
from rw_bot.policy.verdict import GRADE_SURVIVED
from rw_bot.policy.workforce import DEFAULT_MAX_WORKERS, EXPAND_RETRY_SAMPLES, Workforce
from rw_bot.wire.command import AttackOrder, BuildOrder, MoveOrder, ProduceOrder
from rw_bot.wire.state import Entity, Sample


def _send_plan_step(channel: AgentChannel, step: PlanStep) -> None:
    """Send whatever the opening plan decided on, if anything.

    Args:
        channel: An open connection to the agent.
        step: What the plan decided this observation.

    Raises:
        OSError: When the connection fails.
    """
    if step["produce"] is not None:
        channel.send_produce(step["produce"])
    if step["build"] is not None:
        channel.send_build(step["build"])


def _send_moves(channel: AgentChannel, moves: Sequence[MoveOrder]) -> int:
    """Send every move order and report how many.

    Args:
        channel: An open connection to the agent.
        moves: The orders to send.

    Returns:
        How many were sent.

    Raises:
        OSError: When the connection fails.
    """
    for move in moves:
        channel.send_move(move)
    return len(moves)


def _send_attacks(channel: AgentChannel, attacks: Sequence[AttackOrder]) -> int:
    """Send every attack order and report how many.

    Args:
        channel: An open connection to the agent.
        attacks: The orders to send.

    Returns:
        How many were sent.

    Raises:
        OSError: When the connection fails.
    """
    for attack in attacks:
        channel.send_attack(attack)
    return len(attacks)


def _send_produces(channel: AgentChannel, orders: Sequence[ProduceOrder]) -> int:
    """Send every produce order and report how many.

    Args:
        channel: An open connection to the agent.
        orders: The orders to send.

    Returns:
        How many were sent.

    Raises:
        OSError: When the connection fails.
    """
    for order in orders:
        channel.send_produce(order)
    return len(orders)


def _send_builds(channel: AgentChannel, orders: Sequence[BuildOrder]) -> None:
    """Send every build order.

    Args:
        channel: An open connection to the agent.
        orders: The orders to send.

    Raises:
        OSError: When the connection fails.
    """
    for order in orders:
        channel.send_build(order)


def play(
    channel: AgentChannel,
    plan: Sequence[str],
    catalogue: Mapping[str, UnitStats],
    placements: Mapping[str, TypePlacement],
    profiles: Mapping[str, CombatProfile],
    max_samples: int,
    *,
    reinforce: Sequence[str] = (),
    reserve: int = 0,
    expand: bool = True,
    max_workers: int = DEFAULT_MAX_WORKERS,
    counter: bool = False,
    cover: bool = True,
    intercept: bool = False,
    guard_cap: int = 0,
    aa_cover: bool = False,
    forward: bool = False,
    scout: bool = False,
    raid: int = 0,
    rush: bool = False,
    creep: bool = False,
    riposte: bool = False,
    tech: bool = False,
    stop_when_plan_done: bool = False,
    stall_samples: int = DEFAULT_STALL_SAMPLES,
    afford_samples: int = AFFORD_STALL_SAMPLES,
    ladder: Sequence[int] = WAVE_SIZES,
    trace: Path | None = None,
) -> MatchReport:
    """Play the match: one observation, one arbitration, one dispatch, repeat.

    Every layer runs on every observation, which removes three defects at
    once: defenceless while building, unable to build once fighting, unable
    to fight when the opening plan stalls.

    Order within a tick is the spending priority, and it is expressed by call
    order rather than by weights, because the order *is* the policy:

    1. **The plan**, protected. Its prerequisites gate everything else.
    2. **Replacing losses**, protected. An army dying now cannot wait for
       income.
    3. **Expansion**, unprotected. Income first, then throughput: income
       compounds and throughput does not, and buying the second ahead of the
       first takes the one builder away from the only asset that grows
       ([[policy-production]]). Neither may take the credits held for the army
       ([[policy-budget]]).

    Attacking costs nothing and is not arbitrated at all.

    Args:
        channel: An open connection to the agent.
        plan: What to make, in order. Entries may be structures or units.
        catalogue: Unit stats by type name, for prices and mobility.
        placements: Placement rules by type name, for where each may stand.
        profiles: Combat profiles by type name, for reach, for the threat
            filter, and for which targets the army can engage at all.
        max_samples: Stop after this many observations regardless.
        reinforce: Type names idle producers should keep making. Empty means
            fight with what exists and make nothing.
        reserve: Credits held back from expansion for the army.
        max_workers: The most builders worth holding. Every one past the point
            where they can be usefully employed is credits standing still
            instead of fighting ([[policy-production]]).
        counter: Tilt production toward the layers the opponent fields.
        cover: Buy turrets beside bare structures at all.
        intercept: Turn the reserve on a raider inside our outpost radius.
        guard_cap: The most reserve units an interception commits; 0 is all.
        aa_cover: Add an anti-air turret to cover once aircraft are shown.
        forward: Post the reserve at the frontier extractor, not the base.
        scout: Keep a scout walking the pools, feeding the counter tilt.
        rush: March released waves at the estimated enemy start.
        raid: The raid party's size, or zero for no raiding.
        creep: Walk turrets toward the enemy start, one covered step each.
        riposte: Release the whole reserve the moment an intrusion ends.
        tech: Unlock the factories' next tier through the ability verb.

        Each of these is one doctrine field; the reasoning and the
        measurements behind every flag live on
        :class:`~rw_bot.policy.doctrine.Doctrine`, written once rather than
        twice ([[policy-doctrine]]).
        expand: Whether to play the economy at all. False is the control arm of
            the A/B that measures whether expanding helps, and what a probe
            passes when it wants the economy held still ([[policy-economy]]).
        stop_when_plan_done: Whether finishing the plan ends the run. False for
            a match, because a finished opening is when playing starts. True is
            what a probe passes when the plan *is* the task -- the income probe
            builds an extractor and then wants control back to measure, and
            playing on would spend the credits it is trying to observe.
        stall_samples: Observations of no visible progress before the plan is
            called stalled.
        afford_samples: Observations a price wait may persist without its
            shortfall shrinking before the plan is ruled blocked and its
            worker released ([[policy-economy]]).
        ladder: How many units each successive wave waits for. Defaults to the
            shipped AI's ([[engine-ai-triggers]]).
        trace: Where to write the per-sample record, or None to keep none.

    Returns:
        The match report.

    Raises:
        ChannelError: When the agent closes the connection mid-match.
        OSError: When the connection fails.
    """
    tracker = OrderTracker(plan, stall_samples, afford_samples)
    expander = Expander(catalogue, profiles, expand, aa_cover, cover)
    workforce = Workforce(EXPAND_RETRY_SAMPLES)
    recorder = Recorder(trace)
    scores = Scorekeeper(catalogue, profiles)
    waves = WaveController(
        ladder, intercept=intercept, guard_cap=guard_cap, forward=forward, riposte=riposte
    )
    intel = Intel()
    scouts = ScoutRunner()
    # Sized by the doctrine; at zero the raid gate below never fires and the
    # raider is never consulted, so the size is safe to construct with.
    raiders = Raider(size=raid) if raid else Raider()
    rusher = Rusher()
    creeper = Creeper()

    produced = 0
    refused = 0
    outlays = Outlays()
    # Conversions already ordered, as (structure, tier). A conversion never
    # fills the queue, so without this the same order is re-sent every
    # observation -- and it is keyed by the pair rather than by the structure
    # because a conversion keeps the engine identity, so remembering the unit
    # alone would bar it from ever taking a second step up the chain.
    upgraded: set[tuple[int, str]] = set()
    teched: set[int] = set()
    completed = 0
    build_outcome = "building"
    build_reason = "no sample seen yet"
    outcome = "sample_limit"

    while scores.samples_seen < max_samples:
        sample = channel.next_sample()

        # Acknowledged on every exit, including the ones that break out. In
        # lockstep the agent holds the simulation until this arrives
        # ([[policy-determinism]]).
        try:
            army = find_army(sample, catalogue, profiles)
            if scout:
                # The scout is eyes, not a soldier: left in the army it would
                # be counted toward a wave and marched into the fight the
                # moment enough of it gathers.
                army = tuple(unit for unit in army if unit["type_name"] != SCOUT_TYPE)
            targets = find_targets(sample)
            if scout or raid:
                intel.observe(sample)
            scores.observe(sample, army, targets, workforce.size(sample))
            completed = tracker.completed(sample)

            # Read unconditionally, on every sample. Movement is what tells
            # both the plan and the economy that an order is still being carried
            # out, so every worker has to be sampled even on observations that
            # never reach a decision.
            free = workforce.free(sample)

            budget = Budget(sample["credits"], reserve)

            plan_step = build_plan(
                sample,
                tracker,
                budget,
                catalogue,
                placements,
                profiles,
                free,
                workforce,
            )
            build_outcome = plan_step["outcome"]
            build_reason = plan_step["reason"]
            plan_holds_worker = plan_step["holds_worker"]
            _send_plan_step(channel, plan_step)
            # Production runs before the army check, so a wave that has just
            # been wiped still queues its replacements on the sample that
            # notices.
            need = worker_need(
                free,
                workforce.size(sample),
                budget.spendable(),
                catalogue,
                max_workers,
            )
            # A wanted builder joins the composition rather than sitting in a
            # channel of its own. The separate channel was reachable only by a
            # producer that could make nothing in the army mix -- the Command
            # Center and nothing else -- so a Land Factory, which can always
            # make a tank, never fell through to it and the bot ran the whole
            # match on one builder ([[policy-production]]).
            composition_now: tuple[str, ...] = (
                *need,
                *(scouts.need(sample, workforce.size(sample)) if scout else ()),
                *reinforce,
            )
            if counter:
                threats = mobile_threats(intel, catalogue) if scout else tuple(targets)
                composition_now = counter_composition(composition_now, threats, profiles)
            capable = wanted_producers(sample, composition_now)
            queues_open = sum(
                1
                for entity in sample["entities"]
                if entity["unit_id"] in set(capable) and entity["queued"] == 0
            )
            # **Upgrading claims before production, and this is the third
            # ordering this pair has held.** Production-first left the tier
            # three conversion asked eighteen hundred times a match and
            # granted never -- `upgrade:extractorT3 asked 1816 got 0` -- while
            # produce drained every credit above the reserve into units that
            # traded even against a 1.8x income and equilibrated
            # ([[policy-economy]], log 2026-07-31). A 4,000-credit conversion
            # returns +8 a second on ground already held: ~500 in-game seconds
            # to pay back inside matches that run ~900. The reserve still
            # protects the replacement of a loss -- upgrades claim past it --
            # so production is deferred, not starved.
            # Tech claims before income conversions. The unlock saves toward
            # itself when refused, and the T2 extractor conversion funds at a
            # 2,300 balance where the unlock needs 2,900 -- ordered the other
            # way round, every accrual is sniped just short of the goal and
            # the tech arm never reaches the roster it exists for.
            _send_tech(channel, tech, sample, budget, teched)
            _send_produces(channel, upgrade_income(sample, catalogue, budget, upgraded))
            _advance_creep(
                channel, creep, sample, catalogue, profiles, budget, free, workforce, creeper
            )
            produce_orders = replace_losses(sample, catalogue, budget, composition_now)
            ordered_now = _send_produces(channel, produce_orders)
            produced += ordered_now
            # **Upgrading also claims before expanding, and the arithmetic
            # says the opposite.** A new extractor is 700 for +8 credits a
            # second; converting one is 1,400 for +4 and then 4,000 for +8, so
            # pools are six times better per credit and [[policy-economy]]
            # states the rule outright -- "take every free pool before
            # upgrading anything".
            #
            # **It was reordered on that arithmetic and measured worse.** Twelve
            # seeds at Very Hard: 7 won with upgrades first, 5 with expansion
            # first, same two losses, routs 3 -> 2, median win 2,207 -> 2,362.
            # Inside the noise floor, so not a refutation -- but it is certainly
            # not the improvement the per-credit figure promised.
            #
            # The mechanism the arithmetic omits is **risk**, and it is the one
            # thing every rung of this ladder turns on. Matches are decided by
            # how many extractors are LOST -- winners drop nought to four, the
            # rest six or more ([[policy-holding-ground]]). A new extractor is
            # income that can be destroyed; a conversion is income on ground
            # already held, needing no builder and crossing no contested map.
            # Six times the price for income that cannot be taken away is a
            # different trade from six times the price for nothing.
            #
            # So the order stands, and the reasoning is recorded because the
            # arithmetic against it is correct and still lost.
            _send_builds(
                channel,
                expander.step(
                    sample,
                    budget,
                    free,
                    plan_holds_worker,
                    composition_now,
                    workforce,
                    plan_step["wants_worker"],
                ),
            )
            refused_now = sum(1 for claim in budget.ledger() if not claim["granted"])
            refused += refused_now
            # **The reasons are kept now, not just the count.** Every claim
            # carries a sentence saying what it wanted and why it did not get
            # it, and this loop used to reduce a whole tick of that to the one
            # number above -- about four thousand sentences a match, discarded
            # ([[policy-economy]]).
            outlays.add(budget.ledger())
            recorder.step(
                sample,
                scores.army_end,
                scores.targets_end,
                scores.extractors_end,
                len(capable),
                queues_open,
                ordered_now,
                refused_now,
                scores.worth_end,
                scores.rival_worth_end,
            )

            # The engine's verdict is the only thing that ends a match early.
            #
            # The two-loop version also stopped on "no army left" and on
            # "nothing hostile in sight", and neither survives the move to one
            # loop. Nothing hostile in sight is the *opening* position of every
            # match -- the map is fogged and the opponents are across it -- so it
            # would have ended the run on the first observation. And an army of
            # zero is no longer terminal now that production runs every tick:
            # losing a wave is a setback to rebuild from, not a reason to stop
            # playing. Both were proxies for a verdict the engine states
            # outright ([[policy-verdict]]).
            if scores.verdict != GRADE_SURVIVED:
                outcome = scores.verdict
                break
            if stop_when_plan_done and build_outcome != "building":
                # The plan is the caller's whole task and it has settled, one
                # way or another. Only a probe asks for this; a match treats a
                # finished opening as the point where playing begins.
                outcome = build_outcome
                break

            # Fill, then commit; gather, then hold one target. The rules and
            # their memory live on the controller; what the loop owns is only
            # that attacking runs last and costs nothing, which is why it is
            # not arbitrated at all ([[policy-combat]]).
            if scout:
                _send_moves(channel, scouts.patrol(sample, catalogue))
            fighting = army
            if raid:
                fighting = _draft_raid(channel, sample, catalogue, intel, army, waves, raiders)
            moves, attacks = waves.command(sample, catalogue, profiles, fighting)
            _send_moves(channel, moves)
            _send_attacks(channel, attacks)
            if rush:
                _march_rush(channel, sample, catalogue, waves, rusher, fighting, targets)
        finally:
            channel.send_ack()

    recorder.write()
    return scores.report(
        completed=completed,
        planned=len(tracker.plan),
        build_orders=tracker.orders_sent,
        build_outcome=build_outcome,
        build_reason=build_reason,
        produced=produced,
        expanded=expander.count,
        expanded_factories=expander.factories,
        expand_reason=expander.reason,
        attack_orders=waves.attack_orders,
        rallied=waves.rallied,
        intercepts=waves.intercepts,
        sightings=intel.sightings_taken,
        raids=raiders.raids,
        marches=raiders.marches + rusher.marches,
        killed=waves.killed(scores.visible_now),
        refused_claims=refused,
        outlays=outlays.rows(),
        reaches=expander.reaches.rows(),
        outcome=outcome,
    )


__all__ = ["play"]


def _draft_raid(
    channel: AgentChannel,
    sample: Sample,
    catalogue: Mapping[str, UnitStats],
    intel: Intel,
    army: tuple[Entity, ...],
    waves: WaveController,
    raiders: Raider,
) -> tuple[Entity, ...]:
    """Advance the raid and return the units the waves may still command.

    Whether the army can SPARE a party is decided here, against the wave
    controller's own figure, because nothing asked it in v1 and that cost
    every seat in the batch: the party came out of the wave gate and the
    guard, and the raid was refuted 0/12 for it (log: 2026-07-29). The gate
    arbitrates drafting only -- a party already out is managed to its end
    regardless. A unit cannot serve two commanders: whatever the raid drafted
    is withheld from the waves, and returns the moment the raid has nothing
    left to assault.
    """
    spare = len(army) >= waves.need() + raiders.size
    for order in raiders.strike(sample, intel, army, catalogue, spare):
        channel.send_attack_move(order)
    drafted = raiders.party()
    return tuple(u for u in army if u["unit_id"] not in drafted)


def _send_tech(
    channel: AgentChannel,
    tech: bool,
    sample: Sample,
    budget: Budget,
    teched: set[int],
) -> None:
    """Fire the factories' tier unlocks, when the doctrine plays tech.

    The flag is judged here rather than in the loop so the loop stays under
    its complexity bound; an arm without the verb costs one call and no
    claim ([[mechanics-build-actions]]).
    """
    if not tech:
        return
    for unlock in unlock_tech(sample, budget, teched):
        channel.send_ability(unlock)


def _advance_creep(
    channel: AgentChannel,
    creep: bool,
    sample: Sample,
    catalogue: Mapping[str, UnitStats],
    profiles: Mapping[str, CombatProfile],
    budget: Budget,
    free: Sequence[Entity],
    workforce: Workforce,
    creeper: Creeper,
) -> None:
    """Advance the turret line one covered step, when the doctrine plays it.

    Claimed before the army on purpose: for a creep style the turret line IS
    the army, and a claim placed after production would starve on the same
    every-tick drain the tier-three conversion did ([[policy-creep]]). The
    flag is judged here rather than in the loop so the loop stays under its
    complexity bound; an arm without the verb costs one call and no claim.
    """
    if not creep:
        return
    _send_builds(channel, creeper.advance(sample, catalogue, profiles, budget, free, workforce))


def _march_rush(
    channel: AgentChannel,
    sample: Sample,
    catalogue: Mapping[str, UnitStats],
    waves: WaveController,
    rusher: Rusher,
    fighting: tuple[Entity, ...],
    targets: Sequence[Entity],
) -> None:
    """March this tick's released units at the enemy start until contact.

    After the waves, so the released set is this tick's. While nothing is
    visible the released units march at the mirror of our base; on contact
    the engagement policy re-tasks them, the engine running the newest
    waypoint ([[policy-combat]]).
    """
    cleared = waves.released()
    marching = tuple(u for u in fighting if u["unit_id"] in cleared)
    for order in rusher.march(sample, catalogue, marching, bool(targets)):
        channel.send_attack_move(order)
