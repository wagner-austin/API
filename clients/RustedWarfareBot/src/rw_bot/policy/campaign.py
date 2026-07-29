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
from rw_bot.policy.counter import counter_composition
from rw_bot.policy.dispatch import WaveController
from rw_bot.policy.expander import Expander
from rw_bot.policy.ledger import Outlays
from rw_bot.policy.match_report import MatchReport
from rw_bot.policy.production import wanted_producers
from rw_bot.policy.recorder import Recorder
from rw_bot.policy.runner import DEFAULT_STALL_SAMPLES, OrderTracker
from rw_bot.policy.scorekeeper import Scorekeeper
from rw_bot.policy.spending import (
    PlanStep,
    build_plan,
    replace_losses,
    upgrade_income,
    worker_need,
)
from rw_bot.policy.verdict import GRADE_SURVIVED
from rw_bot.policy.workforce import Workforce
from rw_bot.wire.command import AttackOrder, BuildOrder, MoveOrder, ProduceOrder

#: Samples a stationary builder may sit on an unstarted expansion before the
#: order is presumed lost and sent again.
#:
#: The same reasoning as the plan's stall window, used for the opposite purpose:
#: there it ends the plan, here it retries. A builder that has neither moved nor
#: started building for this many samples is not on its way anywhere, and the
#: cost of being wrong is one duplicate order the engine collapses onto the same
#: waypoint ([[policy-loop]]).
EXPAND_RETRY_SAMPLES = 45

#: The most builders worth holding.
#:
#: Set by measurement, not by argument. Uncapped, the bot bought 33 in a
#: 1500-sample match -- 16,500 credits of labour placing 13 extractors, while the
#: army it was supposed to be funding stayed at a dozen units
#: ([[policy-production]]).
DEFAULT_MAX_WORKERS = 4


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
    stop_when_plan_done: bool = False,
    stall_samples: int = DEFAULT_STALL_SAMPLES,
    ladder: Sequence[int] = WAVE_SIZES,
    trace: Path | None = None,
) -> MatchReport:
    """Play the match: one observation, one arbitration, one dispatch, repeat.

    Every layer runs on every observation. That is the whole of the change this
    function represents, and it is what removes three defects at once: the bot
    is no longer defenceless while building, no longer unable to build once
    fighting, and no longer unable to fight at all when the opening plan stalls.

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

    Attacking costs nothing and is therefore not arbitrated at all.

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
        counter: Whether production tilts toward the layers the opponent is
            seen fielding, repeating anti-air already in the mix until its
            share covers the visible air threat. False holds the stated mix
            regardless, which is the behaviour every measurement so far was
            taken under ([[mechanics-combat-profile]]).
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
        ladder: How many units each successive wave waits for. Defaults to the
            shipped AI's ([[engine-ai-triggers]]).
        trace: Where to write the per-sample record, or None to keep none.

    Returns:
        The match report.

    Raises:
        ChannelError: When the agent closes the connection mid-match.
        OSError: When the connection fails.
    """
    tracker = OrderTracker(plan, stall_samples)
    expander = Expander(catalogue, profiles, expand)
    workforce = Workforce(EXPAND_RETRY_SAMPLES)
    recorder = Recorder(trace)
    scores = Scorekeeper(catalogue, profiles)
    waves = WaveController(ladder)

    produced = 0
    refused = 0
    outlays = Outlays()
    # Conversions already ordered, as (structure, tier). A conversion never
    # fills the queue, so without this the same order is re-sent every
    # observation -- and it is keyed by the pair rather than by the structure
    # because a conversion keeps the engine identity, so remembering the unit
    # alone would bar it from ever taking a second step up the chain.
    upgraded: set[tuple[int, str]] = set()
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
            targets = find_targets(sample)
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
            composition_now: tuple[str, ...] = (*need, *reinforce)
            if counter:
                composition_now = counter_composition(composition_now, targets, profiles)
            capable = wanted_producers(sample, composition_now)
            queues_open = sum(
                1
                for entity in sample["entities"]
                if entity["unit_id"] in set(capable) and entity["queued"] == 0
            )
            produce_orders = replace_losses(sample, catalogue, budget, composition_now)
            ordered_now = _send_produces(channel, produce_orders)
            produced += ordered_now
            # **Upgrading claims before expanding, and the arithmetic says the
            # opposite.** A new extractor is 700 for +8 credits a second;
            # converting one is 1,400 for +4 and then 4,000 for +8, so pools are
            # six times better per credit and [[policy-economy]] states the rule
            # outright -- "take every free pool before upgrading anything".
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
            _send_produces(channel, upgrade_income(sample, catalogue, budget, upgraded))
            _send_builds(
                channel,
                expander.step(sample, budget, free, plan_holds_worker, composition_now, workforce),
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
            moves, attacks = waves.command(sample, catalogue, profiles, army)
            _send_moves(channel, moves)
            _send_attacks(channel, attacks)
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
        killed=waves.killed(scores.visible_now),
        refused_claims=refused,
        outlays=outlays.rows(),
        reaches=expander.reaches.rows(),
        outcome=outcome,
    )


__all__ = ["EXPAND_RETRY_SAMPLES", "play"]
