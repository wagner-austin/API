"""Play a whole match in one tick: perceive, arbitrate, dispatch, acknowledge.

There used to be two loops -- build, then fight -- and the seam between them
was the bot's largest structural defect: defenceless while building, unable to
build once fighting, and a stalled plan meant a match that never fought
([[policy-production]]). So there is one loop, and everything runs on every
observation for as long as the match lasts.

**Spending is arbitrated, not raced.** Every decision that costs credits claims
against one :class:`~rw_bot.policy.budget.Budget` per observation, in priority
order: the plan first because its prerequisites gate everything, then replacing
losses, then more income, then more throughput. Before this, the production pass
and the expansion pass each budgeted against ``sample["credits"]`` independently
and committed the same credit twice ([[policy-budget]]).

This module is orchestration only. What to build is
:mod:`rw_bot.policy.build_order`, what to attack is
:mod:`rw_bot.policy.combat`, what to claim is :mod:`rw_bot.policy.economy`,
and all of them are pure. Orders leave through the loop's sending arm,
:mod:`rw_bot.policy.dispatching` -- the pair is the loop, and the
architecture guard names exactly the two of them so a third sender can
never grow unnoticed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from rw_bot.control.channel import AgentChannel
from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.mechanics.combat_profile import CombatProfile
from rw_bot.mechanics.placement import TypePlacement
from rw_bot.policy.assess import AirWatch
from rw_bot.policy.budget import Budget
from rw_bot.policy.combat import WAVE_SIZES, find_army, find_targets
from rw_bot.policy.counter import (
    FLEET_BLOOD,
    counter_composition,
    fleet_types,
    layer_counts,
    mobile_threats,
)
from rw_bot.policy.creep import Creeper
from rw_bot.policy.decoy import Decoys, scout_shortfall
from rw_bot.policy.dispatch import WaveController
from rw_bot.policy.dispatching import (
    advance_creep,
    fight,
    send_builds,
    send_nukes,
    send_plan_step,
    send_postures,
    send_produces,
    send_recon,
    send_tech,
)
from rw_bot.policy.doctrine import NAVTILT_OFF, NAVTILT_PREDICTED
from rw_bot.policy.doom import DoomLatch, DoomModel
from rw_bot.policy.expander import Expander
from rw_bot.policy.intel import Intel
from rw_bot.policy.ledger import Outlays
from rw_bot.policy.lurk import Lurker
from rw_bot.policy.match_report import MatchReport
from rw_bot.policy.nuker import Nuker
from rw_bot.policy.production import wanted_producers
from rw_bot.policy.quartermaster import Quartermaster
from rw_bot.policy.raid import Raider
from rw_bot.policy.reclaim import Razed
from rw_bot.policy.recorder import Recorder
from rw_bot.policy.runner import AFFORD_STALL_SAMPLES, DEFAULT_STALL_SAMPLES, OrderTracker
from rw_bot.policy.rush import Rusher
from rw_bot.policy.scoreboard import local_player, rival_income
from rw_bot.policy.scorekeeper import Scorekeeper
from rw_bot.policy.scouting import SCOUT_TYPE, ScoutRunner
from rw_bot.policy.situation import Closer, Momentum
from rw_bot.policy.spending import (
    build_plan,
    replace_losses,
    upgrade_income,
    worker_need,
)
from rw_bot.policy.verdict import GRADE_SURVIVED
from rw_bot.policy.workforce import DEFAULT_MAX_WORKERS, EXPAND_RETRY_SAMPLES, Workforce


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
    navtilt: int = NAVTILT_OFF,
    doom: DoomModel | None = None,
    cover: bool = True,
    intercept: bool = False,
    guard_cap: int = 0,
    aa_cover: bool = False,
    forward: bool = False,
    scout: bool = False,
    raid: int = 0,
    rush: bool = False,
    creep: int = 0,
    hold: int = 0,
    riposte: bool = False,
    tech: int = 0,
    lurk: int = 0,
    decoys: int = 0,
    kite: bool = False,
    hp_floor: int = 0,
    allin: int = 0,
    strike: int = 0,
    medics: int = 0,
    navy: int = 0,
    battery: int = 0,
    bunkers: int = 0,
    flame: int = 0,
    close: int = 0,
    guns: int = 0,
    nukes: int = 0,
    rebuild: int = 0,
    income_ladder: bool = False,
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
        max_workers: The most builders worth holding. See Doctrine.
        counter: Tilt production toward the layers the opponent fields.
        navtilt: When the counter tilt's naval clause runs. See Doctrine.
        cover: Buy turrets beside bare structures at all.
        intercept: Turn the reserve on a raider inside our outpost radius.
        guard_cap: The most reserve units an interception commits; 0 is all.
        aa_cover: Add an anti-air turret to cover once aircraft are shown.
        forward: Post the reserve at the frontier extractor, not the base.
        scout: Keep a scout walking the pools, feeding the counter tilt.
        rush: March released waves at the estimated enemy start.
        raid: The raid party's size, or zero for no raiding.
        creep: Walk turrets toward the enemy start. See Doctrine.
        hold: Percent of the line the reserve stands at. See Doctrine.
        riposte: Release the whole reserve the moment an intrusion ends.
        tech: Factories to unlock a tier on, zero for none. See Doctrine.
        lurk: Scouts kept alive at the enemy start, zero for none. See Doctrine.
        decoys: Scatter scouts kept alive on our half, zero for none. See Doctrine.
        kite: Reflex: armed mobile units hold the reach band. See Doctrine.
        hp_floor: Reflex: flee below this percent of health. See Doctrine.
        allin: Observation the whole reserve releases from, zero never. See Doctrine.
        strike: Rival army-value drop that opens the release window. See Doctrine.
        medics: Combat engineers kept alive via saving hires. See Doctrine.
        navy: Attack submarines kept alive on the water. See Doctrine.
        battery: Artillery batteries stood on the shore, at most one per
            match. See Doctrine.
        bunkers: Mobile turrets kept alive the same way. See Doctrine.
        flame: Flame turrets held by converting ground turrets. See Doctrine.
        close: Dominance multiple that releases and marches everything. See
            Doctrine.
        guns: Top-tier gun turrets held by walking the turret chain. See
            Doctrine.
        nukes: Nuke launchers stood and kept firing at the priciest hostile
            structure in sight. See Doctrine.
        rebuild: Rival army-value drop required before a razed pool may be
            re-claimed. See Doctrine.
        income_ladder: Refused extractor conversions save toward themselves.
            See Doctrine.

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
    expander = Expander(catalogue, profiles, expand, aa_cover, cover, rebuild)
    workforce = Workforce(EXPAND_RETRY_SAMPLES)
    recorder = Recorder(trace, profiles)
    scores = Scorekeeper(catalogue, profiles)
    # Every WATER-moving type name seen this match, the bloodied gate's
    # accumulating half ([[policy-exact-timing]], the naval wall).
    fleet_seen: set[str] = set()
    # Decision codes issued since the previous trace row was written --
    # consumed by recorder.step, so each row carries what was decided in
    # the window it closes (log 2026-08-09).
    pending_events: set[str] = set()
    # The doom latch: fed the recorder's own figures, scored once at the
    # model's window, holding its answer for the match (law eight: one
    # decision for a match-reshaping response). None when mode 3 is off.
    doom_latch = DoomLatch(doom) if doom is not None and navtilt == NAVTILT_PREDICTED else None
    waves = WaveController(
        ladder,
        intercept=intercept,
        guard_cap=guard_cap,
        forward=forward,
        hold=hold,
        riposte=riposte,
        allin_at=allin,
    )
    intel = Intel()
    scouts = ScoutRunner()
    lurkers = Lurker()
    scatter = Decoys()
    momentum = Momentum()
    # Pools taken from us, read by the expander's rebuild gate; observed
    # unconditionally like momentum (:mod:`rw_bot.policy.reclaim`).
    razed_pools = Razed()
    quartermaster = Quartermaster(
        medics=medics, navy=navy, bunkers=bunkers, flame=flame, guns=guns, battery=battery
    )
    closer = Closer(close)
    # Sized by the doctrine; at zero the raid gate below never fires and the
    # raider is never consulted, so the size is safe to construct with.
    raiders = Raider(size=raid) if raid else Raider()
    rusher = Rusher()
    creeper = Creeper()
    nuker = Nuker()
    airwatch = AirWatch()

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

    send_postures(channel, kite, hp_floor, catalogue, profiles)
    while scores.samples_seen < max_samples:
        sample = channel.next_sample()

        # Acknowledged on every exit, including the ones that break out. In
        # lockstep the agent holds the simulation until this arrives
        # ([[policy-determinism]]).
        try:
            # The engine's refusals land in the ledger before anything
            # decides: a refusal reported in THIS sample must already be
            # excluded by this sample's site choices, or the tick spends an
            # order on a site the engine just declined.
            for refusal in sample["refusals"]:
                workforce.record_refusal((refusal["x"], refusal["y"]))
            army = find_army(sample, catalogue, profiles)
            if scout or lurk or decoys:
                # The scout is eyes, the lurker a leash, the decoy a ticket
                # in the enemy's target lottery -- none of them soldiers:
                # left in the army they would be counted toward a wave and
                # marched into the fight the moment enough of it gathers.
                army = tuple(unit for unit in army if unit["type_name"] != SCOUT_TYPE)
            targets = find_targets(sample)
            if scout or raid:
                intel.observe(sample)
            momentum.observe(sample)
            razed_pools.observe(sample)
            airwatch.observe(sample)
            # The closer: dominance decays, so a decided match is ended
            # while it is decided -- eleven of nineteen dominant Very Hard
            # positions lost when the game ran long. Latched by the Closer
            # on SUSTAINED dominance only: the un-debounced latch turned
            # early-game ratio noise into lifelong premature all-ins
            # ([[policy-situation]]). Observed at the top of the tick since
            # the finisher funds from it: the commitment IS the surplus
            # signal at this rung (`runs/sweeps/vh-nuke`, log 2026-08-05).
            committed_close = closer.observe(sample)
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
            send_plan_step(channel, plan_step)
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
                # One shortfall for all three scout verbs together: each
                # counting the roster against its own figure would leave
                # every one satisfied by the others' scouts.
                *scout_shortfall(sample, int(scout) + lurk + decoys),
                *reinforce,
            )
            if counter:
                threats = mobile_threats(intel, catalogue) if scout else tuple(targets)
                # The bloodied gate joins two existing records: the fleet
                # types ever seen this match, and the death ledger's kills
                # by those types. A game the fleet never touched can never
                # read bloodied -- the two-panel calibration's whole point.
                fleet_seen.update(fleet_types(threats))
                bloodied = scores.deaths_to(fleet_seen) >= FLEET_BLOOD
                predicted = doom_latch is not None and doom_latch.armed
                untilted = composition_now
                composition_now = counter_composition(
                    composition_now, threats, profiles, navtilt, bloodied, predicted
                )
                if composition_now != untilted:
                    pending_events.add("T")
            capable = wanted_producers(sample, composition_now)
            queues_open = sum(
                1
                for entity in sample["entities"]
                if entity["unit_id"] in set(capable) and entity["queued"] == 0
            )
            # Upgrading claims before production: production-first left the
            # T3 conversion asked 1,816 times and granted never while produce
            # drained every credit into units that traded even and
            # equilibrated. The reserve still protects replacing a loss, so
            # production is deferred, not starved ([[policy-economy]]).
            # Tech claims before income conversions. The unlock saves toward
            # itself when refused, and the T2 extractor conversion funds at a
            # 2,300 balance where the unlock needs 2,900 -- ordered the other
            # way round, every accrual is sniped just short of the goal and
            # the tech arm never reaches the roster it exists for.
            send_tech(channel, tech, sample, budget, teched)
            # The finisher claims right after tech: its 45,000 withhold is
            # the doctrine's stated intent -- the fortress survives on what
            # already stands while the launcher funds -- and an end-of-chain
            # save would be drained by every channel below, exactly as the
            # probe measured cover doing (`runs/nuke-probe.out`).
            send_nukes(
                channel,
                nuker.advance(sample, catalogue, budget, free, workforce, nukes, committed_close),
            )
            # The standing purchases, in the quartermaster's stated order:
            # guard before subs, battery's fork last so its re-send wins a
            # contested holder ([[policy-holding-ground]]).
            send_produces(channel, quartermaster.produces(sample, catalogue, budget))
            # Defence saves toward the turret it was refused last tick, early
            # enough to bind the spenders below -- withheld here rather than
            # where defence claims (last), because a fresh budget every tick
            # means an end-of-tick withhold binds nobody
            # ([[policy-budget]], log 2026-08-01).
            expander.fund_cover(budget)
            send_produces(
                channel,
                upgrade_income(sample, catalogue, budget, upgraded, ladder=income_ladder),
            )
            advance_creep(
                channel, creep, sample, catalogue, profiles, budget, free, workforce, creeper
            )
            produce_orders = replace_losses(sample, catalogue, budget, composition_now)
            ordered_now = send_produces(channel, produce_orders)
            produced += ordered_now
            # Upgrading also claims before expanding, although pools are
            # cheaper per credit: the arithmetic omits risk, and matches are
            # decided by extractors LOST ([[policy-economy]],
            # [[policy-holding-ground]]).
            send_builds(
                channel,
                expander.step(
                    sample,
                    budget,
                    free,
                    plan_holds_worker,
                    composition_now,
                    workforce,
                    plan_step["wants_worker"],
                    air_seen=airwatch.seen(),
                    wave_drop=momentum.drop(),
                    razed=razed_pools.positions(),
                ),
            )
            # The walks send AFTER the expander, never before: the engine
            # holds one order per unit and whoever sends last holds the
            # builder. v3 learned this (navy96b) and v4 forgot it by
            # moving the call above the expander block -- all 48 navy96d
            # walks exhausted with the builder re-tasked every tick while
            # navy96c's factories stood 24/26 (log 2026-08-10).
            send_builds(channel, quartermaster.builds(sample, catalogue, budget))
            refused_now = sum(1 for claim in budget.ledger() if not claim["granted"])
            refused += refused_now
            # **The reasons are kept now, not just the count.** Every claim
            # carries a sentence saying what it wanted and why it did not get
            # it, and this loop used to reduce a whole tick of that to the one
            # number above -- about four thousand sentences a match, discarded
            # ([[policy-economy]]).
            outlays.add(budget.ledger())
            # The enemy-shape columns read from live contact regardless of
            # the counter knob: the trace records what was SEEN, and the
            # fleet memory unions live contact with whatever the tilt path
            # already remembered (log 2026-08-09).
            fleet_seen.update(fleet_types(tuple(targets)))
            navy_seen, air_seen = layer_counts(tuple(targets))
            navy_blood = scores.deaths_to(fleet_seen)
            if doom_latch is not None:
                pilot = local_player(sample)
                # The trace's numeric columns, in doom.COLUMNS order: what
                # the model was fitted on is what the latch is fed, by the
                # same figures the recorder writes.
                doom_latch.feed(
                    (
                        scores.army_end,
                        sample["credits"],
                        scores.targets_end,
                        scores.extractors_end,
                        scores.losses_now,
                        len(capable),
                        queues_open,
                        ordered_now,
                        refused_now,
                        scores.worth_end,
                        scores.rival_worth_end,
                        0 if pilot is None else pilot["income"],
                        rival_income(sample),
                        workforce.size(sample),
                        navy_seen,
                        air_seen,
                        navy_blood,
                    )
                )
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
                build_outcome,
                workforce.size(sample),
                navy_seen,
                air_seen,
                navy_blood,
                "".join(sorted(pending_events)) or "-",
            )
            pending_events.clear()

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
            send_recon(channel, scout, lurk, decoys, scouts, lurkers, scatter, sample, catalogue)
            fight(
                channel,
                sample,
                catalogue,
                profiles,
                intel,
                army,
                targets,
                waves,
                raiders,
                rusher,
                momentum,
                raid=raid,
                rush=rush,
                allin=allin,
                strike=strike,
                committed_close=committed_close,
                pending_events=pending_events,
            )
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
