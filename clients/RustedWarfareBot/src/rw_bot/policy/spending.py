"""The spenders that run before expansion: the plan, production, upgrades.

Each turns one kind of want into orders against the tick's single budget, in the
order the policy ranks them -- the opening plan first because its prerequisites
gate everything, then replacing losses because an army dying now cannot wait for
income, then the upgrades that need no builder at all ([[policy-budget]]).

Expansion itself is the fourth and lives in :mod:`rw_bot.policy.expander`, which
is a longer story: it has three claimants of its own and a census of which were
reached.

What they all have in common is that each needs a **worker that is actually
free**, and that is one question with one owner ([[policy-loop]]). Two spenders
that each decided a builder was available both ordered it, the engine ran
whichever waypoint arrived last, and a measured run produced four expansion
orders against a plan still stuck at three of eight.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.mechanics.combat_profile import CombatProfile
from rw_bot.mechanics.placement import TypePlacement
from rw_bot.policy.budget import Budget
from rw_bot.policy.build_order import BUILDER_TYPE, decide
from rw_bot.policy.economy import upgradeable
from rw_bot.policy.production import sustain
from rw_bot.policy.runner import OrderTracker
from rw_bot.policy.workforce import Workforce
from rw_bot.wire.command import (
    AbilityOrder,
    BuildOrder,
    ProduceOrder,
    ability_order,
    build_order,
    produce_order,
)
from rw_bot.wire.state import Entity, Sample


def upgrade_income(
    sample: Sample,
    catalogue: Mapping[str, UnitStats],
    budget: Budget,
    ordered: set[tuple[int, str]],
    *,
    ladder: bool = False,
) -> tuple[ProduceOrder, ...]:
    """Order every extractor that offers to upgrade itself and can be afforded.

    **The best income the bot can buy, and it was invisible until the agent
    stopped filtering actions.** An extractor converting itself to tier two
    pays 12 credits a second against 8, for 1,400. It needs no builder, crosses
    no contested ground and claims no new pool -- which matters on a map where
    the opponents finish holding 44 of the 46 pools and where 247 expansion
    orders leave the bot with one extractor ([[policy-holding-ground]]).

    Claimed unprotected, like expansion: it is an investment that pays back
    over the rest of the match, and it must not take the credits held to
    replace a loss now ([[policy-budget]]).

    Ordering stops at the first refusal rather than skipping to a cheaper
    extractor, because every upgrade costs the same and a refusal means the
    budget is out.

    **Once per structure per tier, and that is not an optimisation.** A
    conversion does not fill the production queue the way building a unit does,
    so ``queued`` stays at zero for as long as it runs and the structure keeps
    offering the upgrade it is already performing. Re-ordering it every
    observation sent a stream of duplicates, and one of them arrived after the
    conversion had finished -- addressed to a unit that was now an
    ``extractorT2`` and could only make an ``extractorT3``. The agent refuses
    an order naming something its subject cannot make, which is right, and the
    refusal crashed the match ([[policy-holding-ground]]).

    That crash is also what establishes the key. **A conversion preserves the
    engine identity** -- the duplicate reached the *same* unit, now a tier two
    -- so remembering the unit alone would bar every structure from ever taking
    a second step, and the walk would stop at tier two exactly as it did
    before. The pair is remembered instead: the tier two it has been told to
    become, not the fact that it was told something.

    **A conversion is not priced at what the result costs to build, and reading
    it that way over-claimed by half.** This asked the catalogue for
    ``extractorT2``'s ``price`` -- 2,100, what it costs to *build* one, which
    nothing in this game can do to an extractor. The conversion is a separate
    figure the engine prints beside it: ``T2 Upgrade Price: $1400`` on the
    tier one, matching ``action_upgradeT2``'s declared price in
    ``.game/assets/units/extractor/extractor.ini``. Claiming 2,100 for a 1,400
    purchase refuses the upgrade on every tick where the balance falls between
    the two, and this budget refuses 1,185 to 1,685 claims a match already
    ([[policy-holding-ground]]).

    So the price comes from the **holder**, not the target. The first entry of
    a unit's ``upgrade_prices`` is the cost of its own next conversion: 1,400
    on the tier one, 4,000 on the tier two, 8,000 on the tier three, each
    matching the ``convertTo`` action in the asset. That is a positional fact
    rather than a labelled one, deliberately -- the engine's own labels are
    inconsistent, printing a tier three's overclock cost under "T2 Upgrade
    Price" ([[mechanics-unit-value]]).

    A holder the catalogue prices with no upgrade at all is skipped rather
    than guessed at, which is the same shape as an absent type.

    Args:
        sample: One observation of the world.
        catalogue: Unit stats by type name, for the conversion's price.
        budget: The tick's credits.
        ordered: Structures already told to upgrade, extended in place. A
            structure leaves this set only by leaving the roster, which is what
            makes a converted one stop being asked.
        ladder: Whether a refused TIER-THREE conversion saves toward itself
            (lower tiers fund organically and never save). Off is the
            Impossible measurement (see the refusal comment below); on is the
            Very Hard counter-measurement: our worth ceilinged at ~30-35k on
            every seed while ``upgrade:extractorT3 asked 1,559-2,788 got 0``
            in every ledger -- the T3 tier never once funded, income
            flatlined at T2, and the matches where the opponent's compounding
            passed the ceiling were unwinnable regardless of tactics
            (`runs/traces/vh-debounce`, log 2026-08-02). Behind a wall that
            holds, the tier the refusal protects against buying is the
            ceiling itself.

    Returns:
        The produce orders to send, in roster order.
    """
    holders = {entity["unit_id"]: entity["type_name"] for entity in sample["entities"]}
    orders: list[ProduceOrder] = []
    for step in upgradeable(sample):
        key = (step["unit_id"], step["produces"])
        if key in ordered:
            continue
        holder = catalogue.get(holders[step["unit_id"]])
        if holder is None or not holder["upgrade_prices"]:
            continue
        claim = budget.claim(f"upgrade:{step['produces']}", holder["upgrade_prices"][0])
        if not claim["granted"]:
            # A refused conversion does NOT save toward itself by default,
            # and this is a measured refutation, not an omission. Withholding
            # the price (:meth:`~rw_bot.policy.budget.Budget.withhold`)
            # bought six T2 and six T3 conversions a match and doubled income
            # to 98/s -- and the army pauses it forced let the enemy's
            # economy live untouched, rival scores rising to the worst of the
            # whole screening arc while drops fell. Income that displaces
            # early pressure is income for a longer strangle
            # ([[policy-budget]], log 2026-07-31). Unconditionally, saving
            # lost -- at Impossible. The ``ladder`` gate is the Very Hard
            # counter-case (see the docstring), and it saves toward the T3
            # tier ONLY: the raw ladder went 2 won / 12 lost, and the deaths
            # were the early T2-phase withholds starving the wall and army
            # in the window survival is decided -- while T2 conversions fund
            # organically in every match without saving. The two wins broke
            # the worth ceiling (83k at 138/s; a never-won seed at 68k), so
            # the tier that never funds keeps the saving and the tier that
            # always funds keeps its timing (`runs/sweeps/vh-t3`,
            # log 2026-08-02).
            if ladder and step["produces"] == "extractorT3":
                budget.withhold(holder["upgrade_prices"][0])
            break
        ordered.add(key)
        orders.append(produce_order(unit_id=step["unit_id"], type_name=step["produces"]))
    return tuple(orders)


#: The factory whose tier-two unlock the tech channel fires.
#:
#: One type to start, because it is the one that was measured: the land
#: factory's 2,000-credit upgrade converts into no type -- it flips a flag on
#: the same building -- and unlocks the heavy roster behind it
#: ([[mechanics-build-actions]], log 2026-07-31). The air, sea and mech
#: factories carry upgrades too; they join this tuple when an arm asks the
#: question.
TECH_TYPES: tuple[str, ...] = ("landFactory",)


def unlock_tech(
    sample: Sample,
    budget: Budget,
    ordered: set[int],
    *,
    limit: int,
) -> tuple[AbilityOrder, ...]:
    """Order every tech factory to unlock its next tier, when affordable.

    The verb the ability order exists for. The unlock arrives on the option
    stream as an action concerning no type -- ``produces`` empty, the
    engine's own selector index attached -- **and it is not the only such
    action**. A rally point is also no-type, non-placed, non-producing, and
    the first live probe took the first match and spent four unlock budgets
    setting rally points. Price is the reading that tells them apart: the
    engine's cost accessor is abstract on the action base class, a rally
    answers zero and the tier upgrade answers its tier's price, so the
    unlock is the no-type action that costs something -- and the claim is
    for the engine's own figure rather than a catalogue guess
    ([[mechanics-build-actions]]).

    Once per structure, for the produce-duplicate reason: the unlock never
    fills the queue, so the factory keeps offering it while it runs, and a
    duplicate arriving after completion would name an action that no longer
    exists.

    **At most ``limit`` structures, ever.** The unlock is per building, and
    the first one already opens production of the tier behind it -- the flag
    form bought all four factories' unlocks in one probe, 8,000 credits of
    saving pauses for a roster the first 2,000 had opened. How many
    factories' throughput the heavy mix deserves is the doctrine's question
    ([[policy-budget]]).

    **A refused unlock saves toward itself**, and this is the gated use the
    income-conversion refutation reserved the mechanism for
    (:meth:`~rw_bot.policy.budget.Budget.withhold`). Without it the unlock
    is unreachable: measured live, ``tech:landFactory asked 37 got 0`` --
    every spender drained the balance to the reserve each tick, so 2,000
    never accumulated (log 2026-07-31). Unlike the income ladder, which
    paused the army once per conversion across an unbounded run of them,
    this is a single bounded purchase per factory, and it is the whole
    point of the arm that carries it.

    Args:
        sample: One observation of the world.
        budget: The tick's credits.
        ordered: Factories already told to unlock, extended in place.
        limit: The most factories the arm ever unlocks.

    Returns:
        The ability orders to send, in roster order.
    """
    offers: dict[int, tuple[str, int]] = {}
    for option in sample["options"]:
        if (
            option["produces"] == ""
            and option["available"]
            and not option["placed"]
            and not option["makes_something"]
            and option["price"] > 0
            and option["key"] != ""
            and option["unit_id"] not in offers
        ):
            offers[option["unit_id"]] = (option["key"], option["price"])
    orders: list[AbilityOrder] = []
    for entity in sample["entities"]:
        if len(ordered) >= limit:
            break
        if not entity["mine"] or not entity["complete"] or entity["queued"] != 0:
            continue
        if entity["type_name"] not in TECH_TYPES or entity["unit_id"] in ordered:
            continue
        offer = offers.get(entity["unit_id"])
        if offer is None:
            continue
        key, price = offer
        claim = budget.claim(f"tech:{entity['type_name']}", price)
        if not claim["granted"]:
            budget.withhold(price)
            break
        ordered.add(entity["unit_id"])
        orders.append(ability_order(unit_id=entity["unit_id"], key=key))
    return tuple(orders)


def worker_need(
    free: Sequence[Entity],
    workers: int,
    available: int,
    catalogue: Mapping[str, UnitStats],
    ceiling: int,
) -> tuple[str, ...]:
    """Return where a builder belongs in the production order, if anywhere.

    **Another worker is bought on the same test another factory is.** A worker
    used to be asked for only when *none* was alive, so the bot played every
    match with the single builder it spawned with -- and that one unit was the
    sole channel through which credits became structures. It walked to each pool
    in turn while the bank grew at 63 credits a second against a spend rate of
    26 ([[policy-production]]).

    **"All of them are busy" is not a shortage; it is success.** That was the
    first test tried here and it bought 33 workers in a 1500-sample match --
    16,500 credits of labour to place 13 extractors, with 80% of the reported
    army value turning out to be builders rather than anything that fights
    ([[policy-production]]). A healthy economy has every worker busy nearly all
    the time, and this map carries 46 pools, so unclaimed work never runs out
    and the rule never stopped buying.

    So a ceiling is passed in. It is a number set by measurement rather than
    derived, and it is a parameter rather than a constant for exactly that
    reason: the value that wins is the one the A/B picks, not the one the
    argument here picks.

    **A spare worker used to be a fallback, and a fallback is unreachable
    here.** The reasoning for it was sound in isolation: the army mix is a
    ratio, so anything inside it is owed a share of the roster, and a builder
    owed a share is a land factory spending the match making builders. So a
    spare was offered only to a producer that could make *nothing* in the
    composition.

    Nothing on this build is ever in that position except the Command Center.
    A land factory can always make a ``c_tank``, so it never falls through, and
    the spare was therefore only ever buildable by one structure. That is why a
    traced match shows **98 builders lost and the worker count never
    accumulating**, and why the producer count never passed two: the bot was
    replacing its single builder over and over and never running a second one
    ([[policy-holding-ground]]).

    The same defect in its acute form lost three matches outright. When the
    Command Center died, the twenty-two Land Factories that can each build a
    builder went on building tanks forever, and the run ended ``plan blocked:
    nothing the player owns can make extractorT1`` with ``workers 0``.

    **A builder therefore goes into the composition, and the ceiling is what
    bounds it.** The objection above is answered by the ceiling rather than by
    a separate channel: below it a builder is wanted and any producer that can
    make one will, at it the type leaves the composition entirely and the mix
    returns to the army. An earlier attempt at "all of them are busy means buy
    another" bought 33 workers, but that ran with no ceiling at all
    ([[policy-production]]).

    This is what the engine's own AI does, and it is not close. It runs several
    bases, each an independent site targeting **two** builders, each claiming
    what is near it -- roughly a dozen builders working in parallel against our
    one ([[ai-opponent-strategy]]). One builder serialises the whole economy:
    it can only ever be walking to one pool, which is also why re-prioritising
    its time cannot help and was measured not to ([[policy-production]]).

    Args:
        free: Workers not already carrying out an order. One standing idle is
            not a shortage, so none is bought while any is free.
        workers: How many workers are owned at all.
        available: Credits still unclaimed after the higher-priority spenders.
        catalogue: Unit stats by type name, for the worker's price.
        ceiling: The most workers worth holding. Set by measurement, and passed
            in rather than named here so an A/B decides it.

    Returns:
        The types to prepend to the army composition this observation, empty
        when no builder is wanted.
    """
    if workers == 0:
        # The economy is over unless this is answered, so it outranks the army
        # and is bought regardless of surplus: there is nothing else worth
        # spending on that does not need a worker first.
        return (BUILDER_TYPE,)
    stats = catalogue.get(BUILDER_TYPE)
    if workers >= ceiling or free or stats is None or available < stats["price"]:
        return ()
    return (BUILDER_TYPE,)


def replace_losses(
    sample: Sample,
    catalogue: Mapping[str, UnitStats],
    budget: Budget,
    composition: Sequence[str],
) -> tuple[ProduceOrder, ...]:
    """Decide what idle producers should make of what the plan keeps wanting.

    Claims are protected: replacing a loss is what the reserve is held for.
    Ordering stops at the first refusal rather than skipping to a cheaper
    producer, because buying the thing nobody asked for on the grounds that the
    wanted one is unaffordable is a decision nobody asked for either.

    Args:
        sample: One observation of the world.
        catalogue: Unit stats by type name, for prices.
        budget: The tick's credits.
        composition: The army mix to hold, repeats meaningful as a ratio. A
            wanted builder is part of it rather than a separate channel; see
            :func:`worker_need` for why the separate channel was unreachable.

    Returns:
        The produce orders to send, in preference order and stopping at the
        first the budget refused.
    """
    orders: list[ProduceOrder] = []
    for order in sustain(sample, catalogue, composition):
        claim = budget.claim(f"produce:{order['type_name']}", order["price"], protected=True)
        if not claim["granted"]:
            break
        orders.append(produce_order(unit_id=order["unit_id"], type_name=order["type_name"]))
    return tuple(orders)


class PlanStep(TypedDict):
    """What the opening plan decided on one observation.

    Attributes:
        outcome: How the plan stands.
        reason: The plan's own words for why.
        holds_worker: The worker the plan has taken this observation, zero when
            it has taken none. The economy needs it: two callers would otherwise
            both order the same builder, the engine runs whichever waypoint
            arrived last, and neither order arrives ([[policy-loop]]).

            **A count, not a flag, and the difference was most of the economy.**
            This was ``claims_builder: bool`` and the expander answered it by
            switching itself off entirely -- every spender, however many workers
            were free. Instrumented, that skipped the expander on 572 of 800
            samples while six workers stood idle. Naming the one worker lets the
            rest be used ([[policy-economy]]).
        wants_worker: Whether the plan is waiting for the next worker to
            free. The signal that gives the plan worker priority to match its
            credit priority: without it, a rich match kept all eight workers
            employed on defence, the plan's factory never met a free worker,
            and the army was never built -- wins 1/10 at a rung the same
            doctrine had won 10/12 (log: 2026-07-31). The expander stands
            down while this is set, so the next freed worker is the plan's.
        build: The structure to place, or None.
        produce: The unit to queue, or None.
    """

    outcome: str
    reason: str
    holds_worker: int
    wants_worker: bool
    build: BuildOrder | None
    produce: ProduceOrder | None


def build_plan(
    sample: Sample,
    tracker: OrderTracker,
    budget: Budget,
    catalogue: Mapping[str, UnitStats],
    placements: Mapping[str, TypePlacement],
    profiles: Mapping[str, CombatProfile],
    free: Sequence[Entity],
    workforce: Workforce,
) -> PlanStep:
    """Advance the opening plan by at most one order.

    Claimed first and protected, because the plan's prerequisites gate
    everything else: a factory that is not built is production that cannot
    happen, however many credits the economy earns.

    Args:
        sample: One observation of the world.
        tracker: What has already been ordered, and whether it moved.
        budget: The tick's credits.
        catalogue: Unit stats by type name, for prices.
        placements: Placement rules by type name.
        profiles: Combat profiles by type name, for the threat filter.
        free: Workers not already carrying out an order.
        workforce: Told what the plan sent a worker to build.

    Returns:
        How the plan stands, and at most one order to send for it.
    """
    decision = decide(
        sample, tracker.plan, catalogue, placements, profiles, free, workforce.claims()
    )
    # Movement is judged per worker now, so the plan's own stall clock asks the
    # workforce whether the unit it ordered is the one that is walking.
    step = tracker.assess(sample, decision, workforce.working(decision["unit_id"]))
    # The plan holds the builder for as long as it wants something placed,
    # including while it is merely waiting to afford it. Acting on that is what
    # keeps the two off each other: a live run had the economy re-tasking the
    # builder to its own pool between the plan's own extractors, and the engine
    # runs the newest waypoint, so neither order arrived ([[policy-loop]]).
    # **Which worker, not whether.** A "wait" now names the unit it is holding
    # (:class:`~rw_bot.policy.build_order.Decision`), so the economy can skip
    # that one worker instead of standing down entirely.
    # **And only while the plan is live.** A blocked ruling exists to end a
    # hostage situation -- the savings clock's whole point is that the worker
    # goes back to the economy, so a hold that survived the ruling would undo
    # the fix ([[policy-economy]]). The tracker lifts the ruling on its own
    # when saving resumes, and the hold comes back with it.
    holds_worker = (
        decision["unit_id"]
        if decision["action"] in ("build", "wait") and step["outcome"] == "building"
        else 0
    )
    # The every-capable-unit-is-busy wait is the one wait that names no unit:
    # there is no specific worker to hold, the plan wants WHICHEVER frees
    # first, and saying so is what lets the campaign grant it one.
    wants_worker = decision["action"] == "wait" and decision["unit_id"] == 0
    if not step["act"]:
        return PlanStep(
            outcome=step["outcome"],
            reason=step["reason"],
            holds_worker=holds_worker,
            wants_worker=wants_worker,
            build=None,
            produce=None,
        )

    # Claimed, not tested. The plan claims first and protected, and
    # :func:`~rw_bot.policy.build_order.decide` has already refused to act on a
    # price the sample's own balance cannot cover -- so this claim cannot be
    # refused. It is still made, because committing the credits is what stops
    # the later claimants spending them twice, which is the whole point of the
    # arbiter ([[policy-budget]]).
    stats = catalogue[decision["type_name"]]
    budget.claim(f"plan:{decision['type_name']}", stats["price"], protected=True)

    if decision["action"] == "produce":
        return PlanStep(
            outcome=step["outcome"],
            reason=step["reason"],
            holds_worker=holds_worker,
            wants_worker=wants_worker,
            build=None,
            produce=produce_order(unit_id=decision["unit_id"], type_name=decision["type_name"]),
        )
    workforce.assign(decision["unit_id"], decision["type_name"], (decision["x"], decision["y"]))
    return PlanStep(
        outcome=step["outcome"],
        reason=step["reason"],
        holds_worker=holds_worker,
        wants_worker=wants_worker,
        build=build_order(
            unit_id=decision["unit_id"],
            type_name=decision["type_name"],
            x=decision["x"],
            y=decision["y"],
        ),
        produce=None,
    )


__all__ = [
    "PlanStep",
    "build_plan",
    "replace_losses",
    "upgrade_income",
    "worker_need",
]
