"""The economy through the loop: pools first, throughput from surplus,
defence last, and production that stops at the first refusal.

The spending order is the policy, so these drive whole ticks and read what
was bought -- a unit test of the priorities would only restate the call
order it is meant to check.
"""

from __future__ import annotations

from pathlib import Path

from rw_bot.control.channel import AgentChannel
from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.mechanics.placement import TypePlacement
from rw_bot.policy.campaign import play
from rw_bot.policy.match_report import format_report
from tests.campaign_fixtures import (
    BUILDER,
    CATALOGUE,
    CENTRE,
    ENEMY,
    FACTORY,
    PLACEMENTS,
    PROFILES,
    THEM,
    US,
    WAVE,
    ScriptedPeer,
    defence_world,
    order_lines,
    run_campaign,
    unit_stats,
    verb,
)
from tests.wire_fixtures import (
    enemy,
    entity,
    lines,
    option,
    pool,
    sample,
)


def test_a_claimable_pool_still_outranks_covering_a_structure() -> None:
    """Defence takes the surplus, not the income.

    Covering a structure was tried *ahead* of claiming a pool, on the reasoning
    that a turret is cheaper than the extractor it covers and 247 expansion
    orders were leaving one extractor standing. Measured, it lost every match
    -- four defeats out of
    four against two survivals -- because there is always some uncovered
    structure, so the rule took the builder nearly every tick: expansion
    collapsed from 275 orders to about 40 and income never grew
    ([[policy-holding-ground]]).
    """
    catalogue, placements, profiles = defence_world()
    world = sample(
        CENTRE,
        entity(214, "builder", x=0.0, y=0.0),
        credits=4000,
        pools=(pool(x=60.0, y=0.0),),
        options=(option(214, "c_turret_t1"), option(214, "extractorT1")),
    )
    peer = ScriptedPeer(lines(world))
    play(AgentChannel(peer), (), catalogue, placements, profiles, 1, expand=True)
    built = [line for line in order_lines(peer) if '"kind":"build"' in line]
    assert built == ['{"kind":"build","unit_id":214,"x":60.0,"y":0.0,"type":"extractorT1"}']


def test_a_spare_builder_buys_throughput_without_costing_the_pool() -> None:
    """The fault that left duels unfinished with the bank full.

    Matches ended with a completed plan, an army of 26 and five extractors --
    and **44,660 credits banked against a single factory**, having knocked an
    opponent from a peak of 37,750 down to 6,650 without finishing it
    ([[policy-holding-ground]]). One worker now buys the capacity to spend
    that, and the others keep claiming pools, so the tick produces both orders
    rather than choosing between them.
    """
    catalogue, placements, profiles = defence_world()
    world = sample(
        CENTRE,
        entity(214, "builder", x=0.0, y=0.0),
        entity(215, "builder", x=20.0, y=0.0),
        # The only producer of a wanted type, and it is busy -- which is what
        # `production_bound` asks before calling throughput the constraint.
        entity(300, "landFactory", x=200.0, y=0.0, queued=1),
        credits=40_000,
        pools=(pool(x=60.0, y=0.0),),
        options=(
            # Placed, which is what confines a structure to a *free* worker --
            # the whole reason diverting one is safe.
            option(214, "landFactory", placed=True),
            option(214, "extractorT1", placed=True),
            option(215, "landFactory", placed=True),
            option(215, "extractorT1", placed=True),
            option(300, "c_tank", placed=False),
        ),
    )
    peer = ScriptedPeer(lines(world))
    play(
        AgentChannel(peer),
        (),
        catalogue,
        placements,
        profiles,
        1,
        reinforce=("c_tank",),
        expand=True,
    )
    built = [line for line in order_lines(peer) if '"kind":"build"' in line]
    assert any('"type":"landFactory"' in line for line in built)
    assert any('"type":"extractorT1"' in line for line in built)


def test_a_lone_builder_is_never_diverted_to_throughput() -> None:
    """The guard the earlier attempt at this lacked.

    Reordering the chain to put throughput first was the worst arm measured --
    three wiped and three defeated, expansion collapsing from 307-509 orders to
    2-6 -- because there was **one** builder, so every factory it placed was an
    extractor it did not ([[policy-production]]). With a single worker free,
    the pool still wins.
    """
    catalogue, placements, profiles = defence_world()
    world = sample(
        CENTRE,
        entity(214, "builder", x=0.0, y=0.0),
        credits=40_000,
        pools=(pool(x=60.0, y=0.0),),
        options=(option(214, "landFactory"), option(214, "extractorT1")),
    )
    peer = ScriptedPeer(lines(world))
    play(AgentChannel(peer), (), catalogue, placements, profiles, 1, expand=True)
    built = [line for line in order_lines(peer) if '"kind":"build"' in line]
    assert built == ['{"kind":"build","unit_id":214,"x":60.0,"y":0.0,"type":"extractorT1"}']


def test_defence_takes_the_surplus_when_no_pool_can_be_claimed() -> None:
    """Income compounds and defence does not, so income keeps its place. What
    defence takes is the surplus that was otherwise buying a twenty-second Land
    Factory -- a trade between two things that both fail to compound
    ([[policy-production]]).
    """
    catalogue, placements, profiles = defence_world()
    world = sample(
        CENTRE,
        entity(214, "builder", x=0.0, y=0.0),
        credits=4000,
        # No pools at all, so income has nothing left to claim and the surplus
        # is what defence is spending.
        options=(option(214, "c_turret_t1"),),
    )
    peer = ScriptedPeer(lines(world))
    play(AgentChannel(peer), (), catalogue, placements, profiles, 1, expand=True)
    # The Command Center is nearest the anchor and is itself uncovered, so it is
    # what gets covered. Aiming this at the extractors instead was measured and
    # lost -- wins 4 -> 0 over the same twelve seeds ([[policy-holding-ground]]).
    assert [line for line in order_lines(peer) if '"kind":"build"' in line] == [
        '{"kind":"build","unit_id":214,"x":60.0,"y":0.0,"type":"c_turret_t1"}'
    ]


def test_the_engine_clock_is_reported_beside_the_frame_count() -> None:
    """The pair answers what neither can alone: whether the simulation advances
    per frame or per wall clock. The engine caps itself at 300 frames a second
    and matches run at about 297, so if the clock outruns the wall the cap is a
    real throughput ceiling and if it tracks the wall then removing it would buy
    nothing ([[harness-parallel-matches]]).
    """
    peer = ScriptedPeer(
        lines(
            sample(CENTRE, *WAVE, frame=100, clock_ms=1_000),
            sample(CENTRE, *WAVE, frame=400, clock_ms=6_000),
        )
    )
    report = play(AgentChannel(peer), (), CATALOGUE, PLACEMENTS, PROFILES, 2)
    assert report["frames_elapsed"] == 300
    assert report["clock_elapsed_ms"] == 5_000


def test_what_the_opponents_field_is_reported() -> None:
    """A whole tier of the game turns on this and nothing else can see it.

    Unit types declare a ``techLevel``, and a type's build action is registered
    only into the action lists at or above that level -- so at tech 1 a tier-2
    action is absent rather than refused, which is why an owned extractor
    offers nothing. Whether the *opponents* hold tier-2 types is therefore the
    difference between the bot playing the same game badly and the bot playing
    a smaller game ([[policy-holding-ground]]).
    """
    world = sample(
        CENTRE,
        *WAVE,
        enemy(9, "c_tank", x=100.0),
        enemy(10, "c_tank", x=120.0),
        enemy(11, "extractorT2", x=140.0),
    )
    report, _ = run_campaign(world, times=1)
    assert report["enemy_types_end"] == (("c_tank", 2), ("extractorT2", 1))


def test_an_unseen_enemy_is_reported_as_none_rather_than_blank() -> None:
    """Nothing visible is a real observation, not a missing measurement."""
    report, _ = run_campaign(sample(CENTRE, *WAVE), times=1)
    assert report["enemy_types_end"] == ()
    assert "enemy fields   none" in format_report(report)


def _upgrade_world() -> tuple[dict[str, UnitStats], dict[str, TypePlacement]]:
    """A catalogue that prices the upgrade the way the engine prices it.

    **The two prices are deliberately different, because in the game they are.**
    This fixture used to give ``extractorT2`` a build price of 1,400 -- the cost
    of the *conversion* -- which made the loop's old reading, claiming the
    target's build price, look correct. The engine's dump says ``Price: $2100``
    for a tier two and ``T2 Upgrade Price: $1400`` on the tier one, so the
    fixture described a world the game cannot produce and the test could not
    have caught the substitution ([[mechanics-unit-value]]).
    """
    catalogue = {
        **CATALOGUE,
        "extractorT1": unit_stats("extractorT1", speed=0.0, price=700, upgrade_prices=(1400,)),
        "extractorT2": unit_stats("extractorT2", speed=0.0, price=2100),
    }
    placements = {
        name: TypePlacement(index=i, type_name=name, needs_pool=name.startswith("extractor"))
        for i, name in enumerate(catalogue)
    }
    return catalogue, placements


def test_an_extractor_is_told_to_upgrade_itself() -> None:
    """The income the map cannot take away.

    An extractor converting itself needs no builder, crosses no contested
    ground and claims no new pool -- which is what matters on a map where the
    opponents finish holding 44 of the 46 pools and 247 expansion orders leave
    the bot with one extractor. It was invisible until the agent stopped
    dropping actions that neither place nor "make something"
    ([[policy-holding-ground]]).
    """
    catalogue, placements = _upgrade_world()
    world = sample(
        CENTRE,
        entity(400, "extractorT1"),
        credits=4000,
        options=(option(400, "extractorT2", placed=False, makes_something=False),),
    )
    peer = ScriptedPeer(lines(world))
    play(AgentChannel(peer), (), catalogue, placements, PROFILES, 1)
    assert verb(peer, "produce") == ['{"kind":"produce","unit_id":400,"type":"extractorT2"}']


def test_an_owned_extractor_is_upgraded_before_a_free_pool_is_claimed() -> None:
    """The order the arithmetic argues against and the matches preferred.

    A new extractor is 700 for +8 credits a second; converting one is 1,400 for
    +4 and then 4,000 for +8, so pools are six times better per credit --
    ``#price per credit: $87`` against ``$800`` in the game's own assets -- and
    [[policy-economy]] states the rule outright: take every free pool before
    upgrading anything.

    **Reordered on exactly that arithmetic, it measured worse.** Twelve seeds at
    Very Hard: 7 won with upgrades first against 5 with expansion first, the
    same two losses, routs 3 -> 2, median win 2,207 -> 2,362. That sits inside
    the noise floor, so it refutes nothing -- but it is not the improvement the
    per-credit figure promised either, and two weak signals pointing the same
    way is what this decision rests on.

    **What the arithmetic leaves out is risk**, and risk is the one thing every
    rung of this ladder turns on: matches are decided by extractors *lost*, with
    winners dropping nought to four and the rest six or more
    ([[policy-holding-ground]]). A new extractor is income that can be
    destroyed; a conversion is income on ground already held. Six times the
    price for income that cannot be taken away is a different trade from six
    times the price for nothing.

    Nothing pinned this order before -- swapping the two calls broke no test at
    all -- which is why it is pinned here now, with the measurement behind it.
    """
    catalogue, placements = _upgrade_world()
    world = sample(
        CENTRE,
        BUILDER,
        entity(400, "extractorT1", x=900.0, y=0.0),
        credits=1500,
        pools=(pool(x=300.0),),
        options=(
            option(214, "extractorT1", placed=True),
            option(400, "extractorT2", placed=False, makes_something=False),
        ),
    )
    peer = ScriptedPeer(lines(world))
    play(AgentChannel(peer), (), catalogue, placements, PROFILES, 1)
    assert verb(peer, "produce") == ['{"kind":"produce","unit_id":400,"type":"extractorT2"}']
    # Expansion is not disabled, only outranked: 1,500 covered the conversion
    # and what was left could not also cover a 700 extractor.
    assert verb(peer, "build") == []


def test_an_upgrade_outranks_replacing_the_army() -> None:
    """Income before army, the third ordering this pair has held.

    Production-first left the tier-three conversion asked ~1,800 times a
    match and granted never, while produce drained every credit above the
    reserve into units that traded even against a 1.8x income and
    equilibrated (log 2026-07-31). A conversion pays back inside the match
    on ground already held, and the reserve still protects the replacement
    of a loss -- upgrades claim past it -- so production is deferred, not
    starved. Nothing pinned this order before; it is pinned here with the
    reasoning.
    """
    catalogue, placements = _upgrade_world()
    world = sample(
        CENTRE,
        entity(400, "extractorT1"),
        entity(500, "landFactory"),
        credits=1400,
        options=(
            option(400, "extractorT2", placed=False, makes_something=False),
            option(500, "c_tank"),
        ),
    )
    peer = ScriptedPeer(lines(world))
    play(
        AgentChannel(peer),
        (),
        catalogue,
        placements,
        PROFILES,
        1,
        reinforce=("c_tank",),
    )
    # 1,400 covered the conversion; what remained could not also cover a
    # 350-credit tank, so the army waits one observation.
    assert verb(peer, "produce") == ['{"kind":"produce","unit_id":400,"type":"extractorT2"}']


def test_a_refused_upgrade_does_not_hold_the_armys_credits() -> None:
    """The saving that was built, measured, and taken back out.

    A refused conversion once withheld its price from every later claim, and
    the mechanism did what it promised: six T2 and six T3 conversions a
    match, income doubled to 98/s -- and the army pauses it forced let the
    enemy's economy live untouched, rival scores rising to the worst of the
    whole screening arc ([[policy-budget]], log 2026-07-31). Income that
    displaces early pressure is income for a longer strangle. So a refusal
    is again just a refusal, and the tank the credits do cover is bought.
    """
    catalogue, placements = _upgrade_world()
    world = sample(
        CENTRE,
        entity(400, "extractorT1"),
        entity(500, "landFactory"),
        credits=500,
        options=(
            option(400, "extractorT2", placed=False, makes_something=False),
            option(500, "c_tank"),
        ),
    )
    peer = ScriptedPeer(lines(world))
    play(
        AgentChannel(peer),
        (),
        catalogue,
        placements,
        PROFILES,
        1,
        reinforce=("c_tank",),
    )
    assert verb(peer, "produce") == ['{"kind":"produce","unit_id":500,"type":"c_tank"}']


def test_tech_unlocks_the_factory_and_off_leaves_it_locked() -> None:
    """The tech verb end to end: a factory offering its no-type unlock is
    told to fire it, once, at the price the wire itself carries -- and the
    same world with the flag off sends nothing ([[mechanics-build-actions]]).
    """
    catalogue = {
        **CATALOGUE,
        "landFactory": unit_stats("landFactory", speed=0.0, price=700, upgrade_prices=(2000,)),
    }
    placements = {
        name: TypePlacement(index=i, type_name=name, needs_pool=name.startswith("extractor"))
        for i, name in enumerate(catalogue)
    }
    world = sample(
        CENTRE,
        entity(500, "landFactory"),
        credits=4000,
        options=(option(500, "", key="c_2", placed=False, makes_something=False, price=2000),),
    )
    peer = ScriptedPeer(lines(world, world))
    play(AgentChannel(peer), (), catalogue, placements, PROFILES, 2, tech=True)
    fired = [line for line in peer.sent if '"ability"' in line]
    # Once, not once per observation: the unlock never fills the queue.
    assert fired == ['{"kind":"ability","unit_id":500,"key":"c_2"}']

    held = ScriptedPeer(lines(world))
    play(AgentChannel(held), (), catalogue, placements, PROFILES, 1)
    assert [line for line in held.sent if '"ability"' in line] == []


def test_an_upgrade_is_ordered_once_rather_than_every_observation() -> None:
    """A conversion never fills the production queue.

    ``queued`` stays at zero for as long as the conversion runs, so the
    structure keeps offering the upgrade it is already performing. Re-ordering
    it every observation sent a stream of duplicates, and one arrived after the
    conversion had finished -- addressed to a unit that was now an
    ``extractorT2`` and could only make an ``extractorT3``. The agent refuses an
    order naming something its subject cannot make, and that refusal crashed the
    match ([[policy-holding-ground]]).
    """
    catalogue, placements = _upgrade_world()
    world = sample(
        CENTRE,
        entity(400, "extractorT1"),
        credits=40000,
        options=(option(400, "extractorT2", placed=False, makes_something=False),),
    )
    peer = ScriptedPeer(lines(world, world, world, world))
    play(AgentChannel(peer), (), catalogue, placements, PROFILES, 4)
    assert verb(peer, "produce") == ['{"kind":"produce","unit_id":400,"type":"extractorT2"}']


def test_upgrades_stop_at_the_first_refusal_rather_than_overdrawing() -> None:
    """Every upgrade costs the same, so a refusal means the budget is out."""
    catalogue, placements = _upgrade_world()
    world = sample(
        CENTRE,
        entity(400, "extractorT1"),
        entity(401, "extractorT1"),
        credits=1400,
        options=(
            option(400, "extractorT2", placed=False, makes_something=False),
            option(401, "extractorT2", placed=False, makes_something=False),
        ),
    )
    peer = ScriptedPeer(lines(world))
    play(AgentChannel(peer), (), catalogue, placements, PROFILES, 1)
    assert len(verb(peer, "produce")) == 1


def test_an_upgrade_the_catalogue_cannot_price_is_never_ordered() -> None:
    """Unpriced means unbudgetable, and spending blind is what the budget
    prevents ([[policy-budget]]).
    """
    world = sample(
        CENTRE,
        entity(400, "extractorT1"),
        credits=4000,
        options=(option(400, "extractorT2", placed=False, makes_something=False),),
    )
    # CATALOGUE does not price extractorT2.
    _, peer = run_campaign(world, times=1)
    assert verb(peer, "produce") == []


def test_the_reserve_gathers_at_the_base() -> None:
    """Units waiting near the base are the only defensive posture the bot has."""
    far = entity(1, "c_tank", x=900.0)
    _, peer = run_campaign(sample(CENTRE, far, ENEMY), times=2)
    assert '{"kind":"move","unit_id":1,"x":0.0,"y":0.0}' in verb(peer, "move")


def test_a_unit_already_at_the_base_is_not_told_to_go_there() -> None:
    report, _ = run_campaign(sample(CENTRE, entity(1, "c_tank", x=10.0), ENEMY), times=2)
    assert report["rallied"] == 0


def test_a_lost_builder_is_replaced_before_the_economy_dies() -> None:
    """A lost builder ends the economy permanently, so one is asked for."""
    world = sample(
        CENTRE,
        credits=4000,
        options=(option(213, "builder"),),
    )
    _, peer = run_campaign(world, times=1)
    assert verb(peer, "produce") == ['{"kind":"produce","unit_id":213,"type":"builder"}']


def test_a_factory_builds_the_last_builder_rather_than_another_tank() -> None:
    """The case that lost three matches, and that no fixture had covered.

    Every world here gave the builder option to the Command Center alone, so
    the fallback -- reached only by a producer that can make nothing in the army
    mix -- always fired. A **Land Factory can make both** a tank and a builder,
    and it can always make a tank, so it never falls through. When the Command
    Center died, twenty-two factories went on building tanks while the player
    had no builder: no further extractor, no replacement factory, and no way
    back. The runs end ``plan blocked: nothing the player owns can make
    extractorT1`` with ``workers 0`` and a defeat ([[policy-production]]).
    """
    world = sample(
        FACTORY,
        credits=4000,
        options=(option(300, "c_tank"), option(300, "builder")),
    )
    _, peer = run_campaign(world, times=1, reinforce=("c_tank",))
    assert verb(peer, "produce") == ['{"kind":"produce","unit_id":300,"type":"builder"}']


def test_a_factory_stays_on_the_army_while_a_builder_is_alive() -> None:
    """The emergency is having none, not having few.

    Otherwise the fix trades one runaway for another: a builder inside the mix
    permanently is a factory spending the match on builders, which is the
    33-worker run in a different disguise ([[policy-production]]).
    """
    world = sample(
        BUILDER,
        FACTORY,
        credits=4000,
        options=(option(300, "c_tank"), option(300, "builder")),
    )
    _, peer = run_campaign(world, times=1, reinforce=("c_tank",))
    assert verb(peer, "produce") == ['{"kind":"produce","unit_id":300,"type":"c_tank"}']


def test_the_builder_goes_last_so_factories_stay_on_tanks() -> None:
    """A producer takes the first type it can make, and only the command centre
    -- which cannot make a tank -- falls through to the builder.
    """
    world = sample(
        CENTRE,
        FACTORY,
        credits=4000,
        options=(option(300, "c_tank"), option(213, "builder")),
    )
    _, peer = run_campaign(world, times=1, reinforce=("c_tank",))
    # Roster order, not preference order: what preference decides is *what each
    # producer makes*, and the factory took the tank rather than falling through
    # to the builder the command centre ends up with.
    assert verb(peer, "produce") == [
        '{"kind":"produce","unit_id":213,"type":"builder"}',
        '{"kind":"produce","unit_id":300,"type":"c_tank"}',
    ]


def test_an_unreachable_enemy_is_counted_apart_from_a_visible_one() -> None:
    """The gap between the two is the diagnosis: an army holding the wrong units."""
    flyer = enemy(9, "helicopter", x=100.0, flying=True)
    profiles = {**PROFILES, "helicopter": PROFILES["c_tank"]}
    peer = ScriptedPeer(lines(sample(CENTRE, *WAVE, flyer)))
    report = play(AgentChannel(peer), (), CATALOGUE, PLACEMENTS, profiles, 1)
    assert report["targets_end"] == 1
    assert report["engageable_end"] == 0
    assert verb(peer, "attack") == []


def test_every_observation_is_acknowledged() -> None:
    """In lockstep the agent holds the simulation until the ack arrives."""
    _, peer = run_campaign(sample(CENTRE, *WAVE, ENEMY), times=4)
    assert len([line for line in peer.sent if '"kind":"ack"' in line]) == 4


def test_the_trace_is_written_when_a_path_is_given(tmp_path: Path) -> None:
    target = tmp_path / "trace.txt"
    run_campaign(sample(CENTRE, *WAVE, ENEMY), times=2, trace=target)
    written = target.read_text(encoding="utf-8")
    assert "frame" in written
    assert "army" in written


def test_the_report_renders_as_lines() -> None:
    report, _ = run_campaign(sample(CENTRE, *WAVE, ENEMY, players=(US, THEM)), times=2)
    rendered = format_report(report)
    assert rendered[0].startswith("verdict")
    assert any("best rival     5700 -> 5700" in line for line in rendered)


def test_throughput_is_bought_once_the_map_has_no_pool_left() -> None:
    """Income first, because income compounds and throughput does not.

    Buying factories ahead of pools takes the builder away from the only asset
    that grows: measured on one seed, 4 extractors with 3 factories produced 62
    units and an army worth 6,450, against 9 extractors with 1 factory producing
    28 units and an army worth 8,200 ([[policy-production]]). So the surplus
    buys throughput only when there is no pool left to claim -- which is what
    this world is, its one pool already built on.
    """
    world = sample(
        CENTRE,
        BUILDER,
        entity(300, "landFactory", queued=1),
        entity(400, "extractorT1", x=300.0),
        credits=4000,
        pools=(pool(x=300.0),),
        options=(
            option(300, "c_tank"),
            option(214, "landFactory", placed=True),
            option(214, "extractorT1", placed=True),
        ),
    )
    report, peer = run_campaign(world, times=1, reinforce=("c_tank",))
    assert verb(peer, "build") == [
        '{"kind":"build","unit_id":214,"x":200.0,"y":120.0,"type":"landFactory"}'
    ]
    assert "landFactory" in report["expand_reason"]
    assert report["expanded_factories"] == 1


def test_throughput_is_not_bought_when_nothing_is_wanted() -> None:
    """More capacity to make nothing is a spend with no return.

    The qualifier that made this rule work at all: a producer idle on a type
    nobody wants is not spare capacity, and a producer busy on one is not a
    constraint ([[policy-production]]).
    """
    world = sample(
        CENTRE,
        BUILDER,
        entity(300, "landFactory", queued=1),
        credits=4000,
        pools=(pool(x=300.0),),
        options=(
            option(300, "c_tank"),
            option(214, "landFactory", placed=True),
            option(214, "extractorT1", placed=True),
        ),
    )
    _, peer = run_campaign(world, times=1)
    assert verb(peer, "build") == [
        '{"kind":"build","unit_id":214,"x":300.0,"y":0.0,"type":"extractorT1"}'
    ]


def test_an_expansion_order_is_not_repeated_at_sample_rate() -> None:
    """The builder has been told; re-sending every observation resets the walk."""
    world = sample(
        CENTRE,
        BUILDER,
        credits=4000,
        pools=(pool(x=300.0),),
        options=(option(214, "extractorT1", placed=True),),
    )
    _, peer = run_campaign(world, times=6)
    assert len(verb(peer, "build")) == 1


def test_the_plan_waits_when_the_army_has_taken_the_credits() -> None:
    """The plan waits on price rather than issuing an order it cannot pay for.

    Decided before the budget is opened at all: the plan is the first claimant
    and protected, so a claim it makes cannot be refused. What stops it here is
    its own affordability check against the same balance.
    """
    world = sample(
        CENTRE,
        BUILDER,
        credits=100,
        options=(option(214, "landFactory", placed=True),),
    )
    report, peer = run_campaign(world, times=1, plan=("landFactory",))
    assert verb(peer, "build") == []
    assert report["build_reason"] == "landFactory costs 1000, holding 100"


def test_production_stops_at_the_first_claim_it_cannot_meet() -> None:
    """Preference order is what makes dropping the tail meaningful.

    Two factories, one tank's worth of credits: the first is ordered and the
    second is not, rather than an arbitrary one of the two.
    """
    world = sample(
        CENTRE,
        entity(300, "landFactory"),
        entity(301, "landFactory"),
        credits=350,
        options=(option(300, "c_tank"), option(301, "c_tank")),
    )
    _, peer = run_campaign(world, times=1, reinforce=("c_tank",))
    assert verb(peer, "produce") == ['{"kind":"produce","unit_id":300,"type":"c_tank"}']


def test_a_produced_plan_entry_is_queued_rather_than_placed() -> None:
    """A unit rolls out of the building that made it; the planner sites nothing."""
    world = sample(
        CENTRE,
        BUILDER,
        FACTORY,
        credits=4000,
        options=(option(300, "c_tank"),),
    )
    _, peer = run_campaign(world, times=1, plan=("c_tank",))
    assert verb(peer, "produce") == ['{"kind":"produce","unit_id":300,"type":"c_tank"}']
