"""The economy through the loop: pools first, throughput from surplus,
defence last, and production that stops at the first refusal.

The spending order is the policy, so these drive whole ticks and read what
was bought -- a unit test of the priorities would only restate the call
order it is meant to check.

Converting a structure into its next tier is the same contest for the same
credits and lives in ``test_campaign_upgrades``.
"""

from __future__ import annotations

from pathlib import Path

from rw_bot.control.channel import AgentChannel
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
