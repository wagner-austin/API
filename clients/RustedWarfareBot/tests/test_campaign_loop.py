"""The one tick: every layer on every observation, and honest endings.

What to build, attack and claim are pure functions tested elsewhere; here it
is the loop around them -- credits arbitrated rather than raced, the fight and
the plan sharing one observation, the match ending when the engine says so,
and the report carrying what actually happened.
"""

from __future__ import annotations

from rw_bot.control.channel import AgentChannel
from rw_bot.mechanics.placement import TypePlacement
from rw_bot.policy.campaign import play
from rw_bot.policy.expander import economy_floor
from rw_bot.policy.match_report import format_report
from rw_bot.policy.scorekeeper import Scorekeeper
from rw_bot.wire.state import Sample
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
    run_campaign,
    unit_stats,
    verb,
)
from tests.wire_fixtures import (
    entity,
    lines,
    option,
    player,
    pool,
    profiles_for,
    sample,
)


def test_the_army_is_sent_at_the_enemy() -> None:
    report, peer = run_campaign(sample(*WAVE, ENEMY))
    assert verb(peer, "attack") == [
        '{"kind":"attack","unit_id":1,"target_id":9}',
        '{"kind":"attack","unit_id":2,"target_id":9}',
        '{"kind":"attack","unit_id":3,"target_id":9}',
    ]
    assert report["attack_orders"] == 3


def test_an_attack_is_not_reissued_while_it_stands() -> None:
    """The engine runs a waypoint until it is replaced, so a repeat resets it."""
    _, peer = run_campaign(sample(*WAVE, ENEMY), times=5)
    assert len(verb(peer, "attack")) == 3


def test_the_plan_and_the_fight_run_on_the_same_observation() -> None:
    """The seam this refactor removed.

    The old loop built to completion and only then fought, so the opening was
    played defenceless and the plan stopped the moment fighting began. Both act
    on one tick now ([[policy-loop]]).
    """
    world = sample(
        CENTRE,
        BUILDER,
        *WAVE,
        ENEMY,
        credits=4000,
        options=(option(214, "landFactory", placed=True),),
    )
    _, peer = run_campaign(world, plan=("landFactory",))
    assert verb(peer, "build")
    assert verb(peer, "attack")


def test_the_same_credit_is_not_committed_twice_in_one_tick() -> None:
    """The defect the arbiter exists for.

    One factory can start a 350 tank and the builder can place a 700 extractor,
    but not on 800 credits. Production claims first because it is protected;
    expansion is refused and says so.
    """
    world = sample(
        CENTRE,
        BUILDER,
        FACTORY,
        credits=800,
        pools=(pool(x=300.0),),
        options=(option(300, "c_tank"), option(214, "extractorT1", placed=True)),
    )
    report, peer = run_campaign(world, times=1, reinforce=("c_tank",))
    assert verb(peer, "produce") == ['{"kind":"produce","unit_id":300,"type":"c_tank"}']
    assert verb(peer, "build") == []
    assert report["refused_claims"] == 1
    assert "wanted 700" in report["expand_reason"]


def test_both_are_afforded_when_the_credits_are_there() -> None:
    """The complement, so the refusal above is arbitration and not a dead path."""
    world = sample(
        CENTRE,
        BUILDER,
        FACTORY,
        credits=4000,
        pools=(pool(x=300.0),),
        options=(option(300, "c_tank"), option(214, "extractorT1", placed=True)),
    )
    report, peer = run_campaign(world, times=1, reinforce=("c_tank",))
    assert verb(peer, "produce")
    assert verb(peer, "build")
    assert report["refused_claims"] == 0
    # Income, not throughput: the factory is idle, so more capacity buys nothing.
    assert report["expanded_factories"] == 0


def _race_world(extractors: int, pools_visible: int) -> Sample:
    """A duel_lake-shaped race: extractors standing on the first pools.

    Each owned extractor stands exactly on one pool, so the survey reads the
    rest as free and the census carries the map's true size.
    """
    sites = tuple(pool(index=n, x=300.0 + 200.0 * n) for n in range(pools_visible))
    standing = tuple(entity(400 + n, "extractorT1", x=300.0 + 200.0 * n) for n in range(extractors))
    return sample(
        CENTRE,
        BUILDER,
        *standing,
        credits=1000,
        pools=sites,
        options=(option(214, "extractorT1", placed=True),),
    )


def test_the_reserve_keeps_expansion_off_the_armys_credits_once_there_is_an_economy() -> None:
    """Expansion is investment and may not take what replaces a loss --
    **after** the economy that funds the army exists.

    The line is the map's own answer now, re-measured three times. Four was
    where an economy exists (across 46 duels, final income >= 50/s won 36 of
    36); seven was where duel_lake's expansion race is won -- every winning
    solo trace reached 6-7 extractors by s1500 while every loss stalled at
    4-5 (`runs/traces/vh-solo24`, log 2026-08-03); and carried to four other
    maps that literal seven lost or stalemated every match, because a floor
    the map cannot fund is never crossed (`runs/sweeps/xmap-*`,
    log 2026-08-05). Nine pools with none unreachable derive exactly the
    seven the traces demanded ([[policy-holding-ground]]).
    """
    held_back = _race_world(extractors=7, pools_visible=9)
    _, spent = run_campaign(held_back, times=1, reserve=0)
    assert verb(spent, "build")

    _, held = run_campaign(held_back, times=1, reserve=400)
    assert verb(held, "build") == []


def test_below_the_derived_floor_expansion_still_takes_the_armys_credits() -> None:
    """The race side of the same boundary: at six of duel_lake's nine pools
    the race is not yet won, so the claim is protected and the reserve does
    not hold it back."""
    racing = _race_world(extractors=6, pools_visible=9)
    _, peer = run_campaign(racing, times=1, reserve=400)
    assert verb(peer, "build")


def test_a_small_map_lowers_the_floor_to_what_it_can_fund() -> None:
    """The cross-map fix. Champion flame-close carried duel_lake's literal
    seven to four maps whose extractor peaks were 2-4 and went 0W/5L/3S:
    a floor the map cannot fund is never crossed, so expansion claimed
    protected forever and the army channels starved (`runs/sweeps/xmap-*`,
    log 2026-08-05). Three visible pools derive a floor of one, so the
    second extractor already yields to the reserve."""
    modest = _race_world(extractors=1, pools_visible=3)
    _, peer = run_campaign(modest, times=1, reserve=400)
    assert verb(peer, "build") == []


def test_the_derived_floor_is_the_reachable_pools_less_the_rivals_share() -> None:
    """duel_lake recovered exactly: nine pools, none unreachable, floor seven
    -- the number the solo traces measured (`runs/traces/vh-solo24`). Pools
    the builder cannot walk to at all never funded anyone's race."""
    assert economy_floor(9, 0) == 7
    assert economy_floor(8, 0) == 6
    assert economy_floor(10, 6) == 2


def test_the_derived_floor_never_drops_below_the_first_extractor() -> None:
    """However few pools a map offers, matches with no economy at all lost
    outright -- final income at or below 38/s failed 6 of 7 across 46 duels
    ([[policy-holding-ground]])."""
    assert economy_floor(2, 0) == 1
    assert economy_floor(0, 0) == 1


def test_the_economy_outranks_the_army_until_it_can_pay_for_one() -> None:
    """The asymmetry that starved every hard match.

    ``replace_losses`` claims protected and unbounded; expansion claimed
    unprotected. So the reserve kept expansion off the army's credits and
    nothing kept the army off the economy's -- and with several factories
    feeding a wave that died continuously, production took the whole income.
    Measured at Very Hard: **2,800 credits reached the economy out of roughly
    65,000 spent**, 129 units produced, two alive, income ending at 26/s
    ([[policy-holding-ground]]).

    Below the floor the same world that is held back above still builds.
    """
    world = sample(
        CENTRE,
        BUILDER,
        credits=1000,
        pools=(pool(x=300.0),),
        options=(option(214, "extractorT1", placed=True),),
    )
    _, peer = run_campaign(world, times=1, reserve=400)
    assert verb(peer, "build")


def test_expansion_can_be_switched_off_entirely() -> None:
    """The control arm of the A/B that measures whether expanding helps."""
    world = sample(
        CENTRE,
        BUILDER,
        credits=4000,
        pools=(pool(x=300.0),),
        options=(option(214, "extractorT1", placed=True),),
    )
    report, peer = run_campaign(world, times=1, expand=False)
    assert verb(peer, "build") == []
    assert report["expand_reason"] == "expansion disabled"


def test_losing_the_army_no_longer_ends_the_match() -> None:
    """Production runs every tick, so a wiped wave is a setback to rebuild from.

    The old fight loop stopped on an empty army, which with continuous
    production is a run abandoned rather than a run lost.
    """
    world = sample(
        CENTRE,
        FACTORY,
        ENEMY,
        credits=4000,
        options=(option(300, "c_tank"),),
    )
    report, peer = run_campaign(world, times=3, reinforce=("c_tank",))
    assert report["outcome"] == "sample_limit"
    assert report["samples_seen"] == 3
    assert verb(peer, "produce")


def test_an_empty_field_no_longer_ends_the_match() -> None:
    """Nothing hostile in sight is the opening position of every match.

    The map is fogged and the opponents are across it, so stopping there would
    have ended the run on its first observation.
    """
    report, _ = run_campaign(sample(CENTRE, *WAVE), times=4)
    assert report["outcome"] == "sample_limit"
    assert report["samples_seen"] == 4


def test_the_engines_verdict_ends_the_match() -> None:
    world = sample(CENTRE, *WAVE, ENEMY, defeated=True)
    report, _ = run_campaign(world, times=5)
    assert report["grade"] == "defeated"
    assert report["outcome"] == "defeated"
    assert report["samples_seen"] == 1


def test_a_wipe_is_reported_in_preference_to_a_defeat() -> None:
    world = sample(CENTRE, *WAVE, defeated=True, wiped=True)
    report, _ = run_campaign(world, times=5)
    assert report["grade"] == "wiped"


def test_being_the_last_player_standing_is_a_win() -> None:
    report, _ = run_campaign(sample(CENTRE, *WAVE, players_left=1), times=5)
    assert report["grade"] == "won"
    assert report["outcome"] == "won"


def test_the_probe_stop_condition_ends_on_a_finished_plan() -> None:
    """Only a probe asks for this; a match treats a finished opening as the start."""
    world = sample(CENTRE, BUILDER, FACTORY, credits=4000)
    report, _ = run_campaign(world, times=5, plan=("landFactory",), stop_when_plan_done=True)
    assert report["build_outcome"] == "done"
    assert report["samples_seen"] == 1


def test_the_engine_scoreboard_is_carried_into_the_report() -> None:
    """Our army value against the strongest rival's, which is the comparison
    that says whether the match is being lost. The visible-enemy count cannot:
    it measures our own scouting as much as their army.
    """
    # The rival is listed first, so finding our own row means walking past one
    # that is not ours -- which is the ordinary shape of a five-player lobby.
    world = sample(CENTRE, *WAVE, ENEMY, players=(THEM, US))
    report, _ = run_campaign(world, times=2)
    assert report["army_value_start"] == 500
    assert report["army_value_end"] == 500
    assert report["income_end"] == 18
    # Worth counts what is standing as well as what moves, because a turret is
    # booked as a building and is the best value the bot can buy.
    assert report["worth_end"] == 500 + 3000
    assert report["rival_worth_end"] == 4200 + 1500


def test_a_stream_without_a_scoreboard_reports_no_valuation() -> None:
    """Zero rather than a guess, and distinguishable from a real zero by the
    absence of any player record at all.
    """
    report, _ = run_campaign(sample(CENTRE, *WAVE), times=1)
    assert report["army_value_end"] == 0
    assert report["worth_end"] == 0
    assert report["rival_worth_end"] == 0


def test_eliminations_are_counted_across_the_run() -> None:
    peer = ScriptedPeer(
        lines(
            sample(CENTRE, *WAVE, ENEMY, players_left=6),
            sample(CENTRE, *WAVE, ENEMY, players_left=4),
        )
    )
    report = play(AgentChannel(peer), (), CATALOGUE, PLACEMENTS, PROFILES, 2)
    assert report["players_start"] == 6
    assert report["players_end"] == 4
    assert report["eliminated"] == 2


def test_a_rival_that_is_hurt_and_rebuilds_still_reports_the_dip() -> None:
    """The two endpoint figures cannot answer "are we killing them".

    An opponent that lost half its army and rebuilt reads identically at the
    last observation to one that was never touched, so the run that matters --
    the one where an attack actually landed -- is indistinguishable from the
    one where the army walked out and died. The drawdown is measured against a
    running peak for exactly that reason ([[policy-verdict]]).
    """
    peer = ScriptedPeer(
        lines(
            *(
                sample(CENTRE, *WAVE, ENEMY, players=(player(1, army_value=worth), US))
                for worth in (1000, 3000, 1200, 2600)
            )
        )
    )
    report = play(AgentChannel(peer), (), CATALOGUE, PLACEMENTS, PROFILES, 4)
    assert report["rival_worth_start"] == 1000
    assert report["rival_worth_end"] == 2600
    assert report["rival_worth_peak"] == 3000
    # 3000 down to 1200, not 3000 down to 2600: the deepest fall from the peak,
    # not the one the run happened to end on.
    assert report["rival_worth_drawdown"] == 1800


def test_a_rival_that_only_ever_grows_reports_no_dip() -> None:
    """Zero drawdown is the finding, not a missing measurement.

    It says nothing the bot did ever cost that opponent anything, however many
    attack orders were sent.
    """
    peer = ScriptedPeer(
        lines(
            *(
                sample(CENTRE, *WAVE, ENEMY, players=(player(1, army_value=worth), US))
                for worth in (1000, 2000, 4000)
            )
        )
    )
    report = play(AgentChannel(peer), (), CATALOGUE, PLACEMENTS, PROFILES, 3)
    assert report["rival_worth_peak"] == 4000
    assert report["rival_worth_drawdown"] == 0


def test_the_army_mix_is_reported_so_a_denied_composition_is_visible() -> None:
    """Asking for a mix is not getting one, and the report has to show which.

    A type the engine never offers leaves the army at whatever else was
    makeable, and every other figure in the report reads the same either way
    ([[policy-production]]).
    """
    # A second armed type, added here rather than to the shared fixture: every
    # other test in this file is written against a single-type army and a wider
    # catalogue would quietly change what they exercise.
    catalogue = {**CATALOGUE, "c_artillery": unit_stats("c_artillery", price=900)}
    placements = {
        name: TypePlacement(index=i, type_name=name, needs_pool=name == "extractorT1")
        for i, name in enumerate(catalogue)
    }
    world = sample(CENTRE, *WAVE, entity(7, "c_artillery"), ENEMY)
    peer = ScriptedPeer(lines(world))
    report = play(AgentChannel(peer), (), catalogue, placements, profiles_for(catalogue), 1)
    assert report["composition_end"] == (("c_tank", 3), ("c_artillery", 1))


def test_deaths_are_attributed_to_their_killers_in_the_report() -> None:
    """The death ledger end to end: a damaged tank and a damaged extractor
    vanish and the report names both killers, split mobile against
    standing; the builder that vanished untouched appears in neither table,
    because a unit nothing hit left the roster some other way."""
    before = sample(
        CENTRE,
        entity(1, "c_tank", x=500.0, damaged_by="c_artillery"),
        entity(2, "c_tank", x=520.0, damaged_by="c_artillery"),
        entity(3, "c_tank", x=540.0, damaged_by="c_helicopter"),
        entity(400, "extractorT1", x=300.0, damaged_by="c_bomber"),
        entity(214, "builder", x=50.0),
        credits=1000,
    )
    after = sample(CENTRE, credits=1000)
    peer = ScriptedPeer(lines(before, after))
    report = play(AgentChannel(peer), (), CATALOGUE, PLACEMENTS, PROFILES, 2)
    assert report["units_lost_to"] == (("c_artillery", 2), ("c_helicopter", 1))
    assert report["buildings_lost_to"] == (("c_bomber", 1),)
    printed = format_report(report)
    assert "units lost to  c_artillery x2, c_helicopter x1" in printed
    assert "works lost to  c_bomber x1" in printed


def test_the_death_ledger_answers_the_bloodied_gates_read() -> None:
    """deaths_to counts only the named killers' kills -- the bloodied
    gate's read: the naval clause asks the ledger how much blood the
    fleet types it has seen actually hold (log 2026-08-08)."""
    before = sample(
        CENTRE,
        entity(1, "c_tank", x=500.0, damaged_by="battleShip"),
        entity(2, "c_tank", x=520.0, damaged_by="battleShip"),
        entity(3, "c_tank", x=540.0, damaged_by="c_artillery"),
        credits=1000,
    )
    after = sample(CENTRE, credits=1000)
    scores = Scorekeeper(CATALOGUE, PROFILES)
    for observed in (before, after):
        scores.observe(observed, (), (), 0)
    assert scores.deaths_to({"battleShip"}) == 2
    assert scores.deaths_to({"battleShip", "c_artillery"}) == 3
    assert scores.deaths_to({"gunBoat"}) == 0


def test_the_shipyards_order_lands_after_the_expanders() -> None:
    """The engine holds one order per unit and whoever sends last holds the
    builder -- learned on navy96b, forgotten in v4, and photographed on
    navy96d: all 48 walks exhausted with the builder re-tasked every tick
    while navy96c's factories stood 24/26 (log 2026-08-10). The shipyard's
    build must be the tick's LAST builder order or the factory never
    stands."""
    catalogue = {**CATALOGUE, "seaFactory": unit_stats("seaFactory", speed=0.0, price=1000)}
    placements = {
        name: TypePlacement(index=i, type_name=name, needs_pool=name == "extractorT1")
        for i, name in enumerate(catalogue)
    }
    profiles = profiles_for(catalogue)
    world = sample(
        CENTRE,
        entity(214, "builder", x=0.0, y=0.0),
        credits=8000,
        pools=(pool(x=60.0, y=0.0),),
        options=(option(214, "extractorT1"), option(214, "seaFactory")),
    )
    peer = ScriptedPeer(lines(world))
    play(AgentChannel(peer), (), catalogue, placements, profiles, 1, expand=True, navy=1)
    built = verb(peer, "build")
    assert built, "the tick produced no build orders at all"
    assert '"type":"seaFactory"' in built[-1], (
        f"the shipyard's order must be sent last, got {built}"
    )
    assert any('"type":"extractorT1"' in line for line in built), (
        "the expander never ordered, so the test proves nothing about ordering"
    )


def test_the_mechanism_census_remembers_what_stood_and_died() -> None:
    """Five navy panels ran to verdict without anything proving a paid-for
    factory ever EXISTED: the end-state tables cannot testify for a
    structure that stood and then died (log 2026-08-10). The peak census
    can, and a pilot match is read against it before a panel runs."""
    catalogue = {**CATALOGUE, "seaFactory": unit_stats("seaFactory", speed=0.0, price=1000)}
    profiles = profiles_for(catalogue)
    with_factory = sample(
        CENTRE,
        entity(50, "seaFactory", x=300.0, y=0.0),
        entity(1, "c_tank", x=10.0),
        entity(2, "c_tank", x=20.0),
        credits=1000,
    )
    without = sample(CENTRE, entity(1, "c_tank", x=10.0), credits=1000)
    scores = Scorekeeper(catalogue, profiles)
    for observed in (with_factory, without):
        scores.observe(observed, (), (), 0)
    report = scores.report(
        completed=0,
        planned=0,
        build_orders=0,
        build_outcome="",
        build_reason="",
        produced=0,
        expanded=0,
        expanded_factories=0,
        expand_reason="",
        attack_orders=0,
        rallied=0,
        intercepts=0,
        sightings=0,
        raids=0,
        marches=0,
        killed=0,
        refused_claims=0,
        outlays=(),
        reaches=(),
        outcome="",
    )
    assert report["owned_peak"] == (
        ("c_tank", 2),
        ("commandCenter", 1),
        ("seaFactory", 1),
    )
    assert report["standing_end"] == (("commandCenter", 1),)
    rendered = format_report(report)
    assert "owned peak     c_tank x2, commandCenter x1, seaFactory x1" in rendered
