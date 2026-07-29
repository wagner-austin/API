"""The one tick, driven against a scripted world.

What to build, what to attack and what to claim are tested as pure functions
elsewhere. What is tested here is the loop around them: that every layer runs on
every observation, that the credits are arbitrated rather than raced, and that
the match ends when the engine says so rather than when a proxy for it does.
"""

from __future__ import annotations

from pathlib import Path

from rw_bot.control.channel import AgentChannel
from rw_bot.mechanics.catalogue import UnitStats, Weapon
from rw_bot.mechanics.combat_profile import CombatProfile
from rw_bot.mechanics.placement import TypePlacement
from rw_bot.policy.campaign import EXPAND_RETRY_SAMPLES, play
from rw_bot.policy.match_report import MatchReport, format_report
from rw_bot.wire.state import Sample
from tests.wire_fixtures import (
    enemy,
    entity,
    lines,
    option,
    player,
    pool,
    profile,
    profiles_for,
    sample,
)


def _unit(
    type_name: str,
    *,
    speed: float = 1.0,
    armed: bool = True,
    price: int = 350,
    upgrade_prices: tuple[int, ...] = (),
) -> UnitStats:
    return UnitStats(
        type_name=type_name,
        display_name=type_name,
        description="",
        price=price,
        hp=100,
        speed=speed,
        turn_speed=0.0,
        mass=1,
        upgrade_prices=upgrade_prices,
        weapon=(
            Weapon(
                shoot_delay=50.0,
                attack_range=110.0,
                direct_damage=17.0,
                direct_damage_volley=17.0,
                area_damage=0.0,
                area_damage_volley=0.0,
            )
            if armed
            else None
        ),
    )


_CATALOGUE = {
    "c_tank": _unit("c_tank"),
    "builder": _unit("builder", speed=0.6, armed=False, price=200),
    "commandCenter": _unit("commandCenter", speed=0.0, armed=False, price=0),
    # Priced as the engine's own dump prices them, so the arithmetic below is
    # the arithmetic a real match does.
    "extractorT1": _unit("extractorT1", speed=0.0, armed=False, price=700),
    "landFactory": _unit("landFactory", speed=0.0, armed=False, price=1000),
    "editorOrBuilder": _unit("editorOrBuilder", speed=0.0, armed=False, price=0),
}

_PROFILES = profiles_for(_CATALOGUE)

_PLACEMENTS: dict[str, TypePlacement] = {
    name: TypePlacement(index=i, type_name=name, needs_pool=name == "extractorT1")
    for i, name in enumerate(_CATALOGUE)
}

_BUILDER = entity(214, "builder")
_CENTRE = entity(213, "commandCenter")
_FACTORY = entity(300, "landFactory")
_WAVE = (
    entity(1, "c_tank"),
    entity(2, "c_tank"),
    entity(3, "c_tank"),
)
_ENEMY = enemy(9, "c_tank", x=100.0)

_US = player(0, index=0, local=True, hostile=False, income=18, army_value=500, building_value=3000)
_THEM = player(1, index=1, income=18, army_value=4200, building_value=1500)


class _ScriptedPeer:
    """Serves prepared lines and records what was sent back.

    Attributes:
        sent: Every line the loop wrote, in order.
    """

    def __init__(self, prepared: list[str]) -> None:
        self._lines = list(prepared)
        self.sent: list[str] = []

    def send_line(self, line: str) -> None:
        """Record one written line.

        Args:
            line: Line content, without a newline.
        """
        self.sent.append(line)

    def read_line(self) -> str:
        """Serve the next prepared line, or end of stream.

        Returns:
            The next line, or an empty string once exhausted.
        """
        if not self._lines:
            return ""
        return self._lines.pop(0)

    def close(self) -> None:
        """Release the connection."""


def _orders(peer: _ScriptedPeer) -> list[str]:
    """Everything the loop sent except the per-sample acknowledgements.

    The ack is protocol rather than policy -- in lockstep it is what releases
    the simulation -- so assertions about what the bot decided filter it out
    ([[policy-determinism]]).
    """
    return [line for line in peer.sent if '"kind":"ack"' not in line]


def _verb(peer: _ScriptedPeer, kind: str) -> list[str]:
    return [line for line in _orders(peer) if f'"kind":"{kind}"' in line]


def _run(
    world: Sample,
    *,
    times: int = 3,
    plan: tuple[str, ...] = (),
    reinforce: tuple[str, ...] = (),
    reserve: int = 0,
    expand: bool = True,
    stop_when_plan_done: bool = False,
    trace: Path | None = None,
) -> tuple[MatchReport, _ScriptedPeer]:
    """Play one scripted world for a fixed number of observations."""
    peer = _ScriptedPeer(lines(*(world for _ in range(times))))
    report = play(
        AgentChannel(peer),
        plan,
        _CATALOGUE,
        _PLACEMENTS,
        _PROFILES,
        times,
        reinforce=reinforce,
        reserve=reserve,
        expand=expand,
        stop_when_plan_done=stop_when_plan_done,
        trace=trace,
    )
    return report, peer


def test_the_army_is_sent_at_the_enemy() -> None:
    report, peer = _run(sample(*_WAVE, _ENEMY))
    assert _verb(peer, "attack") == [
        '{"kind":"attack","unit_id":1,"target_id":9}',
        '{"kind":"attack","unit_id":2,"target_id":9}',
        '{"kind":"attack","unit_id":3,"target_id":9}',
    ]
    assert report["attack_orders"] == 3


def test_an_attack_is_not_reissued_while_it_stands() -> None:
    """The engine runs a waypoint until it is replaced, so a repeat resets it."""
    _, peer = _run(sample(*_WAVE, _ENEMY), times=5)
    assert len(_verb(peer, "attack")) == 3


def test_the_plan_and_the_fight_run_on_the_same_observation() -> None:
    """The seam this refactor removed.

    The old loop built to completion and only then fought, so the opening was
    played defenceless and the plan stopped the moment fighting began. Both act
    on one tick now ([[policy-loop]]).
    """
    world = sample(
        _CENTRE,
        _BUILDER,
        *_WAVE,
        _ENEMY,
        credits=4000,
        options=(option(214, "landFactory", placed=True),),
    )
    _, peer = _run(world, plan=("landFactory",))
    assert _verb(peer, "build")
    assert _verb(peer, "attack")


def test_the_same_credit_is_not_committed_twice_in_one_tick() -> None:
    """The defect the arbiter exists for.

    One factory can start a 350 tank and the builder can place a 700 extractor,
    but not on 800 credits. Production claims first because it is protected;
    expansion is refused and says so.
    """
    world = sample(
        _CENTRE,
        _BUILDER,
        _FACTORY,
        credits=800,
        pools=(pool(x=300.0),),
        options=(option(300, "c_tank"), option(214, "extractorT1", placed=True)),
    )
    report, peer = _run(world, times=1, reinforce=("c_tank",))
    assert _verb(peer, "produce") == ['{"kind":"produce","unit_id":300,"type":"c_tank"}']
    assert _verb(peer, "build") == []
    assert report["refused_claims"] == 1
    assert "wanted 700" in report["expand_reason"]


def test_both_are_afforded_when_the_credits_are_there() -> None:
    """The complement, so the refusal above is arbitration and not a dead path."""
    world = sample(
        _CENTRE,
        _BUILDER,
        _FACTORY,
        credits=4000,
        pools=(pool(x=300.0),),
        options=(option(300, "c_tank"), option(214, "extractorT1", placed=True)),
    )
    report, peer = _run(world, times=1, reinforce=("c_tank",))
    assert _verb(peer, "produce")
    assert _verb(peer, "build")
    assert report["refused_claims"] == 0
    # Income, not throughput: the factory is idle, so more capacity buys nothing.
    assert report["expanded_factories"] == 0


def test_the_reserve_keeps_expansion_off_the_armys_credits_once_there_is_an_economy() -> None:
    """Expansion is investment and may not take what replaces a loss --
    **after** the economy that funds the army exists.

    Four extractors is where that line sits, and it is measured rather than
    chosen: across 46 duels a final income at or above 50 credits a second won
    36 of 36, and at or below 38 it failed 6 of 7. Base income is 18 and an
    extractor pays 8, so 50/s is four of them ([[policy-holding-ground]]).
    """
    held_back = sample(
        _CENTRE,
        _BUILDER,
        *(entity(400 + n, "extractorT1", x=900.0 + 60 * n, y=0.0) for n in range(4)),
        credits=1000,
        pools=(pool(x=300.0),),
        options=(option(214, "extractorT1", placed=True),),
    )
    _, spent = _run(held_back, times=1, reserve=0)
    assert _verb(spent, "build")

    _, held = _run(held_back, times=1, reserve=400)
    assert _verb(held, "build") == []


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
        _CENTRE,
        _BUILDER,
        credits=1000,
        pools=(pool(x=300.0),),
        options=(option(214, "extractorT1", placed=True),),
    )
    _, peer = _run(world, times=1, reserve=400)
    assert _verb(peer, "build")


def test_expansion_can_be_switched_off_entirely() -> None:
    """The control arm of the A/B that measures whether expanding helps."""
    world = sample(
        _CENTRE,
        _BUILDER,
        credits=4000,
        pools=(pool(x=300.0),),
        options=(option(214, "extractorT1", placed=True),),
    )
    report, peer = _run(world, times=1, expand=False)
    assert _verb(peer, "build") == []
    assert report["expand_reason"] == "expansion disabled"


def test_losing_the_army_no_longer_ends_the_match() -> None:
    """Production runs every tick, so a wiped wave is a setback to rebuild from.

    The old fight loop stopped on an empty army, which with continuous
    production is a run abandoned rather than a run lost.
    """
    world = sample(
        _CENTRE,
        _FACTORY,
        _ENEMY,
        credits=4000,
        options=(option(300, "c_tank"),),
    )
    report, peer = _run(world, times=3, reinforce=("c_tank",))
    assert report["outcome"] == "sample_limit"
    assert report["samples_seen"] == 3
    assert _verb(peer, "produce")


def test_an_empty_field_no_longer_ends_the_match() -> None:
    """Nothing hostile in sight is the opening position of every match.

    The map is fogged and the opponents are across it, so stopping there would
    have ended the run on its first observation.
    """
    report, _ = _run(sample(_CENTRE, *_WAVE), times=4)
    assert report["outcome"] == "sample_limit"
    assert report["samples_seen"] == 4


def test_the_engines_verdict_ends_the_match() -> None:
    world = sample(_CENTRE, *_WAVE, _ENEMY, defeated=True)
    report, _ = _run(world, times=5)
    assert report["grade"] == "defeated"
    assert report["outcome"] == "defeated"
    assert report["samples_seen"] == 1


def test_a_wipe_is_reported_in_preference_to_a_defeat() -> None:
    world = sample(_CENTRE, *_WAVE, defeated=True, wiped=True)
    report, _ = _run(world, times=5)
    assert report["grade"] == "wiped"


def test_being_the_last_player_standing_is_a_win() -> None:
    report, _ = _run(sample(_CENTRE, *_WAVE, players_left=1), times=5)
    assert report["grade"] == "won"
    assert report["outcome"] == "won"


def test_the_probe_stop_condition_ends_on_a_finished_plan() -> None:
    """Only a probe asks for this; a match treats a finished opening as the start."""
    world = sample(_CENTRE, _BUILDER, _FACTORY, credits=4000)
    report, _ = _run(world, times=5, plan=("landFactory",), stop_when_plan_done=True)
    assert report["build_outcome"] == "done"
    assert report["samples_seen"] == 1


def test_the_engine_scoreboard_is_carried_into_the_report() -> None:
    """Our army value against the strongest rival's, which is the comparison
    that says whether the match is being lost. The visible-enemy count cannot:
    it measures our own scouting as much as their army.
    """
    # The rival is listed first, so finding our own row means walking past one
    # that is not ours -- which is the ordinary shape of a five-player lobby.
    world = sample(_CENTRE, *_WAVE, _ENEMY, players=(_THEM, _US))
    report, _ = _run(world, times=2)
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
    report, _ = _run(sample(_CENTRE, *_WAVE), times=1)
    assert report["army_value_end"] == 0
    assert report["worth_end"] == 0
    assert report["rival_worth_end"] == 0


def test_eliminations_are_counted_across_the_run() -> None:
    peer = _ScriptedPeer(
        lines(
            sample(_CENTRE, *_WAVE, _ENEMY, players_left=6),
            sample(_CENTRE, *_WAVE, _ENEMY, players_left=4),
        )
    )
    report = play(AgentChannel(peer), (), _CATALOGUE, _PLACEMENTS, _PROFILES, 2)
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
    peer = _ScriptedPeer(
        lines(
            *(
                sample(_CENTRE, *_WAVE, _ENEMY, players=(player(1, army_value=worth), _US))
                for worth in (1000, 3000, 1200, 2600)
            )
        )
    )
    report = play(AgentChannel(peer), (), _CATALOGUE, _PLACEMENTS, _PROFILES, 4)
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
    peer = _ScriptedPeer(
        lines(
            *(
                sample(_CENTRE, *_WAVE, _ENEMY, players=(player(1, army_value=worth), _US))
                for worth in (1000, 2000, 4000)
            )
        )
    )
    report = play(AgentChannel(peer), (), _CATALOGUE, _PLACEMENTS, _PROFILES, 3)
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
    catalogue = {**_CATALOGUE, "c_artillery": _unit("c_artillery", price=900)}
    placements = {
        name: TypePlacement(index=i, type_name=name, needs_pool=name == "extractorT1")
        for i, name in enumerate(catalogue)
    }
    world = sample(_CENTRE, *_WAVE, entity(7, "c_artillery"), _ENEMY)
    peer = _ScriptedPeer(lines(world))
    report = play(AgentChannel(peer), (), catalogue, placements, profiles_for(catalogue), 1)
    assert report["composition_end"] == (("c_tank", 3), ("c_artillery", 1))


def _defence_world() -> tuple[
    dict[str, UnitStats], dict[str, TypePlacement], dict[str, CombatProfile]
]:
    """A catalogue, placements and profiles that include a buildable turret."""
    catalogue = {**_CATALOGUE, "c_turret_t1": _unit("c_turret_t1", speed=0.0, price=500)}
    placements = {
        name: TypePlacement(index=i, type_name=name, needs_pool=name == "extractorT1")
        for i, name in enumerate(catalogue)
    }
    profiles = {**profiles_for(catalogue), "c_turret_t1": profile("c_turret_t1", 165.0)}
    return catalogue, placements, profiles


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
    catalogue, placements, profiles = _defence_world()
    world = sample(
        _CENTRE,
        entity(214, "builder", x=0.0, y=0.0),
        credits=4000,
        pools=(pool(x=60.0, y=0.0),),
        options=(option(214, "c_turret_t1"), option(214, "extractorT1")),
    )
    peer = _ScriptedPeer(lines(world))
    play(AgentChannel(peer), (), catalogue, placements, profiles, 1, expand=True)
    built = [line for line in _orders(peer) if '"kind":"build"' in line]
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
    catalogue, placements, profiles = _defence_world()
    world = sample(
        _CENTRE,
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
    peer = _ScriptedPeer(lines(world))
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
    built = [line for line in _orders(peer) if '"kind":"build"' in line]
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
    catalogue, placements, profiles = _defence_world()
    world = sample(
        _CENTRE,
        entity(214, "builder", x=0.0, y=0.0),
        credits=40_000,
        pools=(pool(x=60.0, y=0.0),),
        options=(option(214, "landFactory"), option(214, "extractorT1")),
    )
    peer = _ScriptedPeer(lines(world))
    play(AgentChannel(peer), (), catalogue, placements, profiles, 1, expand=True)
    built = [line for line in _orders(peer) if '"kind":"build"' in line]
    assert built == ['{"kind":"build","unit_id":214,"x":60.0,"y":0.0,"type":"extractorT1"}']


def test_defence_takes_the_surplus_when_no_pool_can_be_claimed() -> None:
    """Income compounds and defence does not, so income keeps its place. What
    defence takes is the surplus that was otherwise buying a twenty-second Land
    Factory -- a trade between two things that both fail to compound
    ([[policy-production]]).
    """
    catalogue, placements, profiles = _defence_world()
    world = sample(
        _CENTRE,
        entity(214, "builder", x=0.0, y=0.0),
        credits=4000,
        # No pools at all, so income has nothing left to claim and the surplus
        # is what defence is spending.
        options=(option(214, "c_turret_t1"),),
    )
    peer = _ScriptedPeer(lines(world))
    play(AgentChannel(peer), (), catalogue, placements, profiles, 1, expand=True)
    # The Command Center is nearest the anchor and is itself uncovered, so it is
    # what gets covered. Aiming this at the extractors instead was measured and
    # lost -- wins 4 -> 0 over the same twelve seeds ([[policy-holding-ground]]).
    assert [line for line in _orders(peer) if '"kind":"build"' in line] == [
        '{"kind":"build","unit_id":214,"x":60.0,"y":0.0,"type":"c_turret_t1"}'
    ]


def test_the_engine_clock_is_reported_beside_the_frame_count() -> None:
    """The pair answers what neither can alone: whether the simulation advances
    per frame or per wall clock. The engine caps itself at 300 frames a second
    and matches run at about 297, so if the clock outruns the wall the cap is a
    real throughput ceiling and if it tracks the wall then removing it would buy
    nothing ([[harness-parallel-matches]]).
    """
    peer = _ScriptedPeer(
        lines(
            sample(_CENTRE, *_WAVE, frame=100, clock_ms=1_000),
            sample(_CENTRE, *_WAVE, frame=400, clock_ms=6_000),
        )
    )
    report = play(AgentChannel(peer), (), _CATALOGUE, _PLACEMENTS, _PROFILES, 2)
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
        _CENTRE,
        *_WAVE,
        enemy(9, "c_tank", x=100.0),
        enemy(10, "c_tank", x=120.0),
        enemy(11, "extractorT2", x=140.0),
    )
    report, _ = _run(world, times=1)
    assert report["enemy_types_end"] == (("c_tank", 2), ("extractorT2", 1))


def test_an_unseen_enemy_is_reported_as_none_rather_than_blank() -> None:
    """Nothing visible is a real observation, not a missing measurement."""
    report, _ = _run(sample(_CENTRE, *_WAVE), times=1)
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
        **_CATALOGUE,
        "extractorT1": _unit("extractorT1", speed=0.0, price=700, upgrade_prices=(1400,)),
        "extractorT2": _unit("extractorT2", speed=0.0, price=2100),
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
        _CENTRE,
        entity(400, "extractorT1"),
        credits=4000,
        options=(option(400, "extractorT2", placed=False, makes_something=False),),
    )
    peer = _ScriptedPeer(lines(world))
    play(AgentChannel(peer), (), catalogue, placements, _PROFILES, 1)
    assert _verb(peer, "produce") == ['{"kind":"produce","unit_id":400,"type":"extractorT2"}']


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
        _CENTRE,
        _BUILDER,
        entity(400, "extractorT1", x=900.0, y=0.0),
        credits=1500,
        pools=(pool(x=300.0),),
        options=(
            option(214, "extractorT1", placed=True),
            option(400, "extractorT2", placed=False, makes_something=False),
        ),
    )
    peer = _ScriptedPeer(lines(world))
    play(AgentChannel(peer), (), catalogue, placements, _PROFILES, 1)
    assert _verb(peer, "produce") == ['{"kind":"produce","unit_id":400,"type":"extractorT2"}']
    # Expansion is not disabled, only outranked: 1,500 covered the conversion
    # and what was left could not also cover a 700 extractor.
    assert _verb(peer, "build") == []


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
        _CENTRE,
        entity(400, "extractorT1"),
        credits=40000,
        options=(option(400, "extractorT2", placed=False, makes_something=False),),
    )
    peer = _ScriptedPeer(lines(world, world, world, world))
    play(AgentChannel(peer), (), catalogue, placements, _PROFILES, 4)
    assert _verb(peer, "produce") == ['{"kind":"produce","unit_id":400,"type":"extractorT2"}']


def test_upgrades_stop_at_the_first_refusal_rather_than_overdrawing() -> None:
    """Every upgrade costs the same, so a refusal means the budget is out."""
    catalogue, placements = _upgrade_world()
    world = sample(
        _CENTRE,
        entity(400, "extractorT1"),
        entity(401, "extractorT1"),
        credits=1400,
        options=(
            option(400, "extractorT2", placed=False, makes_something=False),
            option(401, "extractorT2", placed=False, makes_something=False),
        ),
    )
    peer = _ScriptedPeer(lines(world))
    play(AgentChannel(peer), (), catalogue, placements, _PROFILES, 1)
    assert len(_verb(peer, "produce")) == 1


def test_an_upgrade_the_catalogue_cannot_price_is_never_ordered() -> None:
    """Unpriced means unbudgetable, and spending blind is what the budget
    prevents ([[policy-budget]]).
    """
    world = sample(
        _CENTRE,
        entity(400, "extractorT1"),
        credits=4000,
        options=(option(400, "extractorT2", placed=False, makes_something=False),),
    )
    # _CATALOGUE does not price extractorT2.
    _, peer = _run(world, times=1)
    assert _verb(peer, "produce") == []


def test_the_reserve_gathers_at_the_base() -> None:
    """Units waiting near the base are the only defensive posture the bot has."""
    far = entity(1, "c_tank", x=900.0)
    _, peer = _run(sample(_CENTRE, far, _ENEMY), times=2)
    assert '{"kind":"move","unit_id":1,"x":0.0,"y":0.0}' in _verb(peer, "move")


def test_a_unit_already_at_the_base_is_not_told_to_go_there() -> None:
    report, _ = _run(sample(_CENTRE, entity(1, "c_tank", x=10.0), _ENEMY), times=2)
    assert report["rallied"] == 0


def test_a_lost_builder_is_replaced_before_the_economy_dies() -> None:
    """A lost builder ends the economy permanently, so one is asked for."""
    world = sample(
        _CENTRE,
        credits=4000,
        options=(option(213, "builder"),),
    )
    _, peer = _run(world, times=1)
    assert _verb(peer, "produce") == ['{"kind":"produce","unit_id":213,"type":"builder"}']


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
        _FACTORY,
        credits=4000,
        options=(option(300, "c_tank"), option(300, "builder")),
    )
    _, peer = _run(world, times=1, reinforce=("c_tank",))
    assert _verb(peer, "produce") == ['{"kind":"produce","unit_id":300,"type":"builder"}']


def test_a_factory_stays_on_the_army_while_a_builder_is_alive() -> None:
    """The emergency is having none, not having few.

    Otherwise the fix trades one runaway for another: a builder inside the mix
    permanently is a factory spending the match on builders, which is the
    33-worker run in a different disguise ([[policy-production]]).
    """
    world = sample(
        _BUILDER,
        _FACTORY,
        credits=4000,
        options=(option(300, "c_tank"), option(300, "builder")),
    )
    _, peer = _run(world, times=1, reinforce=("c_tank",))
    assert _verb(peer, "produce") == ['{"kind":"produce","unit_id":300,"type":"c_tank"}']


def test_the_builder_goes_last_so_factories_stay_on_tanks() -> None:
    """A producer takes the first type it can make, and only the command centre
    -- which cannot make a tank -- falls through to the builder.
    """
    world = sample(
        _CENTRE,
        _FACTORY,
        credits=4000,
        options=(option(300, "c_tank"), option(213, "builder")),
    )
    _, peer = _run(world, times=1, reinforce=("c_tank",))
    # Roster order, not preference order: what preference decides is *what each
    # producer makes*, and the factory took the tank rather than falling through
    # to the builder the command centre ends up with.
    assert _verb(peer, "produce") == [
        '{"kind":"produce","unit_id":213,"type":"builder"}',
        '{"kind":"produce","unit_id":300,"type":"c_tank"}',
    ]


def test_an_unreachable_enemy_is_counted_apart_from_a_visible_one() -> None:
    """The gap between the two is the diagnosis: an army holding the wrong units."""
    flyer = enemy(9, "helicopter", x=100.0, flying=True)
    profiles = {**_PROFILES, "helicopter": _PROFILES["c_tank"]}
    peer = _ScriptedPeer(lines(sample(_CENTRE, *_WAVE, flyer)))
    report = play(AgentChannel(peer), (), _CATALOGUE, _PLACEMENTS, profiles, 1)
    assert report["targets_end"] == 1
    assert report["engageable_end"] == 0
    assert _verb(peer, "attack") == []


def test_every_observation_is_acknowledged() -> None:
    """In lockstep the agent holds the simulation until the ack arrives."""
    _, peer = _run(sample(_CENTRE, *_WAVE, _ENEMY), times=4)
    assert len([line for line in peer.sent if '"kind":"ack"' in line]) == 4


def test_the_trace_is_written_when_a_path_is_given(tmp_path: Path) -> None:
    target = tmp_path / "trace.txt"
    _run(sample(_CENTRE, *_WAVE, _ENEMY), times=2, trace=target)
    written = target.read_text(encoding="utf-8")
    assert "frame" in written
    assert "army" in written


def test_the_report_renders_as_lines() -> None:
    report, _ = _run(sample(_CENTRE, *_WAVE, _ENEMY, players=(_US, _THEM)), times=2)
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
        _CENTRE,
        _BUILDER,
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
    report, peer = _run(world, times=1, reinforce=("c_tank",))
    assert _verb(peer, "build") == [
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
        _CENTRE,
        _BUILDER,
        entity(300, "landFactory", queued=1),
        credits=4000,
        pools=(pool(x=300.0),),
        options=(
            option(300, "c_tank"),
            option(214, "landFactory", placed=True),
            option(214, "extractorT1", placed=True),
        ),
    )
    _, peer = _run(world, times=1)
    assert _verb(peer, "build") == [
        '{"kind":"build","unit_id":214,"x":300.0,"y":0.0,"type":"extractorT1"}'
    ]


def test_an_expansion_order_is_not_repeated_at_sample_rate() -> None:
    """The builder has been told; re-sending every observation resets the walk."""
    world = sample(
        _CENTRE,
        _BUILDER,
        credits=4000,
        pools=(pool(x=300.0),),
        options=(option(214, "extractorT1", placed=True),),
    )
    _, peer = _run(world, times=6)
    assert len(_verb(peer, "build")) == 1


def test_the_plan_waits_when_the_army_has_taken_the_credits() -> None:
    """The plan waits on price rather than issuing an order it cannot pay for.

    Decided before the budget is opened at all: the plan is the first claimant
    and protected, so a claim it makes cannot be refused. What stops it here is
    its own affordability check against the same balance.
    """
    world = sample(
        _CENTRE,
        _BUILDER,
        credits=100,
        options=(option(214, "landFactory", placed=True),),
    )
    report, peer = _run(world, times=1, plan=("landFactory",))
    assert _verb(peer, "build") == []
    assert report["build_reason"] == "landFactory costs 1000, holding 100"


def test_production_stops_at_the_first_claim_it_cannot_meet() -> None:
    """Preference order is what makes dropping the tail meaningful.

    Two factories, one tank's worth of credits: the first is ordered and the
    second is not, rather than an arbitrary one of the two.
    """
    world = sample(
        _CENTRE,
        entity(300, "landFactory"),
        entity(301, "landFactory"),
        credits=350,
        options=(option(300, "c_tank"), option(301, "c_tank")),
    )
    _, peer = _run(world, times=1, reinforce=("c_tank",))
    assert _verb(peer, "produce") == ['{"kind":"produce","unit_id":300,"type":"c_tank"}']


def test_a_produced_plan_entry_is_queued_rather_than_placed() -> None:
    """A unit rolls out of the building that made it; the planner sites nothing."""
    world = sample(
        _CENTRE,
        _BUILDER,
        _FACTORY,
        credits=4000,
        options=(option(300, "c_tank"),),
    )
    _, peer = _run(world, times=1, plan=("c_tank",))
    assert _verb(peer, "produce") == ['{"kind":"produce","unit_id":300,"type":"c_tank"}']


def test_the_plan_and_the_economy_do_not_both_drive_the_one_builder() -> None:
    """There is one builder, and the engine runs whichever waypoint arrived last.

    A live 400-sample run had the economy re-tasking the builder to its own pool
    between the plan's own extractors: four expansions ordered, and a plan still
    stuck at 3 of 8 ([[policy-loop]]). Whoever holds the builder holds it alone.
    """
    world = sample(
        _CENTRE,
        _BUILDER,
        credits=4000,
        pools=(pool(x=300.0), pool(x=900.0, index=1)),
        options=(option(214, "extractorT1", placed=True),),
    )
    report, peer = _run(world, times=1, plan=("extractorT1",))
    assert len(_verb(peer, "build")) == 1
    assert report["expanded"] == 0
    # The economy stands down because the plan's worker was the *only* free one,
    # not because it is barred whenever the plan holds any worker at all. That
    # distinction is the fix: with six workers the old rule skipped the expander
    # on 572 of 800 samples ([[policy-economy]]).
    assert report["expand_reason"] == "the opening plan is using the only free worker"


def test_a_second_worker_keeps_expanding_while_the_plan_holds_the_first() -> None:
    """The fix, and the figure that forced it.

    The plan takes **one** worker and the expander used to answer that by
    standing down entirely -- income, defence and throughput together, however
    many others were free. Instrumented over 800 samples with six workers alive,
    the expander was skipped on **572 of them**: those spenders were not
    declining, they were never asked ([[policy-economy]]).

    Two builders here, and both should be working: the plan places its extractor
    with one and the economy claims the second pool with the other. Neither may
    order the same unit, which is the defect the old gate existed to prevent
    ([[policy-loop]]).
    """
    world = sample(
        _CENTRE,
        _BUILDER,
        entity(215, "builder", x=0.0, y=0.0),
        credits=4000,
        pools=(pool(x=300.0), pool(x=900.0, index=1)),
        options=(
            option(214, "extractorT1", placed=True),
            option(215, "extractorT1", placed=True),
        ),
    )
    report, peer = _run(world, times=1, plan=("extractorT1",))
    built = _verb(peer, "build")
    assert len(built) == 2
    assert report["expanded"] == 1
    # The two orders go to different workers. One unit ordered twice in a tick is
    # the original bug: the engine runs whichever waypoint arrived last, so
    # neither order is carried out.
    assert len({line.split('"unit_id":')[1].split(",")[0] for line in built}) == 2


def test_a_cheaper_defence_does_not_jump_the_queue_while_income_is_merely_short() -> None:
    """The inversion that stalled the economy at Hard.

    Income needs the extractor's 700; a turret needs 500. So on every
    observation where the economy was refused for credits, defence was offered
    the same balance, could afford it, and took it. Measured over a Hard batch:
    **29 turrets bought against 4 extractors, 43 of 47 extractor claims refused
    for credits**, income stuck at 34/s while the opponent compounded
    ([[policy-holding-ground]]).

    A refusal for any *other* reason is a different matter -- every pool taken,
    every route exposed, no worker able to place one -- and the surplus really is
    spare then. That is what defence is for, and the case below it still passes.
    """
    catalogue, placements, profiles = _defence_world()
    world = sample(
        _CENTRE,
        entity(214, "builder", x=0.0, y=0.0),
        # An uncovered extractor, so defence has somewhere it wants to spend.
        entity(400, "extractorT1", x=900.0, y=0.0),
        # Enough for the 500 turret, not enough for the 700 extractor.
        credits=600,
        pools=(pool(x=300.0),),
        options=(option(214, "c_turret_t1"), option(214, "extractorT1", placed=True)),
    )
    peer = _ScriptedPeer(lines(world))
    play(AgentChannel(peer), (), catalogue, placements, profiles, 1, expand=True)
    assert [line for line in _orders(peer) if '"kind":"build"' in line] == []


def test_two_workers_do_not_both_claim_the_same_pool() -> None:
    """The waste that unblocking the workforce exposed.

    A pool is judged occupied by what *stands* on it, so one a builder is
    walking toward still reads as free. One worker at a time hid that; several
    at once did not. An instrumented run granted **23 extractor orders, lost
    nothing at all, and ended with four extractors** -- the credits were never
    burnt, since a granted claim is intent, but every duplicate cost a worker
    its travel time ([[policy-holding-ground]]).
    """
    near = pool(x=300.0)
    far = pool(x=900.0, index=1)
    world = sample(
        _CENTRE,
        _BUILDER,
        entity(215, "builder", x=0.0, y=0.0),
        credits=10_000,
        pools=(near, far),
        options=(
            option(214, "extractorT1", placed=True),
            option(215, "extractorT1", placed=True),
        ),
    )
    _, peer = _run(world, times=1, plan=("extractorT1",))
    sites = {line.split('"x":')[1].split(",")[0] for line in _verb(peer, "build")}
    assert len(sites) == 2


def test_the_economy_takes_the_builder_once_the_plan_is_finished() -> None:
    """The complement: standing down is for the opening, not for the match."""
    world = sample(
        _CENTRE,
        _BUILDER,
        entity(400, "extractorT1"),
        credits=4000,
        pools=(pool(x=300.0),),
        options=(option(214, "extractorT1", placed=True),),
    )
    report, peer = _run(world, times=1, plan=("extractorT1",))
    assert _verb(peer, "build") == [
        '{"kind":"build","unit_id":214,"x":300.0,"y":0.0,"type":"extractorT1"}'
    ]
    assert report["expanded"] == 1


def test_the_economy_leaves_the_worker_it_just_sent_to_build() -> None:
    """The worker is busy because it has an outstanding job, across ticks.

    Both rules used to refuse only while their *own* structure was going up,
    and a refusal from one fell straight through to the other, which ordered
    something else and re-tasked the worker off it. Availability is judged once
    now, per worker, by the thing that knows what each was sent to do
    ([[policy-loop]]).
    """
    first = sample(
        _CENTRE,
        _BUILDER,
        credits=100_000,
        pools=(pool(x=300.0), pool(x=900.0, index=1)),
        options=(option(214, "extractorT1", placed=True),),
    )
    # Same worker, same place, and now its extractor is going up where it was
    # sent. It must not be handed a second pool.
    building = sample(
        _CENTRE,
        _BUILDER,
        entity(400, "extractorT1", x=300.0, complete=False),
        credits=100_000,
        pools=(pool(x=300.0), pool(x=900.0, index=1)),
        options=(option(214, "extractorT1", placed=True),),
    )
    peer = _ScriptedPeer(lines(first, building, building))
    report = play(AgentChannel(peer), (), _CATALOGUE, _PLACEMENTS, _PROFILES, 3)
    assert len(_verb(peer, "build")) == 1
    assert report["expand_reason"] == "every worker is already building something"


def test_a_second_worker_builds_while_the_first_is_busy() -> None:
    """What the whole refactor buys: two workers, two jobs at once.

    One builder was an assumption baked into every layer -- the plan found "the"
    builder and so did the economy, both meaning the first in the roster, so a
    second would have stood idle for the entire match ([[policy-production]]).
    """
    second = entity(215, "builder", x=100.0)
    first = sample(
        _CENTRE,
        _BUILDER,
        second,
        credits=100_000,
        pools=(pool(x=300.0), pool(x=900.0, index=1)),
        options=(option(214, "extractorT1", placed=True), option(215, "extractorT1", placed=True)),
    )
    building = sample(
        _CENTRE,
        _BUILDER,
        second,
        entity(400, "extractorT1", x=300.0, complete=False),
        credits=100_000,
        pools=(pool(x=300.0), pool(x=900.0, index=1)),
        options=(option(214, "extractorT1", placed=True), option(215, "extractorT1", placed=True)),
    )
    peer = _ScriptedPeer(lines(first, building))
    play(AgentChannel(peer), (), _CATALOGUE, _PLACEMENTS, _PROFILES, 2)
    ordered = _verb(peer, "build")
    assert len(ordered) == 2
    assert '"unit_id":214' in ordered[0]
    assert '"unit_id":215' in ordered[1]


def test_a_walking_builder_is_left_alone_by_the_economy() -> None:
    """The order it is carrying out is the order that would be sent again."""
    world = sample(
        _CENTRE,
        entity(214, "builder", x=50.0),
        credits=100_000,
        pools=(pool(x=300.0),),
        options=(option(214, "extractorT1", placed=True),),
    )
    moving = sample(
        _CENTRE,
        entity(214, "builder", x=90.0),
        credits=100_000,
        pools=(pool(x=300.0),),
        options=(option(214, "extractorT1", placed=True),),
    )
    peer = _ScriptedPeer(lines(world, moving))
    report = play(AgentChannel(peer), (), _CATALOGUE, _PLACEMENTS, _PROFILES, 2)
    assert report["expand_reason"] == "every worker is already building something"


def test_a_disbanded_wave_is_sent_home_again() -> None:
    """The mark is per stint in the reserve, not per match.

    A survivor handed back by a disbanded wave was previously told nothing: not
    cleared to attack, and already marked as rallied from before its first wave.
    It stood where its wave died until enough reinforcements arrived to release
    it again ([[policy-combat]]).
    """
    far = entity(1, "c_tank", x=4000.0)
    # Three tanks release a wave on the first observation; two die, and the
    # survivor is below FIRST_WAVE so it goes back to the reserve.
    first = sample(_CENTRE, *_WAVE, _ENEMY)
    after = sample(_CENTRE, far, _ENEMY)
    peer = _ScriptedPeer(lines(first, after))
    play(AgentChannel(peer), (), _CATALOGUE, _PLACEMENTS, _PROFILES, 2)
    moves = [line for line in peer.sent if '"kind":"move"' in line]
    assert '{"kind":"move","unit_id":1,"x":0.0,"y":0.0}' in moves


def test_a_worker_sitting_on_an_unstarted_job_is_freed_to_retry() -> None:
    """The engine refuses some placements silently, and says so only in its log.

    A worker that has neither moved nor started building is not on its way
    anywhere. After the same window the plan's stall clock uses, the order is
    presumed lost and the worker is free to be given another.
    """
    world = sample(
        _CENTRE,
        _BUILDER,
        credits=100_000,
        pools=(pool(x=300.0),),
        options=(option(214, "extractorT1", placed=True),),
    )
    # Nothing ever goes up and the worker never moves, so after the retry
    # window the order is reissued.
    peer = _ScriptedPeer(lines(*(world for _ in range(EXPAND_RETRY_SAMPLES + 2))))
    play(
        AgentChannel(peer),
        (),
        _CATALOGUE,
        _PLACEMENTS,
        _PROFILES,
        EXPAND_RETRY_SAMPLES + 2,
    )
    assert len(_verb(peer, "build")) == 2


def test_a_worker_that_dies_is_forgotten() -> None:
    """Its bookkeeping must not outlive it, or an id the engine reuses inherits
    a job that was never given to it.
    """
    with_worker = sample(
        _CENTRE,
        _BUILDER,
        credits=100_000,
        pools=(pool(x=300.0),),
        options=(option(214, "extractorT1", placed=True),),
    )
    without = sample(_CENTRE, credits=100_000, pools=(pool(x=300.0),))
    reborn = sample(
        _CENTRE,
        _BUILDER,
        credits=100_000,
        pools=(pool(x=300.0),),
        options=(option(214, "extractorT1", placed=True),),
    )
    peer = _ScriptedPeer(lines(with_worker, without, reborn))
    play(AgentChannel(peer), (), _CATALOGUE, _PLACEMENTS, _PROFILES, 3)
    # Ordered on the first observation, and again once a worker exists afresh.
    assert len(_verb(peer, "build")) == 2


def test_an_expansion_is_not_reoffered_at_sample_rate() -> None:
    """The same site twice running is the order still being carried out."""
    world = sample(
        _CENTRE,
        _BUILDER,
        entity(400, "extractorT1", x=300.0, complete=False),
        credits=100_000,
        pools=(pool(x=300.0), pool(x=900.0, index=1)),
        options=(option(214, "extractorT1", placed=True),),
    )
    peer = _ScriptedPeer(lines(world, world, world))
    play(AgentChannel(peer), (), _CATALOGUE, _PLACEMENTS, _PROFILES, 3)
    assert len(_verb(peer, "build")) == 1


def test_an_unrelated_structure_does_not_count_as_a_workers_job() -> None:
    """Ownership, completeness and type all have to match, or an opponent's
    half-built building nearby would hold our worker busy forever.
    """
    world = sample(
        _CENTRE,
        _BUILDER,
        enemy(900, "extractorT1", x=300.0, complete=False),
        entity(401, "landFactory", x=300.0, complete=False),
        credits=100_000,
        pools=(pool(x=300.0), pool(x=900.0, index=1)),
        options=(option(214, "extractorT1", placed=True),),
    )
    peer = _ScriptedPeer(lines(world, world))
    play(AgentChannel(peer), (), _CATALOGUE, _PLACEMENTS, _PROFILES, 2)
    # Neither the enemy's nor the wrong-typed structure is this worker's job,
    # so the second observation still finds it free -- and the site it was sent
    # to is unchanged, which is what the repeat guard suppresses.
    assert len(_verb(peer, "build")) == 1


def test_counter_tilts_production_toward_the_air_the_opponent_fields() -> None:
    """The loop's own record, finally read by the loop.

    ``enemy_types_end`` was carried on every report while production stayed
    blind to it: three matches ended with 33 identical ``c_tank`` against
    aircraft none of them could shoot ([[mechanics-combat-profile]]). With the
    doctrine's counter switch on, the same world and the same mix produce the
    anti-air unit instead -- and with it off, the stated mix stands, which is
    the control arm every measurement so far was taken under.
    """
    catalogue = {**_CATALOGUE, "c_aa": _unit("c_aa")}
    profiles = {**profiles_for(catalogue), "c_aa": profile("c_aa", 120.0, air=True)}
    world = sample(
        _CENTRE,
        _FACTORY,
        enemy(9, "heli", x=100.0, flying=True),
        credits=4000,
        options=(option(300, "c_tank"), option(300, "c_aa", index=1)),
    )
    placements = {**_PLACEMENTS}

    held = _ScriptedPeer(lines(world))
    play(AgentChannel(held), (), catalogue, placements, profiles, 1, reinforce=("c_tank", "c_aa"))
    assert _verb(held, "produce") == ['{"kind":"produce","unit_id":300,"type":"c_tank"}']

    tilted = _ScriptedPeer(lines(world))
    play(
        AgentChannel(tilted),
        (),
        catalogue,
        placements,
        profiles,
        1,
        reinforce=("c_tank", "c_aa"),
        counter=True,
    )
    assert _verb(tilted, "produce") == ['{"kind":"produce","unit_id":300,"type":"c_aa"}']
