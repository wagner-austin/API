"""The doctrine switches that send the army somewhere, end to end.

Release, march, raid, lurk, scatter and intercept: each is driven through a
whole tick, beside the others it has to coexist with. The reflex table the
agent is armed with before the first observation is here too, because it is
the same doctrine being announced.

The growth file: a new behaviour flag lands its loop-level test here, unless
it is about answering the opponent's composition -- those live beside the
counter tilt in ``test_campaign_counter``.
"""

from __future__ import annotations

from pathlib import Path

from rw_bot.control.channel import AgentChannel
from rw_bot.policy.campaign import play
from rw_bot.policy.situation import CLOSE_HOLD
from rw_bot.wire.state import Sample
from tests.campaign_fixtures import (
    BUILDER,
    CATALOGUE,
    CENTRE,
    PLACEMENTS,
    PROFILES,
    WAVE,
    ScriptedPeer,
    run_campaign,
    unit_stats,
    verb,
)
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


def test_intercept_turns_the_reserve_on_a_raider_among_the_extractors() -> None:
    """The disease the counter A/B diagnosed: extractors bleed mid-match while
    the reserve stands at the rally point. With the doctrine's intercept
    switch on, a raider inside the outpost radius of our structures pulls the
    reserve onto it; with it off, the same world produces no attack at all,
    which is the behaviour every measurement before this was taken under.
    """
    world = sample(
        CENTRE,
        entity(1, "c_tank", x=80.0, y=0.0),
        entity(2, "c_tank", x=120.0, y=0.0),
        enemy(9, "c_tank", x=200.0),
    )

    held = ScriptedPeer(lines(world))
    report = play(AgentChannel(held), (), CATALOGUE, PLACEMENTS, PROFILES, 1)
    assert verb(held, "attack") == []
    assert report["intercepts"] == 0

    guarded = ScriptedPeer(lines(world))
    report = play(AgentChannel(guarded), (), CATALOGUE, PLACEMENTS, PROFILES, 1, intercept=True)
    assert verb(guarded, "attack") == [
        '{"kind":"attack","unit_id":1,"target_id":9}',
        '{"kind":"attack","unit_id":2,"target_id":9}',
    ]
    assert report["intercepts"] == 2


def test_rush_marches_the_released_wave_at_the_mirror_of_the_base() -> None:
    """The all-in verb: with nothing visible to fight, a released wave used
    to stand at the rally point waiting for an opponent who never needed to
    come -- every measured Impossible match ended without the bot reaching
    the enemy base ([[policy-holding-ground]]). With the switch on, the
    released wave attack-moves at the anchor's reflection through the pool
    centroid; with it off, the same world sends nobody anywhere."""
    world = sample(
        CENTRE,
        *WAVE,
        pools=(pool(x=300.0, y=300.0), pool(x=500.0, y=100.0)),
    )

    held = ScriptedPeer(lines(world))
    play(AgentChannel(held), (), CATALOGUE, PLACEMENTS, PROFILES, 1)
    assert [line for line in held.sent if "attack_move" in line] == []

    rushing = ScriptedPeer(lines(world))
    report = play(AgentChannel(rushing), (), CATALOGUE, PLACEMENTS, PROFILES, 1, rush=True)
    marched = [line for line in rushing.sent if "attack_move" in line]
    # Pool centroid (400,200), anchor (0,0) -> mirror (800,400); the whole
    # first wave marches together.
    assert marched == [
        '{"kind":"attack_move","unit_id":1,"x":800.0,"y":400.0}',
        '{"kind":"attack_move","unit_id":2,"x":800.0,"y":400.0}',
        '{"kind":"attack_move","unit_id":3,"x":800.0,"y":400.0}',
    ]
    assert report["marches"] == 3


def test_the_closer_releases_and_marches_on_dominance() -> None:
    """The verb the vh-close sweep demanded: nineteen dominant positions at
    the 4,000-sample cap, eleven of them LOST at 10,000 -- dominance decays,
    so a decided match is ended while it is decided ([[policy-situation]]).
    Dominance met, the wave releases and marches without the rush switch;
    dominance short of the multiple, the same world sends nobody anywhere.
    """
    scoreboard = (
        player(0, index=0, local=True, hostile=False, army_value=9_000),
        player(1, index=1, hostile=True, army_value=3_000),
    )
    world = sample(
        CENTRE,
        *WAVE,
        pools=(pool(x=300.0, y=300.0), pool(x=500.0, y=100.0)),
        players=scoreboard,
    )

    # CLOSE_HOLD dominant samples: the debounce runs down, the latch
    # commits, and the wave marches on the committing tick.
    closing = ScriptedPeer(lines(*(world for _ in range(CLOSE_HOLD))))
    report = play(AgentChannel(closing), (), CATALOGUE, PLACEMENTS, PROFILES, CLOSE_HOLD, close=3)
    marched = [line for line in closing.sent if "attack_move" in line]
    # A FORCED march aims at the income posts rather than the bare mirror --
    # the all-in's rule, inherited deliberately: the closer exists to end the
    # match, and the match is ended where the economy stands. The party is
    # SPREAD across the posts, so each pool meets part of the wave.
    assert marched == [
        '{"kind":"attack_move","unit_id":1,"x":500.0,"y":100.0}',
        '{"kind":"attack_move","unit_id":2,"x":300.0,"y":300.0}',
        '{"kind":"attack_move","unit_id":3,"x":500.0,"y":100.0}',
    ]
    assert report["marches"] == 3

    contested = sample(
        CENTRE,
        *WAVE,
        pools=(pool(x=300.0, y=300.0), pool(x=500.0, y=100.0)),
        players=(
            player(0, index=0, local=True, hostile=False, army_value=5_000),
            player(1, index=1, hostile=True, army_value=3_000),
        ),
    )
    holding = ScriptedPeer(lines(*(contested for _ in range(CLOSE_HOLD))))
    play(AgentChannel(holding), (), CATALOGUE, PLACEMENTS, PROFILES, CLOSE_HOLD, close=3)
    assert [line for line in holding.sent if "attack_move" in line] == []


def test_the_closer_latches_once_dominance_was_confirmed() -> None:
    """Dominance confirmed once is a decision, not a reading to re-take:
    un-latched, the window flickered as trades moved the ratio and three
    lost matches show 9, 3 and 6 marches dying in dribbles
    (`runs/sweeps/vh-closer`, log 2026-08-01). A reinforcement built AFTER
    the ratio has slipped below the multiple still marches."""
    scoreboard_dominant = (
        player(0, index=0, local=True, hostile=False, army_value=9_000),
        player(1, index=1, hostile=True, army_value=3_000),
    )
    scoreboard_slipped = (
        player(0, index=0, local=True, hostile=False, army_value=5_000),
        player(1, index=1, hostile=True, army_value=3_000),
    )
    pools = (pool(x=300.0, y=300.0), pool(x=500.0, y=100.0))
    dominant = sample(CENTRE, *WAVE, pools=pools, players=scoreboard_dominant)
    # The ratio has slipped AND a fresh wave-sized group has rolled out:
    # without the latch this tick sends nobody anywhere. A wave-sized group
    # rather than a lone tank, because the anti-trickle floor holds under
    # force too -- fewer than a first wave is not a punch
    # ([[policy-combat]]).
    slipped = sample(
        CENTRE,
        *WAVE,
        entity(4, "c_tank", x=60.0),
        entity(5, "c_tank", x=70.0),
        entity(6, "c_tank", x=80.0),
        pools=pools,
        players=scoreboard_slipped,
    )
    script = (*(dominant for _ in range(CLOSE_HOLD)), slipped)
    peer = ScriptedPeer(lines(*script))
    play(AgentChannel(peer), (), CATALOGUE, PLACEMENTS, PROFILES, len(script), close=3)
    marched = "".join(line for line in peer.sent if "attack_move" in line)
    for unit_id in (4, 5, 6):
        assert f'{{"kind":"attack_move","unit_id":{unit_id}' in marched


def test_creep_walks_a_turret_toward_the_mirror_and_off_builds_none() -> None:
    """The creep verb: the documented human answer to the cheating
    difficulties, expressed as one doctrine flag. With it on, a free worker
    is sent to put a turret one turret-reach along the line to the enemy's
    estimated start; with it off, the same world builds nothing
    ([[policy-creep]], [[ai-opponent-strategy]])."""
    catalogue = {**CATALOGUE, "c_turret_t1": unit_stats("c_turret_t1", speed=0.0, price=500)}
    profiles = {**profiles_for(catalogue), "c_turret_t1": profile("c_turret_t1", 165.0)}
    world = sample(
        CENTRE,
        BUILDER,
        credits=4000,
        pools=(pool(x=300.0, y=300.0), pool(x=500.0, y=100.0)),
    )

    held = ScriptedPeer(lines(world))
    play(AgentChannel(held), (), catalogue, PLACEMENTS, profiles, 1)
    assert [line for line in held.sent if "c_turret_t1" in line] == []

    creeping = ScriptedPeer(lines(world))
    play(AgentChannel(creeping), (), catalogue, PLACEMENTS, profiles, 1, creep=100)
    built = [line for line in creeping.sent if '"build"' in line and "c_turret_t1" in line]
    # Anchor (0,0), pool centroid (400,200) -> mirror (800,400): one reach
    # (165) along that line is (147.6..., 73.8...).
    assert len(built) == 1
    assert '"unit_id":214' in built[0]


def test_the_raid_sends_a_party_at_remembered_income() -> None:
    """Sample one shows an enemy extractor; sample two fogs it. The raid
    attack-moves a party at the memory, and the party is withheld from the
    waves -- with the flag off, the same world sends nobody anywhere.

    Six tanks, not three: the draft is arbitrated against the wave gate's
    need plus the party size, because v1 drafted from the gate itself and
    was refuted 0/12 for it (log: 2026-07-29). Three tanks are the gate's.
    """
    surplus = (*WAVE, entity(4, "c_tank"), entity(5, "c_tank"), entity(6, "c_tank"))
    seen = sample(CENTRE, *surplus, enemy(9, "extractorT1", x=800.0))
    fogged = sample(CENTRE, *surplus)

    idle = ScriptedPeer(lines(seen, fogged))
    play(AgentChannel(idle), (), CATALOGUE, PLACEMENTS, PROFILES, 2)
    assert [line for line in idle.sent if "attack_move" in line] == []

    raiding = ScriptedPeer(lines(seen, fogged))
    report = play(AgentChannel(raiding), (), CATALOGUE, PLACEMENTS, PROFILES, 2, raid=3)
    marched = [line for line in raiding.sent if "attack_move" in line]
    assert marched == [
        '{"kind":"attack_move","unit_id":1,"x":800.0,"y":0.0}',
        '{"kind":"attack_move","unit_id":2,"x":800.0,"y":0.0}',
        '{"kind":"attack_move","unit_id":3,"x":800.0,"y":0.0}',
    ]
    assert report["raids"] == 1
    assert report["marches"] == 3


def test_a_save_that_never_progresses_frees_the_worker() -> None:
    """The amphib arm's disease, cured end to end.

    While the plan waits on a price it claims nothing, so production and
    expansion keep spending the income -- and an entry priced beyond the
    economy's reach held the only worker hostage for whole matches:
    ``plan-holds-only-worker reached 255`` and climbing, twelve seeds, none
    ever affording the 11,000-credit prerequisite (log: 2026-07-29). The
    savings clock rules the plan blocked, out loud, and the worker goes back
    to the economy: the hold is reached once here, on the sample before the
    ruling, and never again.
    """
    world = sample(
        CENTRE,
        BUILDER,
        credits=10,
        options=(option(214, "landFactory", placed=True),),
    )
    report, _ = run_campaign(world, times=3, plan=("landFactory",), afford_samples=1)
    assert report["build_outcome"] == "blocked"
    assert report["build_reason"] == (
        "landFactory costs 1000, holding 10; the shortfall never shrank below "
        "990 across 1 samples -- income is spoken for and this save is not "
        "happening; the worker is released"
    )
    holds = [r for r in report["reaches"] if r["stage"] == "plan-holds-only-worker"]
    assert [r["reached"] for r in holds] == [1]


def test_the_raid_stands_down_without_surplus() -> None:
    """The same remembered extractor, the gate's worth of tanks and not one
    more: no party is drafted, and the waves keep every unit."""
    seen = sample(CENTRE, *WAVE, enemy(9, "extractorT1", x=800.0))
    fogged = sample(CENTRE, *WAVE)
    lean = ScriptedPeer(lines(seen, fogged))
    report = play(AgentChannel(lean), (), CATALOGUE, PLACEMENTS, PROFILES, 2, raid=3)
    assert [line for line in lean.sent if "attack_move" in line] == []
    assert report["raids"] == 0
    assert report["marches"] == 0


def test_the_lurker_walks_to_the_enemy_start_and_off_stays_home() -> None:
    """The leash verb end to end: with ``lurk`` a scout is sent to the
    mirrored enemy start, and the same world with it off sends no such
    move -- the AI recalls its armies home for as long as the intruder
    stands there ([[ai-opponent-strategy]]).
    """
    catalogue = {**CATALOGUE, "scout": unit_stats("scout", price=700)}
    profiles = profiles_for(catalogue)
    world = sample(
        CENTRE,
        entity(300, "scout", x=10.0, y=0.0),
        credits=4000,
        pools=(pool(x=500.0, y=0.0),),
    )
    held = ScriptedPeer(lines(world))
    play(AgentChannel(held), (), catalogue, PLACEMENTS, profiles, 1)
    assert [line for line in verb(held, "move") if '"unit_id":300' in line] == []

    leashed = ScriptedPeer(lines(world))
    play(AgentChannel(leashed), (), catalogue, PLACEMENTS, profiles, 1, lurk=1)
    walked = [line for line in verb(leashed, "move") if '"unit_id":300' in line]
    # The post is the zone rim: the mirrored start at (1000, 0), pulled back
    # by the standoff along the line home.
    assert walked == ['{"kind":"move","unit_id":300,"x":620.0,"y":0.0}']


def test_a_committed_marcher_is_withheld_from_the_engagement() -> None:
    """The strike force walks past the fight it was built to walk past.

    Tick one commits the all-in and marches the tank at the enemy's income;
    tick two shows a hostile in reach -- and the marcher is NOT re-tasked
    onto it, where the same world without the all-in attacks it at once
    ([[policy-combat]], log 2026-07-31).
    """

    def _tick(hostile_id: int) -> Sample:
        return sample(
            CENTRE,
            entity(10, "c_tank", x=100.0, y=0.0),
            entity(11, "c_tank", x=110.0, y=0.0),
            entity(12, "c_tank", x=120.0, y=0.0),
            enemy(hostile_id, "c_tank", x=200.0, y=0.0),
            credits=4000,
            pools=(pool(x=500.0, y=0.0),),
        )

    # The release tick may still engage -- commitment is read before the
    # command that sets it, and the march waypoint lands last and wins in
    # the engine. What must hold is every tick AFTER: a fresh hostile
    # appears and the marchers ignore it.
    committed = ScriptedPeer(lines(_tick(9), _tick(8)))
    play(AgentChannel(committed), (), CATALOGUE, PLACEMENTS, PROFILES, 2, allin=1)
    retasked = [line for line in verb(committed, "attack") if '"target_id":8' in line]
    assert retasked == []
    marched = [line for line in committed.sent if "attack_move" in line and '"unit_id":10' in line]
    assert len(marched) == 1

    plain = ScriptedPeer(lines(_tick(9), _tick(8)))
    play(AgentChannel(plain), (), CATALOGUE, PLACEMENTS, PROFILES, 2, intercept=True)
    assert [line for line in verb(plain, "attack") if '"target_id":8' in line] != []


def test_the_decoys_scatter_to_their_posts() -> None:
    """The scatter verb end to end: with ``decoys`` a scout is sent to a
    flank post; off, it stays put. The AI's target lottery is uniform over
    all our units, so the post is a ticket bought on purpose
    ([[ai-opponent-strategy]]).
    """
    catalogue = {**CATALOGUE, "scout": unit_stats("scout", price=700)}
    profiles = profiles_for(catalogue)
    world = sample(
        CENTRE,
        entity(300, "scout", x=10.0, y=0.0),
        credits=4000,
        pools=(pool(x=500.0, y=0.0),),
    )
    held = ScriptedPeer(lines(world))
    play(AgentChannel(held), (), catalogue, PLACEMENTS, profiles, 1)
    assert [line for line in verb(held, "move") if '"unit_id":300' in line] == []

    scattered = ScriptedPeer(lines(world))
    play(AgentChannel(scattered), (), catalogue, PLACEMENTS, profiles, 1, decoys=1)
    walked = [line for line in verb(scattered, "move") if '"unit_id":300' in line]
    assert len(walked) == 1


def test_the_posture_table_is_sent_once_before_the_first_observation() -> None:
    """The reflex layer's table: one row per profiled type, reach from the
    planner's own catalogue, the reflexes only on armed mobile types --
    and with both knobs off, no rows at all
    ([[community-play-strategies]]).
    """
    world = sample(CENTRE, entity(10, "c_tank", x=100.0, y=0.0), credits=4000)

    off = ScriptedPeer(lines(world))
    play(AgentChannel(off), (), CATALOGUE, PLACEMENTS, PROFILES, 1)
    assert [line for line in off.sent if '"posture"' in line] == []

    on = ScriptedPeer(lines(world))
    play(AgentChannel(on), (), CATALOGUE, PLACEMENTS, PROFILES, 1, kite=True, hp_floor=30)
    rows = [line for line in on.sent if '"posture"' in line]
    assert len(rows) == len(PROFILES)
    tank = next(line for line in rows if '"type":"c_tank"' in line)
    assert '"kite":1' in tank
    assert '"hp_floor":30' in tank
    # A structure carries its reach for the threat lookup and no reflexes:
    # it cannot walk, so a flee order would be noise.
    centre = next(line for line in rows if '"type":"commandCenter"' in line)
    assert '"kite":0' in centre
    assert '"hp_floor":0' in centre


def test_an_open_strike_window_lands_in_the_trace_events(tmp_path: Path) -> None:
    """The decision stream, end to end: the rival's army value falls past
    the strike figure, the window opens in the fighting tail, and the NEXT
    trace row carries ``S`` -- each row holds what was decided in the
    window it closes, so the loop's order never bends for the record
    (log 2026-08-09)."""
    us = player(0, index=0, local=True, hostile=False, income=54, building_value=3000)

    def world(their_army: int) -> Sample:
        them = player(1, index=1, income=180, army_value=their_army, building_value=1500)
        return sample(CENTRE, *WAVE, enemy(9, "c_tank", x=100.0), players=(us, them))

    peer = ScriptedPeer(lines(world(4200), world(4000), world(4000)))
    target = tmp_path / "trace.txt"
    play(AgentChannel(peer), (), CATALOGUE, PLACEMENTS, PROFILES, 3, strike=100, trace=target)
    rows = [ln.split() for ln in target.read_text(encoding="utf-8").splitlines()[1:4]]
    events = [row[20] for row in rows]
    assert events[0] == "-"
    assert "S" in events[2]
