"""The doctrine switches, end to end: counter, intercept, scout, raid and
the savings clock, each driven through a whole tick.

The growth file: every new behaviour flag lands its loop-level test here,
beside the others it has to coexist with.
"""

from __future__ import annotations

from rw_bot.control.channel import AgentChannel
from rw_bot.mechanics.placement import TypePlacement
from rw_bot.policy.campaign import play
from tests.campaign_fixtures import (
    BUILDER,
    CATALOGUE,
    CENTRE,
    FACTORY,
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
    pool,
    profile,
    profiles_for,
    sample,
)


def test_counter_tilts_production_toward_the_air_the_opponent_fields() -> None:
    """The loop's own record, finally read by the loop.

    ``enemy_types_end`` was carried on every report while production stayed
    blind to it: three matches ended with 33 identical ``c_tank`` against
    aircraft none of them could shoot ([[mechanics-combat-profile]]). With the
    doctrine's counter switch on, the same world and the same mix produce the
    anti-air unit instead -- and with it off, the stated mix stands, which is
    the control arm every measurement so far was taken under.
    """
    catalogue = {**CATALOGUE, "c_aa": unit_stats("c_aa")}
    profiles = {**profiles_for(catalogue), "c_aa": profile("c_aa", 120.0, air=True)}
    world = sample(
        CENTRE,
        FACTORY,
        enemy(9, "heli", x=100.0, flying=True),
        credits=4000,
        options=(option(300, "c_tank"), option(300, "c_aa", index=1)),
    )
    placements = {**PLACEMENTS}

    held = ScriptedPeer(lines(world))
    play(AgentChannel(held), (), catalogue, placements, profiles, 1, reinforce=("c_tank", "c_aa"))
    assert verb(held, "produce") == ['{"kind":"produce","unit_id":300,"type":"c_tank"}']

    tilted = ScriptedPeer(lines(world))
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
    assert verb(tilted, "produce") == ['{"kind":"produce","unit_id":300,"type":"c_aa"}']


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


def test_aa_cover_places_an_anti_air_turret_once_air_is_seen() -> None:
    """The gap this closes is total: the whole army and the ground turret
    declare ``canAttackFlyingUnits: false``, so before this switch nothing
    the bot could place touched an aircraft at all
    ([[policy-holding-ground]]). Only once the opponent has SHOWN aircraft --
    an anti-air turret cannot hit the ground, so before that it is 600
    credits pointed at a guess -- and with the switch off, the same world
    buys nothing, which is the behaviour every prior measurement was taken
    under.
    """
    catalogue = {
        **CATALOGUE,
        "c_antiAirTurret": unit_stats("c_antiAirTurret", speed=0.0, armed=True, price=600),
    }
    placements = {
        name: TypePlacement(index=i, type_name=name, needs_pool=False)
        for i, name in enumerate(catalogue)
    }
    profiles = {
        **profiles_for(catalogue),
        "c_antiAirTurret": profile("c_antiAirTurret", 250.0, land=False, air=True),
    }
    world = sample(
        CENTRE,
        BUILDER,
        enemy(9, "gunShip", x=700.0, flying=True),
        credits=4000,
        options=(option(214, "c_antiAirTurret", placed=True),),
    )

    blind = ScriptedPeer(lines(world, world))
    play(AgentChannel(blind), (), catalogue, placements, profiles, 2)
    assert verb(blind, "build") == []

    covered = ScriptedPeer(lines(world, world))
    play(AgentChannel(covered), (), catalogue, placements, profiles, 2, aa_cover=True)
    assert verb(covered, "build") == [
        '{"kind":"build","unit_id":214,"x":60.0,"y":0.0,"type":"c_antiAirTurret"}'
    ]


def test_aa_cover_outranks_ground_cover_once_air_is_shown() -> None:
    """V1 put anti-air after ground defence and its own reach line convicted
    it in one batch: reached 50, acted 0, zero AA turrets standing across
    twelve matches -- never reached with 600 credits left. Ground raiders
    already have the guard; nothing else touches an aircraft, so on the
    latch the AA turret is bought first."""
    catalogue = {
        **CATALOGUE,
        "c_turret_t1": unit_stats("c_turret_t1", speed=0.0, armed=True, price=500),
        "c_antiAirTurret": unit_stats("c_antiAirTurret", speed=0.0, armed=True, price=600),
    }
    placements = {
        name: TypePlacement(index=i, type_name=name, needs_pool=False)
        for i, name in enumerate(catalogue)
    }
    profiles = {
        **profiles_for(catalogue),
        "c_turret_t1": profile("c_turret_t1", 165.0),
        "c_antiAirTurret": profile("c_antiAirTurret", 250.0, land=False, air=True),
    }
    world = sample(
        CENTRE,
        BUILDER,
        enemy(9, "gunShip", x=700.0, flying=True),
        credits=4000,
        options=(
            option(214, "c_turret_t1", placed=True),
            option(214, "c_antiAirTurret", placed=True, index=1),
        ),
    )
    peer = ScriptedPeer(lines(world, world))
    play(AgentChannel(peer), (), catalogue, placements, profiles, 2, aa_cover=True)
    builds = verb(peer, "build")
    assert builds
    assert '"type":"c_antiAirTurret"' in builds[0]


def test_ground_cover_still_happens_when_the_aa_turret_is_unaffordable() -> None:
    """The inversion is a preference, not a blockade: an AA turret the balance
    cannot cover falls through to the ground turret it can, rather than
    holding all cover hostage to 600 credits."""
    catalogue = {
        **CATALOGUE,
        "c_turret_t1": unit_stats("c_turret_t1", speed=0.0, armed=True, price=500),
        "c_antiAirTurret": unit_stats("c_antiAirTurret", speed=0.0, armed=True, price=600),
    }
    placements = {
        name: TypePlacement(index=i, type_name=name, needs_pool=False)
        for i, name in enumerate(catalogue)
    }
    profiles = {
        **profiles_for(catalogue),
        "c_turret_t1": profile("c_turret_t1", 165.0),
        "c_antiAirTurret": profile("c_antiAirTurret", 250.0, land=False, air=True),
    }
    world = sample(
        CENTRE,
        BUILDER,
        enemy(9, "gunShip", x=700.0, flying=True),
        credits=550,
        options=(
            option(214, "c_turret_t1", placed=True),
            option(214, "c_antiAirTurret", placed=True, index=1),
        ),
    )
    peer = ScriptedPeer(lines(world, world))
    play(AgentChannel(peer), (), catalogue, placements, profiles, 2, aa_cover=True)
    builds = verb(peer, "build")
    assert builds
    assert '"type":"c_turret_t1"' in builds[0]


def test_aa_cover_waits_until_aircraft_are_actually_shown() -> None:
    """Latched from sight, not assumed: with no aircraft ever seen the switch
    buys nothing, because an anti-air turret cannot hit the ground."""
    catalogue = {
        **CATALOGUE,
        "c_antiAirTurret": unit_stats("c_antiAirTurret", speed=0.0, armed=True, price=600),
    }
    placements = {
        name: TypePlacement(index=i, type_name=name, needs_pool=False)
        for i, name in enumerate(catalogue)
    }
    profiles = {
        **profiles_for(catalogue),
        "c_antiAirTurret": profile("c_antiAirTurret", 250.0, land=False, air=True),
    }
    grounded = sample(
        CENTRE,
        BUILDER,
        enemy(9, "c_tank", x=700.0),
        credits=4000,
        options=(option(214, "c_antiAirTurret", placed=True),),
    )
    peer = ScriptedPeer(lines(grounded, grounded))
    play(AgentChannel(peer), (), catalogue, placements, profiles, 2, aa_cover=True)
    assert verb(peer, "build") == []


def test_cover_off_buys_no_turret_where_cover_on_buys_one() -> None:
    """The question working siting finally made askable: "defence on" meant
    "defence attempted and silently refused" for its whole measured history,
    and the first batch where turrets landed spent 25-45k a match on them
    and won 6/24 at a rung the attempted-defence bot won 10/12
    ([[policy-holding-ground]])."""
    catalogue = {
        **CATALOGUE,
        "c_turret_t1": unit_stats("c_turret_t1", speed=0.0, armed=True, price=500),
    }
    placements = {
        name: TypePlacement(index=i, type_name=name, needs_pool=False)
        for i, name in enumerate(catalogue)
    }
    profiles = {
        **profiles_for(catalogue),
        "c_turret_t1": profile("c_turret_t1", 165.0),
    }
    world = sample(
        CENTRE,
        BUILDER,
        credits=4000,
        options=(option(214, "c_turret_t1", placed=True),),
    )

    covered = ScriptedPeer(lines(world))
    play(AgentChannel(covered), (), catalogue, placements, profiles, 1)
    assert '"type":"c_turret_t1"' in verb(covered, "build")[0]

    bare = ScriptedPeer(lines(world))
    play(AgentChannel(bare), (), catalogue, placements, profiles, 1, cover=False)
    assert verb(bare, "build") == []


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
    play(AgentChannel(creeping), (), catalogue, PLACEMENTS, profiles, 1, creep=True)
    built = [line for line in creeping.sent if '"build"' in line and "c_turret_t1" in line]
    # Anchor (0,0), pool centroid (400,200) -> mirror (800,400): one reach
    # (165) along that line is (147.6..., 73.8...).
    assert len(built) == 1
    assert '"unit_id":214' in built[0]


def test_scouting_remembers_the_air_after_it_fogs() -> None:
    """The tilt reacts to what was seen, not only to what is shooting.

    Sample one shows a helicopter; sample two shows empty sky. Without
    scouting the tilt forgets with the fog and production reverts to the
    stated mix; with it, the remembered sighting keeps the anti-air share up.
    The scout itself is excluded from the army, so it is never marched into a
    wave.
    """
    catalogue = {
        **CATALOGUE,
        "c_aa": unit_stats("c_aa"),
        "scout": unit_stats("scout", speed=1.4),
        # The tilt filters remembered threats to mobile units, so the
        # helicopter has to be priced and moving to count.
        "heli": unit_stats("heli", speed=2.0),
    }
    profiles = {
        **profiles_for(catalogue),
        "c_aa": profile("c_aa", 120.0, air=True),
    }
    seen = sample(
        CENTRE,
        FACTORY,
        enemy(9, "heli", x=100.0, flying=True),
        # A remembered structure must NOT count toward the tilt -- v1 fed the
        # tilt everything and the flying share drowned in buildings.
        enemy(15, "extractorT1", x=600.0),
        credits=4000,
        options=(option(300, "c_tank"), option(300, "c_aa", index=1)),
    )
    fogged = sample(
        CENTRE,
        FACTORY,
        credits=4000,
        options=(option(300, "c_tank"), option(300, "c_aa", index=1)),
    )

    forgetful = ScriptedPeer(lines(seen, fogged))
    play(
        AgentChannel(forgetful),
        (),
        catalogue,
        PLACEMENTS,
        profiles,
        2,
        reinforce=("c_tank", "c_aa"),
        counter=True,
    )
    assert verb(forgetful, "produce")[-1] == '{"kind":"produce","unit_id":300,"type":"c_tank"}'

    remembering = ScriptedPeer(lines(seen, fogged))
    report = play(
        AgentChannel(remembering),
        (),
        catalogue,
        PLACEMENTS,
        profiles,
        2,
        reinforce=("c_tank", "c_aa"),
        counter=True,
        scout=True,
    )
    assert verb(remembering, "produce")[-1] == '{"kind":"produce","unit_id":300,"type":"c_aa"}'
    assert report["sightings"] == 2


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
