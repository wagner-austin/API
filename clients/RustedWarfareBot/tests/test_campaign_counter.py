"""Answering what the opponent fields, end to end.

Three switches read the same fact -- what has been seen in the enemy's air --
and spend on it differently: ``counter`` tilts what the factories produce,
``aa_cover`` buys a turret that can shoot it, and ``scouting`` keeps the
sighting alive after the fog closes. Each is driven through whole ticks,
because the interesting failures are in how they coexist.

Split from ``test_campaign_switches``, which keeps the switches that decide
where the army goes rather than what is bought.
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
    ScriptedPeer,
    unit_stats,
    verb,
)
from tests.wire_fixtures import (
    enemy,
    lines,
    option,
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
