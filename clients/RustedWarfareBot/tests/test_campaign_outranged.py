"""The outranged switch: reach joins the ratio only while the enemy shows it.

The threat-armed sibling of the siege switch (``test_campaign_siege``). The
time gate closed 0-for in the attrition tier because it fired in every long
game; the pinbase48 split (log 2026-09-12) is binary -- 39/39 against
enemies that never field artillery, 0/9 against the scout-plus-artillery
opening -- and the static merge measured catastrophic in both directions
(arty39). So the share joins on the SEEN opening: a hostile land mover
whose gun starts beyond every land gun in the mix.

Same fail-first pairing as the siege module: the control world shows the
threat and produces tanks only, the armed world is identical but for the
knob -- and the armed world WITHOUT the threat is the identity, which is
the property that makes the clause adoptable where the static merge was
not.
"""

from __future__ import annotations

from rw_bot.control.channel import AgentChannel
from rw_bot.policy.campaign import play
from rw_bot.wire.state import Sample
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
from tests.wire_fixtures import enemy, entity, lines, option, pool, profile, profiles_for, sample

#: The fixture catalogue plus the joined share's own unit, priced by the
#: real table ([[mechanics-unit-value]]) -- and the enemy's pieces, because
#: the clause reads the catalogue's speed to tell a mover from a building:
#: the artillery moves, the hall does not.
OUTRANGED_CATALOGUE = {
    **CATALOGUE,
    "c_artillery": unit_stats("c_artillery", price=900),
    "enemy_arty": unit_stats("enemy_arty"),
    "enemy_hall": unit_stats("enemy_hall", speed=0.0),
}

#: Profiles over the widened catalogue, with the enemy's artillery stated at
#: the live game's 290 reach -- beyond every fixture gun (110), which is the
#: standoff the clause reads -- and the hall at a fortress gun's 400, which
#: must NOT read as one.
OUTRANGED_PROFILES = {
    **profiles_for(OUTRANGED_CATALOGUE),
    "enemy_arty": profile("enemy_arty", 290.0),
    "enemy_hall": profile("enemy_hall", 400.0),
}


def _standoff_world() -> Sample:
    """A factory offering both types, with enemy artillery in sight.

    The roster carries one ``c_tank``, so once the artillery share joins
    the ratio the deficit points at artillery; without the join the
    composition is tanks alone and production keeps wanting one.

    Returns:
        The scripted world.
    """
    return sample(
        CENTRE,
        BUILDER,
        FACTORY,
        entity(1, "c_tank"),
        enemy(9, "enemy_arty", x=400.0),
        credits=4000,
        pools=(pool(x=300.0),),
        options=(
            option(300, "c_tank"),
            option(300, "c_artillery"),
            option(214, "extractorT1", placed=True),
        ),
    )


def _quiet_world() -> Sample:
    """The standoff world with the enemy artillery absent.

    Returns:
        The scripted world.
    """
    return sample(
        CENTRE,
        BUILDER,
        FACTORY,
        entity(1, "c_tank"),
        credits=4000,
        pools=(pool(x=300.0),),
        options=(
            option(300, "c_tank"),
            option(300, "c_artillery"),
            option(214, "extractorT1", placed=True),
        ),
    )


def _play(world: Sample, times: int, outranged: bool) -> ScriptedPeer:
    peer = ScriptedPeer(lines(*(world for _ in range(times))))
    play(
        AgentChannel(peer),
        (),
        OUTRANGED_CATALOGUE,
        PLACEMENTS,
        OUTRANGED_PROFILES,
        times,
        reinforce=("c_tank",),
        outranged=outranged,
    )
    return peer


def test_the_identity_never_fields_artillery_even_against_the_standoff() -> None:
    """The control half of the pair, and the fail-first witness: with the
    switch off, the seen artillery changes nothing and the same world
    produces tanks only -- which is exactly how the champion lost the nine
    opener seeds."""
    peer = _play(_standoff_world(), times=4, outranged=False)
    produced = verb(peer, "produce")
    assert any('"type":"c_tank"' in line for line in produced)
    assert not any('"type":"c_artillery"' in line for line in produced)


def test_the_switch_fields_artillery_while_the_enemy_shows_it() -> None:
    """Against the seen standoff the share joins the ratio, and with a tank
    already standing the deficit points at the artillery."""
    peer = _play(_standoff_world(), times=4, outranged=True)
    produced = verb(peer, "produce")
    assert any('"type":"c_artillery"' in line for line in produced)


def test_the_armed_switch_is_the_identity_when_nothing_outranges() -> None:
    """No seen standoff, no join: the armed doctrine plays the champion's
    game bit for bit on a quiet seed, which is the property that separates
    this clause from the static merge arty39 refused."""
    peer = _play(_quiet_world(), times=4, outranged=True)
    produced = verb(peer, "produce")
    assert any('"type":"c_tank"' in line for line in produced)
    assert not any('"type":"c_artillery"' in line for line in produced)


def _hall_world() -> Sample:
    """The quiet world plus the enemy's armed command centre in sight.

    Returns:
        The scripted world.
    """
    return sample(
        CENTRE,
        BUILDER,
        FACTORY,
        entity(1, "c_tank"),
        enemy(9, "enemy_hall", x=500.0),
        credits=4000,
        pools=(pool(x=300.0),),
        options=(
            option(300, "c_tank"),
            option(300, "c_artillery"),
            option(214, "extractorT1", placed=True),
        ),
    )


def test_an_armed_enemy_structure_never_joins_the_share() -> None:
    """The frame-0 regression witness (condprobe13, log 2026-09-12): the
    first wiring read the enemy command centre -- armed, outranging
    everything, visible forever -- and held the join on from the first
    observation of every match, on winning seeds included. A standoff is a
    fight against something that can come to you; the hall cannot."""
    peer = _play(_hall_world(), times=4, outranged=True)
    produced = verb(peer, "produce")
    assert any('"type":"c_tank"' in line for line in produced)
    assert not any('"type":"c_artillery"' in line for line in produced)
