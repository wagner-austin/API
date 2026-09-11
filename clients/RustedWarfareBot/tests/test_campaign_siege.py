"""The siege switch: reach joins the ratio only after the winning regime.

The counter-standoff knob (log 2026-09-10): the VH attrition anatomy read
the ten-loss tail as two-thirds ground standoff with enemy artillery the
single largest killer, while our own artillery -- the only reach we field
-- was correctly bred OUT of the fixed mix as a tempo cost in the games
that close early. ``siege`` returns it conditionally: one ``c_artillery``
share joins production once ``samples_seen`` passes the gate, the exact
sample mechanism ``worker_wait`` proved (``test_campaign_worker_wait``).

Same fail-first pairing as that module: the control world produces tanks
and never artillery, and the gated world is identical but for the knob.
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
from tests.wire_fixtures import entity, lines, option, pool, profiles_for, sample

#: The fixture catalogue plus the siege share's own unit, priced by the
#: real table ([[mechanics-unit-value]]).
SIEGE_CATALOGUE = {**CATALOGUE, "c_artillery": unit_stats("c_artillery", price=900)}

#: Profiles over the widened catalogue, so a world that FIELDS the share
#: (the dose pair's fixture) can be combat-profiled like any roster.
SIEGE_PROFILES = profiles_for(SIEGE_CATALOGUE)


def _siege_world() -> Sample:
    """A factory offering both types, with the tank half already fielded.

    The roster carries one ``c_tank``, so once the artillery share joins
    the ratio the deficit points at artillery; until then the composition
    is tanks alone and production keeps wanting one.

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


def _dosed_world() -> Sample:
    """The siege world with the single share already standing.

    One tank and one artillery on the roster: a single dose is satisfied
    and produces no more reach, so anything artillery produced here is
    the second share and nothing else.

    Returns:
        The scripted world.
    """
    return sample(
        CENTRE,
        BUILDER,
        FACTORY,
        entity(1, "c_tank"),
        entity(2, "c_artillery"),
        credits=4000,
        pools=(pool(x=300.0),),
        options=(
            option(300, "c_tank"),
            option(300, "c_artillery"),
            option(214, "extractorT1", placed=True),
        ),
    )


def _play(world: Sample, times: int, siege: int, siegedose: int = 1) -> ScriptedPeer:
    peer = ScriptedPeer(lines(*(world for _ in range(times))))
    play(
        AgentChannel(peer),
        (),
        SIEGE_CATALOGUE,
        PLACEMENTS,
        SIEGE_PROFILES,
        times,
        reinforce=("c_tank",),
        siege=siege,
        siegedose=siegedose,
    )
    return peer


def test_the_identity_never_fields_artillery() -> None:
    """The control half of the pair, and the fail-first witness: with the
    switch off, the same world produces tanks and artillery never."""
    peer = _play(_siege_world(), times=4, siege=0)
    produced = verb(peer, "produce")
    assert any('"type":"c_tank"' in line for line in produced)
    assert not any('"type":"c_artillery"' in line for line in produced)


def test_the_switch_fields_artillery_once_the_gate_expires() -> None:
    """Past the gate the share joins the ratio, and with a tank already
    standing the deficit points at the artillery."""
    peer = _play(_siege_world(), times=4, siege=2)
    assert any('"type":"c_artillery"' in line for line in verb(peer, "produce"))


def test_a_gate_beyond_the_match_never_switches() -> None:
    """Timing, not permission: the same knob set past the horizon is the
    identity -- the games that close early never pay the share."""
    peer = _play(_siege_world(), times=4, siege=100)
    assert not any('"type":"c_artillery"' in line for line in verb(peer, "produce"))


def test_a_filled_single_dose_is_satisfied() -> None:
    """The control half of the dose pair, and its fail-first witness: with
    one artillery already standing, the shipped single-share switch wants
    no more reach."""
    peer = _play(_dosed_world(), times=4, siege=2)
    assert not any('"type":"c_artillery"' in line for line in verb(peer, "produce"))


def test_the_dose_widens_the_share() -> None:
    """Dose two on the same world produces artillery again: the standing
    piece fills only the first share, and the deficit points at the
    second -- the dose axis the timing family's closure left open."""
    peer = _play(_dosed_world(), times=4, siege=2, siegedose=2)
    assert any('"type":"c_artillery"' in line for line in verb(peer, "produce"))
