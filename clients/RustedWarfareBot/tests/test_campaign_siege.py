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
    PROFILES,
    ScriptedPeer,
    unit_stats,
    verb,
)
from tests.wire_fixtures import entity, lines, option, pool, sample

#: The fixture catalogue plus the siege share's own unit, priced by the
#: real table ([[mechanics-unit-value]]).
SIEGE_CATALOGUE = {**CATALOGUE, "c_artillery": unit_stats("c_artillery", price=900)}


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


def _play(world: Sample, times: int, siege: int) -> ScriptedPeer:
    peer = ScriptedPeer(lines(*(world for _ in range(times))))
    play(
        AgentChannel(peer),
        (),
        SIEGE_CATALOGUE,
        PLACEMENTS,
        PROFILES,
        times,
        reinforce=("c_tank",),
        siege=siege,
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
