"""The nukes doctrine switch, end to end through the campaign tick.

The unit rules live in ``test_policy_nuker``; what these hold is the
wiring -- the knob reaches the channel, its orders leave on the wire in
the agent's exact format, and off is off ([[policy-loop]]).
"""

from __future__ import annotations

from rw_bot.control.channel import AgentChannel
from rw_bot.policy.campaign import play
from rw_bot.policy.head import decode_head_model
from rw_bot.policy.nuker import LAUNCHER_TYPE
from rw_bot.policy.situation import CLOSE_HOLD
from tests.campaign_fixtures import (
    BUILDER,
    CATALOGUE,
    CENTRE,
    PLACEMENTS,
    ScriptedPeer,
    unit_stats,
)
from tests.wire_fixtures import enemy, entity, lines, option, player, profiles_for, sample

_CATALOGUE = {
    **CATALOGUE,
    LAUNCHER_TYPE: unit_stats(LAUNCHER_TYPE, speed=0.0, armed=False, price=45000),
}
_PROFILES = profiles_for(_CATALOGUE)


def test_the_nukes_knob_places_a_funded_launcher_once_committed_and_off_places_none() -> None:
    """The funding gate end to end: the launcher leaves only after the
    closer's commitment latches -- income alone released the save mid-fight
    and bled a baseline win to defeat (`runs/sweeps/vh-nuke`, 90210)."""
    world = sample(
        CENTRE,
        BUILDER,
        credits=90_000,
        options=(option(214, LAUNCHER_TYPE, key="u_nuke", placed=True),),
        players=(
            player(0, index=0, local=True, hostile=False, income=60, army_value=9_000),
            player(1, index=1, income=60, army_value=1_000),
        ),
    )
    script = tuple(world for _ in range(CLOSE_HOLD))

    held = ScriptedPeer(lines(*script))
    play(AgentChannel(held), (), _CATALOGUE, PLACEMENTS, _PROFILES, len(script), close=3)
    assert [line for line in held.sent if LAUNCHER_TYPE in line] == []

    arming = ScriptedPeer(lines(*script))
    play(AgentChannel(arming), (), _CATALOGUE, PLACEMENTS, _PROFILES, len(script), close=3, nukes=1)
    built = [line for line in arming.sent if '"build"' in line and LAUNCHER_TYPE in line]
    assert len(built) == 1
    assert '"unit_id":214' in built[0]


def test_the_bank_funds_the_launcher_through_the_safe_window_and_doom_holds_it() -> None:
    """The third funding gate, end to end: no dominance anywhere (the rival
    is eight times our army), yet with the bank on the launcher leaves the
    moment the head's window fills reading SAFE -- and with a head that
    reads doom, the same world funds nothing (law eight: no prediction,
    no bank)."""
    world = sample(
        CENTRE,
        BUILDER,
        credits=90_000,
        options=(option(214, LAUNCHER_TYPE, key="u_nuke", placed=True),),
        players=(
            player(0, index=0, local=True, hostile=False, income=60, army_value=900),
            player(1, index=1, income=60, army_value=8_000),
        ),
    )
    script = (world, world, world)

    safe = decode_head_model(
        [
            '{"window": 2, "threshold": 0.5, "intercept": -6.0}',
            '{"name": "credits_last", "mean": 90000.0, "std": 1000.0, "coef": 0.0}',
        ]
    )
    banked = ScriptedPeer(lines(*script))
    play(
        AgentChannel(banked),
        (),
        _CATALOGUE,
        PLACEMENTS,
        _PROFILES,
        len(script),
        nukes=1,
        bank=True,
        gate_model=safe,
    )
    built = [line for line in banked.sent if '"build"' in line and LAUNCHER_TYPE in line]
    assert len(built) == 1

    doom = decode_head_model(
        [
            '{"window": 2, "threshold": 0.5, "intercept": 6.0}',
            '{"name": "credits_last", "mean": 90000.0, "std": 1000.0, "coef": 0.0}',
        ]
    )
    held = ScriptedPeer(lines(*script))
    play(
        AgentChannel(held),
        (),
        _CATALOGUE,
        PLACEMENTS,
        _PROFILES,
        len(script),
        nukes=1,
        bank=True,
        gate_model=doom,
    )
    assert [line for line in held.sent if '"build"' in line and LAUNCHER_TYPE in line] == []


def test_a_standing_launcher_arms_and_fires_over_the_wire() -> None:
    """The full chain in agent format: the priced action leaves as a plain
    ability, the launch as the targeted one, aimed at the hostile
    structure."""
    world = sample(
        CENTRE,
        entity(500, LAUNCHER_TYPE, x=120.0),
        enemy(80, "landFactory", x=900.0, y=450.0),
        credits=20_000,
        options=(
            option(500, "", key="c_buildNuke", index=1, price=11000),
            option(500, "", key="c_launchNuke", index=2, price=0),
        ),
    )
    peer = ScriptedPeer(lines(world))
    play(AgentChannel(peer), (), _CATALOGUE, PLACEMENTS, _PROFILES, 1, nukes=1)
    assert '{"kind":"ability","unit_id":500,"key":"c_buildNuke"}' in peer.sent
    assert (
        '{"kind":"ability_at","unit_id":500,"x":900.0,"y":450.0,"key":"c_launchNuke"}' in peer.sent
    )
