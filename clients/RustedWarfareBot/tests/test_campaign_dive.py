"""The dive through the loop: the party closes on the gun, and only the gun.

The identity pairing the dive was built for: the armed world with an
outranging gun in sight sends a party at it, files the objective on the
report and lands the ``D`` code in the trace row; the same world with the
knob off sends nobody; and the armed world with a tank in sight instead
of a gun sends nobody either -- a zero-artillery seed is the champion's
match bit for bit ([[policy-raid]]).
"""

from __future__ import annotations

from pathlib import Path

from rw_bot.control.channel import AgentChannel
from rw_bot.policy.campaign import play
from rw_bot.wire.state import Entity, Sample
from tests.campaign_fixtures import (
    CATALOGUE,
    CENTRE,
    PLACEMENTS,
    ScriptedPeer,
    unit_stats,
)
from tests.wire_fixtures import enemy, entity, lines, profile, profiles_for, sample

#: The fixture catalogue plus the enemy's gun, which moves (the catalogue's
#: speed is how a piece is told from a fortress).
DIVE_CATALOGUE = {**CATALOGUE, "enemy_arty": unit_stats("enemy_arty")}

#: Profiles over the widened catalogue with the gun at the live game's 290
#: reach -- beyond every fixture gun (110), the standoff the dive closes.
DIVE_PROFILES = {**profiles_for(DIVE_CATALOGUE), "enemy_arty": profile("enemy_arty", 290.0)}

#: Five tanks gathered at the anchor: the opening rung's three plus the
#: party's two, because the draft is arbitrated exactly as the hunt's is.
ARMY: tuple[Entity, ...] = tuple(entity(n, "c_tank") for n in range(1, 6))


def _world(hostile: Entity) -> Sample:
    return sample(CENTRE, *ARMY, hostile)


def _play(
    world: Sample,
    dive: int,
    trace: Path | None = None,
    divemargin: int = 0,
    divecap: int = 0,
    diveblood: int = 0,
    *worlds: Sample,
) -> tuple[ScriptedPeer, int]:
    """Play the world twice (or the given sequence): the dive orders on
    the first observation and its decision code lands in the row the
    second one writes, because the fight runs after the row is cut
    ([[policy-loop]])."""
    played = (world, *worlds) if worlds else (world, world)
    peer = ScriptedPeer(lines(*played))
    report = play(
        AgentChannel(peer),
        (),
        DIVE_CATALOGUE,
        PLACEMENTS,
        DIVE_PROFILES,
        len(played),
        dive=dive,
        divemargin=divemargin,
        divecap=divecap,
        diveblood=diveblood,
        trace=trace,
    )
    return peer, report["dives"]


def test_the_blood_gate_waits_for_a_death_to_the_sighted_gun() -> None:
    """The gun stands from the first observation and the line has the
    leave, but no party goes until a unit has died to it: the sixth tank
    is last damaged by the gun on the first observation and gone on the
    second, so the ledger reads one death and the dive fires there, with
    a party drawn from the five that remain."""
    gun = enemy(9, "enemy_arty", x=400.0)
    doomed = entity(6, "c_tank", damaged_by="enemy_arty")
    before = sample(CENTRE, *ARMY, doomed, gun)
    after = sample(CENTRE, *ARMY, gun)
    peer, dives = _play(before, 2, None, 0, 1, 1, after, after)
    marched = [line for line in peer.sent if "attack_move" in line]
    assert marched == [
        '{"kind":"attack_move","unit_id":1,"x":400.0,"y":0.0}',
        '{"kind":"attack_move","unit_id":2,"x":400.0,"y":0.0}',
    ]
    assert dives == 1
    held, none = _play(before, 2, None, 0, 1, 2, after, after)
    assert [line for line in held.sent if "attack_move" in line] == []
    assert none == 0


def test_a_capped_dive_drafts_against_the_opening_rung() -> None:
    """The cap is the brake, so the gate is the opening rung: five tanks
    clear three plus a party of two and the dive goes, exactly as it does
    uncapped on the first rung, and the cap counts the one party."""
    peer, dives = _play(_world(enemy(9, "enemy_arty", x=400.0)), dive=2, divecap=1)
    assert len([line for line in peer.sent if "attack_move" in line]) == 2
    assert dives == 1


def test_the_margin_is_the_doctrines_and_holds_the_party_home() -> None:
    """The 290 gun against the fixture's 110 line is a 180 standoff: a
    margin of 200 reads it as no standoff at all and raises nobody, and
    a margin of 100 reads it as the dive16 artillery and sends the party."""
    held, dives = _play(_world(enemy(9, "enemy_arty", x=400.0)), dive=2, divemargin=200)
    assert [line for line in held.sent if "attack_move" in line] == []
    assert dives == 0
    sent, dives = _play(_world(enemy(9, "enemy_arty", x=400.0)), dive=2, divemargin=100)
    assert len([line for line in sent.sent if "attack_move" in line]) == 2
    assert dives == 1


def test_the_dive_closes_a_party_on_the_outranging_gun(tmp_path: Path) -> None:
    """The party goes once and holds: the second observation re-sends
    nothing while the objective stands."""
    trace = tmp_path / "dive.trace"
    peer, dives = _play(_world(enemy(9, "enemy_arty", x=400.0)), dive=2, trace=trace)
    marched = [line for line in peer.sent if "attack_move" in line]
    assert marched == [
        '{"kind":"attack_move","unit_id":1,"x":400.0,"y":0.0}',
        '{"kind":"attack_move","unit_id":2,"x":400.0,"y":0.0}',
    ]
    assert dives == 1
    rows = trace.read_text(encoding="utf-8").splitlines()
    assert [row.split()[20] for row in rows[1:3]] == ["-", "D"]


def test_the_knob_off_sends_nobody_at_the_same_gun() -> None:
    peer, dives = _play(_world(enemy(9, "enemy_arty", x=400.0)), dive=0)
    assert [line for line in peer.sent if "attack_move" in line] == []
    assert dives == 0


def test_a_tank_in_sight_is_not_a_gun_and_raises_no_party() -> None:
    """The identity property: the armed doctrine on a seed that never
    fields a piece is the champion's match, order for order."""
    peer, dives = _play(_world(enemy(9, "c_tank", x=400.0)), dive=2)
    assert [line for line in peer.sent if "attack_move" in line] == []
    assert dives == 0
