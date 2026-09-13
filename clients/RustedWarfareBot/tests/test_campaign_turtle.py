"""The turtle through the loop: the hold latches at its window and recalls the wave.

The press's opposite, wired the same way: a one-shot read of worth against
the rival's at the turtle window, then the wave controller withholds every
size release and the wave already out is sent home. What is pinned is the
loop's wiring -- the latch reads the scores' own worth pair, the ``W`` code
lands in the trace from the sample it commits, and the recall is three
move orders that a doctrine without the field never sends -- and the
identity: a match the champion is winning at the window never holds
([[policy-situation]], [[very-hard-race]]).
"""

from __future__ import annotations

from pathlib import Path

from rw_bot.control.channel import AgentChannel
from rw_bot.policy.campaign import play
from rw_bot.policy.situation import TURTLE_WINDOW
from rw_bot.wire.state import Sample
from tests.campaign_fixtures import (
    CATALOGUE,
    CENTRE,
    ENEMY,
    PLACEMENTS,
    PROFILES,
    THEM,
    US,
    WAVE,
    ScriptedPeer,
)
from tests.wire_fixtures import entity, lines, player, sample

#: The fixture wave stood forward of the anchor, so a recall is a walk the
#: rally pass has to order: a unit already standing on the post is sent
#: nowhere, and the wiring under test is the order.
FORWARD_WAVE = tuple(entity(tank["unit_id"], "c_tank", x=400.0) for tank in WAVE)
#: A full first wave with a hostile in reach and the rival ahead on worth:
#: 3,500 against 5,700, a ratio of 0.61, under the 85 the arm carries.
BEHIND = sample(CENTRE, *FORWARD_WAVE, ENEMY, players=(US, THEM))
#: The same board with the rival far behind: no read commits.
AHEAD = sample(
    CENTRE,
    *FORWARD_WAVE,
    ENEMY,
    players=(US, player(1, index=1, income=18, army_value=200, building_value=300)),
)


def _play(world: Sample, turtle: int, trace: Path | None = None) -> ScriptedPeer:
    """Play the world through the window and one sample past it."""
    times = TURTLE_WINDOW + 1
    peer = ScriptedPeer(lines(*(world for _ in range(times))))
    play(
        AgentChannel(peer),
        (),
        CATALOGUE,
        PLACEMENTS,
        PROFILES,
        times,
        turtle=turtle,
        trace=trace,
    )
    return peer


def _moves(peer: ScriptedPeer) -> list[str]:
    return [line for line in peer.sent if '"kind":"move"' in line]


def test_the_turtle_latches_at_its_window_and_recalls_the_wave(tmp_path: Path) -> None:
    """The wave attacks on the first observation as it always did; at the
    window the ratio reads 0.61, the hold commits, the three released tanks
    are sent home, and every row after the window carries the W code."""
    trace = tmp_path / "turtle.trace"
    peer = _play(BEHIND, turtle=85, trace=trace)
    assert any('"kind":"attack"' in line for line in peer.sent)
    assert [line.split('"unit_id":')[1].split(",")[0] for line in _moves(peer)] == ["1", "2", "3"]
    rows = trace.read_text(encoding="utf-8").splitlines()
    events = [row.split()[20] for row in rows[1 : TURTLE_WINDOW + 2]]
    assert events[TURTLE_WINDOW - 1] == "-"
    assert events[TURTLE_WINDOW] == "W"


def test_the_champion_and_a_winning_turtle_never_hold() -> None:
    """The field's identity: with the knob at zero, or with the knob armed
    on a board the champion is winning at the window, nobody is recalled."""
    assert _moves(_play(BEHIND, turtle=0)) == []
    assert _moves(_play(AHEAD, turtle=85)) == []
