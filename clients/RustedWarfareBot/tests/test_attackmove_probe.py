"""The attack-move probe, driven against a scripted agent.

The live half of the probe's claim -- that the flag makes a unit engage en
route -- can only be judged against the real engine; what is tested here is
everything else: the subject is chosen from observed state, the destination
overshoots the hostile on the right line, and every way the world can
disappoint the probe exits with its own code.
"""

from __future__ import annotations

import runpy
import sys

import pytest
from scripts.attackmove_probe import (
    EXIT_BAD_USAGE,
    EXIT_NO_PRODUCER,
    EXIT_OK,
    EXIT_TIMEOUT,
    first_hostile,
    main,
    past,
)

from tests.wire_fixtures import ScriptedPeer, StubbedConnect, enemy, entity, lines, sample

_CENTRE = entity(1, "commandCenter", x=1000.0, y=1000.0)
_SCOUT = entity(2, "scout", x=1000.0, y=1000.0)
_RAIDER = enemy(9, "c_tank", x=1400.0, y=1000.0)


def test_the_destination_overshoots_the_hostile_on_the_scouts_line() -> None:
    """Arrival and engagement must not be confusable: the scout is given no
    reason to stop at the hostile unless the flag makes it.
    """
    x, y = past(_SCOUT, _RAIDER)
    assert x == pytest.approx(1700.0)
    assert y == pytest.approx(1000.0)


def test_a_zero_distance_hostile_does_not_divide_by_zero() -> None:
    on_top = enemy(9, "c_tank", x=1000.0, y=1000.0)
    x, y = past(_SCOUT, on_top)
    assert x == pytest.approx(1000.0)
    assert y == pytest.approx(1000.0)


def test_the_first_hostile_is_taken_in_roster_order() -> None:
    world = sample(_CENTRE, _RAIDER, enemy(10, "scout", x=2000.0, y=2000.0))
    chosen = first_hostile(world)
    assert chosen is not None and chosen["unit_id"] == 9


def test_the_probe_orders_produce_then_attack_move(
    capsys: pytest.CaptureFixture[str],
) -> None:
    opening = sample(_CENTRE)
    ready = sample(_CENTRE, _SCOUT, _RAIDER)
    peer = ScriptedPeer(lines(opening, ready, *(ready for _ in range(500))))
    with StubbedConnect(peer):
        assert main(["27200"]) == EXIT_OK
    sent = [line for line in peer.sent if '"kind":"produce"' in line or "attack_move" in line]
    assert sent[0] == '{"kind":"produce","unit_id":1,"type":"scout"}'
    assert sent[1] == '{"kind":"attack_move","unit_id":2,"x":1700.0,"y":1000.0}'
    printed = capsys.readouterr().out
    assert "attack-move scout 2 past hostile 9" in printed


def test_the_observation_series_names_a_dead_scout_and_an_empty_sky(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The series is the ruling's input, so its empty states are named rather
    than blank -- a missing line reads as a probe failure, not a death.
    """
    opening = sample(_CENTRE)
    ready = sample(_CENTRE, _SCOUT, _RAIDER)
    alone = sample(_CENTRE, _SCOUT)
    after = sample(_CENTRE)
    peer = ScriptedPeer(lines(opening, ready, alone, *(after for _ in range(499))))
    with StubbedConnect(peer):
        assert main(["27200"]) == EXIT_OK
    printed = capsys.readouterr().out
    assert "no hostile in sight" in printed
    assert "scout gone" in printed


def test_a_world_without_a_command_center_exits_loudly() -> None:
    peer = ScriptedPeer(lines(sample(_SCOUT)))
    with StubbedConnect(peer):
        assert main(["27200"]) == EXIT_NO_PRODUCER


def test_a_world_that_never_shows_a_hostile_times_out() -> None:
    opening = sample(_CENTRE)
    quiet = sample(_CENTRE, _SCOUT)
    peer = ScriptedPeer(lines(opening, *(quiet for _ in range(1200))))
    with StubbedConnect(peer):
        assert main(["27200"]) == EXIT_TIMEOUT


def test_a_bad_argument_count_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == EXIT_BAD_USAGE
    assert capsys.readouterr().out == "usage: attackmove_probe <port> [catalogue] [type-flags]\n"


def test_module_entry_point_exits_with_the_run_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_argv = sys.argv
    already_imported = sys.modules.pop("scripts.attackmove_probe")
    sys.argv = ["attackmove_probe"]
    try:
        with pytest.raises(SystemExit) as caught:
            runpy.run_module("scripts.attackmove_probe", run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.modules["scripts.attackmove_probe"] = already_imported
    assert caught.value.code == EXIT_BAD_USAGE
    assert capsys.readouterr().out == "usage: attackmove_probe <port> [catalogue] [type-flags]\n"


def test_a_hostile_seen_before_the_scout_exists_keeps_the_wait_alive() -> None:
    """The wait needs both halves; either alone keeps watching."""
    opening = sample(_CENTRE, _RAIDER)
    waiting = sample(_CENTRE, _RAIDER)
    ready = sample(_CENTRE, _SCOUT, _RAIDER)
    peer = ScriptedPeer(lines(opening, waiting, ready, *(ready for _ in range(500))))
    with StubbedConnect(peer):
        assert main(["27200"]) == EXIT_OK
    ordered = [line for line in peer.sent if "attack_move" in line]
    assert ordered == ['{"kind":"attack_move","unit_id":2,"x":1700.0,"y":1000.0}']


def test_the_series_tracks_the_nearest_hostile_not_the_first() -> None:
    """The ruling is about what the scout meets, and what it meets is
    whichever enemy stands closest on the way.
    """
    from scripts.attackmove_probe import series_line

    crowded = sample(
        _CENTRE,
        _SCOUT,
        enemy(9, "c_tank", x=1400.0, y=1000.0),
        enemy(10, "c_tank", x=3000.0, y=3000.0),
    )
    line = series_line(crowded)
    assert line == "frame 1: scout (1000, 1000) hp 100; nearest 9 (c_tank) at 400 hp 100\n"
