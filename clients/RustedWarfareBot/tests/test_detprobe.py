"""The determinism microscope, driven against a scripted agent.

What the campaign leaned on is pinned here: the dump orders entities by
engine id (the one cross-run-stable ordering, so two dumps diff cleanly),
and the ``order`` variant sends exactly one fixed move at sample zero --
the smallest possible dose of the command path.
"""

from __future__ import annotations

import runpy
import sys

import pytest
from scripts.detprobe import EXIT_BAD_USAGE, EXIT_OK, main

from rw_bot.wire.command import encode_ack, encode_move, move_order
from tests.wire_fixtures import ScriptedPeer, StubbedConnect, entity, repeated, sample

_WORLD = sample(
    # Out of id order on purpose: the dump must sort by id, not roster slot.
    entity(30, "c_tank", x=5.0, y=6.5, hp=40.0, complete=False),
    entity(24, "builder", x=992.0, y=2070.0, hp=170.0),
)

_DUMP = [
    "#24 builder (992.000,2070.000) hp=170.000 mine=1 complete=1",
    "#30 c_tank (5.000,6.500) hp=40.000 mine=1 complete=0",
]

#: The prefix every dump line carries: sample index, frame, engine clock. The
#: clock rides along because it is the integral of the engine's step size --
#: two runs whose clocks disagree at one frame have already diverged, whatever
#: the positions say ([[policy-determinism]]).
_HEAD = "f1 c0"


@pytest.mark.parametrize("args", [[], ["1", "2"], ["1", "2", "3", "4", "5", "6"]])
def test_a_bad_argument_count_prints_usage(
    args: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(args) == EXIT_BAD_USAGE
    assert capsys.readouterr().out == (
        "usage: detprobe <port> <catalogue> <type-dump> [samples] [order]\n"
    )


def test_the_dump_prints_every_entity_by_id_for_two_samples_by_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    peer = ScriptedPeer(repeated(_WORLD, 2))
    with StubbedConnect(peer):
        assert main(["27200", "catalogue", "type-dump"]) == EXIT_OK
    assert capsys.readouterr().out.splitlines() == [
        *(f"S0 {_HEAD} {line}" for line in _DUMP),
        *(f"S1 {_HEAD} {line}" for line in _DUMP),
    ]
    # Observation only: each sample acknowledged, nothing else sent.
    assert peer.sent == [encode_ack(), encode_ack()]


def test_the_sample_budget_bounds_the_dump(capsys: pytest.CaptureFixture[str]) -> None:
    peer = ScriptedPeer(repeated(_WORLD, 2))
    with StubbedConnect(peer):
        assert main(["27200", "catalogue", "type-dump", "1"]) == EXIT_OK
    assert capsys.readouterr().out.splitlines() == [f"S0 {_HEAD} {line}" for line in _DUMP]


def test_the_order_variant_moves_the_lowest_id_builder_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One fixed order at sample zero, and never again.

    Two owned builders, and the lower id gets the move -- id is the choice a
    replica reproduces. The second sample must pass without another order,
    because a second dose would smear the fork this probe exists to isolate.
    """
    world = sample(
        entity(40, "builder", x=10.0, y=20.0, hp=170.0),
        entity(30, "c_tank", x=5.0, y=6.5, hp=40.0, complete=False),
        entity(24, "builder", x=992.0, y=2070.0, hp=170.0),
    )
    peer = ScriptedPeer(repeated(world, 2))
    with StubbedConnect(peer):
        assert main(["27200", "catalogue", "type-dump", "2", "order"]) == EXIT_OK
    printed = capsys.readouterr().out.splitlines()
    assert printed[3] == "ordered #24 +100x"
    assert printed.count("ordered #24 +100x") == 1
    assert peer.sent == [
        encode_move(move_order(unit_id=24, x=1092.0, y=2070.0)),
        encode_ack(),
        encode_ack(),
    ]


def test_module_entry_point_exits_with_the_run_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_argv = sys.argv
    already_imported = sys.modules.pop("scripts.detprobe")
    sys.argv = ["detprobe"]
    try:
        with pytest.raises(SystemExit) as caught:
            runpy.run_module("scripts.detprobe", run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.modules["scripts.detprobe"] = already_imported
    assert caught.value.code == EXIT_BAD_USAGE
    assert capsys.readouterr().out == (
        "usage: detprobe <port> <catalogue> <type-dump> [samples] [order]\n"
    )
