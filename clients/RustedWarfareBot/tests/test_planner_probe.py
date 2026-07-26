"""The end-to-end planner probe, driven against a scripted agent.

The probe's whole point is that its subject is chosen from observed state
rather than from a constant, so the tests that matter are the ones that change
the world and check the choice follows.
"""

from __future__ import annotations

import runpy
import sys
from types import TracebackType

import pytest
from scripts.planner_probe import (
    EXIT_BAD_USAGE,
    EXIT_NO_BUILDER,
    EXIT_OK,
    find_builder,
    main,
)

from rw_bot.control import _test_hooks
from rw_bot.wire.state import Entity, Sample, decode_samples


def _entity(index: int, unit_id: int, type_name: str, x: float, y: float) -> str:
    return (
        f'{{"kind":"entity","frame":1,"index":{index},"id":{unit_id},'
        f'"type":"{type_name}","class":"units.x","x":{x},"y":{y},'
        f'"team":0,"mine":true,"hostile":false,"movement":"LAND","group":1,"hp":100.0,"max_hp":100.0,"complete":true,"queued":0}}'
    )


def _sample_lines(*entities: str) -> list[str]:
    frame = (
        f'{{"kind":"frame","frame":1,"clock_ms":10,'
        f'"visible":{len(entities)},"pools":0,"options":0,"credits":4000}}'
    )
    return [frame, *entities]


def _sample(*entities: str) -> Sample:
    return decode_samples(_sample_lines(*entities))[0]


class _ScriptedPeer:
    """Serves prepared lines and records what was sent back.

    Attributes:
        sent: Every line the probe wrote, in order.
        closed: Whether the probe released the connection.
    """

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.sent: list[str] = []
        self.closed = False

    def send_line(self, line: str) -> None:
        """Record one written line.

        Args:
            line: Line content, without a newline.
        """
        self.sent.append(line)

    def read_line(self) -> str:
        """Serve the next prepared line, or end of stream.

        Returns:
            The next line, or an empty string once exhausted.
        """
        if not self._lines:
            return ""
        return self._lines.pop(0)

    def close(self) -> None:
        """Mark the connection released."""
        self.closed = True


class _StubbedConnect:
    """Binds the connect hook to a scripted peer for one block.

    Attributes:
        peer: The peer every connection returns.
    """

    def __init__(self, peer: _ScriptedPeer) -> None:
        self.peer = peer
        self._original: _test_hooks.ConnectProto = _test_hooks.connect

    def __call__(self, host: str, port: int, timeout_s: float) -> _test_hooks.Connection:
        """Return the scripted peer.

        Args:
            host: Ignored.
            port: Ignored.
            timeout_s: Ignored.

        Returns:
            The scripted peer.
        """
        return self.peer

    def __enter__(self) -> _StubbedConnect:
        """Install this stub as the connect hook.

        Returns:
            This stub.
        """
        self._original = _test_hooks.connect
        _test_hooks.connect = self
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Restore the original connect hook.

        Args:
            exc_type: Exception class raised in the block, if any.
            exc: Exception raised in the block, if any.
            traceback: Traceback of the raised exception, if any.
        """
        _test_hooks.connect = self._original


def test_the_builder_is_found_by_type_not_by_position() -> None:
    """Reordering the roster must not change which unit is chosen."""
    forward = _sample(
        _entity(0, 213, "commandCenter", 0.0, 0.0),
        _entity(1, 214, "builder", 5.0, 6.0),
    )
    reversed_roster = _sample(
        _entity(0, 214, "builder", 5.0, 6.0),
        _entity(1, 213, "commandCenter", 0.0, 0.0),
    )
    expected = Entity(
        index=1,
        unit_id=214,
        type_name="builder",
        class_name="units.x",
        x=5.0,
        y=6.0,
        team=0,
        mine=True,
        hostile=False,
        movement="LAND",
        group=1,
        hp=100.0,
        max_hp=100.0,
        complete=True,
        queued=0,
    )
    assert find_builder(forward) == expected
    # Same unit, now first in the roster: the index differs, the choice must not.
    assert find_builder(reversed_roster) == {**expected, "index": 0}


def test_no_builder_in_the_roster_is_an_answer_not_a_crash() -> None:
    assert find_builder(_sample(_entity(0, 213, "commandCenter", 0.0, 0.0))) is None


def test_the_order_targets_an_offset_from_the_builders_own_position(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The destination is derived from observed state, not from a constant."""
    lines = _sample_lines(_entity(0, 214, "builder", 4250.0, 2610.0))
    peer = _ScriptedPeer(lines + _sample_lines() * 8)
    with _StubbedConnect(peer):
        assert main(["27200"]) == EXIT_OK
    assert peer.sent == [
        '{"kind":"build","unit_id":214,"x":4450.0,"y":2730.0,"type":"landFactory"}'
    ]
    printed = capsys.readouterr().out.splitlines()
    assert printed[:3] == [
        "frame 1 clock 10ms: 1 owned",
        "  id=214 builder at (4250.0, 2610.0)",
        "ordering id=214 to build landFactory at (4450.0, 2730.0)",
    ]


def test_a_builder_somewhere_else_moves_the_destination_with_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    lines = _sample_lines(_entity(0, 99, "builder", 0.0, 0.0))
    peer = _ScriptedPeer(lines + _sample_lines() * 8)
    with _StubbedConnect(peer):
        main(["27200"])
    capsys.readouterr()
    assert peer.sent == ['{"kind":"build","unit_id":99,"x":200.0,"y":120.0,"type":"landFactory"}']


def test_a_roster_with_no_builder_exits_without_ordering(
    capsys: pytest.CaptureFixture[str],
) -> None:
    peer = _ScriptedPeer(_sample_lines(_entity(0, 213, "commandCenter", 1.0, 2.0)))
    with _StubbedConnect(peer):
        assert main(["27200"]) == EXIT_NO_BUILDER
    assert peer.sent == []
    assert peer.closed is True
    assert capsys.readouterr().out.splitlines() == [
        "frame 1 clock 10ms: 1 owned",
        "  id=213 commandCenter at (1.0, 2.0)",
        "no builder in the roster",
    ]


def test_no_arguments_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == EXIT_BAD_USAGE
    assert capsys.readouterr().out == "usage: planner_probe <port>\n"


def test_two_arguments_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["1", "2"]) == EXIT_BAD_USAGE
    assert capsys.readouterr().out == "usage: planner_probe <port>\n"


def test_module_entry_point_exits_with_the_probe_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The probe is run as `python -m scripts.planner_probe`, so the __main__
    # block is a real execution path and is covered by executing it.
    original_argv = sys.argv
    already_imported = sys.modules.pop("scripts.planner_probe")
    sys.argv = ["planner_probe"]
    try:
        with pytest.raises(SystemExit) as caught:
            runpy.run_module("scripts.planner_probe", run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.modules["scripts.planner_probe"] = already_imported
    assert caught.value.code == EXIT_BAD_USAGE
    assert capsys.readouterr().out == "usage: planner_probe <port>\n"
