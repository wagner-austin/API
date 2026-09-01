"""The refusal probe, driven against a scripted agent.

The probe's one job is to force the engine's silent placement refusal and
read the report back, so what these tests pin is the aim -- the order goes to
the command centre's own coordinates, the one site the blocked-pair test can
never pass -- and the wait: reading stops the sample the report arrives, and
a chain that never reports is an exit code rather than a hang.
"""

from __future__ import annotations

import runpy
import sys
from types import TracebackType

import pytest
from scripts.refusal_probe import (
    EXIT_BAD_USAGE,
    EXIT_NO_REFUSAL,
    EXIT_NO_UNITS,
    EXIT_OK,
    main,
)

from rw_bot.control import _test_hooks


def _entity(
    index: int, unit_id: int, type_name: str, x: float, y: float, *, mine: bool = True
) -> str:
    return (
        f'{{"kind":"entity","frame":1,"index":{index},"id":{unit_id},'
        f'"type":"{type_name}","class":"units.x","x":{x},"y":{y},'
        f'"team":0,"mine":{str(mine).lower()},"hostile":{str(not mine).lower()},'
        f'"movement":"LAND","group":1,"flying":false,"submerged":false,"touching_water":false,'
        f'"hp":100.0,"max_hp":100.0,"complete":true,"queued":0,"damaged_by":""}}'
    )


def _sample_lines(*entities: str, refusals: tuple[str, ...] = ()) -> list[str]:
    frame = (
        f'{{"kind":"frame","frame":1,"clock_ms":10,'
        f'"visible":{len(entities)},"pools":0,"options":0,"players":0,'
        f'"refused":{len(refusals)},'
        f'"credits":4000,"defeated":false,"wiped":false,"players_left":6}}'
    )
    return [frame, *entities, *refusals]


#: The opponent's base listed FIRST: fog is disabled on some maps, so the
#: enemy's units precede the player's own in a real roster, and picking by
#: type alone aims the probe at units the dispatch refuses to order.
_ROSTER = (
    _entity(0, 22, "commandCenter", 1230.0, 210.0, mine=False),
    _entity(1, 23, "builder", 1300.0, 260.0, mine=False),
    _entity(2, 213, "commandCenter", 500.0, 700.0),
    _entity(3, 214, "builder", 400.0, 650.0),
)

#: The launcher's module contract: port, then the catalogue and type-dump
#: paths it hands every planner module. The probe ignores the paths, so any
#: values satisfy the shape.
_ARGS = ["27200", "unused-catalogue", "unused-type-dump"]

_REFUSED = (
    '{"kind":"refused","frame":1,"index":0,"unit_id":214,"type":"landFactory","x":500.0,"y":700.0}'
)


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


def test_the_order_targets_the_centres_own_footprint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The aim is the whole probe: the one site the engine can never accept."""
    quiet = _sample_lines(*_ROSTER)
    told = _sample_lines(*_ROSTER, refusals=(_REFUSED,))
    peer = _ScriptedPeer(_sample_lines(*_ROSTER) + quiet + told)
    with _StubbedConnect(peer):
        assert main(_ARGS) == EXIT_OK
    assert '{"kind":"build","unit_id":214,"x":500.0,"y":700.0,"type":"landFactory"}' in peer.sent
    out = capsys.readouterr().out.splitlines()
    assert out == [
        "ordering id=214 to build landFactory at (500.0, 700.0) -- the centre's own footprint",
        "frame 1: engine refused landFactory at (500.0, 700.0) for unit 214",
    ]


def test_reading_stops_the_sample_the_report_arrives() -> None:
    """A probe that keeps reading past its answer holds the match hostage."""
    told = _sample_lines(*_ROSTER, refusals=(_REFUSED,))
    leftover = _sample_lines(*_ROSTER)
    peer = _ScriptedPeer(_sample_lines(*_ROSTER) + told + leftover)
    with _StubbedConnect(peer):
        assert main(_ARGS) == EXIT_OK
    # The leftover sample was never read, and the connection is released.
    assert len(peer._lines) == len(leftover)
    assert peer.closed is True


def test_every_read_sample_is_acknowledged() -> None:
    """In lockstep the ack is what releases the simulation."""
    told = _sample_lines(*_ROSTER, refusals=(_REFUSED,))
    peer = _ScriptedPeer(_sample_lines(*_ROSTER) + told)
    with _StubbedConnect(peer):
        main(_ARGS)
    assert peer.sent.count('{"kind":"ack"}') == 2


def test_a_chain_that_never_reports_is_an_exit_code(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Bounded, because a hang looks like a probe still working."""
    quiet = _sample_lines(*_ROSTER)
    peer = _ScriptedPeer(_sample_lines(*_ROSTER) + quiet * 400)
    with _StubbedConnect(peer):
        assert main(_ARGS) == EXIT_NO_REFUSAL
    assert peer.closed is True
    assert capsys.readouterr().out.splitlines()[-1] == "no refusal within 400 samples"


def test_a_roster_without_a_builder_exits_without_ordering(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The opponent's builder is visible and is NOT an answer: only an owned
    # one can take the order.
    peer = _ScriptedPeer(
        _sample_lines(
            _entity(0, 213, "commandCenter", 500.0, 700.0),
            _entity(1, 23, "builder", 1300.0, 260.0, mine=False),
        )
    )
    with _StubbedConnect(peer):
        assert main(_ARGS) == EXIT_NO_UNITS
    assert [line for line in peer.sent if '"kind":"build"' in line] == []
    assert peer.closed is True
    assert capsys.readouterr().out.splitlines()[-1] == (
        "roster lacks a builder or a command centre"
    )


def test_a_roster_without_a_centre_exits_without_ordering() -> None:
    peer = _ScriptedPeer(_sample_lines(_entity(0, 214, "builder", 400.0, 650.0)))
    with _StubbedConnect(peer):
        assert main(_ARGS) == EXIT_NO_UNITS
    assert [line for line in peer.sent if '"kind":"build"' in line] == []


@pytest.mark.parametrize("args", [[], ["1", "2"], ["1", "2", "3", "4"]])
def test_a_bad_argument_count_prints_usage(
    args: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(args) == EXIT_BAD_USAGE
    assert capsys.readouterr().out == (
        "usage: refusal_probe <port> <catalogue-path> <type-dump-path>\n"
    )


def test_module_entry_point_exits_with_the_probe_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The probe runs as `python -m scripts.refusal_probe`, so the __main__
    # block is a real execution path and is covered by executing it.
    original_argv = sys.argv
    already_imported = sys.modules.pop("scripts.refusal_probe")
    sys.argv = ["refusal_probe"]
    try:
        with pytest.raises(SystemExit) as caught:
            runpy.run_module("scripts.refusal_probe", run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.modules["scripts.refusal_probe"] = already_imported
    assert caught.value.code == EXIT_BAD_USAGE
    assert capsys.readouterr().out == (
        "usage: refusal_probe <port> <catalogue-path> <type-dump-path>\n"
    )
