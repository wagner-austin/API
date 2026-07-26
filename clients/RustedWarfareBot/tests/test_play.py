"""The play entry point, driven against a scripted agent.

The catalogue is the real archived ``-printunits`` dump, so the prices these
assert against are the engine's own.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import TracebackType

import pytest
from scripts.play import (
    DEFAULT_PLAN,
    EXIT_BAD_USAGE,
    EXIT_INCOMPLETE,
    EXIT_OK,
    load_catalogue,
    main,
)

from rw_bot.control import _test_hooks

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CATALOGUE_PATH = _PROJECT_ROOT / "wiki" / "sources" / "m0-probe" / "printunits.log"


def _entity_line(frame: int, index: int, unit_id: int, type_name: str) -> str:
    return (
        f'{{"kind":"entity","frame":{frame},"index":{index},"id":{unit_id},'
        f'"type":"{type_name}","class":"units.x","x":100.0,"y":200.0,'
        f'"team":0,"mine":true,"hp":100.0,"max_hp":100.0}}'
    )


def _sample_lines(frame: int, credits: int, *entities: tuple[int, str]) -> list[str]:
    lines = [
        f'{{"kind":"frame","frame":{frame},"clock_ms":{frame * 3},'
        f'"visible":{len(entities)},"credits":{credits}}}'
    ]
    for index, (unit_id, type_name) in enumerate(entities):
        lines.append(_entity_line(frame, index, unit_id, type_name))
    return lines


_BUILDER = (214, "builder")


class _ScriptedPeer:
    """Serves prepared lines and records what was sent back.

    Attributes:
        sent: Every line written, in order.
    """

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.sent: list[str] = []

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
        """Release the connection."""


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
        """Install the stub.

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
        """Restore the original hook.

        Args:
            exc_type: Exception class raised in the block, if any.
            exc: Exception raised in the block, if any.
            traceback: Traceback of the raised exception, if any.
        """
        _test_hooks.connect = self._original


def test_the_real_catalogue_prices_the_whole_plan() -> None:
    """Every planned structure must be priceable, or the run blocks at once."""
    catalogue = load_catalogue(_CATALOGUE_PATH)
    assert [catalogue[name]["price"] for name in DEFAULT_PLAN] == [700, 700, 700]


def test_a_completed_plan_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    built = [(300 + i, "landFactory") for i in range(len(DEFAULT_PLAN))]
    peer = _ScriptedPeer(_sample_lines(1, 9000, _BUILDER, *built))
    with _StubbedConnect(peer):
        assert main(["27200", str(_CATALOGUE_PATH), "5"]) == EXIT_OK
    assert capsys.readouterr().out.splitlines() == [
        "plan: landFactory -> landFactory -> landFactory",
        "  landFactory costs 700",
        "  landFactory costs 700",
        "  landFactory costs 700",
        "outcome        done (all 3 structures built)",
        "completed      3/3",
        "orders sent    0",
        "samples seen   1",
        "frames elapsed 0",
        "credits left   9000",
    ]


def test_an_unfinished_plan_exits_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    peer = _ScriptedPeer(_sample_lines(1, 10, _BUILDER))
    with _StubbedConnect(peer):
        assert main(["27200", str(_CATALOGUE_PATH), "1"]) == EXIT_INCOMPLETE
    assert capsys.readouterr().out.splitlines()[4:] == [
        "outcome        sample_limit (landFactory costs 700, holding 10)",
        "completed      0/3",
        "orders sent    0",
        "samples seen   1",
        "frames elapsed 0",
        "credits left   10",
    ]


def test_the_sample_budget_defaults_when_not_given(
    capsys: pytest.CaptureFixture[str],
) -> None:
    built = [(300 + i, "landFactory") for i in range(len(DEFAULT_PLAN))]
    peer = _ScriptedPeer(_sample_lines(1, 9000, _BUILDER, *built))
    with _StubbedConnect(peer):
        assert main(["27200", str(_CATALOGUE_PATH)]) == EXIT_OK
    assert capsys.readouterr().out.splitlines()[4:] == [
        "outcome        done (all 3 structures built)",
        "completed      3/3",
        "orders sent    0",
        "samples seen   1",
        "frames elapsed 0",
        "credits left   9000",
    ]


@pytest.mark.parametrize("args", [[], ["27200"], ["a", "b", "c", "d"]])
def test_a_bad_argument_count_prints_usage(
    args: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(args) == EXIT_BAD_USAGE
    assert capsys.readouterr().out == "usage: play <port> <catalogue-path> [max-samples]\n"


def test_module_entry_point_exits_with_the_run_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_argv = sys.argv
    already_imported = sys.modules.pop("scripts.play")
    sys.argv = ["play"]
    try:
        with pytest.raises(SystemExit) as caught:
            runpy.run_module("scripts.play", run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.modules["scripts.play"] = already_imported
    assert caught.value.code == EXIT_BAD_USAGE
    assert capsys.readouterr().out == "usage: play <port> <catalogue-path> [max-samples]\n"
