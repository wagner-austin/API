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
    DEFAULT_GOALS,
    EXIT_BAD_USAGE,
    EXIT_INCOMPLETE,
    EXIT_OK,
    load_catalogue,
    load_placements,
    main,
)

from rw_bot.control import _test_hooks

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CATALOGUE_PATH = _PROJECT_ROOT / "wiki" / "sources" / "m0-probe" / "printunits.log"
_PLACEMENT_PATH = _PROJECT_ROOT / "wiki" / "sources" / "m11-pools" / "type-flags.ndjson"


def _entity_line(frame: int, index: int, unit_id: int, type_name: str) -> str:
    return (
        f'{{"kind":"entity","frame":{frame},"index":{index},"id":{unit_id},'
        f'"type":"{type_name}","class":"units.x","x":100.0,"y":200.0,'
        f'"team":0,"mine":true,"hostile":false,"hp":100.0,"max_hp":100.0,"complete":true,"queued":0}}'
    )


def _pool_line(frame: int, index: int, tile_x: int, tile_y: int) -> str:
    return (
        f'{{"kind":"pool","frame":{frame},"index":{index},'
        f'"tile_x":{tile_x},"tile_y":{tile_y},'
        f'"x":{tile_x * 20 + 10}.0,"y":{tile_y * 20 + 10}.0}}'
    )


#: What the Builder offers by default here -- the plan's own types, which the
#: live capture confirms unit 214 reports.
_BUILDER_OFFERS = ("extractorT1", "landFactory", "c_tank")


def _option_line(frame: int, index: int, unit_id: int, produces: str) -> str:
    return (
        f'{{"kind":"option","frame":{frame},"index":{index},"unit_id":{unit_id},'
        f'"produces":"{produces}","action":1,"placed":true,"available":true}}'
    )


def _sample_lines(
    frame: int,
    credits: int,
    *entities: tuple[int, str],
    pools: tuple[tuple[int, int], ...] = (),
    options: tuple[tuple[int, str], ...] | None = None,
) -> list[str]:
    if options is None:
        options = tuple((214, name) for name in _BUILDER_OFFERS)
    lines = [
        f'{{"kind":"frame","frame":{frame},"clock_ms":{frame * 3},'
        f'"visible":{len(entities)},"pools":{len(pools)},'
        f'"options":{len(options)},"credits":{credits}}}'
    ]
    for index, (unit_id, type_name) in enumerate(entities):
        lines.append(_entity_line(frame, index, unit_id, type_name))
    for index, (tile_x, tile_y) in enumerate(pools):
        lines.append(_pool_line(frame, index, tile_x, tile_y))
    for index, (unit_id, produces) in enumerate(options):
        lines.append(_option_line(frame, index, unit_id, produces))
    return lines


_BUILDER = (214, "builder")

#: What DEFAULT_GOALS expands to against the real tree from the opening roster.
#: Stated here so a change to either the goals or the tree fails loudly rather
#: than quietly changing what these tests drive.
_EXPANDED = (
    "extractorT1",
    "extractorT1",
    "extractorT1",
    "landFactory",
    "c_tank",
    "c_tank",
)


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


def test_the_real_catalogue_prices_every_goal() -> None:
    """A goal the catalogue cannot price blocks the run at once."""
    catalogue = load_catalogue(_CATALOGUE_PATH)
    assert [catalogue[name]["price"] for name in DEFAULT_GOALS] == [700, 700, 700, 350, 350]


def test_the_real_dump_rules_every_goal() -> None:
    """A goal the placement dump does not cover blocks the run at once."""
    placements = load_placements(_PLACEMENT_PATH)
    assert [placements[name]["needs_pool"] for name in DEFAULT_GOALS] == [
        True,
        True,
        True,
        False,
        False,
    ]


def test_the_goals_open_with_the_structures_that_pay_for_the_rest() -> None:
    """An extractor generates credits; everything else spends them."""
    assert DEFAULT_GOALS[0] == "extractorT1"


def test_the_goals_name_no_factory_but_the_plan_has_one() -> None:
    """The distinction expansion exists to make.

    A tank is asked for; the Land Factory that makes one is not, and writing it
    out by hand is exactly what the build tree exists to stop. If this ever
    passes trivially -- because someone put the factory back in the goals --
    the expansion is no longer being exercised by a real run.
    """
    assert "landFactory" not in DEFAULT_GOALS
    assert "c_tank" in DEFAULT_GOALS


def test_a_completed_plan_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    built = [(300 + i, name) for i, name in enumerate(_EXPANDED)]
    peer = _ScriptedPeer(_sample_lines(1, 9000, _BUILDER, *built) * 2)
    with _StubbedConnect(peer):
        assert main(["27200", str(_CATALOGUE_PATH), str(_PLACEMENT_PATH), "5"]) == EXIT_OK
    # The world already holds a finished Land Factory, so expansion inserts
    # nothing -- the goals are reachable as written. That is the same rule the
    # insertion cases exercise, seen from the other side.
    assert capsys.readouterr().out.splitlines() == [
        "goals: extractorT1 -> extractorT1 -> extractorT1 -> c_tank -> c_tank",
        "plan:  extractorT1 -> extractorT1 -> extractorT1 -> c_tank -> c_tank",
        "  extractorT1 costs 700, goes on a resource pool",
        "  extractorT1 costs 700, goes on a resource pool",
        "  extractorT1 costs 700, goes on a resource pool",
        "  c_tank costs 350, goes on the ring",
        "  c_tank costs 350, goes on the ring",
        "outcome        done (all 5 plan entries satisfied)",
        "completed      5/5",
        "orders sent    0",
        "samples seen   1",
        "frames elapsed 0",
        "credits left   9000",
    ]


def test_an_unfinished_plan_exits_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    peer = _ScriptedPeer(_sample_lines(1, 10, _BUILDER) * 2)
    with _StubbedConnect(peer):
        assert main(["27200", str(_CATALOGUE_PATH), str(_PLACEMENT_PATH), "1"]) == EXIT_INCOMPLETE
    assert capsys.readouterr().out.splitlines()[8:] == [
        "outcome        sample_limit (extractorT1 needs a resource pool and none is visible yet)",
        "completed      0/6",
        "orders sent    0",
        "samples seen   1",
        "frames elapsed 0",
        "credits left   10",
    ]


def test_the_sample_budget_defaults_when_not_given(
    capsys: pytest.CaptureFixture[str],
) -> None:
    built = [(300 + i, name) for i, name in enumerate(_EXPANDED)]
    peer = _ScriptedPeer(_sample_lines(1, 9000, _BUILDER, *built) * 2)
    with _StubbedConnect(peer):
        assert main(["27200", str(_CATALOGUE_PATH), str(_PLACEMENT_PATH)]) == EXIT_OK
    assert capsys.readouterr().out.splitlines()[7:] == [
        "outcome        done (all 5 plan entries satisfied)",
        "completed      5/5",
        "orders sent    0",
        "samples seen   1",
        "frames elapsed 0",
        "credits left   9000",
    ]


@pytest.mark.parametrize("args", [[], ["27200"], ["a", "b", "c", "d", "e"]])
def test_a_bad_argument_count_prints_usage(
    args: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(args) == EXIT_BAD_USAGE
    assert capsys.readouterr().out == (
        "usage: play <port> <catalogue-path> <placement-path> [max-samples]\n"
    )


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
    assert capsys.readouterr().out == (
        "usage: play <port> <catalogue-path> <placement-path> [max-samples]\n"
    )
