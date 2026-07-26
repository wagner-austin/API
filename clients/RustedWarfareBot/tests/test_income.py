"""The income probe, driven against a scripted agent.

The catalogue and placement rules are the real archived dumps, so the price the
payback figure is divided into is the engine's own 700.

What matters here is that the probe never spends inside a window. The whole
method rests on it: a slope taken across a purchase measures the purchase, so
the loop that stands still has to be tested for standing still, not merely for
arriving at a number.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import TracebackType

import pytest
from scripts.income import (
    EXIT_BAD_USAGE,
    EXIT_INCOMPLETE,
    EXIT_OK,
    format_reading,
    main,
    observe,
    report,
)

from rw_bot.control import _test_hooks
from rw_bot.control.channel import AgentChannel, ChannelError
from rw_bot.mechanics.income import Reading

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CATALOGUE_PATH = _PROJECT_ROOT / "wiki" / "sources" / "m0-probe" / "printunits.log"
_PLACEMENT_PATH = _PROJECT_ROOT / "wiki" / "sources" / "m11-pools" / "type-flags.ndjson"


def _entity_line(frame: int, index: int, unit_id: int, type_name: str, x: float) -> str:
    return (
        f'{{"kind":"entity","frame":{frame},"index":{index},"id":{unit_id},'
        f'"type":"{type_name}","class":"units.x","x":{x},"y":200.0,'
        f'"team":0,"mine":true,"hostile":false,"movement":"LAND","group":1,'
        f'"hp":100.0,"max_hp":100.0,"complete":true,"queued":0}}'
    )


def _world(frame: int, clock_ms: int, credits_held: int, *, extractors: int = 0) -> list[str]:
    """One sample: a base, a builder, one free pool, and N finished extractors.

    The pool sits far from every structure, so it reads as unoccupied however
    many extractors the roster carries -- occupancy is a distance test, and
    parking the extractors on the pool would make the world stop offering it
    halfway through the probe.
    """
    roster = [(213, "commandCenter", 100.0), (214, "builder", 120.0)]
    roster += [(300 + i, "extractorT1", 140.0 + i) for i in range(extractors)]
    options = [(214, "extractorT1")]
    lines = [
        f'{{"kind":"frame","frame":{frame},"clock_ms":{clock_ms},'
        f'"visible":{len(roster)},"pools":1,"options":{len(options)},'
        f'"credits":{credits_held},"defeated":false,"wiped":false,"players_left":6}}'
    ]
    for index, (unit_id, type_name, x) in enumerate(roster):
        lines.append(_entity_line(frame, index, unit_id, type_name, x))
    lines.append(
        f'{{"kind":"pool","frame":{frame},"index":0,"tile_x":50,"tile_y":10,'
        f'"x":1000.0,"y":200.0,"group_land":1}}'
    )
    for index, (unit_id, produces) in enumerate(options):
        lines.append(
            f'{{"kind":"option","frame":{frame},"index":{index},"unit_id":{unit_id},'
            f'"produces":"{produces}","action":1,"placed":true,"available":true}}'
        )
    return lines


#: One stage at two samples a window: an opening sample, a window with no
#: extractor, the two samples a build takes, then a window with one.
_PROBE_RUN = (
    _world(1, 1_000, 5_000)
    + _world(2, 2_000, 5_010)
    + _world(3, 3_000, 5_020)
    + _world(4, 4_000, 5_020)
    + _world(5, 5_000, 4_320, extractors=1)
    + _world(6, 6_000, 4_350, extractors=1)
    + _world(7, 7_000, 4_400, extractors=1)
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


def _args(stages: str, idle: str) -> list[str]:
    return ["27200", str(_CATALOGUE_PATH), str(_PLACEMENT_PATH), stages, idle]


def test_observing_orders_nothing_and_acknowledges_everything() -> None:
    """The measurement is only valid if the window is genuinely idle.

    An order inside a window spends credits, and the slope then measures the
    purchase instead of the income. The acknowledgement is not optional either:
    in lockstep it is what releases the simulation.
    """
    peer = _ScriptedPeer(_world(1, 1_000, 5_000) + _world(2, 2_000, 5_050))
    readings = observe(AgentChannel(peer), 3, 2)
    assert peer.sent == ['{"kind":"ack"}', '{"kind":"ack"}']
    assert [reading["window"] for reading in readings] == [3, 3]
    assert [reading["credits"] for reading in readings] == [5_000, 5_050]
    assert [reading["clock_ms"] for reading in readings] == [1_000, 2_000]


def test_a_reading_renders_as_one_ndjson_record() -> None:
    """Written out so the run is re-analysable without replaying the match."""
    reading = Reading(window=2, extractors=3, clock_ms=45_000, credits=8_100)
    assert format_reading(reading) == (
        '{"window":2,"extractors":3,"clock_ms":45000,"credits":8100}'
    )


def test_the_probe_measures_the_marginal_extractor(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The number the whole exercise exists to produce.

    Ten credits a second with none, fifty with one, so an extractor is worth
    forty a second and pays back its 700 in 17.5 seconds.
    """
    peer = _ScriptedPeer(list(_PROBE_RUN))
    with _StubbedConnect(peer):
        assert main(_args("1", "2")) == EXIT_OK
    out = capsys.readouterr().out.splitlines()
    assert "per extractor  40.00 credits/s" in out
    assert "payback        17.5s at 700 credits" in out


def test_the_readings_behind_the_answer_are_printed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    peer = _ScriptedPeer(list(_PROBE_RUN))
    with _StubbedConnect(peer):
        main(_args("1", "2"))
    out = capsys.readouterr().out.splitlines()
    assert '{"window":0,"extractors":0,"clock_ms":2000,"credits":5010}' in out
    assert '{"window":1,"extractors":1,"clock_ms":7000,"credits":4400}' in out


def test_the_extractor_is_actually_ordered_onto_the_pool() -> None:
    """The stage between two windows is a real build, not a simulated one."""
    peer = _ScriptedPeer(list(_PROBE_RUN))
    with _StubbedConnect(peer):
        main(_args("1", "2"))
    assert '{"kind":"build","unit_id":214,"x":1000.0,"y":200.0,"type":"extractorT1"}' in peer.sent


def test_a_stage_that_cannot_finish_is_reported_and_exits_nonzero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A probe that quietly returned partial windows would be read as an answer."""
    # The extractor never appears and the builder never moves, so the build
    # stage runs its stall window out and gives up.
    peer = _ScriptedPeer(_world(1, 1_000, 5_000) + _world(2, 2_000, 5_010) * 60)
    with _StubbedConnect(peer):
        assert main(_args("1", "1")) == EXIT_INCOMPLETE
    # The whole line, not a substring of it. A substring passes on a report
    # that also said something contradictory alongside, which is exactly the
    # failure a probe reporting partial windows would present as.
    stopped = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("# stage 1 stopped:")
    ]
    assert len(stopped) == 1


def test_one_extractor_count_cannot_be_turned_into_a_slope() -> None:
    """Said out loud rather than reported as a rate of zero."""
    readings = (
        Reading(window=0, extractors=0, clock_ms=0, credits=0),
        Reading(window=0, extractors=0, clock_ms=1_000, credits=27),
    )
    assert report(readings, 700)[-1] == ("not enough distinct extractor counts to measure a slope")


def test_an_extractor_that_earns_nothing_is_reported_as_never() -> None:
    """Guarding the divide, and saying the useful thing when it cannot be done."""
    readings = (
        Reading(window=0, extractors=0, clock_ms=0, credits=0),
        Reading(window=0, extractors=0, clock_ms=1_000, credits=100),
        Reading(window=1, extractors=2, clock_ms=0, credits=0),
        Reading(window=1, extractors=2, clock_ms=1_000, credits=100),
    )
    assert report(readings, 700)[-1] == "payback        never -- it earns nothing measurable"


@pytest.mark.parametrize("args", [[], ["27200"], ["a", "b", "c", "d", "e", "f"]])
def test_a_bad_argument_count_prints_usage(
    args: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(args) == EXIT_BAD_USAGE
    assert capsys.readouterr().out == (
        "usage: income <port> <catalogue-path> <placement-path> [stages] [idle-samples]\n"
    )


def test_the_stage_and_idle_counts_default_when_not_given() -> None:
    """Both are optional, and omitting them must not change the argument shape."""
    peer = _ScriptedPeer(list(_PROBE_RUN))
    # Three arguments is a valid call; it fails on the stream running out
    # rather than on usage, which is what proves the defaults were taken.
    with _StubbedConnect(peer), pytest.raises(ChannelError, match="RW-CHAN"):
        main(["27200", str(_CATALOGUE_PATH), str(_PLACEMENT_PATH)])


def test_module_entry_point_exits_with_the_run_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_argv = sys.argv
    already_imported = sys.modules.pop("scripts.income")
    sys.argv = ["income"]
    try:
        with pytest.raises(SystemExit) as caught:
            runpy.run_module("scripts.income", run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.modules["scripts.income"] = already_imported
    assert caught.value.code == EXIT_BAD_USAGE
    assert capsys.readouterr().out == (
        "usage: income <port> <catalogue-path> <placement-path> [stages] [idle-samples]\n"
    )
