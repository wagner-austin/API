"""The trace autopsy: a batch of matches reduced to what decided them.

Three throwaway scripts produced the worth-ceiling, the never-above-1.19
ratio, and the expansion-race findings; this pins the kept version of that
reading against hand-built traces whose figures are checked by eye.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from scripts.autopsy import (
    EXIT_BAD_USAGE,
    EXIT_EMPTY,
    EXIT_OK,
    FRAMES_PER_SAMPLE,
    PROBES,
    MatchAutopsy,
    TracePoint,
    autopsy,
    decode_trace,
    main,
)

_HEADER = (
    "   frame  army  credits  enemies  extractors  lost  producers  idle"
    "  orders  refused    worth    rival       world\n"
)


def _row(sample: int, extractors: int, lost: int, worth: int, rival: int) -> str:
    frame = sample * FRAMES_PER_SAMPLE
    return f"{frame} 0 4000 2 {extractors} {lost} 1 1 0 0 {worth} {rival} 12345\n"


def _trace(*points: tuple[int, int, int, int, int]) -> str:
    return _HEADER + "".join(_row(*point) for point in points)


def test_a_trace_decodes_by_shape_and_skips_the_header() -> None:
    points = decode_trace(_trace((0, 0, 0, 3500, 3500), (1500, 6, 0, 24000, 22000)))
    assert points == (
        TracePoint(sample=0, extractors=0, lost=0, worth=3500, rival=3500),
        TracePoint(sample=1500, extractors=6, lost=0, worth=24000, rival=22000),
    )


def test_the_fifteen_column_shape_reads_at_the_same_indices() -> None:
    """The income pair landed between rival and world so that extractors,
    lost, worth and rival kept their positions -- one decoder reads the
    13-column archive and the current shape alike."""
    frame = 20 * FRAMES_PER_SAMPLE
    row = f"{frame} 0 4000 2 6 1 1 1 0 0 24000 22000 54 180 12345\n"
    assert decode_trace(_HEADER + row) == (
        TracePoint(sample=20, extractors=6, lost=1, worth=24000, rival=22000),
    )


def test_an_autopsy_names_the_peak_the_collapse_and_the_race() -> None:
    """A losing shape: worth peaks mid-game against a larger rival, halves
    later, and the race finished at five extractors -- the stall figure the
    solo-24 traces put on every loss."""
    points = decode_trace(
        _trace(
            (0, 0, 0, 3500, 3500),
            (1500, 5, 0, 22000, 28000),
            (2000, 5, 0, 30000, 45000),
            (2500, 5, 4, 25000, 60000),
            (3400, 5, 9, 14000, 90000),
            (4000, 5, 20, 0, 120000),
        )
    )
    assert autopsy(points, "loss") == MatchAutopsy(
        file="loss",
        samples=4000,
        peak_worth=30000,
        peak_sample=2000,
        rival_at_peak=45000,
        halved_sample=3400,
        extractors_at_race=5,
        ratios=(
            round(22000 / 28000, 2),
            round(30000 / 45000, 2),
            round(25000 / 60000, 2),
            round(14000 / 90000, 2),
        ),
    )


def test_a_win_never_halves_and_a_dead_rival_reads_as_zero_ratio() -> None:
    """The winning shape: the ratio explodes as their army dies on the
    line, and a rival at zero cannot be divided by -- the annihilation
    endgame every recorded win shows."""
    points = decode_trace(
        _trace(
            (1500, 6, 0, 25000, 24000),
            (2000, 7, 0, 30000, 20000),
            (2500, 7, 0, 34000, 5000),
            (3400, 7, 0, 36000, 0),
        )
    )
    assert autopsy(points, "win") == MatchAutopsy(
        file="win",
        samples=3400,
        peak_worth=36000,
        peak_sample=3400,
        rival_at_peak=0,
        halved_sample=-1,
        extractors_at_race=6,
        ratios=(round(25000 / 24000, 2), round(30000 / 20000, 2), round(34000 / 5000, 2), 0.0),
    )


def test_an_empty_trace_is_no_row() -> None:
    """A match that died before its first sample belongs to the launch
    log, not the table."""
    assert autopsy(decode_trace(_HEADER), "wedged") is None


def test_main_prints_the_batch_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    batch = tmp_path / "night"
    batch.mkdir()
    (batch / "a.ndjson").write_text(
        _trace((1500, 6, 0, 25000, 24000), (2000, 7, 0, 30000, 20000)), encoding="utf-8"
    )
    (batch / "empty.ndjson").write_text(_HEADER, encoding="utf-8")
    assert main(["night"], root=tmp_path) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split("\t")[:7] == [
        "file",
        "samples",
        "peak_worth",
        "peak_sample",
        "rival_at_peak",
        "halved_sample",
        "extr_at_race",
    ]
    assert lines[0].split("\t")[7:] == [f"ratio_s{probe}" for probe in PROBES]
    assert len(lines) == 2
    assert lines[1].split("\t")[:3] == ["a", "2000", "30000"]


def test_a_batch_with_no_traces_says_so(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "bare").mkdir()
    assert main(["bare"], root=tmp_path) == EXIT_EMPTY
    assert capsys.readouterr().out == "no traces for that batch\n"


def test_any_other_argument_shape_is_a_usage_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([]) == EXIT_BAD_USAGE
    assert capsys.readouterr().out == "usage: autopsy <batch-name>\n"


def test_module_entry_point_exits_with_the_run_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_argv = sys.argv
    already_imported = sys.modules.pop("scripts.autopsy")
    sys.argv = ["autopsy"]
    try:
        with pytest.raises(SystemExit) as caught:
            runpy.run_module("scripts.autopsy", run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.modules["scripts.autopsy"] = already_imported
    assert caught.value.code == EXIT_BAD_USAGE
    assert capsys.readouterr().out == "usage: autopsy <batch-name>\n"
