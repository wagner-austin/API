"""``tankpit-conformance``: argument handling, the printed report, the JSON and the ratchet."""

from __future__ import annotations

import os
import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from platform_core.json_utils import dump_json_str, load_json_str, narrow_json_to_dict

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.validate.conformance_cli import (
    DEFAULT_OUT,
    DEFAULT_TOP,
    EXIT_USAGE,
    format_report,
    main,
    regressions,
)
from tankpit_bot.validate.conformance_types import (
    ConformanceReportDict,
    DivergenceDict,
    GroupTallyDict,
    ReplaySkipReason,
    SessionResultDict,
    SkippedSessionDict,
    decode_report,
    encode_report,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap
from tests.validate.test_conformance import _handmade


@pytest.fixture()
def open_terrain() -> Generator[None, None, None]:
    """Load every field as open ground for the duration of a test."""
    original = _test_hooks.load_terrain_map

    def load_open_terrain(gif_path: Path) -> TerrainMapProtocol:
        """An all-open field."""
        del gif_path
        return InMemoryTerrainMap()

    _test_hooks.load_terrain_map = load_open_terrain
    yield
    _test_hooks.load_terrain_map = original


def _report(groups: list[GroupTallyDict]) -> ConformanceReportDict:
    """A report holding only these groups."""
    return ConformanceReportDict(sessions=[], skipped=[], groups=groups, divergences=[])


def _archive(root: Path) -> None:
    """One replayable capture (one radar tick that diverges) under ``root``."""
    nested = root / "bot"
    nested.mkdir(parents=True)
    (nested / "a.capture_session.json").write_text(_handmade(), encoding="utf-8")


def test_a_group_below_its_baseline_rate_is_a_regression() -> None:
    """Lower rate regresses; equal or higher does not; an uncompared group is not a law broken."""
    baseline = _report(
        [
            GroupTallyDict(commands="radar", ticks=10, matched=8),
            GroupTallyDict(commands="shoot", ticks=10, matched=5),
            GroupTallyDict(commands="move", ticks=4, matched=1),
        ]
    )
    current = _report(
        [
            GroupTallyDict(commands="radar", ticks=10, matched=7),
            GroupTallyDict(commands="shoot", ticks=20, matched=10),
        ]
    )
    assert regressions(current, baseline) == ["radar: 7/10 (70.0%) below baseline 8/10 (80.0%)"]


def test_the_printed_report_names_rates_divergences_and_skips() -> None:
    """Silent shapes read as such; the skip list names reason and detail."""
    report = ConformanceReportDict(
        sessions=[
            SessionResultDict(session="a", field="field01_r.gif", ticks=4, matched=3, unmodelled=2)
        ],
        skipped=[
            SkippedSessionDict(session="b", reason=ReplaySkipReason.NO_ROOM, detail="no join")
        ],
        groups=[GroupTallyDict(commands="shoot", ticks=4, matched=3)],
        divergences=[
            DivergenceDict(
                commands="shoot",
                live=[],
                sim=["53self"],
                count=1,
                example_session="a",
                example_timestamp_ms=99,
            )
        ],
    )
    assert format_report(report, top=5).splitlines() == [
        "conformance: 1 captures replayed, 1 skipped; 3/4 commanded ticks matched (75.0%),"
        " 2 unmodelled",
        "",
        f"{'commands':<40} {'matched':>8} {'ticks':>7} {'rate':>7}",
        f"{'shoot':<40} {3:>8} {4:>7} {'75.0%':>7}",
        "",
        "top 1 divergences:",
        "     1  shoot  LIVE (silent)  SIM 53self  e.g. a @99",
        "",
        "skipped:",
        "  no_room: b (no join)",
    ]


def test_an_empty_report_prints_without_dividing_by_zero() -> None:
    """Nothing compared reads as 0.0%, and no skip section is printed."""
    assert format_report(_report([]), top=5).splitlines() == [
        "conformance: 0 captures replayed, 0 skipped; 0/0 commanded ticks matched (0.0%),"
        " 0 unmodelled",
        "",
        f"{'commands':<40} {'matched':>8} {'ticks':>7} {'rate':>7}",
        "",
        "top 0 divergences:",
    ]


def test_bad_command_lines_are_refused_with_a_code(capsys: pytest.CaptureFixture[str]) -> None:
    """Unknown flags, missing values and a non-count --top exit with the usage code."""
    assert main(["--bogus"]) == EXIT_USAGE
    assert capsys.readouterr().err == "CONFORMANCE_USAGE: unknown argument '--bogus'\n"
    assert main(["--root"]) == EXIT_USAGE
    assert capsys.readouterr().err == "CONFORMANCE_USAGE: --root needs a value\n"
    assert main(["--top", "many"]) == EXIT_USAGE
    assert capsys.readouterr().err == "CONFORMANCE_USAGE: --top needs a count, not 'many'\n"


def test_an_archive_with_nothing_replayable_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The report is still written, and the run says why it failed."""
    out = tmp_path / "out" / "report.json"
    assert main(["--root", str(tmp_path / "empty"), "--out", str(out)]) == 1
    empty = _report([])
    assert capsys.readouterr().out == (
        format_report(empty, DEFAULT_TOP)
        + f"\n\nwrote {out}\n"
        + "CONFORMANCE_EMPTY: no capture under the roots could be replayed\n"
    )
    assert decode_report(narrow_json_to_dict(load_json_str(out.read_text(encoding="utf-8")))) == (
        empty
    )


def _written(out: Path) -> ConformanceReportDict:
    """The report a run wrote."""
    return decode_report(narrow_json_to_dict(load_json_str(out.read_text(encoding="utf-8"))))


def _printed_report(stdout: str) -> str:
    """Stdout from the report's first line on.

    The replay's world seeding logs through the console handler, which
    shares stdout, so the run's own output is everything from its header.

    Raises:
        ValueError: If the report header was never printed.
    """
    return stdout[stdout.index("conformance: ") :]


@pytest.mark.usefixtures("open_terrain")
def test_a_run_writes_its_report_and_passes_its_own_baseline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run is its own baseline; a better baseline makes it a regression."""
    _archive(tmp_path)
    out = tmp_path / "report.json"
    assert main(["--root", str(tmp_path), "--out", str(out), "--top", "3"]) == 0
    report = _written(out)
    assert [(g["commands"], g["ticks"], g["matched"]) for g in report["groups"]] == [
        ("radar", 1, 0)
    ]
    assert _printed_report(capsys.readouterr().out) == (
        format_report(report, 3) + f"\n\nwrote {out}\n"
    )
    assert main(["--root", str(tmp_path), "--out", str(out), "--baseline", str(out)]) == 0
    better = _report([GroupTallyDict(commands="radar", ticks=1, matched=1)])
    capsys.readouterr()
    baseline = tmp_path / "better.json"
    baseline.write_text(dump_json_str(encode_report(better)), encoding="utf-8")
    capsys.readouterr()
    assert main(["--root", str(tmp_path), "--out", str(out), "--baseline", str(baseline)]) == 1
    assert capsys.readouterr().out.splitlines()[-1] == (
        "CONFORMANCE_REGRESSED: radar: 0/1 (0.0%) below baseline 1/1 (100.0%)"
    )


@pytest.mark.usefixtures("open_terrain")
def test_write_baseline_records_the_rates_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The baseline file holds group rates only, and may be read and rewritten in one run."""
    _archive(tmp_path)
    out = tmp_path / "report.json"
    rates = tmp_path / "rates.json"
    args = ["--root", str(tmp_path), "--out", str(out), "--write-baseline", str(rates)]
    assert main(args) == 0
    assert capsys.readouterr().out.splitlines()[-1] == f"wrote baseline {rates}"
    expected = _report(_written(out)["groups"])
    assert _written(rates) == expected
    assert main([*args, "--baseline", str(rates)]) == 0
    assert _written(rates) == expected


@pytest.mark.usefixtures("open_terrain")
def test_main_reads_sys_argv_and_defaults_to_the_spec_archive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no --root the run reads runs/bot and runs/sniff under the working directory."""
    _archive(tmp_path / "runs")
    original_argv = sys.argv
    original_cwd = Path.cwd()
    sys.argv = ["tankpit-conformance"]
    os.chdir(tmp_path)
    rc = main(None)
    os.chdir(original_cwd)
    sys.argv = original_argv
    assert rc == 0
    report = _written(tmp_path / DEFAULT_OUT)
    assert [s["session"] for s in report["sessions"]] == [
        str(Path("runs") / "bot" / "a.capture_session.json")
    ]
    assert _printed_report(capsys.readouterr().out) == (
        format_report(report, DEFAULT_TOP) + f"\n\nwrote {DEFAULT_OUT}\n"
    )
