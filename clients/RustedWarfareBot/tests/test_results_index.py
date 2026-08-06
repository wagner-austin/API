"""The results index: every ad-hoc probe as one queryable table.

The sweep ledger covers ``runs/sweeps``; the campaign's arms are screened as
single ``runs/<name>.out`` probes first, and the night the basics batch ran,
"which arm ever moved the dip" was answered by grepping a hundred scorecards.
The index parses them once and sorts by the figure verdicts turn on.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from scripts.results import (
    EXIT_BAD_USAGE,
    EXIT_OK,
    ProbeRecord,
    SpendRow,
    collect,
    main,
    parse_probe,
)

_BANNER = (
    "==> play (headless match on port 27700, log: runs/tester.log)\n"
    "& scripts/make/play.ps1 -Port 27700 -Seed 777 -Difficulty 3"
    ' -Map "maps/skirmish/[p2]duel_lake.tmx" -PinDelta 3\n'
)


def _card(label: str, value: str) -> str:
    return f"{label:<15}{value}\n"


def _full_report(arm: str = "tester", dip: int = 9100, samples: int = 3215) -> str:
    return (
        _BANNER
        + f"doctrine: {arm}\n"
        + "Plan Total: 5200\n"  # first char not lowercase: not a label
        + "abcdefghijklmno  x\n"  # column 15 blank: not a label either
        + "no\n"  # shorter than the label width
        + _card("verdict", "wiped (wiped)")
        + _card("samples seen", str(samples))
        + _card("best rival", f"3500 -> 160700 (peak 160700, worst dip {dip})")
        + _card("extractors", "0 -> 2")
        + _card("attack orders", "202")
        + _card("engaged gone", "62")
        + _card("spend", "expand:c_turret_t1      asked     2  got     2  spent    1000")
        + _card("spend", "produce:c_tank          asked   564  got    23  spent    8050")
        + "[play] game stopped\n"
    )


def test_a_full_scorecard_is_reduced_to_the_figures_verdicts_turn_on() -> None:
    assert parse_probe(_full_report(), "tester-run") == ProbeRecord(
        file="tester-run",
        arm="tester",
        seed=777,
        difficulty=3,
        map="[p2]duel_lake",
        verdict="wiped",
        samples=3215,
        dip=9100,
        peak=160700,
        rival_end=160700,
        extractors_end=2,
        attacks=202,
        engaged_gone=62,
        spends=(
            SpendRow(channel="expand:c_turret_t1", asked=2, got=2, spent=1000),
            SpendRow(channel="produce:c_tank", asked=564, got=23, spent=8050),
        ),
    )


def test_a_build_log_is_not_a_match() -> None:
    """An agent build or any other ``.out`` never becomes a row."""
    assert parse_probe("==> agent (javac)\nOK 1 target(s)\n", "agent") is None


def test_a_crashed_run_is_a_visible_row_not_a_silent_skip() -> None:
    """Two safe-mode wedges died before their scorecards on 2026-08-01;

    a table that dropped them would hide exactly the runs that need a
    relaunch."""
    assert parse_probe("==> play (headless match)\nchannel open\n", "wedged") == ProbeRecord(
        file="wedged",
        arm="?",
        seed=-1,
        difficulty=-1,
        map="",
        verdict="incomplete",
        samples=0,
        dip=0,
        peak=0,
        rival_end=0,
        extractors_end=0,
        attacks=0,
        engaged_gone=0,
        spends=(),
    )


def test_the_table_reads_needle_movers_first(tmp_path: Path) -> None:
    """Sorted by dip then survival: the question is 'what moved it'."""
    (tmp_path / "low.out").write_text(_full_report("low", dip=2000, samples=9), encoding="utf-8")
    (tmp_path / "high.out").write_text(_full_report("high", dip=9100), encoding="utf-8")
    (tmp_path / "long.out").write_text(_full_report("long", dip=2000, samples=99), encoding="utf-8")
    (tmp_path / "noise.out").write_text("==> agent build\n", encoding="utf-8")
    order = [record["arm"] for record in collect(tmp_path)]
    assert order == ["high", "long", "low"]


def test_main_prints_the_index_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "one.out").write_text(_full_report(), encoding="utf-8")
    assert main([], root=tmp_path) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split("\t")[:6] == ["file", "arm", "seed", "diff", "map", "verdict"]
    assert lines[1].split("\t") == [
        "one",
        "tester",
        "777",
        "3",
        "[p2]duel_lake",
        "wiped",
        "3215",
        "9100",
        "160700",
        "160700",
        "2",
        "202",
        "62",
    ]


def test_main_prints_the_spend_ledger_long_form(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "one.out").write_text(_full_report(), encoding="utf-8")
    assert main(["spends"], root=tmp_path) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "\t".join(("file", "arm", "seed", "channel", "asked", "got", "spent"))
    assert lines[1] == "\t".join(("one", "tester", "777", "expand:c_turret_t1", "2", "2", "1000"))
    assert lines[2] == "\t".join(("one", "tester", "777", "produce:c_tank", "564", "23", "8050"))


def test_any_other_argument_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["dip"]) == EXIT_BAD_USAGE
    assert capsys.readouterr().out == "usage: results [spends]\n"


def test_module_entry_point_exits_with_the_run_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_argv = sys.argv
    already_imported = sys.modules.pop("scripts.results")
    sys.argv = ["results", "nonsense", "extra"]
    try:
        with pytest.raises(SystemExit) as caught:
            runpy.run_module("scripts.results", run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.modules["scripts.results"] = already_imported
    assert caught.value.code == EXIT_BAD_USAGE
    assert capsys.readouterr().out == "usage: results [spends]\n"
