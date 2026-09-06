"""The dataset export: traces joined with the ledger and the verdict.

The training side splits by match and predicts the verdict, so the rows must
carry the match identity, the label, and the killer attributions the tick
columns cannot -- and a trace without the income pair must be skipped aloud,
because a padded zero income would poison exactly the column the dataset
exists for.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from scripts.export_matches import (
    EXIT_BAD_USAGE,
    EXIT_EMPTY,
    EXIT_OK,
    HEADER,
    main,
    parse_trace,
)

_TICK_HEADER = (
    "   frame  army  credits  enemies  extractors  lost  producers  idle"
    "  orders  refused    worth    rival  income  rival_income       world\n"
)
_LOSS_HEADER = "   frame    unit  type                       x        y  killer\n"


def _tick(frame: int, lost: int = 0, income: int = 54, rival_income: int = 180) -> str:
    return f"{frame} 3 4000 2 5 {lost} 2 1 1 0 3500 9000 {income} {rival_income} 42\n"


def _full_tick(frame: int) -> str:
    """A current-era row: plan through ``rival_army`` after the digest."""
    return _tick(frame).rstrip("\n") + " building 4 0 1 0 WH 12 34 56 8900\n"


def _loss(frame: int, killer: str) -> str:
    return f"{frame} 8 c_tank 900 250 {killer}\n"


def _write_match(
    tmp_path: Path, batch: str, stem: str, trace: str, verdict: str | None
) -> tuple[Path, Path, Path]:
    traces = tmp_path / "traces"
    sweeps = tmp_path / "sweeps"
    jobs = tmp_path / "jobs"
    (traces / batch).mkdir(parents=True, exist_ok=True)
    (sweeps / batch).mkdir(parents=True, exist_ok=True)
    jobs.mkdir(exist_ok=True)
    (traces / batch / f"{stem}.ndjson").write_text(trace, encoding="utf-8")
    if verdict is not None:
        card = f"### {stem}\n{'verdict':<15}{verdict}\n"
        (sweeps / batch / f"{stem}.txt").write_text(card, encoding="utf-8")
    return sweeps, traces, jobs


def test_both_tables_parse_and_headers_are_skipped_by_shape() -> None:
    parsed = parse_trace(
        _TICK_HEADER + _tick(0) + _tick(75, lost=1) + "\n" + _LOSS_HEADER + _loss(75, "c_artillery")
    )
    assert parsed["legacy"] is False
    assert [tick[0] for tick in parsed["ticks"]] == [0, 75]
    assert parsed["ticks"][0][12] == 54
    assert parsed["ticks"][0][13] == 180
    assert parsed["losses"] == ((75, True),)


def test_a_dash_killer_and_a_five_token_row_are_unattributed() -> None:
    """The renderer writes ``-`` for blank, and pre-killer archives have no
    column at all; neither names a killer."""
    parsed = parse_trace(_loss(75, "-") + "150 8 c_tank 900 250\n")
    assert parsed["losses"] == ((75, False), (150, False))


def test_the_pre_income_shape_is_flagged_legacy() -> None:
    thirteen = "0 3 4000 2 5 0 2 1 1 0 3500 9000 42\n"
    parsed = parse_trace(thirteen)
    assert parsed["ticks"] == ()
    assert parsed["legacy"] is True


def test_rows_join_the_verdict_the_ledger_and_the_ticks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One won match: the label rides every row, killed_cum counts only the
    attributed loss, and lost_cum runs over the tick column."""
    trace = (
        _TICK_HEADER
        + _tick(0)
        + _tick(75, lost=2)
        + _tick(150)
        + "\n"
        + _LOSS_HEADER
        + _loss(75, "c_artillery")
        + _loss(75, "-")
    )
    sweeps, traces, jobs = _write_match(
        tmp_path, "night", "flame-close-s777", trace, "won (wiped_out)"
    )
    dest = tmp_path / "dataset"
    assert main(["night"], sweeps=sweeps, traces=traces, dest=dest, jobs=jobs) == EXIT_OK
    lines = (dest / "data.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(HEADER)
    assert len(lines) == 4
    first = lines[1].split(",")
    assert first[:5] == ["night/flame-close-s777", "flame-close", "777", "won", "1"]
    assert first[5:10] == ["0", "3", "4000", "2", "5"]
    by_column = dict(zip(HEADER, lines[2].split(","), strict=True))
    assert by_column["lost"] == "2"
    assert by_column["lost_cum"] == "2"
    assert by_column["killed_cum"] == "1"
    assert by_column["income"] == "54"
    assert by_column["rival_income"] == "180"
    assert dict(zip(HEADER, lines[3].split(","), strict=True))["killed_cum"] == "1"
    assert capsys.readouterr().out == f"wrote 3 rows from 1 matches to {dest / 'data.csv'}\n"


def test_a_lost_match_labels_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sweeps, traces, jobs = _write_match(
        tmp_path, "night", "flame-close-s1", _TICK_HEADER + _tick(0), "defeated"
    )
    dest = tmp_path / "dataset"
    assert main(["night"], sweeps=sweeps, traces=traces, dest=dest, jobs=jobs) == EXIT_OK
    row = (dest / "data.csv").read_text(encoding="utf-8").splitlines()[1].split(",")
    assert row[3:5] == ["defeated", "0"]
    capsys.readouterr()


def test_legacy_cardless_and_empty_traces_are_skipped_aloud(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Three ways a match can fail to export, each named on stdout -- a
    silently shrunk dataset reads as a complete one."""
    sweeps, traces, jobs = _write_match(
        tmp_path, "night", "old-s1", "0 3 4000 2 5 0 2 1 1 0 3500 9000 42\n", "won"
    )
    _write_match(tmp_path, "night", "bare-s2", _TICK_HEADER, "won")
    _write_match(tmp_path, "night", "orphan-s3", _TICK_HEADER + _tick(0), None)
    exit_code = main(["night"], sweeps=sweeps, traces=traces, dest=tmp_path / "d", jobs=jobs)
    assert exit_code == EXIT_EMPTY
    assert capsys.readouterr().out == (
        "skipped night/bare-s2: empty trace\n"
        "skipped night/old-s1: pre-income trace shape\n"
        "skipped night/orphan-s3: no scorecard\n"
        "nothing to export\n"
    )
    assert not (tmp_path / "d").exists()


def test_a_missing_verdict_field_exports_as_a_question_mark(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A card without the field is a fact about the card, carried rather than
    guessed over."""
    sweeps, traces, jobs = _write_match(tmp_path, "night", "odd-s4", _TICK_HEADER + _tick(0), "won")
    card = sweeps / "night" / "odd-s4.txt"
    card.write_text("### odd-s4\n", encoding="utf-8")
    dest = tmp_path / "dataset"
    assert main(["night"], sweeps=sweeps, traces=traces, dest=dest, jobs=jobs) == EXIT_OK
    row = (dest / "data.csv").read_text(encoding="utf-8").splitlines()[1].split(",")
    assert row[3:5] == ["?", "0"]
    capsys.readouterr()


def test_the_difficulty_rides_every_row_when_the_card_states_its_match(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Cards file their setup since 2026-08-06; older cards leave the column
    blank rather than inventing a difficulty."""
    sweeps, traces, jobs = _write_match(
        tmp_path, "night", "flame-close-s777", _TICK_HEADER + _tick(0), "won"
    )
    card = sweeps / "night" / "flame-close-s777.txt"
    stated = "\n".join(
        (
            "### flame-close-s777",
            f"{'match':<15}1 opponent(s) at difficulty 2 (1.8x AI income) on maps/x.tmx",
            f"{'verdict':<15}won (won)",
            "",
        )
    )
    card.write_text(stated, encoding="utf-8")
    dest = tmp_path / "dataset"
    assert main(["night"], sweeps=sweeps, traces=traces, dest=dest, jobs=jobs) == EXIT_OK
    lines = (dest / "data.csv").read_text(encoding="utf-8").splitlines()
    by_column = dict(zip(HEADER, lines[1].split(","), strict=True))
    assert by_column["difficulty"] == "2"
    assert by_column["won"] == "1"
    capsys.readouterr()


def test_full_shape_extras_export_verbatim_and_the_job_file_names_the_doctrine(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A current-era trace carries plan through ``rival_army`` into the CSV
    untouched -- the ``-`` events marker included, because "no events" is a
    measurement -- and the doctrine column joins the committed job file by
    arm and seed."""
    sweeps, traces, jobs = _write_match(
        tmp_path, "night", "flame-close-s777", _TICK_HEADER + _full_tick(0), "won"
    )
    (jobs / "night.txt").write_text(
        "# label | seed | doctrine | samples\n"
        "short|line\n"
        "flame-close|777|doctrines/evolve3-g3m10.doctrine|10000\n",
        encoding="utf-8",
    )
    dest = tmp_path / "dataset"
    assert main(["night"], sweeps=sweeps, traces=traces, dest=dest, jobs=jobs) == EXIT_OK
    by_column = dict(
        zip(
            HEADER,
            (dest / "data.csv").read_text(encoding="utf-8").splitlines()[1].split(","),
            strict=True,
        )
    )
    assert by_column["plan"] == "building"
    assert by_column["workers"] == "4"
    assert by_column["navy_seen"] == "0"
    assert by_column["events"] == "WH"
    assert by_column["eco_covered"] == "12"
    assert by_column["rival_army"] == "8900"
    assert by_column["doctrine"] == "evolve3-g3m10"
    capsys.readouterr()


def test_income_era_rows_leave_the_late_columns_blank_not_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A 15-column trace exports empty strings for every column its era never
    recorded -- a padded zero would claim a worker count nobody measured --
    and a batch with no committed job file carries a blank doctrine."""
    sweeps, traces, jobs = _write_match(
        tmp_path, "night", "flame-close-s777", _TICK_HEADER + _tick(0), "won"
    )
    dest = tmp_path / "dataset"
    assert main(["night"], sweeps=sweeps, traces=traces, dest=dest, jobs=jobs) == EXIT_OK
    by_column = dict(
        zip(
            HEADER,
            (dest / "data.csv").read_text(encoding="utf-8").splitlines()[1].split(","),
            strict=True,
        )
    )
    assert by_column["plan"] == ""
    assert by_column["workers"] == ""
    assert by_column["events"] == ""
    assert by_column["rival_army"] == ""
    assert by_column["doctrine"] == ""
    capsys.readouterr()


def test_no_arguments_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == EXIT_BAD_USAGE
    assert capsys.readouterr().out == "usage: export_matches <batch-name> [<batch-name> ...]\n"


def test_module_entry_point_exits_with_the_run_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_argv = sys.argv
    already_imported = sys.modules.pop("scripts.export_matches")
    sys.argv = ["export_matches"]
    try:
        with pytest.raises(SystemExit) as caught:
            runpy.run_module("scripts.export_matches", run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.modules["scripts.export_matches"] = already_imported
    assert caught.value.code == EXIT_BAD_USAGE
    assert capsys.readouterr().out == "usage: export_matches <batch-name> [<batch-name> ...]\n"
