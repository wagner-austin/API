"""The dense margin: bounded, verdict-ordered, and read from real card shapes.

The bands must never cross -- a margin that could rank a wiped match above
a defeated one would disagree with the verdict it claims to sharpen -- and
the parsing must survive the malformed lines a crashed match leaves behind.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest
from scripts.margin import EXIT_BAD_USAGE, EXIT_OK, main

from rw_bot.harness.margin import (
    batch_margins,
    batch_survivals,
    margin_of,
    pressure_of,
    report,
    scorecard_fields,
)


def test_the_bands_cannot_cross() -> None:
    """The extremes of neighboring verdict bands stay strictly ordered."""
    best_wiped = margin_of("wiped", 1.0, 1000, 1000)
    worst_defeated = margin_of("defeated", 0.0, 0, 1000)
    best_defeated = margin_of("defeated", 1.0, 1000, 1000)
    worst_survived = margin_of("survived", 0.0, 1000, 1000)
    best_survived = margin_of("survived", 1.0, 1000, 1000)
    worst_won = margin_of("won", 0.0, 1000, 1000)
    assert best_wiped == -1.0
    assert worst_defeated == -2.0
    assert best_defeated == 0.0
    assert worst_survived == 1.0
    assert best_survived == 2.0
    assert worst_won == 2.0
    assert best_wiped < worst_survived
    assert best_defeated < worst_survived
    assert best_survived <= worst_won


def test_tempo_rewards_fast_wins_and_long_losses() -> None:
    fast_win = margin_of("won", 0.0, 100, 1000)
    slow_win = margin_of("won", 0.0, 1000, 1000)
    early_wipe = margin_of("wiped", 0.0, 100, 1000)
    late_wipe = margin_of("wiped", 0.0, 1000, 1000)
    assert fast_win == 2.9
    assert slow_win == 2.0
    assert early_wipe == -2.9
    assert late_wipe == -2.0


def test_an_unknown_verdict_is_not_a_measurement() -> None:
    assert margin_of("", 0.5, 100, 1000) is None
    assert margin_of("running", 0.5, 100, 1000) is None


def test_pressure_reads_the_card_line_and_survives_malformed_ones() -> None:
    assert pressure_of("3500 -> 139900 (peak 139900, worst dip 11600)") == 11600 / 139900
    assert pressure_of("3500 -> 0 (peak 0, worst dip 0)") == 0.0
    assert pressure_of("none") == 0.0
    assert pressure_of("3500 -> 139900 (peak x, worst dip y)") == 0.0


def test_a_batch_is_scored_and_paired_from_its_cards(tmp_path: Path) -> None:
    """Two arms on one seed pair by margin, and the report carries both
    the margin delta and the win delta it sharpens."""
    card = (
        "### {name}\n"
        "verdict        {verdict} ({verdict})\n"
        "best rival     3500 -> 100 (peak 10000, worst dip 5000)\n"
        "samples seen   {samples}\n"
    )
    (tmp_path / "control-s7.txt").write_text(
        card.format(name="control-s7", verdict="won", samples=1000), encoding="utf-8"
    )
    (tmp_path / "arm-s7.txt").write_text(
        card.format(name="arm-s7", verdict="wiped", samples=500), encoding="utf-8"
    )
    margins = batch_margins(tmp_path)
    assert margins == {"control": {7: 2.5}, "arm": {7: -2.0}}
    lines = report("demo", margins)
    assert lines == (
        "## demo",
        "arm          n=  1  mean margin -2.000  wins 0/1",
        "control      n=  1  mean margin +2.500  wins 1/1",
        "paired control - arm: n=1  margin delta +4.500 (sd 0.000)  win delta +1",
    )


def test_fields_are_read_by_the_sweeps_own_shape() -> None:
    fields = scorecard_fields("verdict        won (won)\nsamples seen   123\nBadLine\n")
    assert fields == {"verdict": "won (won)", "samples seen": "123"}


def test_a_batch_reads_its_survivals_off_the_same_cards(tmp_path: Path) -> None:
    """The survival fitness's figure: samples stood, per arm and seed --
    losses included, because at Impossible losses are the whole field
    and standing time is the gradient. A card with no readable samples
    line is not a measurement."""
    card = "### {name}\nverdict        wiped (wiped)\nsamples seen   {samples}\n"
    (tmp_path / "control-s7.txt").write_text(
        card.format(name="control-s7", samples=1000), encoding="utf-8"
    )
    (tmp_path / "arm-s7.txt").write_text(card.format(name="arm-s7", samples=9000), encoding="utf-8")
    (tmp_path / "arm-s8.txt").write_text(
        "### arm-s8\nverdict        wiped (wiped)\n", encoding="utf-8"
    )
    assert batch_survivals(tmp_path) == {"control": {7: 1000.0}, "arm": {7: 9000.0}}


def test_stray_files_and_unfinished_cards_are_not_measurements(tmp_path: Path) -> None:
    """A notes file, a non-numeric seed and a running verdict all stay out."""
    (tmp_path / "notes.txt").write_text("verdict        won (won)\n", encoding="utf-8")
    (tmp_path / "arm-sX.txt").write_text("verdict        won (won)\n", encoding="utf-8")
    (tmp_path / "arm-s9.txt").write_text(
        "verdict        running (running)\nsamples seen   10\n", encoding="utf-8"
    )
    assert batch_margins(tmp_path) == {}


def test_arms_without_shared_seeds_pair_nothing(tmp_path: Path) -> None:
    (tmp_path / "a-s1.txt").write_text(
        "verdict        won (won)\nsamples seen   10\n", encoding="utf-8"
    )
    (tmp_path / "b-s2.txt").write_text(
        "verdict        won (won)\nsamples seen   10\n", encoding="utf-8"
    )
    lines = report("demo", batch_margins(tmp_path))
    assert not any(line.startswith("paired") for line in lines)


def test_main_prints_usage_and_reports_a_batch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([]) == EXIT_BAD_USAGE
    assert capsys.readouterr().out == "usage: margin <batch> [batch...]\n"
    batch = tmp_path / "demo"
    batch.mkdir()
    (batch / "control-s7.txt").write_text(
        "verdict        won (won)\nsamples seen   10\n", encoding="utf-8"
    )
    assert main(["demo"], root=tmp_path) == EXIT_OK
    out = capsys.readouterr().out
    assert out == ("## demo\ncontrol      n=  1  mean margin +2.000  wins 1/1\n")


def test_the_module_guard_runs_main() -> None:
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("scripts.margin", run_name="__main__")
    assert caught.value.code == EXIT_BAD_USAGE
