"""Tests for the rw_value derivation script.

Drives the real CLI over a small rw_matches-shaped source and checks the
derived corpus row by row — the target arithmetic, the drop list, and the
stdout report.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from scripts.derive_rw_value import build_parser, derive_rw_value, main

_SOURCE_HEADER = (
    "match,arm,seed,verdict,won,frame,army,credits,enemies,extractors,lost,"
    "lost_cum,killed_cum,producers,idle,orders,refused,worth,rival,income,"
    "rival_income,difficulty"
)


def _write_source(path: Path) -> None:
    """Write a two-match rw_matches-shaped CSV.

    Args:
        path: Destination file path.
    """
    rows = [
        _SOURCE_HEADER,
        "m1,armA,1,won,1,0,5,4000,2,0,0,0,0,1,0,0,0,3500,3500,18,32,veryhard",
        "m1,armA,1,won,1,100,9,3800,2,1,0,0,1,2,0,1,0,3900,3600,22,30,veryhard",
        "m1,armA,1,won,1,250,14,3500,2,2,1,1,2,3,0,2,0,4400,3300,30,25,veryhard",
        "m2,armB,2,wiped,0,0,4,4000,2,0,0,0,0,1,0,0,0,3500,3500,18,32,",
        "m2,armB,2,wiped,0,80,6,3900,2,0,1,1,0,1,0,1,1,3600,3900,19,40,",
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


class TestDeriveRwValue:
    """The derivation's arithmetic and drop list."""

    def test_target_counts_down_to_zero_per_match(self, tmp_path: Path) -> None:
        """frames_remaining is each match's own end frame minus frame."""
        source = tmp_path / "data.csv"
        _write_source(source)
        derived = derive_rw_value(source)
        targets = [int(row[-1]) for row in derived["rows"]]
        assert targets == [250, 150, 0, 80, 0]

    def test_outcome_and_identity_columns_never_reach_the_file(self, tmp_path: Path) -> None:
        """won/verdict/arm/seed/difficulty are dropped; the rest survive."""
        source = tmp_path / "data.csv"
        _write_source(source)
        derived = derive_rw_value(source)
        header = derived["header"]
        for dropped in ("won", "verdict", "arm", "seed", "difficulty"):
            assert dropped not in header
        assert header[0] == "match"
        assert header[-1] == "frames_remaining"
        assert len(header) == 18

    def test_unordered_frames_still_find_the_true_end(self, tmp_path: Path) -> None:
        """The end frame is the match's maximum, not its last row."""
        source = tmp_path / "data.csv"
        rows = [
            _SOURCE_HEADER,
            "m1,armA,1,won,1,100,5,4000,2,0,0,0,0,1,0,0,0,3500,3500,18,32,veryhard",
            "m1,armA,1,won,1,200,9,3800,2,1,0,0,1,2,0,1,0,3900,3600,22,30,veryhard",
            "m1,armA,1,won,1,50,4,4100,2,0,0,0,0,1,0,0,0,3400,3400,17,30,veryhard",
        ]
        source.write_text("\n".join(rows) + "\n", encoding="utf-8")
        derived = derive_rw_value(source)
        targets = [int(row[-1]) for row in derived["rows"]]
        assert targets == [100, 0, 150]

    def test_a_missing_column_is_refused_by_name(self, tmp_path: Path) -> None:
        """A source without the income pair cannot silently derive."""
        source = tmp_path / "data.csv"
        source.write_text("match,frame\nm1,0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing required column 'army'"):
            derive_rw_value(source)


class TestMain:
    """The CLI writes the file and reports the shape."""

    def test_writes_and_reports(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """The derived file lands and stdout states rows, matches, target."""
        source = tmp_path / "data.csv"
        _write_source(source)
        out = tmp_path / "rw_value" / "data.csv"
        exit_code = main(["--source", str(source), "--out", str(out)])
        assert exit_code == 0
        assert out.exists()
        captured = capsys.readouterr()
        lines = captured.out.splitlines()
        assert lines[0] == f"rw_value: 5 rows across 2 matches -> {out}"
        assert lines[1] == "  frames_remaining: mean 96.0, max 250 frames"

    def test_parser_requires_both_paths(self) -> None:
        """--source and --out are both mandatory."""
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_module_entry_point_raises_system_exit(self, tmp_path: Path) -> None:
        """Running as ``__main__`` exits through SystemExit(main())."""
        source = tmp_path / "data.csv"
        _write_source(source)
        out = tmp_path / "rw_value" / "data.csv"
        argv = ["derive_rw_value", "--source", str(source), "--out", str(out)]
        saved = sys.argv
        sys.argv = argv
        try:
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_module("scripts.derive_rw_value", run_name="__main__")
            assert excinfo.value.code == 0
        finally:
            sys.argv = saved
