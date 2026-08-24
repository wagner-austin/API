"""Tests for the voc_match_quality corpus builder.

Drives the real CLI over a small synthetic workbook shaped exactly like
``Aggregated_Summarized_Output.xlsx`` — ten site sheets with the
aggregated peak-table headers — and checks the chromatogram-context
arithmetic, the drop rules, the refusals, and the written manifest.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict, narrow_json_to_str
from scripts.build_voc_corpus import (
    OUTPUT_HEADER,
    SITE_SHEETS,
    build_corpus,
    build_parser,
    main,
    parse_sheet,
)

from ..datasets.test_xlsx_reader import sheet_of_inline_rows, write_workbook

_HEADER = [
    "DataFolderName",
    "DateRun",
    "CartridgeNum",
    "Species",
    "RetentionTime",
    "Match1",
    "Match1.Quality",
    "Match2",
    "Match2.Quality",
    "Match3",
    "Match3.Quality",
    "Comments",
    "Compound",
    "Class",
    "MatchScore",
]


def _peak(run: str, species: str, rt: str, quality: str) -> list[str]:
    """Render one peak row in the aggregated sheet's column layout.

    Args:
        run: The ``DataFolderName``.
        species: The plant species code.
        rt: The retention time, verbatim.
        quality: The ``Match1.Quality``, verbatim.

    Returns:
        The 15-column row.
    """
    return [run, "2025-05-19", "406099", species, rt, "Hit", quality] + [""] * 7 + [quality]


def write_voc_workbook(path: Path) -> None:
    """Write the happy-path synthetic workbook.

    The Angelo sheet carries the edge cases; the other nine site sheets
    carry one plain kept row each (species ``s``, rt ``1.0``, quality
    ``50``). Angelo's runs:

    - run A: three peaks (rts 1.0 / 1.05 / 5.0); the middle one has no
      quality (dropped, but still crowds its neighbours' context).
    - run B: one peak with no species (the misfiled-cartridge drop).
    - run C: one peak with no retention time (unplaceable drop).
    - run D: one peak whose quality 994 is outside NIST's 1-99 scale.
    - run E: two peaks at the same retention time (rank ties break by
      file order; the second's gap to the previous peak is zero).

    An all-empty spreadsheet row is structural padding, skipped.

    Args:
        path: Destination workbook path.
    """
    angelo = [
        _HEADER,
        _peak("runA", "arto", "1.0", "90"),
        _peak("runA", "arto", "1.05", ""),
        _peak("runA", "arto", "5.0", "80"),
        ["" for _ in _HEADER],
        _peak("runB", "", "2.0", "70"),
        _peak("runC", "arto", "", "70"),
        _peak("runD", "arto", "3.0", "994"),
        _peak("runE", "quag", "2.0", "60"),
        _peak("runE", "quag", "2.0", "61"),
    ]
    sheets: dict[str, str] = {}
    for site in SITE_SHEETS:
        if site == "Angelo":
            sheets[site] = sheet_of_inline_rows(angelo)
        else:
            sheets[site] = sheet_of_inline_rows([_HEADER, _peak("run1", "s", "1.0", "50")])
    write_workbook(path, sheets)


class TestParseSheet:
    """Header resolution and structural refusals."""

    def test_rows_read_verbatim_and_padding_skipped(self, tmp_path: Path) -> None:
        """Peak rows land verbatim; the all-empty row does not."""
        path = tmp_path / "book.xlsx"
        write_voc_workbook(path)
        peaks = parse_sheet(path, "Angelo")
        assert len(peaks) == 8
        assert peaks[0] == {
            "run": "runA",
            "species": "arto",
            "rt_text": "1.0",
            "quality_text": "90",
        }

    def test_missing_header_is_refused_by_name(self, tmp_path: Path) -> None:
        """A sheet without the quality column cannot silently build."""
        path = tmp_path / "book.xlsx"
        write_workbook(path, {"Angelo": sheet_of_inline_rows([["DataFolderName", "Species"]])})
        with pytest.raises(
            ValueError, match="sheet 'Angelo' is missing required header 'RetentionTime'"
        ):
            parse_sheet(path, "Angelo")

    def test_empty_sheet_is_refused(self, tmp_path: Path) -> None:
        """A sheet with no rows at all is a defect."""
        path = tmp_path / "book.xlsx"
        write_workbook(path, {"Angelo": sheet_of_inline_rows([])})
        with pytest.raises(ValueError, match="sheet 'Angelo' is empty"):
            parse_sheet(path, "Angelo")


class TestBuildCorpus:
    """The context arithmetic and the drop rules."""

    def test_rows_drops_and_target_mean(self, tmp_path: Path) -> None:
        """Thirteen rows survive; every drop rule fires exactly once."""
        path = tmp_path / "book.xlsx"
        write_voc_workbook(path)
        result = build_corpus(path)
        assert result["header"] == OUTPUT_HEADER
        assert len(result["rows"]) == 13
        assert result["n_sites"] == 10
        assert result["n_runs"] == 5 + 9
        assert result["drops"] == {
            "no_rt": 1,
            "no_species": 1,
            "no_quality": 1,
            "quality_range": 1,
        }
        assert result["target_mean"] == pytest.approx((90 + 80 + 60 + 61 + 50 * 9) / 13)

    def test_chromatogram_context_arithmetic(self, tmp_path: Path) -> None:
        """Ranks, gaps and crowding counts come out exactly."""
        path = tmp_path / "book.xlsx"
        write_voc_workbook(path)
        rows = build_corpus(path)["rows"]
        angelo = [r for r in rows if r[0] == "Angelo"]
        # run A, rt 1.0: first of three placeable peaks — the dropped
        # no-quality peak at 1.05 still counts as a neighbour.
        assert angelo[0] == [
            "Angelo",
            "arto",
            "1.0",
            "3",
            "0.166667",
            "1.000000",
            "1",
            "1",
            "4.000000",
            "90",
        ]
        # run A, rt 5.0: last of three, 3.95 behind the middle peak.
        assert angelo[1] == [
            "Angelo",
            "arto",
            "5.0",
            "3",
            "0.833333",
            "3.950000",
            "0",
            "0",
            "4.000000",
            "80",
        ]
        # run E: an exact retention-time tie — ranks break by file
        # order, the second peak's gap is zero, each crowds the other.
        assert angelo[2] == [
            "Angelo",
            "quag",
            "2.0",
            "2",
            "0.250000",
            "2.000000",
            "1",
            "1",
            "0.000000",
            "60",
        ]
        assert angelo[3] == [
            "Angelo",
            "quag",
            "2.0",
            "2",
            "0.750000",
            "0.000000",
            "1",
            "1",
            "0.000000",
            "61",
        ]

    def test_empty_corpus_is_refused(self, tmp_path: Path) -> None:
        """If every row drops, the builder refuses rather than writing."""
        path = tmp_path / "book.xlsx"
        sheets = {
            site: sheet_of_inline_rows([_HEADER, _peak("run1", "", "1.0", "50")])
            for site in SITE_SHEETS
        }
        write_workbook(path, sheets)
        with pytest.raises(ValueError, match="no rows survived"):
            build_corpus(path)


class TestMain:
    """The CLI writes data.csv plus the pinned manifest and reports."""

    def test_writes_reports_and_pins_source(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """data.csv and MANIFEST.json land; stdout states rows and drops."""
        workbook = tmp_path / "book.xlsx"
        write_voc_workbook(workbook)
        out = tmp_path / "voc_match_quality" / "data.csv"
        exit_code = main(["--workbook", str(workbook), "--out", str(out)])
        assert exit_code == 0
        lines = out.read_text(encoding="utf-8").splitlines()
        assert lines[0] == ",".join(OUTPUT_HEADER)
        assert len(lines) == 14

        manifest = narrow_json_to_dict(
            load_json_str((out.parent / "MANIFEST.json").read_text(encoding="utf-8"))
        )
        assert narrow_json_to_dict(manifest["corpus"]) == {
            "rows": 13,
            "sites": 10,
            "runs": 14,
            "dropped_no_rt": 1,
            "dropped_no_species": 1,
            "dropped_no_quality": 1,
            "dropped_quality_out_of_range": 1,
            "target_mean": round((90 + 80 + 60 + 61 + 50 * 9) / 13, 6),
        }
        pin = narrow_json_to_dict(narrow_json_to_dict(manifest["sources"])["workbook"])
        assert narrow_json_to_str(pin["file_name"]) == "book.xlsx"
        assert len(narrow_json_to_str(pin["sha256"])) == 64

        captured = capsys.readouterr()
        out_lines = captured.out.splitlines()
        assert out_lines[0] == (
            f"voc_match_quality: 13 rows across 10 sites (14 chromatograms) -> {out}"
        )
        assert out_lines[1] == (
            "  dropped: 1 no-rt, 1 no-species, 1 no-quality, 1 quality-out-of-range"
        )
        assert out_lines[2] == "  match1_quality: mean 57.0000"

    def test_parser_requires_both_paths(self) -> None:
        """--workbook and --out are both mandatory."""
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_module_entry_point_raises_system_exit(self, tmp_path: Path) -> None:
        """Running as ``__main__`` exits through SystemExit(main())."""
        workbook = tmp_path / "book.xlsx"
        write_voc_workbook(workbook)
        out = tmp_path / "voc_match_quality" / "data.csv"
        argv = ["build_voc_corpus", "--workbook", str(workbook), "--out", str(out)]
        saved = sys.argv
        sys.argv = argv
        try:
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_module("scripts.build_voc_corpus", run_name="__main__")
            assert excinfo.value.code == 0
        finally:
            sys.argv = saved
