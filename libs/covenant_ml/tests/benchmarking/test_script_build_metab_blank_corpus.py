"""Tests for the metab_blank corpus builder.

Drives the real CLI over a small synthetic workbook shaped like
``Emily_Data_Pruned_Labeled.xlsx``'s Normalized sheet — headers resolved
by name, empty intensity cells as genuine zeros — and checks the lab's
3x blank rule at its boundary, the physicochemical feature arithmetic,
the drop rule, the refusals, and the written manifest.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict, narrow_json_to_str
from scripts.build_metab_blank_corpus import (
    BIOLOGICAL_COLUMNS,
    BLANK_COLUMNS,
    OUTPUT_HEADER,
    build_corpus,
    build_parser,
    main,
)

from ..datasets.test_xlsx_reader import sheet_of_inline_rows, write_workbook

_HEADER = [
    "Compound",
    "m/z",
    "Charge",
    "Retention time (min)",
    "Chromatographic peak width (min)",
    *BIOLOGICAL_COLUMNS,
    *BLANK_COLUMNS,
    "250220_ebtruong_combine",
]


def _peak(
    compound: str,
    mz: str,
    rt: str,
    first_bio: str,
    first_blank: str,
    combine: str = "999",
) -> list[str]:
    """Render one peak row: intensity in the first bio/blank column only.

    The pooled combine column always carries signal, proving it never
    reaches a label.

    Args:
        compound: The Compound id.
        mz: The m/z, verbatim.
        rt: The retention time in minutes, verbatim.
        first_bio: Intensity in the first biological column ("" = none).
        first_blank: Intensity in the first blank column ("" = none).
        combine: Intensity in the pooled combine column.

    Returns:
        The row in ``_HEADER`` order.
    """
    bio = [first_bio] + [""] * (len(BIOLOGICAL_COLUMNS) - 1)
    blanks = [first_blank] + [""] * (len(BLANK_COLUMNS) - 1)
    return [compound, mz, "1", rt, "0.05", *bio, *blanks, combine]


def _write_workbook(path: Path) -> None:
    """Write the happy-path synthetic workbook.

    Four peaks: samples-only (real), exactly at the 3x boundary (real —
    the rule is >=), blank-dominated (blank), and detected only in the
    pooled combine (dropped: undetected in samples and individual
    blanks). A row with an empty Compound cell is structural padding.

    Args:
        path: Destination workbook path.
    """
    rows = [
        _HEADER,
        # s_avg = 46/23 = 2, blanks empty -> real.
        _peak("p1", "100.0", "0.80", "46", ""),
        # s_avg = 69/23 = 3, b_avg = 12/12 = 1 -> exactly 3x -> real.
        _peak("p2", "250.25", "0.83", "69", "12"),
        # s_avg = 1, b_avg = 1 -> below 3x -> blank-dominated.
        _peak("p3", "538.5", "5.20", "23", "12"),
        # Only the pooled combine saw it -> dropped, counted.
        _peak("p4", "300.0", "5.25", "", ""),
        ["", *[""] * (len(_HEADER) - 1)],
    ]
    write_workbook(path, {"Normalized": sheet_of_inline_rows(rows)})


class TestBuildCorpus:
    """The 3x rule, the feature arithmetic, and the refusals."""

    def test_labels_drops_and_groups(self, tmp_path: Path) -> None:
        """Three peaks survive with the rule's verdicts; one drops."""
        path = tmp_path / "emily.xlsx"
        _write_workbook(path)
        result = build_corpus(path)
        assert result["header"] == OUTPUT_HEADER
        assert result["n_real"] == 2
        assert result["n_blank"] == 1
        assert result["n_dropped_undetected"] == 1
        assert result["n_rt_bins"] == 2  # 0.80 and 0.83 share bin 8; 5.20 is bin 52

    def test_physicochemical_features_are_exact(self, tmp_path: Path) -> None:
        """m/z defect and Kendrick (CH2) defect come out to the digit."""
        path = tmp_path / "emily.xlsx"
        _write_workbook(path)
        rows = build_corpus(path)["rows"]
        assert rows[0] == ["8", "100.0", "1", "0.80", "0.05", "0.000000", "0.111661", "1"]
        assert rows[1] == ["8", "250.25", "1", "0.83", "0.05", "0.250000", "0.029431", "1"]
        assert rows[2] == ["52", "538.5", "1", "5.20", "0.05", "0.500000", "0.101294", "0"]

    def test_missing_header_is_refused_by_name(self, tmp_path: Path) -> None:
        """A sheet without a blank column cannot silently build."""
        path = tmp_path / "emily.xlsx"
        header = [h for h in _HEADER if h != "Blk2"]
        write_workbook(path, {"Normalized": sheet_of_inline_rows([header])})
        with pytest.raises(ValueError, match="missing required header 'Blk2'"):
            build_corpus(path)

    def test_missing_required_metadata_is_refused(self, tmp_path: Path) -> None:
        """A peak with no retention time is a source defect, not a skip."""
        path = tmp_path / "emily.xlsx"
        bad = _peak("p9", "100.0", "", "46", "")
        write_workbook(path, {"Normalized": sheet_of_inline_rows([_HEADER, bad])})
        with pytest.raises(
            ValueError,
            match="peak 'p9' has no value for required column 'Retention time",
        ):
            build_corpus(path)

    def test_empty_corpus_is_refused(self, tmp_path: Path) -> None:
        """If every peak drops, the builder refuses rather than writing."""
        path = tmp_path / "emily.xlsx"
        rows = [_HEADER, _peak("p4", "300.0", "5.25", "", "")]
        write_workbook(path, {"Normalized": sheet_of_inline_rows(rows)})
        with pytest.raises(ValueError, match="no rows survived"):
            build_corpus(path)

    def test_empty_sheet_is_refused(self, tmp_path: Path) -> None:
        """A sheet with no rows at all is a defect."""
        path = tmp_path / "emily.xlsx"
        write_workbook(path, {"Normalized": sheet_of_inline_rows([])})
        with pytest.raises(ValueError, match="sheet 'Normalized' is empty"):
            build_corpus(path)


class TestMain:
    """The CLI writes data.csv plus the pinned manifest and reports."""

    def test_writes_reports_and_pins_source(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """data.csv and MANIFEST.json land; stdout states the balance."""
        workbook = tmp_path / "emily.xlsx"
        _write_workbook(workbook)
        out = tmp_path / "metab_blank" / "data.csv"
        exit_code = main(["--workbook", str(workbook), "--out", str(out)])
        assert exit_code == 0
        lines = out.read_text(encoding="utf-8").splitlines()
        assert lines[0] == ",".join(OUTPUT_HEADER)
        assert len(lines) == 4

        manifest = narrow_json_to_dict(
            load_json_str((out.parent / "MANIFEST.json").read_text(encoding="utf-8"))
        )
        assert narrow_json_to_dict(manifest["corpus"]) == {
            "rows": 3,
            "rt_bins": 2,
            "real": 2,
            "blank_dominated": 1,
            "dropped_undetected": 1,
            "positive_ratio": round(2 / 3, 6),
        }
        pin = narrow_json_to_dict(narrow_json_to_dict(manifest["sources"])["workbook"])
        assert narrow_json_to_str(pin["file_name"]) == "emily.xlsx"
        assert len(narrow_json_to_str(pin["sha256"])) == 64

        out_lines = capsys.readouterr().out.splitlines()
        assert out_lines[0] == f"metab_blank: 3 rows across 2 rt bins -> {out}"
        assert out_lines[1] == "  real 2 / blank-dominated 1 (positive ratio 0.6667)"
        assert out_lines[2] == "  dropped: 1 undetected-in-samples-and-blanks"

    def test_parser_requires_both_paths(self) -> None:
        """--workbook and --out are both mandatory."""
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_module_entry_point_raises_system_exit(self, tmp_path: Path) -> None:
        """Running as ``__main__`` exits through SystemExit(main())."""
        workbook = tmp_path / "emily.xlsx"
        _write_workbook(workbook)
        out = tmp_path / "metab_blank" / "data.csv"
        argv = ["build_metab_blank_corpus", "--workbook", str(workbook), "--out", str(out)]
        saved = sys.argv
        sys.argv = argv
        try:
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_module("scripts.build_metab_blank_corpus", run_name="__main__")
            assert excinfo.value.code == 0
        finally:
            sys.argv = saved
