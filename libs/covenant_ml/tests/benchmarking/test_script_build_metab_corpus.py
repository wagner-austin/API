"""Tests for the metab_confidence corpus builder.

Drives the real CLI over small synthetic sources shaped exactly like the
artcal campaign files — the MGF's MS1/MS2 block pairs, the quant table's
``<id>/<mz>mz/<rt>min`` row keys and pooled ``combine.mzML`` column, and
the SIRIUS structures TSV — and checks the join, the drop rules, the
refusals, and the written manifest.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict, narrow_json_to_str
from scripts.build_metab_corpus import (
    OUTPUT_HEADER,
    build_corpus,
    build_parser,
    main,
    parse_mgf,
    parse_quant,
    parse_structures,
)

_STRUCTURE_HEADER = "structurePerIdRank\tmappingFeatureId\tConfidenceScoreExact\tname"


def _mgf_block(
    fid: str,
    mslevel: str,
    rt: str,
    height: str,
    peaks: list[str],
) -> str:
    """Render one MGF block in the mzMine DIA export shape.

    Args:
        fid: The ``FEATURE_ID`` value.
        mslevel: The ``MSLEVEL`` value.
        rt: The ``RTINSECONDS`` value.
        height: The ``FEATURE_MS1_HEIGHT`` value.
        peaks: Peak lines (``mz intensity``).

    Returns:
        The block text, including BEGIN/END markers.
    """
    lines = [
        "BEGIN IONS",
        f"FEATURE_ID={fid}",
        f"MSLEVEL={mslevel}",
        f"RTINSECONDS={rt}",
        f"PEPMASS=100.{fid}0000",
        "CHARGE=1",
        f"FEATURE_MS1_HEIGHT={height}",
        f"Num peaks={len(peaks)}",
        *peaks,
        "END IONS",
    ]
    return "\n".join(lines) + "\n"


def _write_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write the happy-path synthetic sources.

    Four features: 1 keeps (conf 0.5), 2 drops for ``-Infinity``
    confidence, 3 keeps (conf 0.25, a different retention window), 4
    drops for zero biological detections (its only intensity is in the
    pooled ``combine.mzML`` column). A rank-2 structure row and a decoy
    MGF feature (99, plus one block with no FEATURE_ID at all) must
    stream past untouched.

    Args:
        tmp_path: Directory to write into.

    Returns:
        The MGF, quant and structures paths, in order.
    """
    mgf = tmp_path / "spectra.mgf"
    blocks = [
        _mgf_block("1", "1", "30.00", "1000", ["100.1 50", "101.1 5"]),
        _mgf_block("1", "2", "30.00", "1000", ["50.0 10", "60.0 30", "70.0 60", "80.0 100"]),
        _mgf_block("2", "1", "40.00", "10", ["100.2 1"]),
        _mgf_block("2", "2", "40.00", "10", ["50.0 1"]),
        # A blank line inside a block must be ignored.
        _mgf_block("3", "1", "66.60", "100", ["100.3 1"]).replace("CHARGE=1\n", "CHARGE=1\n\n"),
        _mgf_block("3", "2", "66.60", "100", ["55.0 1000"]),
        _mgf_block("4", "1", "12.00", "10", ["100.4 1"]),
        _mgf_block("4", "2", "12.00", "10", ["50.0 10"]),
        _mgf_block("99", "1", "5.00", "1", ["9.9 1"]),
        _mgf_block("99", "2", "5.00", "1", ["9.9 1"]).replace("FEATURE_ID=99\n", ""),
    ]
    mgf.write_text("\n".join(blocks), encoding="utf-8")

    quant = tmp_path / "quant.csv"
    quant_lines = [
        '"Filename",A.mzML,B.mzML,combine.mzML',
        "Organ_Water,Drought_Leaf,Ambient_Root,Combined_Root",
        "1/100.1000mz/0.50min,10,1000,555",
        # A blank interior line must be skipped, not parsed as a feature.
        "",
        "2/100.2000mz/0.67min,1,,",
        "3/100.3000mz/1.11min,100,,",
        "4/100.4000mz/0.20min,,,777",
        "99/9.9000mz/0.08min,1,1,1",
    ]
    quant.write_text("\n".join(quant_lines) + "\n", encoding="utf-8")

    structures = tmp_path / "structures.tsv"
    structure_lines = [
        _STRUCTURE_HEADER,
        '1\t1\t0.5\tcompound "one"',
        "2\t1\t0.5\tsecond-rank row",
        "1\t2\t-Infinity\tno exact confidence",
        "1\t3\t0.25\tthird",
        "1\t4\t0.9\tpooled-only",
    ]
    structures.write_text("\n".join(structure_lines) + "\n", encoding="utf-8")
    return mgf, quant, structures


class TestParseStructures:
    """Rank filtering, quote tolerance, and refusals."""

    def test_rank1_targets_keyed_by_feature_id(self, tmp_path: Path) -> None:
        """Only rank-1 rows land; quotes in names do not break parsing."""
        _, _, structures = _write_sources(tmp_path)
        targets = parse_structures(structures)
        assert targets == {"1": "0.5", "2": "-Infinity", "3": "0.25", "4": "0.9"}

    def test_missing_column_is_refused_by_name(self, tmp_path: Path) -> None:
        """A TSV without the confidence column cannot silently build."""
        structures = tmp_path / "structures.tsv"
        structures.write_text("structurePerIdRank\tmappingFeatureId\n1\t1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing required column 'ConfidenceScoreExact'"):
            parse_structures(structures)

    def test_duplicate_rank1_row_is_refused(self, tmp_path: Path) -> None:
        """Two rank-1 rows for one feature is a source defect."""
        structures = tmp_path / "structures.tsv"
        structures.write_text(
            _STRUCTURE_HEADER + "\n1\t7\t0.5\ta\n1\t7\t0.4\tb\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="feature 7 has more than one rank-1"):
            parse_structures(structures)


class TestParseMgf:
    """Block collection, decoy skipping, and structural refusals."""

    def test_collects_wanted_block_pairs(self, tmp_path: Path) -> None:
        """MS1 and MS2 measurables land per wanted feature; decoys stream past."""
        mgf, _, _ = _write_sources(tmp_path)
        ms1, ms2 = parse_mgf(mgf, frozenset({"1", "2", "3", "4"}))
        assert sorted(ms1) == ["1", "2", "3", "4"]
        assert sorted(ms2) == ["1", "2", "3", "4"]
        assert ms1["1"] == {
            "precursor_mz": "100.10000",
            "rt_seconds": "30.00",
            "ms1_height": 1000.0,
            "n_peaks": 2,
        }
        assert ms2["1"] == {
            "n_peaks": 4,
            "total_intensity": 200.0,
            "max_intensity": 100.0,
            "top3_intensity": 190.0,
        }

    def test_zero_peak_ms2_block_parses_with_zero_intensities(self, tmp_path: Path) -> None:
        """An empty fragment list parses; the corpus refuses it later."""
        mgf = tmp_path / "spectra.mgf"
        mgf.write_text(_mgf_block("6", "2", "10.00", "5", []), encoding="utf-8")
        _, ms2 = parse_mgf(mgf, frozenset({"6"}))
        assert ms2["6"] == {
            "n_peaks": 0,
            "total_intensity": 0.0,
            "max_intensity": 0.0,
            "top3_intensity": 0.0,
        }

    def test_peak_count_disagreement_is_refused(self, tmp_path: Path) -> None:
        """A declared Num peaks that disagrees with the listing is a defect."""
        mgf = tmp_path / "spectra.mgf"
        block = _mgf_block("6", "1", "10.00", "5", ["1.0 1"]).replace("Num peaks=1", "Num peaks=3")
        mgf.write_text(block, encoding="utf-8")
        with pytest.raises(ValueError, match="feature 6 MSLEVEL=1 declares 3 peaks but lists 1"):
            parse_mgf(mgf, frozenset({"6"}))

    def test_duplicate_ms1_block_is_refused(self, tmp_path: Path) -> None:
        """Two MS1 blocks for one feature is a source defect."""
        mgf = tmp_path / "spectra.mgf"
        block = _mgf_block("6", "1", "10.00", "5", ["1.0 1"])
        mgf.write_text(block + "\n" + block, encoding="utf-8")
        with pytest.raises(ValueError, match="feature 6 has more than one MS1 block"):
            parse_mgf(mgf, frozenset({"6"}))

    def test_duplicate_ms2_block_is_refused(self, tmp_path: Path) -> None:
        """Two MS2 blocks for one feature is a source defect."""
        mgf = tmp_path / "spectra.mgf"
        block = _mgf_block("6", "2", "10.00", "5", ["1.0 1"])
        mgf.write_text(block + "\n" + block, encoding="utf-8")
        with pytest.raises(ValueError, match="feature 6 has more than one MS2 block"):
            parse_mgf(mgf, frozenset({"6"}))

    def test_unexpected_mslevel_is_refused(self, tmp_path: Path) -> None:
        """Any MSLEVEL other than 1 or 2 is a source defect."""
        mgf = tmp_path / "spectra.mgf"
        mgf.write_text(_mgf_block("6", "3", "10.00", "5", ["1.0 1"]), encoding="utf-8")
        with pytest.raises(ValueError, match="feature 6 has unexpected MSLEVEL '3'"):
            parse_mgf(mgf, frozenset({"6"}))


class TestParseQuant:
    """Sample statistics, pooled-column exclusion, and refusals."""

    def test_pooled_column_never_counts(self, tmp_path: Path) -> None:
        """combine.mzML's intensities reach no statistic."""
        _, quant, _ = _write_sources(tmp_path)
        stats = parse_quant(quant, frozenset({"1", "4"}))
        assert stats["1"] == {"n_detected": 2, "mean_detected": 505.0, "max_detected": 1000.0}
        assert stats["4"] == {"n_detected": 0, "mean_detected": 0.0, "max_detected": 0.0}

    def test_short_rows_read_as_undetected(self, tmp_path: Path) -> None:
        """A row with trailing cells missing counts them as not detected."""
        quant = tmp_path / "quant.csv"
        quant.write_text(
            '"Filename",A.mzML,B.mzML\nOrgan_Water,x,y\n1/1.0mz/0.1min,7\n',
            encoding="utf-8",
        )
        stats = parse_quant(quant, frozenset({"1"}))
        assert stats["1"] == {"n_detected": 1, "mean_detected": 7.0, "max_detected": 7.0}

    def test_no_biological_columns_is_refused(self, tmp_path: Path) -> None:
        """A table whose only sample column is the pooled one cannot build."""
        quant = tmp_path / "quant.csv"
        quant.write_text('"Filename",combine.mzML\nOrgan_Water,x\n', encoding="utf-8")
        with pytest.raises(ValueError, match="no biological sample columns"):
            parse_quant(quant, frozenset({"1"}))

    def test_duplicate_feature_row_is_refused(self, tmp_path: Path) -> None:
        """Two quant rows for one feature id is a source defect."""
        quant = tmp_path / "quant.csv"
        quant.write_text(
            '"Filename",A.mzML\nOrgan_Water,x\n1/1.0mz/0.1min,7\n1/2.0mz/0.2min,8\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="feature 1 appears twice"):
            parse_quant(quant, frozenset({"1"}))

    def test_non_positive_intensity_is_refused(self, tmp_path: Path) -> None:
        """A zero intensity cannot feed a log feature."""
        quant = tmp_path / "quant.csv"
        quant.write_text('"Filename",A.mzML\nOrgan_Water,x\n1/1.0mz/0.1min,0\n', encoding="utf-8")
        with pytest.raises(ValueError, match=r"non-positive intensity 0\.0 in 'A\.mzML'"):
            parse_quant(quant, frozenset({"1"}))


class TestBuildCorpus:
    """The join, the drop rules, and the missing-source refusals."""

    def _parsed(
        self, tmp_path: Path
    ) -> tuple[
        dict[str, str],
        tuple[Path, Path, Path],
    ]:
        """Parse the happy-path sources.

        Args:
            tmp_path: Directory holding the sources.

        Returns:
            The targets mapping and the source paths.
        """
        paths = _write_sources(tmp_path)
        return parse_structures(paths[2]), paths

    def test_rows_drops_and_target_mean(self, tmp_path: Path) -> None:
        """Features 1 and 3 keep; 2 and 4 drop, each under its own rule."""
        targets, (mgf, quant, _) = self._parsed(tmp_path)
        wanted = frozenset(targets)
        ms1, ms2 = parse_mgf(mgf, wanted)
        result = build_corpus(targets, ms1, ms2, parse_quant(quant, wanted))
        assert result["header"] == OUTPUT_HEADER
        assert result["n_dropped_infinite"] == 1
        assert result["n_dropped_undetected"] == 1
        assert result["n_rt_bins"] == 2
        assert result["target_mean"] == pytest.approx(0.375)
        assert result["rows"] == [
            [
                "5",
                "100.10000",
                "30.00",
                "3.000000",
                "2",
                "4",
                "2.301030",
                "2.000000",
                "0.950000",
                "2",
                "2.703291",
                "3.000000",
                "0.5",
            ],
            [
                "11",
                "100.30000",
                "66.60",
                "2.000000",
                "1",
                "1",
                "3.000000",
                "3.000000",
                "1.000000",
                "1",
                "2.000000",
                "2.000000",
                "0.25",
            ],
        ]

    def test_missing_ms1_block_is_refused(self, tmp_path: Path) -> None:
        """A structure call without an MS1 block is a source defect."""
        targets, (mgf, quant, _) = self._parsed(tmp_path)
        wanted = frozenset(targets)
        ms1, ms2 = parse_mgf(mgf, wanted)
        del ms1["1"]
        with pytest.raises(ValueError, match="feature 1 has a structure call but no MS1 block"):
            build_corpus(targets, ms1, ms2, parse_quant(quant, wanted))

    def test_missing_ms2_block_is_refused(self, tmp_path: Path) -> None:
        """A structure call without an MS2 block is a source defect."""
        targets, (mgf, quant, _) = self._parsed(tmp_path)
        wanted = frozenset(targets)
        ms1, ms2 = parse_mgf(mgf, wanted)
        del ms2["1"]
        with pytest.raises(ValueError, match="feature 1 has a structure call but no MS2 block"):
            build_corpus(targets, ms1, ms2, parse_quant(quant, wanted))

    def test_missing_quant_row_is_refused(self, tmp_path: Path) -> None:
        """A structure call without a quant row is a source defect."""
        targets, (mgf, quant, _) = self._parsed(tmp_path)
        wanted = frozenset(targets)
        ms1, ms2 = parse_mgf(mgf, wanted)
        stats = parse_quant(quant, wanted)
        del stats["1"]
        with pytest.raises(ValueError, match="feature 1 has a structure call but no quant"):
            build_corpus(targets, ms1, ms2, stats)

    def test_zero_intensity_spectrum_is_refused(self, tmp_path: Path) -> None:
        """An empty MS2 fragment list cannot feed the log features."""
        targets, (mgf, quant, _) = self._parsed(tmp_path)
        wanted = frozenset(targets)
        ms1, ms2 = parse_mgf(mgf, wanted)
        ms2["1"] = {
            "n_peaks": 0,
            "total_intensity": 0.0,
            "max_intensity": 0.0,
            "top3_intensity": 0.0,
        }
        with pytest.raises(ValueError, match="feature 1 MS2 total intensity must be positive"):
            build_corpus(targets, ms1, ms2, parse_quant(quant, wanted))

    def test_empty_corpus_is_refused(self, tmp_path: Path) -> None:
        """If every row drops, the builder refuses rather than writing."""
        targets = {"2": "-Infinity"}
        with pytest.raises(ValueError, match="no rows survived"):
            build_corpus(targets, {}, {}, {})


class TestMain:
    """The CLI writes data.csv plus the pinned manifest and reports."""

    def test_writes_reports_and_pins_sources(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """data.csv and MANIFEST.json land; stdout states rows and drops."""
        mgf, quant, structures = _write_sources(tmp_path)
        out = tmp_path / "metab_confidence" / "data.csv"
        exit_code = main(
            [
                "--mgf",
                str(mgf),
                "--quant",
                str(quant),
                "--structures",
                str(structures),
                "--out",
                str(out),
            ]
        )
        assert exit_code == 0
        lines = out.read_text(encoding="utf-8").splitlines()
        assert lines[0] == ",".join(OUTPUT_HEADER)
        assert len(lines) == 3

        manifest = narrow_json_to_dict(
            load_json_str((out.parent / "MANIFEST.json").read_text(encoding="utf-8"))
        )
        assert narrow_json_to_dict(manifest["corpus"]) == {
            "rows": 2,
            "rt_bins": 2,
            "dropped_infinite_confidence": 1,
            "dropped_undetected": 1,
            "target_mean": 0.375,
        }
        pins = narrow_json_to_dict(manifest["sources"])
        assert narrow_json_to_str(narrow_json_to_dict(pins["mgf"])["file_name"]) == "spectra.mgf"
        for source in ("mgf", "quant", "structures"):
            assert len(narrow_json_to_str(narrow_json_to_dict(pins[source])["sha256"])) == 64

        captured = capsys.readouterr()
        out_lines = captured.out.splitlines()
        assert out_lines[0] == f"metab_confidence: 2 rows across 2 rt bins -> {out}"
        assert out_lines[1] == (
            "  dropped: 1 infinite-confidence, 1 undetected-in-biological-samples"
        )
        assert out_lines[2] == "  confidence_exact: mean 0.3750"

    def test_parser_requires_all_paths(self) -> None:
        """All four paths are mandatory."""
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_module_entry_point_raises_system_exit(self, tmp_path: Path) -> None:
        """Running as ``__main__`` exits through SystemExit(main())."""
        mgf, quant, structures = _write_sources(tmp_path)
        out = tmp_path / "metab_confidence" / "data.csv"
        argv = [
            "build_metab_corpus",
            "--mgf",
            str(mgf),
            "--quant",
            str(quant),
            "--structures",
            str(structures),
            "--out",
            str(out),
        ]
        saved = sys.argv
        sys.argv = argv
        try:
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_module("scripts.build_metab_corpus", run_name="__main__")
            assert excinfo.value.code == 0
        finally:
            sys.argv = saved
