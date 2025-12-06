"""Unit tests for Agilent reader with Protocol-based test objects."""

from __future__ import annotations

from pathlib import Path

import pytest

from instrument_io._exceptions import AgilentReadError
from instrument_io._protocols.rainbow import _load_data_directory
from instrument_io.readers.agilent import (
    AgilentReader,
    _build_chromatogram_meta,
    _extract_eic_intensities,
    _find_ms_file,
    _is_agilent_d_directory,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestIsAgilentDDirectory:
    """Tests for _is_agilent_d_directory."""

    def test_valid_d_directory(self, tmp_path: Path) -> None:
        d_dir = tmp_path / "sample.D"
        d_dir.mkdir()
        assert _is_agilent_d_directory(d_dir) is True

    def test_file_not_directory(self, tmp_path: Path) -> None:
        d_file = tmp_path / "sample.D"
        d_file.touch()
        assert _is_agilent_d_directory(d_file) is False

    def test_wrong_extension(self, tmp_path: Path) -> None:
        other_dir = tmp_path / "sample.X"
        other_dir.mkdir()
        assert _is_agilent_d_directory(other_dir) is False


class TestBuildChromatogramMeta:
    """Tests for _build_chromatogram_meta."""

    def test_creates_meta_tic(self) -> None:
        meta = _build_chromatogram_meta("/path/test.D", "TIC", "TIC Detector")
        assert meta["source_path"] == "/path/test.D"
        assert meta["signal_type"] == "TIC"
        assert meta["detector"] == "TIC Detector"

    def test_creates_meta_eic(self) -> None:
        meta = _build_chromatogram_meta("/path/test.D", "EIC", "MS")
        assert meta["signal_type"] == "EIC"

    def test_creates_meta_dad(self) -> None:
        meta = _build_chromatogram_meta("/path/test.D", "DAD", "DAD Detector")
        assert meta["signal_type"] == "DAD"


class TestExtractEicIntensities:
    """Tests for _extract_eic_intensities."""

    def test_empty_data_raises(self) -> None:
        with pytest.raises(AgilentReadError) as exc_info:
            _extract_eic_intensities([], [], 500.0, 0.5, "/path")
        assert "Empty MS data" in str(exc_info.value)

    def test_no_matching_mz_raises(self) -> None:
        ms_data = [[100.0, 200.0]]
        mz_axis = [100.0, 200.0]
        with pytest.raises(AgilentReadError) as exc_info:
            _extract_eic_intensities(ms_data, mz_axis, 500.0, 0.5, "/path")
        assert "No m/z values within" in str(exc_info.value)

    def test_extracts_intensities(self) -> None:
        ms_data = [[100.0, 200.0, 300.0], [150.0, 250.0, 350.0]]
        mz_axis = [499.0, 500.0, 501.0]
        result = _extract_eic_intensities(ms_data, mz_axis, 500.0, 1.0, "/path")
        # All m/z values (499, 500, 501) are within ±1.0 of 500.0
        # First time point: 100+200+300 = 600
        # Second time point: 150+250+350 = 750
        assert result == [600.0, 750.0]


class TestAgilentReader:
    """Tests for AgilentReader class."""

    def test_supports_format_d_directory(self, tmp_path: Path) -> None:
        reader = AgilentReader()
        d_dir = tmp_path / "sample.D"
        d_dir.mkdir()
        assert reader.supports_format(d_dir) is True

    def test_supports_format_non_d_directory(self, tmp_path: Path) -> None:
        reader = AgilentReader()
        other_dir = tmp_path / "sample.X"
        other_dir.mkdir()
        assert reader.supports_format(other_dir) is False

    def test_supports_format_non_existent(self, tmp_path: Path) -> None:
        reader = AgilentReader()
        nonexistent = tmp_path / "missing.D"
        assert reader.supports_format(nonexistent) is False

    def test_find_runs_not_directory_raises(self, tmp_path: Path) -> None:
        reader = AgilentReader()
        fake_file = tmp_path / "notadir.txt"
        fake_file.touch()

        with pytest.raises(AgilentReadError) as exc_info:
            reader.find_runs(fake_file)
        assert "Not a directory" in str(exc_info.value)

    def test_find_runs_empty_directory(self, tmp_path: Path) -> None:
        reader = AgilentReader()
        # Empty directory with no .D subdirectories
        runs = reader.find_runs(tmp_path)
        assert runs == []

    def test_find_runs_returns_run_info(self, tmp_path: Path) -> None:
        reader = AgilentReader()

        # Create a .D directory structure
        d_dir = tmp_path / "site1" / "sample.D"
        d_dir.mkdir(parents=True)

        runs = reader.find_runs(tmp_path)
        assert len(runs) == 1
        assert runs[0]["run_id"] == "sample"
        assert runs[0]["site"] == "site1"


class TestFindMsFile:
    """Tests for _find_ms_file function."""

    def test_find_ms_file_raises_when_no_ms(self) -> None:
        """Test that _find_ms_file raises error when no MS data found.

        brown.D has UV only, no MS data.
        """
        brown_d = FIXTURES_DIR / "brown.D"
        if not brown_d.exists():
            pytest.skip("brown.D fixture not found")

        datadir = _load_data_directory(brown_d)

        with pytest.raises(AgilentReadError) as exc_info:
            _find_ms_file(datadir)
        assert "No MS data file found" in str(exc_info.value)

    def test_find_ms_file_succeeds_with_ms_data(self) -> None:
        """Test that _find_ms_file succeeds when MS data exists.

        sample.D has MS data.
        """
        sample_d = FIXTURES_DIR / "sample.D"
        if not sample_d.exists():
            pytest.skip("sample.D fixture not found")

        datadir = _load_data_directory(sample_d)

        ms_file = _find_ms_file(datadir)
        # Verify we got a valid DataFileProtocol with detector attribute
        detector_name: str = ms_file.detector
        assert detector_name  # Non-empty string
