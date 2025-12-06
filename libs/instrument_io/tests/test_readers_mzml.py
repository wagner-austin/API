"""Tests for readers.mzml module."""

from __future__ import annotations

from pathlib import Path

from instrument_io.readers.mzml import (
    MzMLReader,
    _compute_chromatogram_stats,
    _is_mzml_file,
    _is_mzxml_file,
)


# Test _compute_chromatogram_stats
def test_compute_chromatogram_stats_empty() -> None:
    """Test stats computation with empty data."""
    stats = _compute_chromatogram_stats([], [])

    assert stats["num_points"] == 0
    assert stats["rt_min"] == 0.0
    assert stats["rt_max"] == 0.0
    assert stats["rt_step_mean"] == 0.0
    assert stats["intensity_min"] == 0.0
    assert stats["intensity_max"] == 0.0
    assert stats["intensity_mean"] == 0.0
    assert stats["intensity_p99"] == 0.0


def test_compute_chromatogram_stats_single_point() -> None:
    """Test stats computation with single data point."""
    stats = _compute_chromatogram_stats([1.0], [100.0])

    assert stats["num_points"] == 1
    assert stats["rt_min"] == 1.0
    assert stats["rt_max"] == 1.0
    assert stats["rt_step_mean"] == 0.0  # Single point has no step
    assert stats["intensity_min"] == 100.0
    assert stats["intensity_max"] == 100.0
    assert stats["intensity_mean"] == 100.0
    assert stats["intensity_p99"] == 100.0  # p99_idx >= n case


def test_compute_chromatogram_stats_normal() -> None:
    """Test stats computation with normal data."""
    rt = [0.0, 1.0, 2.0, 3.0, 4.0]
    intensities = [10.0, 20.0, 30.0, 40.0, 50.0]
    stats = _compute_chromatogram_stats(rt, intensities)

    assert stats["num_points"] == 5
    assert stats["rt_min"] == 0.0
    assert stats["rt_max"] == 4.0
    assert stats["rt_step_mean"] == 1.0
    assert stats["intensity_min"] == 10.0
    assert stats["intensity_max"] == 50.0
    assert stats["intensity_mean"] == 30.0


# Test helper functions
def test_is_mzml_file_valid(tmp_path: Path) -> None:
    file = tmp_path / "test.mzML"
    file.touch()
    assert _is_mzml_file(file) is True


def test_is_mzml_file_lowercase(tmp_path: Path) -> None:
    file = tmp_path / "test.mzml"
    file.touch()
    assert _is_mzml_file(file) is True


def test_is_mzml_file_wrong_extension(tmp_path: Path) -> None:
    file = tmp_path / "test.xml"
    file.touch()
    assert _is_mzml_file(file) is False


def test_is_mzml_file_directory(tmp_path: Path) -> None:
    directory = tmp_path / "test.mzML"
    directory.mkdir()
    assert _is_mzml_file(directory) is False


def test_is_mzml_file_not_exists(tmp_path: Path) -> None:
    file = tmp_path / "nonexistent.mzML"
    assert _is_mzml_file(file) is False


def test_is_mzxml_file_valid(tmp_path: Path) -> None:
    file = tmp_path / "test.mzXML"
    file.touch()
    assert _is_mzxml_file(file) is True


def test_is_mzxml_file_lowercase(tmp_path: Path) -> None:
    file = tmp_path / "test.mzxml"
    file.touch()
    assert _is_mzxml_file(file) is True


def test_is_mzxml_file_wrong_extension(tmp_path: Path) -> None:
    file = tmp_path / "test.xml"
    file.touch()
    assert _is_mzxml_file(file) is False


def test_is_mzxml_file_directory(tmp_path: Path) -> None:
    directory = tmp_path / "test.mzXML"
    directory.mkdir()
    assert _is_mzxml_file(directory) is False


# Test MzMLReader class
class TestMzMLReader:
    """Tests for MzMLReader class."""

    def test_supports_format_mzml(self, tmp_path: Path) -> None:
        file = tmp_path / "test.mzML"
        file.touch()
        reader = MzMLReader()
        assert reader.supports_format(file) is True

    def test_supports_format_mzxml(self, tmp_path: Path) -> None:
        file = tmp_path / "test.mzXML"
        file.touch()
        reader = MzMLReader()
        assert reader.supports_format(file) is True

    def test_supports_format_other(self, tmp_path: Path) -> None:
        file = tmp_path / "test.csv"
        file.touch()
        reader = MzMLReader()
        assert reader.supports_format(file) is False

    def test_read_tic_unsupported_format_raises(self, tmp_path: Path) -> None:
        import pytest

        from instrument_io._exceptions import MzMLReadError

        file = tmp_path / "test.csv"
        file.touch()
        reader = MzMLReader()
        with pytest.raises(MzMLReadError) as exc_info:
            reader.read_tic(file)
        assert "Unsupported format" in str(exc_info.value)

    def test_read_eic_unsupported_format_raises(self, tmp_path: Path) -> None:
        import pytest

        from instrument_io._exceptions import MzMLReadError

        file = tmp_path / "test.csv"
        file.touch()
        reader = MzMLReader()
        with pytest.raises(MzMLReadError) as exc_info:
            reader.read_eic(file, 500.0, 0.5)
        assert "Unsupported format" in str(exc_info.value)
