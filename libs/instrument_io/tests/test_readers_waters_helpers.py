"""Tests for readers waters: IsWatersRawDirectory."""

from __future__ import annotations

from pathlib import Path

import pytest

from instrument_io._exceptions import WatersReadError
from instrument_io._protocols.rainbow import DataDirectoryProtocol
from instrument_io.fakes import FakeDataDirectory, FakeDataFile
from instrument_io.readers.waters import (
    WatersReader,
    _build_chromatogram_meta,
    _extract_eic_intensities,
    _find_ms_file,
    _find_ms_file_optional,
    _find_tic_file_optional,
    _find_uv_file,
    _is_waters_raw_directory,
)


class TestIsWatersRawDirectory:
    """Test _is_waters_raw_directory helper."""

    def test_valid_raw_directory(self, tmp_path: Path) -> None:
        """Test detection of valid .raw directory."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()
        assert _is_waters_raw_directory(raw_dir) is True

    def test_lowercase_raw_extension(self, tmp_path: Path) -> None:
        """Test lowercase .raw extension."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()
        assert _is_waters_raw_directory(raw_dir) is True

    def test_uppercase_raw_extension(self, tmp_path: Path) -> None:
        """Test uppercase .RAW extension."""
        raw_dir = tmp_path / "test.RAW"
        raw_dir.mkdir()
        assert _is_waters_raw_directory(raw_dir) is True

    def test_not_directory(self, tmp_path: Path) -> None:
        """Test file with .raw extension is rejected."""
        raw_file = tmp_path / "test.raw"
        raw_file.write_text("not a directory")
        assert _is_waters_raw_directory(raw_file) is False

    def test_wrong_extension(self, tmp_path: Path) -> None:
        """Test directory with wrong extension."""
        d_dir = tmp_path / "test.D"
        d_dir.mkdir()
        assert _is_waters_raw_directory(d_dir) is False


class TestFindTicFileOptional:
    """Test _find_tic_file_optional helper."""

    def test_finds_tic_by_detector(self) -> None:
        """Test finding TIC via get_detector."""
        tic_file = FakeDataFile([1.0], [], [100.0], "TIC", "tic_data.dat")
        datadir: DataDirectoryProtocol = FakeDataDirectory([tic_file], "/test")
        result = _find_tic_file_optional(datadir)
        assert result == tic_file
        assert result.detector == "TIC"

    def test_finds_tic_in_datafiles(self) -> None:
        """Test finding TIC by searching datafiles."""
        tic_file = FakeDataFile([1.0], [], [100.0], "tic_scan", "tic_scan_data.dat")
        datadir: DataDirectoryProtocol = FakeDataDirectory([tic_file], "/test")
        result = _find_tic_file_optional(datadir)
        assert result == tic_file
        assert "tic" in result.detector.lower()

    def test_finds_total_in_datafiles(self) -> None:
        """Test finding TIC via 'total' in detector name."""
        total_file = FakeDataFile([1.0], [], [100.0], "Total Ion", "total_ion.dat")
        datadir: DataDirectoryProtocol = FakeDataDirectory([total_file], "/test")
        result = _find_tic_file_optional(datadir)
        assert result == total_file
        assert "total" in result.detector.lower()

    def test_returns_none_when_not_found(self) -> None:
        """Test returns None when no TIC found."""
        ms_file = FakeDataFile([1.0], [], [100.0], "MS", "ms_data.dat")
        datadir: DataDirectoryProtocol = FakeDataDirectory([ms_file], "/test")
        result = _find_tic_file_optional(datadir)
        assert result is None


class TestFindMsFileOptional:
    """Test _find_ms_file_optional helper."""

    def test_finds_ms_by_detector(self) -> None:
        """Test finding MS via get_detector."""
        ms_file = FakeDataFile([1.0], [100.0], [[50.0]], "MS", "ms_data.dat")
        datadir: DataDirectoryProtocol = FakeDataDirectory([ms_file], "/test")
        result = _find_ms_file_optional(datadir)
        assert result == ms_file
        assert result.detector == "MS"

    def test_finds_ms_in_datafiles(self) -> None:
        """Test finding MS by searching datafiles."""
        ms_file = FakeDataFile([1.0], [100.0], [[50.0]], "ms_scan", "ms_scan.dat")
        datadir: DataDirectoryProtocol = FakeDataDirectory([ms_file], "/test")
        result = _find_ms_file_optional(datadir)
        assert result == ms_file
        assert "ms" in result.detector.lower()

    def test_returns_none_when_not_found(self) -> None:
        """Test returns None when no MS found."""
        uv_file = FakeDataFile([1.0], [], [100.0], "UV", "uv_data.dat")
        datadir: DataDirectoryProtocol = FakeDataDirectory([uv_file], "/test")
        result = _find_ms_file_optional(datadir)
        assert result is None


class TestFindMsFile:
    """Test _find_ms_file helper."""

    def test_raises_when_not_found(self) -> None:
        """Test raises WatersReadError when MS not found."""
        uv_file = FakeDataFile([1.0], [], [100.0], "UV", "uv_data.dat")
        datadir: DataDirectoryProtocol = FakeDataDirectory([uv_file], "/test")
        with pytest.raises(WatersReadError) as exc_info:
            _find_ms_file(datadir, "/test")
        assert "No MS data file found" in str(exc_info.value)


class TestFindUvFile:
    """Test _find_uv_file helper."""

    def test_finds_uv_by_detector(self) -> None:
        """Test finding UV via get_detector."""
        uv_file = FakeDataFile([1.0], [200.0], [[50.0]], "UV", "uv_data.dat")
        datadir: DataDirectoryProtocol = FakeDataDirectory([uv_file], "/test")
        result = _find_uv_file(datadir, "/test")
        assert result == uv_file
        assert result.detector == "UV"

    def test_finds_pda_in_datafiles(self) -> None:
        """Test finding UV via 'pda' in detector name."""
        pda_file = FakeDataFile([1.0], [200.0], [[50.0]], "pda_scan", "pda_scan.dat")
        datadir: DataDirectoryProtocol = FakeDataDirectory([pda_file], "/test")
        result = _find_uv_file(datadir, "/test")
        assert result == pda_file
        assert "pda" in result.detector.lower()

    def test_raises_when_not_found(self) -> None:
        """Test raises WatersReadError when UV not found."""
        ms_file = FakeDataFile([1.0], [100.0], [[50.0]], "MS", "ms_data.dat")
        datadir: DataDirectoryProtocol = FakeDataDirectory([ms_file], "/test")
        with pytest.raises(WatersReadError) as exc_info:
            _find_uv_file(datadir, "/test")
        assert "No UV data file found" in str(exc_info.value)


class TestBuildChromatogramMeta:
    """Test _build_chromatogram_meta helper."""

    def test_builds_meta_correctly(self) -> None:
        """Test building ChromatogramMeta."""
        meta = _build_chromatogram_meta("/path/to/file", "TIC", "MS")
        assert meta["source_path"] == "/path/to/file"
        assert meta["signal_type"] == "TIC"
        assert meta["detector"] == "MS"
        assert meta["instrument"] == ""
        assert meta["method_name"] == ""


class TestExtractEicIntensities:
    """Test _extract_eic_intensities helper."""

    def test_extracts_single_mz(self) -> None:
        """Test extracting EIC for single m/z match."""
        ms_data = [[100.0, 200.0, 300.0], [150.0, 250.0, 350.0]]
        mz_axis = [100.0, 200.0, 300.0]
        result = _extract_eic_intensities(ms_data, mz_axis, 200.0, 0.5, "/test")
        assert result == [200.0, 250.0]

    def test_extracts_multiple_mz(self) -> None:
        """Test extracting EIC summing multiple m/z channels."""
        ms_data = [[100.0, 200.0, 300.0], [150.0, 250.0, 350.0]]
        mz_axis = [199.0, 200.0, 201.0]
        result = _extract_eic_intensities(ms_data, mz_axis, 200.0, 1.5, "/test")
        # Should sum all three channels
        assert result == [600.0, 750.0]

    def test_raises_on_empty_data(self) -> None:
        """Test raises on empty MS data."""
        with pytest.raises(WatersReadError) as exc_info:
            _extract_eic_intensities([], [100.0], 100.0, 1.0, "/test")
        assert "Empty MS data" in str(exc_info.value)

    def test_raises_on_empty_mz_axis(self) -> None:
        """Test raises on empty m/z axis."""
        with pytest.raises(WatersReadError) as exc_info:
            _extract_eic_intensities([[100.0]], [], 100.0, 1.0, "/test")
        assert "Empty MS data" in str(exc_info.value)

    def test_raises_on_no_match(self) -> None:
        """Test raises when no m/z values match."""
        ms_data = [[100.0, 200.0]]
        mz_axis = [100.0, 200.0]
        with pytest.raises(WatersReadError) as exc_info:
            _extract_eic_intensities(ms_data, mz_axis, 500.0, 0.1, "/test")
        assert "No m/z values within" in str(exc_info.value)


class TestWatersReaderSupportsFormat:
    """Test WatersReader.supports_format method."""

    def test_supports_raw_directory(self, tmp_path: Path) -> None:
        """Test supports .raw directory."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()
        reader = WatersReader()
        assert reader.supports_format(raw_dir) is True

    def test_rejects_d_directory(self, tmp_path: Path) -> None:
        """Test rejects .D directory."""
        d_dir = tmp_path / "test.D"
        d_dir.mkdir()
        reader = WatersReader()
        assert reader.supports_format(d_dir) is False
