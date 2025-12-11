"""Tests for readers.thermo module."""

from __future__ import annotations

from pathlib import Path

import pytest

from instrument_io._exceptions import ThermoReadError
from instrument_io.readers.thermo import (
    ThermoReader,
    _compute_chromatogram_stats,
    _is_raw_file,
)
from instrument_io.testing import FakeMzMLReader, hooks
from instrument_io.types.spectrum import MSSpectrum


class TestIsRawFile:
    """Tests for _is_raw_file function."""

    def test_raw_file(self, tmp_path: Path) -> None:
        raw_file = tmp_path / "test.raw"
        raw_file.touch()
        assert _is_raw_file(raw_file) is True

    def test_raw_uppercase(self, tmp_path: Path) -> None:
        raw_file = tmp_path / "test.RAW"
        raw_file.touch()
        assert _is_raw_file(raw_file) is True

    def test_not_raw_file(self, tmp_path: Path) -> None:
        other_file = tmp_path / "test.mzML"
        other_file.touch()
        assert _is_raw_file(other_file) is False

    def test_directory_returns_false(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()
        assert _is_raw_file(raw_dir) is False

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "nonexistent.raw"
        assert _is_raw_file(nonexistent) is False


class TestComputeChromatogramStats:
    """Tests for _compute_chromatogram_stats function."""

    def test_valid_data(self) -> None:
        rt = [0.0, 1.0, 2.0, 3.0]
        intensities = [100.0, 200.0, 300.0, 400.0]

        stats = _compute_chromatogram_stats(rt, intensities)

        assert stats["num_points"] == 4
        assert stats["rt_min"] == 0.0
        assert stats["rt_max"] == 3.0
        assert stats["rt_step_mean"] == 1.0
        assert stats["intensity_min"] == 100.0
        assert stats["intensity_max"] == 400.0
        assert stats["intensity_mean"] == 250.0

    def test_two_points(self) -> None:
        rt = [0.0, 10.0]
        intensities = [500.0, 1500.0]

        stats = _compute_chromatogram_stats(rt, intensities)

        assert stats["num_points"] == 2
        assert stats["rt_step_mean"] == 10.0
        assert stats["intensity_mean"] == 1000.0

    def test_single_point(self) -> None:
        """Test stats with single point (edge case for step mean)."""
        rt = [5.0]
        intensities = [1000.0]

        stats = _compute_chromatogram_stats(rt, intensities)

        assert stats["num_points"] == 1
        assert stats["rt_step_mean"] == 0.0  # No step with single point
        assert stats["intensity_mean"] == 1000.0


class TestThermoReader:
    """Tests for ThermoReader class."""

    def test_supports_format_raw(self, tmp_path: Path) -> None:
        raw_file = tmp_path / "test.raw"
        raw_file.touch()

        reader = ThermoReader()
        assert reader.supports_format(raw_file) is True

    def test_supports_format_mzml_false(self, tmp_path: Path) -> None:
        mzml_file = tmp_path / "test.mzML"
        mzml_file.touch()

        reader = ThermoReader()
        assert reader.supports_format(mzml_file) is False

    def test_read_tic_not_raw_raises(self, tmp_path: Path) -> None:
        not_raw = tmp_path / "test.mzML"
        not_raw.touch()

        reader = ThermoReader()
        with pytest.raises(ThermoReadError) as exc_info:
            reader.read_tic(not_raw)
        assert "Not a .raw file" in str(exc_info.value)

    def test_read_eic_not_raw_raises(self, tmp_path: Path) -> None:
        not_raw = tmp_path / "test.mzML"
        not_raw.touch()

        reader = ThermoReader()
        with pytest.raises(ThermoReadError) as exc_info:
            reader.read_eic(not_raw, target_mz=100.0, mz_tolerance=0.5)
        assert "Not a .raw file" in str(exc_info.value)

    def test_read_spectrum_not_raw_raises(self, tmp_path: Path) -> None:
        not_raw = tmp_path / "test.mzML"
        not_raw.touch()

        reader = ThermoReader()
        with pytest.raises(ThermoReadError) as exc_info:
            reader.read_spectrum(not_raw, scan_number=1)
        assert "Not a .raw file" in str(exc_info.value)

    def test_iter_spectra_not_raw_raises(self, tmp_path: Path) -> None:
        not_raw = tmp_path / "test.mzML"
        not_raw.touch()

        reader = ThermoReader()
        with pytest.raises(ThermoReadError) as exc_info:
            list(reader.iter_spectra(not_raw))
        assert "Not a .raw file" in str(exc_info.value)

    def test_count_spectra_not_raw_raises(self, tmp_path: Path) -> None:
        not_raw = tmp_path / "test.mzML"
        not_raw.touch()

        reader = ThermoReader()
        with pytest.raises(ThermoReadError) as exc_info:
            reader.count_spectra(not_raw)
        assert "Not a .raw file" in str(exc_info.value)


def _make_spectrum(source_path: str, scan_number: int, rt: float, tic: float) -> MSSpectrum:
    """Create a spectrum for testing."""
    return MSSpectrum(
        meta={
            "source_path": source_path,
            "scan_number": scan_number,
            "retention_time": rt,
            "ms_level": 1,
            "polarity": "positive",
            "total_ion_current": tic,
        },
        data={"mz_values": [100.0, 200.0], "intensities": [500.0, 500.0]},
        stats={
            "num_peaks": 2,
            "mz_min": 100.0,
            "mz_max": 200.0,
            "base_peak_mz": 100.0,
            "base_peak_intensity": 500.0,
        },
    )


class TestThermoReaderWithFakes:
    """Tests for ThermoReader using fake MzMLReader via hooks."""

    def test_read_tic_success(self, tmp_path: Path) -> None:
        """Test successful TIC reading with fake conversion."""
        raw_file = tmp_path / "test.raw"
        raw_file.touch()

        mzml_output = tmp_path / "output" / "test.mzML"
        mzml_output.parent.mkdir()
        mzml_output.touch()

        mock_spectrum = _make_spectrum(str(mzml_output), 1, 1.0, 1000.0)
        fake_reader = FakeMzMLReader([mock_spectrum])

        # Set up hooks for testing
        hooks.create_temp_dir = lambda: tmp_path / "output"
        hooks.convert_raw_to_mzml = lambda raw, out_dir: mzml_output
        hooks.cleanup_temp_dir = lambda temp_dir: None
        hooks.mzml_reader_factory = lambda: fake_reader

        reader = ThermoReader()
        tic_data = reader.read_tic(raw_file)

        assert tic_data["meta"]["signal_type"] == "TIC"
        assert tic_data["stats"]["num_points"] == 1
        assert tic_data["data"]["retention_times"] == [1.0]
        assert tic_data["data"]["intensities"] == [1000.0]

    def test_read_tic_no_spectra_raises(self, tmp_path: Path) -> None:
        """Test error when no spectra found."""
        raw_file = tmp_path / "test.raw"
        raw_file.touch()

        mzml_output = tmp_path / "output" / "test.mzML"
        mzml_output.parent.mkdir()
        mzml_output.touch()

        fake_reader = FakeMzMLReader([])  # Empty - no spectra

        # Set up hooks for testing
        hooks.create_temp_dir = lambda: tmp_path / "output"
        hooks.convert_raw_to_mzml = lambda raw, out_dir: mzml_output
        hooks.cleanup_temp_dir = lambda temp_dir: None
        hooks.mzml_reader_factory = lambda: fake_reader

        reader = ThermoReader()
        with pytest.raises(ThermoReadError) as exc_info:
            reader.read_tic(raw_file)
        assert "No spectra found" in str(exc_info.value)

    def test_read_eic_success(self, tmp_path: Path) -> None:
        """Test successful EIC reading."""
        raw_file = tmp_path / "test.raw"
        raw_file.touch()

        mzml_output = tmp_path / "output" / "test.mzML"
        mzml_output.parent.mkdir()
        mzml_output.touch()

        # Spectrum with m/z values, some in range [99.5, 100.5]
        mock_spectrum = MSSpectrum(
            meta={
                "source_path": str(mzml_output),
                "scan_number": 1,
                "retention_time": 1.0,
                "ms_level": 1,
                "polarity": "positive",
                "total_ion_current": 1000.0,
            },
            data={
                "mz_values": [99.8, 100.2, 200.0],
                "intensities": [300.0, 400.0, 500.0],
            },
            stats={
                "num_peaks": 3,
                "mz_min": 99.8,
                "mz_max": 200.0,
                "base_peak_mz": 200.0,
                "base_peak_intensity": 500.0,
            },
        )

        fake_reader = FakeMzMLReader([mock_spectrum])

        # Set up hooks for testing
        hooks.create_temp_dir = lambda: tmp_path / "output"
        hooks.convert_raw_to_mzml = lambda raw, out_dir: mzml_output
        hooks.cleanup_temp_dir = lambda temp_dir: None
        hooks.mzml_reader_factory = lambda: fake_reader

        reader = ThermoReader()
        eic_data = reader.read_eic(raw_file, target_mz=100.0, mz_tolerance=0.5)

        assert eic_data["meta"]["signal_type"] == "EIC"
        assert eic_data["params"]["target_mz"] == 100.0
        assert eic_data["params"]["mz_tolerance"] == 0.5
        # Should sum intensities in range [99.5, 100.5]: 300.0 + 400.0 = 700.0
        assert eic_data["data"]["intensities"] == [700.0]

    def test_read_eic_no_spectra_raises(self, tmp_path: Path) -> None:
        """Test error when no spectra found for EIC."""
        raw_file = tmp_path / "test.raw"
        raw_file.touch()

        mzml_output = tmp_path / "output" / "test.mzML"
        mzml_output.parent.mkdir()
        mzml_output.touch()

        fake_reader = FakeMzMLReader([])

        # Set up hooks for testing
        hooks.create_temp_dir = lambda: tmp_path / "output"
        hooks.convert_raw_to_mzml = lambda raw, out_dir: mzml_output
        hooks.cleanup_temp_dir = lambda temp_dir: None
        hooks.mzml_reader_factory = lambda: fake_reader

        reader = ThermoReader()
        with pytest.raises(ThermoReadError) as exc_info:
            reader.read_eic(raw_file, target_mz=100.0, mz_tolerance=0.5)
        assert "No spectra found" in str(exc_info.value)

    def test_read_spectrum_success(self, tmp_path: Path) -> None:
        """Test successful single spectrum reading."""
        raw_file = tmp_path / "test.raw"
        raw_file.touch()

        mzml_output = tmp_path / "output" / "test.mzML"
        mzml_output.parent.mkdir()
        mzml_output.touch()

        mock_spectrum = _make_spectrum(str(mzml_output), 1, 1.0, 1000.0)
        fake_reader = FakeMzMLReader([mock_spectrum])

        # Set up hooks for testing
        hooks.create_temp_dir = lambda: tmp_path / "output"
        hooks.convert_raw_to_mzml = lambda raw, out_dir: mzml_output
        hooks.cleanup_temp_dir = lambda temp_dir: None
        hooks.mzml_reader_factory = lambda: fake_reader

        reader = ThermoReader()
        spectrum = reader.read_spectrum(raw_file, scan_number=1)

        # Source path should be updated to original .raw file
        assert str(raw_file) in spectrum["meta"]["source_path"]

    def test_iter_spectra_success(self, tmp_path: Path) -> None:
        """Test successful spectrum iteration."""
        raw_file = tmp_path / "test.raw"
        raw_file.touch()

        mzml_output = tmp_path / "output" / "test.mzML"
        mzml_output.parent.mkdir()
        mzml_output.touch()

        mock_spectrum = _make_spectrum(str(mzml_output), 1, 1.0, 1000.0)
        fake_reader = FakeMzMLReader([mock_spectrum])

        # Set up hooks for testing
        hooks.create_temp_dir = lambda: tmp_path / "output"
        hooks.convert_raw_to_mzml = lambda raw, out_dir: mzml_output
        hooks.cleanup_temp_dir = lambda temp_dir: None
        hooks.mzml_reader_factory = lambda: fake_reader

        reader = ThermoReader()
        spectra = list(reader.iter_spectra(raw_file))

        assert len(spectra) == 1
        # Source path should be updated to original .raw file
        assert str(raw_file) in spectra[0]["meta"]["source_path"]

    def test_count_spectra_success(self, tmp_path: Path) -> None:
        """Test successful spectrum counting."""
        raw_file = tmp_path / "test.raw"
        raw_file.touch()

        mzml_output = tmp_path / "output" / "test.mzML"
        mzml_output.parent.mkdir()
        mzml_output.touch()

        # Create 42 mock spectra
        spectra = [_make_spectrum(str(mzml_output), i, float(i), 1000.0) for i in range(42)]
        fake_reader = FakeMzMLReader(spectra)

        # Set up hooks for testing
        hooks.create_temp_dir = lambda: tmp_path / "output"
        hooks.convert_raw_to_mzml = lambda raw, out_dir: mzml_output
        hooks.cleanup_temp_dir = lambda temp_dir: None
        hooks.mzml_reader_factory = lambda: fake_reader

        reader = ThermoReader()
        count = reader.count_spectra(raw_file)

        assert count == 42
