"""Integration tests for Thermo .raw file reader.

These tests require ThermoRawFileParser to be installed or bundled.
They are skipped if the CLI tool is not available.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from instrument_io.readers.thermo import ThermoReader

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SMALL_RAW = FIXTURES_DIR / "small.RAW"
TOOLS_DIR = Path(__file__).parent.parent / "tools" / "ThermoRawFileParser"
BUNDLED_EXE = TOOLS_DIR / "ThermoRawFileParser.exe"


def _thermorawfileparser_available() -> bool:
    """Check if ThermoRawFileParser is available."""
    if BUNDLED_EXE.exists():
        return True
    if shutil.which("ThermoRawFileParser") is not None:
        return True
    if shutil.which("ThermoRawFileParser.exe") is not None:
        return True
    return (Path.home() / ".dotnet" / "tools" / "ThermoRawFileParser.exe").exists()


def _get_raw_file() -> Path:
    """Get path to small.RAW test file, skip if not found."""
    if not SMALL_RAW.exists():
        pytest.skip("small.RAW test fixture not found")
    return SMALL_RAW


# Skip all tests if ThermoRawFileParser not available
pytestmark = pytest.mark.skipif(
    not _thermorawfileparser_available(),
    reason="ThermoRawFileParser CLI not installed",
)


class TestThermoReaderIntegration:
    """Integration tests using real .RAW files."""

    def test_supports_format_real_raw(self) -> None:
        """Test that reader recognizes real .RAW file."""
        raw_file = _get_raw_file()
        reader = ThermoReader()
        assert reader.supports_format(raw_file) is True

    def test_read_tic_real_raw(self) -> None:
        """Test reading TIC from real .RAW file."""
        raw_file = _get_raw_file()
        reader = ThermoReader()
        tic_data = reader.read_tic(raw_file)

        # Verify structure
        assert tic_data["meta"]["signal_type"] == "TIC"
        assert tic_data["meta"]["detector"] == "MS"
        assert str(raw_file) in tic_data["meta"]["source_path"]

        # Verify data has expected number of points (48 scans in small.RAW)
        assert tic_data["stats"]["num_points"] == 48
        assert len(tic_data["data"]["retention_times"]) == 48
        assert len(tic_data["data"]["intensities"]) == 48

        # Verify stats are reasonable
        assert tic_data["stats"]["rt_min"] >= 0.0
        assert tic_data["stats"]["rt_max"] > tic_data["stats"]["rt_min"]
        assert tic_data["stats"]["intensity_max"] > 0.0

    def test_read_eic_real_raw(self) -> None:
        """Test reading EIC from real .RAW file."""
        raw_file = _get_raw_file()
        reader = ThermoReader()

        # Use a common m/z value that should have signal
        target_mz = 400.0
        mz_tolerance = 1.0

        eic_data = reader.read_eic(raw_file, target_mz=target_mz, mz_tolerance=mz_tolerance)

        # Verify structure
        assert eic_data["meta"]["signal_type"] == "EIC"
        assert eic_data["params"]["target_mz"] == target_mz
        assert eic_data["params"]["mz_tolerance"] == mz_tolerance

        # Verify we got data for all 48 scans
        assert eic_data["stats"]["num_points"] == 48

    def test_iter_spectra_real_raw(self) -> None:
        """Test iterating spectra from real .RAW file."""
        raw_file = _get_raw_file()
        reader = ThermoReader()

        spectra = list(reader.iter_spectra(raw_file))

        # small.RAW has 48 spectra
        assert len(spectra) == 48

        # Check first spectrum structure
        sp = spectra[0]
        assert str(raw_file) in sp["meta"]["source_path"]
        assert sp["meta"]["scan_number"] >= 1
        assert sp["meta"]["retention_time"] >= 0.0
        assert sp["meta"]["ms_level"] >= 1
        assert len(sp["data"]["mz_values"]) == len(sp["data"]["intensities"])
        assert sp["stats"]["num_peaks"] >= 1

    def test_count_spectra_real_raw(self) -> None:
        """Test counting spectra in real .RAW file."""
        raw_file = _get_raw_file()
        reader = ThermoReader()

        count = reader.count_spectra(raw_file)

        # small.RAW has 48 spectra
        assert count == 48

    def test_read_spectrum_real_raw(self) -> None:
        """Test reading single spectrum from real .RAW file."""
        raw_file = _get_raw_file()
        reader = ThermoReader()

        # Read first spectrum (scan 1)
        spectrum = reader.read_spectrum(raw_file, scan_number=1)

        assert spectrum["meta"]["scan_number"] == 1
        assert str(raw_file) in spectrum["meta"]["source_path"]
        assert spectrum["stats"]["num_peaks"] >= 1
