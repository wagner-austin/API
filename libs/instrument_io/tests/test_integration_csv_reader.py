"""Integration tests for CSV chromatogram reader.

Tests use real CSV fixture files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from instrument_io.readers.csv import CSVChromatogramReader

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CHROMATOGRAM_CSV = FIXTURES_DIR / "chromatogram.csv"


def _get_csv_file() -> Path:
    """Get path to chromatogram.csv test file, skip if not found."""
    if not CHROMATOGRAM_CSV.exists():
        pytest.skip("chromatogram.csv test fixture not found")
    return CHROMATOGRAM_CSV


class TestCSVReaderIntegration:
    """Integration tests using real CSV files."""

    def test_supports_format_real_csv(self) -> None:
        """Test that reader recognizes real CSV file."""
        csv_file = _get_csv_file()
        reader = CSVChromatogramReader()
        assert reader.supports_format(csv_file) is True

    def test_detect_delimiter_real_csv(self) -> None:
        """Test delimiter detection on real CSV file."""
        csv_file = _get_csv_file()
        reader = CSVChromatogramReader()
        delimiter = reader.detect_delimiter(csv_file)
        assert delimiter == ","

    def test_detect_columns_real_csv(self) -> None:
        """Test column detection on real CSV file."""
        csv_file = _get_csv_file()
        reader = CSVChromatogramReader()
        columns = reader.detect_columns(csv_file)
        assert "Time" in columns
        assert "Intensity" in columns

    def test_read_chromatogram_real_csv(self) -> None:
        """Test reading chromatogram from real CSV file."""
        csv_file = _get_csv_file()
        reader = CSVChromatogramReader()
        tic_data = reader.read_chromatogram(csv_file)

        # Verify structure
        assert tic_data["meta"]["signal_type"] == "TIC"
        assert tic_data["meta"]["detector"] == "CSV"
        assert str(csv_file) in tic_data["meta"]["source_path"]

        # Verify data has expected number of points (21 rows in chromatogram.csv)
        assert tic_data["stats"]["num_points"] == 21
        assert len(tic_data["data"]["retention_times"]) == 21
        assert len(tic_data["data"]["intensities"]) == 21

        # Verify first and last data points
        assert tic_data["data"]["retention_times"][0] == 0.0
        assert tic_data["data"]["retention_times"][-1] == 10.0
        assert tic_data["data"]["intensities"][0] == 1000.0

        # Verify stats are computed correctly
        assert tic_data["stats"]["rt_min"] == 0.0
        assert tic_data["stats"]["rt_max"] == 10.0
        assert tic_data["stats"]["intensity_max"] == 15000.0  # Peak at 6.5 min

    def test_read_chromatogram_custom_signal_type(self) -> None:
        """Test reading with custom signal type."""
        csv_file = _get_csv_file()
        reader = CSVChromatogramReader()
        tic_data = reader.read_chromatogram(csv_file, signal_type="EIC")

        assert tic_data["meta"]["signal_type"] == "EIC"
