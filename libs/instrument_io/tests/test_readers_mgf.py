"""Tests for MGFReader class."""

from __future__ import annotations

from pathlib import Path

import pytest

from instrument_io._exceptions import MGFReadError
from instrument_io.readers.mgf import (
    MGFReader,
    _is_mgf_file,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestIsMgfFile:
    """Tests for _is_mgf_file function."""

    def test_returns_true_for_mgf(self) -> None:
        """Test returns True for .mgf file."""
        file = FIXTURES_DIR / "sample.mgf"
        assert _is_mgf_file(file) is True

    def test_returns_false_for_non_mgf(self) -> None:
        """Test returns False for non-.mgf file."""
        file = FIXTURES_DIR / "tiny.pwiz.1.1.mzML"
        assert _is_mgf_file(file) is False

    def test_returns_false_for_directory(self) -> None:
        """Test returns False for directory."""
        assert _is_mgf_file(FIXTURES_DIR) is False

    def test_returns_false_for_nonexistent(self) -> None:
        """Test returns False for nonexistent file."""
        file = FIXTURES_DIR / "nonexistent.mgf"
        assert _is_mgf_file(file) is False


class TestMGFReaderSupportsFormat:
    """Tests for MGFReader.supports_format method."""

    def test_supports_mgf_file(self) -> None:
        """Test supports_format returns True for .mgf file."""
        reader = MGFReader()
        file = FIXTURES_DIR / "sample.mgf"
        assert reader.supports_format(file) is True

    def test_rejects_non_mgf_file(self) -> None:
        """Test supports_format returns False for non-.mgf file."""
        reader = MGFReader()
        file = FIXTURES_DIR / "tiny.pwiz.1.1.mzML"
        assert reader.supports_format(file) is False


class TestMGFReaderIterSpectra:
    """Tests for MGFReader.iter_spectra method."""

    def test_iterates_over_spectra(self) -> None:
        """Test iterating over all spectra in file."""
        reader = MGFReader()
        file = FIXTURES_DIR / "sample.mgf"

        spectra = list(reader.iter_spectra(file))

        assert len(spectra) == 3
        assert all(s["meta"]["ms_level"] == 2 for s in spectra)

    def test_raises_for_non_mgf_file(self) -> None:
        """Test raises MGFReadError for non-.mgf file."""
        reader = MGFReader()
        file = FIXTURES_DIR / "tiny.pwiz.1.1.mzML"

        with pytest.raises(MGFReadError, match="Not an MGF file"):
            list(reader.iter_spectra(file))

    def test_yields_ms2_spectra(self) -> None:
        """Test iter_spectra yields MS2Spectrum with expected fields."""
        reader = MGFReader()
        file = FIXTURES_DIR / "sample.mgf"

        result = reader.iter_spectra(file)
        first_spectrum = next(result)

        # Verify structure
        assert first_spectrum["meta"]["ms_level"] == 2
        assert first_spectrum["meta"]["source_path"] == str(file)
        assert first_spectrum["precursor"]["mz"] == 500.25
        assert first_spectrum["precursor"]["intensity"] == 10000.0
        assert first_spectrum["precursor"]["charge"] == 2
        assert len(first_spectrum["data"]["mz_values"]) == 4
        assert first_spectrum["stats"]["num_peaks"] == 4

    def test_spectrum_data_values(self) -> None:
        """Test spectrum data contains expected values."""
        reader = MGFReader()
        file = FIXTURES_DIR / "sample.mgf"

        spectra = list(reader.iter_spectra(file))
        first = spectra[0]

        # First spectrum from fixture has these peaks
        assert first["data"]["mz_values"] == [100.1, 200.2, 300.3, 400.4]
        assert first["data"]["intensities"] == [500.0, 1000.0, 750.0, 250.0]

    def test_retention_time_decoded(self) -> None:
        """Test retention time is decoded from rtinseconds."""
        reader = MGFReader()
        file = FIXTURES_DIR / "sample.mgf"

        spectra = list(reader.iter_spectra(file))

        # First spectrum: rtinseconds=120.5 -> 2.0083... minutes
        assert 2.0 < spectra[0]["meta"]["retention_time"] < 2.1
        # Second spectrum: rtinseconds=180.0 -> 3.0 minutes
        assert spectra[1]["meta"]["retention_time"] == 3.0

    def test_scan_number_from_title(self) -> None:
        """Test scan number is extracted from title."""
        reader = MGFReader()
        file = FIXTURES_DIR / "sample.mgf"

        spectra = list(reader.iter_spectra(file))

        # First spectrum title contains "scan=100"
        assert spectra[0]["meta"]["scan_number"] == 100
        # Second spectrum title contains "scan=101"
        assert spectra[1]["meta"]["scan_number"] == 101
        # Third spectrum has no scan in title, uses index+1
        assert spectra[2]["meta"]["scan_number"] == 3


class TestMGFReaderReadSpectrum:
    """Tests for MGFReader.read_spectrum method."""

    def test_reads_spectrum_by_index(self) -> None:
        """Test reading spectrum by 0-based index."""
        reader = MGFReader()
        file = FIXTURES_DIR / "sample.mgf"

        spectrum = reader.read_spectrum(file, 0)

        assert spectrum["meta"]["ms_level"] == 2
        assert spectrum["precursor"]["mz"] == 500.25

    def test_reads_second_spectrum(self) -> None:
        """Test reading second spectrum."""
        reader = MGFReader()
        file = FIXTURES_DIR / "sample.mgf"

        spectrum = reader.read_spectrum(file, 1)

        assert spectrum["precursor"]["mz"] == 600.35

    def test_reads_third_spectrum(self) -> None:
        """Test reading third spectrum."""
        reader = MGFReader()
        file = FIXTURES_DIR / "sample.mgf"

        spectrum = reader.read_spectrum(file, 2)

        assert spectrum["precursor"]["mz"] == 450.5

    def test_raises_for_invalid_index(self) -> None:
        """Test raises MGFReadError for negative index."""
        reader = MGFReader()
        file = FIXTURES_DIR / "sample.mgf"

        with pytest.raises(MGFReadError, match="Invalid index"):
            reader.read_spectrum(file, -1)

    def test_raises_for_out_of_range_index(self) -> None:
        """Test raises MGFReadError for out-of-range index."""
        reader = MGFReader()
        file = FIXTURES_DIR / "sample.mgf"

        with pytest.raises(MGFReadError, match="not found"):
            reader.read_spectrum(file, 100)

    def test_raises_for_non_mgf_file(self) -> None:
        """Test raises MGFReadError for non-.mgf file."""
        reader = MGFReader()
        file = FIXTURES_DIR / "tiny.pwiz.1.1.mzML"

        with pytest.raises(MGFReadError, match="Not an MGF file"):
            reader.read_spectrum(file, 0)


class TestMGFReaderCountSpectra:
    """Tests for MGFReader.count_spectra method."""

    def test_counts_spectra_correctly(self) -> None:
        """Test counting spectra in file."""
        reader = MGFReader()
        file = FIXTURES_DIR / "sample.mgf"

        count = reader.count_spectra(file)

        assert count == 3

    def test_raises_for_non_mgf_file(self) -> None:
        """Test raises MGFReadError for non-.mgf file."""
        reader = MGFReader()
        file = FIXTURES_DIR / "tiny.pwiz.1.1.mzML"

        with pytest.raises(MGFReadError, match="Not an MGF file"):
            reader.count_spectra(file)
