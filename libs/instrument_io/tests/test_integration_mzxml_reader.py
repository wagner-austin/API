"""Integration test for MzMLReader with mzXML file.

Uses sample mzXML file from pyteomics test suite to verify that
MzMLReader.iter_spectra correctly handles mzXML format.

Test file source: https://github.com/levitsky/pyteomics/tree/master/tests

Fixtures:
- test.mzXML: Valid mzXML with 2 spectra
- tiny.mzXML: Invalid/corrupt file for error handling tests
"""

from __future__ import annotations

from pathlib import Path

import pytest

from instrument_io.readers.mzml import MzMLReader

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TINY_MZXML = FIXTURES_DIR / "tiny.mzXML"


def test_iter_spectra_reads_real_mzxml() -> None:
    """Test reading real mzXML file."""
    file = FIXTURES_DIR / "test.mzXML"

    reader = MzMLReader()
    spectra = list(reader.iter_spectra(file))

    # File contains 2 spectra
    assert len(spectra) == 2

    # Verify first spectrum
    sp = spectra[0]
    assert sp["meta"]["ms_level"] == 1
    assert sp["meta"]["scan_number"] == 19
    assert len(sp["data"]["mz_values"]) == 1313
    assert len(sp["data"]["intensities"]) == 1313

    # Verify first values match expected
    assert 400 < sp["data"]["mz_values"][0] < 401
    assert sp["data"]["intensities"][0] == 11411.0


def test_supports_format_mzxml() -> None:
    """Test format detection for mzXML files."""
    file = FIXTURES_DIR / "test.mzXML"

    reader = MzMLReader()
    assert reader.supports_format(file) is True


def test_count_spectra_mzxml() -> None:
    """Test counting spectra in mzXML file."""
    file = FIXTURES_DIR / "test.mzXML"

    reader = MzMLReader()
    count = reader.count_spectra(file)

    assert count == 2


def test_read_spectrum_by_index_mzxml() -> None:
    """Test reading specific spectrum by 1-based index from mzXML."""
    file = FIXTURES_DIR / "test.mzXML"

    reader = MzMLReader()
    # read_spectrum uses 1-based index, not scan number
    sp = reader.read_spectrum(file, scan_number=1)  # First spectrum

    assert sp["meta"]["scan_number"] == 19  # Actual scan number in file
    assert sp["meta"]["ms_level"] == 1
    assert len(sp["data"]["mz_values"]) == 1313


def test_read_tic_from_mzxml() -> None:
    """Test reading TIC from mzXML file."""
    file = FIXTURES_DIR / "test.mzXML"

    reader = MzMLReader()
    tic_data = reader.read_tic(file)

    # Verify structure
    assert tic_data["meta"]["signal_type"] == "TIC"
    assert tic_data["meta"]["detector"] == "MS"

    # File has 2 spectra, so 2 TIC points
    assert tic_data["stats"]["num_points"] == 2
    assert len(tic_data["data"]["retention_times"]) == 2
    assert len(tic_data["data"]["intensities"]) == 2

    # TIC values should be positive
    assert tic_data["data"]["intensities"][0] > 0


def test_read_eic_from_mzxml() -> None:
    """Test reading EIC from mzXML file."""
    file = FIXTURES_DIR / "test.mzXML"

    reader = MzMLReader()
    # Target m/z around 500 with tolerance
    eic_data = reader.read_eic(file, target_mz=500.0, mz_tolerance=10.0)

    # Verify structure
    assert eic_data["meta"]["signal_type"] == "EIC"
    assert eic_data["params"]["target_mz"] == 500.0
    assert eic_data["params"]["mz_tolerance"] == 10.0

    # File has 2 spectra, so 2 EIC points
    assert eic_data["stats"]["num_points"] == 2
    assert len(eic_data["data"]["retention_times"]) == 2


def test_iter_spectra_invalid_mzxml_raises() -> None:
    """Test that invalid mzXML file raises an XML parsing error.

    The reader propagates lxml parsing errors directly without wrapping,
    following the principle of explicit error propagation.
    """
    reader = MzMLReader()

    with pytest.raises(Exception) as exc_info:
        list(reader.iter_spectra(TINY_MZXML))

    # Verify it's an lxml XML syntax error
    assert "XMLSyntaxError" in type(exc_info.value).__name__


def test_count_spectra_invalid_mzxml_raises() -> None:
    """Test that count_spectra on invalid file raises XML parsing error."""
    reader = MzMLReader()

    with pytest.raises(Exception) as exc_info:
        reader.count_spectra(TINY_MZXML)

    # Verify it's an lxml XML syntax error
    assert "XMLSyntaxError" in type(exc_info.value).__name__
