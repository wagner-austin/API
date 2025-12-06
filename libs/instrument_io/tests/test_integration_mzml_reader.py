"""Integration tests for MzMLReader using real mzML files.

Uses sample mzML file from HUPO-PSI/mzML repository to verify that
MzMLReader.iter_spectra returns typed data with arrays and metadata
decoded correctly.

Test file source: https://github.com/HUPO-PSI/mzML/tree/master/examples
"""

from __future__ import annotations

from pathlib import Path

from instrument_io.readers.mzml import MzMLReader

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_iter_spectra_reads_real_mzml() -> None:
    """Test reading real mzML file with multiple spectra."""
    file = FIXTURES_DIR / "tiny.pwiz.1.1.mzML"

    reader = MzMLReader()
    spectra = list(reader.iter_spectra(file))

    # File contains 4 spectra
    assert len(spectra) == 4

    # Verify first spectrum structure
    sp = spectra[0]
    assert sp["meta"]["ms_level"] == 1
    assert sp["meta"]["scan_number"] == 19  # cycle=19 in Waters format
    assert len(sp["data"]["mz_values"]) == 15
    assert len(sp["data"]["intensities"]) == 15

    # Verify first few values are valid (test data starts at m/z 0.0)
    assert sp["data"]["mz_values"][0] == 0.0
    assert sp["data"]["intensities"][0] == 15.0


def test_read_spectrum_by_index() -> None:
    """Test reading specific spectrum by 1-based index."""
    file = FIXTURES_DIR / "tiny.pwiz.1.1.mzML"

    reader = MzMLReader()
    # read_spectrum uses 1-based index, not scan number
    # File has 4 spectra with scan numbers (from cycle values): 19, 20, 21, 22
    sp = reader.read_spectrum(file, scan_number=1)  # First spectrum

    assert sp["meta"]["scan_number"] == 19  # Cycle value from Waters format
    assert sp["meta"]["ms_level"] == 1
    # First spectrum has 15 peaks
    assert len(sp["data"]["mz_values"]) == 15


def test_count_spectra() -> None:
    """Test counting spectra in file."""
    file = FIXTURES_DIR / "tiny.pwiz.1.1.mzML"

    reader = MzMLReader()
    count = reader.count_spectra(file)

    assert count == 4


def test_supports_format() -> None:
    """Test format detection for mzML files."""
    file = FIXTURES_DIR / "tiny.pwiz.1.1.mzML"

    reader = MzMLReader()
    assert reader.supports_format(file) is True


def test_spectrum_stats_computed() -> None:
    """Test that spectrum statistics are computed correctly."""
    file = FIXTURES_DIR / "tiny.pwiz.1.1.mzML"

    reader = MzMLReader()
    spectra = list(reader.iter_spectra(file))
    sp = spectra[0]

    stats = sp["stats"]
    assert stats["num_peaks"] == 15
    # Test data has m/z values from 0.0 to 14.0
    assert stats["mz_min"] == 0.0
    assert stats["mz_max"] == 14.0
    # Base peak is at m/z 0.0 with intensity 15.0
    assert stats["base_peak_intensity"] == 15.0
    assert stats["base_peak_mz"] == 0.0


def test_read_tic_from_mzml() -> None:
    """Test reading TIC from mzML file."""
    file = FIXTURES_DIR / "tiny.pwiz.1.1.mzML"

    reader = MzMLReader()
    tic_data = reader.read_tic(file)

    # Verify structure
    assert tic_data["meta"]["signal_type"] == "TIC"
    assert tic_data["meta"]["detector"] == "MS"
    assert str(file) in tic_data["meta"]["source_path"]

    # File has 4 spectra, so 4 TIC points
    assert tic_data["stats"]["num_points"] == 4
    assert len(tic_data["data"]["retention_times"]) == 4
    assert len(tic_data["data"]["intensities"]) == 4

    # TIC values from metadata should be positive
    assert tic_data["data"]["intensities"][0] > 0
    # All spectra have same TIC value in this test file
    assert tic_data["stats"]["intensity_max"] > 0


def test_read_eic_from_mzml() -> None:
    """Test reading EIC from mzML file."""
    file = FIXTURES_DIR / "tiny.pwiz.1.1.mzML"

    reader = MzMLReader()
    # Target m/z 5.0 with tolerance 2.0 should capture m/z 3, 4, 5, 6, 7
    eic_data = reader.read_eic(file, target_mz=5.0, mz_tolerance=2.0)

    # Verify structure
    assert eic_data["meta"]["signal_type"] == "EIC"
    assert eic_data["params"]["target_mz"] == 5.0
    assert eic_data["params"]["mz_tolerance"] == 2.0

    # File has 4 spectra, so 4 EIC points
    assert eic_data["stats"]["num_points"] == 4
    assert len(eic_data["data"]["retention_times"]) == 4
    assert len(eic_data["data"]["intensities"]) == 4

    # m/z values 3,4,5,6,7 have intensities 12,11,10,9,8 (15-mz)
    # Sum = 12+11+10+9+8 = 50
    assert eic_data["data"]["intensities"][0] == 50.0


def test_read_eic_no_match() -> None:
    """Test EIC with m/z range that has no matches."""
    file = FIXTURES_DIR / "tiny.pwiz.1.1.mzML"

    reader = MzMLReader()
    # Target m/z 100.0 is outside the data range (0-14)
    eic_data = reader.read_eic(file, target_mz=100.0, mz_tolerance=1.0)

    # Should still return data, just with zero intensities
    assert eic_data["stats"]["num_points"] == 4
    assert all(i == 0.0 for i in eic_data["data"]["intensities"])
