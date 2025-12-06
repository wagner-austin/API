"""Integration tests for MGFReader using real MGF files.

Tests reading sample.mgf fixture which contains 3 MS/MS spectra:
- Spectrum 1: scan=100, precursor 500.25 m/z, charge 2+, RT 120.5s, 4 peaks
- Spectrum 2: scan=101, precursor 600.35 m/z, charge 3+, RT 180.0s, 3 peaks
- Spectrum 3: precursor 450.5 m/z, charge 1+, no RT, 2 peaks
"""

from __future__ import annotations

from pathlib import Path

import pytest

from instrument_io._exceptions import MGFReadError
from instrument_io.readers.mgf import MGFReader

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_MGF = FIXTURES_DIR / "sample.mgf"


def _get_mgf_file() -> Path:
    """Get path to sample.mgf, skip if not found."""
    if not SAMPLE_MGF.exists():
        pytest.skip("sample.mgf test fixture not found")
    return SAMPLE_MGF


class TestMGFReaderSupportsFormat:
    """Test MGFReader.supports_format method."""

    def test_supports_format_mgf_file(self) -> None:
        """Test format detection for MGF file."""
        path = _get_mgf_file()
        reader = MGFReader()
        assert reader.supports_format(path) is True

    def test_rejects_non_mgf_file(self) -> None:
        """Test format rejection for non-MGF file."""
        csv_file = FIXTURES_DIR / "chromatogram.csv"
        reader = MGFReader()
        assert reader.supports_format(csv_file) is False


class TestMGFReaderIterSpectra:
    """Test MGFReader.iter_spectra method."""

    def test_iter_spectra_reads_all(self) -> None:
        """Test iterating all spectra from MGF file."""
        path = _get_mgf_file()
        reader = MGFReader()
        spectra = list(reader.iter_spectra(path))

        # File contains 3 spectra
        assert len(spectra) == 3

    def test_iter_spectra_first_spectrum(self) -> None:
        """Test first spectrum data is correct."""
        path = _get_mgf_file()
        reader = MGFReader()
        spectra = list(reader.iter_spectra(path))

        sp = spectra[0]
        # Spectrum 1: scan=100, precursor 500.25, charge 2+
        assert sp["meta"]["ms_level"] == 2
        assert sp["precursor"]["mz"] == 500.25
        assert sp["precursor"]["charge"] == 2
        assert sp["precursor"]["intensity"] == 10000.0
        # RT 120.5s converted to minutes = 2.008 min
        assert 2.0 < sp["meta"]["retention_time"] < 2.1

        # 4 peaks: 100.1, 200.2, 300.3, 400.4
        assert sp["stats"]["num_peaks"] == 4
        assert len(sp["data"]["mz_values"]) == 4
        assert len(sp["data"]["intensities"]) == 4

    def test_iter_spectra_second_spectrum(self) -> None:
        """Test second spectrum data is correct."""
        path = _get_mgf_file()
        reader = MGFReader()
        spectra = list(reader.iter_spectra(path))

        sp = spectra[1]
        # Spectrum 2: scan=101, precursor 600.35, charge 3+
        assert sp["precursor"]["mz"] == 600.35
        assert sp["precursor"]["charge"] == 3
        # RT 180.0s converted to minutes = 3.0 min
        assert sp["meta"]["retention_time"] == 3.0

        # 3 peaks: 150.1, 250.2, 350.3
        assert sp["stats"]["num_peaks"] == 3

    def test_iter_spectra_third_spectrum_no_rt(self) -> None:
        """Test third spectrum with no retention time."""
        path = _get_mgf_file()
        reader = MGFReader()
        spectra = list(reader.iter_spectra(path))

        sp = spectra[2]
        # Spectrum 3: precursor 450.5, charge 1+, no RT
        assert sp["precursor"]["mz"] == 450.5
        assert sp["precursor"]["charge"] == 1
        assert sp["meta"]["retention_time"] == 0.0  # Default when missing

        # 2 peaks: 200.0, 300.0
        assert sp["stats"]["num_peaks"] == 2


class TestMGFReaderReadSpectrum:
    """Test MGFReader.read_spectrum method."""

    def test_read_spectrum_by_index(self) -> None:
        """Test reading specific spectrum by 0-based index."""
        path = _get_mgf_file()
        reader = MGFReader()

        # Read second spectrum (index 1)
        sp = reader.read_spectrum(path, index=1)

        assert sp["precursor"]["mz"] == 600.35
        assert sp["precursor"]["charge"] == 3
        assert sp["stats"]["num_peaks"] == 3

    def test_read_spectrum_first(self) -> None:
        """Test reading first spectrum."""
        path = _get_mgf_file()
        reader = MGFReader()

        sp = reader.read_spectrum(path, index=0)
        assert sp["precursor"]["mz"] == 500.25

    def test_read_spectrum_last(self) -> None:
        """Test reading last spectrum."""
        path = _get_mgf_file()
        reader = MGFReader()

        sp = reader.read_spectrum(path, index=2)
        assert sp["precursor"]["mz"] == 450.5

    def test_read_spectrum_invalid_index_raises(self) -> None:
        """Test that invalid index raises error."""
        path = _get_mgf_file()
        reader = MGFReader()

        with pytest.raises(MGFReadError) as exc_info:
            reader.read_spectrum(path, index=10)
        assert "not found" in str(exc_info.value).lower()


class TestMGFReaderCountSpectra:
    """Test MGFReader.count_spectra method."""

    def test_count_spectra(self) -> None:
        """Test counting spectra in MGF file."""
        path = _get_mgf_file()
        reader = MGFReader()

        count = reader.count_spectra(path)
        assert count == 3


class TestMGFReaderDataQuality:
    """Test data quality and consistency."""

    def test_mz_values_sorted(self) -> None:
        """Test that m/z values are in ascending order."""
        path = _get_mgf_file()
        reader = MGFReader()
        spectra = list(reader.iter_spectra(path))

        for sp in spectra:
            mz_values = sp["data"]["mz_values"]
            assert mz_values == sorted(mz_values), "m/z values should be sorted"

    def test_intensities_positive(self) -> None:
        """Test that intensity values are positive."""
        path = _get_mgf_file()
        reader = MGFReader()
        spectra = list(reader.iter_spectra(path))

        for sp in spectra:
            for intensity in sp["data"]["intensities"]:
                assert intensity > 0, "Intensity values should be positive"

    def test_stats_consistency(self) -> None:
        """Test that stats match actual data."""
        path = _get_mgf_file()
        reader = MGFReader()
        spectra = list(reader.iter_spectra(path))

        for sp in spectra:
            mz_values = sp["data"]["mz_values"]
            intensities = sp["data"]["intensities"]

            assert sp["stats"]["num_peaks"] == len(mz_values)
            assert sp["stats"]["mz_min"] == min(mz_values)
            assert sp["stats"]["mz_max"] == max(mz_values)
            assert sp["stats"]["base_peak_intensity"] == max(intensities)

            # Base peak m/z should correspond to max intensity
            max_idx = intensities.index(max(intensities))
            assert sp["stats"]["base_peak_mz"] == mz_values[max_idx]
