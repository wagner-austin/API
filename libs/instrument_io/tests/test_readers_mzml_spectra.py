"""Tests for readers mzml complete: SpectrumDict."""

from __future__ import annotations

from pathlib import Path

import pytest

from instrument_io._exceptions import MzMLReadError
from instrument_io.readers.mzml import MzMLReader

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestMzMLReaderIterMS2Spectra:
    """Test iter_ms2_spectra method - currently uncovered."""

    def test_iter_ms2_spectra_raises_without_precursor(self) -> None:
        """Test that iter_ms2_spectra raises MzMLReadError when MS2 lacks precursor info."""
        reader = MzMLReader()
        path = FIXTURES_DIR / "small.pwiz.1.1.mzML"

        # small.pwiz.1.1.mzML has MS2 spectra without precursor info
        # This should raise MzMLReadError when trying to convert the first MS2
        with pytest.raises(MzMLReadError) as exc_info:
            list(reader.iter_ms2_spectra(path))

        assert "No precursor info found" in str(exc_info.value)

    def test_iter_ms2_spectra_from_tiny_pwiz(self) -> None:
        """Test iter_ms2_spectra with tiny.pwiz.1.1.mzML (has 1 MS2 spectrum)."""
        reader = MzMLReader()
        path = FIXTURES_DIR / "tiny.pwiz.1.1.mzML"

        # tiny.pwiz.1.1.mzML has 1 MS2 but no precursor info
        with pytest.raises(MzMLReadError) as exc_info:
            list(reader.iter_ms2_spectra(path))

        assert "No precursor info found" in str(exc_info.value)

    def test_iter_ms2_spectra_with_mzxml(self) -> None:
        """Test iter_ms2_spectra mzXML branch by calling with mzXML file.

        Verifies that the mzXML code path (lines 462-468) executes correctly.
        test.mzXML has 1 MS2 spectrum (scan 20) with precursorMz element.
        """
        reader = MzMLReader()
        path = FIXTURES_DIR / "test.mzXML"

        # test.mzXML has MS2 spectra - consume the iterator to execute mzXML branch
        # This will execute lines 462-468 (mzXML branch of iter_ms2_spectra)
        ms2_spectra = list(reader.iter_ms2_spectra(path))

        # test.mzXML has 1 MS2 spectrum with precursor info
        assert len(ms2_spectra) == 1
        assert ms2_spectra[0]["meta"]["ms_level"] == 2
        # Precursor info extracted from mzXML precursorMz element
        assert ms2_spectra[0]["precursor"]["mz"] >= 0.0  # Has precursor info

    def test_iter_ms2_spectra_unsupported_format(self, tmp_path: Path) -> None:
        """Test iter_ms2_spectra with unsupported file type."""
        reader = MzMLReader()
        path = tmp_path / "test.csv"
        path.write_text("not,mzml")

        with pytest.raises(MzMLReadError) as exc_info:
            list(reader.iter_ms2_spectra(path))

        assert "Unsupported format" in str(exc_info.value)

    def test_iter_ms2_spectra_empty_file(self) -> None:
        """Test branch 457->exit: iter_ms2_spectra with file containing no spectra.

        Tests that generator exits cleanly when file has no spectra at all.
        """
        reader = MzMLReader()
        path = FIXTURES_DIR / "empty.mzML"

        # empty.mzML has 0 spectra, so the generator yields nothing
        ms2_spectra = list(reader.iter_ms2_spectra(path))

        # Should return empty list (no MS2 spectra found)
        assert ms2_spectra == []


class TestMzMLReaderReadTICEdgeCases:
    """Test read_tic edge cases and error paths."""

    def test_read_tic_unsupported_format(self, tmp_path: Path) -> None:
        """Test read_tic with unsupported file format."""
        reader = MzMLReader()
        path = tmp_path / "test.csv"
        path.write_text("not,mzml")

        with pytest.raises(MzMLReadError) as exc_info:
            reader.read_tic(path)

        assert "Unsupported format" in str(exc_info.value)

    def test_read_tic_computes_from_intensities_when_zero(self) -> None:
        """Test that TIC is computed from intensities when total_ion_current is 0."""
        reader = MzMLReader()
        path = FIXTURES_DIR / "tiny.pwiz.1.1.mzML"

        tic_data = reader.read_tic(path)

        # Verify TIC was computed
        assert tic_data["stats"]["num_points"] == 4
        assert all(i >= 0.0 for i in tic_data["data"]["intensities"])


class TestMzMLReaderReadEICEdgeCases:
    """Test read_eic edge cases and error paths."""

    def test_read_eic_unsupported_format(self, tmp_path: Path) -> None:
        """Test read_eic with unsupported file format."""
        reader = MzMLReader()
        path = tmp_path / "test.csv"
        path.write_text("not,mzml")

        with pytest.raises(MzMLReadError) as exc_info:
            reader.read_eic(path, target_mz=100.0, mz_tolerance=1.0)

        assert "Unsupported format" in str(exc_info.value)
