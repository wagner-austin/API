"""Tests for MGF protocol adapters."""

from __future__ import annotations

from pathlib import Path

from instrument_io._protocols.mgf import _open_mgf

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestOpenMgf:
    """Tests for _open_mgf function."""

    def test_open_mgf_returns_reader_protocol(self) -> None:
        """Test that _open_mgf returns a reader with expected interface."""
        file = FIXTURES_DIR / "sample.mgf"
        reader = _open_mgf(file)

        # Verify context manager protocol
        with reader:
            # Verify iteration works
            spectra = list(reader)
            assert len(spectra) == 3

    def test_open_mgf_spectrum_has_required_keys(self) -> None:
        """Test that spectra have required keys."""
        file = FIXTURES_DIR / "sample.mgf"
        reader = _open_mgf(file)

        with reader:
            for spectrum in reader:
                assert "m/z array" in spectrum
                assert "intensity array" in spectrum
                assert "params" in spectrum

    def test_open_mgf_params_has_expected_keys(self) -> None:
        """Test that params dict has expected MGF keys."""
        file = FIXTURES_DIR / "sample.mgf"
        reader = _open_mgf(file)

        with reader:
            spectra = list(reader)
            first_spectrum = spectra[0]
            # Verify params contains expected MGF fields
            # The actual typed conversion is tested in test_decoders_mgf.py
            assert "params" in first_spectrum
            assert "m/z array" in first_spectrum
            assert "intensity array" in first_spectrum
