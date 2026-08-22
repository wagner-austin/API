"""Tests for readers mzml complete: ExtractArrayErrors."""

from __future__ import annotations

from pathlib import Path

import pytest

from instrument_io._exceptions import MzMLReadError
from tests._mzml_helpers import SpectrumDict


class TestExtractArrayErrors:
    """Test error paths in array extraction."""

    def test_missing_mz_array_raises(self, tmp_path: Path) -> None:
        """Test that missing m/z array raises MzMLReadError."""
        from instrument_io.readers.mzml_spectrum import _extract_array_from_spectrum

        # Create a minimal spectrum dict without m/z array
        spectrum = SpectrumDict({"intensity array": [1.0, 2.0, 3.0]})

        with pytest.raises(MzMLReadError) as exc_info:
            _extract_array_from_spectrum(spectrum, "m/z array", "/test.mzML")

        assert "Missing required array" in str(exc_info.value)
        assert "m/z array" in str(exc_info.value)

    def test_invalid_array_type_raises(self, tmp_path: Path) -> None:
        """Test that non-array value raises MzMLReadError."""
        from instrument_io.readers.mzml_spectrum import _extract_array_from_spectrum

        # Create spectrum with non-array value
        spectrum = SpectrumDict({"m/z array": "not an array"})

        with pytest.raises(MzMLReadError) as exc_info:
            _extract_array_from_spectrum(spectrum, "m/z array", "/test.mzML")

        assert "Expected array" in str(exc_info.value)


class TestExtractFloatOrZero:
    """Test _extract_float_or_zero for all branches."""

    def test_extract_float_or_zero_with_none(self) -> None:
        """Test extraction when value is None."""
        from instrument_io.readers.mzml_spectrum import _extract_float_or_zero

        spectrum = SpectrumDict({})
        result = _extract_float_or_zero(spectrum, "missing_key")
        assert result == 0.0

    def test_extract_float_or_zero_with_int(self) -> None:
        """Test extraction with int value."""
        from instrument_io.readers.mzml_spectrum import _extract_float_or_zero

        spectrum = SpectrumDict({"value": 42})
        result = _extract_float_or_zero(spectrum, "value")
        assert result == 42.0

    def test_extract_float_or_zero_with_float(self) -> None:
        """Test extraction with float value."""
        from instrument_io.readers.mzml_spectrum import _extract_float_or_zero

        spectrum = SpectrumDict({"value": 3.14})
        result = _extract_float_or_zero(spectrum, "value")
        assert result == 3.14

    def test_extract_float_or_zero_with_non_numeric(self) -> None:
        """Test extraction with non-numeric value returns 0.0."""
        from instrument_io.readers.mzml_spectrum import _extract_float_or_zero

        spectrum = SpectrumDict({"value": "not a number"})
        result = _extract_float_or_zero(spectrum, "value")
        assert result == 0.0


class TestExtractPolarityString:
    """Test _extract_polarity_string for all branches."""

    def test_extract_polarity_positive_string(self) -> None:
        """Test extraction of 'positive' string."""
        from instrument_io.readers.mzml_spectrum import _extract_polarity_string

        spectrum = SpectrumDict({"positive scan": "positive"})
        result = _extract_polarity_string(spectrum)
        assert result == "positive"

    def test_extract_polarity_negative_string(self) -> None:
        """Test extraction of 'negative' string."""
        from instrument_io.readers.mzml_spectrum import _extract_polarity_string

        spectrum = SpectrumDict({"negative scan": "negative"})
        result = _extract_polarity_string(spectrum)
        assert result == "negative"

    def test_extract_polarity_bool_true_positive(self) -> None:
        """Test extraction when 'positive scan' is True."""
        from instrument_io.readers.mzml_spectrum import _extract_polarity_string

        spectrum = SpectrumDict({"positive scan": True})
        result = _extract_polarity_string(spectrum)
        assert result == "positive"

    def test_extract_polarity_bool_true_negative(self) -> None:
        """Test extraction when 'negative scan' is True."""
        from instrument_io.readers.mzml_spectrum import _extract_polarity_string

        spectrum = SpectrumDict({"negative scan": True})
        result = _extract_polarity_string(spectrum)
        assert result == "negative"

    def test_extract_polarity_none(self) -> None:
        """Test extraction when no polarity info present."""
        from instrument_io.readers.mzml_spectrum import _extract_polarity_string

        spectrum = SpectrumDict({})
        result = _extract_polarity_string(spectrum)
        assert result is None

    def test_extract_polarity_using_polarity_key(self) -> None:
        """Test extraction using 'polarity' key."""
        from instrument_io.readers.mzml_spectrum import _extract_polarity_string

        spectrum = SpectrumDict({"polarity": "positive"})
        result = _extract_polarity_string(spectrum)
        assert result == "positive"

    def test_extract_polarity_bool_true_with_polarity_key(self) -> None:
        """Test polarity key with bool True - covers branch 183->178, 186->178.

        When 'polarity' key has bool True, neither 'positive' nor 'negative'
        is in the key name, so the code continues to next key iteration.
        This exercises the branch where bool is True but key doesn't contain
        positive/negative, returning None.
        """
        from instrument_io.readers.mzml_spectrum import _extract_polarity_string

        # polarity key with bool True - doesn't match positive/negative in key
        spectrum = SpectrumDict({"polarity": True})
        result = _extract_polarity_string(spectrum)
        # Returns None because "positive" and "negative" not in "polarity"
        assert result is None

    def test_extract_polarity_bool_false_skipped(self) -> None:
        """Test that bool False values are skipped."""
        from instrument_io.readers.mzml_spectrum import _extract_polarity_string

        # When value is bool but False, condition `isinstance(value, bool) and value`
        # is False, so we continue to next key
        spectrum = SpectrumDict({"positive scan": False, "negative scan": "negative"})
        result = _extract_polarity_string(spectrum)
        assert result == "negative"
