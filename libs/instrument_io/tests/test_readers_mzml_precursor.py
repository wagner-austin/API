"""Tests for readers mzml complete: ExtractPrecursorInfo."""

from __future__ import annotations

from instrument_io._protocols.pyteomics import SpectrumValue
from tests._mzml_helpers import SpectrumDict


class TestExtractPrecursorInfo:
    """Test _extract_precursor_info for all branches."""

    def test_extract_precursor_none_when_missing(self) -> None:
        """Test that None is returned when no precursor info."""
        from instrument_io.readers.mzml_spectrum import _extract_precursor_info

        spectrum = SpectrumDict({})
        result = _extract_precursor_info(spectrum)
        assert result is None

    def test_extract_precursor_from_precursormz_list(self) -> None:
        """Test extraction from precursorMz as list (mzXML style)."""
        from instrument_io.readers.mzml_spectrum import _extract_precursor_info

        spectrum = SpectrumDict({"precursorMz": [500.5]})
        result = _extract_precursor_info(spectrum)

        assert result
        assert result["mz"] == 500.5
        assert result["charge"] is None
        assert result["intensity"] is None

    def test_extract_precursor_from_precursormz_float(self) -> None:
        """Test extraction from precursorMz as float."""
        from instrument_io.readers.mzml_spectrum import _extract_precursor_info

        spectrum = SpectrumDict({"precursorMz": 300.25})
        result = _extract_precursor_info(spectrum)

        assert result
        assert result["mz"] == 300.25

    def test_extract_precursor_from_precursormz_invalid(self) -> None:
        """Test extraction from precursorMz with invalid type."""
        from instrument_io.readers.mzml_spectrum import _extract_precursor_info

        spectrum = SpectrumDict({"precursorMz": "invalid"})
        result = _extract_precursor_info(spectrum)

        assert result
        assert result["mz"] == 0.0

    def test_extract_precursor_from_mzml_style(self) -> None:
        """Test extraction from precursor list (mzML style)."""
        from instrument_io.readers.mzml_spectrum import _extract_precursor_info

        precursor_ion: dict[str, SpectrumValue] = {
            "selected ion m/z": 450.75,
            "charge state": 2,
            "peak intensity": 1000.0,
        }
        precursor_entry: dict[str, SpectrumValue] = {"selectedIons": [precursor_ion]}
        spectrum = SpectrumDict({"precursor": [precursor_entry]})
        result = _extract_precursor_info(spectrum)

        assert result
        assert result["mz"] == 450.75
        assert result["charge"] == 2
        assert result["intensity"] == 1000.0

    def test_extract_precursor_empty_precursor_list(self) -> None:
        """Test line 252: return None when precursor list is empty."""
        from instrument_io.readers.mzml_spectrum import _extract_precursor_info

        # precursor key exists but list is empty
        spectrum = SpectrumDict({"precursor": []})
        result = _extract_precursor_info(spectrum)
        assert result is None

    def test_extract_precursor_first_entry_not_dict(self) -> None:
        """Test line 252: return None when first precursor entry is not dict."""
        from instrument_io.readers.mzml_spectrum import _extract_precursor_info

        # precursor list has entry but it's not a dict
        spectrum = SpectrumDict({"precursor": ["not a dict"]})
        result = _extract_precursor_info(spectrum)
        assert result is None

    def test_extract_precursor_missing_selected_ions(self) -> None:
        """Test line 252: return None when selectedIons is missing."""
        from instrument_io.readers.mzml_spectrum import _extract_precursor_info

        # precursor entry is dict but has no selectedIons
        precursor_entry: dict[str, SpectrumValue] = {"isolationWindow": {}}
        spectrum = SpectrumDict({"precursor": [precursor_entry]})
        result = _extract_precursor_info(spectrum)
        assert result is None

    def test_extract_precursor_empty_selected_ions(self) -> None:
        """Test line 252: return None when selectedIons is empty list."""
        from instrument_io.readers.mzml_spectrum import _extract_precursor_info

        # selectedIons exists but is empty
        precursor_entry: dict[str, SpectrumValue] = {"selectedIons": []}
        spectrum = SpectrumDict({"precursor": [precursor_entry]})
        result = _extract_precursor_info(spectrum)
        assert result is None

    def test_extract_precursor_ion_not_dict(self) -> None:
        """Test line 252: return None when ion entry is not dict."""
        from instrument_io.readers.mzml_spectrum import _extract_precursor_info

        # selectedIons has entry but it's not a dict
        precursor_entry: dict[str, SpectrumValue] = {"selectedIons": ["not a dict"]}
        spectrum = SpectrumDict({"precursor": [precursor_entry]})
        result = _extract_precursor_info(spectrum)
        assert result is None

    def test_extract_precursor_precursormz_list_invalid_first_item(self) -> None:
        """Test line 214: precursorMz list with non-numeric first item."""
        from instrument_io.readers.mzml_spectrum import _extract_precursor_info

        # precursorMz is list but first item is not numeric
        spectrum = SpectrumDict({"precursorMz": ["not a number"]})
        result = _extract_precursor_info(spectrum)

        assert result
        assert result["mz"] == 0.0

    def test_extract_precursor_precursormz_empty_list(self) -> None:
        """Test precursorMz as empty list goes to else branch (line 214).

        When precursorMz is an empty list, the condition `len(precursor_mz) > 0`
        is False, so it falls to the else branch which sets mz_val = 0.0 and
        returns a PrecursorInfo with mz=0.0.
        """
        from instrument_io.readers.mzml_spectrum import _extract_precursor_info

        # precursorMz is empty list - goes to else branch, returns mz=0.0
        spectrum = SpectrumDict({"precursorMz": []})
        result = _extract_precursor_info(spectrum)

        # Returns PrecursorInfo with mz=0.0 - verify expected values
        assert result
        assert result["mz"] == 0.0
        assert result["charge"] is None
        assert result["intensity"] is None
