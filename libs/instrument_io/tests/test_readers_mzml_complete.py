"""Complete coverage tests for MzMLReader focusing on uncovered paths."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from instrument_io._exceptions import MzMLReadError
from instrument_io._protocols.pyteomics import SpectrumValue
from instrument_io.readers.mzml import MzMLReader

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class SpectrumDict:
    """Test helper implementing SpectrumDictProtocol."""

    def __init__(self, data: dict[str, SpectrumValue]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> SpectrumValue:
        return self._data[key]

    def get(self, key: str) -> SpectrumValue:
        return self._data.get(key)

    def keys(self) -> Generator[str, None, None]:
        yield from self._data.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._data


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


class TestExtractArrayErrors:
    """Test error paths in array extraction."""

    def test_missing_mz_array_raises(self, tmp_path: Path) -> None:
        """Test that missing m/z array raises MzMLReadError."""
        from instrument_io.readers.mzml import _extract_array_from_spectrum

        # Create a minimal spectrum dict without m/z array
        spectrum = SpectrumDict({"intensity array": [1.0, 2.0, 3.0]})

        with pytest.raises(MzMLReadError) as exc_info:
            _extract_array_from_spectrum(spectrum, "m/z array", "/test.mzML")

        assert "Missing required array" in str(exc_info.value)
        assert "m/z array" in str(exc_info.value)

    def test_invalid_array_type_raises(self, tmp_path: Path) -> None:
        """Test that non-array value raises MzMLReadError."""
        from instrument_io.readers.mzml import _extract_array_from_spectrum

        # Create spectrum with non-array value
        spectrum = SpectrumDict({"m/z array": "not an array"})

        with pytest.raises(MzMLReadError) as exc_info:
            _extract_array_from_spectrum(spectrum, "m/z array", "/test.mzML")

        assert "Expected array" in str(exc_info.value)


class TestExtractFloatOrZero:
    """Test _extract_float_or_zero for all branches."""

    def test_extract_float_or_zero_with_none(self) -> None:
        """Test extraction when value is None."""
        from instrument_io.readers.mzml import _extract_float_or_zero

        spectrum = SpectrumDict({})
        result = _extract_float_or_zero(spectrum, "missing_key")
        assert result == 0.0

    def test_extract_float_or_zero_with_int(self) -> None:
        """Test extraction with int value."""
        from instrument_io.readers.mzml import _extract_float_or_zero

        spectrum = SpectrumDict({"value": 42})
        result = _extract_float_or_zero(spectrum, "value")
        assert result == 42.0

    def test_extract_float_or_zero_with_float(self) -> None:
        """Test extraction with float value."""
        from instrument_io.readers.mzml import _extract_float_or_zero

        spectrum = SpectrumDict({"value": 3.14})
        result = _extract_float_or_zero(spectrum, "value")
        assert result == 3.14

    def test_extract_float_or_zero_with_non_numeric(self) -> None:
        """Test extraction with non-numeric value returns 0.0."""
        from instrument_io.readers.mzml import _extract_float_or_zero

        spectrum = SpectrumDict({"value": "not a number"})
        result = _extract_float_or_zero(spectrum, "value")
        assert result == 0.0


class TestExtractPolarityString:
    """Test _extract_polarity_string for all branches."""

    def test_extract_polarity_positive_string(self) -> None:
        """Test extraction of 'positive' string."""
        from instrument_io.readers.mzml import _extract_polarity_string

        spectrum = SpectrumDict({"positive scan": "positive"})
        result = _extract_polarity_string(spectrum)
        assert result == "positive"

    def test_extract_polarity_negative_string(self) -> None:
        """Test extraction of 'negative' string."""
        from instrument_io.readers.mzml import _extract_polarity_string

        spectrum = SpectrumDict({"negative scan": "negative"})
        result = _extract_polarity_string(spectrum)
        assert result == "negative"

    def test_extract_polarity_bool_true_positive(self) -> None:
        """Test extraction when 'positive scan' is True."""
        from instrument_io.readers.mzml import _extract_polarity_string

        spectrum = SpectrumDict({"positive scan": True})
        result = _extract_polarity_string(spectrum)
        assert result == "positive"

    def test_extract_polarity_bool_true_negative(self) -> None:
        """Test extraction when 'negative scan' is True."""
        from instrument_io.readers.mzml import _extract_polarity_string

        spectrum = SpectrumDict({"negative scan": True})
        result = _extract_polarity_string(spectrum)
        assert result == "negative"

    def test_extract_polarity_none(self) -> None:
        """Test extraction when no polarity info present."""
        from instrument_io.readers.mzml import _extract_polarity_string

        spectrum = SpectrumDict({})
        result = _extract_polarity_string(spectrum)
        assert result is None

    def test_extract_polarity_using_polarity_key(self) -> None:
        """Test extraction using 'polarity' key."""
        from instrument_io.readers.mzml import _extract_polarity_string

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
        from instrument_io.readers.mzml import _extract_polarity_string

        # polarity key with bool True - doesn't match positive/negative in key
        spectrum = SpectrumDict({"polarity": True})
        result = _extract_polarity_string(spectrum)
        # Returns None because "positive" and "negative" not in "polarity"
        assert result is None

    def test_extract_polarity_bool_false_skipped(self) -> None:
        """Test that bool False values are skipped."""
        from instrument_io.readers.mzml import _extract_polarity_string

        # When value is bool but False, condition `isinstance(value, bool) and value`
        # is False, so we continue to next key
        spectrum = SpectrumDict({"positive scan": False, "negative scan": "negative"})
        result = _extract_polarity_string(spectrum)
        assert result == "negative"


class TestExtractPrecursorInfo:
    """Test _extract_precursor_info for all branches."""

    def test_extract_precursor_none_when_missing(self) -> None:
        """Test that None is returned when no precursor info."""
        from instrument_io.readers.mzml import _extract_precursor_info

        spectrum = SpectrumDict({})
        result = _extract_precursor_info(spectrum)
        assert result is None

    def test_extract_precursor_from_precursormz_list(self) -> None:
        """Test extraction from precursorMz as list (mzXML style)."""
        from instrument_io.readers.mzml import _extract_precursor_info

        spectrum = SpectrumDict({"precursorMz": [500.5]})
        result = _extract_precursor_info(spectrum)

        assert result
        assert result["mz"] == 500.5
        assert result["charge"] is None
        assert result["intensity"] is None

    def test_extract_precursor_from_precursormz_float(self) -> None:
        """Test extraction from precursorMz as float."""
        from instrument_io.readers.mzml import _extract_precursor_info

        spectrum = SpectrumDict({"precursorMz": 300.25})
        result = _extract_precursor_info(spectrum)

        assert result
        assert result["mz"] == 300.25

    def test_extract_precursor_from_precursormz_invalid(self) -> None:
        """Test extraction from precursorMz with invalid type."""
        from instrument_io.readers.mzml import _extract_precursor_info

        spectrum = SpectrumDict({"precursorMz": "invalid"})
        result = _extract_precursor_info(spectrum)

        assert result
        assert result["mz"] == 0.0

    def test_extract_precursor_from_mzml_style(self) -> None:
        """Test extraction from precursor list (mzML style)."""
        from instrument_io.readers.mzml import _extract_precursor_info

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
        from instrument_io.readers.mzml import _extract_precursor_info

        # precursor key exists but list is empty
        spectrum = SpectrumDict({"precursor": []})
        result = _extract_precursor_info(spectrum)
        assert result is None

    def test_extract_precursor_first_entry_not_dict(self) -> None:
        """Test line 252: return None when first precursor entry is not dict."""
        from instrument_io.readers.mzml import _extract_precursor_info

        # precursor list has entry but it's not a dict
        spectrum = SpectrumDict({"precursor": ["not a dict"]})
        result = _extract_precursor_info(spectrum)
        assert result is None

    def test_extract_precursor_missing_selected_ions(self) -> None:
        """Test line 252: return None when selectedIons is missing."""
        from instrument_io.readers.mzml import _extract_precursor_info

        # precursor entry is dict but has no selectedIons
        precursor_entry: dict[str, SpectrumValue] = {"isolationWindow": {}}
        spectrum = SpectrumDict({"precursor": [precursor_entry]})
        result = _extract_precursor_info(spectrum)
        assert result is None

    def test_extract_precursor_empty_selected_ions(self) -> None:
        """Test line 252: return None when selectedIons is empty list."""
        from instrument_io.readers.mzml import _extract_precursor_info

        # selectedIons exists but is empty
        precursor_entry: dict[str, SpectrumValue] = {"selectedIons": []}
        spectrum = SpectrumDict({"precursor": [precursor_entry]})
        result = _extract_precursor_info(spectrum)
        assert result is None

    def test_extract_precursor_ion_not_dict(self) -> None:
        """Test line 252: return None when ion entry is not dict."""
        from instrument_io.readers.mzml import _extract_precursor_info

        # selectedIons has entry but it's not a dict
        precursor_entry: dict[str, SpectrumValue] = {"selectedIons": ["not a dict"]}
        spectrum = SpectrumDict({"precursor": [precursor_entry]})
        result = _extract_precursor_info(spectrum)
        assert result is None

    def test_extract_precursor_precursormz_list_invalid_first_item(self) -> None:
        """Test line 214: precursorMz list with non-numeric first item."""
        from instrument_io.readers.mzml import _extract_precursor_info

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
        from instrument_io.readers.mzml import _extract_precursor_info

        # precursorMz is empty list - goes to else branch, returns mz=0.0
        spectrum = SpectrumDict({"precursorMz": []})
        result = _extract_precursor_info(spectrum)

        # Returns PrecursorInfo with mz=0.0 - verify expected values
        assert result
        assert result["mz"] == 0.0
        assert result["charge"] is None
        assert result["intensity"] is None


class TestSpectrumToMSSpectrumAlternativePaths:
    """Test alternative metadata extraction paths in _spectrum_to_msspectrum."""

    def test_retention_time_from_direct_retentiontime(self) -> None:
        """Test RT extraction from direct 'retentionTime' key (mzXML style)."""
        from instrument_io.readers.mzml import _spectrum_to_msspectrum

        # Create minimal spectrum with retentionTime instead of scanList
        spectrum = SpectrumDict(
            {
                "m/z array": MockArray([100.0, 200.0]),
                "intensity array": MockArray([1000.0, 2000.0]),
                "id": "scan=1",
                "retentionTime": 1.25,
                "ms level": 1,
            }
        )

        result = _spectrum_to_msspectrum(spectrum, "/test.mzML")
        assert result["meta"]["retention_time"] == 1.25

    def test_ms_level_from_mslevel_camelcase(self) -> None:
        """Test MS level extraction from 'msLevel' key (mzXML style)."""
        from instrument_io.readers.mzml import _spectrum_to_msspectrum

        # Create spectrum with msLevel instead of 'ms level'
        spectrum = SpectrumDict(
            {
                "m/z array": MockArray([100.0, 200.0]),
                "intensity array": MockArray([1000.0, 2000.0]),
                "id": "scan=1",
                "msLevel": 2,
            }
        )

        result = _spectrum_to_msspectrum(spectrum, "/test.mzML")
        assert result["meta"]["ms_level"] == 2

    def test_retention_time_from_scanlist_dict(self) -> None:
        """Test RT extraction from scanList dict structure."""
        from instrument_io.readers.mzml import _spectrum_to_msspectrum

        # Create spectrum with scanList structure
        scan_entry: dict[str, SpectrumValue] = {"scan start time": 2.5}
        scanlist: dict[str, SpectrumValue] = {"scan": [scan_entry]}
        spectrum = SpectrumDict(
            {
                "m/z array": MockArray([100.0]),
                "intensity array": MockArray([1000.0]),
                "id": "scan=1",
                "ms level": 1,
                "scanList": scanlist,
            }
        )

        result = _spectrum_to_msspectrum(spectrum, "/test.mzML")
        assert result["meta"]["retention_time"] == 2.5

    def test_scanlist_with_empty_scan_array(self) -> None:
        """Test branch 284->297: scanList with empty scan array.

        When scanList is a dict with 'scan' key but the list is empty,
        the RT extraction falls through to retentionTime or defaults to 0.0.
        """
        from instrument_io.readers.mzml import _spectrum_to_msspectrum

        scanlist: dict[str, SpectrumValue] = {"scan": []}
        spectrum = SpectrumDict(
            {
                "m/z array": MockArray([100.0]),
                "intensity array": MockArray([1000.0]),
                "id": "scan=1",
                "ms level": 1,
                "scanList": scanlist,
            }
        )

        result = _spectrum_to_msspectrum(spectrum, "/test.mzML")
        # Falls through to else branch, no retentionTime, so RT is 0.0
        assert result["meta"]["retention_time"] == 0.0

    def test_scanlist_with_non_dict_first_scan(self) -> None:
        """Test branch 286->297: scanList where first scan is not a dict.

        When scanList['scan'][0] is not a dict, RT extraction falls through.
        """
        from instrument_io.readers.mzml import _spectrum_to_msspectrum

        scanlist: dict[str, SpectrumValue] = {"scan": ["not a dict"]}
        spectrum = SpectrumDict(
            {
                "m/z array": MockArray([100.0]),
                "intensity array": MockArray([1000.0]),
                "id": "scan=1",
                "ms level": 1,
                "scanList": scanlist,
            }
        )

        result = _spectrum_to_msspectrum(spectrum, "/test.mzML")
        # Falls through because first_scan is not dict, RT is 0.0
        assert result["meta"]["retention_time"] == 0.0

    def test_mslevel_camelcase_non_int(self) -> None:
        """Test branch 307->309: msLevel (camelCase) is not an int.

        When neither 'ms level' nor 'msLevel' is an int, ms_level defaults to 1.
        """
        from instrument_io.readers.mzml import _spectrum_to_msspectrum

        spectrum = SpectrumDict(
            {
                "m/z array": MockArray([100.0]),
                "intensity array": MockArray([1000.0]),
                "id": "scan=1",
                "msLevel": "not an int",  # String, not int
            }
        )

        result = _spectrum_to_msspectrum(spectrum, "/test.mzML")
        # ms_level defaults to 1 when not properly set
        assert result["meta"]["ms_level"] == 1

    def test_both_ms_level_keys_missing(self) -> None:
        """Test when neither 'ms level' nor 'msLevel' is present."""
        from instrument_io.readers.mzml import _spectrum_to_msspectrum

        spectrum = SpectrumDict(
            {
                "m/z array": MockArray([100.0]),
                "intensity array": MockArray([1000.0]),
                "id": "scan=1",
            }
        )

        result = _spectrum_to_msspectrum(spectrum, "/test.mzML")
        # ms_level defaults to 1 when not present
        assert result["meta"]["ms_level"] == 1

    def test_mslevel_float_not_used(self) -> None:
        """Test that msLevel as float doesn't get used (branch 307->309)."""
        from instrument_io.readers.mzml import _spectrum_to_msspectrum

        spectrum = SpectrumDict(
            {
                "m/z array": MockArray([100.0]),
                "intensity array": MockArray([1000.0]),
                "id": "scan=1",
                "msLevel": 2.5,  # Float, not int - isinstance(2.5, int) is False
            }
        )

        result = _spectrum_to_msspectrum(spectrum, "/test.mzML")
        # ms_level defaults to 1 because float is not int
        assert result["meta"]["ms_level"] == 1


class MockDType:
    """Mock dtype for MockArray."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name


class MockArray:
    """Mock array object implementing NdArrayProtocol."""

    def __init__(self, data: list[float]) -> None:
        self._data = data

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self._data),)

    @property
    def dtype(self) -> MockDType:
        return MockDType("float64")

    @property
    def ndim(self) -> int:
        return 1

    @property
    def size(self) -> int:
        return len(self._data)

    def tolist(self) -> list[float]:
        return self._data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> float:
        return self._data[idx]
