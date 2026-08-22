"""Tests for readers mzml complete: SpectrumToMSSpectrumAlternativePaths."""

from __future__ import annotations

from instrument_io._protocols.pyteomics import SpectrumValue
from tests._mzml_helpers import SpectrumDict


class TestSpectrumToMSSpectrumAlternativePaths:
    """Test alternative metadata extraction paths in _spectrum_to_msspectrum."""

    def test_retention_time_from_direct_retentiontime(self) -> None:
        """Test RT extraction from direct 'retentionTime' key (mzXML style)."""
        from instrument_io.readers.mzml_spectrum import _spectrum_to_msspectrum

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
        from instrument_io.readers.mzml_spectrum import _spectrum_to_msspectrum

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
        from instrument_io.readers.mzml_spectrum import _spectrum_to_msspectrum

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
        from instrument_io.readers.mzml_spectrum import _spectrum_to_msspectrum

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
        from instrument_io.readers.mzml_spectrum import _spectrum_to_msspectrum

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
        from instrument_io.readers.mzml_spectrum import _spectrum_to_msspectrum

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
        from instrument_io.readers.mzml_spectrum import _spectrum_to_msspectrum

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
        from instrument_io.readers.mzml_spectrum import _spectrum_to_msspectrum

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
