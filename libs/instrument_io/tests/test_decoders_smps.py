"""Tests for SMPS decoder functions."""

from __future__ import annotations

import pytest

from instrument_io._decoders.smps import (
    _decode_smps_data,
    _decode_smps_full,
    _decode_smps_metadata,
    _parse_cell_value,
)
from instrument_io._exceptions import DecodingError


class TestParseCellValue:
    """Tests for _parse_cell_value."""

    def test_empty_returns_none(self) -> None:
        assert _parse_cell_value("") is None

    def test_whitespace_returns_none(self) -> None:
        assert _parse_cell_value("   ") is None

    def test_boolean_true(self) -> None:
        assert _parse_cell_value("true") is True
        assert _parse_cell_value("True") is True
        assert _parse_cell_value("yes") is True
        assert _parse_cell_value("Yes") is True
        assert _parse_cell_value("y") is True
        assert _parse_cell_value("Y") is True

    def test_boolean_false(self) -> None:
        assert _parse_cell_value("false") is False
        assert _parse_cell_value("False") is False
        assert _parse_cell_value("no") is False
        assert _parse_cell_value("No") is False
        assert _parse_cell_value("n") is False
        assert _parse_cell_value("N") is False

    def test_integer(self) -> None:
        assert _parse_cell_value("42") == 42
        assert _parse_cell_value("-123") == -123
        assert _parse_cell_value("0") == 0

    def test_float(self) -> None:
        assert _parse_cell_value("3.14") == 3.14
        assert _parse_cell_value("-2.5") == -2.5

    def test_string_fallback(self) -> None:
        assert _parse_cell_value("hello") == "hello"
        assert _parse_cell_value("abc123") == "abc123"


class TestDecodeSmpsMetadata:
    """Tests for _decode_smps_metadata."""

    def test_fewer_than_3_lines_raises(self) -> None:
        lines = ["line1", "line2"]
        with pytest.raises(DecodingError) as exc_info:
            _decode_smps_metadata(lines)
        assert "fewer than 3 lines" in str(exc_info.value)

    def test_invalid_header_raises(self) -> None:
        lines = [
            "date",  # Only one field, not 3
            "param1\tparam2",
            "value1\tvalue2",
        ]
        with pytest.raises(DecodingError) as exc_info:
            _decode_smps_metadata(lines)
        assert "at least 3 tab-separated" in str(exc_info.value)

    def test_param_mismatch_raises(self) -> None:
        lines = [
            "2024-01-01\t12:00:00\tInstrument",
            "param1\tparam2\tparam3",
            "value1\tvalue2",  # Only 2 values, but 3 params
        ]
        with pytest.raises(DecodingError) as exc_info:
            _decode_smps_metadata(lines)
        assert "Mismatch" in str(exc_info.value)

    def test_valid_metadata(self) -> None:
        lines = [
            "2024-01-01\t12:00:00\tSMPS 3080",
            "Lower Voltage Limit [V]\tUpper Voltage Limit [V]\tSample Duration [s]",
            "10\t10000\t120",
        ]
        result = _decode_smps_metadata(lines)
        assert result["timestamp"] == "2024-01-01 12:00:00"
        assert result["instrument"] == "SMPS 3080"
        assert result["lower_voltage_limit"] == 10.0
        assert result["upper_voltage_limit"] == 10000.0
        assert result["sample_duration"] == 120.0

    def test_missing_optional_params(self) -> None:
        lines = [
            "2024-01-01\t12:00:00\tSMPS 3080",
            "Other Param",
            "Some Value",
        ]
        result = _decode_smps_metadata(lines)
        assert result["lower_voltage_limit"] == 0.0
        assert result["upper_voltage_limit"] == 0.0
        assert result["sample_duration"] == 0.0


class TestDecodeSmpsData:
    """Tests for _decode_smps_data."""

    def test_fewer_than_4_lines_raises(self) -> None:
        lines = ["line1", "line2", "line3"]
        with pytest.raises(DecodingError) as exc_info:
            _decode_smps_data(lines)
        assert "fewer than 4 lines" in str(exc_info.value)

    def test_valid_data(self) -> None:
        lines = [
            "header line 0",
            "header line 1",
            "header line 2",
            "Diameter\tConcentration",  # Column headers at line 3
            "10\t100",
            "20\t200",
        ]
        result = _decode_smps_data(lines)
        assert len(result) == 2
        assert result[0]["Diameter"] == 10
        assert result[0]["Concentration"] == 100
        assert result[1]["Diameter"] == 20
        assert result[1]["Concentration"] == 200

    def test_skips_mismatched_rows(self) -> None:
        lines = [
            "header line 0",
            "header line 1",
            "header line 2",
            "Col1\tCol2",
            "A\tB",
            "X",  # Mismatched - only 1 column
            "C\tD",
        ]
        result = _decode_smps_data(lines)
        # Should have 2 rows, skipping the mismatched one
        assert len(result) == 2


class TestDecodeSmpsFullData:
    """Tests for _decode_smps_full."""

    def test_valid_full_file(self) -> None:
        lines = [
            "2024-01-01\t12:00:00\tSMPS 3080",
            "Lower Voltage Limit [V]\tUpper Voltage Limit [V]",
            "10\t10000",
            "Diameter\tConcentration",
            "10\t100",
        ]
        result = _decode_smps_full(lines)
        assert result["metadata"]["instrument"] == "SMPS 3080"
        assert len(result["data"]) == 1
