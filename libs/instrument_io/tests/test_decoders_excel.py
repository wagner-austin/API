"""Tests for Excel decoder functions."""

from __future__ import annotations

import pytest

from instrument_io._decoders.excel import (
    _decode_cell_value,
    _decode_row_dict,
    _decode_rows,
    _extract_bool,
    _extract_float,
    _extract_float_or_none,
    _extract_int,
    _extract_int_or_none,
    _extract_string,
    _extract_string_or_none,
)
from instrument_io._exceptions import DecodingError
from instrument_io._json_bridge import JSONValue


class TestDecodeCellValue:
    """Tests for _decode_cell_value."""

    def test_string(self) -> None:
        assert _decode_cell_value("hello") == "hello"

    def test_int(self) -> None:
        assert _decode_cell_value(42) == 42

    def test_float(self) -> None:
        assert _decode_cell_value(3.14) == 3.14

    def test_bool(self) -> None:
        assert _decode_cell_value(True) is True

    def test_none(self) -> None:
        assert _decode_cell_value(None) is None


class TestDecodeRowDict:
    """Tests for _decode_row_dict."""

    def test_basic(self) -> None:
        row: dict[str, JSONValue] = {"Name": "Alice", "Age": 30, "Score": 95.5}
        columns = ["Name", "Age", "Score"]
        result = _decode_row_dict(row, columns)
        assert result == {"Name": "Alice", "Age": 30, "Score": 95.5}

    def test_missing_column(self) -> None:
        row: dict[str, JSONValue] = {"Name": "Bob"}
        columns = ["Name", "Age"]
        result = _decode_row_dict(row, columns)
        assert result == {"Name": "Bob", "Age": None}


class TestDecodeRows:
    """Tests for _decode_rows."""

    def test_multiple_rows(self) -> None:
        rows: list[dict[str, JSONValue]] = [
            {"Name": "Alice", "Age": 30},
            {"Name": "Bob", "Age": 25},
        ]
        columns = ["Name", "Age"]
        result = _decode_rows(rows, columns)
        assert len(result) == 2
        assert result[0] == {"Name": "Alice", "Age": 30}
        assert result[1] == {"Name": "Bob", "Age": 25}


class TestExtractString:
    """Tests for _extract_string."""

    def test_string(self) -> None:
        assert _extract_string("hello", "field") == "hello"

    def test_int_converts(self) -> None:
        assert _extract_string(42, "field") == "42"

    def test_float_converts(self) -> None:
        assert _extract_string(3.14, "field") == "3.14"

    def test_bool_converts(self) -> None:
        assert _extract_string(True, "field") == "True"

    def test_none_raises(self) -> None:
        with pytest.raises(DecodingError) as exc_info:
            _extract_string(None, "field")
        assert "expected string" in str(exc_info.value)

    def test_list_raises(self) -> None:
        with pytest.raises(DecodingError) as exc_info:
            _extract_string([1, 2, 3], "field")
        assert "Cannot convert list to string" in str(exc_info.value)


class TestExtractStringOrNone:
    """Tests for _extract_string_or_none."""

    def test_string(self) -> None:
        assert _extract_string_or_none("hello") == "hello"

    def test_int_converts(self) -> None:
        assert _extract_string_or_none(42) == "42"

    def test_float_converts(self) -> None:
        assert _extract_string_or_none(3.14) == "3.14"

    def test_bool_converts(self) -> None:
        assert _extract_string_or_none(True) == "True"

    def test_none_returns_none(self) -> None:
        assert _extract_string_or_none(None) is None

    def test_list_converts_to_string(self) -> None:
        result = _extract_string_or_none([1, 2, 3])
        assert result == "[1, 2, 3]"


class TestExtractInt:
    """Tests for _extract_int."""

    def test_int(self) -> None:
        assert _extract_int(42, "field") == 42

    def test_float_converts(self) -> None:
        assert _extract_int(42.7, "field") == 42

    def test_string_converts(self) -> None:
        assert _extract_int("123", "field") == 123

    def test_none_raises(self) -> None:
        with pytest.raises(DecodingError) as exc_info:
            _extract_int(None, "field")
        assert "expected int" in str(exc_info.value)

    def test_bool_raises(self) -> None:
        with pytest.raises(DecodingError) as exc_info:
            _extract_int(True, "field")
        assert "expected int" in str(exc_info.value)

    def test_invalid_string_raises(self) -> None:
        with pytest.raises(DecodingError) as exc_info:
            _extract_int("abc", "field")
        assert "Cannot parse" in str(exc_info.value)

    def test_list_raises(self) -> None:
        with pytest.raises(DecodingError) as exc_info:
            _extract_int([1, 2, 3], "field")
        assert "Cannot convert list to int" in str(exc_info.value)


class TestExtractIntOrNone:
    """Tests for _extract_int_or_none."""

    def test_int(self) -> None:
        assert _extract_int_or_none(42) == 42

    def test_float_converts(self) -> None:
        assert _extract_int_or_none(42.7) == 42

    def test_string_converts(self) -> None:
        assert _extract_int_or_none("123") == 123

    def test_none_returns_none(self) -> None:
        assert _extract_int_or_none(None) is None

    def test_bool_returns_none(self) -> None:
        assert _extract_int_or_none(True) is None

    def test_invalid_string_returns_none(self) -> None:
        assert _extract_int_or_none("abc") is None

    def test_list_returns_none(self) -> None:
        assert _extract_int_or_none([1, 2, 3]) is None


class TestExtractFloat:
    """Tests for _extract_float."""

    def test_float(self) -> None:
        assert _extract_float(3.14, "field") == 3.14

    def test_int_converts(self) -> None:
        assert _extract_float(42, "field") == 42.0

    def test_string_converts(self) -> None:
        assert _extract_float("3.14", "field") == 3.14

    def test_none_raises(self) -> None:
        with pytest.raises(DecodingError) as exc_info:
            _extract_float(None, "field")
        assert "expected float" in str(exc_info.value)

    def test_bool_raises(self) -> None:
        with pytest.raises(DecodingError) as exc_info:
            _extract_float(True, "field")
        assert "expected float" in str(exc_info.value)

    def test_invalid_string_raises(self) -> None:
        with pytest.raises(DecodingError) as exc_info:
            _extract_float("abc", "field")
        assert "Cannot parse" in str(exc_info.value)

    def test_list_raises(self) -> None:
        with pytest.raises(DecodingError) as exc_info:
            _extract_float([1.0, 2.0], "field")
        assert "Cannot convert list to float" in str(exc_info.value)


class TestExtractFloatOrNone:
    """Tests for _extract_float_or_none."""

    def test_float(self) -> None:
        assert _extract_float_or_none(3.14) == 3.14

    def test_int_converts(self) -> None:
        assert _extract_float_or_none(42) == 42.0

    def test_string_converts(self) -> None:
        assert _extract_float_or_none("3.14") == 3.14

    def test_none_returns_none(self) -> None:
        assert _extract_float_or_none(None) is None

    def test_bool_returns_none(self) -> None:
        assert _extract_float_or_none(True) is None

    def test_invalid_string_returns_none(self) -> None:
        assert _extract_float_or_none("abc") is None

    def test_list_returns_none(self) -> None:
        assert _extract_float_or_none([1.0, 2.0]) is None


class TestExtractBool:
    """Tests for _extract_bool."""

    def test_bool_true(self) -> None:
        assert _extract_bool(True, "field") is True

    def test_bool_false(self) -> None:
        assert _extract_bool(False, "field") is False

    def test_string_true(self) -> None:
        assert _extract_bool("true", "field") is True

    def test_string_yes(self) -> None:
        assert _extract_bool("yes", "field") is True

    def test_string_1(self) -> None:
        assert _extract_bool("1", "field") is True

    def test_string_y(self) -> None:
        assert _extract_bool("y", "field") is True

    def test_string_false(self) -> None:
        assert _extract_bool("false", "field") is False

    def test_string_no(self) -> None:
        assert _extract_bool("no", "field") is False

    def test_string_0(self) -> None:
        assert _extract_bool("0", "field") is False

    def test_string_n(self) -> None:
        assert _extract_bool("n", "field") is False

    def test_string_case_insensitive(self) -> None:
        assert _extract_bool("TRUE", "field") is True
        assert _extract_bool("False", "field") is False

    def test_int_nonzero(self) -> None:
        assert _extract_bool(1, "field") is True

    def test_int_zero(self) -> None:
        assert _extract_bool(0, "field") is False

    def test_float_nonzero(self) -> None:
        assert _extract_bool(3.14, "field") is True

    def test_float_zero(self) -> None:
        assert _extract_bool(0.0, "field") is False

    def test_none_raises(self) -> None:
        with pytest.raises(DecodingError) as exc_info:
            _extract_bool(None, "field")
        assert "expected bool" in str(exc_info.value)

    def test_invalid_string_raises(self) -> None:
        with pytest.raises(DecodingError) as exc_info:
            _extract_bool("maybe", "field")
        assert "Cannot parse" in str(exc_info.value)

    def test_list_raises(self) -> None:
        with pytest.raises(DecodingError) as exc_info:
            _extract_bool([True, False], "field")
        assert "Cannot convert list to bool" in str(exc_info.value)
