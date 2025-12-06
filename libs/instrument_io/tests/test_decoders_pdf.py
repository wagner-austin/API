"""Tests for PDF decoder functions."""

from __future__ import annotations

from instrument_io._decoders.pdf import (
    _decode_pdf_cell,
    _decode_pdf_row,
    _decode_pdf_table,
    _is_float_string,
    _is_integer_string,
)


class TestDecodePdfCell:
    """Tests for _decode_pdf_cell."""

    def test_none_returns_none(self) -> None:
        assert _decode_pdf_cell(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert _decode_pdf_cell("") is None

    def test_whitespace_returns_none(self) -> None:
        assert _decode_pdf_cell("   ") is None

    def test_boolean_true(self) -> None:
        assert _decode_pdf_cell("true") is True
        assert _decode_pdf_cell("True") is True
        assert _decode_pdf_cell("yes") is True
        assert _decode_pdf_cell("Yes") is True
        assert _decode_pdf_cell("y") is True
        assert _decode_pdf_cell("Y") is True

    def test_boolean_false(self) -> None:
        assert _decode_pdf_cell("false") is False
        assert _decode_pdf_cell("False") is False
        assert _decode_pdf_cell("no") is False
        assert _decode_pdf_cell("No") is False
        assert _decode_pdf_cell("n") is False
        assert _decode_pdf_cell("N") is False

    def test_integer(self) -> None:
        assert _decode_pdf_cell("42") == 42
        assert _decode_pdf_cell("-123") == -123
        assert _decode_pdf_cell("0") == 0

    def test_float(self) -> None:
        assert _decode_pdf_cell("3.14") == 3.14
        assert _decode_pdf_cell("-2.5") == -2.5
        assert _decode_pdf_cell("0.5") == 0.5

    def test_string(self) -> None:
        assert _decode_pdf_cell("hello") == "hello"
        assert _decode_pdf_cell("  world  ") == "world"


class TestIsIntegerString:
    """Tests for _is_integer_string."""

    def test_empty_returns_false(self) -> None:
        assert _is_integer_string("") is False

    def test_positive_integer(self) -> None:
        assert _is_integer_string("123") is True

    def test_negative_integer(self) -> None:
        assert _is_integer_string("-123") is True

    def test_just_minus_returns_false(self) -> None:
        assert _is_integer_string("-") is False

    def test_float_returns_false(self) -> None:
        assert _is_integer_string("3.14") is False

    def test_non_digit_returns_false(self) -> None:
        assert _is_integer_string("abc") is False


class TestIsFloatString:
    """Tests for _is_float_string."""

    def test_empty_returns_false(self) -> None:
        assert _is_float_string("") is False

    def test_just_decimal_returns_false(self) -> None:
        assert _is_float_string(".") is False

    def test_just_minus_returns_false(self) -> None:
        assert _is_float_string("-") is False

    def test_valid_float(self) -> None:
        assert _is_float_string("3.14") is True
        assert _is_float_string("-2.5") is True
        assert _is_float_string("0.5") is True

    def test_multiple_decimals_returns_false(self) -> None:
        assert _is_float_string("1.2.3") is False

    def test_integer_returns_false(self) -> None:
        assert _is_float_string("123") is False

    def test_invalid_chars_returns_false(self) -> None:
        assert _is_float_string("3.14abc") is False

    def test_no_digits_returns_false(self) -> None:
        assert _is_float_string("-.") is False

    def test_leading_decimal(self) -> None:
        assert _is_float_string(".5") is True

    def test_trailing_decimal(self) -> None:
        assert _is_float_string("5.") is True


class TestDecodePdfRow:
    """Tests for _decode_pdf_row."""

    def test_decodes_all_cells(self) -> None:
        row = ["42", "hello", None, "true"]
        result = _decode_pdf_row(row)
        assert result == [42, "hello", None, True]


class TestDecodePdfTable:
    """Tests for _decode_pdf_table."""

    def test_empty_table_returns_empty(self) -> None:
        table: list[list[str | None]] = []
        result = _decode_pdf_table(table)
        assert result == []

    def test_single_header_row_returns_empty(self) -> None:
        table: list[list[str | None]] = [["Name", "Value"]]
        result = _decode_pdf_table(table)
        assert result == []

    def test_header_with_none(self) -> None:
        table: list[list[str | None]] = [["Name", None, "Value"], ["Alice", "X", "100"]]
        result = _decode_pdf_table(table)
        assert len(result) == 1
        assert result[0]["Name"] == "Alice"
        assert result[0]["Value"] == 100

    def test_basic_table(self) -> None:
        table: list[list[str | None]] = [
            ["Name", "Age", "Active"],
            ["Alice", "30", "true"],
            ["Bob", "25", "false"],
        ]
        result = _decode_pdf_table(table)
        assert len(result) == 2
        assert result[0] == {"Name": "Alice", "Age": 30, "Active": True}
        assert result[1] == {"Name": "Bob", "Age": 25, "Active": False}

    def test_skips_empty_rows(self) -> None:
        table: list[list[str | None]] = [
            ["Name", "Value"],
            [None, None],  # Empty row
            ["Test", "123"],
        ]
        result = _decode_pdf_table(table)
        # Empty row results in empty dict which is filtered out
        assert any(row.get("Name") == "Test" for row in result)
