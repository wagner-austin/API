"""Tests for Word document decoder functions."""

from __future__ import annotations

from instrument_io._decoders.docx import (
    _decode_cell_value,
    _is_float_format,
)


class TestDecodeCellValue:
    """Tests for _decode_cell_value."""

    def test_empty_returns_none(self) -> None:
        assert _decode_cell_value("") is None

    def test_whitespace_returns_none(self) -> None:
        assert _decode_cell_value("   ") is None

    def test_boolean_true(self) -> None:
        assert _decode_cell_value("true") is True
        assert _decode_cell_value("yes") is True
        assert _decode_cell_value("y") is True

    def test_boolean_false(self) -> None:
        assert _decode_cell_value("false") is False
        assert _decode_cell_value("no") is False
        assert _decode_cell_value("n") is False

    def test_integer(self) -> None:
        assert _decode_cell_value("42") == 42
        assert _decode_cell_value("-123") == -123

    def test_float(self) -> None:
        assert _decode_cell_value("3.14") == 3.14
        assert _decode_cell_value("-2.5") == -2.5

    def test_scientific_notation(self) -> None:
        assert _decode_cell_value("1.5e3") == 1500.0
        assert _decode_cell_value("1.5E3") == 1500.0

    def test_string(self) -> None:
        assert _decode_cell_value("hello") == "hello"


class TestIsFloatFormat:
    """Tests for _is_float_format."""

    def test_empty_returns_false(self) -> None:
        assert _is_float_format("") is False

    def test_no_decimal_or_exp_returns_false(self) -> None:
        assert _is_float_format("123") is False

    def test_just_sign_returns_false(self) -> None:
        assert _is_float_format("-") is False
        assert _is_float_format("+") is False

    def test_valid_decimal(self) -> None:
        assert _is_float_format("3.14") is True
        assert _is_float_format("-2.5") is True
        assert _is_float_format(".5") is True
        assert _is_float_format("5.") is True

    def test_valid_scientific(self) -> None:
        assert _is_float_format("1e3") is True
        assert _is_float_format("1.5e3") is True
        assert _is_float_format("1E-3") is True

    def test_invalid_scientific(self) -> None:
        assert _is_float_format("e3") is False
        assert _is_float_format("1e") is False
        assert _is_float_format("1ee3") is False

    def test_multiple_decimals_returns_false(self) -> None:
        assert _is_float_format("1.2.3") is False
