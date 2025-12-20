"""Tests for Polars utilities module."""

from __future__ import annotations

from covenant_ml.datasets.loaders._polars_utils import (
    _is_simple_numeric,
    is_numeric_string,
)


class TestIsNumericString:
    """Tests for is_numeric_string function."""

    def test_missing_values_are_numeric(self) -> None:
        """Missing values like NA, ?, empty string are treated as numeric."""
        assert is_numeric_string("NA") is True
        assert is_numeric_string("") is True
        assert is_numeric_string("?") is True
        assert is_numeric_string("N/A") is True
        assert is_numeric_string("nan") is True

    def test_infinity_values_are_numeric(self) -> None:
        """Infinity values are treated as numeric."""
        assert is_numeric_string("inf") is True
        assert is_numeric_string("-inf") is True
        assert is_numeric_string("+inf") is True
        assert is_numeric_string("infinity") is True
        assert is_numeric_string("-infinity") is True
        assert is_numeric_string("+infinity") is True
        assert is_numeric_string("INF") is True

    def test_plus_minus_only_not_numeric(self) -> None:
        """Strings with only +/- signs are not numeric."""
        assert is_numeric_string("+") is False
        assert is_numeric_string("-") is False
        assert is_numeric_string("++") is False
        assert is_numeric_string("--") is False

    def test_scientific_notation_with_invalid_parts(self) -> None:
        """Invalid scientific notation is not numeric."""
        # More than one 'e'
        assert is_numeric_string("1e2e3") is False
        # Non-numeric mantissa
        assert is_numeric_string("abce5") is False
        # Non-numeric exponent
        assert is_numeric_string("1ex") is False

    def test_valid_scientific_notation(self) -> None:
        """Valid scientific notation is numeric."""
        assert is_numeric_string("1e5") is True
        assert is_numeric_string("1.5e-3") is True
        assert is_numeric_string("-2.5e+10") is True

    def test_simple_numbers_are_numeric(self) -> None:
        """Simple numbers are numeric."""
        assert is_numeric_string("123") is True
        assert is_numeric_string("1.5") is True
        assert is_numeric_string("-3.14") is True
        assert is_numeric_string("+42") is True

    def test_non_numeric_strings(self) -> None:
        """Non-numeric strings are detected."""
        assert is_numeric_string("abc") is False
        assert is_numeric_string("12a") is False


class TestIsSimpleNumeric:
    """Tests for _is_simple_numeric function."""

    def test_empty_string_not_numeric(self) -> None:
        """Empty string is not numeric."""
        assert _is_simple_numeric("") is False

    def test_multiple_decimal_points_not_numeric(self) -> None:
        """Multiple decimal points are not numeric."""
        assert _is_simple_numeric("1.2.3") is False
        assert _is_simple_numeric("...") is False

    def test_non_digit_parts_not_numeric(self) -> None:
        """Parts with non-digit characters are not numeric."""
        assert _is_simple_numeric("12a") is False
        assert _is_simple_numeric("a.b") is False

    def test_valid_simple_numbers(self) -> None:
        """Valid simple numbers are detected."""
        assert _is_simple_numeric("123") is True
        assert _is_simple_numeric("1.5") is True
        assert _is_simple_numeric(".5") is True
        assert _is_simple_numeric("5.") is True
