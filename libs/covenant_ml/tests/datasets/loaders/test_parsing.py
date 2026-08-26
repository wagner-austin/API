"""Tests for shared parsing utilities.

Tests the internal _parsing module functions that are shared
across CSV, ARFF, and time-series loaders.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from covenant_ml.datasets.loaders._parsing import (
    CATEGORICAL_MISSING,
    MISSING_VALUES,
    build_categorical_encodings,
    build_encoding_lookup,
    detect_categorical_columns,
    encode_categorical_value,
    encode_label,
    find_column_index,
    is_numeric_value,
    is_simple_numeric,
    parse_numeric_value,
)
from covenant_ml.datasets.types import CategoricalEncoding, TargetColumnSpec


class TestMissingValues:
    """Tests for MISSING_VALUES constant."""

    def test_missing_values_contains_empty_string(self) -> None:
        """Empty string is a missing value."""
        assert "" in MISSING_VALUES

    def test_missing_values_contains_question_mark(self) -> None:
        """Question mark (ARFF missing) is a missing value."""
        assert "?" in MISSING_VALUES

    def test_missing_values_contains_na_variants(self) -> None:
        """NA/N/A variants are missing values."""
        assert "NA" in MISSING_VALUES
        assert "N/A" in MISSING_VALUES
        assert "n/a" in MISSING_VALUES

    def test_missing_values_contains_nan_variants(self) -> None:
        """NaN variants are missing values."""
        assert "NaN" in MISSING_VALUES
        assert "nan" in MISSING_VALUES

    def test_missing_values_contains_null_variants(self) -> None:
        """Null variants are missing values."""
        assert "null" in MISSING_VALUES
        assert "NULL" in MISSING_VALUES
        assert "None" in MISSING_VALUES

    def test_missing_values_contains_dot(self) -> None:
        """Single dot (SAS missing) is a missing value."""
        assert "." in MISSING_VALUES


class TestFindColumnIndex:
    """Tests for find_column_index function."""

    def test_find_column_exact_match(self) -> None:
        """Find column with exact case match."""
        headers = ["id", "name", "value"]
        assert find_column_index(headers, "name") == 1

    def test_find_column_case_insensitive(self) -> None:
        """Find column with different case."""
        headers = ["ID", "Name", "Value"]
        assert find_column_index(headers, "name") == 1
        assert find_column_index(headers, "NAME") == 1
        assert find_column_index(headers, "NaMe") == 1

    def test_find_column_first_position(self) -> None:
        """Find column at first position."""
        headers = ["target", "feature1", "feature2"]
        assert find_column_index(headers, "target") == 0

    def test_find_column_last_position(self) -> None:
        """Find column at last position."""
        headers = ["feature1", "feature2", "target"]
        assert find_column_index(headers, "target") == 2

    def test_find_column_not_found_raises(self) -> None:
        """Raise ValueError when column not found."""
        headers = ["id", "name", "value"]
        with pytest.raises(ValueError, match="Column 'missing' not found"):
            find_column_index(headers, "missing")

    def test_find_column_shows_available_columns(self) -> None:
        """Error message includes available column names."""
        headers = ["id", "name", "value"]
        with pytest.raises(ValueError, match=r"Available: \['id', 'name', 'value'\]"):
            find_column_index(headers, "missing")


class TestParseNumericValue:
    """Tests for parse_numeric_value function."""

    def test_parse_integer(self) -> None:
        """Parse integer string."""
        assert parse_numeric_value("42") == 42.0

    def test_parse_float(self) -> None:
        """Parse float string."""
        assert parse_numeric_value("3.14") == 3.14

    def test_parse_negative(self) -> None:
        """Parse negative number."""
        assert parse_numeric_value("-5.5") == -5.5

    def test_parse_with_whitespace(self) -> None:
        """Parse value with leading/trailing whitespace."""
        assert parse_numeric_value("  42  ") == 42.0

    def test_parse_with_thousands_separator(self) -> None:
        """Parse value with comma thousands separator."""
        assert parse_numeric_value("1,234,567") == 1234567.0

    def test_parse_missing_empty(self) -> None:
        """Missing empty string returns 0.0."""
        assert parse_numeric_value("") == 0.0

    def test_parse_missing_question_mark(self) -> None:
        """Missing question mark returns 0.0."""
        assert parse_numeric_value("?") == 0.0

    def test_parse_missing_na(self) -> None:
        """Missing NA returns 0.0."""
        assert parse_numeric_value("NA") == 0.0

    def test_parse_infinity_returns_zero(self) -> None:
        """Infinity value is replaced with 0.0."""
        assert parse_numeric_value("inf") == 0.0
        assert parse_numeric_value("-inf") == 0.0

    def test_parse_scientific_notation(self) -> None:
        """Parse scientific notation."""
        assert parse_numeric_value("1e-5") == pytest.approx(0.00001)
        assert parse_numeric_value("2.5E10") == pytest.approx(2.5e10)


class TestIsNumericValue:
    """Tests for is_numeric_value function."""

    def test_integer_is_numeric(self) -> None:
        """Integer string is numeric."""
        assert is_numeric_value("42") is True

    def test_float_is_numeric(self) -> None:
        """Float string is numeric."""
        assert is_numeric_value("3.14") is True

    def test_negative_is_numeric(self) -> None:
        """Negative number is numeric."""
        assert is_numeric_value("-5.5") is True

    def test_positive_sign_is_numeric(self) -> None:
        """Positive signed number is numeric."""
        assert is_numeric_value("+42") is True

    def test_scientific_notation_is_numeric(self) -> None:
        """Scientific notation is numeric."""
        assert is_numeric_value("1e-5") is True
        assert is_numeric_value("2.5E10") is True
        assert is_numeric_value("-1.5e+3") is True

    def test_infinity_is_numeric(self) -> None:
        """Infinity values are numeric."""
        assert is_numeric_value("inf") is True
        assert is_numeric_value("-inf") is True
        assert is_numeric_value("+inf") is True
        assert is_numeric_value("infinity") is True

    def test_with_thousands_separator_is_numeric(self) -> None:
        """Value with comma thousands separator is numeric."""
        assert is_numeric_value("1,234,567") is True

    def test_text_is_not_numeric(self) -> None:
        """Plain text is not numeric."""
        assert is_numeric_value("hello") is False

    def test_mixed_text_numbers_is_not_numeric(self) -> None:
        """Mixed text and numbers is not numeric."""
        assert is_numeric_value("abc123") is False

    def test_empty_string_is_not_numeric(self) -> None:
        """Empty string is not numeric."""
        assert is_numeric_value("") is False

    def test_just_sign_is_not_numeric(self) -> None:
        """Just a sign character is not numeric."""
        assert is_numeric_value("-") is False
        assert is_numeric_value("+") is False

    def test_multiple_decimal_points_is_not_numeric(self) -> None:
        """A version string must not read as a measurement."""
        assert is_numeric_value("1.2.3") is False

    def test_a_digit_glued_to_letters_is_not_numeric(self) -> None:
        """Unit suffixes are the common case: `1.2a` is a value and a unit."""
        assert is_numeric_value("1.2a") is False

    def test_repeated_exponent_marker_is_not_numeric(self) -> None:
        assert is_numeric_value("1e2e3") is False

    def test_a_non_numeric_mantissa_is_not_numeric(self) -> None:
        """`abce5` splits on 'e' into two halves that look like the shape."""
        assert is_numeric_value("abce5") is False

    def test_a_non_numeric_or_absent_exponent_is_not_numeric(self) -> None:
        assert is_numeric_value("1eabc") is False
        assert is_numeric_value("1e") is False


class TestIsSimpleNumeric:
    """Tests for is_simple_numeric function."""

    def test_integer_is_simple_numeric(self) -> None:
        """Integer string is simple numeric."""
        assert is_simple_numeric("42") is True

    def test_decimal_is_simple_numeric(self) -> None:
        """Decimal string is simple numeric."""
        assert is_simple_numeric("3.14") is True

    def test_leading_decimal_is_simple_numeric(self) -> None:
        """Leading decimal point is simple numeric."""
        assert is_simple_numeric(".5") is True

    def test_trailing_decimal_is_simple_numeric(self) -> None:
        """Trailing decimal point is simple numeric."""
        assert is_simple_numeric("5.") is True

    def test_empty_string_is_not_simple_numeric(self) -> None:
        """Empty string is not simple numeric."""
        assert is_simple_numeric("") is False

    def test_multiple_decimals_is_not_simple_numeric(self) -> None:
        """Multiple decimal points is not simple numeric."""
        assert is_simple_numeric("1.2.3") is False

    def test_non_digit_is_not_simple_numeric(self) -> None:
        """Non-digit characters are not simple numeric."""
        assert is_simple_numeric("abc") is False
        assert is_simple_numeric("1a2") is False

    def test_a_lone_decimal_point_is_not_simple_numeric(self) -> None:
        """`.5` and `5.` are both valid, so the split alone is not enough --
        one side must still carry a digit."""
        assert is_simple_numeric(".") is False


class TestEncodeLabel:
    """Tests for encode_label function."""

    def test_encode_positive_int(self) -> None:
        """Encode positive integer label."""
        spec = TargetColumnSpec(
            column_name="target",
            label_type="binary_int",
            positive_values=(1,),
            negative_values=(0,),
        )
        assert encode_label("1", spec, 0, Path("test.csv")) == 1

    def test_encode_negative_int(self) -> None:
        """Encode negative integer label."""
        spec = TargetColumnSpec(
            column_name="target",
            label_type="binary_int",
            positive_values=(1,),
            negative_values=(0,),
        )
        assert encode_label("0", spec, 0, Path("test.csv")) == 0

    def test_encode_positive_string(self) -> None:
        """Encode positive string label."""
        spec = TargetColumnSpec(
            column_name="target",
            label_type="binary_str",
            positive_values=("yes", "true"),
            negative_values=("no", "false"),
        )
        assert encode_label("yes", spec, 0, Path("test.csv")) == 1
        assert encode_label("true", spec, 0, Path("test.csv")) == 1

    def test_encode_negative_string(self) -> None:
        """Encode negative string label."""
        spec = TargetColumnSpec(
            column_name="target",
            label_type="binary_str",
            positive_values=("yes",),
            negative_values=("no",),
        )
        assert encode_label("no", spec, 0, Path("test.csv")) == 0

    def test_encode_case_insensitive(self) -> None:
        """Label encoding is case-insensitive for strings."""
        spec = TargetColumnSpec(
            column_name="target",
            label_type="binary_str",
            positive_values=("YES",),
            negative_values=("NO",),
        )
        assert encode_label("yes", spec, 0, Path("test.csv")) == 1
        assert encode_label("Yes", spec, 0, Path("test.csv")) == 1

    def test_encode_with_whitespace(self) -> None:
        """Label encoding strips whitespace."""
        spec = TargetColumnSpec(
            column_name="target",
            label_type="binary_int",
            positive_values=(1,),
            negative_values=(0,),
        )
        assert encode_label("  1  ", spec, 0, Path("test.csv")) == 1

    def test_encode_unknown_raises(self) -> None:
        """Raise ValueError for unknown label value."""
        spec = TargetColumnSpec(
            column_name="target",
            label_type="binary_int",
            positive_values=(1,),
            negative_values=(0,),
        )
        with pytest.raises(ValueError, match="Unknown label value"):
            encode_label("unknown", spec, 0, Path("test.csv"))

    def test_encode_unknown_includes_row_number(self) -> None:
        """Error message includes row number."""
        spec = TargetColumnSpec(
            column_name="target",
            label_type="binary_int",
            positive_values=(1,),
            negative_values=(0,),
        )
        with pytest.raises(ValueError, match="at row 42"):
            encode_label("unknown", spec, 42, Path("test.csv"))


class TestDetectCategoricalColumns:
    """Tests for detect_categorical_columns function."""

    def test_detect_all_numeric(self) -> None:
        """No categorical columns when all numeric."""
        rows = [["1", "2.5", "3"], ["4", "5.5", "6"]]
        feature_indices = [0, 1, 2]
        result = detect_categorical_columns(rows, feature_indices)
        assert result == set()

    def test_detect_all_categorical(self) -> None:
        """All columns categorical when all text."""
        rows = [["a", "b", "c"], ["d", "e", "f"]]
        feature_indices = [0, 1, 2]
        result = detect_categorical_columns(rows, feature_indices)
        assert result == {0, 1, 2}

    def test_detect_mixed(self) -> None:
        """Detect mixed numeric and categorical."""
        rows = [["1", "a", "2.5"], ["2", "b", "3.5"]]
        feature_indices = [0, 1, 2]
        result = detect_categorical_columns(rows, feature_indices)
        assert result == {1}

    def test_detect_skips_missing_values(self) -> None:
        """Missing values don't affect categorical detection."""
        rows = [["1", "?", "a"], ["2", "NA", "b"], ["3", "1.5", "c"]]
        feature_indices = [0, 1, 2]
        result = detect_categorical_columns(rows, feature_indices)
        # Column 1 has missing values and one numeric, so not categorical
        # Column 2 is all text (categorical)
        assert result == {2}

    def test_detect_handles_short_rows(self) -> None:
        """Handle rows shorter than expected."""
        rows = [["1", "a"], ["2"]]  # Second row missing columns
        feature_indices = [0, 1]
        result = detect_categorical_columns(rows, feature_indices)
        assert result == {1}


class TestBuildCategoricalEncodings:
    """Tests for build_categorical_encodings function."""

    def test_build_single_column(self) -> None:
        """Build encoding for single categorical column."""
        rows = [["1", "a"], ["2", "b"], ["3", "a"]]
        feature_indices = [0, 1]
        feature_names = ["num", "cat"]
        categorical_columns = {1}

        result = build_categorical_encodings(
            rows, feature_indices, feature_names, categorical_columns
        )

        assert len(result) == 1
        assert result[0]["column_name"] == "cat"
        assert result[0]["n_categories"] == 2
        # Values sorted alphabetically: a=0, b=1
        mapping_dict = dict(result[0]["mapping"])
        assert mapping_dict["a"] == 0
        assert mapping_dict["b"] == 1

    def test_build_includes_missing_category(self) -> None:
        """Build encoding includes missing value category."""
        rows = [["1", "a"], ["2", "?"], ["3", "b"]]
        feature_indices = [0, 1]
        feature_names = ["num", "cat"]
        categorical_columns = {1}

        result = build_categorical_encodings(
            rows, feature_indices, feature_names, categorical_columns
        )

        assert result[0]["n_categories"] == 3
        mapping_dict = dict(result[0]["mapping"])
        assert mapping_dict[CATEGORICAL_MISSING] == 0
        assert mapping_dict["a"] == 1
        assert mapping_dict["b"] == 2

    def test_build_multiple_columns(self) -> None:
        """Build encodings for multiple categorical columns."""
        rows = [["x", "p"], ["y", "q"], ["z", "r"]]
        feature_indices = [0, 1]
        feature_names = ["cat1", "cat2"]
        categorical_columns = {0, 1}

        result = build_categorical_encodings(
            rows, feature_indices, feature_names, categorical_columns
        )

        assert len(result) == 2
        assert result[0]["column_name"] == "cat1"
        assert result[1]["column_name"] == "cat2"


class TestEncodeCategoricalValue:
    """Tests for encode_categorical_value function."""

    def test_encode_known_value(self) -> None:
        """Encode known categorical value."""
        mapping = {"a": 0, "b": 1, "c": 2}
        assert encode_categorical_value("a", mapping) == 0.0
        assert encode_categorical_value("b", mapping) == 1.0
        assert encode_categorical_value("c", mapping) == 2.0

    def test_encode_strips_whitespace(self) -> None:
        """Encoding strips whitespace from value."""
        mapping = {"a": 0, "b": 1}
        assert encode_categorical_value("  a  ", mapping) == 0.0

    def test_encode_missing_value(self) -> None:
        """Missing values use CATEGORICAL_MISSING mapping."""
        mapping = {CATEGORICAL_MISSING: 0, "a": 1, "b": 2}
        assert encode_categorical_value("?", mapping) == 0.0
        assert encode_categorical_value("NA", mapping) == 0.0
        assert encode_categorical_value("", mapping) == 0.0

    def test_encode_missing_without_mapping_returns_zero(self) -> None:
        """Missing value without CATEGORICAL_MISSING returns 0."""
        mapping = {"a": 1, "b": 2}  # No CATEGORICAL_MISSING key
        assert encode_categorical_value("?", mapping) == 0.0


class TestBuildEncodingLookup:
    """Tests for build_encoding_lookup function."""

    def test_build_single_encoding(self) -> None:
        """Build lookup for single encoding."""
        encodings: list[CategoricalEncoding] = [
            CategoricalEncoding(
                column_name="cat",
                mapping=(("a", 0), ("b", 1)),
                n_categories=2,
            )
        ]
        feature_names = ["num", "cat"]

        result = build_encoding_lookup(encodings, feature_names)

        assert len(result) == 1
        assert 1 in result
        assert result[1] == {"a": 0, "b": 1}

    def test_build_multiple_encodings(self) -> None:
        """Build lookup for multiple encodings."""
        encodings: list[CategoricalEncoding] = [
            CategoricalEncoding(
                column_name="cat1",
                mapping=(("x", 0), ("y", 1)),
                n_categories=2,
            ),
            CategoricalEncoding(
                column_name="cat2",
                mapping=(("p", 0), ("q", 1)),
                n_categories=2,
            ),
        ]
        feature_names = ["cat1", "num", "cat2"]

        result = build_encoding_lookup(encodings, feature_names)

        assert len(result) == 2
        assert 0 in result
        assert 2 in result
        assert result[0] == {"x": 0, "y": 1}
        assert result[2] == {"p": 0, "q": 1}

    def test_build_empty_encodings(self) -> None:
        """Build empty lookup for no encodings."""
        encodings: list[CategoricalEncoding] = []
        feature_names = ["num1", "num2"]

        result = build_encoding_lookup(encodings, feature_names)

        assert result == {}
