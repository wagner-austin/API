"""Tests for CSV decoder functions."""

from __future__ import annotations

import pytest

from instrument_io._decoders.csv import (
    _compute_chromatogram_stats_from_data,
    _detect_delimiter,
    _find_column_index,
    _make_chromatogram_data,
    _parse_csv_line,
    _parse_float_column,
    _parse_float_value,
    _percentile,
)
from instrument_io._exceptions import CSVReadError, DecodingError


class TestDetectDelimiter:
    """Tests for _detect_delimiter."""

    def test_tab_delimiter(self) -> None:
        assert _detect_delimiter("A\tB\tC") == "\t"

    def test_comma_delimiter(self) -> None:
        assert _detect_delimiter("A,B,C") == ","

    def test_no_delimiter_raises(self) -> None:
        with pytest.raises(DecodingError) as exc_info:
            _detect_delimiter("ABC")
        assert "No tab or comma" in str(exc_info.value)

    def test_tab_takes_precedence(self) -> None:
        # If both tab and comma present, tab wins (TSV format)
        assert _detect_delimiter("A\tB,C") == "\t"


class TestParseCsvLine:
    """Tests for _parse_csv_line."""

    def test_comma_split(self) -> None:
        result = _parse_csv_line("A,B,C", ",")
        assert result == ["A", "B", "C"]

    def test_tab_split(self) -> None:
        result = _parse_csv_line("A\tB\tC", "\t")
        assert result == ["A", "B", "C"]

    def test_strips_whitespace(self) -> None:
        result = _parse_csv_line(" A , B , C ", ",")
        assert result == ["A", "B", "C"]


class TestFindColumnIndex:
    """Tests for _find_column_index."""

    def test_finds_exact_match(self) -> None:
        headers = ["Name", "Age", "Score"]
        assert _find_column_index(headers, "Age", "test.csv") == 1

    def test_case_insensitive(self) -> None:
        headers = ["Name", "AGE", "Score"]
        assert _find_column_index(headers, "age", "test.csv") == 1

    def test_not_found_raises(self) -> None:
        headers = ["Name", "Age", "Score"]
        with pytest.raises(CSVReadError) as exc_info:
            _find_column_index(headers, "Missing", "test.csv")
        assert "Column 'Missing' not found" in str(exc_info.value)
        assert "Available columns" in str(exc_info.value)


class TestParseFloatValue:
    """Tests for _parse_float_value."""

    def test_integer_string(self) -> None:
        result = _parse_float_value("42", 1, "col", "test.csv")
        assert result == 42.0

    def test_negative_integer(self) -> None:
        result = _parse_float_value("-42", 1, "col", "test.csv")
        assert result == -42.0

    def test_float_string(self) -> None:
        result = _parse_float_value("3.14", 1, "col", "test.csv")
        assert result == 3.14

    def test_scientific_notation(self) -> None:
        result = _parse_float_value("1.5e3", 1, "col", "test.csv")
        assert result == 1500.0

    def test_scientific_notation_uppercase(self) -> None:
        result = _parse_float_value("1.5E3", 1, "col", "test.csv")
        assert result == 1500.0

    def test_empty_raises(self) -> None:
        with pytest.raises(CSVReadError) as exc_info:
            _parse_float_value("", 1, "col", "test.csv")
        assert "Empty value" in str(exc_info.value)

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(CSVReadError) as exc_info:
            _parse_float_value("   ", 1, "col", "test.csv")
        assert "Empty value" in str(exc_info.value)

    def test_multiple_decimals_raises(self) -> None:
        with pytest.raises(CSVReadError) as exc_info:
            _parse_float_value("1.2.3", 1, "col", "test.csv")
        assert "Invalid float" in str(exc_info.value)

    def test_invalid_chars_raises(self) -> None:
        with pytest.raises(CSVReadError) as exc_info:
            _parse_float_value("12abc", 1, "col", "test.csv")
        assert "Invalid float" in str(exc_info.value)

    def test_thousands_separator_removed(self) -> None:
        result = _parse_float_value("1,000", 1, "col", "test.csv")
        assert result == 1000.0


class TestParseFloatColumn:
    """Tests for _parse_float_column."""

    def test_extracts_column(self) -> None:
        rows = [["A", "1.0", "X"], ["B", "2.0", "Y"]]
        result = _parse_float_column(rows, 1, "value", "test.csv")
        assert result == [1.0, 2.0]

    def test_column_out_of_range_raises(self) -> None:
        rows = [["A", "B"]]  # Only 2 columns
        with pytest.raises(CSVReadError) as exc_info:
            _parse_float_column(rows, 5, "value", "test.csv")
        assert "Row 2 has 2 columns" in str(exc_info.value)


class TestComputeChromatogramStatsFromData:
    """Tests for _compute_chromatogram_stats_from_data."""

    def test_empty_data_raises(self) -> None:
        with pytest.raises(CSVReadError) as exc_info:
            _compute_chromatogram_stats_from_data([], [], "test.csv")
        assert "Empty chromatogram data" in str(exc_info.value)

    def test_empty_intensities_raises(self) -> None:
        with pytest.raises(CSVReadError) as exc_info:
            _compute_chromatogram_stats_from_data([1.0], [], "test.csv")
        assert "Empty chromatogram data" in str(exc_info.value)

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(CSVReadError) as exc_info:
            _compute_chromatogram_stats_from_data([1.0, 2.0], [100.0], "test.csv")
        assert "Mismatched lengths" in str(exc_info.value)

    def test_single_point(self) -> None:
        stats = _compute_chromatogram_stats_from_data([1.0], [100.0], "test.csv")
        assert stats["num_points"] == 1
        assert stats["rt_step_mean"] == 0.0

    def test_multiple_points(self) -> None:
        rt = [0.0, 1.0, 2.0]
        intensities = [100.0, 200.0, 150.0]
        stats = _compute_chromatogram_stats_from_data(rt, intensities, "test.csv")
        assert stats["num_points"] == 3
        assert stats["rt_min"] == 0.0
        assert stats["rt_max"] == 2.0
        assert stats["rt_step_mean"] == 1.0


class TestPercentile:
    """Tests for _percentile."""

    def test_single_value(self) -> None:
        assert _percentile([100.0], 0.99) == 100.0

    def test_99th_percentile(self) -> None:
        values = [float(i) for i in range(100)]
        result = _percentile(values, 0.99)
        assert 98.0 <= result <= 99.0


class TestMakeChromatogramData:
    """Tests for _make_chromatogram_data."""

    def test_creates_typeddict(self) -> None:
        data = _make_chromatogram_data([0.0, 1.0], [100.0, 200.0])
        assert data["retention_times"] == [0.0, 1.0]
        assert data["intensities"] == [100.0, 200.0]
