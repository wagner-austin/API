"""Tests for CSV parsing functions."""

from __future__ import annotations

import tempfile
from pathlib import Path

from scripts.discover_datasets.parsers.csv import (
    detect_csv_delimiter,
    read_csv_header_and_sample,
    read_data_header_and_sample,
    strip_quotes,
)


class TestDetectCsvDelimiter:
    """Tests for detect_csv_delimiter function."""

    def test_comma_delimiter(self) -> None:
        """Test detecting comma delimiter."""
        result = detect_csv_delimiter("a,b,c,d")
        assert result == ","

    def test_semicolon_delimiter(self) -> None:
        """Test detecting semicolon delimiter."""
        result = detect_csv_delimiter("a;b;c;d")
        assert result == ";"

    def test_tab_delimiter(self) -> None:
        """Test detecting tab delimiter."""
        result = detect_csv_delimiter("a\tb\tc\td")
        assert result == "\t"

    def test_semicolon_more_than_comma(self) -> None:
        """Test semicolon wins when more frequent than comma."""
        result = detect_csv_delimiter("a;b;c,d")
        assert result == ";"

    def test_tab_more_than_comma(self) -> None:
        """Test tab wins when more frequent than comma."""
        result = detect_csv_delimiter("a\tb\tc,d")
        assert result == "\t"

    def test_comma_default_on_empty(self) -> None:
        """Test comma is default when no delimiters found."""
        result = detect_csv_delimiter("abcd")
        assert result == ","

    def test_comma_wins_tie(self) -> None:
        """Test comma wins on tie with semicolon."""
        result = detect_csv_delimiter("a,b;c")
        assert result == ","

    def test_space_not_used_by_default(self) -> None:
        """Test space is not selected when prefer_space is False."""
        result = detect_csv_delimiter("a b c d")
        assert result == ","  # Falls back to comma

    def test_space_used_with_prefer_space(self) -> None:
        """Test space is selected when prefer_space is True and no other delimiters."""
        result = detect_csv_delimiter("a b c d", prefer_space=True)
        assert result == " "

    def test_comma_preferred_over_space(self) -> None:
        """Test comma is preferred even when there are more spaces."""
        # This simulates CSV with spaces in column names
        result = detect_csv_delimiter("Bankrupt?, ROA before tax, Net income")
        assert result == ","

    def test_space_not_used_when_comma_present(self) -> None:
        """Test space is not used even with prefer_space when comma is present."""
        result = detect_csv_delimiter("a,b c d", prefer_space=True)
        assert result == ","


class TestStripQuotes:
    """Tests for strip_quotes function."""

    def test_double_quotes(self) -> None:
        """Test stripping double quotes."""
        result = strip_quotes('"hello"')
        assert result == "hello"

    def test_single_quotes(self) -> None:
        """Test stripping single quotes."""
        result = strip_quotes("'hello'")
        assert result == "hello"

    def test_no_quotes(self) -> None:
        """Test value without quotes."""
        result = strip_quotes("hello")
        assert result == "hello"

    def test_whitespace_with_quotes(self) -> None:
        """Test stripping whitespace and quotes."""
        result = strip_quotes('  "hello"  ')
        assert result == "hello"

    def test_short_string(self) -> None:
        """Test short string (length < 2)."""
        result = strip_quotes("a")
        assert result == "a"

    def test_mismatched_quotes(self) -> None:
        """Test mismatched quotes are not stripped."""
        result = strip_quotes("\"hello'")
        assert result == "\"hello'"


class TestReadCsvHeaderAndSample:
    """Tests for read_csv_header_and_sample function."""

    def test_empty_file(self) -> None:
        """Test reading empty CSV file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            path = Path(f.name)

        columns, n_rows, sample = read_csv_header_and_sample(path, "utf-8")
        path.unlink()

        assert columns == ()
        assert n_rows == 0
        assert sample == ()

    def test_header_only(self) -> None:
        """Test reading CSV with only header."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write("a,b,c\n")
            path = Path(f.name)

        columns, n_rows, sample = read_csv_header_and_sample(path, "utf-8")
        path.unlink()

        assert columns == ("a", "b", "c")
        assert n_rows == 0
        assert sample == ()

    def test_with_data_rows(self) -> None:
        """Test reading CSV with header and data rows."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write("a,b,c\n1,2,3\n4,5,6\n")
            path = Path(f.name)

        columns, n_rows, sample = read_csv_header_and_sample(path, "utf-8")
        path.unlink()

        assert columns == ("a", "b", "c")
        assert n_rows == 2
        assert sample == (("1", "2", "3"), ("4", "5", "6"))

    def test_utf8_sig_encoding(self) -> None:
        """Test reading CSV with UTF-8 BOM encoding."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as f:
            f.write(b"\xef\xbb\xbfa,b\n1,2\n")
            path = Path(f.name)

        columns, n_rows, _sample = read_csv_header_and_sample(path, "utf-8-sig")
        path.unlink()

        assert columns == ("a", "b")
        assert n_rows == 1

    def test_semicolon_delimiter(self) -> None:
        """Test reading CSV with semicolon delimiter."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write("a;b;c\n1;2;3\n4;5;6\n")
            path = Path(f.name)

        columns, n_rows, sample = read_csv_header_and_sample(path, "utf-8")
        path.unlink()

        assert columns == ("a", "b", "c")
        assert n_rows == 2
        assert sample == (("1", "2", "3"), ("4", "5", "6"))

    def test_tab_delimiter(self) -> None:
        """Test reading CSV with tab delimiter."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write("a\tb\tc\n1\t2\t3\n")
            path = Path(f.name)

        columns, n_rows, sample = read_csv_header_and_sample(path, "utf-8")
        path.unlink()

        assert columns == ("a", "b", "c")
        assert n_rows == 1
        assert sample == (("1", "2", "3"),)

    def test_quoted_column_names(self) -> None:
        """Test reading CSV with quoted column names."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write('"a","b","c"\n"1","2","3"\n')
            path = Path(f.name)

        columns, n_rows, sample = read_csv_header_and_sample(path, "utf-8")
        path.unlink()

        assert columns == ("a", "b", "c")
        assert n_rows == 1
        assert sample == (("1", "2", "3"),)

    def test_large_csv_limits_sample_size(self) -> None:
        """Test that large CSV files are sampled to MAX_SAMPLE_ROWS.

        When a CSV has more than MAX_SAMPLE_ROWS (1000) data rows,
        only the first 1000 rows are sampled but total count is accurate.
        """
        n_rows = 1100  # More than MAX_SAMPLE_ROWS

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write("a,b\n")
            for i in range(n_rows):
                f.write(f"{i},value\n")
            path = Path(f.name)

        columns, total_rows, sample = read_csv_header_and_sample(path, "utf-8")
        path.unlink()

        assert columns == ("a", "b")
        assert total_rows == n_rows
        assert len(sample) == 1000  # MAX_SAMPLE_ROWS
        # First row should be index 0
        assert sample[0][0] == "0"


class TestReadDataHeaderAndSample:
    """Tests for read_data_header_and_sample function (space-delimited .data files)."""

    def test_basic_data_file(self) -> None:
        """Test reading basic space-delimited .data file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".data", delete=False, encoding="utf-8"
        ) as f:
            f.write("1 2 3 4\n5 6 7 8\n")
            path = Path(f.name)

        columns, n_rows, sample = read_data_header_and_sample(path, "utf-8")
        path.unlink()

        # .data files have no header, columns are X1, X2, ..., class (last is target)
        assert len(columns) == 4
        assert columns == ("X1", "X2", "X3", "class")
        assert n_rows == 2
        assert sample == (("1", "2", "3", "4"), ("5", "6", "7", "8"))

    def test_empty_data_file(self) -> None:
        """Test reading empty .data file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".data", delete=False, encoding="utf-8"
        ) as f:
            path = Path(f.name)

        columns, n_rows, _sample = read_data_header_and_sample(path, "utf-8")
        path.unlink()

        assert columns == ()
        assert n_rows == 0

    def test_data_file_with_target(self) -> None:
        """Test reading .data file where last column is target."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".data", delete=False, encoding="utf-8"
        ) as f:
            # Last column is the class (1 or 2)
            f.write("1 0.5 0.3 1\n2 0.6 0.4 2\n3 0.7 0.5 1\n")
            path = Path(f.name)

        columns, n_rows, _sample = read_data_header_and_sample(path, "utf-8")
        path.unlink()

        # Last column should be named 'class' for target detection
        assert columns[-1] == "class"
        assert n_rows == 3

    def test_single_row_data_file(self) -> None:
        """Test reading .data file with only one row (no loop iteration)."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".data", delete=False, encoding="utf-8"
        ) as f:
            f.write("1 2 3 4\n")
            path = Path(f.name)

        columns, n_rows, sample = read_data_header_and_sample(path, "utf-8")
        path.unlink()

        assert columns == ("X1", "X2", "X3", "class")
        assert n_rows == 1
        assert sample == (("1", "2", "3", "4"),)

    def test_large_data_file_limits_sample_size(self) -> None:
        """Test that large .data files are sampled to MAX_SAMPLE_ROWS.

        When a .data file has more than MAX_SAMPLE_ROWS (1000) data rows,
        only the first 1000 rows are sampled but total count is accurate.
        """
        n_rows = 1100  # More than MAX_SAMPLE_ROWS

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".data", delete=False, encoding="utf-8"
        ) as f:
            for i in range(n_rows):
                f.write(f"{i} value 1\n")
            path = Path(f.name)

        columns, total_rows, sample = read_data_header_and_sample(path, "utf-8")
        path.unlink()

        assert columns == ("X1", "X2", "class")
        assert total_rows == n_rows
        assert len(sample) == 1000  # MAX_SAMPLE_ROWS
        # First row should be index 0
        assert sample[0][0] == "0"
