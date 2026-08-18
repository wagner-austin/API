"""Tests for the growth-policy file readers.

Every test writes a real file under ``tmp_path`` and reads it back through the
real ``polars`` boundary. Nothing is substituted: the point of this module is
that a real reader is reached and narrowed, so reading through a stand-in would
test the stand-in's narrowing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.growth_policy.reading import (
    read_frame,
    read_numeric_columns,
    read_text_column,
    read_whitespace_rows,
    require_columns,
)
from covenant_ml.growth_policy.types import (
    ERR_EMPTY_DATASET,
    ERR_MISSING_COLUMN,
    ERR_MISSING_VALUE,
)

from .numeric import as_float_list


class TestReadFrame:
    """Parsing a CSV into a frame."""

    def test_reads_every_data_row(self, tmp_path: Path) -> None:
        """The header should not be counted as a row."""
        target = tmp_path / "rows.csv"
        target.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

        assert len(read_frame(target)) == 2

    def test_exposes_the_header_as_columns(self, tmp_path: Path) -> None:
        """Column names should follow the header, in order."""
        target = tmp_path / "rows.csv"
        target.write_text("a,b\n1,2\n", encoding="utf-8")

        assert read_frame(target).columns == ["a", "b"]

    def test_rejects_a_header_with_no_rows(self, tmp_path: Path) -> None:
        """A header-only file carries no data and should be refused."""
        target = tmp_path / "empty.csv"
        target.write_text("a,b\n", encoding="utf-8")

        with pytest.raises(ValueError, match=ERR_EMPTY_DATASET):
            read_frame(target)


class TestRequireColumns:
    """Header checking."""

    def test_accepts_a_complete_header(self, tmp_path: Path) -> None:
        """Every required column present should pass without raising."""
        target = tmp_path / "rows.csv"
        target.write_text("a,b\n1,2\n", encoding="utf-8")

        require_columns(read_frame(target), ("a", "b"), target)

    def test_names_the_missing_column(self, tmp_path: Path) -> None:
        """The error should identify which column is absent."""
        target = tmp_path / "rows.csv"
        target.write_text("a\n1\n", encoding="utf-8")

        with pytest.raises(ValueError, match=ERR_MISSING_COLUMN) as excinfo:
            require_columns(read_frame(target), ("a", "b"), target)

        assert "'b'" in str(excinfo.value)


class TestReadColumns:
    """Taking columns out of a frame."""

    def test_reads_numeric_columns_in_the_requested_order(self, tmp_path: Path) -> None:
        """Selection order should drive the matrix's column order."""
        target = tmp_path / "rows.csv"
        target.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

        matrix = read_numeric_columns(read_frame(target), ["b", "a"])

        first_row: NDArray[np.float64] = matrix[0]
        second_row: NDArray[np.float64] = matrix[1]
        assert as_float_list(first_row) == [2.0, 1.0]
        assert as_float_list(second_row) == [4.0, 3.0]

    def test_rejects_a_null_cell_from_a_short_row(self, tmp_path: Path) -> None:
        """Polars pads a short row rather than failing, so this layer must catch it."""
        target = tmp_path / "rows.csv"
        target.write_text("a,b\n1,2\n3\n", encoding="utf-8")

        with pytest.raises(ValueError, match=ERR_MISSING_VALUE) as excinfo:
            read_numeric_columns(read_frame(target), ["a", "b"])

        assert "1 of 4" in str(excinfo.value)

    def test_rejects_a_blank_cell(self, tmp_path: Path) -> None:
        """A blank field is absent data rather than a zero."""
        target = tmp_path / "rows.csv"
        target.write_text("a,b\n1,\n3,4\n", encoding="utf-8")

        with pytest.raises(ValueError, match=ERR_MISSING_VALUE):
            read_numeric_columns(read_frame(target), ["a", "b"])

    def test_accepts_a_fully_populated_selection(self, tmp_path: Path) -> None:
        """A complete numeric selection should pass the finiteness check."""
        target = tmp_path / "rows.csv"
        target.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

        matrix = read_numeric_columns(read_frame(target), ["a", "b"])

        assert matrix.shape == (2, 2)

    def test_reads_a_text_column(self, tmp_path: Path) -> None:
        """A string column should come back as a list of strings."""
        target = tmp_path / "rows.csv"
        target.write_text("name,value\nalpha,1\nbeta,2\n", encoding="utf-8")

        assert read_text_column(read_frame(target), "name") == ["alpha", "beta"]

    def test_reads_a_quoted_field_containing_a_comma(self, tmp_path: Path) -> None:
        """Quoting must be honoured, or a company name would split into two rows."""
        target = tmp_path / "rows.csv"
        target.write_text('name,value\n"beta, inc",1\n', encoding="utf-8")

        assert read_text_column(read_frame(target), "name") == ["beta, inc"]


class TestReadWhitespaceRows:
    """Reading a whitespace-separated file."""

    def test_splits_on_runs_of_whitespace(self, tmp_path: Path) -> None:
        """Multiple spaces should not produce empty fields."""
        target = tmp_path / "german.data"
        target.write_text("A11   6  1169 1\n", encoding="utf-8")

        assert read_whitespace_rows(target) == [["A11", "6", "1169", "1"]]

    def test_drops_blank_lines(self, tmp_path: Path) -> None:
        """A blank line should not become an empty row."""
        target = tmp_path / "german.data"
        target.write_text("A11 1 2\n\n   \nA12 3 4\n", encoding="utf-8")

        assert read_whitespace_rows(target) == [["A11", "1", "2"], ["A12", "3", "4"]]

    def test_rejects_an_empty_file(self, tmp_path: Path) -> None:
        """A file with no non-blank lines should be refused."""
        target = tmp_path / "german.data"
        target.write_text("\n  \n", encoding="utf-8")

        with pytest.raises(ValueError, match=ERR_EMPTY_DATASET):
            read_whitespace_rows(target)
