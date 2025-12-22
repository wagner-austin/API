"""Tests for file format detection functions."""

from __future__ import annotations

import tempfile
from pathlib import Path

from scripts.discover_datasets import scanner as scanner_mod


class TestDetectFileFormat:
    """Tests for _detect_file_format function."""

    def test_csv_format(self) -> None:
        """Test CSV format detection."""
        result = scanner_mod._detect_file_format(Path("data.csv"))
        assert result == "csv"

    def test_csv_format_uppercase(self) -> None:
        """Test CSV format detection with uppercase extension."""
        result = scanner_mod._detect_file_format(Path("DATA.CSV"))
        assert result == "csv"

    def test_arff_format(self) -> None:
        """Test ARFF format detection."""
        result = scanner_mod._detect_file_format(Path("data.arff"))
        assert result == "arff"

    def test_xlsx_format(self) -> None:
        """Test XLSX format detection."""
        result = scanner_mod._detect_file_format(Path("data.xlsx"))
        assert result == "xlsx"

    def test_xls_format(self) -> None:
        """Test XLS format detection."""
        result = scanner_mod._detect_file_format(Path("data.xls"))
        assert result == "xls"

    def test_data_format(self) -> None:
        """Test .data format detection (space/tab delimited)."""
        result = scanner_mod._detect_file_format(Path("german.data"))
        assert result == "data"

    def test_unknown_format(self) -> None:
        """Test unknown format detection."""
        result = scanner_mod._detect_file_format(Path("data.txt"))
        assert result == "unknown"


class TestFindDataFile:
    """Tests for _find_data_file function."""

    def test_no_data_files(self) -> None:
        """Test when folder has no data files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "readme.txt").write_text("readme")

            result, message = scanner_mod._find_data_file(folder)

            assert result is None
            assert message == "No data files found"

    def test_single_csv_file(self) -> None:
        """Test when folder has single CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            csv_file = folder / "data.csv"
            csv_file.write_text("a,b\n1,2\n")

            result, message = scanner_mod._find_data_file(folder)

            assert result == csv_file
            assert message == "Single data file found"

    def test_prefers_data_csv(self) -> None:
        """Test preference for data.csv among multiple files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "other.csv").write_text("a,b\n1,2\n")
            data_csv = folder / "data.csv"
            data_csv.write_text("a,b\n1,2\n")

            result, message = scanner_mod._find_data_file(folder)

            assert result == data_csv
            assert "Selected data.csv" in message

    def test_prefers_train_csv(self) -> None:
        """Test preference for train.csv among multiple files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "other.csv").write_text("a,b\n1,2\n")
            train_csv = folder / "train.csv"
            train_csv.write_text("a,b\n1,2\n")

            result, message = scanner_mod._find_data_file(folder)

            assert result == train_csv
            assert "Selected train.csv" in message

    def test_selects_largest_file(self) -> None:
        """Test selecting largest file when no preferred names match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            small = folder / "small.csv"
            small.write_text("a\n1\n")
            large = folder / "large.csv"
            large.write_text("a,b,c,d,e\n1,2,3,4,5\n" * 100)

            result, message = scanner_mod._find_data_file(folder)

            assert result == large
            assert "Selected largest file" in message

    def test_selects_largest_file_second_larger(self) -> None:
        """Test that second file is selected when it's larger than first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            first = folder / "aaa.csv"
            first.write_text("x\n1\n")
            second = folder / "bbb.csv"
            second.write_text("a,b,c,d,e\n1,2,3,4,5\n" * 100)

            result, message = scanner_mod._find_data_file(folder)

            assert result == second
            assert "Selected largest file" in message

    def test_finds_arff_file(self) -> None:
        """Test finding ARFF file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            arff_file = folder / "data.arff"
            arff_file.write_text("@relation test\n@data\n")

            result, _message = scanner_mod._find_data_file(folder)

            assert result == arff_file

    def test_finds_excel_file(self) -> None:
        """Test finding Excel file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            xlsx_file = folder / "data.xlsx"
            xlsx_file.write_bytes(b"PK")

            result, _message = scanner_mod._find_data_file(folder)

            assert result == xlsx_file

    def test_finds_data_file(self) -> None:
        """Test finding .data file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            data_file = folder / "german.data"
            data_file.write_text("1 2 3 4\n5 6 7 8\n")

            result, _message = scanner_mod._find_data_file(folder)

            assert result == data_file
