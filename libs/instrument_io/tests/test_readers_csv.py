"""Tests for readers.csv module."""

from __future__ import annotations

from pathlib import Path

from instrument_io.readers.csv import (
    CSVChromatogramReader,
    _build_chromatogram_meta,
    _is_csv_file,
)


# Test _is_csv_file
class TestIsCsvFile:
    """Tests for _is_csv_file function."""

    def test_csv_file(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.csv"
        csv_file.touch()
        assert _is_csv_file(csv_file) is True

    def test_tsv_file(self, tmp_path: Path) -> None:
        tsv_file = tmp_path / "test.tsv"
        tsv_file.touch()
        assert _is_csv_file(tsv_file) is True

    def test_txt_file(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "test.txt"
        txt_file.touch()
        assert _is_csv_file(txt_file) is True

    def test_uppercase_extension(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.CSV"
        csv_file.touch()
        assert _is_csv_file(csv_file) is True

    def test_non_csv_file(self, tmp_path: Path) -> None:
        other_file = tmp_path / "test.xlsx"
        other_file.touch()
        assert _is_csv_file(other_file) is False

    def test_directory_returns_false(self, tmp_path: Path) -> None:
        csv_dir = tmp_path / "test.csv"
        csv_dir.mkdir()
        assert _is_csv_file(csv_dir) is False

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "nonexistent.csv"
        assert _is_csv_file(nonexistent) is False


# Test _build_chromatogram_meta
class TestBuildChromatogramMeta:
    """Tests for _build_chromatogram_meta function."""

    def test_builds_meta(self) -> None:
        meta = _build_chromatogram_meta("/path/to/file.csv", "TIC", "CSV")
        assert meta["source_path"] == "/path/to/file.csv"
        assert meta["signal_type"] == "TIC"
        assert meta["detector"] == "CSV"
        assert meta["instrument"] == ""
        assert meta["method_name"] == ""
        assert meta["sample_name"] == ""
        assert meta["acquisition_date"] == ""


# Test CSVChromatogramReader class
class TestCSVChromatogramReader:
    """Tests for CSVChromatogramReader class."""

    def test_supports_format_csv(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.csv"
        csv_file.touch()
        reader = CSVChromatogramReader()
        assert reader.supports_format(csv_file) is True

    def test_supports_format_tsv(self, tmp_path: Path) -> None:
        tsv_file = tmp_path / "test.tsv"
        tsv_file.touch()
        reader = CSVChromatogramReader()
        assert reader.supports_format(tsv_file) is True

    def test_supports_format_txt(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "test.txt"
        txt_file.touch()
        reader = CSVChromatogramReader()
        assert reader.supports_format(txt_file) is True

    def test_supports_format_xlsx_false(self, tmp_path: Path) -> None:
        xlsx_file = tmp_path / "test.xlsx"
        xlsx_file.touch()
        reader = CSVChromatogramReader()
        assert reader.supports_format(xlsx_file) is False

    def test_detect_columns_empty_file_raises(self, tmp_path: Path) -> None:
        import pytest

        from instrument_io._exceptions import CSVReadError

        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")
        reader = CSVChromatogramReader()
        with pytest.raises(CSVReadError) as exc_info:
            reader.detect_columns(csv_file)
        assert "Empty file" in str(exc_info.value)

    def test_detect_columns_whitespace_only_headers_raises(self, tmp_path: Path) -> None:
        import pytest

        from instrument_io._exceptions import CSVReadError

        csv_file = tmp_path / "whitespace.csv"
        csv_file.write_text("   ,   ,   \n")
        reader = CSVChromatogramReader()
        with pytest.raises(CSVReadError) as exc_info:
            reader.detect_columns(csv_file)
        assert "No columns found" in str(exc_info.value)

    def test_detect_delimiter_empty_file_raises(self, tmp_path: Path) -> None:
        import pytest

        from instrument_io._exceptions import CSVReadError

        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")
        reader = CSVChromatogramReader()
        with pytest.raises(CSVReadError) as exc_info:
            reader.detect_delimiter(csv_file)
        assert "Empty file" in str(exc_info.value)

    def test_read_chromatogram_empty_file_raises(self, tmp_path: Path) -> None:
        import pytest

        from instrument_io._exceptions import CSVReadError

        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")
        reader = CSVChromatogramReader()
        with pytest.raises(CSVReadError) as exc_info:
            reader.read_chromatogram(csv_file)
        assert "Empty file" in str(exc_info.value)

    def test_read_chromatogram_header_only_raises(self, tmp_path: Path) -> None:
        import pytest

        from instrument_io._exceptions import CSVReadError

        csv_file = tmp_path / "header_only.csv"
        csv_file.write_text("Time,Intensity\n")
        reader = CSVChromatogramReader()
        with pytest.raises(CSVReadError) as exc_info:
            reader.read_chromatogram(csv_file)
        assert "no data rows" in str(exc_info.value)

    def test_read_chromatogram_success(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("Time,Intensity\n0.0,100.0\n1.0,200.0\n")
        reader = CSVChromatogramReader()
        result = reader.read_chromatogram(csv_file)
        assert result["data"]["retention_times"] == [0.0, 1.0]
        assert result["data"]["intensities"] == [100.0, 200.0]

    def test_detect_columns_success(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("Time,Intensity,Extra\n0.0,100.0,A\n")
        reader = CSVChromatogramReader()
        columns = reader.detect_columns(csv_file)
        assert columns == ["Time", "Intensity", "Extra"]

    def test_detect_delimiter_tab(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.tsv"
        csv_file.write_text("Time\tIntensity\n0.0\t100.0\n")
        reader = CSVChromatogramReader()
        delimiter = reader.detect_delimiter(csv_file)
        assert delimiter == "\t"

    def test_read_chromatogram_with_blank_lines(self) -> None:
        """Test reading CSV with blank lines between data rows.

        Uses real fixture file with embedded blank lines.
        Covers branch 48->46 (empty line skipped).
        """
        fixtures = Path(__file__).parent / "fixtures"
        csv_file = fixtures / "chromatogram_with_blanks.csv"
        reader = CSVChromatogramReader()
        result = reader.read_chromatogram(csv_file)
        # Should have 3 data points (blank lines skipped)
        assert result["data"]["retention_times"] == [0.0, 0.5, 1.0]
        assert result["data"]["intensities"] == [1000.0, 1200.0, 3500.0]
