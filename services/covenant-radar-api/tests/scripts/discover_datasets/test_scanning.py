"""Tests for dataset folder and directory scanning functions."""

from __future__ import annotations

import tempfile
from pathlib import Path

from scripts.discover_datasets import scanner as scanner_mod

from .conftest import get_workbook_ctor, get_xlwt_workbook_ctor


class TestScanDatasetFolder:
    """Tests for scan_dataset_folder function."""

    def test_empty_folder(self) -> None:
        """Test scanning empty folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)

            result = scanner_mod.scan_dataset_folder(folder)

            assert result["status"] == "error"
            assert result["message"] == "No data files found"
            assert result["file_name"] == ""

    def test_csv_with_target(self) -> None:
        """Test scanning folder with CSV containing target column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            csv_file = folder / "data.csv"
            csv_file.write_text("id,feature1,target\n1,0.5,0\n2,0.7,1\n")

            result = scanner_mod.scan_dataset_folder(folder)

            assert result["status"] == "success"
            assert result["file_name"] == "data.csv"
            assert result["file_format"] == "csv"
            assert result["n_rows"] == 2
            assert result["n_columns"] == 3
            assert result["recommended_target"] == "target"
            assert "id" in result["recommended_exclude"]

    def test_csv_no_target(self) -> None:
        """Test scanning folder with CSV without target column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            csv_file = folder / "data.csv"
            csv_file.write_text("feature1,feature2,feature3\n1,2,3\n")

            result = scanner_mod.scan_dataset_folder(folder)

            assert result["status"] == "warning"
            assert "No target column" in result["message"]

    def test_csv_non_binary_target(self) -> None:
        """Test scanning folder with non-binary target."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            csv_file = folder / "data.csv"
            csv_file.write_text("feature,class\n1,A\n2,B\n3,C\n")

            result = scanner_mod.scan_dataset_folder(folder)

            assert result["status"] == "warning"
            assert "No binary target" in result["message"]

    def test_arff_file(self) -> None:
        """Test scanning folder with ARFF file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            arff_file = folder / "data.arff"
            arff_file.write_text(
                """@relation test
@attribute feature numeric
@attribute target {0,1}
@data
1.0,0
2.0,1
"""
            )

            result = scanner_mod.scan_dataset_folder(folder)

            assert result["file_format"] == "arff"
            assert result["n_rows"] == 2
            assert result["recommended_target"] == "target"

    def test_excel_file(self) -> None:
        """Test scanning folder with Excel file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            xlsx_file = folder / "data.xlsx"
            ctor = get_workbook_ctor()
            wb = ctor()
            ws = wb.active
            ws.cell(row=1, column=1, value="feature")
            ws.cell(row=1, column=2, value="target")
            ws.cell(row=2, column=1, value=1)
            ws.cell(row=2, column=2, value=0)
            ws.cell(row=3, column=1, value=2)
            ws.cell(row=3, column=2, value=1)
            wb.save(xlsx_file)
            wb.close()

            result = scanner_mod.scan_dataset_folder(folder)

            assert result["file_format"] == "xlsx"
            assert result["n_rows"] == 2
            assert result["recommended_target"] == "target"
            assert result["status"] == "success"

    def test_xls_file(self) -> None:
        """Test scanning folder with legacy .xls file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            xls_file = folder / "data.xls"
            ctor = get_xlwt_workbook_ctor()
            wb = ctor()
            ws = wb.add_sheet("Sheet1")
            ws.write(0, 0, "feature")
            ws.write(0, 1, "target")
            ws.write(1, 0, 1)
            ws.write(1, 1, 0)
            ws.write(2, 0, 2)
            ws.write(2, 1, 1)
            wb.save(str(xls_file))

            result = scanner_mod.scan_dataset_folder(folder)

            assert result["file_format"] == "xls"
            assert result["n_rows"] == 2
            assert result["recommended_target"] == "target"
            assert result["status"] == "success"

    def test_data_file(self) -> None:
        """Test scanning folder with .data file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            data_file = folder / "german.data"
            # Last column is class (1 or 2)
            data_file.write_text("1 0.5 0.3 1\n2 0.6 0.4 2\n3 0.7 0.5 1\n")

            result = scanner_mod.scan_dataset_folder(folder)

            assert result["file_format"] == "data"
            assert result["n_rows"] == 3
            # Should detect class column

    def test_no_supported_files(self) -> None:
        """Test scanning folder with no supported file formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            parquet_file = folder / "data.parquet"
            parquet_file.write_bytes(b"PAR1")

            result = scanner_mod.scan_dataset_folder(folder)

            assert result["status"] == "error"
            assert "No data files found" in result["message"]


class TestScanExternalDir:
    """Tests for scan_external_dir function."""

    def test_nonexistent_directory(self) -> None:
        """Test scanning non-existent directory."""
        result = scanner_mod.scan_external_dir(Path("/nonexistent/path"))

        assert result["n_total"] == 0
        assert len(result["datasets"]) == 0

    def test_empty_directory(self) -> None:
        """Test scanning empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            external_dir = Path(tmpdir)

            result = scanner_mod.scan_external_dir(external_dir)

            assert result["n_total"] == 0

    def test_multiple_datasets(self) -> None:
        """Test scanning directory with multiple dataset folders."""
        with tempfile.TemporaryDirectory() as tmpdir:
            external_dir = Path(tmpdir)

            ds1 = external_dir / "dataset1"
            ds1.mkdir()
            (ds1 / "data.csv").write_text("feature,target\n1,0\n2,1\n")

            ds2 = external_dir / "dataset2"
            ds2.mkdir()
            (ds2 / "data.csv").write_text("a,b\n1,2\n")

            ds3 = external_dir / "dataset3"
            ds3.mkdir()

            result = scanner_mod.scan_external_dir(external_dir)

            assert result["n_total"] == 3
            assert result["n_success"] == 1
            assert result["n_warning"] == 1
            assert result["n_error"] == 1

    def test_ignores_hidden_folders(self) -> None:
        """Test that hidden folders are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            external_dir = Path(tmpdir)

            hidden = external_dir / ".hidden"
            hidden.mkdir()
            (hidden / "data.csv").write_text("a,target\n1,0\n")

            visible = external_dir / "visible"
            visible.mkdir()
            (visible / "data.csv").write_text("a,target\n1,0\n")

            result = scanner_mod.scan_external_dir(external_dir)

            assert result["n_total"] == 1
            assert result["datasets"][0]["folder_name"] == "visible"

    def test_ignores_files_at_root(self) -> None:
        """Test that files at root level are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            external_dir = Path(tmpdir)

            (external_dir / "readme.txt").write_text("readme")

            ds = external_dir / "dataset"
            ds.mkdir()
            (ds / "data.csv").write_text("a,target\n1,0\n")

            result = scanner_mod.scan_external_dir(external_dir)

            assert result["n_total"] == 1


class TestDetectTargetInfo:
    """Tests for _detect_target_info function."""

    def test_empty_positive_value_skips_ratio_calculation(self) -> None:
        """Test that empty positive value skips ratio calculation."""
        from scripts.discover_datasets.scanner import _detect_target_info
        from scripts.discover_datasets.types import TargetColumnCandidate

        # Create candidate with one empty value - empty string is alphabetically first
        # so it becomes the "positive" value, but empty string is falsy
        candidate: TargetColumnCandidate = {
            "column_name": "target",
            "unique_values": ("", "valid"),  # Empty string is positive
            "n_unique": 2,
            "is_binary": True,
        }
        candidates = (candidate,)
        sample_rows: tuple[tuple[str, ...], ...] = (
            ("1", ""),
            ("2", "valid"),
        )
        columns = ("feature", "target")

        result = _detect_target_info(candidates, "target", sample_rows, columns)

        # Empty positive value means ratio calculation is skipped
        assert result["positive_value"] == ""
        assert result["positive_ratio"] == 0.0

    def test_no_recommended_target_returns_empty_info(self) -> None:
        """Test that no recommended target returns empty target info."""
        from scripts.discover_datasets.scanner import _detect_target_info
        from scripts.discover_datasets.types import TargetColumnCandidate

        candidate: TargetColumnCandidate = {
            "column_name": "target",
            "unique_values": ("0", "1"),
            "n_unique": 2,
            "is_binary": True,
        }
        candidates = (candidate,)
        sample_rows: tuple[tuple[str, ...], ...] = (("1", "0"),)
        columns = ("feature", "target")

        result = _detect_target_info(candidates, "", sample_rows, columns)

        assert result["positive_value"] == ""
        assert result["negative_value"] == ""
        assert result["positive_ratio"] == 0.0


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports(self) -> None:
        """Test __all__ exports are available."""
        assert "scan_dataset_folder" in scanner_mod.__all__
        assert "scan_external_dir" in scanner_mod.__all__
