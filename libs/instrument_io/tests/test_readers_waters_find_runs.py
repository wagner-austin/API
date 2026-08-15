"""Tests for readers waters: WatersReaderFindRuns."""

from __future__ import annotations

from pathlib import Path

import pytest

from instrument_io._exceptions import WatersReadError
from instrument_io.readers.waters import (
    WatersReader,
)


class TestWatersReaderFindRuns:
    """Test WatersReader.find_runs method."""

    def test_find_runs_not_directory_raises(self, tmp_path: Path) -> None:
        """Test find_runs raises when path is not a directory."""
        not_dir = tmp_path / "not_a_directory.raw"
        not_dir.write_text("not a directory")

        reader = WatersReader()
        with pytest.raises(WatersReadError) as exc_info:
            reader.find_runs(not_dir)
        assert "Not a directory" in str(exc_info.value)

    def test_find_runs_empty_directory(self, tmp_path: Path) -> None:
        """Test find_runs returns empty list for empty directory."""
        reader = WatersReader()
        runs = reader.find_runs(tmp_path)
        assert runs == []

    def test_find_runs_with_raw_directories(self, tmp_path: Path) -> None:
        """Test find_runs finds .raw directories."""
        # Create .raw directories
        raw1 = tmp_path / "sample1.raw"
        raw1.mkdir()
        raw2 = tmp_path / "sample2.raw"
        raw2.mkdir()

        reader = WatersReader()
        runs = reader.find_runs(tmp_path)

        assert len(runs) == 2
        run_ids = [r["run_id"] for r in runs]
        assert "sample1" in run_ids
        assert "sample2" in run_ids

    def test_find_runs_with_data_files(self, tmp_path: Path) -> None:
        """Test find_runs detects TIC, MS, and DAD files."""
        raw_dir = tmp_path / "sample.raw"
        raw_dir.mkdir()

        # Create files that indicate different data types
        (raw_dir / "_FUNC001.DAT").write_text("tic data")  # TIC
        (raw_dir / "ms_data.idx").write_text("ms data")  # MS
        (raw_dir / "PDA_spectrum.dat").write_text("pda data")  # DAD

        reader = WatersReader()
        runs = reader.find_runs(tmp_path)

        assert len(runs) == 1
        run = runs[0]
        assert run["has_tic"] is True
        assert run["has_ms"] is True
        assert run["has_dad"] is True
        assert run["file_count"] == 3

    def test_find_runs_detects_uv_as_dad(self, tmp_path: Path) -> None:
        """Test find_runs detects UV files as DAD."""
        raw_dir = tmp_path / "sample.raw"
        raw_dir.mkdir()

        (raw_dir / "uv_data.dat").write_text("uv data")

        reader = WatersReader()
        runs = reader.find_runs(tmp_path)

        assert len(runs) == 1
        assert runs[0]["has_dad"] is True

    def test_find_runs_skips_raw_file(self, tmp_path: Path) -> None:
        """Test find_runs skips .raw files (not directories)."""
        # Create a .raw file (not directory)
        raw_file = tmp_path / "sample.raw"
        raw_file.write_text("not a directory")

        reader = WatersReader()
        runs = reader.find_runs(tmp_path)

        assert runs == []

    def test_find_runs_extracts_site_from_parent(self, tmp_path: Path) -> None:
        """Test find_runs extracts site from parent directory name."""
        site_dir = tmp_path / "site_A"
        site_dir.mkdir()
        raw_dir = site_dir / "sample.raw"
        raw_dir.mkdir()

        reader = WatersReader()
        runs = reader.find_runs(tmp_path)

        assert len(runs) == 1
        assert runs[0]["site"] == "site_A"

    def test_find_runs_ignores_subdirectories(self, tmp_path: Path) -> None:
        """Test find_runs only counts files, not subdirectories."""
        raw_dir = tmp_path / "sample.raw"
        raw_dir.mkdir()

        # Create a file and a subdirectory
        (raw_dir / "data.dat").write_text("data")
        (raw_dir / "subdir").mkdir()

        reader = WatersReader()
        runs = reader.find_runs(tmp_path)

        assert len(runs) == 1
        # Only the file should be counted, not the subdirectory
        assert runs[0]["file_count"] == 1
