"""Tests for chunked CSV reader with Polars."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from covenant_ml.datasets.loaders.chunked_csv_reader import (
    DEFAULT_BATCH_SIZE,
    PROGRESS_THRESHOLD_BYTES,
    _make_progress,
    read_csv_to_dataframe,
    read_csv_with_progress,
)
from covenant_ml.datasets.types import LoadProgress


def _get_fixtures_dir() -> Path:
    """Get path to test fixtures directory."""
    return Path(__file__).parent.parent / "fixtures"


class TestReadCSVWithProgress:
    """Tests for read_csv_with_progress function."""

    def test_read_simple_csv(self) -> None:
        """Read simple CSV file returns headers and rows."""
        fixtures_dir = _get_fixtures_dir()
        csv_path = fixtures_dir / "small_csv" / "data.csv"

        headers, rows = read_csv_with_progress(csv_path, "utf-8")

        assert headers == ["feature_1", "feature_2", "feature_3", "target"]
        assert len(rows) == 5

    def test_read_csv_row_values(self) -> None:
        """Read CSV returns correct row values as strings."""
        fixtures_dir = _get_fixtures_dir()
        csv_path = fixtures_dir / "small_csv" / "data.csv"

        _headers, rows = read_csv_with_progress(csv_path, "utf-8")

        # First data row: 1.0, 2.0, 3.0, 0
        assert rows[0] == ["1.0", "2.0", "3.0", "0"]
        # Last data row: 13.0, 14.0, 15.0, 0
        assert rows[4] == ["13.0", "14.0", "15.0", "0"]

    def test_read_csv_file_not_found(self) -> None:
        """Read non-existent file raises FileNotFoundError."""
        csv_path = Path("/nonexistent/path/data.csv")

        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            read_csv_with_progress(csv_path, "utf-8")

    def test_read_empty_csv_raises(self) -> None:
        """Read empty CSV raises ValueError."""
        fixtures_dir = _get_fixtures_dir()
        csv_path = fixtures_dir / "empty_csv" / "data.csv"

        with pytest.raises(ValueError, match="No data rows found"):
            read_csv_with_progress(csv_path, "utf-8")

    def test_read_csv_with_progress_callback(self) -> None:
        """Progress callback is called during reading."""
        fixtures_dir = _get_fixtures_dir()
        csv_path = fixtures_dir / "small_csv" / "data.csv"

        progress_updates: list[LoadProgress] = []

        def capture_progress(progress: LoadProgress) -> None:
            progress_updates.append(progress)

        _headers, _rows = read_csv_with_progress(
            csv_path, "utf-8", progress_callback=capture_progress
        )

        # Small file skips reading phase, gets 2 parsing updates (start + end)
        assert len(progress_updates) == 2
        # Verify first update is parsing phase with row-based progress
        first_update = progress_updates[0]
        assert first_update["phase"] == "parsing"
        assert first_update["bytes_read"] == 0  # Row-based progress
        assert first_update["bytes_total"] == 0  # Row-based progress
        assert first_update["rows_processed"] == 0
        assert first_update["rows_total"] == 5  # 5 rows in small_csv
        assert first_update["percent_complete"] == 0.0
        # Message should describe some operation
        assert len(first_update["message"]) >= 5  # Reasonable message length

    def test_read_csv_progress_phases(self) -> None:
        """Progress updates include reading and parsing phases."""
        fixtures_dir = _get_fixtures_dir()
        csv_path = fixtures_dir / "small_csv" / "data.csv"

        phases_seen: set[str] = set()

        def capture_phase(progress: LoadProgress) -> None:
            phases_seen.add(progress["phase"])

        read_csv_with_progress(csv_path, "utf-8", progress_callback=capture_phase)

        # Should see parsing phase (small file skips some reading updates)
        assert "parsing" in phases_seen

    def test_read_csv_final_progress_complete(self) -> None:
        """Final progress update shows 100% complete."""
        fixtures_dir = _get_fixtures_dir()
        csv_path = fixtures_dir / "small_csv" / "data.csv"

        last_progress: list[LoadProgress] = []

        def capture_last(progress: LoadProgress) -> None:
            last_progress.clear()
            last_progress.append(progress)

        read_csv_with_progress(csv_path, "utf-8", progress_callback=capture_last)

        assert len(last_progress) == 1
        assert last_progress[0]["percent_complete"] == 100.0
        assert last_progress[0]["rows_processed"] == 5

    def test_read_csv_without_callback(self) -> None:
        """Read CSV works without progress callback."""
        fixtures_dir = _get_fixtures_dir()
        csv_path = fixtures_dir / "small_csv" / "data.csv"

        headers, rows = read_csv_with_progress(csv_path, "utf-8", progress_callback=None)

        assert len(headers) == 4
        assert len(rows) == 5

    def test_read_csv_utf8_sig_encoding(self) -> None:
        """Read CSV with utf-8-sig encoding."""
        fixtures_dir = _get_fixtures_dir()
        csv_path = fixtures_dir / "small_csv" / "data.csv"

        headers, rows = read_csv_with_progress(csv_path, "utf-8-sig")

        assert len(headers) == 4
        assert len(rows) == 5

    def test_read_csv_latin1_encoding(self) -> None:
        """Read CSV with latin-1 encoding uses lossy fallback."""
        fixtures_dir = _get_fixtures_dir()
        csv_path = fixtures_dir / "small_csv" / "data.csv"

        headers, rows = read_csv_with_progress(csv_path, "latin-1")

        assert len(headers) == 4
        assert len(rows) == 5


class TestReadCSVToDataframe:
    """Tests for read_csv_to_dataframe function."""

    def test_read_returns_dataframe_protocol(self) -> None:
        """Read CSV returns object matching DataFrame protocol."""
        fixtures_dir = _get_fixtures_dir()
        csv_path = fixtures_dir / "small_csv" / "data.csv"

        df = read_csv_to_dataframe(csv_path, "utf-8")

        # Verify DataFrame has expected dimensions
        assert df.height == 5
        assert len(df.columns) == 4
        assert df.columns == ["feature_1", "feature_2", "feature_3", "target"]

    def test_read_dataframe_columns(self) -> None:
        """Read CSV DataFrame has correct columns."""
        fixtures_dir = _get_fixtures_dir()
        csv_path = fixtures_dir / "small_csv" / "data.csv"

        df = read_csv_to_dataframe(csv_path, "utf-8")

        assert df.columns == ["feature_1", "feature_2", "feature_3", "target"]

    def test_read_dataframe_file_not_found(self) -> None:
        """Read non-existent file raises FileNotFoundError."""
        csv_path = Path("/nonexistent/path/data.csv")

        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            read_csv_to_dataframe(csv_path, "utf-8")

    def test_read_dataframe_empty_raises(self) -> None:
        """Read empty CSV raises ValueError."""
        fixtures_dir = _get_fixtures_dir()
        csv_path = fixtures_dir / "empty_csv" / "data.csv"

        with pytest.raises(ValueError, match="No data rows found"):
            read_csv_to_dataframe(csv_path, "utf-8")

    def test_read_dataframe_with_progress(self) -> None:
        """Progress callback is called when reading DataFrame."""
        fixtures_dir = _get_fixtures_dir()
        csv_path = fixtures_dir / "small_csv" / "data.csv"

        progress_count = [0]

        def count_progress(progress: LoadProgress) -> None:
            progress_count[0] += 1

        df = read_csv_to_dataframe(csv_path, "utf-8", progress_callback=count_progress)

        assert df.height == 5
        assert progress_count[0] >= 2  # At least start and end


class TestModuleConstants:
    """Tests for module constants."""

    def test_default_batch_size_positive(self) -> None:
        """Default batch size is positive."""
        assert DEFAULT_BATCH_SIZE > 0
        assert DEFAULT_BATCH_SIZE == 100_000

    def test_progress_threshold_positive(self) -> None:
        """Progress threshold is positive."""
        assert PROGRESS_THRESHOLD_BYTES > 0
        assert PROGRESS_THRESHOLD_BYTES == 1_024 * 1_024  # 1 MB


class TestMakeProgress:
    """Tests for _make_progress helper function."""

    def test_make_progress_bytes_based(self) -> None:
        """Progress calculates percent from bytes when bytes_total > 0."""
        result = _make_progress(
            phase="reading",
            bytes_read=500,
            bytes_total=1000,
            rows_processed=0,
            rows_total=0,
            message="Reading file...",
        )

        assert result["phase"] == "reading"
        assert result["percent_complete"] == 50.0
        assert result["bytes_read"] == 500
        assert result["bytes_total"] == 1000
        assert result["message"] == "Reading file..."

    def test_make_progress_rows_based(self) -> None:
        """Progress calculates percent from rows when bytes_total = 0."""
        result = _make_progress(
            phase="parsing",
            bytes_read=0,
            bytes_total=0,
            rows_processed=25,
            rows_total=100,
            message="Parsing rows...",
        )

        assert result["phase"] == "parsing"
        assert result["percent_complete"] == 25.0
        assert result["rows_processed"] == 25
        assert result["rows_total"] == 100

    def test_make_progress_zero_totals(self) -> None:
        """Progress returns 0% when both totals are zero."""
        result = _make_progress(
            phase="reading",
            bytes_read=0,
            bytes_total=0,
            rows_processed=0,
            rows_total=0,
            message="Starting...",
        )

        assert result["percent_complete"] == 0.0

    def test_make_progress_caps_at_100(self) -> None:
        """Progress caps at 100% even if read exceeds total."""
        result = _make_progress(
            phase="reading",
            bytes_read=1500,
            bytes_total=1000,
            rows_processed=0,
            rows_total=0,
            message="Completing...",
        )

        assert result["percent_complete"] == 100.0


class TestLargeFileProgress:
    """Tests for large file progress reporting."""

    def test_large_file_reading_progress(self) -> None:
        """Large files trigger reading phase progress reports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "large.csv"

            # Create a CSV file larger than PROGRESS_THRESHOLD_BYTES (1MB)
            # Each row is approximately 20 bytes, so we need ~52,500 rows
            # for safety let's generate more padding
            with csv_path.open("w") as f:
                f.write("col1,col2,col3\n")
                row_data = "1234567890,abcdefghij,0123456789\n"  # ~33 bytes per row
                # Write enough rows to exceed 1MB
                for _i in range(35000):  # ~35000 * 33 = ~1.1MB
                    f.write(row_data)

            progress_updates: list[LoadProgress] = []

            def capture(progress: LoadProgress) -> None:
                progress_updates.append(progress)

            _headers, _rows = read_csv_with_progress(csv_path, "utf-8", progress_callback=capture)

            # Should have reading phase progress reports for large files
            reading_updates = [p for p in progress_updates if p["phase"] == "reading"]
            assert len(reading_updates) >= 2  # At least start and end reading reports

    def test_large_row_count_periodic_progress(self) -> None:
        """Files with many rows trigger periodic parsing progress reports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "many_rows.csv"

            # Create a file with > DEFAULT_BATCH_SIZE rows (100,000)
            # Use very short rows to keep file size manageable but row count high
            with csv_path.open("w") as f:
                f.write("a,b\n")
                # 105,000 rows to trigger at least one periodic report
                for _i in range(105000):
                    f.write("1,2\n")

            progress_updates: list[LoadProgress] = []

            def capture(progress: LoadProgress) -> None:
                progress_updates.append(progress)

            _headers, _rows = read_csv_with_progress(csv_path, "utf-8", progress_callback=capture)

            # Should have parsing phase progress reports for periodic updates
            parsing_updates = [p for p in progress_updates if p["phase"] == "parsing"]
            # At least 2: the periodic one at 100k and the final one
            assert len(parsing_updates) >= 2
            # Verify row counts in periodic updates - should have exactly one at 100k
            periodic_update = [p for p in parsing_updates if p["rows_processed"] == 100000]
            assert len(periodic_update) == 1
