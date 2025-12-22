"""Tests for ARFF parsing functions."""

from __future__ import annotations

import tempfile
from pathlib import Path

from scripts.discover_datasets.parsers.arff import read_arff_header_and_sample


class TestReadArffHeaderAndSample:
    """Tests for read_arff_header_and_sample function."""

    def test_basic_arff(self) -> None:
        """Test reading basic ARFF file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".arff", delete=False, encoding="utf-8"
        ) as f:
            f.write(
                """@relation test
@attribute feature1 numeric
@attribute feature2 numeric
@attribute class {0,1}
@data
1.0,2.0,0
3.0,4.0,1
"""
            )
            path = Path(f.name)

        columns, n_rows, sample = read_arff_header_and_sample(path)
        path.unlink()

        assert columns == ("feature1", "feature2", "class")
        assert n_rows == 2
        assert sample == (("1.0", "2.0", "0"), ("3.0", "4.0", "1"))

    def test_arff_with_comments(self) -> None:
        """Test reading ARFF file with comments."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".arff", delete=False, encoding="utf-8"
        ) as f:
            f.write(
                """% This is a comment
@relation test
@attribute a numeric
@data
% Another comment
1.0
"""
            )
            path = Path(f.name)

        columns, n_rows, _sample = read_arff_header_and_sample(path)
        path.unlink()

        assert columns == ("a",)
        assert n_rows == 1

    def test_arff_empty_data(self) -> None:
        """Test reading ARFF file with no data rows."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".arff", delete=False, encoding="utf-8"
        ) as f:
            f.write(
                """@relation empty
@attribute a numeric
@data
"""
            )
            path = Path(f.name)

        columns, n_rows, _sample = read_arff_header_and_sample(path)
        path.unlink()

        assert columns == ("a",)
        assert n_rows == 0

    def test_arff_malformed_attribute(self) -> None:
        """Test reading ARFF file with malformed attribute line."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".arff", delete=False, encoding="utf-8"
        ) as f:
            f.write(
                """@relation test
@attribute
@attribute valid numeric
@data
1.0
"""
            )
            path = Path(f.name)

        columns, n_rows, _sample = read_arff_header_and_sample(path)
        path.unlink()

        assert columns == ("valid",)
        assert n_rows == 1

    def test_arff_large_file_samples_from_start_and_end(self) -> None:
        """Test ARFF with > MAX_SAMPLE_ROWS samples from both start and end.

        This ensures sorted/imbalanced datasets have representation from
        both the beginning and end of the data.
        """
        # MAX_SAMPLE_ROWS is 1000, so we need > 1000 rows
        n_rows = 1200

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".arff", delete=False, encoding="utf-8"
        ) as f:
            f.write("@relation large_test\n")
            f.write("@attribute value numeric\n")
            f.write("@attribute class {0,1}\n")
            f.write("@data\n")
            # First 600 rows have class 0, last 600 have class 1
            for i in range(600):
                f.write(f"{i},0\n")
            for i in range(600, 1200):
                f.write(f"{i},1\n")
            path = Path(f.name)

        columns, total_rows, sample = read_arff_header_and_sample(path)
        path.unlink()

        assert columns == ("value", "class")
        assert total_rows == n_rows

        # Sample should have 1000 rows (500 from start, 500 from end)
        assert len(sample) == 1000

        # First half of sample should be from start (class 0)
        first_half = sample[:500]
        assert all(row[1] == "0" for row in first_half)

        # Second half of sample should be from end (class 1)
        second_half = sample[500:]
        assert all(row[1] == "1" for row in second_half)
