"""Tests for loading the benchmark dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.benchmarking.dataset import (
    encode_group_codes,
    file_sha256,
    load_bankruptcy_dataset,
)
from covenant_ml.benchmarking.types import ERR_MISSING_COLUMN

CSV_HEADER = "company_name,status_label,year,X1,X2\n"
CSV_ROWS = (
    "C_1,alive,1999,1.0,2.0\n"
    "C_1,failed,2000,3.0,4.0\n"
    "C_2,alive,1999,5.0,6.0\n"
    "C_3,failed,2001,7.0,8.0\n"
)


def write_csv(directory: Path, content: str) -> Path:
    """Write a CSV file into a directory.

    Args:
        directory: Destination directory.
        content: File contents.

    Returns:
        Path to the written file.
    """
    path = directory / "data.csv"
    path.write_text(content, encoding="utf-8")
    return path


def test_file_sha256_matches_hashlib(tmp_path: Path) -> None:
    path = write_csv(tmp_path, CSV_HEADER + CSV_ROWS)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert file_sha256(path) == expected


def test_group_codes_are_assigned_in_first_appearance_order() -> None:
    codes = encode_group_codes(["b", "a", "b", "c"])
    expected: list[int] = [0, 1, 0, 2]
    expected_array: NDArray[np.int64] = np.asarray(expected, dtype=np.int64)
    assert np.array_equal(codes, expected_array)


def test_group_codes_handle_a_single_group() -> None:
    codes = encode_group_codes(["only", "only"])
    expected: list[int] = [0, 0]
    expected_array: NDArray[np.int64] = np.asarray(expected, dtype=np.int64)
    assert np.array_equal(codes, expected_array)


def test_group_codes_handle_no_rows() -> None:
    codes = encode_group_codes([])
    assert len(codes) == 0


def test_identifier_columns_are_excluded_from_features(tmp_path: Path) -> None:
    path = write_csv(tmp_path, CSV_HEADER + CSV_ROWS)
    dataset = load_bankruptcy_dataset(path)
    # X1 and X2 only; company_name, status_label and year are identifiers.
    assert dataset.info["n_features"] == 2
    assert dataset.features.shape == (4, 2)


def test_failed_rows_become_the_positive_class(tmp_path: Path) -> None:
    path = write_csv(tmp_path, CSV_HEADER + CSV_ROWS)
    dataset = load_bankruptcy_dataset(path)
    expected: list[int] = [0, 1, 0, 1]
    expected_array: NDArray[np.int64] = np.asarray(expected, dtype=np.int64)
    assert np.array_equal(dataset.labels, expected_array)


def test_rows_of_one_company_share_a_code(tmp_path: Path) -> None:
    path = write_csv(tmp_path, CSV_HEADER + CSV_ROWS)
    dataset = load_bankruptcy_dataset(path)
    # C_1 contributes the first two rows, then C_2 and C_3 one each.
    expected: list[int] = [0, 0, 1, 2]
    expected_array: NDArray[np.int64] = np.asarray(expected, dtype=np.int64)
    assert np.array_equal(dataset.company_codes, expected_array)


def test_dataset_info_records_identity(tmp_path: Path) -> None:
    path = write_csv(tmp_path, CSV_HEADER + CSV_ROWS)
    dataset = load_bankruptcy_dataset(path)
    assert dataset.info["n_rows"] == 4
    assert dataset.info["sha256"] == file_sha256(path)


def test_missing_label_column_raises(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "company_name,year,X1\nC_1,1999,1.0\n")
    with pytest.raises(ValueError, match=ERR_MISSING_COLUMN):
        load_bankruptcy_dataset(path)


def test_missing_group_column_raises(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "status_label,year,X1\nalive,1999,1.0\n")
    with pytest.raises(ValueError, match=ERR_MISSING_COLUMN):
        load_bankruptcy_dataset(path)
