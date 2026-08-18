"""The three datasets the growth-policy experiment is measured on.

Each loader reproduces exactly what the recorded experiment read, so the tables
in ``libs/cleargbm/docs/EXPERIMENT_2026-08-17_growth_policy_xgb_instrument.md``
remain reproducible from this code. That is why the American-bankruptcy loader
lives here rather than delegating to
:func:`covenant_ml.benchmarking.dataset.load_bankruptcy_dataset`: the benchmark
loader reads through ``polars`` and produces a three-way partition, while the
figures on record came from a ``csv`` read and a two-way company-disjoint
partition. Delegating would change the numbers the document reports.

Every path is a parameter. The recorded multi-dataset run hardcoded an absolute
``C:\\Users\\...`` path, which made it unrunnable on any other machine and on
the measurement fleet.

All file reading goes through :mod:`covenant_ml.growth_policy.csv_io`, which is
the only module here that touches a file and the only one that narrows the
standard library's loosely typed CSV rows.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from .reading import (
    read_frame,
    read_numeric_columns,
    read_text_column,
    read_whitespace_rows,
    require_columns,
)
from .types import (
    ERR_EMPTY_DATASET,
    ERR_EMPTY_SPLIT,
    ERR_RAGGED_ROWS,
    DatasetInfo,
)

#: Number of feature columns in the American-bankruptcy CSV (``X1`` .. ``X18``).
BANKRUPTCY_FEATURE_COUNT = 18

#: Columns the American-bankruptcy loader requires to be present.
BANKRUPTCY_REQUIRED_COLUMNS = (
    "company_name",
    "status_label",
    *[f"X{index}" for index in range(1, BANKRUPTCY_FEATURE_COUNT + 1)],
)

#: Fraction of groups assigned to the training partition.
TRAIN_FRACTION = 0.70

#: The status value marking a company that did not fail.
_ALIVE_LABEL = "alive"

#: Matches a value the datasets encode as a number; anything else is a
#: category. A predicate rather than a ``float()`` inside a ``try``: whether a
#: column is numeric is a property of the data, so it is decided by asking
#: rather than by provoking a failure and catching it.
_NUMERIC_PATTERN = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


class GroupedDataset(NamedTuple):
    """A dataset whose rows carry a grouping key.

    Args:
        features: Feature matrix, shape (n_samples, n_features).
        labels: Binary labels (0 or 1), shape (n_samples,).
        groups: Grouping key per row, used to keep a partition disjoint.
    """

    features: NDArray[np.float64]
    labels: NDArray[np.int64]
    groups: list[str]


class PlainDataset(NamedTuple):
    """A dataset with no grouping key.

    Args:
        features: Feature matrix, shape (n_samples, n_features).
        labels: Binary labels (0 or 1), shape (n_samples,).
    """

    features: NDArray[np.float64]
    labels: NDArray[np.int64]


def describe_dataset(
    name: str,
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> DatasetInfo:
    """Summarise a loaded dataset's shape for the report header.

    Args:
        name: Human-readable dataset name.
        features: Feature matrix, shape (n_samples, n_features).
        labels: Binary labels, shape (n_samples,).

    Returns:
        The dataset description.

    Raises:
        ValueError: If the dataset has no rows, which leaves every downstream
            statistic undefined rather than merely small.
    """
    row_count = int(features.shape[0])
    if row_count == 0:
        raise ValueError(f"[{ERR_EMPTY_DATASET}] Dataset '{name}' has no rows")
    return {
        "name": name,
        "row_count": row_count,
        "feature_count": int(features.shape[1]),
        "positive_rate": float(np.sum(labels)) / row_count,
    }


def load_bankruptcy(csv_path: Path) -> GroupedDataset:
    """Load the American-bankruptcy dataset with its company key.

    Args:
        csv_path: Path to ``american_bankruptcy.csv``.

    Returns:
        Features, labels, and the company name for each row.

    Raises:
        ValueError: If the file yields no data rows, or if a required column is
            absent, either of which means the path named a different dataset.
    """
    frame = read_frame(csv_path)
    require_columns(frame, BANKRUPTCY_REQUIRED_COLUMNS, csv_path)
    feature_columns = [f"X{index}" for index in range(1, BANKRUPTCY_FEATURE_COUNT + 1)]
    features = read_numeric_columns(frame, feature_columns)
    statuses = read_text_column(frame, "status_label")
    companies = read_text_column(frame, "company_name")
    flags: list[int] = [0 if status.strip() == _ALIVE_LABEL else 1 for status in statuses]
    return GroupedDataset(
        features=features,
        labels=np.asarray(flags, dtype=np.int64),
        groups=companies,
    )


def sorted_group_codes(groups: list[str]) -> tuple[NDArray[np.int64], int]:
    """Assign each distinct group an integer code in sorted-name order.

    Deliberately not
    :func:`covenant_ml.benchmarking.dataset.encode_group_codes`, which assigns
    codes in first-appearance order. The recorded run permuted
    ``sorted(set(names))``, so a first-appearance encoding would permute a
    different sequence under the same seed and put different companies in the
    training fold, changing every figure the write-up reports.

    Args:
        groups: Group name per row.

    Returns:
        The integer code per row, and the number of distinct groups.
    """
    unique: list[str] = sorted(set(groups))
    position: dict[str, int] = {name: index for index, name in enumerate(unique)}
    codes: list[int] = [position[name] for name in groups]
    return np.asarray(codes, dtype=np.int64), len(unique)


def company_disjoint_indices(
    groups: list[str],
    seed: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Partition row indices so no group appears on both sides.

    A company contributes several yearly rows, so partitioning rows at random
    would place one company in both folds and leak its outcome into the
    held-out score.

    Args:
        groups: Grouping key per row.
        seed: Seed controlling the group permutation.

    Returns:
        Train and test row indices, in that order.

    Raises:
        ValueError: If either side is empty, which leaves the fit or the score
            undefined rather than merely small.
    """
    codes, group_count = sorted_group_codes(groups)
    rng = np.random.default_rng(seed)
    permutation: NDArray[np.int64] = rng.permutation(group_count)
    cut = int(TRAIN_FRACTION * group_count)
    train_codes: NDArray[np.int64] = permutation[:cut]
    mask: NDArray[np.bool_] = np.isin(codes, train_codes)
    indices: NDArray[np.int64] = np.arange(len(groups), dtype=np.int64)
    train_index: NDArray[np.int64] = indices[mask]
    test_index: NDArray[np.int64] = indices[~mask]
    if len(train_index) == 0 or len(test_index) == 0:
        raise ValueError(
            f"[{ERR_EMPTY_SPLIT}] Group-disjoint partition at seed {seed} left an "
            f"empty side: {len(train_index)} train rows, {len(test_index)} test rows"
        )
    return train_index, test_index


def _reject_ragged(rows: list[list[str]], path: Path) -> int:
    """Fail when a file's rows differ in width.

    Emptiness is not re-checked here. :func:`read_whitespace_rows` is the only
    way rows reach this function and it already refuses a file with no
    non-blank lines, so a second guard would be unreachable duplication of a
    rule that already has one owner.

    Args:
        rows: The rows to check. Must be non-empty.
        path: Source path, named in the error message.

    Returns:
        The common row width.

    Raises:
        ValueError: If any row's width differs from the first, which would
            misalign every feature column.
    """
    width = len(rows[0])
    for row in rows:
        if len(row) != width:
            raise ValueError(
                f"[{ERR_RAGGED_ROWS}] {path} has rows of differing width: "
                f"expected {width}, found {len(row)}"
            )
    return width


def load_taiwan_bankruptcy(csv_path: Path) -> PlainDataset:
    """Load the Taiwan-bankruptcy dataset.

    The label is the first column; every remaining column is a feature.

    Args:
        csv_path: Path to the dataset's ``data.csv``.

    Returns:
        Features and labels.

    Raises:
        ValueError: If the file holds no data rows.
    """
    frame = read_frame(csv_path)
    columns = frame.columns
    label_column = columns[0]
    feature_columns = columns[1:]
    features = read_numeric_columns(frame, feature_columns)
    label_matrix = read_numeric_columns(frame, [label_column])
    label_values: NDArray[np.float64] = label_matrix[:, 0]
    return PlainDataset(
        features=features,
        labels=label_values.astype(np.int64),
    )


def load_german_credit(data_path: Path) -> PlainDataset:
    """Load the German-credit dataset, ordinal-encoding its categorical columns.

    The file is whitespace-separated with the label last, encoded ``1`` for good
    and ``2`` for bad credit; the label is mapped so bad credit is the positive
    class. Categorical columns are ordinal-encoded over their sorted value set,
    which is the encoding the recorded figures were measured under.

    Args:
        data_path: Path to ``german.data``.

    Returns:
        Features and labels.

    Raises:
        ValueError: If the file holds no rows, or if its rows are ragged.
    """
    rows = read_whitespace_rows(data_path)
    width = _reject_ragged(rows, data_path)
    labels: list[int] = [1 if row[-1] == "2" else 0 for row in rows]
    feature_rows: list[list[str]] = [row[:-1] for row in rows]
    columns: list[list[str]] = [[row[index] for row in feature_rows] for index in range(width - 1)]
    encoded: list[list[float]] = [encode_column(column) for column in columns]
    features: NDArray[np.float64] = np.asarray(encoded, dtype=np.float64).T
    return PlainDataset(
        features=features,
        labels=np.asarray(labels, dtype=np.int64),
    )


def encode_column(column: list[str]) -> list[float]:
    """Convert one raw column to floats, ordinal-encoding it when categorical.

    A column counts as numeric only when every value parses as a number. One
    non-numeric value makes the whole column categorical, because a column read
    half as magnitudes and half as codes would compare values that mean
    different things.

    Args:
        column: The column's raw string values.

    Returns:
        The column as floats.
    """
    if all(_NUMERIC_PATTERN.match(value) is not None for value in column):
        return [float(value) for value in column]
    codes = {value: index for index, value in enumerate(sorted(set(column)))}
    return [float(codes[value]) for value in column]


__all__ = [
    "BANKRUPTCY_FEATURE_COUNT",
    "BANKRUPTCY_REQUIRED_COLUMNS",
    "TRAIN_FRACTION",
    "GroupedDataset",
    "PlainDataset",
    "company_disjoint_indices",
    "describe_dataset",
    "encode_column",
    "load_bankruptcy",
    "load_german_credit",
    "load_taiwan_bankruptcy",
    "sorted_group_codes",
]
