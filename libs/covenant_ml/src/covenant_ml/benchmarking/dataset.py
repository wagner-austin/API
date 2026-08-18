"""Loading of the bankruptcy benchmark dataset.

The only module in this package that touches a file or a dataframe. It lowers
the CSV into plain numpy arrays plus a content hash, so every layer above it
is pure and testable without fixtures on disk.

Polars is reached through :func:`__import__` with its members assigned to
Protocol-typed names. ``DataFrame.to_numpy`` is declared as returning
``ndarray[Any, Any]`` upstream, which strict mode rejects; naming the exact
array type at this one boundary keeps every layer above it precisely typed
without stubs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import NamedTuple, Protocol

import numpy as np
from numpy.typing import NDArray

from .types import ERR_MISSING_COLUMN, DatasetInfo

#: Columns that identify a row rather than describe it. Everything else in the
#: CSV is a feature.
IDENTIFIER_COLUMNS = ("company_name", "year", "status_label")

#: Label column, and the value of it that denotes the positive class.
LABEL_COLUMN = "status_label"
POSITIVE_LABEL = "failed"

#: Column whose value groups rows that must not be split across folds.
GROUP_COLUMN = "company_name"


class SeriesProto(Protocol):
    """Protocol for the polars ``Series`` members this module reads."""

    def to_list(self) -> list[str]:
        """Materialise the column as Python strings.

        Returns:
            One string per row.
        """
        ...


class DataFrameProto(Protocol):
    """Protocol for the polars ``DataFrame`` members this module reads."""

    @property
    def columns(self) -> list[str]:
        """Return the column names in order.

        Returns:
            Column names.
        """
        ...

    def select(self, columns: list[str]) -> DataFrameProto:
        """Project the frame down to the named columns.

        Args:
            columns: Column names to keep.

        Returns:
            The projected frame.
        """
        ...

    def get_column(self, name: str) -> SeriesProto:
        """Return one column.

        Args:
            name: Column name.

        Returns:
            The column.
        """
        ...

    def to_numpy(self) -> NDArray[np.float64]:
        """Materialise the frame as a dense array.

        Returns:
            Array of shape (n_rows, n_columns).
        """
        ...

    def __len__(self) -> int:
        """Return the row count.

        Returns:
            Number of rows.
        """
        ...


class ReadCsvProto(Protocol):
    """Protocol for ``polars.read_csv``."""

    def __call__(self, source: Path) -> DataFrameProto:
        """Read a CSV file into a frame.

        Args:
            source: Path to the CSV.

        Returns:
            The loaded frame.
        """
        ...


def load_read_csv() -> ReadCsvProto:
    """Resolve ``polars.read_csv`` as a Protocol-typed callable.

    Returns:
        The CSV reader.
    """
    module = __import__("polars", fromlist=["read_csv"])
    read_csv: ReadCsvProto = module.read_csv
    return read_csv


class LoadedDataset(NamedTuple):
    """The benchmark input, lowered to arrays.

    Args:
        features: Feature matrix, shape (n_rows, n_features).
        labels: Binary labels (1 where the company failed), shape (n_rows,).
        company_codes: Integer company identifier per row, shape (n_rows,).
        info: Identity of the source file and the loaded shape.
    """

    features: NDArray[np.float64]
    labels: NDArray[np.int64]
    company_codes: NDArray[np.int64]
    info: DatasetInfo


def file_sha256(path: Path) -> str:
    """Hash a file's contents.

    Recorded in every manifest so two runs can be proven to have used the
    same input.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_group_codes(names: list[str]) -> NDArray[np.int64]:
    """Assign a stable integer code to each distinct group name.

    Codes are assigned in first-appearance order, so the mapping depends only
    on the file's contents and not on hash ordering.

    Args:
        names: Group name per row.

    Returns:
        Integer code per row, shape (n_rows,).
    """
    assigned: dict[str, int] = {}
    codes: list[int] = []
    for name in names:
        code = assigned.get(name)
        if code is None:
            code = len(assigned)
            assigned[name] = code
        codes.append(code)
    return np.asarray(codes, dtype=np.int64)


def load_bankruptcy_dataset(csv_path: Path) -> LoadedDataset:
    """Load the bankruptcy CSV into arrays.

    Args:
        csv_path: Path to the source CSV.

    Returns:
        Features, labels, company codes and dataset identity.

    Raises:
        ValueError: If the CSV is missing the label or grouping column, which
            would otherwise produce a silently mislabelled benchmark.
    """
    read_csv = load_read_csv()
    frame = read_csv(csv_path)
    columns = frame.columns

    for required in (LABEL_COLUMN, GROUP_COLUMN):
        if required not in columns:
            raise ValueError(
                f"[{ERR_MISSING_COLUMN}] Column '{required}' is required but "
                f"'{csv_path.name}' has {columns}"
            )

    feature_columns = [name for name in columns if name not in IDENTIFIER_COLUMNS]
    features: NDArray[np.float64] = frame.select(feature_columns).to_numpy()

    label_values = frame.get_column(LABEL_COLUMN).to_list()
    label_flags: list[int] = [1 if value == POSITIVE_LABEL else 0 for value in label_values]
    labels: NDArray[np.int64] = np.asarray(label_flags, dtype=np.int64)

    company_codes = encode_group_codes(frame.get_column(GROUP_COLUMN).to_list())

    info: DatasetInfo = {
        "sha256": file_sha256(csv_path),
        "n_rows": len(frame),
        "n_features": len(feature_columns),
    }
    return LoadedDataset(
        features=features,
        labels=labels,
        company_codes=company_codes,
        info=info,
    )


__all__ = [
    "GROUP_COLUMN",
    "IDENTIFIER_COLUMNS",
    "LABEL_COLUMN",
    "POSITIVE_LABEL",
    "DataFrameProto",
    "LoadedDataset",
    "ReadCsvProto",
    "SeriesProto",
    "encode_group_codes",
    "file_sha256",
    "load_bankruptcy_dataset",
    "load_read_csv",
]
