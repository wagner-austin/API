"""File reading for the growth-policy datasets.

The only module here that touches a file.

CSV goes through ``polars``, reusing the typed boundary
:mod:`covenant_ml.benchmarking.dataset` already declares for it. The standard
library's ``csv`` module is deliberately not used: its readers are typed as
iterators of loosely-typed rows, which under ``disallow_any_expr`` would put an
``Any`` between the file and a published figure, and the repository's import
rules bar ``Iterable`` and ``Iterator`` outright. The polars boundary hands
back concrete frames, columns and arrays instead.

The German-credit file is whitespace-separated rather than delimited, so it is
read as text and split, which needs no reader at all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..benchmarking.dataset import DataFrameProto, load_read_csv
from .types import ERR_EMPTY_DATASET, ERR_MISSING_COLUMN, ERR_MISSING_VALUE


def read_frame(path: Path) -> DataFrameProto:
    """Read a CSV file into a frame.

    Args:
        path: File to read.

    Returns:
        The parsed frame.

    Raises:
        ValueError: If the file holds no data rows, which means the path named
            something other than the dataset it was expected to name.
    """
    frame = load_read_csv()(path)
    if len(frame) == 0:
        raise ValueError(f"[{ERR_EMPTY_DATASET}] No data rows read from {path}")
    return frame


def require_columns(frame: DataFrameProto, required: tuple[str, ...], path: Path) -> None:
    """Fail unless every required column is present.

    Checking the header up front means a wrong file fails naming the column it
    lacks, rather than raising somewhere deeper with no locator.

    Args:
        frame: Frame to check.
        required: Column names that must be present.
        path: Source path, named in the error message.

    Raises:
        ValueError: If any required column is absent.
    """
    columns = frame.columns
    for name in required:
        if name not in columns:
            raise ValueError(f"[{ERR_MISSING_COLUMN}] Column '{name}' is absent from {path}")


def read_numeric_columns(frame: DataFrameProto, columns: list[str]) -> NDArray[np.float64]:
    """Take a set of columns as a float matrix, rejecting absent values.

    The finiteness check is load-bearing rather than defensive. Polars pads a
    short CSV row with ``null`` instead of failing, so a truncated line becomes
    a ``NaN`` feature that XGBoost silently routes down a default branch while
    ClearGBM and the metrics see a value that is not a number. The reader this
    replaced parsed each cell with ``float()``, which raised on an empty field,
    so accepting nulls here would be a real loss of strictness rather than a
    change of style. A row with too many fields is rejected by polars itself.

    Args:
        frame: Frame to read.
        columns: Column names to select, in order.

    Returns:
        The selected columns, shape (n_rows, len(columns)).

    Raises:
        ValueError: If any selected cell is absent or not finite, which means
            the file is truncated, padded, or carries a sentinel this
            experiment has no rule for.
    """
    matrix: NDArray[np.float64] = frame.select(columns).to_numpy()
    finite: NDArray[np.bool_] = np.isfinite(matrix)
    if not bool(np.all(finite)):
        bad_count = int(np.size(finite)) - int(np.count_nonzero(finite))
        raise ValueError(
            f"[{ERR_MISSING_VALUE}] {bad_count} of {np.size(finite)} selected cells "
            f"are absent or not finite across columns {columns}"
        )
    return matrix


def read_text_column(frame: DataFrameProto, name: str) -> list[str]:
    """Take one column as a list of strings.

    Args:
        frame: Frame to read.
        name: Column to take.

    Returns:
        The column's values.
    """
    return frame.get_column(name).to_list()


def read_whitespace_rows(path: Path) -> list[list[str]]:
    """Read a whitespace-separated file as a list of rows.

    Args:
        path: File to read.

    Returns:
        One list of fields per non-blank line.

    Raises:
        ValueError: If the file holds no non-blank lines.
    """
    with path.open(encoding="utf-8") as handle:
        rows = [line.split() for line in handle if len(line.strip()) > 0]
    if len(rows) == 0:
        raise ValueError(f"[{ERR_EMPTY_DATASET}] No rows read from {path}")
    return rows


__all__ = [
    "read_frame",
    "read_numeric_columns",
    "read_text_column",
    "read_whitespace_rows",
    "require_columns",
]
