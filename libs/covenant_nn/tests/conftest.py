"""Shared test fixtures for covenant_nn tests."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Final, TypedDict

import numpy as np
import pytest
from numpy.typing import NDArray
from platform_ml import configure_torch_threads

# Path to test data directory
DATA_DIR = Path(__file__).parent / "data"

#: Rows of the US bankruptcy CSV a training test sees.
#:
#: A STRATIFIED SAMPLE, NOT THE WHOLE FILE. The CSV holds 78,682 company-years
#: (5,220 failed), and every training test used to fit on all of them: on the
#: hub one LSTM early-stopping test took 140 s alone and the suite 500 s under
#: xdist, against the operator's five-minute make check (board task e80453a0).
#: 4,000 rows keep 265 failures, enough for a 15 percent validation split to
#: carry about 40 positives, and every backend still learns on them: measured
#: on one torch thread, the outcome test's MLP reached val AUC 0.66 in 1.6 s
#: and the no-progress test's LSTM 0.64 in 3.4 s. A 2,000-row draw (133
#: failures) left that LSTM at AUC 0.55.
US_BANKRUPTCY_SAMPLE_ROWS: Final[int] = 4000

#: Seed of the sample, so every test and every run sees the same rows.
_SAMPLE_SEED: Final[int] = 0

#: Intra-op threads torch may use inside one test process.
#:
#: ONE, BECAUSE XDIST IS ALREADY THE PARALLELISM. torch sizes its pool to every
#: core by default, so ``-n auto`` on the hub's 24 cores ran 24 workers of 24
#: threads each on models too small to divide a matmul between threads. The
#: same pin, and the measurement behind it, is in Model-Trainer's conftest
#: (MCPs board task 2f90d785).
_TORCH_THREADS_PER_TEST: Final[int] = 1


@pytest.fixture(autouse=True)
def _one_torch_thread_per_test() -> None:
    """Pin torch's intra-op pool to one thread before every test."""
    configure_torch_threads({"threads": _TORCH_THREADS_PER_TEST})


class USBankruptcyDataset(TypedDict):
    """US bankruptcy dataset loaded from CSV."""

    x: NDArray[np.float64]
    y: NDArray[np.int64]
    feature_names: list[str]
    n_samples: int
    n_features: int
    n_bankrupt: int
    n_healthy: int


def _dataset(
    x: NDArray[np.float64], y: NDArray[np.int64], feature_names: list[str]
) -> USBankruptcyDataset:
    """Build the dataset record, counting its rows and classes."""
    n_samples = int(y.shape[0])
    bankrupt_mask: NDArray[np.bool_] = y == 1
    n_bankrupt = int(np.count_nonzero(bankrupt_mask))
    return {
        "x": x,
        "y": y,
        "feature_names": list(feature_names),
        "n_samples": n_samples,
        "n_features": int(x.shape[1]),
        "n_bankrupt": n_bankrupt,
        "n_healthy": n_samples - n_bankrupt,
    }


#: The parsed CSV by path, filled by the first sample a test process loads.
_PARSED: dict[Path, USBankruptcyDataset] = {}


def _parse_us_bankruptcy_csv(data_path: Path) -> USBankruptcyDataset:
    """Parse the whole US bankruptcy CSV, once per test process.

    Format: ``company_name,status_label,year,X1,...,X18``; ``status_label`` is
    ``failed`` (label 1) or ``alive`` (label 0), and every ``X`` cell is a
    number.

    Args:
        data_path: The CSV to read.

    Returns:
        The whole file. Callers index it, which copies, and never hand these
        arrays to code under test.

    Raises:
        FileNotFoundError: If the dataset file is missing.
        ValueError: If a feature cell is not a number.
    """
    if data_path in _PARSED:
        return _PARSED[data_path]
    with open(data_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        headers = [h.strip() for h in next(reader)]
        rows = list(reader)

    status_idx = headers.index("status_label")
    feature_cols = [h for h in headers if h.startswith("X")]
    feature_indices = [headers.index(col) for col in feature_cols]

    x_array: NDArray[np.float64] = np.zeros((len(rows), len(feature_cols)), dtype=np.float64)
    y_array: NDArray[np.int64] = np.zeros(len(rows), dtype=np.int64)
    for i, row in enumerate(rows):
        y_array[i] = 1 if row[status_idx] == "failed" else 0
        for j, col_idx in enumerate(feature_indices):
            x_array[i, j] = float(row[col_idx])

    parsed = _dataset(x_array, y_array, feature_cols)
    _PARSED[data_path] = parsed
    return parsed


def load_us_bankruptcy_sample() -> USBankruptcyDataset:
    """Load a stratified sample of the US bankruptcy dataset for training tests.

    Returns:
        :data:`US_BANKRUPTCY_SAMPLE_ROWS` rows in file order, with failed and
        alive companies in the file's proportion, drawn with a fixed seed.
        The arrays are fresh copies on every call.

    Raises:
        FileNotFoundError: If the dataset file is missing.
    """
    full = _parse_us_bankruptcy_csv(DATA_DIR / "american_bankruptcy.csv")
    rng = np.random.default_rng(_SAMPLE_SEED)
    bankrupt_mask: NDArray[np.bool_] = full["y"] == 1
    positives: NDArray[np.intp] = np.flatnonzero(bankrupt_mask)
    negatives: NDArray[np.intp] = np.flatnonzero(~bankrupt_mask)
    n_positive = round(US_BANKRUPTCY_SAMPLE_ROWS * full["n_bankrupt"] / full["n_samples"])
    positive_order: NDArray[np.intp] = rng.permutation(len(positives))
    negative_order: NDArray[np.intp] = rng.permutation(len(negatives))
    chosen_positives: NDArray[np.intp] = positives[positive_order[:n_positive]]
    chosen_negatives: NDArray[np.intp] = negatives[
        negative_order[: US_BANKRUPTCY_SAMPLE_ROWS - n_positive]
    ]
    chosen: NDArray[np.intp] = np.sort(np.concatenate((chosen_positives, chosen_negatives)))
    x_sample: NDArray[np.float64] = full["x"][chosen]
    y_sample: NDArray[np.int64] = full["y"][chosen]
    return _dataset(x_sample, y_sample, full["feature_names"])
