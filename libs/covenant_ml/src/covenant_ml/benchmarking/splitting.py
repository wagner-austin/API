"""Company-disjoint partitioning of the benchmark dataset.

Pure numpy, so the partition logic is testable without touching a CSV.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .protocols import DataSplit
from .types import ERR_EMPTY_SPLIT, ERR_LENGTH_MISMATCH


def company_disjoint_split(
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    company_codes: NDArray[np.int64],
    seed: int,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> DataSplit:
    """Partition rows so that no company appears in two folds.

    Rows are company-years. A random row split would leak a company's own
    later years into training and inflate held-out scores, so companies --
    not rows -- are the unit that gets partitioned.

    Args:
        features: Feature matrix, shape (n_rows, n_features).
        labels: Binary labels (0 or 1), shape (n_rows,).
        company_codes: Integer company identifier per row, shape (n_rows,).
            Rows sharing a code always land in the same fold.
        seed: Seed for the company permutation.
        train_fraction: Share of companies assigned to training.
        val_fraction: Share of companies assigned to validation. The
            remainder becomes the held-out test fold.

    Returns:
        The three-way partition.

    Raises:
        ValueError: If the three arrays differ in length, or if the requested
            fractions leave any fold empty for this dataset.
    """
    n_rows = len(features)
    n_labels = len(labels)
    n_codes = len(company_codes)
    if n_labels != n_rows or n_codes != n_rows:
        raise ValueError(
            f"[{ERR_LENGTH_MISMATCH}] features, labels and company_codes must have equal "
            f"length, got {n_rows}, {n_labels} and {n_codes}"
        )

    unique_companies: NDArray[np.int64] = np.unique(company_codes)
    rng = np.random.default_rng(seed)
    permuted: NDArray[np.int64] = rng.permutation(unique_companies)

    n_companies = len(permuted)
    train_end = int(n_companies * train_fraction)
    val_end = int(n_companies * (train_fraction + val_fraction))

    train_companies: NDArray[np.int64] = permuted[:train_end]
    val_companies: NDArray[np.int64] = permuted[train_end:val_end]
    test_companies: NDArray[np.int64] = permuted[val_end:]

    train_mask: NDArray[np.bool_] = np.isin(company_codes, train_companies)
    val_mask: NDArray[np.bool_] = np.isin(company_codes, val_companies)
    test_mask: NDArray[np.bool_] = np.isin(company_codes, test_companies)

    n_train = int(np.sum(train_mask))
    n_val = int(np.sum(val_mask))
    n_test = int(np.sum(test_mask))
    if n_train == 0 or n_val == 0 or n_test == 0:
        raise ValueError(
            f"[{ERR_EMPTY_SPLIT}] Company-disjoint split produced an empty fold: "
            f"train={n_train}, val={n_val}, test={n_test} from {n_companies} companies"
        )

    return DataSplit(
        x_train=features[train_mask],
        y_train=labels[train_mask],
        x_val=features[val_mask],
        y_val=labels[val_mask],
        x_test=features[test_mask],
        y_test=labels[test_mask],
    )


__all__ = ["company_disjoint_split"]
