"""Company-disjoint partitioning of the benchmark dataset.

Pure numpy, so the partition logic is testable without touching a CSV.

The partition is computed once, as three row masks, by
:func:`company_disjoint_masks`; :func:`company_disjoint_split` applies them to
the arrays. The masks are public because a paired evaluation needs the
held-out rows' COMPANY CODES as well as their features -- an interval that
resamples rows of one company independently treats ten years of one firm as
ten independent firms -- and a second implementation of the permutation to
recover them would be a fork of this one.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from .protocols import DataSplit
from .types import ERR_EMPTY_SPLIT, ERR_LENGTH_MISMATCH


class SplitMasks(NamedTuple):
    """Row membership of each fold.

    Args:
        train: True where the row is in the training fold.
        val: True where the row is in the validation fold.
        test: True where the row is in the held-out fold.
    """

    train: NDArray[np.bool_]
    val: NDArray[np.bool_]
    test: NDArray[np.bool_]


def company_disjoint_masks(
    n_rows: int,
    company_codes: NDArray[np.int64],
    seed: int,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> SplitMasks:
    """Assign every row to a fold so that no company appears in two folds.

    Rows are company-years. A random row split would leak a company's own
    later years into training and inflate held-out scores, so companies --
    not rows -- are the unit that gets partitioned.

    Args:
        n_rows: Row count the masks must cover.
        company_codes: Integer company identifier per row, shape (n_rows,).
            Rows sharing a code always land in the same fold.
        seed: Seed for the company permutation.
        train_fraction: Share of companies assigned to training.
        val_fraction: Share of companies assigned to validation. The
            remainder becomes the held-out test fold.

    Returns:
        One boolean mask per fold.

    Raises:
        ValueError: If ``company_codes`` does not cover ``n_rows`` rows, or if
            the requested fractions leave any fold empty for this dataset.
    """
    n_codes = len(company_codes)
    if n_codes != n_rows:
        raise ValueError(
            f"[{ERR_LENGTH_MISMATCH}] company_codes must cover {n_rows} rows, got {n_codes}"
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

    masks = SplitMasks(
        train=np.isin(company_codes, train_companies),
        val=np.isin(company_codes, val_companies),
        test=np.isin(company_codes, test_companies),
    )

    n_train = int(np.sum(masks.train))
    n_val = int(np.sum(masks.val))
    n_test = int(np.sum(masks.test))
    if n_train == 0 or n_val == 0 or n_test == 0:
        raise ValueError(
            f"[{ERR_EMPTY_SPLIT}] Company-disjoint split produced an empty fold: "
            f"train={n_train}, val={n_val}, test={n_test} from {n_companies} companies"
        )
    return masks


def apply_split_masks(
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    masks: SplitMasks,
) -> DataSplit:
    """Cut the arrays into the three folds the masks name.

    Args:
        features: Feature matrix, shape (n_rows, n_features).
        labels: Binary labels (0 or 1), shape (n_rows,).
        masks: Fold membership per row.

    Returns:
        The three-way partition.

    Raises:
        ValueError: If the labels or masks do not cover the feature rows.
    """
    n_rows = len(features)
    lengths = (len(labels), len(masks.train), len(masks.val), len(masks.test))
    if any(length != n_rows for length in lengths):
        raise ValueError(
            f"[{ERR_LENGTH_MISMATCH}] labels and masks must cover {n_rows} rows, got {lengths}"
        )
    return DataSplit(
        x_train=features[masks.train],
        y_train=labels[masks.train],
        x_val=features[masks.val],
        y_val=labels[masks.val],
        x_test=features[masks.test],
        y_test=labels[masks.test],
    )


def company_disjoint_split(
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    company_codes: NDArray[np.int64],
    seed: int,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> DataSplit:
    """Partition rows so that no company appears in two folds.

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
    masks = company_disjoint_masks(n_rows, company_codes, seed, train_fraction, val_fraction)
    return apply_split_masks(features, labels, masks)


__all__ = [
    "SplitMasks",
    "apply_split_masks",
    "company_disjoint_masks",
    "company_disjoint_split",
]
