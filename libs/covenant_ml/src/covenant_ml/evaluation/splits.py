"""Splits that carry the held-out rows' unit codes.

A paired interval needs to know which held-out rows belong together, which a
bare :class:`~covenant_ml.benchmarking.protocols.DataSplit` does not say.
:class:`GroupedSplit` is that split plus one code per held-out row.

Two partitions are provided:

* :class:`CompanyDisjointGroupedSplits` -- the benchmark's own company-disjoint
  split (the same masks, via
  :func:`~covenant_ml.benchmarking.splitting.company_disjoint_masks`), with
  each held-out row's company as its code.
* :class:`ChronologicalGroupedSplits` -- past trains, future tests, for data
  that arrives in time: market days, contract resolutions. An ``embargo`` of
  distinct time keys is dropped at each boundary, because a label that looks ahead (a
  five-day return, a contract resolving next week) computed on the last
  training rows overlaps the first test rows. Rows come back in time order,
  and each held-out row's code is its time key, so rows sharing a date can be
  treated as one unit.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

import numpy as np
from numpy.typing import NDArray

from ..benchmarking.protocols import DataSplit
from ..benchmarking.splitting import (
    SplitMasks,
    apply_split_masks,
    company_disjoint_masks,
)
from .types import (
    ERR_INVALID_EMBARGO,
    ERR_INVALID_FRACTION,
    ERR_LENGTH_MISMATCH,
    ERR_TOO_FEW_UNITS,
)


class GroupedSplit(NamedTuple):
    """A partition and the unit code of every held-out row.

    Args:
        split: The partition.
        test_groups: One code per held-out row, shape (n_test,).
    """

    split: DataSplit
    test_groups: NDArray[np.int64]


class GroupedSplitFactoryProto(Protocol):
    """Builds the grouped partition for one seed."""

    def __call__(self, seed: int) -> GroupedSplit:
        """Partition the dataset for a seed.

        Args:
            seed: Seed of the partition, where it has randomness.

        Returns:
            The grouped partition.
        """
        ...


class CompanyDisjointGroupedSplits:
    """Company-disjoint partitions, each held-out row coded by its company."""

    def __init__(
        self,
        features: NDArray[np.float64],
        labels: NDArray[np.int64],
        company_codes: NDArray[np.int64],
    ) -> None:
        """Bind the dataset the factory partitions.

        Args:
            features: Feature matrix, shape (n_rows, n_features).
            labels: Binary labels, shape (n_rows,).
            company_codes: Integer company identifier per row.
        """
        self._features = features
        self._labels = labels
        self._company_codes = company_codes

    def __call__(self, seed: int) -> GroupedSplit:
        """Partition by company for one seed.

        Args:
            seed: Seed of the company permutation.

        Returns:
            The partition with each held-out row's company code.

        Raises:
            ValueError: If the arrays are misaligned or a fold comes out empty.
        """
        masks = company_disjoint_masks(len(self._features), self._company_codes, seed)
        return GroupedSplit(
            split=apply_split_masks(self._features, self._labels, masks),
            test_groups=self._company_codes[masks.test],
        )


def chronological_masks(
    n_rows: int,
    train_fraction: float,
    val_fraction: float,
    embargo: int,
) -> SplitMasks:
    """Assign rows already in time order to train, validation and test.

    Args:
        n_rows: Rows, in time order.
        train_fraction: Share of rows, from the start, for training.
        val_fraction: Share of rows after training for validation; the rest,
            less the embargoes, is the test fold.
        embargo: Rows dropped after training and after validation.

    Returns:
        One mask per fold; embargoed rows are in none.

    Raises:
        ValueError: If a fraction is outside ``(0, 1)``, the two leave no test
            share, ``embargo`` is negative, or a fold comes out empty.
    """
    if not 0.0 < train_fraction < 1.0 or not 0.0 < val_fraction < 1.0:
        raise ValueError(
            f"[{ERR_INVALID_FRACTION}] fractions must lie in (0, 1), got "
            f"{train_fraction} and {val_fraction}"
        )
    if train_fraction + val_fraction >= 1.0:
        raise ValueError(
            f"[{ERR_INVALID_FRACTION}] train and validation shares leave no test rows: "
            f"{train_fraction} + {val_fraction}"
        )
    if embargo < 0:
        raise ValueError(f"[{ERR_INVALID_EMBARGO}] embargo must be non-negative, got {embargo}")
    train_end = int(n_rows * train_fraction)
    val_start = train_end + embargo
    val_end = int(n_rows * (train_fraction + val_fraction))
    test_start = val_end + embargo
    positions: NDArray[np.int64] = np.arange(n_rows, dtype=np.int64)
    masks = SplitMasks(
        train=positions < train_end,
        val=(positions >= val_start) & (positions < val_end),
        test=positions >= test_start,
    )
    sizes = (int(np.sum(masks.train)), int(np.sum(masks.val)), int(np.sum(masks.test)))
    if min(sizes) == 0:
        raise ValueError(
            f"[{ERR_TOO_FEW_UNITS}] chronological split left an empty fold: "
            f"train, val, test = {sizes} from {n_rows} rows with embargo {embargo}"
        )
    return masks


class ChronologicalGroupedSplits:
    """Partition whole time groups with an embargo measured in distinct time keys."""

    def __init__(
        self,
        features: NDArray[np.float64],
        labels: NDArray[np.int64],
        time_keys: NDArray[np.int64],
        train_fraction: float,
        val_fraction: float,
        embargo: int,
    ) -> None:
        """Order the dataset in time once.

        Args:
            features: Feature matrix, shape (n_rows, n_features).
            labels: Binary labels, shape (n_rows,).
            time_keys: Sortable time per row (a date as YYYYMMDD, an epoch
                second). Rows sharing a key keep their input order.
            train_fraction: Share of distinct time keys, earliest first, for training.
            val_fraction: Share of distinct time keys after training for validation.
            embargo: Whole time keys dropped at each boundary. Callers must choose
                keys and embargo to cover label availability; this is not a duration.

        Raises:
            ValueError: If the arrays differ in length.
        """
        n_rows = len(features)
        if len(labels) != n_rows or len(time_keys) != n_rows:
            raise ValueError(
                f"[{ERR_LENGTH_MISMATCH}] features, labels and time_keys must have equal "
                f"length, got {n_rows}, {len(labels)} and {len(time_keys)}"
            )
        order: NDArray[np.intp] = np.argsort(time_keys, kind="stable")
        self._features: NDArray[np.float64] = features[order]
        self._labels: NDArray[np.int64] = labels[order]
        self._time_keys: NDArray[np.int64] = time_keys[order]
        self._train_fraction = train_fraction
        self._val_fraction = val_fraction
        self._embargo = embargo

    def __call__(self, seed: int) -> GroupedSplit:
        """Build the partition, which is the same at every seed.

        The seed still varies the models fitted on it; only the partition is
        fixed, because the future cannot be resampled into the past.

        Args:
            seed: Unused by the partition; accepted for the factory protocol.

        Returns:
            The partition, rows in time order, coded by time key.

        Raises:
            ValueError: If the fractions or embargo are invalid or leave a
                fold empty.
        """
        keys: NDArray[np.int64] = np.unique(self._time_keys)
        group_masks = chronological_masks(
            len(keys), self._train_fraction, self._val_fraction, self._embargo
        )
        return self._partition(
            keys[group_masks.train], keys[group_masks.val], keys[group_masks.test]
        )

    def validation_split(self, inner_train_fraction: float) -> GroupedSplit:
        """Score only outer validation, with stopping rows drawn from training.

        Args:
            inner_train_fraction: Fraction of outer training keys used for fitting.

        Returns:
            Training, early-stopping and scoring arrays with no final-test rows.

        Raises:
            ValueError: Invalid fraction or an empty inner fold after its embargo.
        """
        if not 0 < inner_train_fraction < 1:
            raise ValueError(f"[{ERR_INVALID_FRACTION}] inner training fraction must be in (0, 1)")
        keys: NDArray[np.int64] = np.unique(self._time_keys)
        outer = chronological_masks(
            len(keys), self._train_fraction, self._val_fraction, self._embargo
        )
        training = keys[outer.train]
        cut = int(len(training) * inner_train_fraction)
        stopping = cut + self._embargo
        if cut < 1 or stopping >= len(training):
            raise ValueError(f"[{ERR_TOO_FEW_UNITS}] inner training or stopping fold is empty")
        return self._partition(training[:cut], training[stopping:], keys[outer.val])

    def _partition(
        self, training: NDArray[np.int64], stopping: NDArray[np.int64], scoring: NDArray[np.int64]
    ) -> GroupedSplit:
        """Apply one key partition through the shared array splitter.

        Args:
            training: Keys available to model fitting.
            stopping: Separate keys for early stopping.
            scoring: Held-out keys scored by the evaluator.

        Returns:
            Partition and scoring keys, preserving whole time groups.
        """
        masks = SplitMasks(
            train=np.isin(self._time_keys, training),
            val=np.isin(self._time_keys, stopping),
            test=np.isin(self._time_keys, scoring),
        )
        return GroupedSplit(
            split=apply_split_masks(self._features, self._labels, masks),
            test_groups=self._time_keys[masks.test],
        )


__all__ = [
    "ChronologicalGroupedSplits",
    "CompanyDisjointGroupedSplits",
    "GroupedSplit",
    "GroupedSplitFactoryProto",
    "chronological_masks",
]
