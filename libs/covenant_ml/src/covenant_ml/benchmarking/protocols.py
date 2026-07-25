"""Protocols describing the benchmark's injected collaborators.

Every boundary the runner depends on is expressed here as a ``Protocol`` with
the exact signature of the real implementation, so the runner never names a
concrete gradient-boosting library and never sees an untyped value.

Concrete implementations live in :mod:`covenant_ml.benchmarking.adapters`;
they are constructed and injected by
:mod:`covenant_ml.benchmarking.factory`.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

import numpy as np
from numpy.typing import NDArray

from .types import BenchmarkModelName


class DataSplit(NamedTuple):
    """A company-disjoint train/validation/test partition.

    Immutable, and never crosses a JSON boundary, so it is a ``NamedTuple``
    rather than a TypedDict with a codec pair.

    Args:
        x_train: Training features, shape (n_train, n_features).
        y_train: Training labels (0 or 1), shape (n_train,).
        x_val: Validation features, shape (n_val, n_features).
        y_val: Validation labels (0 or 1), shape (n_val,).
        x_test: Held-out features, shape (n_test, n_features).
        y_test: Held-out labels (0 or 1), shape (n_test,).
    """

    x_train: NDArray[np.float64]
    y_train: NDArray[np.int64]
    x_val: NDArray[np.float64]
    y_val: NDArray[np.int64]
    x_test: NDArray[np.float64]
    y_test: NDArray[np.int64]


class MonotonicClockProto(Protocol):
    """Protocol for the monotonic clock used to time fits.

    Injected so tests can drive the timing logic from a deterministic
    sequence instead of wall-clock readings.
    """

    def __call__(self) -> float:
        """Read the clock.

        Returns:
            Seconds from an arbitrary fixed origin. Only differences between
            two readings are meaningful.
        """
        ...


class TrainedModelProto(Protocol):
    """Protocol for a fitted model, as the benchmark needs to use it."""

    def predict_positive_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict the positive-class probability for each row.

        Args:
            x: Feature matrix, shape (n_samples, n_features).

        Returns:
            Positive-class probabilities, shape (n_samples,).
        """
        ...

    def mean_leaves(self) -> float:
        """Report the mean number of leaves per tree in the fitted ensemble.

        This is the work-per-tree normalizer that makes a depth-wise learner
        comparable to a leaf-wise one: without it, a wall-clock ratio conflates
        "slower per unit of work" with "doing more work per tree".

        Returns:
            Mean leaves per tree across the ensemble.
        """
        ...


class TrainerProto(Protocol):
    """Protocol for a gradient-boosting implementation under measurement."""

    @property
    def model_name(self) -> BenchmarkModelName:
        """Name recorded for this trainer's results.

        Returns:
            The model's manifest name.
        """
        ...

    def fit(self, split: DataSplit, seed: int) -> TrainedModelProto:
        """Fit the model on the split's training partition.

        This call is what the benchmark times, so implementations must do the
        whole fit here and no lazy work afterwards.

        Args:
            split: The partition to train on.
            seed: Seed for the model's internal randomness.

        Returns:
            The fitted model.
        """
        ...


class SplitFactoryProto(Protocol):
    """Protocol for the callable that partitions the dataset for a seed."""

    def __call__(self, seed: int) -> DataSplit:
        """Build the partition for one seed.

        Args:
            seed: Seed controlling the company permutation.

        Returns:
            The three-way partition.
        """
        ...


__all__ = [
    "DataSplit",
    "MonotonicClockProto",
    "SplitFactoryProto",
    "TrainedModelProto",
    "TrainerProto",
]
