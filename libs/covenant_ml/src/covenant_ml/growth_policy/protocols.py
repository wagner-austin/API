"""Protocols naming every boundary the growth-policy runner depends on.

The runner never names a concrete gradient-boosting library and never sees an
untyped value. Concrete implementations live in
:mod:`covenant_ml.growth_policy.adapters` and are constructed and injected by
:mod:`covenant_ml.growth_policy.factory`.

Two boundaries are imported from :mod:`covenant_ml.benchmarking.protocols`
rather than restated. A fitted model is used here exactly as the benchmark uses
one -- score the held-out fold, report mean leaves per tree -- and the clock is
read the same way, so declaring second copies would let the two harnesses drift
into disagreeing about a shape they share.
"""

from __future__ import annotations

from typing import Literal, NamedTuple, Protocol

import numpy as np
from numpy.typing import NDArray

from ..benchmarking.protocols import MonotonicClockProto, TrainedModelProto


class TwoWaySplit(NamedTuple):
    """A train/test partition.

    Immutable and never crossing a JSON boundary, so it is a ``NamedTuple``
    rather than a ``TypedDict`` with a codec pair. The experiment holds out one
    fold and does not tune, so there is no validation partition to carry.

    Args:
        x_train: Training features, shape (n_train, n_features).
        y_train: Training labels (0 or 1), shape (n_train,).
        x_test: Held-out features, shape (n_test, n_features).
        y_test: Held-out labels (0 or 1), shape (n_test,).
    """

    x_train: NDArray[np.float64]
    y_train: NDArray[np.int64]
    x_test: NDArray[np.float64]
    y_test: NDArray[np.int64]


class ArmSpec(NamedTuple):
    """The configuration distinguishing one arm from the others.

    Only the growth policy and its two budgets vary across the XGBoost arms;
    every other hyperparameter is held identical, which is what makes the
    contrast attributable to growth policy.

    Args:
        name: Display name, unique within a report.
        grow_policy: ``depthwise`` grows level by level to ``max_depth``;
            ``lossguide`` expands the highest-gain leaf until ``max_leaves``.
        max_depth: Depth budget. ``0`` disables the depth bound, which is what
            a pure leaf-wise arm requires.
        max_leaves: Leaf budget. ``0`` disables the leaf bound, which is what a
            pure depth-wise arm requires.
    """

    name: str
    grow_policy: Literal["depthwise", "lossguide"]
    max_depth: int
    max_leaves: int


class ArmTrainerProto(Protocol):
    """Protocol for one measurable arm of the experiment."""

    @property
    def arm_name(self) -> str:
        """Name recorded for this arm's results.

        Returns:
            The arm's display name.
        """
        ...

    def fit(self, split: TwoWaySplit, seed: int) -> TrainedModelProto:
        """Fit this arm on the split's training partition.

        This call is what the experiment times, so implementations must do the
        whole fit here and no lazy work afterwards.

        Args:
            split: The partition to train on.
            seed: Seed for the model's internal randomness.

        Returns:
            The fitted model.
        """
        ...


class SplitFactoryProto(Protocol):
    """Protocol for the callable that partitions a dataset for one seed."""

    def __call__(self, seed: int) -> TwoWaySplit:
        """Build the partition for one seed.

        Args:
            seed: Seed controlling the partition.

        Returns:
            The train/test partition.
        """
        ...


class MetricsProto(Protocol):
    """Protocol for the three metrics every arm is scored on."""

    def auc_roc(self, y_true: NDArray[np.int64], positive_proba: NDArray[np.float64]) -> float:
        """Score area under the ROC curve.

        Args:
            y_true: True binary labels, shape (n_samples,).
            positive_proba: Positive-class probabilities, shape (n_samples,).

        Returns:
            The metric value.
        """
        ...

    def auc_pr(self, y_true: NDArray[np.int64], positive_proba: NDArray[np.float64]) -> float:
        """Score area under the precision-recall curve.

        Args:
            y_true: True binary labels, shape (n_samples,).
            positive_proba: Positive-class probabilities, shape (n_samples,).

        Returns:
            The metric value.
        """
        ...

    def log_loss(self, y_true: NDArray[np.int64], positive_proba: NDArray[np.float64]) -> float:
        """Score log loss.

        Args:
            y_true: True binary labels, shape (n_samples,).
            positive_proba: Positive-class probabilities, shape (n_samples,).

        Returns:
            The metric value.
        """
        ...


__all__ = [
    "ArmSpec",
    "ArmTrainerProto",
    "MetricsProto",
    "MonotonicClockProto",
    "SplitFactoryProto",
    "TrainedModelProto",
    "TwoWaySplit",
]
