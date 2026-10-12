"""How held-out quality grows with training data.

A model that is still improving at the full training set is limited by data,
and its loss to a simpler model may be a budget result rather than a verdict
on the method. A model that flattened early is not. The curve is what tells
the two apart before anyone concludes that "simpler is better".

Fractions are NESTED within a seed: one permutation of the training fold is
drawn per seed and every fraction takes a prefix of it, so a larger fraction
always contains the smaller one's rows and a dip in the curve is the model,
not a different draw. Validation and held-out folds are never subsampled.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from ..benchmarking.protocols import DataSplit, TrainerProto
from ..benchmarking.quality import compute_quality
from .types import ERR_INVALID_FRACTION, LearningCurvePoint


def require_fractions(fractions: Sequence[float]) -> None:
    """Refuse fractions outside ``(0, 1]`` or out of ascending order.

    Args:
        fractions: Training shares to fit at.

    Raises:
        ValueError: If no fraction is given, one is outside ``(0, 1]``, or
            they do not strictly ascend.
    """
    if not fractions:
        raise ValueError(f"[{ERR_INVALID_FRACTION}] at least one training fraction is needed")
    if any(not 0.0 < fraction <= 1.0 for fraction in fractions):
        raise ValueError(
            f"[{ERR_INVALID_FRACTION}] training fractions must lie in (0, 1], got {fractions}"
        )
    if any(later <= earlier for earlier, later in pairwise(fractions)):
        raise ValueError(
            f"[{ERR_INVALID_FRACTION}] training fractions must strictly ascend, got {fractions}"
        )


def training_prefix(split: DataSplit, order: NDArray[np.int64], fraction: float) -> DataSplit:
    """The partition with its training fold cut to a prefix of ``order``.

    Args:
        split: The full partition.
        order: A permutation of the training rows.
        fraction: Share of the training rows kept, in ``(0, 1]``; at least
            one row is always kept.

    Returns:
        The partition with fewer training rows.
    """
    n_keep = max(1, math.ceil(fraction * len(split.y_train)))
    rows: NDArray[np.int64] = order[:n_keep]
    return DataSplit(
        x_train=split.x_train[rows],
        y_train=split.y_train[rows],
        x_val=split.x_val,
        y_val=split.y_val,
        x_test=split.x_test,
        y_test=split.y_test,
    )


def learning_curve_seed(
    arm: str,
    trainer: TrainerProto,
    split: DataSplit,
    seed: int,
    fractions: Sequence[float],
) -> list[LearningCurvePoint]:
    """Fit one learner at each training fraction of one partition.

    Args:
        arm: The name the points are reported under.
        trainer: The learner.
        split: The partition.
        seed: Seed of the training permutation and of the model.
        fractions: Training shares, strictly ascending in ``(0, 1]``.

    Returns:
        One point per fraction.

    Raises:
        ValueError: If the fractions are invalid.
    """
    require_fractions(fractions)
    order: NDArray[np.int64] = np.random.default_rng(seed).permutation(len(split.y_train))
    points: list[LearningCurvePoint] = []
    for fraction in fractions:
        reduced = training_prefix(split, order, fraction)
        proba = trainer.fit(reduced, seed).predict_positive_proba(split.x_test)
        points.append(
            {
                "arm": arm,
                "seed": seed,
                "train_fraction": fraction,
                "n_train": len(reduced.y_train),
                "quality": compute_quality(split.y_test, proba),
            }
        )
    return points


__all__ = ["learning_curve_seed", "require_fractions", "training_prefix"]
