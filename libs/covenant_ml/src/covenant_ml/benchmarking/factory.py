"""Construction and wiring of the benchmark's collaborators.

The single place that names concrete implementations. Everything else in the
package depends on Protocols, so swapping a learner or a partitioning
strategy is a change here and nowhere else.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .adapters import ClearGbmTrainer, LightGbmTrainer
from .protocols import DataSplit, SplitFactoryProto, TrainerProto
from .splitting import company_disjoint_split
from .types import BenchmarkConfig

#: Defaults reproducing the workload the ClearGBM performance work is tuned
#: against: 200 depth-6 trees over 64 histogram bins, single-threaded.
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_REPEATS = 5
DEFAULT_WARMUPS = 2


class _CompanyDisjointSplitFactory:
    """Partitions one loaded dataset, re-permuting companies per seed."""

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

    def __call__(self, seed: int) -> DataSplit:
        """Build the partition for one seed.

        Args:
            seed: Seed controlling the company permutation.

        Returns:
            The three-way partition.
        """
        return company_disjoint_split(
            self._features,
            self._labels,
            self._company_codes,
            seed,
        )


def make_benchmark_config(
    n_estimators: int = 200,
    max_depth: int = 6,
    max_bins: int = 64,
    num_leaves: int = 31,
    repeats: int = DEFAULT_REPEATS,
    warmups: int = DEFAULT_WARMUPS,
) -> BenchmarkConfig:
    """Build the shared configuration both learners are held to.

    Args:
        n_estimators: Boosting rounds.
        max_depth: Maximum tree depth.
        max_bins: Histogram bin count.
        num_leaves: Leaf cap. Binds LightGBM's leaf-wise growth only.
        repeats: Timed fits per learner per seed.
        warmups: Discarded fits before timing begins.

    Returns:
        The shared configuration.
    """
    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": 0.05,
        "max_bins": max_bins,
        "min_data_in_leaf": 20,
        "num_leaves": num_leaves,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "n_jobs": 1,
        "repeats": repeats,
        "warmups": warmups,
    }


def make_trainers(config: BenchmarkConfig) -> tuple[TrainerProto, TrainerProto]:
    """Construct both learners under the shared configuration.

    Args:
        config: Hyperparameters held identical across both learners.

    Returns:
        The ClearGBM and LightGBM trainers, in that order.
    """
    return ClearGbmTrainer(config), LightGbmTrainer(config)


def make_split_factory(
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    company_codes: NDArray[np.int64],
) -> SplitFactoryProto:
    """Construct the per-seed partitioner for a loaded dataset.

    Args:
        features: Feature matrix, shape (n_rows, n_features).
        labels: Binary labels, shape (n_rows,).
        company_codes: Integer company identifier per row.

    Returns:
        A callable producing the partition for a seed.
    """
    return _CompanyDisjointSplitFactory(features, labels, company_codes)


__all__ = [
    "DEFAULT_REPEATS",
    "DEFAULT_SEEDS",
    "DEFAULT_WARMUPS",
    "make_benchmark_config",
    "make_split_factory",
    "make_trainers",
]
