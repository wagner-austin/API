"""Construction and wiring of the growth-policy experiment's collaborators.

The single place that names a concrete learner, a concrete metric, or a
concrete partitioning strategy. Everything else in the package depends on the
Protocols in :mod:`covenant_ml.growth_policy.protocols`, so swapping an arm or
a scorer is a change here and nowhere else.

The defaults reproduce the run recorded in
``libs/cleargbm/docs/EXPERIMENT_2026-08-17_growth_policy_xgb_instrument.md``:
200 trees, learning rate 0.05, 64 histogram bins, min-leaf 20, no L1 or L2,
single-threaded, seeds 42/43/44, median of three timed fits after one warmup.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .datasets import company_disjoint_indices
from .metrics import SklearnMetrics
from .protocols import ArmSpec, ArmTrainerProto, MetricsProto, SplitFactoryProto, TwoWaySplit
from .trainers import ClearGbmAnchorTrainer, LgbAnchorTrainer, XgbArmTrainer
from .types import ERR_DUPLICATE_ARM, ERR_NO_ARMS, ExperimentConfig
from .vendors import (
    load_average_precision,
    load_lgb_ctor,
    load_log_loss,
    load_roc_auc,
    load_stratified_split,
    load_xgb_ctor,
)

#: Seeds the recorded run measured at.
DEFAULT_SEEDS = (42, 43, 44)

#: Timed fits per arm per seed, and discarded fits before timing begins.
DEFAULT_REPEATS = 3
DEFAULT_WARMUPS = 1

#: Depth budget for the depth-wise arms, and leaf budgets for the leaf-wise
#: ones. 31 is LightGBM's default shape; 47 is ClearGBM's measured mean leaf
#: count, so the pair brackets the range the comparison cares about.
DEFAULT_MAX_DEPTH = 6
DEFAULT_LEAF_BUDGETS = (31, 47)

#: Fraction of rows held out by the stratified splitter.
STRATIFIED_TEST_SIZE = 0.30


class _GroupDisjointSplitFactory:
    """Partitions a grouped dataset, re-permuting groups per seed."""

    def __init__(
        self,
        features: NDArray[np.float64],
        labels: NDArray[np.int64],
        groups: list[str],
    ) -> None:
        """Bind the dataset this factory partitions.

        Args:
            features: Feature matrix, shape (n_rows, n_features).
            labels: Binary labels, shape (n_rows,).
            groups: Grouping key per row.
        """
        self._features = features
        self._labels = labels
        self._groups = groups

    def __call__(self, seed: int) -> TwoWaySplit:
        """Build the partition for one seed.

        Args:
            seed: Seed controlling the group permutation.

        Returns:
            The train/test partition.
        """
        train_index, test_index = company_disjoint_indices(self._groups, seed)
        return TwoWaySplit(
            x_train=self._features[train_index],
            y_train=self._labels[train_index],
            x_test=self._features[test_index],
            y_test=self._labels[test_index],
        )


class _StratifiedSplitFactory:
    """Partitions an ungrouped dataset, preserving class proportions."""

    def __init__(
        self,
        features: NDArray[np.float64],
        labels: NDArray[np.int64],
    ) -> None:
        """Bind the dataset this factory partitions.

        Args:
            features: Feature matrix, shape (n_rows, n_features).
            labels: Binary labels, shape (n_rows,).
        """
        self._features = features
        self._labels = labels
        self._split = load_stratified_split()

    def __call__(self, seed: int) -> TwoWaySplit:
        """Build the partition for one seed.

        Args:
            seed: Seed controlling the permutation.

        Returns:
            The train/test partition.
        """
        folds = self._split(
            self._features,
            self._labels,
            test_size=STRATIFIED_TEST_SIZE,
            random_state=seed,
            stratify=self._labels,
        )
        return TwoWaySplit(
            x_train=np.asarray(folds[0], dtype=np.float64),
            y_train=np.asarray(folds[2], dtype=np.int64),
            x_test=np.asarray(folds[1], dtype=np.float64),
            y_test=np.asarray(folds[3], dtype=np.int64),
        )


def make_experiment_config(
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_bins: int = 64,
    min_leaf: int = 20,
    repeats: int = DEFAULT_REPEATS,
    warmups: int = DEFAULT_WARMUPS,
) -> ExperimentConfig:
    """Build the configuration every arm is held to.

    Args:
        n_estimators: Boosting rounds.
        learning_rate: Shrinkage applied to each tree's contribution.
        max_bins: Histogram bin count.
        min_leaf: Minimum-child constraint.
        repeats: Timed fits per arm per seed.
        warmups: Discarded fits before timing begins.

    Returns:
        The shared configuration.
    """
    return {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_bins": max_bins,
        "min_leaf": min_leaf,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "n_jobs": 1,
        "repeats": repeats,
        "warmups": warmups,
    }


def make_arm_specs(
    max_depth: int = DEFAULT_MAX_DEPTH,
    leaf_budgets: Sequence[int] = DEFAULT_LEAF_BUDGETS,
) -> list[ArmSpec]:
    """Build the XGBoost arm specifications.

    One depth-wise arm bounded by ``max_depth``, then one leaf-wise arm per
    leaf budget. Each arm disables the budget it is not using, so no arm leaves
    a bound implicit.

    Args:
        max_depth: Depth budget for the depth-wise arm.
        leaf_budgets: Leaf budgets, one leaf-wise arm each.

    Returns:
        The arm specifications, depth-wise first.

    Raises:
        ValueError: If no arms would be produced, or if two arms would share a
            name, which would silently merge them in the summary.
    """
    specs = [
        ArmSpec(
            name=f"xgb depthwise d{max_depth}",
            grow_policy="depthwise",
            max_depth=max_depth,
            max_leaves=0,
        )
    ]
    for budget in leaf_budgets:
        specs.append(
            ArmSpec(
                name=f"xgb lossguide L{budget}",
                grow_policy="lossguide",
                max_depth=0,
                max_leaves=budget,
            )
        )
    _reject_duplicate_names([spec.name for spec in specs])
    return specs


def _reject_duplicate_names(names: Sequence[str]) -> None:
    """Fail when two arms share a display name.

    Args:
        names: The arm names to check.

    Raises:
        ValueError: If the list is empty, or if any name repeats. A repeated
            name would merge two arms into one summary row, reporting a mean
            over two different configurations as though it were one.
    """
    if len(names) == 0:
        raise ValueError(f"[{ERR_NO_ARMS}] At least one arm is required")
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(f"[{ERR_DUPLICATE_ARM}] Arm name '{name}' is not unique")
        seen.add(name)


def make_xgb_trainers(
    config: ExperimentConfig,
    specs: Sequence[ArmSpec],
) -> list[ArmTrainerProto]:
    """Construct one trainer per XGBoost arm.

    Args:
        config: Hyperparameters shared across arms.
        specs: The arm specifications to build.

    Returns:
        The trainers, in specification order.

    Raises:
        ValueError: If the specifications are empty or carry a repeated name.
    """
    _reject_duplicate_names([spec.name for spec in specs])
    constructor = load_xgb_ctor()
    return [XgbArmTrainer(spec, config, constructor) for spec in specs]


def make_anchor_trainers(
    config: ExperimentConfig,
    num_leaves: int = 31,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> list[ArmTrainerProto]:
    """Construct the LightGBM and ClearGBM anchor arms.

    Args:
        config: Hyperparameters shared across arms.
        num_leaves: Leaf cap for the LightGBM anchor.
        max_depth: Depth cap for both anchors.

    Returns:
        The LightGBM anchor followed by the ClearGBM anchor.
    """
    return [
        LgbAnchorTrainer(num_leaves, max_depth, config, load_lgb_ctor()),
        ClearGbmAnchorTrainer(max_depth, config),
    ]


def make_metrics() -> MetricsProto:
    """Construct the scorer for the held-out fold.

    Returns:
        The scikit-learn-backed metric set.
    """
    return SklearnMetrics(load_roc_auc(), load_average_precision(), load_log_loss())


def make_group_split_factory(
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    groups: list[str],
) -> SplitFactoryProto:
    """Construct the group-disjoint partitioner for a loaded dataset.

    Args:
        features: Feature matrix, shape (n_rows, n_features).
        labels: Binary labels, shape (n_rows,).
        groups: Grouping key per row.

    Returns:
        A callable producing the partition for a seed.
    """
    return _GroupDisjointSplitFactory(features, labels, groups)


def make_stratified_split_factory(
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> SplitFactoryProto:
    """Construct the stratified partitioner for a loaded dataset.

    Used for the datasets that carry no grouping key, where a group-disjoint
    partition is not defined.

    Args:
        features: Feature matrix, shape (n_rows, n_features).
        labels: Binary labels, shape (n_rows,).

    Returns:
        A callable producing the partition for a seed.
    """
    return _StratifiedSplitFactory(features, labels)


__all__ = [
    "DEFAULT_LEAF_BUDGETS",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_REPEATS",
    "DEFAULT_SEEDS",
    "DEFAULT_WARMUPS",
    "STRATIFIED_TEST_SIZE",
    "make_anchor_trainers",
    "make_arm_specs",
    "make_experiment_config",
    "make_group_split_factory",
    "make_metrics",
    "make_stratified_split_factory",
    "make_xgb_trainers",
]
