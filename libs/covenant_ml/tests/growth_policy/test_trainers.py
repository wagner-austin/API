"""Tests for the three arms, driven against the real learners.

Nothing is substituted here. Each trainer's whole job is to turn a real
vendor's real fit into the experiment's Protocol, so a trainer tested against
a stand-in would establish only that the stand-in was converted. The datasets
are small so the fits are quick, but they are genuinely learnable, which is
what lets a test assert that a fitted model discriminates rather than merely
returns an array of the right shape.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.growth_policy.protocols import ArmSpec, TwoWaySplit
from covenant_ml.growth_policy.trainers import (
    ClearGbmAnchorTrainer,
    LgbAnchorTrainer,
    XgbArmTrainer,
    XgbTrainedModel,
)
from covenant_ml.growth_policy.vendors import load_lgb_ctor, load_xgb_ctor

from .factories import make_config, make_separable_dataset
from .numeric import equal, label_mask, mean_of, select


def make_split(row_count: int = 120, feature_count: int = 4) -> TwoWaySplit:
    """Build a learnable partition with disjoint folds.

    Args:
        row_count: Rows per fold.
        feature_count: Feature columns.

    Returns:
        The partition.
    """
    features, labels = make_separable_dataset(row_count * 2, feature_count, seed=1)
    return TwoWaySplit(
        x_train=features[:row_count],
        y_train=labels[:row_count],
        x_test=features[row_count:],
        y_test=labels[row_count:],
    )


def assert_discriminates(proba: NDArray[np.float64], labels: NDArray[np.int64]) -> None:
    """Assert a prediction separates the two classes on average.

    Args:
        proba: Positive-class probabilities.
        labels: True labels.

    Raises:
        AssertionError: If positives do not score above negatives on average.
    """
    positives = mean_of(select(proba, label_mask(labels, 1)))
    negatives = mean_of(select(proba, label_mask(labels, 0)))
    assert positives > negatives


class TestXgbArmTrainer:
    """The instrument's arms."""

    def test_depthwise_arm_fits_and_discriminates(self) -> None:
        """A depth-wise fit should produce probabilities that separate the classes."""
        spec = ArmSpec("xgb depthwise d3", "depthwise", 3, 0)
        trainer = XgbArmTrainer(spec, make_config(), load_xgb_ctor())
        split = make_split()

        model = trainer.fit(split, 42)
        proba = model.predict_positive_proba(split.x_test)

        assert proba.shape == (len(split.x_test),)
        assert_discriminates(proba, split.y_test)

    def test_reports_the_spec_name(self) -> None:
        """The arm name should come from the spec, unchanged."""
        spec = ArmSpec("xgb lossguide L7", "lossguide", 0, 7)

        trainer = XgbArmTrainer(spec, make_config(), load_xgb_ctor())

        assert trainer.arm_name == "xgb lossguide L7"

    def test_leaf_budget_bounds_the_fitted_trees(self) -> None:
        """A lossguide arm must respect max_leaves, or the arms measure the same thing."""
        spec = ArmSpec("xgb lossguide L4", "lossguide", 0, 4)
        trainer = XgbArmTrainer(spec, make_config(), load_xgb_ctor())

        model = trainer.fit(make_split(row_count=200), 42)

        assert model.mean_leaves() <= 4.0

    def test_depth_budget_bounds_the_fitted_trees(self) -> None:
        """A depthwise arm at depth 2 cannot exceed four leaves per tree."""
        spec = ArmSpec("xgb depthwise d2", "depthwise", 2, 0)
        trainer = XgbArmTrainer(spec, make_config(), load_xgb_ctor())

        model = trainer.fit(make_split(row_count=200), 42)

        assert model.mean_leaves() <= 4.0

    def test_is_deterministic_at_a_fixed_seed(self) -> None:
        """Two fits at one seed should agree, or no comparison between arms holds."""
        spec = ArmSpec("xgb depthwise d3", "depthwise", 3, 0)
        trainer = XgbArmTrainer(spec, make_config(), load_xgb_ctor())
        split = make_split()

        first = trainer.fit(split, 42).predict_positive_proba(split.x_test)
        second = trainer.fit(split, 42).predict_positive_proba(split.x_test)

        assert equal(first, second)


class TestXgbTrainedModel:
    """The fitted-model wrapper."""

    def test_returns_the_positive_column(self) -> None:
        """Column one of predict_proba is the positive class."""
        constructor = load_xgb_ctor()
        split = make_split()
        estimator = constructor(
            n_estimators=3,
            learning_rate=0.3,
            max_bin=16,
            min_child_weight=2,
            tree_method="hist",
            grow_policy="depthwise",
            max_depth=3,
            max_leaves=0,
            reg_alpha=0.0,
            reg_lambda=0.0,
            n_jobs=1,
            random_state=42,
            eval_metric="logloss",
        )
        fitted = estimator.fit(split.x_train, split.y_train)

        model = XgbTrainedModel(fitted)
        proba = model.predict_positive_proba(split.x_test)

        raw: NDArray[np.float64] = np.asarray(fitted.predict_proba(split.x_test), dtype=np.float64)
        expected: NDArray[np.float64] = raw[:, 1]
        assert equal(proba, expected)


class TestLgbAnchorTrainer:
    """The LightGBM anchor."""

    def test_fits_and_discriminates(self) -> None:
        """A LightGBM fit should separate the classes."""
        trainer = LgbAnchorTrainer(4, 3, make_config(), load_lgb_ctor())
        split = make_split()

        proba = trainer.fit(split, 42).predict_positive_proba(split.x_test)

        assert proba.shape == (len(split.x_test),)
        assert_discriminates(proba, split.y_test)

    def test_names_itself_by_leaf_count(self) -> None:
        """The anchor's name should carry its leaf budget."""
        trainer = LgbAnchorTrainer(31, 6, make_config(), load_lgb_ctor())

        assert trainer.arm_name == "lgb leafwise L31"

    def test_reports_mean_leaves(self) -> None:
        """The wrapper should read a real leaf count from the booster."""
        trainer = LgbAnchorTrainer(4, 3, make_config(), load_lgb_ctor())

        model = trainer.fit(make_split(), 42)

        assert model.mean_leaves() > 0.0


class TestClearGbmAnchorTrainer:
    """The ClearGBM anchor."""

    def test_fits_and_discriminates(self) -> None:
        """A ClearGBM fit should separate the classes."""
        trainer = ClearGbmAnchorTrainer(3, make_config())
        split = make_split()

        proba = trainer.fit(split, 42).predict_positive_proba(split.x_test)

        assert proba.shape == (len(split.x_test),)
        assert_discriminates(proba, split.y_test)

    def test_names_itself_by_depth(self) -> None:
        """The anchor's name should carry its depth budget."""
        trainer = ClearGbmAnchorTrainer(6, make_config())

        assert trainer.arm_name == "cleargbm depthwise d6"

    def test_reports_mean_leaves(self) -> None:
        """The wrapper should read a real leaf count from the exported model."""
        trainer = ClearGbmAnchorTrainer(3, make_config())

        model = trainer.fit(make_split(), 42)

        assert model.mean_leaves() > 0.0

    def test_is_deterministic_at_a_fixed_seed(self) -> None:
        """Two fits at one seed should agree exactly."""
        trainer = ClearGbmAnchorTrainer(3, make_config())
        split = make_split()

        first = trainer.fit(split, 42).predict_positive_proba(split.x_test)
        second = trainer.fit(split, 42).predict_positive_proba(split.x_test)

        assert equal(first, second)


@pytest.mark.parametrize("depth", [1, 2])
def test_deeper_trees_grow_more_leaves(depth: int) -> None:
    """Leaf count should rise with the depth budget, confirming the budget binds."""
    shallow = XgbArmTrainer(ArmSpec("a", "depthwise", depth, 0), make_config(), load_xgb_ctor())
    deeper = XgbArmTrainer(ArmSpec("b", "depthwise", depth + 2, 0), make_config(), load_xgb_ctor())
    split = make_split(row_count=200)

    assert shallow.fit(split, 42).mean_leaves() <= deeper.fit(split, 42).mean_leaves()
