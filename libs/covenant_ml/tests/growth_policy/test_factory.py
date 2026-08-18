"""Tests for the wiring layer."""

from __future__ import annotations

import numpy as np
import pytest

from covenant_ml.growth_policy.factory import (
    DEFAULT_LEAF_BUDGETS,
    DEFAULT_MAX_DEPTH,
    STRATIFIED_TEST_SIZE,
    make_anchor_trainers,
    make_arm_specs,
    make_experiment_config,
    make_group_split_factory,
    make_metrics,
    make_stratified_split_factory,
    make_xgb_trainers,
)
from covenant_ml.growth_policy.protocols import ArmSpec
from covenant_ml.growth_policy.types import ERR_DUPLICATE_ARM, ERR_NO_ARMS

from .factories import make_config, make_separable_dataset
from .numeric import as_int_list, columns_of, equal, floats, ints


class TestMakeExperimentConfig:
    """The shared configuration."""

    def test_defaults_match_the_recorded_run(self) -> None:
        """The defaults are the values the published tables were measured under."""
        config = make_experiment_config()

        assert config["n_estimators"] == 200
        assert config["learning_rate"] == 0.05
        assert config["max_bins"] == 64
        assert config["min_leaf"] == 20
        assert config["reg_alpha"] == 0.0
        assert config["reg_lambda"] == 0.0
        assert config["n_jobs"] == 1

    def test_overrides_are_honoured(self) -> None:
        """Every parameter should be overridable for a quicker run."""
        config = make_experiment_config(n_estimators=5, repeats=1, warmups=0)

        assert config["n_estimators"] == 5
        assert config["repeats"] == 1
        assert config["warmups"] == 0


class TestMakeArmSpecs:
    """The arm specifications."""

    def test_builds_one_depthwise_arm_then_one_arm_per_budget(self) -> None:
        """Depth-wise leads, then a leaf-wise arm per budget in order."""
        specs = make_arm_specs(6, [31, 47])

        assert [spec.name for spec in specs] == [
            "xgb depthwise d6",
            "xgb lossguide L31",
            "xgb lossguide L47",
        ]

    def test_depthwise_arm_disables_the_leaf_budget(self) -> None:
        """A depth-wise arm must not carry a leaf bound."""
        specs = make_arm_specs(6, [31])

        assert specs[0].grow_policy == "depthwise"
        assert specs[0].max_depth == 6
        assert specs[0].max_leaves == 0

    def test_leafwise_arm_disables_the_depth_budget(self) -> None:
        """A leaf-wise arm must not carry a depth bound."""
        specs = make_arm_specs(6, [31])

        assert specs[1].grow_policy == "lossguide"
        assert specs[1].max_depth == 0
        assert specs[1].max_leaves == 31

    def test_defaults_bracket_the_measured_leaf_counts(self) -> None:
        """The default budgets are LightGBM's shape and ClearGBM's measured mean."""
        specs = make_arm_specs()

        assert len(specs) == 1 + len(DEFAULT_LEAF_BUDGETS)
        assert specs[0].max_depth == DEFAULT_MAX_DEPTH

    def test_rejects_a_repeated_leaf_budget(self) -> None:
        """Two identical budgets would collapse into one summary row."""
        with pytest.raises(ValueError, match=ERR_DUPLICATE_ARM):
            make_arm_specs(6, [31, 31])


class TestMakeXgbTrainers:
    """Trainer construction from specs."""

    def test_builds_one_trainer_per_spec(self) -> None:
        """Names should follow the specs, in order."""
        specs = make_arm_specs(3, [4])

        trainers = make_xgb_trainers(make_config(), specs)

        assert [trainer.arm_name for trainer in trainers] == [spec.name for spec in specs]

    def test_rejects_duplicate_arm_names(self) -> None:
        """Two specs sharing a name would merge in the summary."""
        spec = ArmSpec("same", "depthwise", 3, 0)

        with pytest.raises(ValueError, match=ERR_DUPLICATE_ARM):
            make_xgb_trainers(make_config(), [spec, spec])

    def test_rejects_an_empty_spec_list(self) -> None:
        """An experiment with no arms would state nothing."""
        with pytest.raises(ValueError, match=ERR_NO_ARMS):
            make_xgb_trainers(make_config(), [])


class TestMakeAnchorTrainers:
    """Anchor construction."""

    def test_builds_the_lightgbm_and_cleargbm_anchors(self) -> None:
        """Both anchors should be present, LightGBM first."""
        anchors = make_anchor_trainers(make_config(), num_leaves=8, max_depth=3)

        assert [anchor.arm_name for anchor in anchors] == [
            "lgb leafwise L8",
            "cleargbm depthwise d3",
        ]


class TestMakeGroupSplitFactory:
    """Group-disjoint partitioning through the factory."""

    def test_partitions_rows_without_splitting_a_group(self) -> None:
        """No company may appear in both folds."""
        features, labels = make_separable_dataset(row_count=80, feature_count=3)
        groups = [f"C{index % 10}" for index in range(80)]

        split = make_group_split_factory(features, labels, groups)(42)

        assert len(split.x_train) + len(split.x_test) == 80
        assert columns_of(split.x_train) == 3

    def test_is_reproducible_at_a_fixed_seed(self) -> None:
        """One seed must give one partition, or arms are not comparable."""
        features, labels = make_separable_dataset(row_count=80, feature_count=3)
        groups = [f"C{index % 10}" for index in range(80)]
        factory = make_group_split_factory(features, labels, groups)

        first = factory(42)
        second = factory(42)

        assert equal(first.x_train, second.x_train)
        assert as_int_list(first.y_test) == as_int_list(second.y_test)

    def test_different_seeds_give_different_partitions(self) -> None:
        """Re-permuting per seed is what makes three seeds three measurements."""
        features, labels = make_separable_dataset(row_count=200, feature_count=3)
        groups = [f"C{index % 40}" for index in range(200)]
        factory = make_group_split_factory(features, labels, groups)

        assert not equal(factory(42).x_train, factory(43).x_train)


class TestMakeStratifiedSplitFactory:
    """Stratified partitioning through the factory."""

    def test_holds_out_the_configured_fraction(self) -> None:
        """The test fold should be the configured share of the rows."""
        features, labels = make_separable_dataset(row_count=100, feature_count=3)

        split = make_stratified_split_factory(features, labels)(42)

        assert len(split.x_test) == int(100 * STRATIFIED_TEST_SIZE)
        assert len(split.x_train) == 100 - int(100 * STRATIFIED_TEST_SIZE)

    def test_assigns_features_and_labels_to_the_right_folds(self) -> None:
        """The splitter returns features before labels; the factory must not swap them."""
        features, labels = make_separable_dataset(row_count=100, feature_count=3)

        split = make_stratified_split_factory(features, labels)(42)

        assert columns_of(split.x_train) == 3
        assert columns_of(split.x_test) == 3
        assert split.y_train.ndim == 1
        assert split.y_test.ndim == 1

    def test_labels_keep_an_integer_dtype(self) -> None:
        """A float label would misfit every learner that expects a class."""
        features, labels = make_separable_dataset(row_count=100, feature_count=3)

        split = make_stratified_split_factory(features, labels)(42)

        assert split.y_train.dtype == np.int64
        assert split.y_test.dtype == np.int64

    def test_is_reproducible_at_a_fixed_seed(self) -> None:
        """One seed must give one partition."""
        features, labels = make_separable_dataset(row_count=100, feature_count=3)
        factory = make_stratified_split_factory(features, labels)

        assert equal(factory(42).x_train, factory(42).x_train)


class TestMakeMetrics:
    """The scorer factory."""

    def test_builds_a_usable_scorer(self) -> None:
        """The constructed scorer should score a real prediction."""
        metrics = make_metrics()
        labels = ints([0, 1])
        scores = floats([0.2, 0.8])

        assert metrics.auc_roc(labels, scores) == 1.0
