"""Tests for cleargbm.types: the config axes.

The growth-policy, leaf-budget and objective spellings — enumeration
constants, their require_* validators, and the decode paths that pair
them (regression weight pairing, growth/num_leaves pairing).
"""

from __future__ import annotations

import pytest

from cleargbm.types import (
    GROWTH_STRATEGIES,
    OBJECTIVES,
    JSONDict,
    JSONTypeError,
    decode_gradient_boosting_config,
    require_growth_strategy,
    require_leaf_budget,
    require_objective,
)


class TestConfigAxes:
    """The growth, budget and objective axes through decode and require_*."""

    def test_growth_strategies_enumerates_both_policies(self) -> None:
        """The closed literal and its runtime tuple must not drift apart."""
        assert GROWTH_STRATEGIES == ("depth_wise", "leaf_wise")

    def test_require_growth_strategy_accepts_every_enumerated_value(self) -> None:
        """Every value in the tuple must survive narrowing."""
        narrowed = [
            require_growth_strategy(value, "growth_strategy") for value in GROWTH_STRATEGIES
        ]
        assert narrowed == ["depth_wise", "leaf_wise"]

    def test_require_growth_strategy_rejects_unknown_value(self) -> None:
        """An unknown policy names itself and the accepted set."""
        with pytest.raises(ValueError, match="growth_strategy must be one of"):
            require_growth_strategy("lossguide", "growth_strategy")

    def test_require_leaf_budget_accepts_the_smallest_usable_budget(self) -> None:
        """Two leaves is one split: the smallest tree that is not a stump."""
        assert require_leaf_budget(2, "num_leaves") == 2

    def test_require_leaf_budget_rejects_a_budget_of_one(self) -> None:
        with pytest.raises(ValueError, match="num_leaves must be >= 2"):
            require_leaf_budget(1, "num_leaves")

    def test_objectives_enumerates_every_loss(self) -> None:
        """The closed literal and its runtime tuple must not drift apart."""
        assert OBJECTIVES == (
            "binary_log_loss",
            "squared_error",
            "multiclass_softmax",
            "lambdarank",
        )

    def test_require_objective_accepts_every_enumerated_value(self) -> None:
        """Every value in the tuple must survive narrowing."""
        narrowed = [require_objective(value, "objective") for value in OBJECTIVES]
        assert narrowed == [
            "binary_log_loss",
            "squared_error",
            "multiclass_softmax",
            "lambdarank",
        ]

    def test_require_objective_rejects_unknown_value(self) -> None:
        """An unknown objective names itself and the accepted set."""
        with pytest.raises(ValueError, match="objective must be one of"):
            require_objective("reg:squarederror", "objective")

    def test_decode_regression_config_with_null_weight(self) -> None:
        """The squared-error pairing decodes: null weight, regression loss.

        The pairing itself is enforced at the Rust boundary; this layer must
        pass a well-typed regression config through unchanged.
        """
        raw: JSONDict = {
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "colsample_bytree": None,
            "categorical_features": None,
            "n_classes": None,
            "lambdarank_truncation_level": None,
            "goss_top_rate": None,
            "goss_other_rate": None,
            "quantized_gradient_bins": None,
            "min_data_in_bin": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 0,
            "monotonic_constraints": None,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 1,
            "early_stopping_rounds": None,
            "growth_strategy": "depth_wise",
            "num_leaves": None,
            "objective": "squared_error",
            "scale_pos_weight": None,
        }
        decoded = decode_gradient_boosting_config(raw)
        assert decoded["objective"] == "squared_error"
        assert decoded["scale_pos_weight"] is None

    def test_decode_rejects_a_non_positive_weight(self) -> None:
        """A present weight must still be a finite positive float."""
        raw: JSONDict = {
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "colsample_bytree": None,
            "categorical_features": None,
            "n_classes": None,
            "lambdarank_truncation_level": None,
            "goss_top_rate": None,
            "goss_other_rate": None,
            "quantized_gradient_bins": None,
            "min_data_in_bin": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 0,
            "monotonic_constraints": None,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 1,
            "early_stopping_rounds": None,
            "growth_strategy": "depth_wise",
            "num_leaves": None,
            "objective": "binary_log_loss",
            "scale_pos_weight": 0.0,
        }
        with pytest.raises(ValueError, match="scale_pos_weight"):
            decode_gradient_boosting_config(raw)

    def test_decode_rejects_an_unknown_objective(self) -> None:
        """An unknown objective in a payload should raise ValueError."""
        raw: JSONDict = {
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "colsample_bytree": None,
            "categorical_features": None,
            "n_classes": None,
            "lambdarank_truncation_level": None,
            "goss_top_rate": None,
            "goss_other_rate": None,
            "quantized_gradient_bins": None,
            "min_data_in_bin": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 0,
            "monotonic_constraints": None,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 1,
            "early_stopping_rounds": None,
            "growth_strategy": "depth_wise",
            "num_leaves": None,
            "objective": "regression",  # invalid
            "scale_pos_weight": 1.0,
        }
        with pytest.raises(ValueError, match="objective must be one of"):
            decode_gradient_boosting_config(raw)

    def test_decode_growth_strategy_invalid(self) -> None:
        """An unknown policy in a payload should raise ValueError."""
        raw: JSONDict = {
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "colsample_bytree": None,
            "categorical_features": None,
            "n_classes": None,
            "lambdarank_truncation_level": None,
            "goss_top_rate": None,
            "goss_other_rate": None,
            "quantized_gradient_bins": None,
            "min_data_in_bin": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 0,
            "monotonic_constraints": None,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 1,
            "early_stopping_rounds": None,
            "growth_strategy": "lossguide",  # invalid
            "num_leaves": None,
            "objective": "binary_log_loss",
            "scale_pos_weight": 1.0,
        }
        with pytest.raises(ValueError, match="growth_strategy must be one of"):
            decode_gradient_boosting_config(raw)

    def test_decode_growth_strategy_wrong_type(self) -> None:
        """A non-string policy should raise JSONTypeError."""
        raw: JSONDict = {
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "colsample_bytree": None,
            "categorical_features": None,
            "n_classes": None,
            "lambdarank_truncation_level": None,
            "goss_top_rate": None,
            "goss_other_rate": None,
            "quantized_gradient_bins": None,
            "min_data_in_bin": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 0,
            "monotonic_constraints": None,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 1,
            "early_stopping_rounds": None,
            "growth_strategy": 1,  # invalid
            "num_leaves": None,
            "objective": "binary_log_loss",
            "scale_pos_weight": 1.0,
        }
        with pytest.raises(JSONTypeError, match="growth_strategy must be str"):
            decode_gradient_boosting_config(raw)

    def test_decode_leaf_wise_is_accepted_by_the_python_layer(self) -> None:
        """Python validates the vocabulary; Rust owns the implementation gate.

        ``leaf_wise`` decodes cleanly here and is refused at the Rust boundary.
        Keeping the rejection in one place means the Python layer never has to
        be edited when the builder gains the policy.
        """
        raw: JSONDict = {
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "colsample_bytree": None,
            "categorical_features": None,
            "n_classes": None,
            "lambdarank_truncation_level": None,
            "goss_top_rate": None,
            "goss_other_rate": None,
            "quantized_gradient_bins": None,
            "min_data_in_bin": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 0,
            "monotonic_constraints": None,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 1,
            "early_stopping_rounds": None,
            "growth_strategy": "leaf_wise",
            "num_leaves": 31,
            "objective": "binary_log_loss",
            "scale_pos_weight": 1.0,
        }
        decoded = decode_gradient_boosting_config(raw)
        assert decoded["growth_strategy"] == "leaf_wise"
        assert decoded["num_leaves"] == 31

    def test_decode_num_leaves_below_two(self) -> None:
        """A leaf budget of 1 cannot describe a tree with a split."""
        raw: JSONDict = {
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "colsample_bytree": None,
            "categorical_features": None,
            "n_classes": None,
            "lambdarank_truncation_level": None,
            "goss_top_rate": None,
            "goss_other_rate": None,
            "quantized_gradient_bins": None,
            "min_data_in_bin": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 0,
            "monotonic_constraints": None,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 1,
            "early_stopping_rounds": None,
            "growth_strategy": "leaf_wise",
            "num_leaves": 1,  # invalid
        }
        with pytest.raises(ValueError, match="num_leaves must be >= 2"):
            decode_gradient_boosting_config(raw)
