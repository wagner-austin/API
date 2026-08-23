"""Tests for cleargbm.types: model configuration and payloads."""

from __future__ import annotations

import pytest

from cleargbm.types import (
    GROWTH_STRATEGIES,
    OBJECTIVES,
    GradientBoostingConfig,
    JSONDict,
    JSONTypeError,
    JSONValue,
    decode_gradient_boosting_config,
    encode_gradient_boosting_config,
    require_growth_strategy,
    require_leaf_budget,
    require_objective,
)

# =============================================================================
# GradientBoostingConfig Tests
# =============================================================================


class TestGradientBoostingConfig:
    """Tests for GradientBoostingConfig encode/decode."""

    def test_encode_decode_roundtrip(self) -> None:
        """Encode then decode should preserve data."""
        original: GradientBoostingConfig = {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "max_features": 3,
            "colsample_bytree": 0.7,
            "categorical_features": (1, 3),
            "max_bins": 64,
            "subsample": 0.8,
            "random_state": 42,
            "monotonic_constraints": (1, -1, 0),
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "n_jobs": 4,
            "early_stopping_rounds": 10,
            "growth_strategy": "depth_wise",
            "num_leaves": None,
            "objective": "binary_log_loss",
            "scale_pos_weight": 1.0,
        }
        encoded = encode_gradient_boosting_config(original)
        decoded = decode_gradient_boosting_config(encoded)

        assert decoded["n_estimators"] == 100
        assert decoded["max_depth"] == 4
        assert decoded["learning_rate"] == 0.1
        assert decoded["min_samples_split"] == 10
        assert decoded["min_samples_leaf"] == 5
        assert decoded["max_features"] == 3
        assert decoded["colsample_bytree"] == 0.7
        assert decoded["categorical_features"] == (1, 3)
        assert decoded["max_bins"] == 64
        assert decoded["subsample"] == 0.8
        assert decoded["random_state"] == 42
        assert decoded["monotonic_constraints"] == (1, -1, 0)
        assert decoded["reg_alpha"] == 0.1
        assert decoded["reg_lambda"] == 1.0
        assert decoded["n_jobs"] == 4
        assert decoded["early_stopping_rounds"] == 10
        assert decoded["growth_strategy"] == "depth_wise"
        assert decoded["num_leaves"] is None

    def test_encode_decode_with_none_optionals(self) -> None:
        """None values for optional fields should roundtrip."""
        original: GradientBoostingConfig = {
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "colsample_bytree": None,
            "categorical_features": None,
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
            "scale_pos_weight": 1.0,
        }
        encoded = encode_gradient_boosting_config(original)
        decoded = decode_gradient_boosting_config(encoded)

        assert decoded["max_features"] is None
        assert decoded["colsample_bytree"] is None
        assert decoded["monotonic_constraints"] is None
        assert decoded["reg_alpha"] == 0.0
        assert decoded["reg_lambda"] == 0.0
        assert decoded["n_jobs"] == 1
        assert decoded["early_stopping_rounds"] is None

    def test_decode_invalid_monotonic_constraint_value(self) -> None:
        """Monotonic constraint value not in {-1, 0, 1} should raise ValueError."""
        raw: JSONDict = {
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "colsample_bytree": None,
            "categorical_features": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 0,
            "monotonic_constraints": [2],  # invalid value
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 1,
            "early_stopping_rounds": None,
            "growth_strategy": "depth_wise",
            "num_leaves": None,
            "objective": "binary_log_loss",
            "scale_pos_weight": 1.0,
        }
        with pytest.raises(ValueError, match=r"monotonic_constraints\[0\] must be -1, 0, or 1"):
            decode_gradient_boosting_config(raw)

    def test_decode_monotonic_constraints_not_list(self) -> None:
        """monotonic_constraints not a list should raise TypeError."""
        raw: JSONDict = {
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "colsample_bytree": None,
            "categorical_features": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 0,
            "monotonic_constraints": "not a list",
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 1,
            "early_stopping_rounds": None,
            "growth_strategy": "depth_wise",
            "num_leaves": None,
            "objective": "binary_log_loss",
            "scale_pos_weight": 1.0,
        }
        with pytest.raises(JSONTypeError, match="monotonic_constraints must be list or None"):
            decode_gradient_boosting_config(raw)

    def test_decode_monotonic_constraint_not_int(self) -> None:
        """Monotonic constraint item not an int should raise TypeError."""
        raw: JSONDict = {
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "colsample_bytree": None,
            "categorical_features": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 0,
            "monotonic_constraints": ["not an int"],
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 1,
            "early_stopping_rounds": None,
            "growth_strategy": "depth_wise",
            "num_leaves": None,
            "objective": "binary_log_loss",
            "scale_pos_weight": 1.0,
        }
        with pytest.raises(JSONTypeError, match=r"monotonic_constraints\[0\] must be int"):
            decode_gradient_boosting_config(raw)

    def test_decode_bool_in_monotonic_constraints_fails(self) -> None:
        """Boolean in monotonic_constraints should raise TypeError."""
        raw: JSONDict = {
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "colsample_bytree": None,
            "categorical_features": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 0,
            "monotonic_constraints": [True],  # bool, not int
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 1,
            "early_stopping_rounds": None,
            "growth_strategy": "depth_wise",
            "num_leaves": None,
            "objective": "binary_log_loss",
            "scale_pos_weight": 1.0,
        }
        with pytest.raises(JSONTypeError, match=r"monotonic_constraints\[0\] must be int"):
            decode_gradient_boosting_config(raw)

    def test_encode_decode_n_jobs_minus_one(self) -> None:
        """n_jobs=-1 (all cores) should roundtrip."""
        original: GradientBoostingConfig = {
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "colsample_bytree": None,
            "categorical_features": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 0,
            "monotonic_constraints": None,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": -1,
            "early_stopping_rounds": None,
            "growth_strategy": "depth_wise",
            "num_leaves": None,
            "objective": "binary_log_loss",
            "scale_pos_weight": 1.0,
        }
        encoded = encode_gradient_boosting_config(original)
        decoded = decode_gradient_boosting_config(encoded)

        assert decoded["n_jobs"] == -1

    def test_decode_n_jobs_invalid(self) -> None:
        """Invalid n_jobs should raise ValueError."""
        raw: JSONDict = {
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "colsample_bytree": None,
            "categorical_features": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 0,
            "monotonic_constraints": None,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 0,  # invalid
            "early_stopping_rounds": None,
            "growth_strategy": "depth_wise",
            "num_leaves": None,
            "objective": "binary_log_loss",
            "scale_pos_weight": 1.0,
        }
        with pytest.raises(ValueError, match="n_jobs must be -1 or positive"):
            decode_gradient_boosting_config(raw)

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

    def test_objectives_enumerates_both_losses(self) -> None:
        """The closed literal and its runtime tuple must not drift apart."""
        assert OBJECTIVES == ("binary_log_loss", "squared_error")

    def test_require_objective_accepts_every_enumerated_value(self) -> None:
        """Every value in the tuple must survive narrowing."""
        narrowed = [require_objective(value, "objective") for value in OBJECTIVES]
        assert narrowed == ["binary_log_loss", "squared_error"]

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


# =============================================================================
# GradientBoostingModel Tests
# =============================================================================


class TestCategoricalFeaturesDecode:
    """Validation of the categorical_features index list."""

    @staticmethod
    def _raw(value: JSONValue) -> JSONDict:
        return {
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "colsample_bytree": None,
            "categorical_features": value,
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
            "scale_pos_weight": 1.0,
        }

    def test_rejects_a_non_list(self) -> None:
        """A dict is neither a list nor null."""
        with pytest.raises(JSONTypeError, match="categorical_features must be list or None"):
            decode_gradient_boosting_config(self._raw({"a": 1}))

    def test_rejects_an_empty_list(self) -> None:
        """Null is the only spelling of all-numeric."""
        with pytest.raises(ValueError, match="non-empty"):
            decode_gradient_boosting_config(self._raw([]))

    def test_rejects_a_non_int_element(self) -> None:
        """Strings and bools are not feature indices."""
        with pytest.raises(JSONTypeError, match=r"categorical_features\[1\] must be int"):
            decode_gradient_boosting_config(self._raw([0, "one"]))
        with pytest.raises(JSONTypeError, match=r"categorical_features\[0\] must be int"):
            decode_gradient_boosting_config(self._raw([True]))

    def test_rejects_a_negative_index(self) -> None:
        """Feature indices are non-negative."""
        with pytest.raises(ValueError, match=r"must be >= 0, got -1"):
            decode_gradient_boosting_config(self._raw([-1]))

    def test_rejects_an_unsorted_list(self) -> None:
        """Strictly ascending is the one canonical spelling of a set."""
        with pytest.raises(ValueError, match="strictly ascending"):
            decode_gradient_boosting_config(self._raw([2, 1]))
        with pytest.raises(ValueError, match="strictly ascending"):
            decode_gradient_boosting_config(self._raw([1, 1]))
