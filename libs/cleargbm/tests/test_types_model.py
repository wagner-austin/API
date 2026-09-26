"""Tests for cleargbm.types: model configuration and payloads."""

from __future__ import annotations

import pytest

from cleargbm.types import (
    GradientBoostingConfig,
    GrowthStrategy,
    JSONDict,
    JSONTypeError,
    Objective,
    decode_gradient_boosting_config,
    encode_gradient_boosting_config,
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
            "n_classes": None,
            "lambdarank_truncation_level": None,
            "goss_top_rate": None,
            "goss_other_rate": None,
            "quantized_gradient_bins": None,
            "min_data_in_bin": None,
            "max_bins": 64,
            "subsample": 0.8,
            "random_state": 42,
            "monotonic_constraints": (1, -1, 0),
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "n_jobs": 4,
            "early_stopping_rounds": 10,
            "growth_strategy": GrowthStrategy.DEPTH_WISE,
            "num_leaves": None,
            "objective": Objective.BINARY_LOG_LOSS,
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
        assert decoded["growth_strategy"] is GrowthStrategy.DEPTH_WISE
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
            "growth_strategy": GrowthStrategy.DEPTH_WISE,
            "num_leaves": None,
            "objective": Objective.BINARY_LOG_LOSS,
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
            "n_classes": None,
            "lambdarank_truncation_level": None,
            "goss_top_rate": None,
            "goss_other_rate": None,
            "quantized_gradient_bins": None,
            "min_data_in_bin": None,
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
            "n_classes": None,
            "lambdarank_truncation_level": None,
            "goss_top_rate": None,
            "goss_other_rate": None,
            "quantized_gradient_bins": None,
            "min_data_in_bin": None,
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
            "n_classes": None,
            "lambdarank_truncation_level": None,
            "goss_top_rate": None,
            "goss_other_rate": None,
            "quantized_gradient_bins": None,
            "min_data_in_bin": None,
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
            "n_classes": None,
            "lambdarank_truncation_level": None,
            "goss_top_rate": None,
            "goss_other_rate": None,
            "quantized_gradient_bins": None,
            "min_data_in_bin": None,
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
            "n_jobs": -1,
            "early_stopping_rounds": None,
            "growth_strategy": GrowthStrategy.DEPTH_WISE,
            "num_leaves": None,
            "objective": Objective.BINARY_LOG_LOSS,
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
            "n_jobs": 0,  # invalid
            "early_stopping_rounds": None,
            "growth_strategy": "depth_wise",
            "num_leaves": None,
            "objective": "binary_log_loss",
            "scale_pos_weight": 1.0,
        }
        with pytest.raises(ValueError, match="n_jobs must be -1 or positive"):
            decode_gradient_boosting_config(raw)
