"""Tests for _parse_external_train_config function."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError, dump_json_str

from covenant_radar_api.worker._train_external_parsers import (
    parse_external_train_config as _parse_external_train_config,
)


class TestLightGBMConfig:
    """Tests for parsing LightGBM configuration."""

    def test_valid(self) -> None:
        """Parse valid config for LightGBM backend."""
        config_json = dump_json_str(
            {
                "backend": "lightgbm",
                "dataset": "taiwan",
                "learning_rate": 0.1,
                "max_depth": 6,
                "n_estimators": 100,
                "num_leaves": 31,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }
        )
        result = _parse_external_train_config(config_json)
        if result["backend"] != "lightgbm":
            raise AssertionError("Expected lightgbm backend")
        assert result["dataset"] == "taiwan"
        assert result["config"]["learning_rate"] == 0.1
        assert result["config"]["num_leaves"] == 31
        assert result["config"]["min_child_samples"] == 20
        assert result["config"]["early_stopping_rounds"] == 10
        assert result["config"]["reg_alpha"] == 0.0
        assert result["config"]["reg_lambda"] == 1.0

    def test_with_regularization(self) -> None:
        """Parse LightGBM config with custom regularization."""
        config_json = dump_json_str(
            {
                "backend": "lightgbm",
                "dataset": "us",
                "learning_rate": 0.05,
                "max_depth": 8,
                "n_estimators": 200,
                "num_leaves": 63,
                "min_child_samples": 10,
                "subsample": 0.9,
                "colsample_bytree": 0.7,
                "random_state": 123,
                "device": "cuda",
                "early_stopping_rounds": 20,
                "reg_alpha": 1.0,
                "reg_lambda": 5.0,
            }
        )
        result = _parse_external_train_config(config_json)
        if result["backend"] != "lightgbm":
            raise AssertionError("Expected lightgbm backend")
        assert result["config"]["device"] == "cuda"
        assert result["config"]["early_stopping_rounds"] == 20
        assert result["config"]["reg_alpha"] == 1.0
        assert result["config"]["reg_lambda"] == 5.0


class TestClearGBMConfig:
    """Tests for parsing ClearGBM configuration."""

    def test_valid(self) -> None:
        """Parse valid config for ClearGBM backend."""
        config_json = dump_json_str(
            {
                "backend": "cleargbm",
                "dataset": "taiwan",
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "max_features": 0.8,
                "subsample": 0.8,
                "random_state": 42,
            }
        )
        result = _parse_external_train_config(config_json)
        if result["backend"] != "cleargbm":
            raise AssertionError("Expected cleargbm backend")
        assert result["dataset"] == "taiwan"
        assert result["config"]["n_estimators"] == 100
        assert result["config"]["max_depth"] == 6
        assert result["config"]["learning_rate"] == 0.1
        assert result["config"]["min_samples_split"] == 10
        assert result["config"]["min_samples_leaf"] == 5
        assert result["config"]["max_features"] == 0.8
        assert result["config"]["subsample"] == 0.8
        assert result["config"]["max_bins"] == 64
        assert result["config"]["reg_alpha"] == 0.0
        assert result["config"]["reg_lambda"] == 1.0
        assert result["config"]["n_jobs"] == -1
        assert result["config"]["early_stopping_rounds"] == 10
        assert result["config"]["growth_strategy"] == "depth_wise"
        assert result["config"]["num_leaves"] is None

    def test_with_monotonic_constraints(self) -> None:
        """Parse ClearGBM config with monotonic constraints."""
        config_json = dump_json_str(
            {
                "backend": "cleargbm",
                "dataset": "us",
                "n_estimators": 50,
                "max_depth": 4,
                "learning_rate": 0.05,
                "min_samples_split": 5,
                "min_samples_leaf": 3,
                "max_features": 10,
                "subsample": 0.9,
                "random_state": 7,
                "monotonic_constraints": {"feature_a": 1, "feature_b": -1},
            }
        )
        result = _parse_external_train_config(config_json)
        if result["backend"] != "cleargbm":
            raise AssertionError("Expected cleargbm backend")
        assert result["config"]["monotonic_constraints"] == {
            "feature_a": 1,
            "feature_b": -1,
        }
        assert result["config"]["max_features"] == 10

    def test_with_null_max_features(self) -> None:
        """Parse ClearGBM config with null max_features."""
        config_json = dump_json_str(
            {
                "backend": "cleargbm",
                "dataset": "polish",
                "n_estimators": 50,
                "max_depth": 4,
                "learning_rate": 0.05,
                "min_samples_split": 5,
                "min_samples_leaf": 3,
                "max_features": None,
                "subsample": 0.9,
                "random_state": 7,
            }
        )
        result = _parse_external_train_config(config_json)
        if result["backend"] != "cleargbm":
            raise AssertionError("Expected cleargbm backend")
        assert result["config"]["max_features"] is None

    def test_with_null_monotonic_constraints(self) -> None:
        """Parse ClearGBM config with null monotonic_constraints."""
        config_json = dump_json_str(
            {
                "backend": "cleargbm",
                "dataset": "taiwan",
                "n_estimators": 50,
                "max_depth": 4,
                "learning_rate": 0.05,
                "min_samples_split": 5,
                "min_samples_leaf": 3,
                "max_features": 0.5,
                "subsample": 0.9,
                "random_state": 7,
                "monotonic_constraints": None,
            }
        )
        result = _parse_external_train_config(config_json)
        if result["backend"] != "cleargbm":
            raise AssertionError("Expected cleargbm backend")
        assert result["config"]["monotonic_constraints"] is None

    def test_with_custom_optional_fields(self) -> None:
        """Parse ClearGBM config with custom optional values."""
        config_json = dump_json_str(
            {
                "backend": "cleargbm",
                "dataset": "taiwan",
                "n_estimators": 200,
                "max_depth": 8,
                "learning_rate": 0.01,
                "min_samples_split": 20,
                "min_samples_leaf": 10,
                "max_features": 0.5,
                "max_bins": 128,
                "subsample": 1.0,
                "random_state": 99,
                "reg_alpha": 0.5,
                "reg_lambda": 2.0,
                "n_jobs": 4,
                "early_stopping_rounds": 20,
            }
        )
        result = _parse_external_train_config(config_json)
        if result["backend"] != "cleargbm":
            raise AssertionError("Expected cleargbm backend")
        assert result["config"]["max_bins"] == 128
        assert result["config"]["reg_alpha"] == 0.5
        assert result["config"]["reg_lambda"] == 2.0
        assert result["config"]["n_jobs"] == 4
        assert result["config"]["early_stopping_rounds"] == 20

    def test_leaf_wise_with_budget(self) -> None:
        """leaf_wise growth with a num_leaves budget parses through."""
        config_json = dump_json_str(
            {
                "backend": "cleargbm",
                "dataset": "taiwan",
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "max_features": None,
                "subsample": 0.8,
                "random_state": 42,
                "growth_strategy": "leaf_wise",
                "num_leaves": 31,
            }
        )
        result = _parse_external_train_config(config_json)
        if result["backend"] != "cleargbm":
            raise AssertionError("Expected cleargbm backend")
        assert result["config"]["growth_strategy"] == "leaf_wise"
        assert result["config"]["num_leaves"] == 31

    def test_explicit_depth_wise(self) -> None:
        """An explicit depth_wise growth_strategy parses through unchanged."""
        config_json = dump_json_str(
            {
                "backend": "cleargbm",
                "dataset": "taiwan",
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "max_features": None,
                "subsample": 0.8,
                "random_state": 42,
                "growth_strategy": "depth_wise",
            }
        )
        result = _parse_external_train_config(config_json)
        if result["backend"] != "cleargbm":
            raise AssertionError("Expected cleargbm backend")
        assert result["config"]["growth_strategy"] == "depth_wise"
        assert result["config"]["num_leaves"] is None

    def test_leaf_wise_without_budget_raises(self) -> None:
        """leaf_wise growth without num_leaves is a config error."""
        config_json = dump_json_str(
            {
                "backend": "cleargbm",
                "dataset": "taiwan",
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "max_features": None,
                "subsample": 0.8,
                "random_state": 42,
                "growth_strategy": "leaf_wise",
            }
        )
        with pytest.raises(JSONTypeError, match="leaf_wise growth requires num_leaves"):
            _parse_external_train_config(config_json)

    def test_depth_wise_with_budget_raises(self) -> None:
        """depth_wise growth with a num_leaves budget is a config error."""
        config_json = dump_json_str(
            {
                "backend": "cleargbm",
                "dataset": "taiwan",
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "max_features": None,
                "subsample": 0.8,
                "random_state": 42,
                "num_leaves": 31,
            }
        )
        with pytest.raises(JSONTypeError, match="depth_wise growth takes no num_leaves budget"):
            _parse_external_train_config(config_json)

    def test_unknown_growth_strategy_raises(self) -> None:
        """An unrecognized growth_strategy value is a config error."""
        config_json = dump_json_str(
            {
                "backend": "cleargbm",
                "dataset": "taiwan",
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "max_features": None,
                "subsample": 0.8,
                "random_state": 42,
                "growth_strategy": "best_first",
            }
        )
        with pytest.raises(JSONTypeError, match="growth_strategy must be one of"):
            _parse_external_train_config(config_json)

    def test_non_integer_num_leaves_raises(self) -> None:
        """A non-integer num_leaves is a config error."""
        config_json = dump_json_str(
            {
                "backend": "cleargbm",
                "dataset": "taiwan",
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "max_features": None,
                "subsample": 0.8,
                "random_state": 42,
                "growth_strategy": "leaf_wise",
                "num_leaves": True,
            }
        )
        with pytest.raises(JSONTypeError, match="num_leaves must be an integer or null"):
            _parse_external_train_config(config_json)

    def test_stray_track_contributions_is_tolerated(self) -> None:
        """The removed track_contributions field is ignored, not required.

        The knob was removed 2026-08-22 (contribution extraction is a
        post-hoc explainer capability, never a training one); older clients
        still sending it must not break, and the parsed config must not
        carry it.
        """
        config_json = dump_json_str(
            {
                "backend": "cleargbm",
                "dataset": "taiwan",
                "n_estimators": 50,
                "max_depth": 4,
                "learning_rate": 0.05,
                "min_samples_split": 5,
                "min_samples_leaf": 3,
                "max_features": 0.5,
                "subsample": 0.9,
                "random_state": 7,
            }
        )
        result = _parse_external_train_config(config_json)
        if result["backend"] != "cleargbm":
            raise AssertionError("Expected cleargbm backend")
        assert "track_contributions" not in result["config"]

    def test_with_colsample_bytree_fraction(self) -> None:
        """Parse ClearGBM config with a per-tree feature fraction."""
        config_json = dump_json_str(
            {
                "backend": "cleargbm",
                "dataset": "taiwan",
                "n_estimators": 50,
                "max_depth": 4,
                "learning_rate": 0.05,
                "min_samples_split": 5,
                "min_samples_leaf": 3,
                "max_features": None,
                "colsample_bytree": 0.5,
                "subsample": 0.9,
                "random_state": 7,
            }
        )
        result = _parse_external_train_config(config_json)
        if result["backend"] != "cleargbm":
            raise AssertionError("Expected cleargbm backend")
        assert result["config"]["colsample_bytree"] == 0.5

    def test_invalid_colsample_bytree_type_raises(self) -> None:
        """Invalid colsample_bytree type raises JSONTypeError."""
        config_json = dump_json_str(
            {
                "backend": "cleargbm",
                "dataset": "taiwan",
                "n_estimators": 50,
                "max_depth": 4,
                "learning_rate": 0.05,
                "min_samples_split": 5,
                "min_samples_leaf": 3,
                "max_features": None,
                "colsample_bytree": "half",
                "subsample": 0.9,
                "random_state": 7,
            }
        )
        with pytest.raises(JSONTypeError, match="colsample_bytree must be"):
            _parse_external_train_config(config_json)

    def test_with_categorical_features_names(self) -> None:
        """Parse ClearGBM config with categorical column names."""
        config_json = dump_json_str(
            {
                "backend": "cleargbm",
                "dataset": "taiwan",
                "n_estimators": 50,
                "max_depth": 4,
                "learning_rate": 0.05,
                "min_samples_split": 5,
                "min_samples_leaf": 3,
                "max_features": None,
                "categorical_features": ["industry", "region"],
                "subsample": 0.9,
                "random_state": 7,
            }
        )
        result = _parse_external_train_config(config_json)
        if result["backend"] != "cleargbm":
            raise AssertionError("Expected cleargbm backend")
        assert result["config"]["categorical_features"] == ["industry", "region"]

    def test_invalid_categorical_features_type_raises(self) -> None:
        """Non-list and non-string entries raise JSONTypeError."""
        base = {
            "backend": "cleargbm",
            "dataset": "taiwan",
            "n_estimators": 50,
            "max_depth": 4,
            "learning_rate": 0.05,
            "min_samples_split": 5,
            "min_samples_leaf": 3,
            "max_features": None,
            "subsample": 0.9,
            "random_state": 7,
        }
        with pytest.raises(JSONTypeError, match="categorical_features must be"):
            _parse_external_train_config(
                dump_json_str({**base, "categorical_features": "industry"})
            )
        with pytest.raises(JSONTypeError, match="entries must be strings"):
            _parse_external_train_config(dump_json_str({**base, "categorical_features": [3]}))

    def test_invalid_max_features_type_raises(self) -> None:
        """Invalid max_features type raises JSONTypeError."""
        config_json = dump_json_str(
            {
                "backend": "cleargbm",
                "dataset": "taiwan",
                "n_estimators": 50,
                "max_depth": 4,
                "learning_rate": 0.05,
                "min_samples_split": 5,
                "min_samples_leaf": 3,
                "max_features": "invalid",
                "subsample": 0.9,
                "random_state": 7,
            }
        )
        with pytest.raises(JSONTypeError, match="max_features must be"):
            _parse_external_train_config(config_json)

    def test_invalid_monotonic_constraints_type_raises(self) -> None:
        """Invalid monotonic_constraints type raises JSONTypeError."""
        config_json = dump_json_str(
            {
                "backend": "cleargbm",
                "dataset": "taiwan",
                "n_estimators": 50,
                "max_depth": 4,
                "learning_rate": 0.05,
                "min_samples_split": 5,
                "min_samples_leaf": 3,
                "max_features": 0.5,
                "subsample": 0.9,
                "random_state": 7,
                "monotonic_constraints": "invalid",
            }
        )
        with pytest.raises(JSONTypeError, match="monotonic_constraints must be"):
            _parse_external_train_config(config_json)

    def test_monotonic_constraints_non_int_values_raises(self) -> None:
        """Monotonic constraints with non-int values raises JSONTypeError."""
        config_json = dump_json_str(
            {
                "backend": "cleargbm",
                "dataset": "taiwan",
                "n_estimators": 50,
                "max_depth": 4,
                "learning_rate": 0.05,
                "min_samples_split": 5,
                "min_samples_leaf": 3,
                "max_features": 0.5,
                "subsample": 0.9,
                "random_state": 7,
                "monotonic_constraints": {"feature_a": "not_int"},
            }
        )
        with pytest.raises(JSONTypeError, match="monotonic_constraints values must be ints"):
            _parse_external_train_config(config_json)
