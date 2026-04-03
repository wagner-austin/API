"""Tests for worker/_train_external_regression_parsers.py.

Tests use dependency injection via worker/_regression_hooks to verify actual
code paths. All code paths are tested with strong assertions on actual behavior.

Strict typing only: no Any, no casts, no type: ignore, no stubs, no mocks.
"""

from __future__ import annotations

import pytest
from covenant_ml.datasets import (
    RegressionDatasetConfig,
    RegressionDatasetRegistry,
)
from covenant_ml.datasets.types import RegressionTargetSpec
from platform_core.json_utils import JSONTypeError, dump_json_str

from covenant_radar_api.worker import _regression_hooks as hooks
from covenant_radar_api.worker._train_external_regression_parsers import (
    parse_external_regression_train_config,
)

# =============================================================================
# Fake Implementations for Testing
# =============================================================================


def _make_fake_regression_config(name: str) -> RegressionDatasetConfig:
    """Create fake regression dataset config.

    Args:
        name: Dataset name.

    Returns:
        RegressionDatasetConfig for testing.
    """
    return RegressionDatasetConfig(
        name=name,
        display_name=f"Fake {name}",
        folder=f"{name}_data",
        file_name="data.csv",
        file_format="csv",
        encoding="utf-8",
        target=RegressionTargetSpec(column_name="target"),
        exclude_columns=(),
        n_samples_expected=80,
        n_features_expected=6,
        target_mean_expected=0.5,
    )


def _make_fake_regression_registry() -> RegressionDatasetRegistry:
    """Create fake regression dataset registry with one dataset.

    Returns:
        RegressionDatasetRegistry with financial_distress.
    """
    configs = (_make_fake_regression_config("financial_distress"),)
    return RegressionDatasetRegistry(configs)


# =============================================================================
# Fixtures
# =============================================================================


class _HookGuard:
    """RAII guard for installing/restoring regression hooks."""

    def __init__(self) -> None:
        self._orig_registry = hooks.regression_registry_factory

    def install(self) -> None:
        """Install fake hooks."""
        hooks.regression_registry_factory = _make_fake_regression_registry

    def restore(self) -> None:
        """Restore original hooks."""
        hooks.regression_registry_factory = self._orig_registry


# =============================================================================
# Tests: XGBoost regressor config parsing
# =============================================================================


class TestParseXGBoostRegConfig:
    """Tests for parsing XGBoost regression config."""

    def setup_method(self) -> None:
        """Install fake regression registry."""
        self._guard = _HookGuard()
        self._guard.install()

    def teardown_method(self) -> None:
        """Restore original hooks."""
        self._guard.restore()

    def test_minimal_xgboost_reg_config(self) -> None:
        """Minimal config defaults to xgboost_reg backend."""
        config_json = dump_json_str(
            {
                "dataset": "financial_distress",
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }
        )
        result = parse_external_regression_train_config(config_json)

        assert result["backend"] == "xgboost_reg"
        assert result["dataset"] == "financial_distress"
        assert result["config"]["learning_rate"] == 0.1
        assert result["config"]["max_depth"] == 3
        assert result["config"]["n_estimators"] == 10

    def test_explicit_xgboost_reg_backend(self) -> None:
        """Explicit 'xgboost_reg' backend is accepted."""
        config_json = dump_json_str(
            {
                "dataset": "financial_distress",
                "backend": "xgboost_reg",
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }
        )
        result = parse_external_regression_train_config(config_json)
        assert result["backend"] == "xgboost_reg"

    def test_xgboost_reg_custom_split_ratios(self) -> None:
        """Custom split ratios are parsed."""
        config_json = dump_json_str(
            {
                "dataset": "financial_distress",
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "train_ratio": 0.8,
                "val_ratio": 0.1,
                "test_ratio": 0.1,
            }
        )
        result = parse_external_regression_train_config(config_json)
        assert result["config"]["train_ratio"] == 0.8
        assert result["config"]["val_ratio"] == 0.1
        assert result["config"]["test_ratio"] == 0.1

    def test_xgboost_reg_cuda_device(self) -> None:
        """CUDA device is accepted."""
        config_json = dump_json_str(
            {
                "dataset": "financial_distress",
                "device": "cuda",
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }
        )
        result = parse_external_regression_train_config(config_json)
        assert result["config"]["device"] == "cuda"


# =============================================================================
# Tests: LightGBM regressor config parsing
# =============================================================================


class TestParseLightGBMRegConfig:
    """Tests for parsing LightGBM regression config."""

    def setup_method(self) -> None:
        """Install fake regression registry."""
        self._guard = _HookGuard()
        self._guard.install()

    def teardown_method(self) -> None:
        """Restore original hooks."""
        self._guard.restore()

    def test_lightgbm_reg_config(self) -> None:
        """LightGBM regressor config is parsed correctly."""
        config_json = dump_json_str(
            {
                "dataset": "financial_distress",
                "backend": "lightgbm_reg",
                "device": "cpu",
                "learning_rate": 0.05,
                "max_depth": 5,
                "n_estimators": 100,
                "num_leaves": 31,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }
        )
        result = parse_external_regression_train_config(config_json)

        assert result["backend"] == "lightgbm_reg"
        assert result["dataset"] == "financial_distress"
        assert result["config"]["num_leaves"] == 31
        assert result["config"]["min_child_samples"] == 20
        assert result["config"]["learning_rate"] == 0.05

    def test_lightgbm_reg_with_regularization(self) -> None:
        """LightGBM regressor config with explicit regularization."""
        config_json = dump_json_str(
            {
                "dataset": "financial_distress",
                "backend": "lightgbm_reg",
                "device": "cpu",
                "learning_rate": 0.05,
                "max_depth": 5,
                "n_estimators": 100,
                "num_leaves": 31,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "reg_alpha": 0.1,
                "reg_lambda": 2.0,
            }
        )
        result = parse_external_regression_train_config(config_json)
        assert result["config"]["reg_alpha"] == 0.1
        assert result["config"]["reg_lambda"] == 2.0


# =============================================================================
# Tests: Error cases
# =============================================================================


class TestParseRegressionTrainErrors:
    """Tests for error handling in regression train config parsing."""

    def setup_method(self) -> None:
        """Install fake regression registry."""
        self._guard = _HookGuard()
        self._guard.install()

    def teardown_method(self) -> None:
        """Restore original hooks."""
        self._guard.restore()

    def test_invalid_dataset_raises_value_error(self) -> None:
        """Unknown dataset raises ValueError."""
        config_json = dump_json_str(
            {
                "dataset": "nonexistent",
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }
        )
        with pytest.raises(ValueError, match="dataset must be one of"):
            parse_external_regression_train_config(config_json)

    def test_invalid_backend_raises_value_error(self) -> None:
        """Invalid regressor backend raises ValueError."""
        config_json = dump_json_str(
            {
                "dataset": "financial_distress",
                "backend": "xgboost",
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }
        )
        with pytest.raises(ValueError, match="backend must be one of"):
            parse_external_regression_train_config(config_json)

    def test_mlp_reg_backend_raises_value_error(self) -> None:
        """mlp_reg backend raises ValueError (not supported for training)."""
        config_json = dump_json_str(
            {
                "dataset": "financial_distress",
                "backend": "mlp_reg",
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }
        )
        with pytest.raises(ValueError, match="backend must be one of"):
            parse_external_regression_train_config(config_json)

    def test_non_string_backend_raises_json_type_error(self) -> None:
        """Non-string backend raises JSONTypeError."""
        config_json = dump_json_str(
            {
                "dataset": "financial_distress",
                "backend": 42,
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }
        )
        with pytest.raises(JSONTypeError, match="backend must be a string"):
            parse_external_regression_train_config(config_json)

    def test_bad_split_ratios_raises_value_error(self) -> None:
        """Split ratios not summing to 1.0 raises ValueError."""
        config_json = dump_json_str(
            {
                "dataset": "financial_distress",
                "train_ratio": 0.5,
                "val_ratio": 0.1,
                "test_ratio": 0.1,
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }
        )
        with pytest.raises(ValueError, match=r"Split ratios must sum to 1\.0"):
            parse_external_regression_train_config(config_json)

    def test_non_object_json_raises_json_type_error(self) -> None:
        """Non-object JSON raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="config must be a JSON object"):
            parse_external_regression_train_config('"just a string"')

    def test_missing_dataset_raises_json_type_error(self) -> None:
        """Missing dataset field raises JSONTypeError."""
        config_json = dump_json_str(
            {
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }
        )
        with pytest.raises(JSONTypeError):
            parse_external_regression_train_config(config_json)

    def test_invalid_device_raises_value_error(self) -> None:
        """Invalid device raises ValueError."""
        config_json = dump_json_str(
            {
                "dataset": "financial_distress",
                "device": "tpu",
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }
        )
        with pytest.raises(ValueError, match="device must be one of"):
            parse_external_regression_train_config(config_json)


# =============================================================================
# Tests: Backend name parsing
# =============================================================================


class TestParseRegressionTrainBackend:
    """Tests for _parse_regression_train_backend."""

    def test_none_defaults_to_xgboost_reg(self) -> None:
        """None input defaults to xgboost_reg."""
        from covenant_radar_api.worker._train_external_regression_parsers import (
            _parse_regression_train_backend,
        )

        assert _parse_regression_train_backend(None) == "xgboost_reg"

    def test_xgboost_reg(self) -> None:
        """'xgboost_reg' is accepted."""
        from covenant_radar_api.worker._train_external_regression_parsers import (
            _parse_regression_train_backend,
        )

        assert _parse_regression_train_backend("xgboost_reg") == "xgboost_reg"

    def test_lightgbm_reg(self) -> None:
        """'lightgbm_reg' is accepted."""
        from covenant_radar_api.worker._train_external_regression_parsers import (
            _parse_regression_train_backend,
        )

        assert _parse_regression_train_backend("lightgbm_reg") == "lightgbm_reg"

    def test_invalid_raises_value_error(self) -> None:
        """Invalid backend raises ValueError."""
        from covenant_radar_api.worker._train_external_regression_parsers import (
            _parse_regression_train_backend,
        )

        with pytest.raises(ValueError, match="backend must be one of"):
            _parse_regression_train_backend("xgboost")

    def test_non_string_raises_json_type_error(self) -> None:
        """Non-string input raises JSONTypeError."""
        from covenant_radar_api.worker._train_external_regression_parsers import (
            _parse_regression_train_backend,
        )

        with pytest.raises(JSONTypeError, match="backend must be a string"):
            _parse_regression_train_backend(42)
