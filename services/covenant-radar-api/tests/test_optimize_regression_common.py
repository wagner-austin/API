"""Tests for worker/_optimize_regression_common.py regression dataset loading and parsing.

Tests use dependency injection via worker/_regression_hooks to verify actual code paths.
All code paths are tested with strong assertions on actual behavior.

Strict typing only: no Any, no casts, no type: ignore, no stubs, no mocks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from covenant_ml.datasets import (
    RegressionDatasetConfig,
    RegressionDatasetRegistry,
    RegressionLoadedDataset,
)
from covenant_ml.datasets.protocol import ProgressCallbackProtocol
from covenant_ml.datasets.types import RegressionDatasetMeta, RegressionTargetSpec
from numpy.typing import NDArray
from platform_core.json_utils import JSONTypeError

from covenant_radar_api.worker import _regression_hooks as hooks
from covenant_radar_api.worker._optimize_regression_common import (
    load_regression_dataset,
    parse_regression_dataset_name,
    parse_regressor_backend_name,
)

# =============================================================================
# Fake Implementations for Testing
# =============================================================================


def _make_fake_regression_dataset(name: str = "financial_distress") -> RegressionLoadedDataset:
    """Create fake regression dataset for testing.

    Args:
        name: Dataset name.

    Returns:
        RegressionLoadedDataset with synthetic data.
    """
    rng = np.random.default_rng(42)
    x: NDArray[np.float64] = rng.random((80, 6)).astype(np.float64)
    y: NDArray[np.float64] = rng.random(80).astype(np.float64)
    meta: RegressionDatasetMeta = {
        "name": name,
        "n_samples": 80,
        "n_features": 6,
        "feature_names": tuple(f"feature_{i}" for i in range(6)),
        "target_mean": 0.5,
        "target_std": 0.3,
        "target_min": 0.0,
        "target_max": 1.0,
        "categorical_encodings": (),
    }
    return {"meta": meta, "x": x, "y": y}


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


def _make_fake_regression_loader(
    config: RegressionDatasetConfig,
    external_dir: Path,
    progress_callback: ProgressCallbackProtocol | None = None,
) -> RegressionLoadedDataset:
    """Fake regression dataset loader that returns synthetic data.

    Args:
        config: Dataset config (used for name).
        external_dir: Ignored.
        progress_callback: Ignored.

    Returns:
        Fake RegressionLoadedDataset.
    """
    return _make_fake_regression_dataset(config["name"])


# =============================================================================
# Tests: parse_regressor_backend_name
# =============================================================================


class TestParseRegressorBackendName:
    """Tests for parse_regressor_backend_name function."""

    def test_none_defaults_to_xgboost_reg(self) -> None:
        """None input returns 'xgboost_reg'."""
        assert parse_regressor_backend_name(None) == "xgboost_reg"

    def test_xgboost_reg(self) -> None:
        """'xgboost_reg' is accepted."""
        assert parse_regressor_backend_name("xgboost_reg") == "xgboost_reg"

    def test_lightgbm_reg(self) -> None:
        """'lightgbm_reg' is accepted."""
        assert parse_regressor_backend_name("lightgbm_reg") == "lightgbm_reg"

    def test_mlp_reg(self) -> None:
        """'mlp_reg' is accepted."""
        assert parse_regressor_backend_name("mlp_reg") == "mlp_reg"

    def test_lstm_reg(self) -> None:
        """'lstm_reg' is accepted."""
        assert parse_regressor_backend_name("lstm_reg") == "lstm_reg"

    def test_invalid_backend_raises_value_error(self) -> None:
        """Invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="backend must be one of"):
            parse_regressor_backend_name("xgboost")

    def test_non_string_raises_json_type_error(self) -> None:
        """Non-string input raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="backend must be a string"):
            parse_regressor_backend_name(42)


# =============================================================================
# Tests: parse_regression_dataset_name
# =============================================================================


class TestParseRegressionDatasetName:
    """Tests for parse_regression_dataset_name function."""

    def setup_method(self) -> None:
        """Install fake regression registry before each test."""
        self._orig_registry = hooks.regression_registry_factory
        hooks.regression_registry_factory = _make_fake_regression_registry

    def teardown_method(self) -> None:
        """Restore original hooks after each test."""
        hooks.regression_registry_factory = self._orig_registry

    def test_valid_dataset_name(self) -> None:
        """Valid dataset name returns the name."""
        result = parse_regression_dataset_name("financial_distress")
        assert result == "financial_distress"

    def test_invalid_dataset_raises_value_error(self) -> None:
        """Invalid dataset name raises ValueError."""
        with pytest.raises(ValueError, match="dataset must be one of"):
            parse_regression_dataset_name("nonexistent")


# =============================================================================
# Tests: load_regression_dataset
# =============================================================================


class TestLoadRegressionDataset:
    """Tests for load_regression_dataset function."""

    def setup_method(self) -> None:
        """Install fake hooks before each test."""
        self._orig_registry = hooks.regression_registry_factory
        self._orig_loader = hooks.regression_dataset_loader
        hooks.regression_registry_factory = _make_fake_regression_registry
        hooks.regression_dataset_loader = _make_fake_regression_loader

    def teardown_method(self) -> None:
        """Restore original hooks after each test."""
        hooks.regression_registry_factory = self._orig_registry
        hooks.regression_dataset_loader = self._orig_loader

    def test_loads_regression_dataset(self, tmp_path: Path) -> None:
        """Loading regression dataset returns correct structure."""
        dataset = load_regression_dataset("financial_distress", tmp_path)

        assert dataset["meta"]["name"] == "financial_distress"
        assert dataset["meta"]["n_samples"] == 80
        assert dataset["meta"]["n_features"] == 6
        assert dataset["x"].shape == (80, 6)
        assert dataset["y"].shape == (80,)
        assert dataset["y"].dtype == np.float64

    def test_loads_without_progress_callback(self, tmp_path: Path) -> None:
        """Loading without progress callback works."""
        dataset = load_regression_dataset("financial_distress", tmp_path, None)

        assert dataset["meta"]["n_samples"] == 80

    def test_loads_with_progress_callback(self, tmp_path: Path) -> None:
        """Loading with progress callback passes it through."""
        callback_calls: list[str] = []

        def _fake_loader_with_callback(
            config: RegressionDatasetConfig,
            external_dir: Path,
            progress_callback: ProgressCallbackProtocol | None = None,
        ) -> RegressionLoadedDataset:
            if progress_callback is not None:
                callback_calls.append("called")
            return _make_fake_regression_dataset(config["name"])

        hooks.regression_dataset_loader = _fake_loader_with_callback

        from covenant_ml.datasets.types import LoadProgress

        def _progress(progress: LoadProgress) -> None:
            pass

        progress_cb: ProgressCallbackProtocol = _progress

        dataset = load_regression_dataset("financial_distress", tmp_path, progress_cb)

        assert dataset["meta"]["n_samples"] == 80
        assert len(callback_calls) == 1
