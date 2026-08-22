"""Tests for worker _regression_hooks module.

Tests the real regression hook implementations: regression dataset registry,
regression dataset loader, regressor registry, and regressor objective factory.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from covenant_ml.types_regression import RegressorBackendName
from numpy.typing import NDArray

from covenant_radar_api.worker._regression_hooks import (
    _real_regression_dataset_loader,
    _real_regression_explainer_registry,
    _real_regression_registry,
    _real_regressor_objective_factory,
    _real_regressor_registry,
    regression_dataset_loader,
    regression_explainer_registry_factory,
    regression_registry_factory,
    regressor_objective_factory,
    regressor_registry_factory,
)
from covenant_radar_api.worker.optimize_regression_types import (
    UnifiedRegressionOptimizeParseResult,
)

# =============================================================================
# Tests: Regression Dataset Registry Hook
# =============================================================================


class TestRegressionRegistryHook:
    """Tests for regression_registry_factory hook."""

    def test_hook_defaults_to_real(self) -> None:
        """regression_registry_factory defaults to _real_regression_registry."""
        assert regression_registry_factory is _real_regression_registry

    def test_real_registry_returns_registry_with_financial_distress(self) -> None:
        """_real_regression_registry returns registry containing financial_distress."""
        registry = _real_regression_registry()
        names = registry.list_names()
        assert "financial_distress" in names


# =============================================================================
# Tests: Regression Dataset Loader Hook
# =============================================================================


class TestRegressionDatasetLoaderHook:
    """Tests for regression_dataset_loader hook."""

    def test_hook_defaults_to_real(self) -> None:
        """regression_dataset_loader defaults to _real_regression_dataset_loader."""
        assert regression_dataset_loader is _real_regression_dataset_loader

    def test_real_loader_file_not_found(self, tmp_path: Path) -> None:
        """_real_regression_dataset_loader raises FileNotFoundError for missing file."""
        from covenant_ml.datasets.types import RegressionDatasetConfig, RegressionTargetSpec

        config = RegressionDatasetConfig(
            name="test_regression",
            display_name="Test Regression",
            folder="nonexistent_folder",
            file_name="nonexistent.csv",
            file_format="csv",
            encoding="utf-8",
            target=RegressionTargetSpec(column_name="target"),
            exclude_columns=(),
            n_samples_expected=100,
            n_features_expected=5,
            target_mean_expected=0.0,
        )
        with pytest.raises(FileNotFoundError):
            _real_regression_dataset_loader(config, tmp_path)


# =============================================================================
# Tests: Regressor Registry Hook
# =============================================================================


class TestRegressorRegistryHook:
    """Tests for regressor_registry_factory hook."""

    def test_hook_defaults_to_real(self) -> None:
        """regressor_registry_factory defaults to _real_regressor_registry."""
        assert regressor_registry_factory is _real_regressor_registry

    def test_real_registry_has_xgboost_reg(self) -> None:
        """_real_regressor_registry has xgboost_reg backend."""
        registry = _real_regressor_registry()
        names = registry.list_backends()
        assert "xgboost_reg" in names

    def test_real_registry_has_lightgbm_reg(self) -> None:
        """_real_regressor_registry has lightgbm_reg backend."""
        registry = _real_regressor_registry()
        names = registry.list_backends()
        assert "lightgbm_reg" in names

    def test_real_registry_has_mlp_reg(self) -> None:
        """_real_regressor_registry has mlp_reg backend."""
        registry = _real_regressor_registry()
        names = registry.list_backends()
        assert "mlp_reg" in names

    def test_real_registry_has_lstm_reg(self) -> None:
        """_real_regressor_registry has lstm_reg backend."""
        registry = _real_regressor_registry()
        names = registry.list_backends()
        assert "lstm_reg" in names


# =============================================================================
# Tests: Regressor Objective Factory Hook
# =============================================================================


class TestRegressorObjectiveFactoryHook:
    """Tests for regressor_objective_factory hook and _real_regressor_objective_factory."""

    def _make_config(
        self,
        backend: RegressorBackendName = "xgboost_reg",
    ) -> UnifiedRegressionOptimizeParseResult:
        """Create minimal config for regression objective factory tests.

        Args:
            backend: Regressor backend name.

        Returns:
            UnifiedRegressionOptimizeParseResult with default values.
        """
        return UnifiedRegressionOptimizeParseResult(
            backend=backend,
            dataset="us_bankruptcy",
            n_trials=1,
            timeout_seconds=None,
            device="cpu",
            feature_preset="none",
            random_state=42,
            early_stopping_rounds=2,
            n_jobs=1,
            precision="fp32",
            nn_optimizer="adamw",
            n_epochs=5,
            early_stopping_patience=2,
            sequence_length=3,
            bidirectional=False,
        )

    def _make_regression_data(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
        """Create small test data for regression objective factory.

        Returns:
            Tuple of (features, targets, feature_names).
        """
        rng = np.random.RandomState(42)
        x: NDArray[np.float64] = rng.randn(50, 5).astype(np.float64)
        y: NDArray[np.float64] = rng.randn(50).astype(np.float64)
        names = [f"f{i}" for i in range(5)]
        return x, y, names

    def test_hook_defaults_to_real(self) -> None:
        """regressor_objective_factory defaults to _real_regressor_objective_factory."""
        assert regressor_objective_factory is _real_regressor_objective_factory

    def test_xgboost_reg_objective(self) -> None:
        """_real_regressor_objective_factory creates XGBoost regressor objective."""
        x, y, names = self._make_regression_data()
        config = self._make_config()
        obj = _real_regressor_objective_factory("xgboost_reg", x, y, names, config)
        assert obj.n_features == 5

    def test_lightgbm_reg_objective(self) -> None:
        """_real_regressor_objective_factory creates LightGBM regressor objective."""
        x, y, names = self._make_regression_data()
        config = self._make_config()
        obj = _real_regressor_objective_factory("lightgbm_reg", x, y, names, config)
        assert obj.n_features == 5

    def test_mlp_reg_objective(self) -> None:
        """_real_regressor_objective_factory creates MLP regressor objective."""
        x, y, names = self._make_regression_data()
        config = self._make_config(backend="mlp_reg")
        obj = _real_regressor_objective_factory("mlp_reg", x, y, names, config)
        assert obj.n_features == 5

    def test_lstm_reg_objective(self) -> None:
        """_real_regressor_objective_factory creates LSTM regressor objective."""
        x, y, names = self._make_regression_data()
        config = self._make_config(backend="lstm_reg")
        objective = _real_regressor_objective_factory("lstm_reg", x, y, names, config)
        assert objective.n_features == 5


# =============================================================================
# Tests: Regression Explainer Registry Hook
# =============================================================================


class TestRegressionExplainerRegistryHook:
    """Tests for regression_explainer_registry_factory hook."""

    def test_hook_defaults_to_real(self) -> None:
        """regression_explainer_registry_factory defaults to real."""
        assert regression_explainer_registry_factory is _real_regression_explainer_registry

    def test_real_registry_has_all_explainers(self) -> None:
        """Real registry contains all 4 explainer types."""
        registry = _real_regression_explainer_registry()
        explainers = registry.list_explainers()
        assert "permutation" in explainers
        assert "gradient" in explainers
        assert "integrated_gradients" in explainers
        assert "shap_tree" in explainers

    def test_xgboost_reg_compatible(self) -> None:
        """xgboost_reg is compatible with permutation and shap_tree."""
        registry = _real_regression_explainer_registry()
        compatible = registry.list_compatible_explainers("xgboost_reg")
        assert "permutation" in compatible
        assert "shap_tree" in compatible
        assert "gradient" not in compatible

    def test_mlp_reg_compatible(self) -> None:
        """mlp_reg is compatible with gradient, IG, and permutation."""
        registry = _real_regression_explainer_registry()
        compatible = registry.list_compatible_explainers("mlp_reg")
        assert "gradient" in compatible
        assert "integrated_gradients" in compatible
        assert "permutation" in compatible
        assert "shap_tree" not in compatible
