"""Test hooks for regression worker components.

Regression-specific hooks for dataset loading, regressor registry,
regressor objective factory, and regression explainer registry.
Separated from _test_hooks.py (classifier) for clear separation of concerns.

Production code uses real implementations; tests can override these
module-level symbols to inject fakes without conditionals in core logic.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from covenant_ml.backends.regressor_protocol import RegressorProgressCallback
from covenant_ml.backends.regressor_registry import (
    RegressorBackendFactory,
    RegressorBackendRegistration,
    RegressorRegistry,
    default_regressor_registry,
)
from covenant_ml.datasets import (
    RegressionDatasetConfig,
    RegressionDatasetRegistry,
    RegressionLoadedDataset,
    create_regression_csv_loader,
    make_default_regression_registry,
)
from covenant_ml.datasets.protocol import ProgressCallbackProtocol
from covenant_ml.explainers.regression_registry import (
    RegressionExplainerRegistry,
    default_regression_explainer_registry,
)
from covenant_ml.features import FeaturePreset
from covenant_ml.types import RegressorBackendName
from numpy.typing import NDArray

from covenant_radar_api.worker._test_hooks import ObjectiveWithFeatureCount
from covenant_radar_api.worker.optimize_regression_types import (
    UnifiedRegressionOptimizeParseResult,
)

# =============================================================================
# Regression Dataset Registry Hook
# =============================================================================


class RegressionRegistryFactoryProtocol(Protocol):
    """Protocol for regression dataset registry factory function."""

    def __call__(self) -> RegressionDatasetRegistry:
        """Create a RegressionDatasetRegistry with dataset configurations.

        Returns:
            RegressionDatasetRegistry instance.
        """
        ...


def _real_regression_registry() -> RegressionDatasetRegistry:
    """Real implementation returning production regression dataset registry.

    Returns:
        RegressionDatasetRegistry with all verified regression dataset configurations.
    """
    return make_default_regression_registry()


regression_registry_factory: RegressionRegistryFactoryProtocol = _real_regression_registry


# =============================================================================
# Regression Dataset Loader Hook
# =============================================================================


class RegressionDatasetLoaderCallable(Protocol):
    """Protocol for callable regression dataset loader function.

    Defines the signature of a callable function for loading regression datasets.
    Supports optional progress callback for granular loading progress.
    """

    def __call__(
        self,
        config: RegressionDatasetConfig,
        external_dir: Path,
        progress_callback: ProgressCallbackProtocol | None = None,
    ) -> RegressionLoadedDataset:
        """Load a regression dataset from disk.

        Args:
            config: Regression dataset configuration from registry.
            external_dir: Root directory containing dataset folders.
            progress_callback: Optional callback for loading progress updates.

        Returns:
            RegressionLoadedDataset with features, continuous targets, and metadata.

        Raises:
            FileNotFoundError: If dataset file doesn't exist.
            ValueError: If data doesn't match expected format.
        """
        ...


def _real_regression_dataset_loader(
    config: RegressionDatasetConfig,
    external_dir: Path,
    progress_callback: ProgressCallbackProtocol | None = None,
) -> RegressionLoadedDataset:
    """Real implementation using covenant_ml.datasets regression loader.

    Args:
        config: Regression dataset configuration from registry.
        external_dir: Root directory containing dataset folders.
        progress_callback: Optional callback for loading progress updates.

    Returns:
        RegressionLoadedDataset with features, continuous targets, and metadata.

    Raises:
        FileNotFoundError: If dataset file doesn't exist.
        ValueError: If data doesn't match expected format.
    """
    loader = create_regression_csv_loader()
    return loader.load(config, external_dir, progress_callback)


regression_dataset_loader: RegressionDatasetLoaderCallable = _real_regression_dataset_loader


# =============================================================================
# Regressor Registry Hook
# =============================================================================


class RegressorRegistryFactory(Protocol):
    """Protocol for regressor backend registry factory."""

    def __call__(self) -> RegressorRegistry:
        """Create a RegressorRegistry with regressor backend implementations.

        Returns:
            RegressorRegistry instance.
        """
        ...


def _real_regressor_registry() -> RegressorRegistry:
    """Build registry with all regressor backends including PyTorch (covenant_nn).

    Returns:
        RegressorRegistry with tree-based backends from covenant_ml
        and neural backends (MLP regressor, LSTM regressor) from covenant_nn.
    """
    reg = default_regressor_registry()
    nn_mod = __import__(
        "covenant_nn",
        fromlist=["create_mlp_regressor_backend", "create_lstm_regressor_backend"],
    )
    create_mlp_reg: RegressorBackendFactory = nn_mod.create_mlp_regressor_backend
    create_lstm_reg: RegressorBackendFactory = nn_mod.create_lstm_regressor_backend
    reg.register("mlp_reg", RegressorBackendRegistration(create_mlp_reg))
    reg.register("lstm_reg", RegressorBackendRegistration(create_lstm_reg))
    return reg


regressor_registry_factory: RegressorRegistryFactory = _real_regressor_registry


# =============================================================================
# Regressor Objective Factory Hook
# =============================================================================


class RegressorObjectiveFactoryProtocol(Protocol):
    """Protocol for regression objective factory function.

    Creates per-backend regression objective functions.
    The returned objective must have an n_features property and
    conform to ObjectiveProtocol (int64 y in __call__, ignored).
    """

    def __call__(
        self,
        backend_name: RegressorBackendName,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        feature_names: list[str],
        config: UnifiedRegressionOptimizeParseResult,
    ) -> ObjectiveWithFeatureCount:
        """Create an objective function for the specified regressor backend.

        Args:
            backend_name: Regressor backend to create objective for.
            x: Feature matrix.
            y: Continuous target values (float64).
            feature_names: Feature column names.
            config: Parsed regression optimization config.

        Returns:
            Objective callable with n_features property.
        """
        ...


def _real_regressor_objective_factory(
    backend_name: RegressorBackendName,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    feature_names: list[str],
    config: UnifiedRegressionOptimizeParseResult,
) -> ObjectiveWithFeatureCount:
    """Create per-backend regressor objective using dynamic imports.

    Dispatches to the appropriate create_*_regressor_objective factory.

    Args:
        backend_name: Regressor backend to create objective for.
        x: Feature matrix.
        y: Continuous target values (float64).
        feature_names: Feature column names.
        config: Parsed regression optimization config.

    Returns:
        Objective callable with n_features property.

    Raises:
        ValueError: If backend_name is not recognized.
    """
    if backend_name == "xgboost_reg":
        from covenant_ml.optimizer.objectives.xgboost_regressor_objective import (
            create_xgboost_regressor_objective,
        )

        return create_xgboost_regressor_objective(
            x,
            y,
            feature_names,
            config["device"],
            config["feature_preset"],
        )

    if backend_name == "lightgbm_reg":
        from covenant_ml.optimizer.objectives.lightgbm_regressor_objective import (
            create_lightgbm_regressor_objective,
        )

        return create_lightgbm_regressor_objective(
            x,
            y,
            feature_names,
            config["device"],
            config["feature_preset"],
            early_stopping_rounds=config["early_stopping_rounds"],
            n_jobs=config["n_jobs"],
        )

    if backend_name == "mlp_reg":
        nn_mod = __import__("covenant_nn", fromlist=["create_mlp_regressor_objective"])
        create_mlp: _CreateMLPRegressorObjectiveProto = nn_mod.create_mlp_regressor_objective
        return create_mlp(
            x,
            y,
            feature_names,
            config["device"],
            config["precision"],
            config["feature_preset"],
            config["n_epochs"],
            config["early_stopping_patience"],
            optimizer_name=config["nn_optimizer"],
        )

    # backend_name == "lstm_reg"
    nn_mod = __import__("covenant_nn", fromlist=["create_lstm_regressor_objective"])
    create_lstm: _CreateLSTMRegressorObjectiveProto = nn_mod.create_lstm_regressor_objective
    return create_lstm(
        x,
        y,
        feature_names,
        config["device"],
        config["precision"],
        config["feature_preset"],
        config["n_epochs"],
        config["early_stopping_patience"],
        config["sequence_length"],
        bidirectional=config["bidirectional"],
    )


class _CreateMLPRegressorObjectiveProto(Protocol):
    """Protocol for covenant_nn.create_mlp_regressor_objective."""

    def __call__(
        self,
        x_features: NDArray[np.float64],
        y_targets: NDArray[np.float64],
        feature_names: list[str],
        device: Literal["cpu", "cuda", "auto"],
        precision: Literal["fp32", "fp16", "bf16", "auto"],
        feature_preset: FeaturePreset,
        n_epochs: int,
        early_stopping_patience: int,
        optimizer_name: Literal["adamw", "adam", "sgd"] = ...,
        epoch_callback: RegressorProgressCallback | None = ...,
    ) -> ObjectiveWithFeatureCount: ...


class _CreateLSTMRegressorObjectiveProto(Protocol):
    """Protocol for covenant_nn.create_lstm_regressor_objective."""

    def __call__(
        self,
        x_features: NDArray[np.float64],
        y_targets: NDArray[np.float64],
        feature_names: list[str],
        device: Literal["cpu", "cuda", "auto"],
        precision: Literal["fp32", "fp16", "bf16", "auto"],
        feature_preset: FeaturePreset,
        n_epochs: int,
        early_stopping_patience: int,
        sequence_length: int,
        bidirectional: bool = ...,
        epoch_callback: RegressorProgressCallback | None = ...,
    ) -> ObjectiveWithFeatureCount: ...


regressor_objective_factory: RegressorObjectiveFactoryProtocol = _real_regressor_objective_factory


# =============================================================================
# Regression Explainer Registry Hook
# =============================================================================


class RegressionExplainerRegistryFactory(Protocol):
    """Protocol for regression explainer registry factory."""

    def __call__(self) -> RegressionExplainerRegistry:
        """Create a RegressionExplainerRegistry with explainer registrations.

        Returns:
            RegressionExplainerRegistry instance.
        """
        ...


def _real_regression_explainer_registry() -> RegressionExplainerRegistry:
    """Build regression explainer registry with all adapters.

    Returns:
        RegressionExplainerRegistry with permutation, gradient,
        integrated_gradients, and shap_tree adapters registered.
    """
    return default_regression_explainer_registry()


regression_explainer_registry_factory: RegressionExplainerRegistryFactory = (
    _real_regression_explainer_registry
)


__all__ = [
    "RegressionDatasetLoaderCallable",
    "RegressionExplainerRegistryFactory",
    "RegressionRegistryFactoryProtocol",
    "RegressorObjectiveFactoryProtocol",
    "RegressorRegistryFactory",
    "regression_dataset_loader",
    "regression_explainer_registry_factory",
    "regression_registry_factory",
    "regressor_objective_factory",
    "regressor_registry_factory",
]
