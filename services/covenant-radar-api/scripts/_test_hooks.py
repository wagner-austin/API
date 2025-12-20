"""Test hooks for CLI optimization scripts.

Production code uses real implementations; tests override these module-level
symbols to inject fakes without conditionals in core logic.

Each backend has its own runner hook with strongly-typed result types.
Import and set hooks at application/test startup.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from covenant_ml.backends.registry import (
    ClassifierRegistry,
)
from covenant_ml.backends.registry import (
    default_registry as default_backend_registry,
)
from covenant_ml.datasets import (
    DatasetConfig,
    DatasetRegistry,
    LoadedDataset,
    TimeSeriesDatasetConfig,
    TimeSeriesDatasetRegistry,
    create_dataset_loader,
    create_timeseries_csv_loader,
    make_default_registry,
    make_default_timeseries_registry,
)
from covenant_ml.explainers.registry import (
    ExplainerRegistry,
    default_explainer_registry,
)

from covenant_radar_api.worker.optimize_lightgbm_job import (
    LightGBMLoadingProgressCallbackProtocol,
    LightGBMLoadingProgressInfo,
    LightGBMOptimizationResult,
    LightGBMPhaseCallbackProtocol,
    LightGBMPhaseInfo,
    LightGBMTrialProgressCallbackProtocol,
    LightGBMTrialProgressInfo,
    run_lightgbm_optimization,
)
from covenant_radar_api.worker.optimize_lstm_job import (
    LSTMLoadingProgressCallbackProtocol,
    LSTMLoadingProgressInfo,
    LSTMOptimizationResult,
    LSTMPhaseCallbackProtocol,
    LSTMPhaseInfo,
    LSTMTrialProgressCallbackProtocol,
    LSTMTrialProgressInfo,
    run_lstm_optimization,
)
from covenant_radar_api.worker.optimize_mlp_job import (
    MLPLoadingProgressCallbackProtocol,
    MLPLoadingProgressInfo,
    MLPOptimizationResult,
    MLPPhaseCallbackProtocol,
    MLPPhaseInfo,
    MLPTrialProgressCallbackProtocol,
    MLPTrialProgressInfo,
    run_mlp_optimization,
)
from covenant_radar_api.worker.optimize_xgboost_job import (
    OptimizationResult as XGBoostOptimizationResult,
)
from covenant_radar_api.worker.optimize_xgboost_job import (
    TrialProgressCallbackProtocol as XGBoostProgressCallbackProtocol,
)
from covenant_radar_api.worker.optimize_xgboost_job import (
    TrialProgressInfo as XGBoostProgressInfo,
)
from covenant_radar_api.worker.optimize_xgboost_job import (
    XGBoostLoadingProgressCallbackProtocol,
    XGBoostLoadingProgressInfo,
    XGBoostPhaseCallbackProtocol,
    XGBoostPhaseInfo,
)
from covenant_radar_api.worker.optimize_xgboost_job import (
    run_optimization as run_xgboost_optimization,
)

# =============================================================================
# XGBoost Runner Protocol and Hook
# =============================================================================


class XGBoostRunnerProtocol(Protocol):
    """Protocol for XGBoost optimization runner function."""

    def __call__(
        self,
        config_json: str,
        external_dir: Path,
        output_dir: Path,
        progress_callback: XGBoostProgressCallbackProtocol | None = None,
        phase_callback: XGBoostPhaseCallbackProtocol | None = None,
        loading_progress_callback: XGBoostLoadingProgressCallbackProtocol | None = None,
    ) -> XGBoostOptimizationResult:
        """Run XGBoost hyperparameter optimization.

        Args:
            config_json: JSON configuration string.
            external_dir: Directory with external datasets.
            output_dir: Directory for output files.
            progress_callback: Optional callback for trial progress updates.
            phase_callback: Optional callback for phase transitions.
            loading_progress_callback: Optional callback for loading progress.

        Returns:
            Optimization result with best XGBoost hyperparameters.
        """
        ...


xgboost_runner: XGBoostRunnerProtocol = run_xgboost_optimization


# =============================================================================
# MLP Runner Protocol and Hook
# =============================================================================


class MLPRunnerProtocol(Protocol):
    """Protocol for MLP optimization runner function."""

    def __call__(
        self,
        config_json: str,
        external_dir: Path,
        output_dir: Path,
        progress_callback: MLPTrialProgressCallbackProtocol | None = None,
        phase_callback: MLPPhaseCallbackProtocol | None = None,
        loading_progress_callback: MLPLoadingProgressCallbackProtocol | None = None,
    ) -> MLPOptimizationResult:
        """Run MLP hyperparameter optimization.

        Args:
            config_json: JSON configuration string.
            external_dir: Directory with external datasets.
            output_dir: Directory for output files.
            progress_callback: Optional callback for trial progress updates.
            phase_callback: Optional callback for phase transitions.
            loading_progress_callback: Optional callback for loading progress.

        Returns:
            Optimization result with best MLP hyperparameters.
        """
        ...


mlp_runner: MLPRunnerProtocol = run_mlp_optimization


# =============================================================================
# LightGBM Runner Protocol and Hook
# =============================================================================


class LightGBMRunnerProtocol(Protocol):
    """Protocol for LightGBM optimization runner function."""

    def __call__(
        self,
        config_json: str,
        external_dir: Path,
        output_dir: Path,
        progress_callback: LightGBMTrialProgressCallbackProtocol | None = None,
        phase_callback: LightGBMPhaseCallbackProtocol | None = None,
        loading_progress_callback: LightGBMLoadingProgressCallbackProtocol | None = None,
    ) -> LightGBMOptimizationResult:
        """Run LightGBM hyperparameter optimization.

        Args:
            config_json: JSON configuration string.
            external_dir: Directory with external datasets.
            output_dir: Directory for output files.
            progress_callback: Optional callback for trial progress updates.
            phase_callback: Optional callback for phase transitions.
            loading_progress_callback: Optional callback for loading progress.

        Returns:
            Optimization result with best LightGBM hyperparameters.
        """
        ...


lightgbm_runner: LightGBMRunnerProtocol = run_lightgbm_optimization


# =============================================================================
# LSTM Runner Protocol and Hook
# =============================================================================


class LSTMRunnerProtocol(Protocol):
    """Protocol for LSTM optimization runner function."""

    def __call__(
        self,
        config_json: str,
        external_dir: Path,
        output_dir: Path,
        progress_callback: LSTMTrialProgressCallbackProtocol | None = None,
        phase_callback: LSTMPhaseCallbackProtocol | None = None,
        loading_progress_callback: LSTMLoadingProgressCallbackProtocol | None = None,
    ) -> LSTMOptimizationResult:
        """Run LSTM hyperparameter optimization.

        Args:
            config_json: JSON configuration string.
            external_dir: Directory with external datasets.
            output_dir: Directory for output files.
            progress_callback: Optional callback for trial progress updates.
            phase_callback: Optional callback for phase transitions.
            loading_progress_callback: Optional callback for loading progress.

        Returns:
            Optimization result with best LSTM hyperparameters.
        """
        ...


lstm_runner: LSTMRunnerProtocol = run_lstm_optimization


# =============================================================================
# Explainer Registry Factory Protocol and Hook
# =============================================================================


class ExplainerRegistryFactoryProtocol(Protocol):
    """Protocol for explainer registry factory function."""

    def __call__(self) -> ExplainerRegistry:
        """Create an ExplainerRegistry with all supported explainers.

        Returns:
            ExplainerRegistry with permutation, gradient, integrated_gradients,
            and shap_tree explainers registered.
        """
        ...


explainer_registry_factory: ExplainerRegistryFactoryProtocol = default_explainer_registry


# =============================================================================
# Dataset Loader Protocol and Hook
# =============================================================================


class DatasetLoaderCallable(Protocol):
    """Protocol for callable dataset loader function.

    Unlike DatasetLoaderProtocol (which defines a class with a load method),
    this protocol defines the signature of a callable function.
    """

    def __call__(
        self,
        config: DatasetConfig,
        external_dir: Path,
    ) -> LoadedDataset:
        """Load a dataset from disk.

        Args:
            config: Dataset configuration from registry.
            external_dir: Root directory containing dataset folders.

        Returns:
            LoadedDataset with features, labels, and metadata.

        Raises:
            FileNotFoundError: If dataset file doesn't exist.
            ValueError: If data doesn't match expected format.
        """
        ...


def _real_dataset_loader(config: DatasetConfig, external_dir: Path) -> LoadedDataset:
    """Real implementation using covenant_ml.datasets loader.

    Args:
        config: Dataset configuration from registry.
        external_dir: Root directory containing dataset folders.

    Returns:
        LoadedDataset with features, labels, and metadata.

    Raises:
        FileNotFoundError: If dataset file doesn't exist.
        ValueError: If data doesn't match expected format.
    """
    loader = create_dataset_loader()
    return loader.load(config, external_dir)


dataset_loader: DatasetLoaderCallable = _real_dataset_loader


def _real_dataset_registry() -> DatasetRegistry:
    """Real implementation returning production dataset registry.

    Returns:
        DatasetRegistry with all verified dataset configurations.
    """
    return make_default_registry()


class DatasetRegistryFactoryProtocol(Protocol):
    """Protocol for dataset registry factory function."""

    def __call__(self) -> DatasetRegistry:
        """Create a DatasetRegistry with dataset configurations.

        Returns:
            DatasetRegistry instance.
        """
        ...


dataset_registry_factory: DatasetRegistryFactoryProtocol = _real_dataset_registry


# =============================================================================
# Time-Series Dataset Registry Factory Protocol and Hook
# =============================================================================


class TimeSeriesRegistryFactoryProtocol(Protocol):
    """Protocol for time-series dataset registry factory function."""

    def __call__(self) -> TimeSeriesDatasetRegistry:
        """Create a TimeSeriesDatasetRegistry with dataset configurations.

        Returns:
            TimeSeriesDatasetRegistry instance.
        """
        ...


def _real_timeseries_registry() -> TimeSeriesDatasetRegistry:
    """Real implementation returning production time-series dataset registry.

    Returns:
        TimeSeriesDatasetRegistry with all verified time-series dataset configs.
    """
    return make_default_timeseries_registry()


timeseries_registry_factory: TimeSeriesRegistryFactoryProtocol = _real_timeseries_registry


# =============================================================================
# Time-Series Dataset Loader Protocol and Hook
# =============================================================================


class TimeSeriesLoaderCallable(Protocol):
    """Protocol for callable time-series dataset loader function."""

    def __call__(
        self,
        config: TimeSeriesDatasetConfig,
        external_dir: Path,
    ) -> LoadedDataset:
        """Load a time-series dataset from disk.

        Args:
            config: Time-series dataset configuration from registry.
            external_dir: Root directory containing dataset folders.

        Returns:
            LoadedDataset with aggregated features, labels, and metadata.

        Raises:
            FileNotFoundError: If dataset file doesn't exist.
            ValueError: If data doesn't match expected format.
        """
        ...


def _real_timeseries_loader(config: TimeSeriesDatasetConfig, external_dir: Path) -> LoadedDataset:
    """Real implementation using covenant_ml.datasets time-series loader.

    Args:
        config: Time-series dataset configuration from registry.
        external_dir: Root directory containing dataset folders.

    Returns:
        LoadedDataset with aggregated features, labels, and metadata.

    Raises:
        FileNotFoundError: If dataset file doesn't exist.
        ValueError: If data doesn't match expected format.
    """
    loader = create_timeseries_csv_loader()
    return loader.load(config, external_dir)


timeseries_loader: TimeSeriesLoaderCallable = _real_timeseries_loader


# =============================================================================
# Backend Registry Factory Protocol and Hook
# =============================================================================


class BackendRegistryFactoryProtocol(Protocol):
    """Protocol for backend registry factory function."""

    def __call__(self) -> ClassifierRegistry:
        """Create a ClassifierRegistry with all supported backends.

        Returns:
            ClassifierRegistry with xgboost, mlp, lightgbm, and lstm backends.
        """
        ...


backend_registry_factory: BackendRegistryFactoryProtocol = default_backend_registry


__all__ = [
    "BackendRegistryFactoryProtocol",
    "ClassifierRegistry",
    "DatasetConfig",
    "DatasetLoaderCallable",
    "DatasetRegistry",
    "DatasetRegistryFactoryProtocol",
    "ExplainerRegistry",
    "ExplainerRegistryFactoryProtocol",
    "LSTMLoadingProgressCallbackProtocol",
    "LSTMLoadingProgressInfo",
    "LSTMOptimizationResult",
    "LSTMPhaseCallbackProtocol",
    "LSTMPhaseInfo",
    "LSTMRunnerProtocol",
    "LSTMTrialProgressCallbackProtocol",
    "LSTMTrialProgressInfo",
    "LightGBMLoadingProgressCallbackProtocol",
    "LightGBMLoadingProgressInfo",
    "LightGBMOptimizationResult",
    "LightGBMPhaseCallbackProtocol",
    "LightGBMPhaseInfo",
    "LightGBMRunnerProtocol",
    "LightGBMTrialProgressCallbackProtocol",
    "LightGBMTrialProgressInfo",
    "LoadedDataset",
    "MLPLoadingProgressCallbackProtocol",
    "MLPLoadingProgressInfo",
    "MLPOptimizationResult",
    "MLPPhaseCallbackProtocol",
    "MLPPhaseInfo",
    "MLPRunnerProtocol",
    "MLPTrialProgressCallbackProtocol",
    "MLPTrialProgressInfo",
    "TimeSeriesDatasetConfig",
    "TimeSeriesDatasetRegistry",
    "TimeSeriesLoaderCallable",
    "TimeSeriesRegistryFactoryProtocol",
    "XGBoostLoadingProgressCallbackProtocol",
    "XGBoostLoadingProgressInfo",
    "XGBoostOptimizationResult",
    "XGBoostPhaseCallbackProtocol",
    "XGBoostPhaseInfo",
    "XGBoostProgressCallbackProtocol",
    "XGBoostProgressInfo",
    "XGBoostRunnerProtocol",
    "backend_registry_factory",
    "dataset_loader",
    "dataset_registry_factory",
    "explainer_registry_factory",
    "lightgbm_runner",
    "lstm_runner",
    "mlp_runner",
    "timeseries_loader",
    "timeseries_registry_factory",
    "xgboost_runner",
]
