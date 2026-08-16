"""Dependency hooks for AMEX competition pipeline.

Production code uses real implementations; tests can override these module-level
symbols to inject fakes without conditionals in core logic.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from covenant_ml.backends.protocol import ClassifierBackend
from covenant_ml.backends.registry import ClassifierRegistry
from covenant_ml.datasets import (
    LoadedDataset,
    TimeSeriesDatasetConfig,
    create_timeseries_csv_loader,
)
from covenant_ml.ensemble import _hooks as ensemble_hooks
from covenant_ml.ensemble._hooks import _MinimizeFnProtocol
from covenant_ml.ensemble.types import EnsembleOOFData, OptimizationConfig, OptimizationResult
from covenant_ml.types import BackendName

# =============================================================================
# Console Hook
# =============================================================================


class ConsoleProtocol(Protocol):
    """Protocol for console output."""

    def write(self, message: str) -> None:
        """Write a message to the console.

        Args:
            message: Message to write.
        """
        ...


class _RichConsoleAdapter:
    """Adapter that wraps Rich console to implement ConsoleProtocol."""

    def write(self, message: str) -> None:
        """Write a message using Rich console.

        Args:
            message: Message to write.
        """
        from platform_core.logging import get_rich_console

        console = get_rich_console()
        console_print = console.print
        console_print(message)


class ConsoleHookCallable(Protocol):
    """Protocol for console hook factory function."""

    def __call__(self) -> ConsoleProtocol:
        """Create a console for output.

        Returns:
            ConsoleProtocol implementation.
        """
        ...


def _default_console_factory() -> ConsoleProtocol:
    """Default console factory returning Rich console adapter.

    Returns:
        RichConsoleAdapter instance.
    """
    return _RichConsoleAdapter()


console_hook: ConsoleHookCallable = _default_console_factory


def get_console() -> ConsoleProtocol:
    """Get the current console via hook.

    Returns:
        ConsoleProtocol implementation from current hook.
    """
    return console_hook()


# =============================================================================
# Project Root Hook
# =============================================================================


class ProjectRootCallable(Protocol):
    """Protocol for project root path factory function."""

    def __call__(self) -> Path:
        """Get the project root path.

        Returns:
            Path to project root directory.
        """
        ...


def _default_project_root() -> Path:
    """Default project root factory.

    Returns:
        Path to covenant-radar-api root.
    """
    return Path(__file__).parent.parent.parent


project_root_hook: ProjectRootCallable = _default_project_root


def get_project_root() -> Path:
    """Get project root path via hook.

    Returns:
        Path to project root directory.
    """
    return project_root_hook()


# =============================================================================
# Registry Hook
# =============================================================================


class RegistryProtocol(Protocol):
    """Protocol for classifier registry that only requires get method.

    This allows fakes to implement just the get method.
    """

    def get(self, name: BackendName) -> ClassifierBackend:
        """Get a backend by name.

        Args:
            name: Backend name (must be a valid BackendName literal).

        Returns:
            ClassifierBackend instance.
        """
        ...


class RegistryHookCallable(Protocol):
    """Protocol for registry hook factory function."""

    def __call__(self) -> RegistryProtocol:
        """Create a classifier registry.

        Returns:
            Registry implementation with get method.
        """
        ...


def _default_registry_factory() -> RegistryProtocol:
    """Default registry factory returning real ClassifierRegistry.

    Returns:
        ClassifierRegistry from covenant_ml.
    """
    from covenant_ml.backends.registry import default_registry

    return default_registry()


registry_hook: RegistryHookCallable = _default_registry_factory


def get_registry() -> RegistryProtocol:
    """Get the current registry via hook.

    Returns:
        Registry with get method from current hook.
    """
    return registry_hook()


# =============================================================================
# Time-Series Loader Hook
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


def _real_timeseries_loader(
    config: TimeSeriesDatasetConfig,
    external_dir: Path,
) -> LoadedDataset:
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


timeseries_loader_hook: TimeSeriesLoaderCallable = _real_timeseries_loader


def get_timeseries_loader() -> TimeSeriesLoaderCallable:
    """Get the current time-series loader via hook.

    Returns:
        TimeSeriesLoaderCallable from current hook.
    """
    return timeseries_loader_hook


# =============================================================================
# Ensemble Optimizer Hook
# =============================================================================


class EnsembleOptimizerCallable(Protocol):
    """Protocol for ensemble optimizer function."""

    def __call__(
        self,
        oof_data: EnsembleOOFData,
        config: OptimizationConfig,
    ) -> OptimizationResult:
        """Optimize ensemble weights.

        Args:
            oof_data: Out-of-fold predictions from all models.
            config: Optimization configuration.

        Returns:
            OptimizationResult with optimized weights.

        Raises:
            ValueError: If OOF data is invalid.
        """
        ...


def _real_ensemble_optimizer(
    oof_data: EnsembleOOFData,
    config: OptimizationConfig,
) -> OptimizationResult:
    """Real implementation using covenant_ml ensemble optimizer.

    Args:
        oof_data: Out-of-fold predictions from all models.
        config: Optimization configuration.

    Returns:
        OptimizationResult with optimized weights.

    Raises:
        ValueError: If OOF data is invalid.
    """
    from covenant_ml.ensemble.optimizer import optimize_ensemble_weights

    return optimize_ensemble_weights(oof_data, config)


ensemble_optimizer_hook: EnsembleOptimizerCallable = _real_ensemble_optimizer


def get_ensemble_optimizer() -> EnsembleOptimizerCallable:
    """Get the current ensemble optimizer via hook.

    Returns:
        EnsembleOptimizerCallable from current hook.
    """
    return ensemble_optimizer_hook


# =============================================================================
# Scipy Minimize Hook (re-exported from ensemble.optimizer)
# =============================================================================


def configure_minimize_hook(hook: _MinimizeFnProtocol) -> None:
    """Configure a custom minimize function for testing.

    Args:
        hook: The minimize function to use.
    """
    ensemble_hooks.minimize = hook


def restore_real_minimize() -> None:
    """Restore the real scipy solver covenant_ml binds."""
    ensemble_hooks.minimize = ensemble_hooks._real_minimize


__all__ = [
    "ClassifierRegistry",
    "ConsoleHookCallable",
    "ConsoleProtocol",
    "EnsembleOptimizerCallable",
    "ProjectRootCallable",
    "RegistryHookCallable",
    "RegistryProtocol",
    "TimeSeriesLoaderCallable",
    "configure_minimize_hook",
    "console_hook",
    "ensemble_optimizer_hook",
    "get_console",
    "get_ensemble_optimizer",
    "get_project_root",
    "get_registry",
    "get_timeseries_loader",
    "project_root_hook",
    "registry_hook",
    "restore_real_minimize",
    "timeseries_loader_hook",
]
