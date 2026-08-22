"""Test hooks for AMEX pipeline components.

Provides fake implementations for dependency injection in tests.
Production code uses real implementations; tests set these module-level
hooks to inject fakes without conditionals in core logic.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path

from covenant_ml.ensemble.testing import fake_minimize

import scripts.amex._hooks as hooks_module
from scripts.amex._hook_protocols import (
    FakeDatasetSpec,
)
from scripts.amex._test_fakes import (
    FakeConsole,
    FakeRegistry,
    FakeTimeseriesLoader,
    make_fake_optimizer,
)


def configure_fake_console() -> FakeConsole:
    """Configure fake console hook and return the fake console.

    Returns:
        FakeConsole instance.
    """
    fake_console = FakeConsole()
    hooks_module.console_hook = lambda: fake_console
    return fake_console


def configure_fake_project_root(path: Path) -> None:
    """Configure fake project root hook.

    Args:
        path: Fake project root path.
    """
    hooks_module.project_root_hook = lambda: path


def configure_fake_registry(output_dir: Path, random_state: int = 42) -> FakeRegistry:
    """Configure fake registry hook and return the fake registry.

    Args:
        output_dir: Directory for model outputs.
        random_state: Random seed.

    Returns:
        FakeRegistry instance.
    """
    fake_registry = FakeRegistry(output_dir, random_state)

    def _fake_registry_factory() -> hooks_module.RegistryProtocol:
        return fake_registry

    hooks_module.registry_hook = _fake_registry_factory
    return fake_registry


def configure_fake_timeseries_loader(
    train_spec: FakeDatasetSpec,
    test_spec: FakeDatasetSpec,
) -> FakeTimeseriesLoader:
    """Configure fake time-series loader hook and return the fake loader.

    Args:
        train_spec: Specification for training dataset.
        test_spec: Specification for test dataset.

    Returns:
        FakeTimeseriesLoader instance.
    """
    fake_loader = FakeTimeseriesLoader(train_spec, test_spec)
    hooks_module.timeseries_loader_hook = fake_loader
    return fake_loader


def configure_fake_ensemble_optimizer(random_state: int = 42) -> None:
    """Configure fake ensemble optimizer hook.

    Args:
        random_state: Random seed.
    """
    hooks_module.ensemble_optimizer_hook = make_fake_optimizer(random_state)


def configure_fake_scipy() -> None:
    """Configure fake scipy minimize hook."""
    hooks_module.configure_minimize_hook(fake_minimize)


def configure_all_fakes(
    project_root: Path,
    output_dir: Path,
    train_spec: FakeDatasetSpec,
    test_spec: FakeDatasetSpec,
    random_state: int = 42,
) -> FakeConsole:
    """Configure all fake hooks for testing.

    Args:
        project_root: Fake project root.
        output_dir: Directory for model outputs.
        train_spec: Specification for training dataset.
        test_spec: Specification for test dataset.
        random_state: Random seed.

    Returns:
        FakeConsole for inspecting output.
    """
    fake_console = configure_fake_console()
    configure_fake_project_root(project_root)
    configure_fake_registry(output_dir, random_state)
    configure_fake_timeseries_loader(train_spec, test_spec)
    configure_fake_ensemble_optimizer(random_state)
    configure_fake_scipy()
    return fake_console


__all__ = [
    "configure_all_fakes",
    "configure_fake_console",
    "configure_fake_ensemble_optimizer",
    "configure_fake_project_root",
    "configure_fake_registry",
    "configure_fake_scipy",
    "configure_fake_timeseries_loader",
]
