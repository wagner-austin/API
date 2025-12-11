"""Public test utilities for platform_ml.

Provides Protocol types and hooks for testing ML components against real code paths.
HTTP transport fakes are defined in tests/ to avoid httpx import in src/.

Usage:
    # For wandb publisher tests:
    from platform_ml.testing import hooks, reset_hooks, WandbModuleProtocol

    def fake_load_wandb() -> WandbModuleProtocol:
        return FakeWandbModule()

    hooks.load_wandb_module = fake_load_wandb
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Protocol

from platform_core.json_utils import JSONValue

# ---------------------------------------------------------------------------
# Wandb hooks - external service, needs hooks
# ---------------------------------------------------------------------------


class WandbRunProtocol(Protocol):
    """Protocol for wandb.Run interface."""

    @property
    def id(self) -> str:
        """Run ID assigned by wandb."""
        ...


class WandbConfigProtocol(Protocol):
    """Protocol for wandb.config interface."""

    def update(self, d: Mapping[str, JSONValue]) -> None:
        """Update config with dictionary."""
        ...


class WandbTableProtocol(Protocol):
    """Protocol for wandb.Table result."""

    @property
    def columns(self) -> list[str]:
        """Column headers."""
        ...

    @property
    def data(self) -> list[list[float | int | str | bool]]:
        """Table data rows."""
        ...


class WandbTableCtorProtocol(Protocol):
    """Protocol for wandb.Table constructor."""

    def __call__(
        self,
        columns: list[str],
        data: list[list[float | int | str | bool]],
    ) -> WandbTableProtocol:
        """Create a new wandb Table."""
        ...


class WandbModuleProtocol(Protocol):
    """Protocol for wandb module interface."""

    @property
    def run(self) -> WandbRunProtocol | None:
        """Current active run, or None if not initialized."""
        ...

    @property
    def config(self) -> WandbConfigProtocol:
        """Config object for the current run."""
        ...

    @property
    def table_ctor(self) -> WandbTableCtorProtocol:
        """Table constructor for creating wandb Tables."""
        ...

    def init(self, *, project: str, name: str) -> WandbRunProtocol:
        """Initialize a new wandb run."""
        ...

    def log(self, data: Mapping[str, float | int | str | bool | WandbTableProtocol]) -> None:
        """Log metrics to the current run."""
        ...

    def finish(self) -> None:
        """Finish the current run."""
        ...


class LoadWandbModuleCallable(Protocol):
    """Protocol for loading wandb module."""

    def __call__(self) -> WandbModuleProtocol:
        """Load and return the wandb module."""
        ...


class ImportWandbCallable(Protocol):
    """Protocol for importing wandb module."""

    def __call__(self) -> WandbModuleProtocol:
        """Import and return raw wandb module."""
        ...


class CheckWandbAvailableCallable(Protocol):
    """Protocol for checking if wandb is available."""

    def __call__(self) -> bool:
        """Return True if wandb is available, False otherwise."""
        ...


class _Hooks:
    """Mutable container for test hooks.

    Only for external services that cannot be tested otherwise (wandb).
    """

    load_wandb_module: LoadWandbModuleCallable
    import_wandb: ImportWandbCallable
    check_wandb_available: CheckWandbAvailableCallable


def _production_import_wandb() -> WandbModuleProtocol:
    """Production implementation that imports real wandb."""
    raw_wandb: WandbModuleProtocol = __import__("wandb")
    return raw_wandb


def _production_check_wandb_available() -> bool:
    """Production implementation that checks if wandb is installed."""
    import importlib.util

    spec = importlib.util.find_spec("wandb")
    return spec is not None


class _WandbModuleAdapter:
    """Adapter that wraps real wandb module to match WandbModuleProtocol.

    Uses getattr for all attribute access to avoid mypy issues with
    dynamically loaded modules and PascalCase attributes like Table.
    """

    _TABLE_ATTR: Final[str] = "Table"

    def __init__(self, wandb_mod: WandbModuleProtocol) -> None:
        self._wandb = wandb_mod

    @property
    def run(self) -> WandbRunProtocol | None:
        result: WandbRunProtocol | None = getattr(self._wandb, "run", None)
        return result

    @property
    def config(self) -> WandbConfigProtocol:
        result: WandbConfigProtocol = self._wandb.config
        return result

    @property
    def table_ctor(self) -> WandbTableCtorProtocol:
        result: WandbTableCtorProtocol = getattr(self._wandb, self._TABLE_ATTR)
        return result

    def init(self, *, project: str, name: str) -> WandbRunProtocol:
        init_func = self._wandb.init
        result: WandbRunProtocol = init_func(project=project, name=name)
        return result

    def log(self, data: Mapping[str, float | int | str | bool | WandbTableProtocol]) -> None:
        log_func = self._wandb.log
        log_func(data)

    def finish(self) -> None:
        finish_func = self._wandb.finish
        finish_func()


def _production_load_wandb_module() -> WandbModuleProtocol:
    """Production implementation that loads real wandb."""
    from platform_ml.wandb_publisher import WandbUnavailableError

    if not hooks.check_wandb_available():
        raise WandbUnavailableError("wandb package is not installed")

    raw_wandb = hooks.import_wandb()
    return _WandbModuleAdapter(raw_wandb)


# Global hooks instance - only for wandb (external service)
hooks: Final[_Hooks] = _Hooks()


def set_production_hooks() -> None:
    """Set all hooks to production implementations."""
    hooks.check_wandb_available = _production_check_wandb_available
    hooks.import_wandb = _production_import_wandb
    hooks.load_wandb_module = _production_load_wandb_module


def reset_hooks() -> None:
    """Reset hooks to production implementations."""
    set_production_hooks()


# Initialize with production hooks by default
set_production_hooks()


__all__ = [
    "LoadWandbModuleCallable",
    "WandbConfigProtocol",
    "WandbModuleProtocol",
    "WandbRunProtocol",
    "WandbTableCtorProtocol",
    "WandbTableProtocol",
    "hooks",
    "reset_hooks",
    "set_production_hooks",
]
