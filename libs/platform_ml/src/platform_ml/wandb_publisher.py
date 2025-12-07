"""Protocol-based wandb integration for ML services.

This module provides a reusable wandb publisher that can be shared across
ML services (Model-Trainer, handwriting-ai, etc.) with strict typing.

The publisher uses Protocol-based typing to avoid direct wandb type imports,
enabling strict mypy compliance without Any, cast, or type: ignore.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final, Literal, Protocol, TypedDict

from platform_core.json_utils import JSONValue

from .wandb_types import WandbInitResult


class _WandbRunProto(Protocol):
    """Protocol for wandb.Run interface."""

    @property
    def id(self) -> str:
        """Run ID assigned by wandb."""
        ...


class _WandbConfigProto(Protocol):
    """Protocol for wandb.config interface."""

    def update(self, d: Mapping[str, JSONValue]) -> None:
        """Update config with dictionary."""
        ...


class _WandbTableProto(Protocol):
    """Protocol for wandb.Table constructor result."""


class _WandbTableCtorProto(Protocol):
    """Protocol for wandb.Table constructor."""

    def __call__(
        self,
        columns: list[str],
        data: list[list[float | int | str | bool]],
    ) -> _WandbTableProto:
        """Create a new wandb Table."""
        ...


class _WandbModuleProto(Protocol):
    """Protocol for wandb module interface."""

    @property
    def run(self) -> _WandbRunProto | None:
        """Current active run, or None if not initialized."""
        ...

    @property
    def config(self) -> _WandbConfigProto:
        """Config object for the current run."""
        ...

    def init(
        self,
        *,
        project: str,
        name: str,
    ) -> _WandbRunProto:
        """Initialize a new wandb run."""
        ...

    def log(self, data: Mapping[str, float | int | str | bool | _WandbTableProto]) -> None:
        """Log metrics to the current run."""
        ...

    def finish(self) -> None:
        """Finish the current run."""
        ...


class WandbUnavailableError(Exception):
    """Raised when wandb is requested but not installed."""


def _load_wandb_module() -> _WandbModuleProto:
    """Load wandb module with Protocol-typed interface.

    Returns:
        The wandb module typed as _WandbModuleProto.

    Raises:
        WandbUnavailableError: If wandb package is not installed.
    """
    # Import check - this will raise ImportError if wandb not installed
    # We convert to our domain error for clarity
    import importlib.util

    spec = importlib.util.find_spec("wandb")
    if spec is None:
        raise WandbUnavailableError("wandb package is not installed")

    mod = __import__("wandb")
    wandb_module: _WandbModuleProto = mod
    return wandb_module


class _MetricsData(TypedDict, total=False):
    """Internal typed dict for metrics logging."""

    # Step metrics
    train_loss: float
    train_ppl: float
    grad_norm: float
    samples_per_sec: float
    learning_rate: float
    global_step: int
    epoch: int

    # Validation metrics
    val_loss: float
    val_ppl: float
    best_val_loss: float
    epochs_no_improve: int

    # Final metrics
    test_loss: float
    test_ppl: float
    early_stopped: bool


class WandbPublisher:
    """Reusable wandb integration for ML services.

    This publisher provides a typed interface for logging training metrics
    to Weights & Biases. It uses Protocol-based typing to maintain strict
    type safety without direct wandb type imports.

    When enabled=False, all methods are no-ops.
    When enabled=True and wandb is not installed, raises WandbUnavailableError.
    """

    _enabled: Final[bool]
    _wandb: _WandbModuleProto | None
    _run_id: str | None

    def __init__(
        self,
        *,
        project: str,
        run_name: str,
        enabled: bool = True,
    ) -> None:
        """Initialize the wandb publisher.

        Args:
            project: Wandb project name.
            run_name: Name for this run.
            enabled: Whether to enable wandb logging. If False, all methods are no-ops.

        Raises:
            WandbUnavailableError: If enabled=True but wandb is not installed.
        """
        self._enabled = enabled
        self._wandb = None
        self._run_id = None

        if not enabled:
            return

        # Load wandb module - raises WandbUnavailableError if not installed
        self._wandb = _load_wandb_module()

        # Initialize the run
        run = self._wandb.init(project=project, name=run_name)
        self._run_id = run.id

    def get_init_result(self) -> WandbInitResult:
        """Get the initialization result.

        Returns:
            WandbInitResult with status and run_id.
        """
        if not self._enabled:
            return WandbInitResult(status="disabled", run_id=None)
        # When enabled=True, _wandb is always set (constructor raises if unavailable)
        return WandbInitResult(status="enabled", run_id=self._run_id)

    def log_config(self, config: Mapping[str, JSONValue]) -> None:
        """Log configuration to wandb.

        Args:
            config: Configuration dictionary to log.
        """
        if not self._enabled or self._wandb is None:
            return
        if self._wandb.run is None:
            return
        self._wandb.config.update(config)

    def log_step(self, metrics: Mapping[str, float | int]) -> None:
        """Log per-step training metrics.

        Args:
            metrics: Metrics dictionary with train_loss, train_ppl, grad_norm, etc.
        """
        if not self._enabled or self._wandb is None:
            return
        if self._wandb.run is None:
            return
        self._wandb.log(metrics)

    def log_epoch(self, metrics: Mapping[str, float | int]) -> None:
        """Log epoch-end metrics with validation results.

        Args:
            metrics: Metrics dictionary with val_loss, val_ppl, best_val_loss, etc.
        """
        if not self._enabled or self._wandb is None:
            return
        if self._wandb.run is None:
            return
        self._wandb.log(metrics)

    def log_final(self, metrics: Mapping[str, float | int | bool]) -> None:
        """Log final training metrics.

        Args:
            metrics: Final metrics with test_loss, test_ppl, early_stopped.
        """
        if not self._enabled or self._wandb is None:
            return
        if self._wandb.run is None:
            return
        # Convert bool to int for wandb compatibility
        log_data: dict[str, float | int] = {}
        for k, v in metrics.items():
            if isinstance(v, bool):
                log_data[k] = 1 if v else 0
            else:
                log_data[k] = v
        self._wandb.log(log_data)

    def log_table(
        self,
        name: str,
        columns: list[str],
        data: Sequence[Sequence[float | int]],
    ) -> None:
        """Log a summary table to wandb.

        Args:
            name: Name for the table (e.g., "epoch_summary").
            columns: Column headers.
            data: List of rows, each row is a list of values.
        """
        if not self._enabled or self._wandb is None:
            return
        if self._wandb.run is None:
            return

        # Get Table constructor via getattr to avoid PascalCase property in Protocol
        wandb_mod = __import__("wandb")
        table_ctor: _WandbTableCtorProto = wandb_mod.Table

        # Convert sequences to lists for wandb.Table
        data_lists: list[list[float | int | str | bool]] = [list(row) for row in data]
        table = table_ctor(columns=columns, data=data_lists)
        self._wandb.log({name: table})

    def finish(self) -> None:
        """Finish the wandb run.

        Should be called at the end of training to properly close the run.
        """
        if not self._enabled or self._wandb is None:
            return
        if self._wandb.run is None:
            return
        self._wandb.finish()

    @property
    def is_enabled(self) -> bool:
        """Whether wandb logging is enabled and active."""
        return self._enabled and self._wandb is not None and self._wandb.run is not None


class DisabledPublisherStatus(TypedDict):
    """Status for a disabled publisher."""

    status: Literal["disabled"]
    run_id: None


class EnabledPublisherStatus(TypedDict):
    """Status for an enabled publisher."""

    status: Literal["enabled"]
    run_id: str


PublisherStatus = DisabledPublisherStatus | EnabledPublisherStatus


__all__ = [
    "DisabledPublisherStatus",
    "EnabledPublisherStatus",
    "PublisherStatus",
    "WandbPublisher",
    "WandbUnavailableError",
]
