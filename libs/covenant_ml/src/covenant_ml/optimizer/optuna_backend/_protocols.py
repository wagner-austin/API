"""Optuna protocol type definitions.

Strict typing only: no Any, no casts, no stubs.
Defines Protocol types for Optuna library interfaces.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


class OptunaSamplerProtocol(Protocol):
    """Protocol for Optuna sampler."""

    ...


class OptunaTrialProtocol(Protocol):
    """Protocol for Optuna trial object."""

    @property
    def number(self) -> int: ...

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        log: bool = False,
    ) -> int: ...

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
    ) -> float: ...

    def suggest_categorical(
        self,
        name: str,
        choices: tuple[float, ...] | tuple[int, ...] | tuple[str, ...],
    ) -> float | int | str: ...

    def report(self, value: float, step: int) -> None: ...

    def should_prune(self) -> bool: ...


class OptunaStudyProtocol(Protocol):
    """Protocol for Optuna study object."""

    @property
    def best_trial(self) -> OptunaTrialProtocol: ...

    @property
    def best_value(self) -> float: ...

    @property
    def best_params(self) -> dict[str, float | int | str]: ...

    def optimize(
        self,
        func: Callable[[OptunaTrialProtocol], float],
        n_trials: int,
        timeout: float | None = None,
        callbacks: list[Callable[[OptunaStudyProtocol, OptunaTrialProtocol], None]] | None = None,
    ) -> None: ...

    def get_trials(
        self,
        deepcopy: bool = True,
        states: tuple[str, ...] | None = None,
    ) -> list[OptunaTrialProtocol]: ...


class OptunaCreateStudyProtocol(Protocol):
    """Protocol for optuna.create_study function."""

    def __call__(
        self,
        *,
        direction: str,
        sampler: OptunaSamplerProtocol,
        pruner: OptunaPrunerProtocol | None = None,
    ) -> OptunaStudyProtocol: ...


class OptunaTPESamplerProtocol(Protocol):
    """Protocol for TPESampler constructor."""

    def __call__(
        self,
        *,
        seed: int,
        n_startup_trials: int,
    ) -> OptunaSamplerProtocol: ...


class OptunaPrunerProtocol(Protocol):
    """Protocol for Optuna pruner."""

    ...


class OptunaMedianPrunerProtocol(Protocol):
    """Protocol for MedianPruner constructor."""

    def __call__(
        self,
        *,
        n_startup_trials: int,
        n_warmup_steps: int,
    ) -> OptunaPrunerProtocol: ...


class OptunaModuleProtocol(Protocol):
    """Protocol for optuna module."""

    @property
    def create_study(self) -> OptunaCreateStudyProtocol: ...


__all__ = [
    "OptunaCreateStudyProtocol",
    "OptunaMedianPrunerProtocol",
    "OptunaModuleProtocol",
    "OptunaPrunerProtocol",
    "OptunaSamplerProtocol",
    "OptunaStudyProtocol",
    "OptunaTPESamplerProtocol",
    "OptunaTrialProtocol",
]
