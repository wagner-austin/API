"""The strategy package's injection seams and the protocol types they use.

Strict typing only: no Any, no casts, no stubs.

``optuna_factories`` is bound to the real optuna import, so the strategy
reaches optuna without anything being wired first. Tests rebind this module's
attribute and restore it afterwards.

These protocol types are a narrower view of the same library than
``optuna_backend._protocols`` takes -- that one also requires report,
should_prune and get_trials, which this strategy never calls. Merging the two
would widen what every fake here has to implement, so they are left separate.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from ..types import SampledFloatParams, SampledIntParams, SampledStringParams, SearchSpace

GridTuple = tuple[SampledIntParams, SampledFloatParams, SampledStringParams]


class BuildGridProtocol(Protocol):
    """Protocol for grid building function."""

    def __call__(
        self,
        search_space: SearchSpace,
        n_points: int,
    ) -> list[GridTuple]: ...


def _real_build_grid(search_space: SearchSpace, n_points: int) -> list[GridTuple]:
    """Build the parameter grid the search space calls for.

    grid_search is imported on the call rather than on import, because that
    module reads this one for the seam.

    Args:
        search_space: Space to enumerate.
        n_points: Points per dimension.

    Returns:
        One tuple of sampled parameters per grid point.
    """
    from .grid_search import _build_grid

    return _build_grid(search_space, n_points)


# The grid builder, bound to the real one. Tests rebind this module attribute.
build_grid: BuildGridProtocol = _real_build_grid


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


class OptunaPrunerProtocol(Protocol):
    """Protocol for Optuna pruner."""

    ...


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


class OptunaMedianPrunerProtocol(Protocol):
    """Protocol for MedianPruner constructor."""

    def __call__(
        self,
        *,
        n_startup_trials: int,
        n_warmup_steps: int,
    ) -> OptunaPrunerProtocol: ...


def _real_optuna_factories() -> tuple[
    OptunaCreateStudyProtocol,
    OptunaTPESamplerProtocol,
    OptunaMedianPrunerProtocol,
]:
    """Get real Optuna factories via dynamic import.

    Returns:
        Tuple of (create_study, TPESampler, MedianPruner) factories.
    """
    optuna_mod = __import__("optuna")
    create_study: OptunaCreateStudyProtocol = optuna_mod.create_study

    samplers_submod = __import__("optuna.samplers", fromlist=["TPESampler"])
    tpe_sampler: OptunaTPESamplerProtocol = samplers_submod.TPESampler

    pruners_submod = __import__("optuna.pruners", fromlist=["MedianPruner"])
    median_pruner: OptunaMedianPrunerProtocol = pruners_submod.MedianPruner

    return create_study, tpe_sampler, median_pruner


class OptunaFactoriesProtocol(Protocol):
    """Protocol for the optuna factory provider."""

    def __call__(
        self,
    ) -> tuple[
        OptunaCreateStudyProtocol,
        OptunaTPESamplerProtocol,
        OptunaMedianPrunerProtocol,
    ]:
        """Provide the optuna entry points the strategy needs.

        Returns:
            Tuple of (create_study, TPESampler, MedianPruner) factories.
        """
        ...


optuna_factories: OptunaFactoriesProtocol = _real_optuna_factories


__all__ = [
    "BuildGridProtocol",
    "GridTuple",
    "OptunaCreateStudyProtocol",
    "OptunaFactoriesProtocol",
    "OptunaMedianPrunerProtocol",
    "OptunaPrunerProtocol",
    "OptunaSamplerProtocol",
    "OptunaStudyProtocol",
    "OptunaTPESamplerProtocol",
    "OptunaTrialProtocol",
    "build_grid",
    "optuna_factories",
]
