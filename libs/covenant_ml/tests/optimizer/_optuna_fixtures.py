"""Shared fixtures and helpers for test_optuna_tpe splits."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from covenant_ml.optimizer.strategies._hooks import (
    OptunaCreateStudyProtocol,
    OptunaMedianPrunerProtocol,
    OptunaPrunerProtocol,
    OptunaSamplerProtocol,
    OptunaStudyProtocol,
    OptunaTPESamplerProtocol,
    OptunaTrialProtocol,
)


class _FakeSampler:
    """Fake TPE sampler for testing."""

    def __init__(self, seed: int, n_startup_trials: int) -> None:
        self._seed = seed
        self._n_startup_trials = n_startup_trials


class _FakePruner:
    """Fake pruner for testing."""

    def __init__(self, n_startup_trials: int, n_warmup_steps: int) -> None:
        self._n_startup_trials = n_startup_trials
        self._n_warmup_steps = n_warmup_steps


class _FakeTrial:
    """Fake Optuna trial for testing."""

    def __init__(self, trial_number: int, rng_seed: int) -> None:
        self._number = trial_number
        self._rng = np.random.default_rng(rng_seed + trial_number)
        self._params: dict[str, float | int | str] = {}

    @property
    def number(self) -> int:
        return self._number

    def get_params(self) -> dict[str, float | int | str]:
        """Get all suggested params."""
        return dict(self._params)

    def suggest_int(self, name: str, low: int, high: int, *, log: bool = False) -> int:
        """Suggest integer parameter."""
        value = int(self._rng.integers(low, high + 1))
        self._params[name] = value
        return value

    def suggest_float(self, name: str, low: float, high: float, *, log: bool = False) -> float:
        """Suggest float parameter."""
        if log:
            import math

            log_low = math.log(low)
            log_high = math.log(high)
            uniform_val: float = float(self._rng.uniform(log_low, log_high))
            value = math.exp(uniform_val)
        else:
            value = float(self._rng.uniform(low, high))
        self._params[name] = value
        return value

    def suggest_categorical(
        self,
        name: str,
        choices: tuple[float, ...] | tuple[int, ...] | tuple[str, ...],
    ) -> float | int | str:
        """Suggest categorical parameter."""
        idx = int(self._rng.integers(0, len(choices)))
        value = choices[idx]
        self._params[name] = value
        return value


class _FakeStudy:
    """Fake Optuna study for testing."""

    def __init__(
        self,
        direction: str,
        sampler: OptunaSamplerProtocol,
        pruner: OptunaPrunerProtocol | None = None,
    ) -> None:
        self._direction = direction
        self._sampler = sampler
        self._pruner = pruner
        self._best_trial: OptunaTrialProtocol | None = None
        self._best_value = float("-inf") if direction == "maximize" else float("inf")
        self._best_params: dict[str, float | int | str] = {}
        self._rng_seed = 42

    @property
    def best_trial(self) -> OptunaTrialProtocol:
        if self._best_trial is None:
            return _FakeTrial(0, self._rng_seed)
        return self._best_trial

    @property
    def best_value(self) -> float:
        return self._best_value

    @property
    def best_params(self) -> dict[str, float | int | str]:
        return self._best_params

    def optimize(
        self,
        func: Callable[[OptunaTrialProtocol], float],
        n_trials: int,
        timeout: float | None = None,
        callbacks: list[Callable[[OptunaStudyProtocol, OptunaTrialProtocol], None]] | None = None,
    ) -> None:
        """Run optimization."""
        for i in range(n_trials):
            fake_trial = _FakeTrial(i, self._rng_seed)
            trial: OptunaTrialProtocol = fake_trial
            value = func(trial)

            is_better = (self._direction == "maximize" and value > self._best_value) or (
                self._direction == "minimize" and value < self._best_value
            )
            if is_better:
                self._best_value = value
                self._best_trial = trial
                self._best_params = fake_trial.get_params()


def _make_fake_optuna_hook() -> Callable[
    [],
    tuple[
        OptunaCreateStudyProtocol,
        OptunaTPESamplerProtocol,
        OptunaMedianPrunerProtocol,
    ],
]:
    """Create fake Optuna hook for testing."""

    def create_study(
        *,
        direction: str,
        sampler: OptunaSamplerProtocol,
        pruner: OptunaPrunerProtocol | None = None,
    ) -> OptunaStudyProtocol:
        return _FakeStudy(direction=direction, sampler=sampler, pruner=pruner)

    def tpe_sampler(*, seed: int, n_startup_trials: int) -> OptunaSamplerProtocol:
        return _FakeSampler(seed=seed, n_startup_trials=n_startup_trials)

    def median_pruner(*, n_startup_trials: int, n_warmup_steps: int) -> OptunaPrunerProtocol:
        return _FakePruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps)

    def hook() -> tuple[
        OptunaCreateStudyProtocol,
        OptunaTPESamplerProtocol,
        OptunaMedianPrunerProtocol,
    ]:
        cs: OptunaCreateStudyProtocol = create_study
        ts: OptunaTPESamplerProtocol = tpe_sampler
        mp: OptunaMedianPrunerProtocol = median_pruner
        return cs, ts, mp

    return hook
