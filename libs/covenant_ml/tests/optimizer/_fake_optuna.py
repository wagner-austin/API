"""Fake Optuna study/trial/sampler machinery for optimizer tests."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from covenant_ml.optimizer.optuna_backend._protocols import (
    OptunaCreateStudyProtocol,
    OptunaMedianPrunerProtocol,
    OptunaPrunerProtocol,
    OptunaSamplerProtocol,
    OptunaStudyProtocol,
    OptunaTPESamplerProtocol,
    OptunaTrialProtocol,
)
from covenant_ml.optimizer.types import (
    OptimizationConfig,
)


class FakeTrial:
    """Fake Optuna trial that returns deterministic values."""

    def __init__(self, trial_number: int) -> None:
        self._number = trial_number
        self._suggestions: dict[str, float | int | str] = {}

    @property
    def number(self) -> int:
        """Return trial number."""
        return self._number

    @property
    def suggestions(self) -> dict[str, float | int | str]:
        """Return all suggestions made during this trial."""
        return self._suggestions

    def suggest_int(self, name: str, low: int, high: int, *, log: bool = False) -> int:
        """Suggest an integer parameter."""
        _ = log
        value = low + (self._number % (high - low + 1))
        self._suggestions[name] = value
        return value

    def suggest_float(self, name: str, low: float, high: float, *, log: bool = False) -> float:
        """Suggest a float parameter."""
        _ = log
        ratio = (self._number % 10) / 10.0
        value = low + ratio * (high - low)
        self._suggestions[name] = value
        return value

    def suggest_categorical(
        self, name: str, choices: tuple[float, ...] | tuple[int, ...] | tuple[str, ...]
    ) -> float | int | str:
        """Suggest a categorical parameter."""
        index = self._number % len(choices)
        value = choices[index]
        self._suggestions[name] = value
        return value

    def report(self, value: float, step: int) -> None:
        """Report intermediate value (no-op)."""
        _ = value, step

    def should_prune(self) -> bool:
        """Check if trial should be pruned (always False)."""
        return False


class FakeSampler:
    """Fake Optuna sampler."""

    def __init__(self, *, seed: int, n_startup_trials: int) -> None:
        self.seed = seed
        self.n_startup_trials = n_startup_trials


class FakePruner:
    """Fake Optuna pruner."""

    def __init__(self, *, n_startup_trials: int, n_warmup_steps: int) -> None:
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps


class FakeStudy:
    """Fake Optuna study that runs trials deterministically."""

    def __init__(
        self,
        *,
        direction: str,
        sampler: OptunaSamplerProtocol,
        pruner: OptunaPrunerProtocol | None,
    ) -> None:
        self._direction = direction
        self._sampler = sampler
        self._pruner = pruner
        self._trials: list[FakeTrial] = []
        self._values: list[float] = []
        self._best_idx = 0

    @property
    def best_trial(self) -> OptunaTrialProtocol:
        """Return the best trial."""
        return self._trials[self._best_idx]

    @property
    def best_value(self) -> float:
        """Return the best value."""
        return self._values[self._best_idx]

    @property
    def best_params(self) -> dict[str, float | int | str]:
        """Return the best trial's parameters."""
        return self._trials[self._best_idx].suggestions

    def optimize(
        self,
        func: Callable[[OptunaTrialProtocol], float],
        n_trials: int,
        timeout: float | None = None,
        callbacks: list[Callable[[OptunaStudyProtocol, OptunaTrialProtocol], None]] | None = None,
    ) -> None:
        """Run optimization by calling func for each trial."""
        _ = timeout, callbacks
        for i in range(n_trials):
            trial = FakeTrial(i)
            value = func(trial)
            self._trials.append(trial)
            self._values.append(value)
            if self._direction == "maximize":
                if value > self._values[self._best_idx]:
                    self._best_idx = i
            else:
                if value < self._values[self._best_idx]:
                    self._best_idx = i

    def get_trials(
        self, deepcopy: bool = True, states: tuple[str, ...] | None = None
    ) -> list[OptunaTrialProtocol]:
        """Return all trials."""
        _ = deepcopy, states
        return list(self._trials)


def fake_create_study(
    *, direction: str, sampler: OptunaSamplerProtocol, pruner: OptunaPrunerProtocol | None = None
) -> OptunaStudyProtocol:
    """Create a fake Optuna study."""
    return FakeStudy(direction=direction, sampler=sampler, pruner=pruner)


def fake_tpe_sampler(*, seed: int, n_startup_trials: int) -> OptunaSamplerProtocol:
    """Create a fake TPE sampler."""
    return FakeSampler(seed=seed, n_startup_trials=n_startup_trials)


def fake_median_pruner(*, n_startup_trials: int, n_warmup_steps: int) -> OptunaPrunerProtocol:
    """Create a fake median pruner."""
    return FakePruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps)


def get_fake_optuna_factories() -> tuple[
    OptunaCreateStudyProtocol, OptunaTPESamplerProtocol, OptunaMedianPrunerProtocol
]:
    """Return fake Optuna factory functions for testing."""
    return fake_create_study, fake_tpe_sampler, fake_median_pruner


def make_optuna_test_data(
    n_samples: int = 50, n_features: int = 4, seed: int = 42
) -> tuple[NDArray[np.float64], NDArray[np.int64], list[str]]:
    """Create test dataset for Optuna backend optimizer tests.

    Args:
        n_samples: Number of samples to generate.
        n_features: Number of features.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (features, labels, feature_names).
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    n_positive = n_samples // 3
    y = np.zeros(n_samples, dtype=np.int64)
    y[:n_positive] = 1
    rng.shuffle(y)
    return x, y, [f"feat_{i}" for i in range(n_features)]


def make_optuna_config(
    n_trials: int = 5, pruning_enabled: bool = True, timeout_seconds: int | None = None
) -> OptimizationConfig:
    """Create test optimization config for Optuna backend tests.

    Args:
        n_trials: Number of trials to run.
        pruning_enabled: Whether to enable pruning.
        timeout_seconds: Optional timeout in seconds.

    Returns:
        OptimizationConfig with test defaults.
    """
    return {
        "n_trials": n_trials,
        "timeout_seconds": timeout_seconds,
        "n_startup_trials": 10,
        "random_state": 42,
        "direction": "maximize",
        "pruning_enabled": pruning_enabled,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
    }
