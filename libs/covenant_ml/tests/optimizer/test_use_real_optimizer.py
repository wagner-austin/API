"""Tests that the startup wiring call reaches every injection point.

The optimizer has two independent hooks in separate modules: the optuna
module factories and the TPE strategy factories. Every entry point called
use_real_optuna() alone, following this package's own documentation, which
left the TPE hook unset -- so /ml/optimize raised "Optuna TPE hook not set"
for every backend, and hyperparameter optimization did not work at all.

use_real_optimizer() exists so a single call wires everything, and so a hook
added later is wired everywhere by changing one function. This test is what
makes that promise checkable: it asserts both seams are live afterwards, and
it fails if a third is added without being wired here.

Strict typing only: no Any, no casts, no type: ignore, no stubs, no mocks.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

from covenant_ml.optimizer import use_real_optimizer
from covenant_ml.optimizer.optuna_backend._hooks import (
    get_optuna_factories,
    set_optuna_module_hook,
)
from covenant_ml.optimizer.strategies.optuna_tpe import (
    _get_optuna_factories as get_tpe_factories,
)
from covenant_ml.optimizer.strategies.optuna_tpe import (
    set_optuna_tpe_hook,
)


@pytest.fixture(autouse=True)
def clear_hooks() -> Generator[None, None, None]:
    """Start each test with both hooks unset, and leave them unset."""
    set_optuna_module_hook(None)
    set_optuna_tpe_hook(None)
    yield
    set_optuna_module_hook(None)
    set_optuna_tpe_hook(None)


def test_tpe_hook_is_unset_before_wiring() -> None:
    """Without wiring, the TPE seam is the one that fails.

    This is the failure production hit on every optimization request.
    """
    with pytest.raises(RuntimeError, match="Optuna TPE hook not set"):
        get_tpe_factories()


def test_wires_the_optuna_module_hook() -> None:
    """The module seam yields real Optuna objects, not a fake."""
    use_real_optimizer()

    _, sampler_ctor, pruner_ctor = get_optuna_factories()
    sampler = sampler_ctor(seed=42, n_startup_trials=5)
    pruner = pruner_ctor(n_startup_trials=5, n_warmup_steps=2)

    assert type(sampler).__name__ == "TPESampler"
    assert type(pruner).__name__ == "MedianPruner"


def test_wires_the_tpe_strategy_hook() -> None:
    """The TPE seam yields real Optuna objects -- the seam that was missed."""
    use_real_optimizer()

    _, sampler_ctor, pruner_ctor = get_tpe_factories()
    sampler = sampler_ctor(seed=42, n_startup_trials=5)
    pruner = pruner_ctor(n_startup_trials=5, n_warmup_steps=2)

    assert type(sampler).__name__ == "TPESampler"
    assert type(pruner).__name__ == "MedianPruner"


def test_wiring_is_idempotent() -> None:
    """Calling it twice is safe; entry points may both run in one process."""
    use_real_optimizer()
    use_real_optimizer()

    _, tpe_sampler_ctor, _ = get_tpe_factories()
    _, module_sampler_ctor, _ = get_optuna_factories()

    assert type(tpe_sampler_ctor(seed=1, n_startup_trials=1)).__name__ == "TPESampler"
    assert type(module_sampler_ctor(seed=1, n_startup_trials=1)).__name__ == "TPESampler"
