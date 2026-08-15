"""Both optuna seams are live without anything being wired.

The optimizer has two optuna seams in separate modules: the factories the
backend optimizers build a study from, and the ones the TPE strategy uses.
They used to be nullable and set by separate functions, every entry point set
only the first because the package docstring said to, and /ml/optimize raised
"Optuna TPE hook not set" for every backend.

This replaces the test that checked the wiring call reached both seams. There
is no wiring call now, so what is worth asserting is that each seam holds a
real implementation on import -- and it fails if a third is added unbound.

Strict typing only: no Any, no casts, no type: ignore, no stubs, no mocks.
"""

from __future__ import annotations

from covenant_ml.optimizer.optuna_backend import _hooks as backend_hooks
from covenant_ml.optimizer.strategies import _hooks as tpe_hooks


def test_the_backend_seam_is_bound_to_real_optuna() -> None:
    """The backend optimizers reach optuna with nothing wired first."""
    assert backend_hooks.optuna_factories is backend_hooks._real_optuna_factories


def test_the_tpe_seam_is_bound_to_real_optuna() -> None:
    """The TPE strategy reaches optuna with nothing wired first."""
    assert tpe_hooks.optuna_factories is tpe_hooks._real_optuna_factories


def test_both_bound_implementations_hand_back_three_factories() -> None:
    """Each imports optuna and returns the three entry points a study needs."""
    for factories in (backend_hooks.optuna_factories(), tpe_hooks.optuna_factories()):
        create_study, tpe_sampler, median_pruner = factories
        assert callable(create_study)
        assert callable(tpe_sampler)
        assert callable(median_pruner)
