"""The optuna factories the backend optimizers build their study from.

Strict typing only: no Any, no casts, no stubs.

The hook is bound to the real optuna import, so a caller reaches optuna
without wiring anything first. Tests rebind ``optuna_factories`` and restore
it afterwards; read it through the module rather than importing the name, so
the rebinding is visible at the call site.
"""

from __future__ import annotations

from typing import Protocol

from ._protocols import (
    OptunaCreateStudyProtocol,
    OptunaMedianPrunerProtocol,
    OptunaModuleProtocol,
    OptunaTPESamplerProtocol,
)


class OptunaFactoriesProtocol(Protocol):
    """Protocol for the optuna factory provider."""

    def __call__(
        self,
    ) -> tuple[
        OptunaCreateStudyProtocol,
        OptunaTPESamplerProtocol,
        OptunaMedianPrunerProtocol,
    ]:
        """Provide the optuna entry points an optimizer needs.

        Returns:
            Tuple of (create_study, TPESampler, MedianPruner) factories.
        """
        ...


def _real_optuna_factories() -> tuple[
    OptunaCreateStudyProtocol,
    OptunaTPESamplerProtocol,
    OptunaMedianPrunerProtocol,
]:
    """Get real Optuna factories via dynamic import.

    Uses __import__ with Protocol type assignment to avoid Any types.

    Returns:
        Tuple of (create_study, TPESampler, MedianPruner) factories.
    """
    optuna_mod: OptunaModuleProtocol = __import__("optuna")
    create_study: OptunaCreateStudyProtocol = optuna_mod.create_study

    samplers_submod = __import__("optuna.samplers", fromlist=["TPESampler"])
    tpe_sampler: OptunaTPESamplerProtocol = samplers_submod.TPESampler

    pruners_submod = __import__("optuna.pruners", fromlist=["MedianPruner"])
    median_pruner: OptunaMedianPrunerProtocol = pruners_submod.MedianPruner

    return create_study, tpe_sampler, median_pruner


optuna_factories: OptunaFactoriesProtocol = _real_optuna_factories


__all__ = [
    "OptunaFactoriesProtocol",
    "optuna_factories",
]
