"""Optuna module hook mechanism for dependency injection.

Strict typing only: no Any, no casts, no stubs.
Production code sets the hook to real Optuna at startup.
Tests can set a fake implementation.
"""

from __future__ import annotations

from collections.abc import Callable

from ._protocols import (
    OptunaCreateStudyProtocol,
    OptunaMedianPrunerProtocol,
    OptunaModuleProtocol,
    OptunaTPESamplerProtocol,
)

_optuna_module_hook: (
    Callable[
        [],
        tuple[
            OptunaCreateStudyProtocol,
            OptunaTPESamplerProtocol,
            OptunaMedianPrunerProtocol,
        ],
    ]
    | None
) = None


def set_optuna_module_hook(
    hook: Callable[
        [],
        tuple[
            OptunaCreateStudyProtocol,
            OptunaTPESamplerProtocol,
            OptunaMedianPrunerProtocol,
        ],
    ]
    | None,
) -> None:
    """Set hook for Optuna module access.

    Production code sets this to real Optuna at startup.
    Tests can set a fake implementation.

    Args:
        hook: Callable returning (create_study, TPESampler, MedianPruner)
    """
    global _optuna_module_hook
    _optuna_module_hook = hook


def get_optuna_factories() -> tuple[
    OptunaCreateStudyProtocol,
    OptunaTPESamplerProtocol,
    OptunaMedianPrunerProtocol,
]:
    """Get Optuna factories via hook.

    The hook MUST be set before calling this function.

    Returns:
        Tuple of (create_study, TPESampler, MedianPruner) factories.

    Raises:
        RuntimeError: If hook is not set.
    """
    if _optuna_module_hook is None:
        raise RuntimeError(
            "Optuna module hook not set. "
            "Call set_optuna_module_hook() or use_real_optuna() before optimization."
        )
    return _optuna_module_hook()


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


def use_real_optuna() -> None:
    """Set the hook to use real Optuna.

    Call this at application startup before running optimization.
    """
    set_optuna_module_hook(_real_optuna_factories)


__all__ = [
    "get_optuna_factories",
    "set_optuna_module_hook",
    "use_real_optuna",
]
