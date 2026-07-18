"""Contract enforcement: the ``require`` helper and ``@enforce_contract``.

Two mechanisms, per the Phase 1 spec:

* :func:`require` -- module-level assertion that raises a specific
  :class:`~tankpit_bot.contracts.base.ContractError` subclass with the
  caller's ``file:line`` recorded as the violation site. Used inside
  typed code where the values are already narrowed.
* :func:`enforce_contract` -- decorator that runs a named
  :class:`Contract` check over a function's arguments before the
  function body executes. A contract's ``check`` carries the same
  typed signature as the functions it guards (the protocol is generic
  over a ``ParamSpec``), so enforcement adds no type erasure. The
  guard rule (``scripts/contract_rules.py``) requires this decorator
  on public mutation functions in ``facts/``, ``ledger/`` and
  ``memory/``.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Generic, Protocol, TypeVar

from typing_extensions import ParamSpec

from tankpit_bot.contracts.base import ContractError

P = ParamSpec("P")
R = TypeVar("R")


class Contract(Protocol[P]):
    """A named runtime contract over a callable's arguments.

    Implementations give ``check`` the same typed signature as the
    function the contract guards.
    """

    @property
    def name(self) -> str:
        """Name of the contract (matches the error's ``contract_name``)."""

    def check(self, *args: P.args, **kwargs: P.kwargs) -> None:
        """Validate the arguments of an enforced call.

        Args:
            *args: Positional arguments of the enforced function.
            **kwargs: Keyword arguments of the enforced function.

        Raises:
            ContractError: If the arguments violate the contract.
        """


def require(
    condition: bool,
    error: type[ContractError],
    **details: str,
) -> None:
    """Raise ``error`` with the caller's location unless ``condition`` holds.

    Args:
        condition: The contract condition that must hold.
        error: ContractError subclass to raise on violation.
        **details: Per-violation key/value details.

    Raises:
        ContractError: The given subclass, when ``condition`` is False.
    """
    if condition:
        return
    caller = traceback.extract_stack(limit=2)[0]
    violated_at = f"{Path(caller.filename).name}:{caller.lineno}"
    raise error(violated_at=violated_at, details=dict(details))


class _EnforcedFunction(Generic[P, R]):
    """A callable wrapper that runs a contract check before the call."""

    def __init__(self, contract: Contract[P], fn: Callable[P, R]) -> None:
        """Wrap ``fn`` with ``contract`` enforcement.

        Args:
            contract: Contract to check before every call.
            fn: The enforced function.
        """
        self._contract = contract
        self._fn = fn
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Check the contract, then invoke the enforced function.

        Args:
            *args: Positional arguments of the enforced function.
            **kwargs: Keyword arguments of the enforced function.

        Returns:
            Whatever the enforced function returns.

        Raises:
            ContractError: If the contract check fails.
        """
        self._contract.check(*args, **kwargs)
        return self._fn(*args, **kwargs)


class _ContractDecorator(Generic[P]):
    """Decorator object produced by :func:`enforce_contract`."""

    def __init__(self, contract: Contract[P]) -> None:
        """Hold the contract for a later decoration.

        Args:
            contract: Contract to enforce on the decorated function.
        """
        self._contract = contract

    def __call__(self, fn: Callable[P, R]) -> _EnforcedFunction[P, R]:
        """Decorate ``fn`` with contract enforcement.

        Args:
            fn: Function to enforce the contract on.

        Returns:
            The enforced wrapper.
        """
        return _EnforcedFunction(self._contract, fn)


def enforce_contract(contract: Contract[P]) -> _ContractDecorator[P]:
    """Build a decorator that enforces ``contract`` on a function.

    Args:
        contract: Contract to enforce.

    Returns:
        A decorator wrapping the function with a pre-call check.
    """
    return _ContractDecorator(contract)


__all__ = [
    "Contract",
    "enforce_contract",
    "require",
]
