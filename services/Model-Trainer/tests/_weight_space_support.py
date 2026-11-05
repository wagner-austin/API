"""What the weight-space tests share: typed literal tensors and command runners.

The runners execute a command's real entry point and its real ``__main__``
guard, the two lines a direct ``main()`` call never reaches.

``torch.tensor`` takes its data as ``Any``, so a list literal passed straight
to it is inferred as ``list[Any]`` and refused under ``disallow_any_expr``.
Passing the literal through a parameter with a declared type first is what
makes a hand-written expected value a typed expression, and doing it here
once keeps every test file from carrying its own copy.
"""

from __future__ import annotations

import runpy
import sys
from collections.abc import Callable

import pytest
import torch


def floats(values: list[float] | list[list[float]], dtype: torch.dtype) -> torch.Tensor:
    """Build a floating tensor from hand-written numbers.

    Args:
        values: A vector or a matrix.
        dtype: ``torch.float32`` or ``torch.float64``.

    Returns:
        The tensor.
    """
    return torch.tensor(values, dtype=dtype)


def longs(values: list[int] | list[list[int]]) -> torch.Tensor:
    """Build an integer tensor, of actions or token ids.

    Args:
        values: A vector or a matrix.

    Returns:
        The ``torch.long`` tensor.
    """
    return torch.tensor(values, dtype=torch.long)


def run_entrypoint(entry: Callable[[], None], argv: list[str]) -> str | int | None:
    """Run a console-script entry point under the given arguments.

    Args:
        entry: The command's ``entrypoint``, which always raises ``SystemExit``.
        argv: The arguments, without the program name.

    Returns:
        The exit code it raised.
    """
    saved = sys.argv
    sys.argv = ["entrypoint", *argv]
    try:
        with pytest.raises(SystemExit) as raised:
            entry()
    finally:
        sys.argv = saved
    return raised.value.code


def run_as_module(module_name: str, argv: list[str]) -> str | int | None:
    """Run a command the way ``python -m`` does, through its ``__main__`` guard.

    Without the guard the module imports, runs nothing and exits 0, which is
    indistinguishable from a successful run; this executes the guard itself.

    Args:
        module_name: The command's dotted module path.
        argv: The arguments, without the program name.

    Returns:
        The exit code it raised.
    """
    saved_argv = sys.argv
    saved_module = sys.modules.pop(module_name, None)
    sys.argv = [module_name, *argv]
    try:
        with pytest.raises(SystemExit) as raised:
            runpy.run_module(module_name, run_name="__main__", alter_sys=False)
    finally:
        sys.argv = saved_argv
        if saved_module is not None:
            sys.modules[module_name] = saved_module
    return raised.value.code


__all__ = ["floats", "longs", "run_as_module", "run_entrypoint"]
