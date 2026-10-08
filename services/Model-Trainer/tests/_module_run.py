"""Run a command-line module the way ``python -m`` does, in one place.

The suite's CLI tests each proved the same thing by hand: put the command
line in ``sys.argv``, drop the module from ``sys.modules`` so its
``__main__`` guard runs on a fresh execution, run it under
:func:`runpy.run_module` and read the code ``SystemExit`` carried. Every copy
saved and restored the same two pieces of process state, so this is that
block once.

Why ``python -m`` is the form worth executing: HPC3 jobs 55595084 and
55595086 each "succeeded" in six seconds having written no record, because a
module without its ``__main__`` guard is imported, runs nothing and exits 0.
Only executing the module as ``__main__`` and demanding its output catches
that, and that same execution also runs the console ``entrypoint()`` the
guard calls and the ``main()`` that reads the process arguments. So a suite
that runs a measurement this way once has exercised all three invocation
forms on one walk.
"""

from __future__ import annotations

import runpy
import sys
from collections.abc import Sequence

import pytest


def run_module_as_main(module_name: str, argv: Sequence[str]) -> int | str | None:
    """Execute a module as ``__main__`` with the given command line.

    Args:
        module_name: Dotted name of the module, as ``python -m`` takes it.
        argv: The arguments after the program name.

    Returns:
        The code the module's ``SystemExit`` carried.

    Raises:
        pytest.fail.Exception: When the module ran to the end without raising
            ``SystemExit``, which is what a missing ``__main__`` guard does.
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
