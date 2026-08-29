"""Module entry point for running as `python -m scripts.optimize`.

PINS BEFORE IT IMPORTS, and the order is the whole point. The BLAS thread
count decides how a reduction is partitioned, so it decides the arithmetic of
every fit this package performs -- and the variables are read once, when numpy
loads. A pin written after that is a pin nobody reads.

This is why ``scripts/optimize/__init__`` holds no re-exports: it is imported
before this module, and while it pulled ``runner`` it pulled numpy with it,
putting the load above any line this file could execute. The fix was not
missing from here; it was unreachable from here.

Until 2026-08-29 this entry point pinned nothing at all, so every
hyperparameter search it ran took its arithmetic from whatever thread count
the shell happened to inherit. The five benchmark entry points beside it had
already been fixed.
"""

from __future__ import annotations

import os
from collections.abc import Container, Sequence
from typing import Protocol

from platform_core.determinism_cpu import apply_cpu_determinism
from platform_core.determinism_env import SINGLE_THREAD
from platform_core.determinism_record import DeterminismRecord

# NOTHING FROM scripts.optimize IS IMPORTED AT MODULE SCOPE. `runner` pulls
# covenant_ml, which pulls numpy, and an import here would put that load above
# the pin in the one file whose job is to be above it.


class PinProtocol(Protocol):
    """Protocol for pinning this process's CPU reduction order."""

    def __call__(self) -> DeterminismRecord:
        """Pin the thread count and report what was pinned.

        Returns:
            The record naming every variable that was set.
        """
        ...


def pin(modules: Container[str] | None = None) -> DeterminismRecord:
    """Pin this process's CPU reduction order before any numeric import.

    Args:
        modules: The importer's module table, defaulting to ``sys.modules``.
            Injected only so a test can state that the natives are absent
            without unloading them from the worker it shares.

    Returns:
        The record naming every thread variable that was set.

    Raises:
        NativeLibrariesAlreadyLoadedError: When a native numeric library is
            already imported, so the write cannot take effect. Propagated
            rather than softened: a search that reports a pinned posture it
            does not have is worse than one that refuses to start, because
            its numbers look reproducible and are not.
    """
    return apply_cpu_determinism(os.putenv, SINGLE_THREAD, modules)


def run(argv: Sequence[str] | None = None, pin_cpu: PinProtocol = pin) -> int:
    """Pin, then run the optimizer.

    Args:
        argv: Command-line arguments, or None to read the process arguments.
        pin_cpu: How to pin. Injected for the same reason the benchmark entry
            points inject theirs: a test process has already imported numpy,
            so the real pin correctly refuses there, and asserting that
            refusal is a different test from asserting what a pinned run
            does.

    Returns:
        Process exit code.
    """
    pin_cpu()

    # Imported AFTER the pin. Nothing from this package or from covenant_ml
    # may appear at module scope above this line.
    from scripts.optimize.main import main

    return main(argv)


if __name__ == "__main__":
    raise SystemExit(run())
