"""Internal dependency-injection hooks for the growth-policy package.

Hooks are bound to their real implementations at import time, so production
code calls them directly with no conditional and no optional indirection.
Tests rebind an attribute to a real alternative, run, and restore it.

This module is private (underscore prefix) and is not part of the package's
public surface.

Usage in tests:
    from covenant_ml.growth_policy import _test_hooks

    previous = _test_hooks.monotonic_clock
    _test_hooks.monotonic_clock = StepClock([0.0, 1.0])
    try:
        ...
    finally:
        _test_hooks.monotonic_clock = previous
"""

from __future__ import annotations

import time

from ..benchmarking.protocols import MonotonicClockProto

#: Clock bracketing each timed fit. Bound to :func:`time.perf_counter`, the
#: highest-resolution monotonic counter available, so a timing cannot be moved
#: by a wall-clock adjustment mid-run. Tests rebind this to drive the timing
#: logic from a fixed sequence rather than from real elapsed time.
monotonic_clock: MonotonicClockProto = time.perf_counter


__all__ = ["monotonic_clock"]
