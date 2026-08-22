"""Internal dependency-injection hooks for the benchmarking package.

Hooks are bound to their real implementations at import time, so production
code calls them directly with no conditional and no optional indirection.
Tests rebind an attribute to a fake, run, and restore it.

This module is private (underscore prefix) and is not part of the package's
public surface.

Usage in tests:
    from covenant_ml.benchmarking import _test_hooks

    previous = _test_hooks.monotonic_clock
    _test_hooks.monotonic_clock = FakeClock([0.0, 1.0])
    try:
        ...
    finally:
        _test_hooks.monotonic_clock = previous
"""

from __future__ import annotations

import time

from .power import opt_out_of_power_throttling
from .protocols import MonotonicClockProto, PowerThrottlingOptOutProto

#: Clock used to bracket each timed fit. Bound to :func:`time.perf_counter`,
#: the highest-resolution monotonic counter available, so timings are immune
#: to wall-clock adjustments. Tests rebind this to drive the timing logic from
#: a fixed sequence.
monotonic_clock: MonotonicClockProto = time.perf_counter

#: Opt-out from system-managed power throttling, requested once per run before
#: any fit is timed. Bound to the real Win32 call; tests rebind it so the
#: suite never alters the host's power state and can assert the request was
#: made rather than inferring it from a wall-clock effect.
power_throttling_opt_out: PowerThrottlingOptOutProto = opt_out_of_power_throttling


__all__ = ["monotonic_clock", "power_throttling_opt_out"]
