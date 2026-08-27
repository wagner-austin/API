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

import os
import time

from platform_core.environment_record import (
    HostProbe,
    VersionReader,
    stdlib_host_probe,
)
from platform_core.environment_record import (
    installed_version as _installed_version,
)

from .power import opt_out_of_power_throttling
from .protocols import HostProbeProto, MonotonicClockProto, PowerThrottlingOptOutProto

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


def _default_host_probe() -> HostProbe:
    """Build the probe that reads this machine.

    ``os.cpu_count`` is injected rather than read inside the probe so the
    arm that refuses a machine reporting no count stays reachable.

    Returns:
        The stdlib probe.
    """
    return stdlib_host_probe(os.cpu_count)


#: Reader for the machine a benchmark ran on. Tests rebind it so a
#: fingerprint assertion does not depend on whose box ran the suite.
host_probe: HostProbeProto = _default_host_probe

#: Reader for one distribution's installed version. Propagates on a
#: missing distribution rather than softening to a sentinel, because a
#: sentinel is a non-empty string that would compare EQUAL between two
#: environments that each failed to find the library.
installed_version: VersionReader = _installed_version


__all__ = [
    "host_probe",
    "installed_version",
    "monotonic_clock",
    "power_throttling_opt_out",
]
