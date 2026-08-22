"""Opt the measuring process out of system-managed power throttling.

Windows classifies a long-running console process as background work and
applies EcoQoS power throttling to it. The demotion lands mid-measurement and
never lifts, so a benchmark that does not opt out reports two different power
regimes as though they were one.

Measured on this workload (LightGBM, 200 trees, 78,682x18, single-threaded,
the identical fit repeated in one process):

    fit 0  0.547s        fit 3  3.794s
    fit 1  0.536s        fit 4  6.496s
    fit 2  0.540s        fit 5  7.108s   <- 13x, and it does not recover
    -- opted out --
    fit 6  0.540s        fit 8  0.491s
    fit 7  0.521s        fit 9  0.503s

RSS (233 MB) and thread count (75) were flat across the whole sequence, and 90
seconds of idle did not restore speed, so this is neither a leak nor thermal
recovery. Opting out restores full speed on the very next fit.

Why this cannot be left to the rotation protocol: the runner rotates arm order
across seeds so no arm systematically occupies the cold-CPU slot, which
cancels a *symmetric* effect. Throttling is a one-way step change part-way
through a run, so whichever arms are measured before the step keep the fast
regime and the rest never see it again. Rotation spreads that damage around
rather than removing it, and the resulting per-arm medians straddle the step
in an order-dependent way.

There is deliberately no fallback. A run that could not opt out is measuring
an unknown mix of two power regimes, and a number nobody can attribute is
worse than no number.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

from .protocols import ProcessInformationSetterProto, SetProcessInformationProto
from .types import ERR_POWER_THROTTLING

#: ``ProcessPowerThrottling`` from ``PROCESS_INFORMATION_CLASS``.
PROCESS_POWER_THROTTLING: int = 4

#: ``PROCESS_POWER_THROTTLING_EXECUTION_SPEED``.
EXECUTION_SPEED: int = 0x1

#: ``PROCESS_POWER_THROTTLING_CURRENT_VERSION``.
STATE_VERSION: int = 1

#: Byte length of ``PROCESS_POWER_THROTTLING_STATE``: three ``ULONG``.
STATE_SIZE: int = 12

#: What ``GetCurrentProcess`` returns: the documented ``(HANDLE)-1``
#: pseudo-handle for the calling process.
CURRENT_PROCESS_PSEUDO_HANDLE: int = -1


class PowerThrottlingState(ctypes.Structure):
    """``PROCESS_POWER_THROTTLING_STATE`` as declared in ``processthreadsapi.h``.

    The pair of masks encodes three distinct requests, and the difference
    between two of them is the whole point of this module:

    * ``ControlMask = 0`` -- the process expresses no preference and Windows
      decides. This is the default, and it is what throttles.
    * ``ControlMask = EXECUTION_SPEED``, ``StateMask = EXECUTION_SPEED`` --
      always throttle.
    * ``ControlMask = EXECUTION_SPEED``, ``StateMask = 0`` -- never throttle.

    Reading the state back therefore does not reveal whether the process is
    *currently* being throttled: a default-managed process reports
    ``StateMask = 0``, identical to one that has explicitly opted out.

    The struct is built and consumed entirely inside
    :func:`win32_process_information_setter`; the injection boundary carries
    the three masks as plain integers so no caller and no test has to reach
    through a ``ctypes`` field descriptor.
    """

    _fields_ = (
        ("Version", wintypes.ULONG),
        ("ControlMask", wintypes.ULONG),
        ("StateMask", wintypes.ULONG),
    )


def win32_process_information_setter(
    version: int,
    control_mask: int,
    state_mask: int,
) -> int:
    """Apply a power-throttling state to the current process via Win32.

    Args:
        version: ``PROCESS_POWER_THROTTLING_STATE.Version``.
        control_mask: Which policies the process is expressing a preference
            about.
        state_mask: The preference itself, for the policies named by
            ``control_mask``.

    Returns:
        The Win32 error code, or ``0`` when the request was accepted. Returned
        rather than raised so the decision to fail belongs to
        :func:`disable_power_throttling`, which owns the error message.
    """
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    # Assigned straight to a Protocol-typed name: the annotation is where the
    # concrete return type comes from, the same pattern the package uses for
    # vendor constructors in `adapters`.
    set_process_information: SetProcessInformationProto = kernel32.SetProcessInformation

    state = PowerThrottlingState(
        Version=version,
        ControlMask=control_mask,
        StateMask=state_mask,
    )
    # Every argument is an explicitly widthed ctypes instance, so no argtypes
    # declaration is needed and nothing is marshalled at a default width. The
    # process handle is the documented pseudo-handle `(HANDLE)-1` that
    # GetCurrentProcess returns, passed directly rather than fetched: one
    # fewer untyped boundary, and it cannot be truncated to 32 bits.
    accepted = set_process_information(
        ctypes.c_void_p(CURRENT_PROCESS_PSEUDO_HANDLE),
        ctypes.c_int(PROCESS_POWER_THROTTLING),
        ctypes.c_void_p(ctypes.addressof(state)),
        ctypes.c_uint32(ctypes.sizeof(state)),
    )
    if accepted != 0:
        return 0
    return ctypes.get_last_error()


def disable_power_throttling(setter: ProcessInformationSetterProto) -> None:
    """Opt the current process out of system-managed power throttling.

    Requests ``ControlMask = EXECUTION_SPEED`` with ``StateMask = 0``, the
    documented encoding for "never throttle this process". Setting both masks
    would request the exact opposite, one bit away, so the encoding is
    asserted in tests rather than left to review.

    Args:
        setter: Applies the state. Injected so the refusal path is reachable
            in tests without altering the host's power state.

    Returns:
        None. The call is made for its effect on the process.

    Raises:
        RuntimeError: Carrying :data:`~covenant_ml.benchmarking.types.ERR_POWER_THROTTLING`
            and the Win32 error code, if the request is refused. Raised rather
            than ignored: continuing would time an unknown mix of two power
            regimes.
    """
    code = setter(STATE_VERSION, EXECUTION_SPEED, 0)
    if code != 0:
        raise RuntimeError(
            f"[{ERR_POWER_THROTTLING}] Could not opt out of process power throttling "
            f"(win32 {code}); fit times would mix two power regimes"
        )


def opt_out_of_power_throttling() -> None:
    """Opt out using the real Win32 boundary.

    The zero-argument shape the runner's injection hook is bound to.

    Returns:
        None. The call is made for its effect on the process.

    Raises:
        RuntimeError: If the platform refuses the request.
    """
    disable_power_throttling(win32_process_information_setter)


__all__ = [
    "CURRENT_PROCESS_PSEUDO_HANDLE",
    "EXECUTION_SPEED",
    "PROCESS_POWER_THROTTLING",
    "STATE_SIZE",
    "STATE_VERSION",
    "PowerThrottlingState",
    "disable_power_throttling",
    "opt_out_of_power_throttling",
    "win32_process_information_setter",
]
