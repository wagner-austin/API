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

from .protocols import (
    CloseHandleProto,
    OpenProcessProto,
    ProcessInformationSetterProto,
    SetProcessInformationProto,
    TargetedProcessInformationSetterProto,
)
from .types import (
    ERR_POWER_TARGET_REFUSED,
    ERR_POWER_TARGET_UNREACHABLE,
    ERR_POWER_THROTTLING,
)

#: ``ProcessPowerThrottling`` from ``PROCESS_INFORMATION_CLASS``.
PROCESS_POWER_THROTTLING: int = 4

#: ``PROCESS_POWER_THROTTLING_EXECUTION_SPEED``.
EXECUTION_SPEED: int = 0x1

#: ``PROCESS_POWER_THROTTLING_CURRENT_VERSION``.
STATE_VERSION: int = 1

#: Byte length of ``PROCESS_POWER_THROTTLING_STATE``: three ``ULONG``.
STATE_SIZE: int = 12

#: ``PROCESS_SET_INFORMATION``: the single right needed to change another
#: process's power state. Deliberately not ``PROCESS_ALL_ACCESS`` -- this
#: opens someone else's process, and asking for more than the one right the
#: call uses is how a handle becomes useful for something nobody reviewed.
PROCESS_SET_INFORMATION: int = 0x0200

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


def win32_targeted_process_information_setter(
    pid: int,
    version: int,
    control_mask: int,
    state_mask: int,
) -> tuple[bool, int]:
    """Apply a power-throttling state to ANOTHER process, by pid, via Win32.

    The by-pid counterpart of :func:`win32_process_information_setter`. That
    one reaches its target through the ``(HANDLE)-1`` pseudo-handle, which
    names the caller and nothing else; this one must open a real handle, and
    so must close it.

    The handle is closed on BOTH paths. Leaking it on the failure path would
    be the harder bug to find, because the failure it accompanies already
    explains the symptom.

    Args:
        pid: The target process.
        version: ``PROCESS_POWER_THROTTLING_STATE.Version``.
        control_mask: Which policies the target expresses a preference about.
        state_mask: The preference itself, for the policies named by
            ``control_mask``.

    Returns:
        ``(opened, code)`` as ``TargetedProcessInformationSetterProto`` defines
        it: whether the process could be opened, and the Win32 error code or
        ``0`` on acceptance.
    """
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process: OpenProcessProto = kernel32.OpenProcess
    set_process_information: SetProcessInformationProto = kernel32.SetProcessInformation
    close_handle: CloseHandleProto = kernel32.CloseHandle

    handle = open_process(
        ctypes.c_uint32(PROCESS_SET_INFORMATION),
        ctypes.c_int(0),
        ctypes.c_uint32(pid),
    )
    if handle == 0:
        return (False, ctypes.get_last_error())

    state = PowerThrottlingState(
        Version=version,
        ControlMask=control_mask,
        StateMask=state_mask,
    )
    accepted = set_process_information(
        ctypes.c_void_p(handle),
        ctypes.c_int(PROCESS_POWER_THROTTLING),
        ctypes.c_void_p(ctypes.addressof(state)),
        ctypes.c_uint32(ctypes.sizeof(state)),
    )
    code = 0 if accepted != 0 else ctypes.get_last_error()
    close_handle(ctypes.c_void_p(handle))
    return (True, code)


def disable_power_throttling_for(pid: int, setter: TargetedProcessInformationSetterProto) -> None:
    """Opt one already-running process out of power throttling.

    :func:`disable_power_throttling` can only reach the calling process, so it
    cannot help a measurement that is already under way -- which is the case
    that matters most, because the throttle lands part-way through a long run
    and the run is exactly what one does not want to restart. Applied
    successfully to a job hours in, without interrupting it.

    Requests the same ``ControlMask = EXECUTION_SPEED`` with ``StateMask = 0``
    the current-process form does; the two encodings must not drift apart, and
    a test asserts they are identical.

    Args:
        pid: The process to opt out.
        setter: Applies the state. Injected so both refusal paths are
            reachable in tests without altering any real process.

    Returns:
        None. The call is made for its effect on the target.

    Raises:
        RuntimeError: Carrying
            :data:`~covenant_ml.benchmarking.types.ERR_POWER_TARGET_UNREACHABLE`
            when the process could not be opened, or
            :data:`~covenant_ml.benchmarking.types.ERR_POWER_TARGET_REFUSED`
            when it was opened and refused the change. Two codes rather than
            one because the remedies differ: an unreachable pid has usually
            exited or belongs to another user, while a refusal is a platform
            policy answer about a process that is right there.
    """
    opened, code = setter(pid, STATE_VERSION, EXECUTION_SPEED, 0)
    if not opened:
        raise RuntimeError(
            f"[{ERR_POWER_TARGET_UNREACHABLE}] Could not open process {pid} to lift power "
            f"throttling (win32 {code}); it may have exited or belong to another user"
        )
    if code != 0:
        raise RuntimeError(
            f"[{ERR_POWER_TARGET_REFUSED}] Process {pid} refused the power-throttling "
            f"opt-out (win32 {code}); its timings would mix two power regimes"
        )


def opt_process_out_of_power_throttling(pid: int) -> None:
    """Opt one running process out, using the real Win32 boundary.

    The single-argument shape a caller binds when it has a pid rather than
    being the process in question.

    Args:
        pid: The process to opt out.

    Returns:
        None. The call is made for its effect on the target.

    Raises:
        RuntimeError: If the process cannot be opened or refuses the request.
    """
    disable_power_throttling_for(pid, win32_targeted_process_information_setter)


__all__ = [
    "CURRENT_PROCESS_PSEUDO_HANDLE",
    "EXECUTION_SPEED",
    "PROCESS_POWER_THROTTLING",
    "PROCESS_SET_INFORMATION",
    "STATE_SIZE",
    "STATE_VERSION",
    "PowerThrottlingState",
    "disable_power_throttling",
    "disable_power_throttling_for",
    "opt_out_of_power_throttling",
    "opt_process_out_of_power_throttling",
    "win32_process_information_setter",
    "win32_targeted_process_information_setter",
]
