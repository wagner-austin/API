"""Tests for opting the measuring process out of power throttling.

The real Win32 boundary is exercised directly where the subject *is* that call
succeeding on this platform; the refusal path is reached by injecting a setter
through the package's Protocol, not by patching. Nothing here is mocked.
"""

from __future__ import annotations

import ctypes
import os

import pytest

from covenant_ml.benchmarking.power import (
    CURRENT_PROCESS_PSEUDO_HANDLE,
    EXECUTION_SPEED,
    PROCESS_POWER_THROTTLING,
    PROCESS_SET_INFORMATION,
    STATE_SIZE,
    STATE_VERSION,
    PowerThrottlingState,
    disable_power_throttling,
    disable_power_throttling_for,
    opt_out_of_power_throttling,
    opt_process_out_of_power_throttling,
    win32_process_information_setter,
    win32_targeted_process_information_setter,
)
from covenant_ml.benchmarking.types import (
    ERR_POWER_TARGET_REFUSED,
    ERR_POWER_TARGET_UNREACHABLE,
    ERR_POWER_THROTTLING,
)


class RecordingSetter:
    """A setter that records what it was asked for and reports an outcome."""

    def __init__(self, code: int) -> None:
        """Bind the Win32 code this setter will report.

        Args:
            code: Value to return; ``0`` means the request was accepted.
        """
        self._code = code
        self.requests: list[tuple[int, int, int]] = []

    def __call__(self, version: int, control_mask: int, state_mask: int) -> int:
        """Record the requested state and report the bound outcome.

        Args:
            version: Structure version requested.
            control_mask: Policies the caller expressed a preference about.
            state_mask: The preference itself.

        Returns:
            The bound Win32 code.
        """
        self.requests.append((version, control_mask, state_mask))
        return self._code


def test_the_request_is_control_set_and_state_clear() -> None:
    """``ControlMask`` set with ``StateMask`` clear means "never throttle".

    Setting both masks would request *always* throttle -- the exact opposite,
    one bit away -- so the encoding that reaches the system call is asserted
    rather than trusted.
    """
    setter = RecordingSetter(0)
    disable_power_throttling(setter)
    assert setter.requests == [(1, 0x1, 0)]
    assert (STATE_VERSION, EXECUTION_SPEED) == (1, 0x1)


def test_a_refused_request_raises_with_the_win32_code() -> None:
    """No fallback: an unattributable measurement must not proceed."""
    setter = RecordingSetter(87)
    with pytest.raises(RuntimeError, match=ERR_POWER_THROTTLING) as caught:
        disable_power_throttling(setter)
    message = str(caught.value)
    assert "win32 87" in message
    assert "mix two power regimes" in message


def test_an_accepted_request_returns_without_raising() -> None:
    setter = RecordingSetter(0)
    disable_power_throttling(setter)
    assert len(setter.requests) == 1


def test_the_real_win32_boundary_accepts_the_opt_out() -> None:
    """The platform must actually accept this request.

    If Windows ever stops accepting it, every fit time the harness reports
    silently becomes a mix of two power regimes. That regression has to fail
    here rather than surface as unexplained variance in a manifest.
    """
    assert win32_process_information_setter(STATE_VERSION, EXECUTION_SPEED, 0) == 0


def test_the_real_boundary_reports_a_win32_code_for_a_bad_request() -> None:
    """A malformed request must come back as a code, not an exception.

    Version 0 is not a valid ``PROCESS_POWER_THROTTLING_STATE`` version, so
    Windows rejects it. This pins the contract that the boundary reports
    failure by return value, leaving the raise to the caller.
    """
    assert win32_process_information_setter(0, EXECUTION_SPEED, 0) != 0


def test_opting_out_through_the_real_boundary_is_idempotent() -> None:
    """The runner calls this once per run; repeated runs share a process."""
    opt_out_of_power_throttling()
    opt_out_of_power_throttling()


def test_state_struct_is_three_ulongs_wide() -> None:
    """``SetProcessInformation`` validates the buffer length.

    A layout change is otherwise rejected at runtime with an opaque Win32
    error rather than anything pointing at this struct. The field *order* is
    pinned behaviourally instead of by introspection: the real boundary
    accepts the documented encoding and rejects an invalid version, which
    could not both hold if the masks were transposed.
    """
    assert ctypes.sizeof(PowerThrottlingState()) == STATE_SIZE
    assert STATE_SIZE == 12
    assert PROCESS_POWER_THROTTLING == 4
    assert CURRENT_PROCESS_PSEUDO_HANDLE == -1


class RecordingTargetedSetter:
    """A by-pid setter that records requests and reports a two-part outcome."""

    def __init__(self, *, opened: bool, code: int) -> None:
        """Bind the outcome this setter will report.

        Args:
            opened: Whether the target could be opened at all.
            code: Win32 code; ``0`` means the request was accepted.
        """
        self._opened = opened
        self._code = code
        self.requests: list[tuple[int, int, int, int]] = []

    def __call__(
        self, pid: int, version: int, control_mask: int, state_mask: int
    ) -> tuple[bool, int]:
        """Record the requested state and report the bound outcome.

        Args:
            pid: Target process.
            version: Structure version requested.
            control_mask: Policies the caller expressed a preference about.
            state_mask: The preference itself.

        Returns:
            The bound ``(opened, code)`` pair.
        """
        self.requests.append((pid, version, control_mask, state_mask))
        return (self._opened, self._code)


def test_the_by_pid_request_is_the_same_encoding_as_the_current_process_one() -> None:
    """The two forms must not drift apart.

    They encode the same intent -- never throttle -- and a change to one that
    missed the other would leave a fleet where opting a running job out did
    the opposite of opting the caller out.
    """
    current = RecordingSetter(0)
    targeted = RecordingTargetedSetter(opened=True, code=0)

    disable_power_throttling(current)
    disable_power_throttling_for(4321, targeted)

    assert current.requests == [(STATE_VERSION, EXECUTION_SPEED, 0)]
    assert targeted.requests == [(4321, STATE_VERSION, EXECUTION_SPEED, 0)]


def test_an_unreachable_target_names_the_pid_and_its_own_code() -> None:
    """A pid that cannot be opened is a different failure from a refusal.

    The remedy differs -- an exited or other-user process versus a platform
    policy answer -- so the two must not share one error code.
    """
    targeted = RecordingTargetedSetter(opened=False, code=5)
    with pytest.raises(RuntimeError, match=ERR_POWER_TARGET_UNREACHABLE) as caught:
        disable_power_throttling_for(9999, targeted)
    message = str(caught.value)
    assert "process 9999" in message
    assert "win32 5" in message
    assert ERR_POWER_TARGET_REFUSED not in message


def test_a_target_that_refuses_is_reported_as_a_refusal_not_as_unreachable() -> None:
    targeted = RecordingTargetedSetter(opened=True, code=87)
    with pytest.raises(RuntimeError, match=ERR_POWER_TARGET_REFUSED) as caught:
        disable_power_throttling_for(4321, targeted)
    message = str(caught.value)
    assert "win32 87" in message
    assert "mix two power regimes" in message
    assert ERR_POWER_TARGET_UNREACHABLE not in message


def test_an_accepted_by_pid_request_returns_without_raising() -> None:
    targeted = RecordingTargetedSetter(opened=True, code=0)
    disable_power_throttling_for(4321, targeted)
    assert len(targeted.requests) == 1


def test_the_real_by_pid_boundary_opts_out_this_very_process() -> None:
    """Exercised against a real pid, because the subject is the Win32 call.

    Uses our own pid: it is guaranteed to exist, guaranteed to be openable,
    and the effect -- never throttle -- is one the test process wants anyway.
    """
    opened, code = win32_targeted_process_information_setter(
        os.getpid(), STATE_VERSION, EXECUTION_SPEED, 0
    )
    assert (opened, code) == (True, 0)


def test_the_real_by_pid_boundary_reports_an_unopenable_pid_rather_than_raising() -> None:
    """Pid 0 is the System Idle Process and cannot be opened for this right.

    Pins the contract that an unreachable target comes back as
    ``(False, code)`` rather than an exception, leaving the raise to the
    caller that owns the message.
    """
    opened, code = win32_targeted_process_information_setter(0, STATE_VERSION, EXECUTION_SPEED, 0)
    assert opened is False
    assert code != 0


def test_the_real_by_pid_boundary_reports_a_code_for_a_bad_request() -> None:
    """Opened successfully, then refused: the ``(True, code)`` half.

    Version 0 is not a valid state version, so the process opens and the state
    change is rejected -- which is the only way to reach that branch against
    the real boundary.
    """
    opened, code = win32_targeted_process_information_setter(os.getpid(), 0, EXECUTION_SPEED, 0)
    assert opened is True
    assert code != 0


def test_opting_a_process_out_through_the_real_boundary_is_idempotent() -> None:
    opt_process_out_of_power_throttling(os.getpid())
    opt_process_out_of_power_throttling(os.getpid())


def test_only_the_set_information_right_is_requested() -> None:
    """Opening someone else's process asks for one right and no more.

    ``PROCESS_ALL_ACCESS`` would work and would hand back a handle useful for
    reading memory and terminating the target, neither of which this call
    does.
    """
    assert PROCESS_SET_INFORMATION == 0x0200
