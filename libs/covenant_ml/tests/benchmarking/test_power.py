"""Tests for opting the measuring process out of power throttling.

The real Win32 boundary is exercised directly where the subject *is* that call
succeeding on this platform; the refusal path is reached by injecting a setter
through the package's Protocol, not by patching. Nothing here is mocked.
"""

from __future__ import annotations

import ctypes

import pytest

from covenant_ml.benchmarking.power import (
    CURRENT_PROCESS_PSEUDO_HANDLE,
    EXECUTION_SPEED,
    PROCESS_POWER_THROTTLING,
    STATE_SIZE,
    STATE_VERSION,
    PowerThrottlingState,
    disable_power_throttling,
    opt_out_of_power_throttling,
    win32_process_information_setter,
)
from covenant_ml.benchmarking.types import ERR_POWER_THROTTLING


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
