"""The kill-on-close decision logic, and the real kernel32 binding where it loads."""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Sequence

import pytest

from maketools.job import (
    EXPECTED_SIZES,
    EXTENDED_LIMIT_INFORMATION_CLASS,
    KILL_ON_JOB_CLOSE,
    ExtendedLimitInformation,
    Kernel32JobApi,
    join_kill_on_close_job,
    kernel32_job_api,
    kill_on_close_information,
)
from tests.conftest import FakeJobApi


def test_join_creates_sets_and_assigns_in_order() -> None:
    api = FakeJobApi(handle=7)
    assert join_kill_on_close_job(api) == ""
    assert api.calls == ["create", "set", "assign"]
    assert api.handles == [7, 7]


def test_join_names_a_failed_creation() -> None:
    api = FakeJobApi(handle=0)
    assert join_kill_on_close_job(api) == "CreateJobObject returned NULL"
    assert api.calls == ["create"]


def test_join_names_a_refused_limit_with_the_win32_error() -> None:
    api = FakeJobApi(set_ok=False, error=24)
    assert join_kill_on_close_job(api) == "SetInformationJobObject failed (win32 24)"
    assert api.calls == ["create", "set", "error"]


def test_join_names_a_refused_assignment_with_the_win32_error() -> None:
    api = FakeJobApi(assign_ok=False, error=5)
    assert join_kill_on_close_job(api) == "AssignProcessToJobObject failed (win32 5)"
    assert api.calls == ["create", "set", "assign", "error"]


def test_the_structure_has_the_size_the_original_computed_by_hand() -> None:
    information = ExtendedLimitInformation()
    assert ctypes.sizeof(information) == EXPECTED_SIZES[ctypes.sizeof(ctypes.c_void_p())]


def test_the_limit_record_carries_only_the_kill_flag() -> None:
    information = kill_on_close_information()
    flags: int = information.LimitFlags
    assert flags == KILL_ON_JOB_CLOSE
    raw = bytes(information)
    # LimitFlags sits at offset 16 in both layouts; everything else is zero.
    assert raw[16:20] == KILL_ON_JOB_CLOSE.to_bytes(4, "little")
    assert raw[:16] == bytes(16)
    assert raw[20:] == bytes(len(raw) - 20)


class FakeExport:
    """One exported function as the binding sees it: signature slots, a call log.

    Args:
        answers: What successive calls return, in order.
    """

    def __init__(self, *answers: int | None) -> None:
        """Start unbound, with the scripted answers."""
        self.restype: type[ctypes.c_void_p] | type[ctypes.c_int] | type[ctypes.c_uint32] | None = (
            None
        )
        self.argtypes: Sequence[
            type[ctypes.c_void_p]
            | type[ctypes.c_wchar_p]
            | type[ctypes.c_int]
            | type[ctypes.c_uint32]
        ] = ()
        self.calls: list[tuple[int | None, ...]] = []
        self._answers = list(answers)

    def __call__(self, *args: int | None) -> int | None:
        """Record the call and answer the next scripted value."""
        self.calls.append(args)
        return self._answers.pop(0)


class FakeSetInformation(FakeExport):
    """``SetInformationJobObject`` as a fake that copies the record it is handed.

    The binding passes an address and a length; the structure behind them is
    the callee's frame-local, gone once the call returns, so the only moment
    its bytes can be checked is inside the call. Copied here, compared later.
    """

    def __init__(self, *answers: int | None) -> None:
        """Start with the scripted answers and no records."""
        super().__init__(*answers)
        self.records: list[bytes] = []

    def __call__(self, *args: int | None) -> int | None:
        """Copy the record at ``args[2]`` of ``args[3]`` bytes, then answer."""
        address, size = args[2], args[3]
        if address is None or size is None:
            raise AssertionError(f"SetInformationJobObject was handed no record: {args!r}")
        self.records.append(ctypes.string_at(address, size))
        return super().__call__(*args)


class FakeKernel32(ctypes.CDLL):
    """A ``kernel32`` that loads nothing and exports the five names the binding reads.

    THE BINDING, NOT THE KERNEL, IS UNDER TEST HERE. ``Kernel32JobApi`` wires
    five exports with their signatures and turns their raw answers into the
    ``JobApi`` contract; that wiring is the same on every platform, and the
    real kernel32 exists on one. The real-binding test below keeps the Win32
    contract where it can run; this keeps the wiring covered where it cannot.
    ``CDLL.__init__`` is deliberately not called: it would ``dlopen`` a name,
    and there is none to open.
    """

    def __init__(self) -> None:
        """Export the five functions with their scripted answers."""
        self.CreateJobObjectW = FakeExport(42, None)
        self.SetInformationJobObject = FakeSetInformation(1, 0)
        self.AssignProcessToJobObject = FakeExport(1, 0)
        self.GetCurrentProcess = FakeExport(99, 99)
        self.GetLastError = FakeExport(6)


def test_the_binding_wires_every_export_and_maps_the_raw_answers() -> None:
    library = FakeKernel32()
    api = Kernel32JobApi(library)
    pointer_size = ctypes.sizeof(ctypes.c_void_p())

    assert library.CreateJobObjectW.restype is ctypes.c_void_p
    assert list(library.CreateJobObjectW.argtypes) == [ctypes.c_void_p, ctypes.c_wchar_p]
    assert library.SetInformationJobObject.restype is ctypes.c_int
    assert list(library.SetInformationJobObject.argtypes) == [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    assert library.AssignProcessToJobObject.restype is ctypes.c_int
    assert list(library.AssignProcessToJobObject.argtypes) == [ctypes.c_void_p, ctypes.c_void_p]
    assert library.GetCurrentProcess.restype is ctypes.c_void_p
    assert list(library.GetCurrentProcess.argtypes) == []
    assert library.GetLastError.restype is ctypes.c_uint32
    assert list(library.GetLastError.argtypes) == []

    assert api.create_job_object() == 42
    assert api.create_job_object() == 0
    assert library.CreateJobObjectW.calls == [(None, None), (None, None)]

    assert api.set_kill_on_close(42) is True
    assert api.set_kill_on_close(42) is False
    first, second = library.SetInformationJobObject.calls
    assert first[0] == 42
    assert first[1] == EXTENDED_LIMIT_INFORMATION_CLASS
    assert first[3] == EXPECTED_SIZES[pointer_size]
    assert second[0] == 42
    # The address handed over pointed at the kill-on-close record itself.
    expected = bytes(kill_on_close_information())
    assert library.SetInformationJobObject.records == [expected, expected]

    assert api.assign_current_process(42) is True
    assert api.assign_current_process(42) is False
    assert library.GetCurrentProcess.calls == [(), ()]
    assert library.AssignProcessToJobObject.calls == [(42, 99), (42, 99)]

    assert api.last_error() == 6
    assert library.GetLastError.calls == [()]


@pytest.mark.skipif(sys.platform != "win32", reason="kernel32 exists on Windows only")
def test_the_real_binding_joins_this_process_to_a_job() -> None:
    # Joining is harmless here: the handle belongs to this process, and the
    # kill-on-close fires only when the last handle closes, which is exit.
    api = kernel32_job_api()
    assert join_kill_on_close_job(api) == ""
    # The last error after a successful sequence is whatever the kernel left
    # there; the binding's job is to return an integer, not a particular one.
    assert api.last_error() >= 0
    # An invalid handle is refused, and the error is then a real code.
    assert api.set_kill_on_close(0) is False
    assert api.last_error() == 6  # ERROR_INVALID_HANDLE


@pytest.mark.skipif(sys.platform == "win32", reason="the refusal is the off-Windows behaviour")
def test_the_real_binding_cannot_load_off_windows() -> None:
    with pytest.raises(OSError):
        kernel32_job_api()
