"""The group ownership decision logic, and the real libc binding where it loads."""

from __future__ import annotations

import ctypes
import errno
import sys
from collections.abc import Sequence

import pytest

from platform_core.process_group import (
    LIBC_NAME,
    SIGKILL,
    LibcGroupApi,
    isolate_in_new_group,
    kill_group,
    libc_group_api,
)
from platform_core.testing_process_group import FakeGroupApi


def test_isolate_is_one_setsid_call() -> None:
    api = FakeGroupApi(session=77)
    assert isolate_in_new_group(api) == ""
    assert api.calls == ["setsid"]


def test_isolate_names_a_refused_setsid_with_its_errno() -> None:
    api = FakeGroupApi(session=-1, code=errno.EPERM)
    assert isolate_in_new_group(api) == f"setsid failed (errno {errno.EPERM})"
    assert api.calls == ["setsid", "errno"]


def test_kill_group_sends_sigkill_to_the_group() -> None:
    api = FakeGroupApi()
    assert kill_group(api, 4242) == ""
    assert api.signalled == [(4242, SIGKILL)]
    assert api.calls == ["killpg"]


def test_kill_group_reads_an_empty_group_as_a_completed_sweep() -> None:
    api = FakeGroupApi(kill_result=-1, code=errno.ESRCH)
    assert kill_group(api, 4242) == ""
    assert api.calls == ["killpg", "errno"]


def test_kill_group_names_any_other_refusal_with_its_errno() -> None:
    api = FakeGroupApi(kill_result=-1, code=errno.EPERM)
    assert kill_group(api, 4242) == f"killpg failed (errno {errno.EPERM})"
    assert api.calls == ["killpg", "errno"]


class FakeExport:
    """One exported function as the binding sees it: signature slots, a call log.

    Args:
        answers: What successive calls return, in order.
    """

    def __init__(self, *answers: int) -> None:
        """Start unbound, with the scripted answers."""
        self.restype: type[ctypes.c_int] | None = None
        self.argtypes: Sequence[type[ctypes.c_int]] = ()
        self.calls: list[tuple[int, ...]] = []
        self._answers = list(answers)

    def __call__(self, *args: int) -> int:
        """Record the call and answer the next scripted value."""
        self.calls.append(args)
        return self._answers.pop(0)


class FakeLibc(ctypes.CDLL):
    """A libc that loads nothing and exports the two names the binding reads.

    THE BINDING, NOT LIBC, IS UNDER TEST HERE. ``LibcGroupApi`` wires two
    exports with their signatures and hands their raw answers through; that
    wiring is the same on every platform, and libc.so.6 exists on one. The
    real-binding test below keeps the POSIX contract where it can run; this
    keeps the wiring covered where it cannot. ``CDLL.__init__`` is
    deliberately not called: it would ``dlopen`` a name, and there is none.
    """

    def __init__(self) -> None:
        """Export the two functions with their scripted answers."""
        self.setsid = FakeExport(5000, -1)
        self.killpg = FakeExport(0, -1)


def test_the_binding_wires_both_exports_and_hands_the_raw_answers_through() -> None:
    library = FakeLibc()
    api = LibcGroupApi(library)

    assert library.setsid.restype is ctypes.c_int
    assert list(library.setsid.argtypes) == []
    assert library.killpg.restype is ctypes.c_int
    assert list(library.killpg.argtypes) == [ctypes.c_int, ctypes.c_int]

    assert api.setsid() == 5000
    assert api.setsid() == -1
    assert library.setsid.calls == [(), ()]

    assert api.killpg(5000, SIGKILL) == 0
    assert api.killpg(5000, SIGKILL) == -1
    assert library.killpg.calls == [(5000, SIGKILL), (5000, SIGKILL)]

    # errno is ctypes' private copy, which nothing here has set.
    ctypes.set_errno(0)
    assert api.last_errno() == 0


@pytest.mark.skipif(sys.platform == "win32", reason=f"{LIBC_NAME} exists off Windows only")
def test_the_real_binding_signals_a_group_and_reports_an_absent_one() -> None:
    api = libc_group_api()
    # This process is already in a group; signalling a pid that cannot
    # exist asks the kernel for a group it has not got, which is ESRCH.
    assert api.killpg(2**22 + 1, 0) == -1
    assert api.last_errno() == errno.ESRCH
    assert kill_group(api, 2**22 + 1) == ""
    # Signal 0 to this process's own group is permission-checked and sent to
    # nobody: a real success from the real libc.
    assert api.killpg(0, 0) == 0


@pytest.mark.skipif(sys.platform != "win32", reason="the refusal is the Windows behaviour")
def test_the_real_binding_cannot_load_on_windows() -> None:
    with pytest.raises(OSError):
        libc_group_api()
