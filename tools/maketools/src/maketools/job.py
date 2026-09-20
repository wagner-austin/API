"""The Windows kill-on-close job object the launcher joins before pytest.

The launcher assigns ITSELF to a job object with
``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``. Every descendant -- poetry, pytest,
and all execnet workers -- inherits membership, so when this process dies
for ANY reason the OS tears the whole tree down.

This is the part a ``finally`` block cannot do. On 2026-08-18 three ``make
check`` runs left 101 live processes holding ~112 GB of commit for 23 hours;
the launching shell was already dead while make, poetry, pytest and 92
workers stayed alive, so a finally-block reaper would never have run.
Verified: with self-assignment, force-killing the launcher killed
grandchildren two levels down, 3 of 3, with no finally involved.

Assigning SELF rather than the child is deliberate: it removes the window
between spawning a child and assigning it, during which a grandchild could
escape the job.

KNOWN GAP, deliberately left open: this process surviving while the process
that LAUNCHED it is killed. Such a run is above the job, so nothing tears it
down; it runs to completion with nobody reading the output. Per-test
timeouts bound the cost.

THE STRUCTURE IS DECLARED, NOT SIZED BY HAND. The PowerShell original
computed ``JOBOBJECT_EXTENDED_LIMIT_INFORMATION``'s size as 144 or 112 by
pointer width, because GNU make is a 32-bit binary and resolves
``powershell.exe`` to the 32-bit shell; a hard-coded 144 failed on every
``make check`` while succeeding in any hand-run shell. ``ctypes`` lays the
structure out for whichever interpreter is running, so the size is right by
construction on both.

The OS calls are reached through :class:`JobApi` so the decision logic is
exercised on the platform that has no ``kernel32``; :func:`kernel32_job_api`
is the real binding and is tested only where it can load.
"""

from __future__ import annotations

import ctypes
from typing import Final, Protocol

#: ``JobObjectExtendedLimitInformation``.
EXTENDED_LIMIT_INFORMATION_CLASS: Final[int] = 9

#: ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``.
KILL_ON_JOB_CLOSE: Final[int] = 0x2000


class JobApi(Protocol):
    """The four kernel calls a kill-on-close self-assignment needs."""

    def create_job_object(self) -> int:
        """Create an anonymous job.

        Returns:
            The handle, or 0 when the call failed.
        """
        ...

    def set_kill_on_close(self, handle: int) -> bool:
        """Set ``KILL_ON_JOB_CLOSE`` on a job.

        Args:
            handle: The job.

        Returns:
            Whether the call succeeded.
        """
        ...

    def assign_current_process(self, handle: int) -> bool:
        """Put this process in a job.

        Args:
            handle: The job.

        Returns:
            Whether the call succeeded.
        """
        ...

    def last_error(self) -> int:
        """Read the last Win32 error.

        Returns:
            The code.
        """
        ...


def join_kill_on_close_job(api: JobApi) -> str:
    """Create a kill-on-close job and put this process in it.

    Args:
        api: The kernel calls.

    Returns:
        The empty string on success, otherwise why it failed, naming the
        call and the Win32 error.
    """
    handle = api.create_job_object()
    if handle == 0:
        return "CreateJobObject returned NULL"
    if not api.set_kill_on_close(handle):
        return f"SetInformationJobObject failed (win32 {api.last_error()})"
    if not api.assign_current_process(handle):
        return f"AssignProcessToJobObject failed (win32 {api.last_error()})"
    return ""


class ExtendedLimitInformation(ctypes.Structure):
    """``JOBOBJECT_EXTENDED_LIMIT_INFORMATION``, flattened.

    The Windows declaration nests a ``JOBOBJECT_BASIC_LIMIT_INFORMATION`` and
    an ``IO_COUNTERS``; the members are listed here in the same order with
    the same primitive types, which ctypes lays out to the same offsets
    (each nested record begins on an 8-byte boundary, and so does the first
    8-byte member after the 4-byte ``SchedulingClass`` here). Flat because
    the type checker cannot see through a nested Structure field.

    Sizes by pointer width, which the layout reproduces: 144 on 64-bit, 112
    on 32-bit.
    """

    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


#: The structure's size by pointer width, the figure the PowerShell original
#: computed by hand; a test asserts ctypes agrees on whichever interpreter runs.
EXPECTED_SIZES: Final[dict[int, int]] = {8: 144, 4: 112}


def kill_on_close_information() -> ExtendedLimitInformation:
    """Build the limit record with only ``KILL_ON_JOB_CLOSE`` set.

    Returns:
        The zeroed structure with that flag.
    """
    return ExtendedLimitInformation(LimitFlags=KILL_ON_JOB_CLOSE)


class Kernel32JobApi:
    """The real binding, over ``kernel32``.

    Loaded through ``ctypes.CDLL`` rather than ``WinDLL`` because the latter
    name does not exist on the platforms the type checker also runs on; on
    64-bit Windows there is one calling convention and the two are the same.
    """

    def __init__(self, library: ctypes.CDLL) -> None:
        """Bind the four functions with their signatures.

        Args:
            library: ``kernel32``.
        """
        self._create = library.CreateJobObjectW
        self._create.restype = ctypes.c_void_p
        self._create.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        self._set_information = library.SetInformationJobObject
        self._set_information.restype = ctypes.c_int
        self._set_information.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        self._assign = library.AssignProcessToJobObject
        self._assign.restype = ctypes.c_int
        self._assign.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._current_process = library.GetCurrentProcess
        self._current_process.restype = ctypes.c_void_p
        self._current_process.argtypes = []
        self._get_last_error = library.GetLastError
        self._get_last_error.restype = ctypes.c_uint32
        self._get_last_error.argtypes = []

    def create_job_object(self) -> int:
        """Create an anonymous job.

        Returns:
            The handle, or 0 on failure (a NULL pointer comes back as None).
        """
        handle: int | None = self._create(None, None)
        return 0 if handle is None else handle

    def set_kill_on_close(self, handle: int) -> bool:
        """Set ``KILL_ON_JOB_CLOSE`` on a job.

        Args:
            handle: The job.

        Returns:
            Whether the call succeeded.
        """
        information = kill_on_close_information()
        # The address rather than ``byref``: the parameter is declared a void
        # pointer, an address is one, and it keeps every argument an integer
        # or None, which is what lets a fake library record the call exactly.
        # ``information`` outlives the call because it is this frame's local.
        result: int = self._set_information(
            handle,
            EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.addressof(information),
            ctypes.sizeof(information),
        )
        return result != 0

    def assign_current_process(self, handle: int) -> bool:
        """Put this process in a job.

        Args:
            handle: The job.

        Returns:
            Whether the call succeeded.
        """
        current: int | None = self._current_process()
        result: int = self._assign(handle, current)
        return result != 0

    def last_error(self) -> int:
        """Read the last Win32 error.

        Returns:
            The code.
        """
        code: int = self._get_last_error()
        return code


def kernel32_job_api() -> JobApi:
    """Load ``kernel32`` and bind it.

    Returns:
        The real API.

    Raises:
        OSError: Off Windows, where there is no ``kernel32`` to load; the
            launcher never asks for it there.
    """
    return Kernel32JobApi(ctypes.CDLL("kernel32"))


__all__ = [
    "EXPECTED_SIZES",
    "EXTENDED_LIMIT_INFORMATION_CLASS",
    "KILL_ON_JOB_CLOSE",
    "ExtendedLimitInformation",
    "JobApi",
    "Kernel32JobApi",
    "join_kill_on_close_job",
    "kernel32_job_api",
    "kill_on_close_information",
]
