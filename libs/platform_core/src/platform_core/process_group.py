"""POSIX process-group ownership for a process that will spawn workers.

:mod:`platform_core.process_job` gives a Windows process a kill-on-close
job so its whole tree dies with it. POSIX has no such object; what it has
is the process group. A process that calls ``setsid`` before creating any
child becomes the leader of a new session and a new group, and every
descendant inherits the group id. The owner that later kills the leader
then sends ``SIGKILL`` to the group and every surviving member goes with
it, whoever its OS parent became in the meantime.

The members this exists for are PyTorch ``DataLoader`` workers. A worker's
``ManagerWatchdog`` watches its OS parent and exits when that parent dies,
so killing the candidate that made the workers used to leave them behind
only when a forkserver sat between them; with the candidate as their parent
they exit on their own, and the group sweep is what removes the ones that
have not yet noticed. Board task 49899fd2 measured the leak this closes.

The calls are reached through :class:`GroupApi` so the decision logic is
exercised on the platform that has no libc; :func:`libc_group_api` is the
real binding, bound with ``ctypes`` rather than through :mod:`os` because
the type checker's :mod:`os` has no ``setsid`` or ``killpg`` on Windows,
and it is tested only where it can load.
"""

from __future__ import annotations

import ctypes
import errno
from typing import Final, Protocol

#: ``SIGKILL``, the one signal a group sweep sends.
SIGKILL: Final[int] = 9

#: The libc a Linux process links; Windows has no file by this name.
LIBC_NAME: Final[str] = "libc.so.6"


class GroupApi(Protocol):
    """The two libc calls group ownership needs, and the errno they leave."""

    def setsid(self) -> int:
        """Start a new session and group with this process as leader.

        Returns:
            The new session id, or -1 when the call failed.
        """
        ...

    def killpg(self, pgid: int, signal: int) -> int:
        """Send a signal to every member of a group.

        Args:
            pgid: The group.
            signal: The signal number.

        Returns:
            0 on success, -1 when the call failed.
        """
        ...

    def last_errno(self) -> int:
        """Read the errno the last call left.

        Returns:
            The code.
        """
        ...


def isolate_in_new_group(api: GroupApi) -> str:
    """Make this process the leader of its own session and group.

    Args:
        api: The libc calls.

    Returns:
        The empty string on success, otherwise why it failed, naming the
        call and the errno.
    """
    if api.setsid() == -1:
        return f"setsid failed (errno {api.last_errno()})"
    return ""


def kill_group(api: GroupApi, pgid: int) -> str:
    """Send ``SIGKILL`` to every member of a group.

    A group nobody belongs to any more answers ``ESRCH``; that is a completed
    sweep, not a failure, because the members this removes are the ones that
    outlived their leader and there are none.

    Args:
        api: The libc calls.
        pgid: The group, which is the leader's pid after ``setsid``.

    Returns:
        The empty string when every member is signalled or none remains,
        otherwise why it failed, naming the call and the errno.
    """
    if api.killpg(pgid, SIGKILL) == 0:
        return ""
    code = api.last_errno()
    if code == errno.ESRCH:
        return ""
    return f"killpg failed (errno {code})"


class LibcGroupApi:
    """The real binding, over libc.

    Loaded with ``use_errno=True`` so the errno a failed call leaves is read
    from ctypes' private copy, which the interpreter's own calls between the
    failure and the read cannot overwrite.
    """

    def __init__(self, library: ctypes.CDLL) -> None:
        """Bind the two functions with their signatures.

        Args:
            library: libc.
        """
        self._setsid = library.setsid
        self._setsid.restype = ctypes.c_int
        self._setsid.argtypes = []
        self._killpg = library.killpg
        self._killpg.restype = ctypes.c_int
        self._killpg.argtypes = [ctypes.c_int, ctypes.c_int]

    def setsid(self) -> int:
        """Start a new session and group with this process as leader.

        Returns:
            The new session id, or -1 when the call failed.
        """
        result: int = self._setsid()
        return result

    def killpg(self, pgid: int, signal: int) -> int:
        """Send a signal to every member of a group.

        Args:
            pgid: The group.
            signal: The signal number.

        Returns:
            0 on success, -1 when the call failed.
        """
        result: int = self._killpg(pgid, signal)
        return result

    def last_errno(self) -> int:
        """Read the errno the last call left.

        Returns:
            The code.
        """
        return ctypes.get_errno()


def libc_group_api() -> GroupApi:
    """Load libc and bind it.

    Returns:
        The real API.

    Raises:
        OSError: Off Linux, where there is no ``libc.so.6`` to load; a caller
            on Windows asks :mod:`platform_core.process_job` instead.
    """
    return LibcGroupApi(ctypes.CDLL(LIBC_NAME, use_errno=True))


__all__ = [
    "LIBC_NAME",
    "SIGKILL",
    "GroupApi",
    "LibcGroupApi",
    "isolate_in_new_group",
    "kill_group",
    "libc_group_api",
]
