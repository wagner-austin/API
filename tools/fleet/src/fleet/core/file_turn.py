"""A turn held across processes and threads by an exclusive ``flock`` on a lock file.

Lifted out of :mod:`fleet.core.host_claims` (MCPs board task a85ef09e), where
it was written for a host's claim turn, when the lease file needed the same
thing (MCPs board task b1089284): every node runner is its own process on the
hub, and a file several of them read and rewrite needs a turn none of
``threading``'s locks can give.

THE TURN IS AN EXCLUSIVE ``flock`` ON A FILE THAT STAYS (MCPs board task
5280a25d). Until then it was the lock file's exclusive creation, given back
by removing it, and a waiter looked again on a clock. The hub's records,
mirrors and their locks live on emerald's NFS export, where creating a file
and removing it each wait for the server's disk: measured on lacrymal on
2026-11-06, a turn of create, write, read and remove took a median 56 ms and
up to 477 ms with nobody else waiting, against 2.9 ms to open the file, take
its ``flock`` and close it, and as the push gates of MCPs 4a35dc099 and
ad145c288 launched their jobs on every node at once (02:55Z and 03:33Z) the
node runners logged lease turns taken after 15 s and append turns after 11 s,
all of it between a job's claim and its first phase, which that task holds
to 10 s.

A WAITER SLEEPS IN THE KERNEL, which wakes it when the holder lets go:
measured 1.7 ms after the release on that mount, for a thread waiting behind
another thread of its own process too, since a ``flock`` belongs to the open
file and not to the process. Nothing looks again on a clock.

A HOLDER THAT ENDS LETS GO AS IT ENDS. The kernel releases a ``flock`` when
the last descriptor of its open file closes, which a process's exit does,
SIGKILL included, and an NFS server ends the locks of a client that stops
renewing its lease. So no lock is aged, no holder's pid is asked about, and
no turn is taken from a live holder: what a holder runs inside its turn is
bounded by that work's own deadlines (MCPs board task 86f70457 is the stall
of 2026-10-22 that the age bound and the pid probe answered until then).

A HOLDER MAY REMOVE THE LOCK FILE INSIDE ITS TURN, as a run's retire turn
does once it is over (:mod:`fleet.core.retire_owed`), so one file per run
does not stay behind. A waiter woken holding the lock of a file no longer at
its path holds nothing another turn can see, so it takes the path's file
again (:func:`_still_at`).
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import pathlib
import time
from collections.abc import Generator
from typing import Final, TypedDict

from platform_core.logging import get_logger

from fleet.core import _test_hooks

_log = get_logger(__name__)

#: The mode a lock file is made with when no turn has made it yet.
LOCK_FILE_MODE: Final = 0o644


class TurnBounds(TypedDict):
    """What one kind of turn is called in a log line.

    Attributes:
        what: The turn's name, e.g. ``claim turn``.
        waited_for: Who a wait was for, e.g. ``another process claiming for its host``.
    """

    what: str
    waited_for: str


@contextlib.contextmanager
def held(lock: pathlib.Path, *, holder: str, bounds: TurnBounds) -> Generator[None, None, None]:
    """Hold the turn for the span of the ``with`` block.

    Waits in the kernel while another turn holds the lock, and gives it back
    on leaving the block however it is left: closing the descriptor releases
    its ``flock``.

    Args:
        lock: The lock file, made when absent and never removed here.
        holder: Who takes the turn, named in the line logged after a wait.
        bounds: The turn's names.

    Yields:
        Nothing; the turn is held while the block runs.

    Raises:
        OSError: When the lock file can be neither opened nor locked.
    """
    descriptor = _take(lock, holder=holder, bounds=bounds)
    try:
        yield
    finally:
        os.close(descriptor)


def _take(lock: pathlib.Path, *, holder: str, bounds: TurnBounds) -> int:
    """Lock the file at the path, waiting for the turn that holds it.

    Args:
        lock: The lock file.
        holder: Who takes the turn.
        bounds: The turn's names.

    Returns:
        The descriptor whose ``flock`` is the turn.
    """
    _test_hooks.make_directory(lock.parent)
    asked = time.monotonic()
    waited = False
    while True:
        descriptor = os.open(lock, os.O_RDWR | os.O_CREAT, LOCK_FILE_MODE)
        kept = False
        try:
            waited = _locked(descriptor) or waited
            kept = _still_at(lock, descriptor)
        finally:
            if not kept:
                os.close(descriptor)
        if kept:
            break
    if waited:
        _log.info(
            "%s took its %s after waiting %d ms for %s",
            holder,
            bounds["what"],
            round((time.monotonic() - asked) * 1000),
            bounds["waited_for"],
        )
    return descriptor


def _locked(descriptor: int) -> bool:
    """Take the descriptor's exclusive ``flock``, sleeping while another holds it.

    Args:
        descriptor: An open descriptor of the lock file.

    Returns:
        Whether another turn held it when asked, so this one waited.
    """
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        return True
    return False


def _still_at(lock: pathlib.Path, descriptor: int) -> bool:
    """Whether the locked file is the one at the path now.

    A holder that removed the file inside its turn leaves every waiter
    already woken on it, or opened before the removal, holding a file no
    later turn opens.

    Args:
        lock: The lock file's path.
        descriptor: The locked descriptor.

    Returns:
        True when the path names the locked file.
    """
    opened = os.fstat(descriptor)
    try:
        named = os.stat(lock)
    except FileNotFoundError:
        return False
    return (opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino)


__all__ = [
    "LOCK_FILE_MODE",
    "TurnBounds",
    "held",
]
