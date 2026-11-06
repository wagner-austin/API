"""Another process's turn in a test: a ``flock`` on its own open file (MCPs task 5280a25d).

A turn is an exclusive ``flock`` on a lock file (:mod:`fleet.core.file_turn`),
and a ``flock`` belongs to the open file, so a descriptor this test opens and
locks excludes every turn the code under test takes exactly as another
runner's process would. :class:`OtherTurn` holds one from its construction
and lets it go on a thread of its own after a stated span, having done its
own work first, as the other process's turn would.
"""

from __future__ import annotations

import fcntl
import os
import pathlib
import threading
import time
from collections.abc import Callable

from fleet.core import file_turn


def _nothing() -> None:
    """The work of a turn that writes nothing."""


def is_held(lock: pathlib.Path) -> bool:
    """Whether a turn holds the lock now, asked without waiting.

    Args:
        lock: The lock file.

    Returns:
        True when another open file holds its ``flock``; False when none
        does, or no turn has made the file.
    """
    try:
        descriptor = os.open(lock, os.O_RDWR)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(descriptor)
    return False


class OtherTurn:
    """A turn held on a lock file by a descriptor of the test's own."""

    def __init__(self, lock: pathlib.Path) -> None:
        """Take the turn at once, making the lock file when absent.

        Args:
            lock: The lock file.
        """
        lock.parent.mkdir(parents=True, exist_ok=True)
        self._descriptor = os.open(lock, os.O_RDWR | os.O_CREAT, file_turn.LOCK_FILE_MODE)
        fcntl.flock(self._descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def end_after(self, seconds: float, work: Callable[[], object] = _nothing) -> threading.Thread:
        """Do the turn's work and let it go after a span, on a thread of its own.

        Args:
            seconds: How long the turn is held from now.
            work: What the turn does just before it lets go; what it returns
                is not read.

        Returns:
            The started thread, for the case to join.
        """

        def ending() -> None:
            time.sleep(seconds)
            work()
            os.close(self._descriptor)

        thread = threading.Thread(target=ending)
        thread.start()
        return thread
