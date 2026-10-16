"""Make the platform refuse to remove or create a file, the way a racing process can.

Support module, not a test module (MCPs board task a85ef09e). On Windows a
file another process holds open cannot be removed: measured on austinpc,
Python 3.11.9, on 2026-10-08, ``os.unlink`` raises ``PermissionError`` while a
plain ``open`` of the file is held, which is what a host's other runner does
for the moment it reads a claim (:func:`fleet.core.host_claims.discharge`).
Linux removes an open file, so there the same refusal is made by taking the
write bit off the file's directory, which refuses the removal for a user
other than root, as every fleet node's runner account is.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import stat
import sys
from collections.abc import Callable, Generator


@contextlib.contextmanager
def refused_removal(path: pathlib.Path) -> Generator[Callable[[], None], None, None]:
    """Refuse every removal of a file until released, or the block ends.

    Args:
        path: An existing file.

    Yields:
        The release, which a case calls when the refusal should end; calling
        it again does nothing.
    """
    if sys.platform == "win32":
        handle = path.open(encoding="utf-8")
        released = [False]

        def release() -> None:
            if not released[0]:
                handle.close()
                released[0] = True

    else:
        directory = path.parent
        writable = directory.stat().st_mode
        os.chmod(directory, writable & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))

        def release() -> None:
            os.chmod(directory, writable)

    try:
        yield release
    finally:
        release()


@contextlib.contextmanager
def refused_creation(path: pathlib.Path) -> Generator[Callable[[], None], None, None]:
    """Refuse an exclusive create of a file with ``PermissionError`` until released.

    What Windows does while another turn's give-back is still removing a lock
    (:func:`fleet.core.file_turn.held`), made to last: on Windows a directory
    standing at the path refuses ``os.open(O_CREAT | O_EXCL)`` with
    ``PermissionError`` (measured on austinpc, Python 3.11.9, 2026-10-16),
    and on Linux taking the write bit off the parent refuses it the same way
    for a user other than root.

    Args:
        path: A file that does not exist, in a directory that does.

    Yields:
        The release, which a case calls when the refusal should end; calling
        it again does nothing.
    """
    if sys.platform == "win32":
        path.mkdir()
        released = [False]

        def release() -> None:
            if not released[0]:
                path.rmdir()
                released[0] = True

    else:
        directory = path.parent
        writable = directory.stat().st_mode
        os.chmod(directory, writable & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))

        def release() -> None:
            os.chmod(directory, writable)

    try:
        yield release
    finally:
        release()


__all__ = ["refused_creation", "refused_removal"]
