"""One read of a file another process rewrites with ``os.replace``.

The fleet's shared files (sibling ``knowledge.json`` reports, claim
contents) are rewritten by staging a file and replacing the destination.
A reader that lands inside that swap does not see the file, and how it
fails to see it depends on the filesystem (board task b651224a):

* a Windows host refuses the open with ``PermissionError`` (arterial
  tick 264, 2026-08-26 03:01:06);
* the container fleet's runs mount, sedona's Windows directory
  bind-mounted into Linux by Docker Desktop, shows the file MISSING
  (``ENOENT``), EMPTY (a zero-length read, never a partial one), or
  unreadable with ``ENODATA``. Measured in the fleet container with one
  writer replacing a 20 KB file while two readers read it for 120 s:
  52 ENOENT and 10 empty reads, against 0 of any kind over 2.8 million
  reads of the same file on the container's own ``/tmp``. ENODATA
  killed bot p2 at tick 24 of a live Practice run, 2026-09-29 05:56:43.

:func:`read_once` names which of those four hid the file, and raises any
other failure. What a caller does about a window is its own contract.
"""

from __future__ import annotations

import errno
from enum import StrEnum
from pathlib import Path

from typing_extensions import TypedDict

from tankpit_bot import _test_hooks


class ReplaceWindow(StrEnum):
    """How the writer's ``os.replace`` window hid a file from one read."""

    PERMISSION_DENIED = "permission_denied"
    MISSING = "missing"
    NO_DATA = "no_data"
    EMPTY = "empty"


class WindowedReadDict(TypedDict):
    """The outcome of one read of a replaced file.

    Attributes:
        text: The file's content; empty exactly when ``window`` is set.
        window: The replace window that hid the file, or ``None`` when
            the read saw it.
    """

    text: str
    window: ReplaceWindow | None


def read_once(path: Path) -> WindowedReadDict:
    """Read a replaced file once, naming the replace window if one hid it.

    Args:
        path: The file.

    Returns:
        The text, or the window that hid it.

    Raises:
        OSError: Any read failure other than the four forms
            :class:`ReplaceWindow` names.
    """
    try:
        text = _test_hooks.read_text(path)
    except PermissionError:
        return WindowedReadDict(text="", window=ReplaceWindow.PERMISSION_DENIED)
    except FileNotFoundError:
        return WindowedReadDict(text="", window=ReplaceWindow.MISSING)
    except OSError as error:
        if error.errno != errno.ENODATA:
            raise
        return WindowedReadDict(text="", window=ReplaceWindow.NO_DATA)
    if not text:
        return WindowedReadDict(text="", window=ReplaceWindow.EMPTY)
    return WindowedReadDict(text=text, window=None)


__all__ = [
    "ReplaceWindow",
    "WindowedReadDict",
    "read_once",
]
