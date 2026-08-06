"""Injectable seams for the archive-analysis layer.

Production wires the real filesystem at import time; tests install
fakes with :func:`set_analysis_hooks` and restore with
:func:`reset_analysis_hooks`. There is no conditional anywhere in the
package — callers invoke the hook directly, so the test path and the
production path execute the same code.

Only genuine boundaries live here. Frame splitting, XOR bring-up and
message decoding are pure functions over bytes and are exercised
directly rather than through a seam.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ReadTextFn(Protocol):
    """Read a UTF-8 text file whole."""

    def __call__(self, path: Path) -> str:
        """Return the file's decoded contents.

        Args:
            path: File to read.

        Returns:
            The file's contents decoded as UTF-8.

        Raises:
            OSError: If the file cannot be read.
        """
        ...


class ListSessionPathsFn(Protocol):
    """Enumerate capture-session files under a directory."""

    def __call__(self, directory: Path) -> list[Path]:
        """Return every capture-session file, in a stable order.

        Args:
            directory: Directory to enumerate.

        Returns:
            Matching paths sorted by name, so a scan of the same
            archive twice produces the same order.
        """
        ...


def _read_text_impl(path: Path) -> str:
    """Production :class:`ReadTextFn` — read the file as UTF-8.

    Args:
        path: File to read.

    Returns:
        The file's contents.

    Raises:
        OSError: If the file cannot be read.
    """
    return path.read_text(encoding="utf-8")


def _list_session_paths_impl(directory: Path) -> list[Path]:
    """Production :class:`ListSessionPathsFn` — glob and sort.

    Args:
        directory: Directory to enumerate.

    Returns:
        Every ``*.capture_session.json`` path, sorted by name.
    """
    return sorted(directory.glob("*.capture_session.json"))


read_text: ReadTextFn = _read_text_impl
list_session_paths: ListSessionPathsFn = _list_session_paths_impl


def set_analysis_hooks(
    *,
    read_text_fn: ReadTextFn,
    list_session_paths_fn: ListSessionPathsFn,
) -> None:
    """Install analysis IO hooks.

    Args:
        read_text_fn: Replacement file reader.
        list_session_paths_fn: Replacement directory enumerator.
    """
    global read_text, list_session_paths
    read_text = read_text_fn
    list_session_paths = list_session_paths_fn


def reset_analysis_hooks() -> None:
    """Restore the production analysis IO hooks."""
    global read_text, list_session_paths
    read_text = _read_text_impl
    list_session_paths = _list_session_paths_impl


__all__ = [
    "ListSessionPathsFn",
    "ReadTextFn",
    "list_session_paths",
    "read_text",
    "reset_analysis_hooks",
    "set_analysis_hooks",
]
