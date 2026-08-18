"""Dependency-injection hooks for this package's boundary effects.

Every symbol is bound to its real implementation at import time and called
unconditionally. Production wires nothing; tests rebind and restore. There is
no ``if hook is not None`` branch anywhere, because a conditional hook is a
second code path that production never exercises.

Only genuine boundary effects belong here. The simulator is not a hook: it is
injected as a parameter through :class:`navprobe.rollout.SimulatorProtocol`,
which is the stronger form of the same discipline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ReadTextProtocol(Protocol):
    """Read a file's entire contents as UTF-8 text."""

    def __call__(self, path: Path) -> str:
        """Read the file.

        Args:
            path: File to read.

        Returns:
            The file's contents.

        Raises:
            OSError: When the file cannot be read.
        """
        ...


class WriteTextProtocol(Protocol):
    """Write UTF-8 text to a file, replacing any existing contents."""

    def __call__(self, path: Path, text: str) -> None:
        """Write the file.

        Args:
            path: File to write.
            text: Contents to write.

        Raises:
            OSError: When the file cannot be written.
        """
        ...


class MakeParentDirsProtocol(Protocol):
    """Create a path's parent directories."""

    def __call__(self, path: Path) -> None:
        """Create every missing parent directory of ``path``.

        Args:
            path: File whose parents should exist.

        Raises:
            OSError: When a directory cannot be created.
        """
        ...


def _read_text_impl(path: Path) -> str:
    """Production implementation of :class:`ReadTextProtocol`.

    Args:
        path: File to read.

    Returns:
        The file's contents.

    Raises:
        OSError: When the file cannot be read.
    """
    return path.read_text(encoding="utf-8")


def _write_text_impl(path: Path, text: str) -> None:
    """Production implementation of :class:`WriteTextProtocol`.

    Args:
        path: File to write.
        text: Contents to write.

    Raises:
        OSError: When the file cannot be written.
    """
    path.write_text(text, encoding="utf-8")


def _make_parent_dirs_impl(path: Path) -> None:
    """Production implementation of :class:`MakeParentDirsProtocol`.

    Args:
        path: File whose parents should exist.

    Raises:
        OSError: When a directory cannot be created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


read_text: ReadTextProtocol = _read_text_impl
write_text: WriteTextProtocol = _write_text_impl
make_parent_dirs: MakeParentDirsProtocol = _make_parent_dirs_impl


__all__ = [
    "MakeParentDirsProtocol",
    "ReadTextProtocol",
    "WriteTextProtocol",
    "make_parent_dirs",
    "read_text",
    "write_text",
]
