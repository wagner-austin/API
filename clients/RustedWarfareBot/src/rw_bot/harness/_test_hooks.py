"""Dependency-injection hooks for the harness package.

Every non-pure operation the harness performs is reached through a
module-level symbol here, bound to its real implementation at import time.
Callers always invoke the hook directly — there is no ``if testing`` branch
anywhere, so the production and test code paths are byte-identical in shape.

Tests rebind a symbol to a fake and restore it afterwards. This module is
private to the package; consumers outside it must not import from here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Protocol


class ReadTextLinesProto(Protocol):
    """Read a UTF-8 text file and return its lines."""

    def __call__(self, path: Path) -> tuple[str, ...]:
        """Read every line of a text file.

        Args:
            path: File to read.

        Returns:
            The file's lines with trailing newlines removed, in file order.

        Raises:
            OSError: When the file cannot be read.
            UnicodeDecodeError: When the file is not valid UTF-8.
        """
        ...


class WriteLineProto(Protocol):
    """Emit one line to the process's standard output."""

    def __call__(self, text: str) -> None:
        """Write one line.

        Args:
            text: Line content, without a trailing newline.
        """
        ...


class PathExistsProto(Protocol):
    """Report whether a filesystem path exists."""

    def __call__(self, path: Path) -> bool:
        """Test one path for existence.

        Args:
            path: Path to test.

        Returns:
            ``True`` when the path exists, ``False`` otherwise.
        """
        ...


class ReadArgvProto(Protocol):
    """Read this process's command-line arguments, excluding the program name."""

    def __call__(self) -> list[str]:
        """Return the argument list.

        Returns:
            Arguments after the program name, in order.
        """
        ...


def _read_argv_impl() -> list[str]:
    """Production implementation of :class:`ReadArgvProto`.

    Returns:
        ``sys.argv`` after the program name.
    """
    return list(sys.argv[1:])


def _read_text_lines_impl(path: Path) -> tuple[str, ...]:
    """Production implementation of :class:`ReadTextLinesProto`.

    Decoding is strict: a log that is not valid UTF-8 is a real problem with
    the launcher's ``-Dfile.encoding`` setting, and silently replacing bad
    bytes would hide it.

    Args:
        path: File to read.

    Returns:
        The file's lines with trailing newlines removed, in file order.

    Raises:
        OSError: When the file cannot be read.
        UnicodeDecodeError: When the file is not valid UTF-8.
    """
    return tuple(path.read_text(encoding="utf-8").splitlines())


def _path_exists_impl(path: Path) -> bool:
    """Production implementation of :class:`PathExistsProto`.

    Args:
        path: Path to test.

    Returns:
        ``True`` when the path exists, ``False`` otherwise.
    """
    return path.exists()


def _write_line_impl(text: str) -> None:
    """Production implementation of :class:`WriteLineProto`.

    Args:
        text: Line content, without a trailing newline.
    """
    sys.stdout.write(f"{text}\n")


path_exists: PathExistsProto = _path_exists_impl
read_argv: ReadArgvProto = _read_argv_impl
read_text_lines: ReadTextLinesProto = _read_text_lines_impl
write_line: WriteLineProto = _write_line_impl


__all__ = [
    "PathExistsProto",
    "ReadArgvProto",
    "ReadTextLinesProto",
    "WriteLineProto",
    "path_exists",
    "read_argv",
    "read_text_lines",
    "write_line",
]
