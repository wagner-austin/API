"""Filesystem hooks: read/write/append text + path existence checks.

All four operations are routed through Protocol-typed module-level
symbols. Production code uses ``pathlib.Path``-backed implementations;
tests replace the symbols with in-memory fakes to keep filesystem
access out of the test loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class WriteTextProtocol(Protocol):
    """Protocol for writing text to a file."""

    def __call__(self, path: Path, content: str) -> None:
        """Write text content to a file.

        Args:
            path: File path to write to.
            content: Text content to write.
        """
        ...


class WriteBytesProtocol(Protocol):
    """Protocol for writing binary content to a file."""

    def __call__(self, path: Path, content: bytes) -> None:
        """Write binary content to a file.

        Args:
            path: File path to write to.
            content: Binary content to write.
        """
        ...


class ReadTextProtocol(Protocol):
    """Protocol for reading text from a file."""

    def __call__(self, path: Path) -> str:
        """Read text content from a file.

        Args:
            path: File path to read from.

        Returns:
            Text content of the file.

        Raises:
            FileNotFoundError: If file does not exist.
        """
        ...


class AppendTextProtocol(Protocol):
    """Protocol for appending text to a file."""

    def __call__(self, path: Path, content: str) -> None:
        """Append text content to a file.

        Args:
            path: File path to append to.
            content: Text content to append.
        """
        ...


class PathExistsProtocol(Protocol):
    """Protocol for checking if a path exists."""

    def __call__(self, path: Path) -> bool:
        """Check if path exists.

        Args:
            path: Path to check.

        Returns:
            True if path exists, False otherwise.
        """
        ...


class RemoveFileProtocol(Protocol):
    """Protocol for deleting a file."""

    def __call__(self, path: Path) -> None:
        """Delete a file, tolerating absence.

        Args:
            path: File path to delete.
        """
        ...


class GlobPathsProtocol(Protocol):
    """Protocol for listing files in a directory by glob pattern."""

    def __call__(self, directory: Path, pattern: str) -> list[Path]:
        """List files in a directory matching a glob pattern.

        Args:
            directory: Directory to list.
            pattern: Glob pattern matched against file names.

        Returns:
            Matching paths in sorted order; empty when the directory
            does not exist or nothing matches.
        """
        ...


def _real_write_text(path: Path, content: str) -> None:
    """Real implementation using Path.write_text().

    Args:
        path: File path to write to.
        content: Text content to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _real_write_bytes(path: Path, content: bytes) -> None:
    """Real implementation using Path.write_bytes().

    Args:
        path: File path to write to.
        content: Binary content to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _real_read_text(path: Path) -> str:
    """Real implementation using Path.read_text().

    Args:
        path: File path to read from.

    Returns:
        Text content of the file.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    return path.read_text(encoding="utf-8")


def _real_append_text(path: Path, content: str) -> None:
    """Real implementation using append mode.

    Args:
        path: File path to append to.
        content: Text content to append.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(content)


def _real_path_exists(path: Path) -> bool:
    """Real implementation using Path.exists().

    Args:
        path: Path to check.

    Returns:
        True if path exists, False otherwise.
    """
    return path.exists()


def _real_remove_file(path: Path) -> None:
    """Real implementation using Path.unlink().

    Args:
        path: File path to delete; a missing file is a no-op.
    """
    path.unlink(missing_ok=True)


def _real_glob_paths(directory: Path, pattern: str) -> list[Path]:
    """Real implementation using Path.glob().

    Args:
        directory: Directory to list.
        pattern: Glob pattern matched against file names.

    Returns:
        Matching paths in sorted order; empty when the directory does
        not exist or nothing matches.
    """
    return sorted(directory.glob(pattern))


write_text: WriteTextProtocol = _real_write_text
write_bytes: WriteBytesProtocol = _real_write_bytes
read_text: ReadTextProtocol = _real_read_text
append_text: AppendTextProtocol = _real_append_text
path_exists: PathExistsProtocol = _real_path_exists
remove_file: RemoveFileProtocol = _real_remove_file
glob_paths: GlobPathsProtocol = _real_glob_paths


__all__ = [
    "AppendTextProtocol",
    "GlobPathsProtocol",
    "PathExistsProtocol",
    "ReadTextProtocol",
    "RemoveFileProtocol",
    "WriteBytesProtocol",
    "WriteTextProtocol",
    "append_text",
    "glob_paths",
    "path_exists",
    "read_text",
    "remove_file",
    "write_bytes",
    "write_text",
]
