"""Filesystem hooks: read/write/append text + path existence checks.

All four operations are routed through Protocol-typed module-level
symbols. Production code uses ``pathlib.Path``-backed implementations;
tests replace the symbols with in-memory fakes to keep filesystem
access out of the test loop.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from platform_core.logging import get_logger

_fs_log = get_logger(__name__)


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


class FileMarkerProtocol(Protocol):
    """Protocol for identifying a file and measuring it in one call."""

    def __call__(self, path: Path) -> tuple[int, int]:
        """Return the file's identity and current size.

        One call rather than an exists-then-stat pair: a growing file
        that is replaced between the two calls would otherwise report
        the new run's size against the old run's identity.

        Args:
            path: File to inspect.

        Returns:
            ``(identity, size_bytes)``. ``identity`` is the
            filesystem's own file number, which changes when the path
            is re-created for a new run — the signal an incremental
            reader uses to know its consumed offset is meaningless.

        Raises:
            OSError: If the file does not exist.
        """
        ...


class ReadBytesFromProtocol(Protocol):
    """Protocol for reading a file's bytes from a starting offset."""

    def __call__(self, path: Path, offset: int) -> bytes:
        """Read from ``offset`` to end of file.

        Args:
            path: File to read.
            offset: Byte offset to start at; a offset at or past the
                end yields empty bytes.

        Returns:
            Every byte from ``offset`` onward.

        Raises:
            OSError: If the file does not exist.
        """
        ...


class CreateTextExclusiveProtocol(Protocol):
    """Protocol for atomically creating a file that must not exist."""

    def __call__(self, path: Path, content: str) -> bool:
        """Create ``path`` with ``content`` only if no file is there.

        The existence check and the creation are ONE atomic filesystem
        operation (``O_CREAT | O_EXCL``), so when several processes
        race to create the same path exactly one wins — the mutex
        primitive the fleet's authoritative container claim is built
        on ([[fleet-forage-allocation]]). Only the CREATION is atomic:
        a concurrent reader can observe the file after the exclusive
        open and before the content lands, so protocols built on this
        primitive must treat existence as the lock and content as
        metadata.

        Args:
            path: File path that must not already exist.
            content: Text content to write into the new file.

        Returns:
            True when this call created the file; False when a file
            was already there.
        """
        ...


class ReplaceTextProtocol(Protocol):
    """Protocol for atomically replacing a file's text content."""

    def __call__(self, path: Path, content: str) -> None:
        """Write text and atomically move it into place.

        Readers of ``path`` see either the previous complete content
        or the new complete content, never a partial write — the
        invariant the fleet knowledge exchange relies on to make its
        strict decode-and-raise semantics sound.

        Args:
            path: Final file path.
            content: Text content to write.
        """
        ...


def _real_replace_text(path: Path, content: str) -> None:
    """Real implementation: write a per-process temp file, then os.replace().

    The staging name carries the pid so concurrent writers of the
    same destination never collide on the temp file. On Windows,
    ``os.replace`` raises ``PermissionError`` while a READER holds
    the destination open (CPython opens files without
    ``FILE_SHARE_DELETE``) — for the heartbeat-style files this hook
    serves (the fleet knowledge exchange rewrites every tick), the
    contention law is: DROP this beat. The previous complete content
    stays current, the staging file is removed, and the next tick's
    write refreshes — readers still never observe a torn file, which
    is the contract.

    Args:
        path: Final file path.
        content: Text content to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    staging.write_text(content, encoding="utf-8")
    try:
        staging.replace(path)
    except PermissionError:
        _fs_log.info(
            "replace_text contention on %s: destination held open by a "
            "reader; beat dropped, previous content stays current",
            path,
        )
        staging.unlink(missing_ok=True)


def _real_create_text_exclusive(path: Path, content: str) -> bool:
    """Real implementation using ``os.open`` with ``O_CREAT | O_EXCL``.

    Args:
        path: File path that must not already exist.
        content: Text content to write into the new file.

    Returns:
        True when this call created the file; False when a file was
        already there.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(content)
    return True


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


def _real_file_marker(path: Path) -> tuple[int, int]:
    """Real implementation using a single Path.stat().

    Args:
        path: File to inspect.

    Returns:
        ``(st_ino, st_size)``. On Windows ``st_ino`` is the NTFS file
        index, so it distinguishes a re-created path from a grown one.

    Raises:
        OSError: If the file does not exist.
    """
    status = path.stat()
    return (status.st_ino, status.st_size)


def _real_read_bytes_from(path: Path, offset: int) -> bytes:
    """Real implementation seeking to ``offset`` and reading to EOF.

    Args:
        path: File to read.
        offset: Byte offset to start at.

    Returns:
        Every byte from ``offset`` onward; empty when the offset is at
        or past the end.

    Raises:
        OSError: If the file does not exist.
    """
    with path.open("rb") as handle:
        handle.seek(offset)
        return handle.read()


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


create_text_exclusive: CreateTextExclusiveProtocol = _real_create_text_exclusive
replace_text: ReplaceTextProtocol = _real_replace_text
write_text: WriteTextProtocol = _real_write_text
write_bytes: WriteBytesProtocol = _real_write_bytes
read_text: ReadTextProtocol = _real_read_text
append_text: AppendTextProtocol = _real_append_text
path_exists: PathExistsProtocol = _real_path_exists
remove_file: RemoveFileProtocol = _real_remove_file
glob_paths: GlobPathsProtocol = _real_glob_paths
file_marker: FileMarkerProtocol = _real_file_marker
read_bytes_from: ReadBytesFromProtocol = _real_read_bytes_from


__all__ = [
    "AppendTextProtocol",
    "CreateTextExclusiveProtocol",
    "GlobPathsProtocol",
    "PathExistsProtocol",
    "ReadTextProtocol",
    "RemoveFileProtocol",
    "ReplaceTextProtocol",
    "WriteBytesProtocol",
    "WriteTextProtocol",
    "append_text",
    "create_text_exclusive",
    "glob_paths",
    "path_exists",
    "read_text",
    "remove_file",
    "replace_text",
    "write_bytes",
    "write_text",
]
