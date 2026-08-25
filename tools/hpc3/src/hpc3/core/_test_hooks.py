"""Dependency-injection seam for the core.

Every impure act -- running a command on the cluster, sending bytes to it,
reading a local file -- is reached through a symbol here, bound to its real
implementation at import time. Production calls the hook directly, so there is
no conditional dispatch and no second code path; a test rebinds the symbol and
exercises the same lines.

Nothing here catches an exception. ``subprocess.run`` is called with
``check=False`` and the return code is inspected explicitly, so a remote
failure becomes a typed :class:`~platform_core.errors.AppError` at the call
site that knows what the command was for, rather than a
``CalledProcessError`` caught and re-raised somewhere that does not.
"""

from __future__ import annotations

import pathlib
import subprocess
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from platform_core.logging import get_logger
from typing_extensions import TypedDict

_logger = get_logger("hpc3")


class CommandResult(TypedDict):
    """What running a command produced.

    Attributes:
        returncode: Process exit status.
        stdout: Standard output, decoded as UTF-8.
        stderr: Standard error, decoded as UTF-8. Carried because Slurm and
            ssh put the reason for a rejection here, and a failure that
            discards it forces the reader back to the cluster to find out
            what happened.
    """

    returncode: int
    stdout: str
    stderr: str


class RunProtocol(Protocol):
    """Protocol for running a local command to completion."""

    def __call__(self, argv: Sequence[str], *, stdin_bytes: bytes | None = None) -> CommandResult:
        """Run a command and collect its result.

        Args:
            argv: Executable and arguments. Never a shell string: the payload
                may hold arbitrary text and shell interpretation of it would
                be a defect, not a feature.
            stdin_bytes: Bytes to write to the process's standard input, or
                None to provide no input.

        Returns:
            The command's exit status and captured streams. A non-zero status
            is returned, never raised -- the caller decides what it means.
        """
        ...


def _default_run(argv: Sequence[str], *, stdin_bytes: bytes | None = None) -> CommandResult:
    """Real implementation running a command via subprocess.

    Args:
        argv: Executable and arguments.
        stdin_bytes: Bytes for standard input, or None.

    Returns:
        The command's exit status and captured streams, decoded as UTF-8 with
        undecodable bytes replaced -- a mangled character in a diagnostic is
        better than losing the diagnostic.
    """
    completed = subprocess.run(
        list(argv),
        check=False,
        input=stdin_bytes,
        capture_output=True,
    )
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout.decode("utf-8", errors="replace"),
        stderr=completed.stderr.decode("utf-8", errors="replace"),
    )


class LogEventProtocol(Protocol):
    """Protocol for emitting one structured audit event."""

    def __call__(self, event: str, fields: Mapping[str, str | int | float | bool]) -> None:
        """Emit a structured event.

        Args:
            event: Stable event name, the thing a log query filters on.
            fields: Structured context. Scalars only: a nested object would
                be flattened differently by every log reader, and the point
                of structure is that it survives the trip.
        """
        ...


def _default_log_event(event: str, fields: Mapping[str, str | int | float | bool]) -> None:
    """Real implementation writing through the platform logger.

    Args:
        event: Stable event name.
        fields: Structured context.
    """
    _logger.info(event, extra=dict(fields))


def _default_read_bytes(path: pathlib.Path) -> bytes:
    """Read a file's raw bytes.

    Args:
        path: File to read.

    Returns:
        The file's bytes, for the caller to digest or transfer.
    """
    return path.read_bytes()


def _default_append_text(path: pathlib.Path, text: str) -> None:
    """Append text to a file, creating it and its parents if absent.

    Opened in append mode per call and closed immediately, so a record is on
    disk before the next one is built. A held-open handle would buffer the
    sweep's earlier submissions into a process that may not survive to flush
    them -- and those jobs are already running.

    Args:
        path: File to append to.
        text: Text to append, including its own trailing newline.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _default_file_exists(path: pathlib.Path) -> bool:
    """Report whether a path names an existing file.

    Args:
        path: Path to test.

    Returns:
        True when the path exists and is a regular file.
    """
    return path.is_file()


def _default_write_text(path: pathlib.Path, text: str) -> None:
    """Write text to a file, replacing any existing contents.

    Written with an explicit LF newline. A rendered definition or shell
    script authored on Windows and translated to CRLF makes the cluster's
    kernel report the interpreter as missing, which reads as a broken image
    rather than a line-ending problem.

    Args:
        path: File to write. Its parent must already exist; creating it
            silently here would let a typo in a destination produce a
            plausible-looking tree in the wrong place.
        text: Complete file contents, including its own trailing newline.
    """
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _default_make_dir(path: pathlib.Path) -> None:
    """Create a directory and any missing parents.

    Args:
        path: Directory to create. Existing is not an error: rendering the
            same build twice is an ordinary act, and the files inside are
            replaced rather than merged.
    """
    path.mkdir(parents=True, exist_ok=True)


run: RunProtocol = _default_run
log_event: LogEventProtocol = _default_log_event
read_bytes: Callable[[pathlib.Path], bytes] = _default_read_bytes
append_text: Callable[[pathlib.Path, str], None] = _default_append_text
file_exists: Callable[[pathlib.Path], bool] = _default_file_exists
write_text: Callable[[pathlib.Path, str], None] = _default_write_text
make_dir: Callable[[pathlib.Path], None] = _default_make_dir


def reset_hooks() -> None:
    """Rebind every hook to its production implementation."""
    global run, log_event, read_bytes, append_text, file_exists, write_text, make_dir
    run = _default_run
    log_event = _default_log_event
    read_bytes = _default_read_bytes
    append_text = _default_append_text
    file_exists = _default_file_exists
    write_text = _default_write_text
    make_dir = _default_make_dir


__all__ = [
    "CommandResult",
    "LogEventProtocol",
    "RunProtocol",
    "append_text",
    "file_exists",
    "log_event",
    "make_dir",
    "read_bytes",
    "reset_hooks",
    "run",
    "write_text",
]
