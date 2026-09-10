"""Injection points for what this package does outside the process itself.

Module-level names, rebound by ``tests.conftest`` before each test and
restored after, exactly as in ``fleet_wake._test_hooks`` and
``hpc_wake._test_hooks``. Production binds the real implementations at
import; a test binds fakes. There is no conditional anywhere -- the call site
calls the hook.

FIVE SEAMS, AND THE PROCESS RUNNER IS THE ONE WORTH EXPLAINING. This bridge
reads GitHub through the ``gh`` CLI rather than over HTTPS with a token of
its own, and that is a deliberate trade rather than the lazy option:

* ``gh`` is already authenticated on this machine, under the operator's own
  account, with the scopes the work needs. A second long-lived GitHub token
  would be a second credential to mint, store beside the board secrets,
  rotate, and eventually find expired -- to read data the first credential
  already reaches.
* Every other reader of CI in this workspace is a ``gh api`` call. The wiki
  page this package's announcements are shaped by cites ``gh api`` a dozen
  times. A bridge that spoke a different dialect to the same endpoint would
  be the one surface whose answers could not be reproduced by pasting its
  command into a terminal.

The honest cost, stated because a fallback would be worse: ``gh`` must be
installed and logged in wherever the cycle runs, and when it is not, this
package fails loudly with ``GH_COMMAND_FAILED`` carrying the CLI's own
stderr. It does not degrade, retry, or skip -- a bridge that quietly
announced nothing because it could not ask is indistinguishable from a
bridge with nothing to announce, which is the silence the whole family
exists to remove.

THE ENROLMENT RECORD AND THE POSITION RECORD SHARE THESE SEAMS AND ARE
DIFFERENT FILES ON PURPOSE. One is written by a git hook on every push and
grows with attempts; the other is written by the cycle and grows with
announcements. A test can therefore give them different behaviour -- a
readable enrolment record beside a position file that fails to write -- which
is the case that decides whether an announcement is repeated or lost.
"""

from __future__ import annotations

import datetime
import pathlib
import subprocess
import sys
from collections.abc import Sequence
from typing import Protocol

from platform_core.mcp_client import McpPostProtocol, urllib_mcp_post


class CompletedProto(Protocol):
    """The three fields this package reads off a finished process."""

    @property
    def stdout(self) -> str:
        """Captured standard output."""

    @property
    def stderr(self) -> str:
        """Captured standard error."""

    @property
    def returncode(self) -> int:
        """The process's exit status."""


class RunProcessProtocol(Protocol):
    """The slice of :func:`subprocess.run` the ``gh`` boundary uses.

    ``check`` is deliberately absent. A non-zero exit is read and turned into
    a coded refusal carrying the CLI's stderr, which says what went wrong;
    ``check=True`` would raise a ``CalledProcessError`` whose message is the
    argv and nothing else, and the useful half would have to be recovered
    from an attribute afterwards.
    """

    def __call__(
        self,
        args: Sequence[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: int,
    ) -> CompletedProto:
        """Run the command to completion and return its outcome.

        Args:
            args: The full argument vector, program first.
            capture_output: Always True here; both streams are read.
            text: Always True here; both streams are decoded as text.
            timeout: Seconds before the process is abandoned.

        Returns:
            The finished process.
        """
        ...


class ReadTextProtocol(Protocol):
    """Read a whole file as UTF-8 text."""

    def __call__(self, path: pathlib.Path) -> str:
        """Read it.

        Args:
            path: Absolute path to read.

        Returns:
            The file's contents.
        """
        ...


class AppendTextProtocol(Protocol):
    """Append one line to a file."""

    def __call__(self, path: pathlib.Path, line: str) -> None:
        """Append it.

        Args:
            path: Absolute path to append to.
            line: The line, without a trailing newline.
        """
        ...


class FileExistsProtocol(Protocol):
    """Report whether a path is an existing file."""

    def __call__(self, path: pathlib.Path) -> bool:
        """Check it.

        Args:
            path: Absolute path to test.

        Returns:
            True when the file exists.
        """
        ...


class EmitProtocol(Protocol):
    """Write one line to the cycle's report stream."""

    def __call__(self, line: str) -> None:
        """Write it.

        Args:
            line: The line, without a trailing newline.
        """
        ...


class NowProtocol(Protocol):
    """Read the wall clock."""

    def __call__(self) -> int:
        """Read it.

        Returns:
            Whole seconds since the epoch, UTC. Integer seconds because
            every comparison this package makes is an age in seconds, and a
            float would invite arithmetic nobody needs.
        """
        ...


def _default_read_text(path: pathlib.Path) -> str:
    """Read a real file as UTF-8.

    Args:
        path: Absolute path to read.

    Returns:
        The file's contents.
    """
    return path.read_text(encoding="utf-8")


def _default_append_text(path: pathlib.Path, line: str) -> None:
    """Append one line to a real file.

    The parent directory is created if absent, because both records are
    created by their first write and a machine whose bridge has never run is
    the ordinary first-run case rather than a mistake.

    Args:
        path: Absolute path to append to.
        line: The line, without a trailing newline.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


def _default_file_exists(path: pathlib.Path) -> bool:
    """Test a real path.

    Args:
        path: Absolute path to test.

    Returns:
        True when it is an existing file.
    """
    return path.is_file()


def _default_emit(line: str) -> None:
    """Write one line to standard output and flush it.

    The flush is required: a scheduler or Monitor reads this process's stdout
    as a stream, and a buffered line is an event that has not happened yet as
    far as the subscriber is concerned.

    Args:
        line: The line, without a trailing newline.
    """
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _default_now() -> int:
    """Read the wall clock.

    Returns:
        Whole seconds since the epoch, UTC.
    """
    return int(datetime.datetime.now(tz=datetime.UTC).timestamp())


http_post: McpPostProtocol = urllib_mcp_post
run_process: RunProcessProtocol = subprocess.run
read_text: ReadTextProtocol = _default_read_text
append_text: AppendTextProtocol = _default_append_text
file_exists: FileExistsProtocol = _default_file_exists
emit: EmitProtocol = _default_emit
now: NowProtocol = _default_now


def reset_hooks() -> None:
    """Rebind every hook to its production implementation."""
    global http_post, run_process, read_text, append_text, file_exists, emit, now
    http_post = urllib_mcp_post
    run_process = subprocess.run
    read_text = _default_read_text
    append_text = _default_append_text
    file_exists = _default_file_exists
    emit = _default_emit
    now = _default_now


__all__ = [
    "AppendTextProtocol",
    "CompletedProto",
    "EmitProtocol",
    "FileExistsProtocol",
    "NowProtocol",
    "ReadTextProtocol",
    "RunProcessProtocol",
    "append_text",
    "emit",
    "file_exists",
    "http_post",
    "now",
    "read_text",
    "reset_hooks",
    "run_process",
]
