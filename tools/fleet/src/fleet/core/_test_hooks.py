"""Dependency-injection seam for the core.

Every impure act -- running a command, reading the clock, touching a file --
is reached through a symbol here, bound to its real implementation at import
time. Production calls the hook directly, so there is no conditional dispatch
and no second code path; a test rebinds the symbol and exercises the same
lines.

THE CLOCK IS A HOOK AND THAT IS NOT INCIDENTAL. A lease has an expiry, so
every question this package answers about whether a resource is free is a
question about the current time. A test that could not control the clock could
only assert that an unexpired lease is unexpired, which is the case that never
breaks. Controlling it is how the expiry boundary gets tested at all.

NOTHING HERE CATCHES. The one named exception in this package is the
deadline, and it now lives with the command runner in
:mod:`fleet.core._command`, whose docstring states it: ``subprocess`` has
no non-raising way to report an expired ``timeout``, so that one report is
converted into the result's own ``timed_out`` field at the boundary.
Nothing is retried, softened or defaulted; the caller reads ``timed_out``
exactly as it reads ``returncode``.

EVERY COMMAND CARRIES A DEADLINE, AND THE PARAMETER HAS NO DEFAULT. Measured
2026-09-17 11:15Z (board tasks 35940277 and 41ac6ed2): one ssh whose peer
went away mid-command sat in ESTABLISHED for three days, the tick around it
never exited, the scheduled task's IgnoreNew refused every later tick, and
the dispatch queue drained nothing until the task's 72-hour execution limit
ended the parent. A default would be the value a caller reaches for without
deciding, which is how that ssh had none.

AND A DEADLINE IS ONLY REAL IF THE CALL CANNOT BLOCK OUTSIDE IT, which is
the half the rule above does not state and did not hold until 2026-09-24
(board task 1e57ebe5). That reasoning, and the measurement behind it,
belong with the code they govern: see :mod:`fleet.core._command`, which
this module re-exports :class:`~fleet.core._command.CommandResult` and
:data:`~fleet.core._command.TIMED_OUT_RETURNCODE` from.
"""

from __future__ import annotations

import pathlib
import socket
import subprocess
import tempfile
import time
from collections.abc import Sequence
from typing import Protocol

from platform_core.config import _optional_env_str, config_test_hooks
from platform_core.mcp_client import McpPostProtocol, urllib_mcp_post

from fleet.core._command import TIMED_OUT_RETURNCODE, CommandResult, _awaited


class EnvProtocol(Protocol):
    """Read one process environment variable.

    Implementations MUST normalise a variable set to whitespace to None. An
    exported-but-blank variable is the unset case as far as every caller here
    is concerned, and a fake that returned ``""`` where the real reader
    returns None would let a blank credential reach the queue.
    """

    def __call__(self, name: str) -> str | None:
        """Read it.

        Args:
            name: The variable name.

        Returns:
            Its trimmed value, or None when unset or blank.
        """
        ...


class RunProtocol(Protocol):
    """Runs a local command to completion and collects its output."""

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: int,
        stdin_bytes: bytes | None = None,
        unset_env: Sequence[str] = (),
        set_env: Sequence[tuple[str, str]] = (),
    ) -> CommandResult:
        """Run a command.

        Args:
            argv: Executable and arguments. Never a shell string: a project
                path or a node name is arbitrary text, and shell
                interpretation of it would be a defect rather than a feature.
            timeout_seconds: The deadline. A command still running when it
                passes is ended and reported with ``timed_out`` set; the
                caller names the value because only it knows what the
                command is for. Required, so a call cannot omit the decision.
            stdin_bytes: Bytes to write to the process's standard input, or
                None to give the child a closed stdin (``DEVNULL``) rather
                than this process's own, which under a scheduled task is a
                handle a remote shell can wait on forever.
            unset_env: Names of environment variables the child must NOT
                inherit; every other variable of this process reaches it
                unchanged. Measured 2026-09-17: the agent itself runs under
                ``poetry run``, which exports ``VIRTUAL_ENV``, and poetry
                honours an activated venv over a ``-C`` project's own, so a
                session-audit invocation inherited the fleet venv and could
                not import ``session_audit``. There is no argv-level way to
                drop a variable on Windows, so the seam carries it.
            set_env: ``(name, value)`` pairs the child receives on top of
                what it inherits, applied after ``unset_env``. There is no
                argv-level way to set one on Windows either: a session verb
                puts the committed extraction of session-audit first on
                ``PYTHONPATH`` through this (:mod:`fleet.core.published_tree`).

        Returns:
            Exit status and captured streams. A non-zero status is returned
            rather than raised; the caller decides what it means.
        """


class NowProtocol(Protocol):
    """Reads the wall clock in whole seconds since the epoch."""

    def __call__(self) -> int:
        """Read the current time.

        Returns:
            Whole seconds since the epoch. Whole rather than fractional
            because every consumer is a lease boundary measured in seconds,
            and a float would invite comparisons that differ in their last
            bit between two readers of one file.
        """


class ReadTextProtocol(Protocol):
    """Reads a file's whole contents as UTF-8."""

    def __call__(self, path: pathlib.Path) -> str:
        """Read a file.

        Args:
            path: Absolute path to read.

        Returns:
            The file's contents.

        Raises:
            OSError: If the file cannot be read. Propagated rather than
                translated: the three files this package reads are its own
                records, and one being unreadable is an operational fault
                whose own message names the path and the reason.
        """


class AppendTextProtocol(Protocol):
    """Appends one line to an append-only file, creating it if absent."""

    def __call__(self, path: pathlib.Path, line: str) -> None:
        """Append a line.

        Args:
            path: Absolute path to append to.
            line: The line, without a trailing newline; the implementation
                adds it. Taking the line without one is what makes it
                impossible to append two records that share a line.

        Raises:
            OSError: If the file cannot be written.
        """


class ReadBytesProtocol(Protocol):
    """Reads a file's whole contents as bytes."""

    def __call__(self, path: pathlib.Path) -> bytes:
        """Read a file without decoding it.

        Distinct from :class:`ReadTextProtocol` rather than a mode on it,
        because the one caller is reading a gzip archive and the text reader
        decodes as UTF-8 with replacement -- which is right for a diagnostic
        and silently destroys an archive.

        Args:
            path: Absolute path to read.

        Returns:
            The file's contents.

        Raises:
            OSError: If the file cannot be read.
        """


class FileExistsProtocol(Protocol):
    """Reports whether a path names an existing regular file."""

    def __call__(self, path: pathlib.Path) -> bool:
        """Test a path.

        Args:
            path: Absolute path to test.

        Returns:
            True when the path exists and is a regular file. A directory at
            that path is False rather than an error: the caller's next act
            would be to read it, and letting that fail with its own message
            is better than inventing one here.
        """


class DirectoryExistsProtocol(Protocol):
    """Reports whether a path names an existing directory."""

    def __call__(self, path: pathlib.Path) -> bool:
        """Test a path.

        Separate from :class:`FileExistsProtocol` rather than a flag on it,
        because the two answer different questions and conflating them would
        let a manifest satisfy a check for the directory beside it. The one
        caller is checking that a shared directory a Makefile names is still
        where the Makefile says.

        Args:
            path: Absolute path to test.

        Returns:
            True when the path exists and is a directory.
        """


class TempRootProtocol(Protocol):
    """Reports the system's directory for scratch files."""

    def __call__(self) -> pathlib.Path:
        """Locate the temporary directory.

        A hook because it reads the environment, and this package routes
        every impure act through the seam. The one caller is the staging
        archive, which is scratch by construction: tar writes it, its bytes
        are read once and sent, and nothing opens it again.

        Returns:
            The directory scratch files belong in.
        """


class HostnameProtocol(Protocol):
    """Reports this machine's hostname, lowercased."""

    def __call__(self) -> str:
        """Read the hostname.

        A hook because it reads the machine, and the one caller is the node
        runner's registering check-in, which names the machine it runs on in
        the harness's ``<platform>:<hostname>`` spelling (MCPs board task
        fd5cabfa); a test binds a name rather than reading the test host's.

        Returns:
            The hostname, lowercased as the harness spells it.
        """


class MakeDirectoryProtocol(Protocol):
    """Creates a directory and every parent it needs."""

    def __call__(self, path: pathlib.Path) -> None:
        """Make a directory, tolerating one that is already there.

        The text writers create their own parents, so this exists for the one
        act that cannot: ``tar`` will not create the directory it is asked to
        write an archive into. A workspace whose records directory does not
        exist yet is the ordinary first-run case -- the archive is written
        before any record is appended -- so its first dispatch would otherwise
        fail at tar with a message about a path rather than about staging.

        Args:
            path: Absolute path to create.

        Raises:
            OSError: If it cannot be created.
        """


class WriteTextProtocol(Protocol):
    """Replaces a file's whole contents, creating it if absent."""

    def __call__(self, path: pathlib.Path, text: str) -> None:
        """Write a file.

        Args:
            path: Absolute path to write.
            text: The complete new contents.

        Raises:
            OSError: If the file cannot be written.
        """


def _default_run(
    argv: Sequence[str],
    *,
    timeout_seconds: int,
    stdin_bytes: bytes | None = None,
    unset_env: Sequence[str] = (),
    set_env: Sequence[tuple[str, str]] = (),
) -> CommandResult:
    """Run a command with the real subprocess module.

    Args:
        argv: Executable and arguments.
        timeout_seconds: The deadline; the child is killed when it passes.
        stdin_bytes: Bytes for standard input, or None for a closed stdin.
        unset_env: Variable names withheld from the child's environment.
        set_env: ``(name, value)`` pairs set in the child's environment
            after the withheld names are removed.

    Returns:
        The command's exit status and captured streams, as
        :func:`fleet.core._command._awaited` describes.
    """
    withheld = frozenset(unset_env)
    # The parent environment comes from the monorepo's one permitted reader
    # (the ``env`` guard bans ``os.environ`` everywhere else); the copy it
    # hands back is filtered here, never mutated.
    parent = config_test_hooks.get_environment()
    environment = {name: value for name, value in parent.items() if name not in withheld}
    environment.update(set_env)
    if stdin_bytes is None:
        return _awaited(
            argv,
            stdin_source=subprocess.DEVNULL,
            environment=environment,
            timeout_seconds=timeout_seconds,
        )
    # A FILE THE CHILD READS, NOT BYTES THIS PROCESS WRITES, and the deadline
    # depends on it: ``fleet.core._command`` carries the measurement and why.
    with tempfile.TemporaryFile() as payload:
        payload.write(stdin_bytes)
        payload.seek(0)
        return _awaited(
            argv,
            stdin_source=payload,
            environment=environment,
            timeout_seconds=timeout_seconds,
        )


def _default_now() -> int:
    """Read the real wall clock.

    Returns:
        Whole seconds since the epoch.
    """
    return int(time.time())


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

    The parent directory is created if absent, because the three records this
    package appends to are created by their first write and a workspace
    pointing at a fresh directory is the ordinary first-run case rather than
    a mistake.

    Args:
        path: Absolute path to append to.
        line: The line, without a trailing newline.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


def _default_read_bytes(path: pathlib.Path) -> bytes:
    """Read a real file without decoding it.

    Args:
        path: Absolute path to read.

    Returns:
        The file's contents.
    """
    return path.read_bytes()


def _default_file_exists(path: pathlib.Path) -> bool:
    """Report whether a real path names an existing file.

    Args:
        path: Absolute path to test.

    Returns:
        True when the path exists and is a regular file.
    """
    return path.is_file()


def _default_directory_exists(path: pathlib.Path) -> bool:
    """Report whether a real path names an existing directory.

    Args:
        path: Absolute path to test.

    Returns:
        True when the path exists and is a directory.
    """
    return path.is_dir()


def _default_temp_root() -> pathlib.Path:
    """Locate the real temporary directory.

    Returns:
        The system scratch directory.
    """
    return pathlib.Path(tempfile.gettempdir())


def _default_hostname() -> str:
    """Read the real hostname.

    Returns:
        ``socket.gethostname()`` lowercased, the spelling the session
        observer's script uses on every node.
    """
    return socket.gethostname().lower()


def _default_make_directory(path: pathlib.Path) -> None:
    """Create a real directory and its parents.

    Args:
        path: Absolute path to create.
    """
    path.mkdir(parents=True, exist_ok=True)


def _default_write_text(path: pathlib.Path, text: str) -> None:
    """Replace a real file's contents.

    Args:
        path: Absolute path to write.
        text: The complete new contents.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _default_env(name: str) -> str | None:
    """Read a process environment variable.

    Delegates to ``platform_core.config``, which is the monorepo's single
    permitted reader of the process environment -- the ``env`` guard rule
    names it explicitly rather than exempting it. A second reader here would
    be the fork that rule exists to prevent.

    Args:
        name: The variable name.

    Returns:
        Its trimmed value, or None when unset OR set to whitespace. The
        normalisation is the shared reader's, and it is why callers here test
        only for None.
    """
    return _optional_env_str(name)


run: RunProtocol = _default_run
now: NowProtocol = _default_now
# The dispatch queue's network seam. The implementation is
# ``platform_core.mcp_client.urllib_mcp_post``, shared with tools/board-watch:
# the SEAM belongs to this package (production binds the real thing, a test
# binds a fake) while the JSON-RPC-over-SSE transport behind it is one
# implementation everywhere, down to the error processor that hands back a
# 401 instead of raising.
http_post: McpPostProtocol = urllib_mcp_post
env: EnvProtocol = _default_env
read_text: ReadTextProtocol = _default_read_text
read_bytes: ReadBytesProtocol = _default_read_bytes
file_exists: FileExistsProtocol = _default_file_exists
directory_exists: DirectoryExistsProtocol = _default_directory_exists
make_directory: MakeDirectoryProtocol = _default_make_directory
temp_root: TempRootProtocol = _default_temp_root
append_text: AppendTextProtocol = _default_append_text
write_text: WriteTextProtocol = _default_write_text
hostname: HostnameProtocol = _default_hostname


__all__ = [
    "TIMED_OUT_RETURNCODE",
    "AppendTextProtocol",
    "CommandResult",
    "DirectoryExistsProtocol",
    "FileExistsProtocol",
    "HostnameProtocol",
    "MakeDirectoryProtocol",
    "NowProtocol",
    "ReadBytesProtocol",
    "ReadTextProtocol",
    "RunProtocol",
    "TempRootProtocol",
    "WriteTextProtocol",
    "append_text",
    "directory_exists",
    "file_exists",
    "hostname",
    "make_directory",
    "now",
    "read_bytes",
    "read_text",
    "run",
    "temp_root",
    "write_text",
]
