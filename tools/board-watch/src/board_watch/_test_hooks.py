"""Injection points for everything this package cannot do in a test.

Module-level names, rebound by :mod:`tests.conftest` before each test and
restored after. There is no conditional anywhere in the package asking
whether it is under test: production binds the real implementations at
import and a test binds fakes, and the call site just calls the hook.

Four kinds of seam, one per thing that reaches outside the process: the
network, the environment, the filesystem holding the cursor, and standard
output. Nothing else is hooked, because nothing else leaves.

THE NETWORK SEAM IS DECLARED HERE AND IMPLEMENTED NEXT DOOR. ``http_post``
binds :func:`platform_core.mcp_client.urllib_mcp_post`, lifted out of this
module on 2026-09-05 when a second package needed the same POST. The SEAM
stays here -- production binds the real thing, a test binds a fake, and no
call site asks which -- while the implementation, down to the
non-raising error processor, is shared. A second copy of that would be a
second place for the SSE transport's quirks to be got wrong.

There is deliberately no clock and no sleep. This command reads once and
exits; the interval belongs to the shell loop that calls it, where it is
visible at the call site. See :mod:`board_watch.cli.watch`.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Protocol

from platform_core.config import _optional_env_str
from platform_core.mcp_client import McpPostProtocol, urllib_mcp_post


class EnvProtocol(Protocol):
    """Read one process environment variable.

    Implementations MUST normalise a variable that is set to whitespace to
    None. An exported-but-blank variable is the unset case as far as every
    caller here is concerned, and a fake that returned ``""`` where the real
    reader returns None would let a blank credential reach the board.
    """

    def __call__(self, name: str) -> str | None:
        """Read it.

        Args:
            name: The variable name.

        Returns:
            Its trimmed value, or None when unset or blank.
        """
        ...


class ReadTextProtocol(Protocol):
    """Read a UTF-8 file that is known to exist."""

    def __call__(self, path: pathlib.Path) -> str:
        """Read it.

        Args:
            path: The file.

        Returns:
            Its whole contents.
        """
        ...


class WriteTextProtocol(Protocol):
    """Write a UTF-8 file, creating its parent directory."""

    def __call__(self, path: pathlib.Path, content: str) -> None:
        """Write it.

        Args:
            path: The file.
            content: What to write.
        """
        ...


class FileExistsProtocol(Protocol):
    """Report whether a path is an existing file."""

    def __call__(self, path: pathlib.Path) -> bool:
        """Check it.

        Args:
            path: The path.

        Returns:
            True when it exists and is a file.
        """
        ...


class EmitProtocol(Protocol):
    """Write one line to the watcher's event stream."""

    def __call__(self, line: str) -> None:
        """Write it.

        Args:
            line: The line, without a trailing newline.
        """
        ...


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
        normalisation is the shared reader's, and it is why callers here
        test only for None.
    """
    return _optional_env_str(name)


def _default_read_text(path: pathlib.Path) -> str:
    """Read a UTF-8 file.

    Args:
        path: The file.

    Returns:
        Its whole contents.
    """
    return path.read_text(encoding="utf-8")


def _default_write_text(path: pathlib.Path, content: str) -> None:
    """Write a UTF-8 file, creating its parent directory.

    Args:
        path: The file.
        content: What to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _default_file_exists(path: pathlib.Path) -> bool:
    """Report whether a path is an existing file.

    Args:
        path: The path.

    Returns:
        True when it exists and is a file.
    """
    return path.is_file()


def _default_emit(line: str) -> None:
    """Write one line to standard output and flush it.

    The flush is required, not tidiness. Monitor reads this process's stdout
    as a stream of events, and a buffered line is an event that has not
    happened yet as far as the subscriber is concerned.

    Args:
        line: The line, without a trailing newline.
    """
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


http_post: McpPostProtocol = urllib_mcp_post
env: EnvProtocol = _default_env
read_text: ReadTextProtocol = _default_read_text
write_text: WriteTextProtocol = _default_write_text
file_exists: FileExistsProtocol = _default_file_exists
emit: EmitProtocol = _default_emit
