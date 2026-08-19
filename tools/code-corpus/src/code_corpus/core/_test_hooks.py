"""Dependency-injection seam for the core.

Every non-pure operation the core performs -- running git, reading a file,
writing an output -- is reached through a symbol here, bound to its real
implementation at import time. Production calls the hook directly, so there
is no conditional dispatch and no separate production path; a test rebinds
the symbol to a fake and calls the same code.

The randomness behind the holdout sample and the train-order shuffle is NOT
here. It is passed in as a seed and turned into a ``random.Random`` inside
the function that uses it, because an emission must be reproducible from its
recorded seed alone. Hiding it behind a rebindable hook would make the seed
advisory.
"""

from __future__ import annotations

import pathlib
import subprocess
from collections.abc import Callable, Sequence
from typing import Protocol


class RunGitProtocol(Protocol):
    """Protocol for running a git subcommand inside a repository."""

    def __call__(self, repo_root: pathlib.Path, args: Sequence[str]) -> str:
        """Run git and return its standard output.

        Args:
            repo_root: Repository to run inside.
            args: Git subcommand and its arguments.

        Returns:
            The command's standard output, undecoded beyond UTF-8.

        Raises:
            subprocess.CalledProcessError: If git exits non-zero. A corpus
                whose input state cannot be pinned is not a corpus worth
                emitting, so there is no fallback.
        """
        ...


def _default_run_git(repo_root: pathlib.Path, args: Sequence[str]) -> str:
    """Real implementation running git via subprocess.

    Args:
        repo_root: Repository to run inside.
        args: Git subcommand and its arguments.

    Returns:
        The command's standard output.

    Raises:
        subprocess.CalledProcessError: If git exits non-zero.
    """
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _default_read_bytes(path: pathlib.Path) -> bytes:
    """Read a file's raw bytes.

    Args:
        path: File to read.

    Returns:
        The file's bytes, for the caller to decode or digest.
    """
    return path.read_bytes()


def _default_write_text(path: pathlib.Path, text: str) -> None:
    """Write text as UTF-8 with LF endings, creating parent directories.

    Args:
        path: File to write.
        text: Contents to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


run_git: RunGitProtocol = _default_run_git
read_bytes: Callable[[pathlib.Path], bytes] = _default_read_bytes
write_text: Callable[[pathlib.Path, str], None] = _default_write_text


def reset_hooks() -> None:
    """Rebind every hook to its production implementation."""
    global run_git, read_bytes, write_text
    run_git = _default_run_git
    read_bytes = _default_read_bytes
    write_text = _default_write_text


__all__ = [
    "RunGitProtocol",
    "read_bytes",
    "reset_hooks",
    "run_git",
    "write_text",
]
