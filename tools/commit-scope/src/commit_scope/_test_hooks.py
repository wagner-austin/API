"""Injection points for everything this package cannot do in a test.

Module-level names, rebound by :mod:`tests.conftest` before each test and
restored after. There is no conditional anywhere in the package asking whether
it is under test: production binds the real implementations at import and a
test binds fakes, and the call site just calls the hook.

Three seams, one per thing that leaves the process: git, the environment
holding the declaration, and standard output. Nothing else is hooked, because
nothing else leaves.

THE GIT SEAM IS ONE CALL, NOT A CLIENT. This package asks git exactly one
question -- what is in the index -- and a general-purpose git wrapper here
would be a second, thinner copy of something no caller needs. The seam is the
answer, not the tool that produced it.

There is deliberately no clock. A hook runs once and exits; nothing here is
scheduled, retried, or timed out, because a check that retries a broken git is
a check that reports success late instead of failing now.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Protocol

from platform_core.config import _optional_env_str
from platform_core.error_codes_tooling import CommitScopeErrorCode
from platform_core.errors import AppError


class RunGitProtocol(Protocol):
    """Run one git command and return its standard output."""

    def __call__(self, arguments: tuple[str, ...]) -> str:
        """Run it.

        Args:
            arguments: Arguments after the program name.

        Returns:
            Standard output, decoded as UTF-8.

        Raises:
            AppError: When git cannot answer.
        """
        ...


class EnvProtocol(Protocol):
    """Read one process environment variable.

    Implementations MUST normalise a variable set to whitespace to None. An
    exported-but-blank declaration is the undeclared case for every caller
    here, and a fake returning ``""`` where the real reader returns None would
    make an unenforced commit look enforced.
    """

    def __call__(self, name: str) -> str | None:
        """Read it.

        Args:
            name: The variable name.

        Returns:
            Its trimmed value, or None when unset or blank.
        """
        ...


class EmitProtocol(Protocol):
    """Write one line of the report."""

    def __call__(self, line: str) -> None:
        """Write it.

        Args:
            line: The line, without a trailing newline.
        """
        ...


def _default_run_git(arguments: tuple[str, ...]) -> str:
    """Run git and return stdout, raising with a specific code when it cannot.

    ``check=True`` rather than inspecting the return code, and the resulting
    ``CalledProcessError`` is translated rather than swallowed: a hook that
    treated a failed git as an empty index would report "nothing staged" and
    wave every commit through, which is worse than the defect this package
    removes.

    The two codes distinguish the two fixes. ``rev-parse`` failing means this
    is not a work tree, which the operator fixes by running the hook somewhere
    else; anything else failing means git itself could not answer.

    Args:
        arguments: Arguments after the program name.

    Returns:
        Standard output, decoded as UTF-8.

    Raises:
        AppError: ``GIT_REPO_ROOT_UNRESOLVED`` when the working directory is
            not inside a git work tree, ``GIT_INDEX_UNREADABLE`` when git
            failed for any other reason.
    """
    completed = subprocess.run(
        ("git", *arguments),
        capture_output=True,
        check=False,
        encoding="utf-8",
    )
    if completed.returncode == 0:
        return completed.stdout
    if "rev-parse" in arguments:
        raise AppError(
            code=CommitScopeErrorCode.GIT_REPO_ROOT_UNRESOLVED,
            message=(
                "git could not resolve a repository root from this directory, so "
                f"there is no index to check. git said: {completed.stderr.strip()!r}"
            ),
        )
    raise AppError(
        code=CommitScopeErrorCode.GIT_INDEX_UNREADABLE,
        message=(
            f"git {' '.join(arguments)} failed with status {completed.returncode}, so "
            f"the staged set is unknown. git said: {completed.stderr.strip()!r}"
        ),
    )


def _default_env(name: str) -> str | None:
    """Read a process environment variable.

    Delegates to ``platform_core.config``, the monorepo's single permitted
    reader of the process environment. A second reader here would be the fork
    the ``env`` guard rule exists to prevent.

    Args:
        name: The variable name.

    Returns:
        Its trimmed value, or None when unset or set to whitespace.
    """
    return _optional_env_str(name)


def _default_emit(line: str) -> None:
    """Write one line to standard output and flush it.

    The flush is required rather than tidy: git shows a hook's output as it
    arrives, and a buffered refusal is a refusal the author has not read yet
    when their editor opens.

    Args:
        line: The line, without a trailing newline.
    """
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


run_git: RunGitProtocol = _default_run_git
env: EnvProtocol = _default_env
emit: EmitProtocol = _default_emit
