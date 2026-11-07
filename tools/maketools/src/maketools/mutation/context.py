"""What one run of the mutation step is bound to, and how it runs a command (MCPs task e52e5148).

Every command the step runs (git, cosmic-ray, coverage, pytest) goes
through :data:`maketools._test_hooks.run_answering`, in the package
directory, under :data:`CHILD_WALL_SECONDS`. The refusals carry
:class:`platform_core.error_codes_tooling.MutationErrorCode`, whose values
are MCPs' own.
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, NamedTuple

from platform_core.error_codes_tooling import MutationErrorCode
from platform_core.errors import AppError

from maketools import _test_hooks
from maketools.bounded_spans import MUTATION_BOUND_SECONDS
from maketools.commands import CommandResult

#: The wall on any one command the step runs: its whole span's bound, so a
#: single hung command fails as the span past its bound would, and sooner.
CHILD_WALL_SECONDS: Final[int] = MUTATION_BOUND_SECONDS


class MutationRun(NamedTuple):
    """What one run of the step is bound to.

    Attributes:
        package_dir: The package it mutates, where every command runs.
        python: The interpreter of the package's environment, beside which
            cosmic-ray is installed.
        environment: The suite's environment: its coverage file, thread
            caps and caches, which every command the step runs inherits.
        run_dir: The run's scratch directory, removed after it.
    """

    package_dir: Path
    python: str
    environment: Mapping[str, str]
    run_dir: Path


def tail(result: CommandResult) -> str:
    """How a command ended, for a refusal.

    Args:
        result: How it ended.

    Returns:
        Its exit and the last 20 lines of its output and errors.
    """
    lines = (result["stdout"] + result["stderr"]).splitlines()[-20:]
    return f"exited {result['returncode']}: " + "\n".join(lines)


def run_command(run: MutationRun, argv: Sequence[str], env: Mapping[str, str]) -> CommandResult:
    """Run one command in the package directory.

    Args:
        run: The step's run.
        argv: The command.
        env: Its whole environment.

    Returns:
        How it ended.
    """
    return _test_hooks.run_answering(
        argv, cwd=run.package_dir, env=env, timeout_seconds=CHILD_WALL_SECONDS
    )


def run_or_refuse(
    run: MutationRun, argv: Sequence[str], env: Mapping[str, str], code: MutationErrorCode
) -> str:
    """Run a command and require it to succeed.

    Args:
        run: The step's run.
        argv: The command.
        env: Its whole environment.
        code: The refusal its failure is.

    Returns:
        What it printed.

    Raises:
        AppError: ``code``, naming the command and its last lines.
    """
    result = run_command(run, argv, env)
    if result["returncode"]:
        raise AppError(code, f"{shlex.join(argv)} {tail(result)}")
    return result["stdout"]


__all__ = ["CHILD_WALL_SECONDS", "MutationRun", "run_command", "run_or_refuse", "tail"]
