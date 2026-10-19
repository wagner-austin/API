"""Internal hooks for dependency injection (underscore = private).

One seam: the process runner the scheduled entry point hands the bridge
command to. Production binds :func:`_default_run_process`; tests rebind it
to a recording fake, which is how the entry point's own lines are covered
without this suite ever posting to the board. The clock seam beside it went
with the health record it stamped (MCPs board task 6a2eb905): the log
header is the one stamp left, and no test controls it.
"""

from __future__ import annotations

import pathlib
import subprocess
from collections.abc import Mapping, Sequence
from typing import Protocol


class CompletedProto(Protocol):
    """The three fields the entry point reads off a finished process."""

    @property
    def stdout(self) -> str:
        """Captured standard output."""

    @property
    def stderr(self) -> str:
        """Captured standard error."""

    @property
    def returncode(self) -> int:
        """The process's exit status."""


class RunProcess(Protocol):
    """Runs one publisher to completion, both streams captured as text."""

    def __call__(
        self,
        args: Sequence[str],
        *,
        cwd: pathlib.Path,
        env: Mapping[str, str],
        timeout: int,
    ) -> CompletedProto:
        """Run the command to completion and return its outcome.

        Args:
            args: The publisher's command, program first.
            cwd: The directory ``poetry run`` resolves its project from.
            env: The child's complete environment.
            timeout: Seconds before the child is killed.

        Returns:
            The finished process, whatever its status.

        Raises:
            subprocess.TimeoutExpired: When it outlasts ``timeout``.
        """
        ...


def _default_run_process(
    args: Sequence[str],
    *,
    cwd: pathlib.Path,
    env: Mapping[str, str],
    timeout: int,
) -> CompletedProto:
    """Run a real publisher, capturing both streams as text.

    A named implementation rather than ``subprocess.run`` bound straight to
    the hook, so a test can run the production runner itself through a
    failure (``effect-seam-twin``, board task cc7222ca). It also carries the
    deadline the bare binding never had: a publisher that never returned
    held the pump forever, the unbounded-child shape that stalled the fleet
    queue for three days in September 2026.

    Args:
        args: The publisher's command, program first.
        cwd: The directory it runs in.
        env: Its complete environment.
        timeout: Seconds before it is killed.

    Returns:
        The finished process, whatever its status.

    Raises:
        subprocess.TimeoutExpired: When it outlasts ``timeout``.
    """
    return subprocess.run(
        list(args),
        cwd=cwd,
        env=dict(env),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


run_process: RunProcess = _default_run_process
