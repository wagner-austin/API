"""Internal hooks for dependency injection (underscore = private).

Two seams. The process runner the scheduled entry point hands the bridge
command to: production leaves it bound to :func:`subprocess.run`; tests
rebind it to a recording fake, which is how the entry point's own lines
are covered without this suite ever posting to the board. And the clock,
since 2026-09-21, because the tick now writes a health record carrying an
instant, and a test asserting a stamp it cannot control asserts nothing.
"""

from __future__ import annotations

import datetime
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
    """The slice of :func:`subprocess.run` the entry point uses."""

    def __call__(
        self,
        args: Sequence[str],
        *,
        cwd: pathlib.Path,
        env: Mapping[str, str],
        capture_output: bool,
        text: bool,
    ) -> CompletedProto:
        """Run the command to completion and return its outcome."""
        ...


class Now(Protocol):
    """The clock the tick stamps its log header and health record with."""

    def __call__(self) -> datetime.datetime:
        """Return the current instant, timezone-aware UTC."""
        ...


def _utc_now() -> datetime.datetime:
    """Production's clock.

    Returns:
        The current instant, timezone-aware UTC.
    """
    return datetime.datetime.now(datetime.UTC)


run_process: RunProcess = subprocess.run
now: Now = _utc_now
