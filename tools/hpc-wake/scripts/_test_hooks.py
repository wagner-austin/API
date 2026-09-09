"""Internal hooks for dependency injection (underscore = private).

One seam: the process runner the scheduled entry point hands the bridge
command to. Production leaves it bound to :func:`subprocess.run`; tests
rebind it to a recording fake, which is how the entry point's own lines
are covered without this suite ever posting to the board.
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


run_process: RunProcess = subprocess.run
