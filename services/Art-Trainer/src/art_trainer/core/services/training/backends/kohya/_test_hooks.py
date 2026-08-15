"""Test hooks for the Kohya backend.

Each hook is bound to its real implementation here, so callers invoke it
directly with no conditional. Tests rebind a hook to a fake and call
reset_hooks() to restore the real implementations.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Protocol

from art_trainer.core.services.training.backends.kohya.config import KohyaConfig


class SubprocessResult(Protocol):
    """Protocol for subprocess result."""

    @property
    def returncode(self) -> int:
        """Return code from subprocess."""
        ...

    @property
    def stdout(self) -> str:
        """Standard output from subprocess."""
        ...

    @property
    def stderr(self) -> str:
        """Standard error from subprocess."""
        ...


class SubprocessRunner(Protocol):
    """Protocol for running subprocesses."""

    def __call__(
        self,
        args: list[str],
        *,
        cwd: Path | None = None,
        timeout: int | None = None,
    ) -> SubprocessResult:
        """Run a subprocess.

        Args:
            args: Command and arguments.
            cwd: Working directory.
            timeout: Timeout in seconds.

        Returns:
            Subprocess result with returncode, stdout, stderr.
        """
        ...


class ConfigWriter(Protocol):
    """Protocol for writing TOML config files."""

    def __call__(self, config: KohyaConfig, path: Path) -> None:
        """Write config to TOML file.

        Args:
            config: Kohya configuration TypedDict.
            path: Path to write the TOML file.
        """
        ...


def _real_subprocess_runner(
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout: int | None = None,
) -> SubprocessResult:
    """Run a subprocess and capture its output.

    Args:
        args: Command and arguments.
        cwd: Working directory.
        timeout: Timeout in seconds.

    Returns:
        Subprocess result with returncode, stdout, stderr.
    """
    from art_trainer.core.services.training.backends.kohya.runner import SubprocessResultImpl

    completed = subprocess.run(
        args,
        cwd=str(cwd) if cwd is not None else None,
        timeout=timeout,
        capture_output=True,
        text=True,
        check=False,
    )
    return SubprocessResultImpl(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _real_config_writer(config: KohyaConfig, path: Path) -> None:
    """Write a Kohya config to a TOML file.

    Args:
        config: Kohya configuration TypedDict.
        path: Path to write the TOML file.
    """
    import toml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        toml.dump(config, handle)


class Hooks:
    """Container for test hooks.

    Production code sets these to real implementations.
    Tests set these to fakes for isolation.
    """

    subprocess_runner: SubprocessRunner = _real_subprocess_runner
    config_writer: ConfigWriter = _real_config_writer


def reset_hooks() -> None:
    """Restore every hook to its real implementation."""
    Hooks.subprocess_runner = _real_subprocess_runner
    Hooks.config_writer = _real_config_writer


__all__ = [
    "ConfigWriter",
    "Hooks",
    "SubprocessResult",
    "SubprocessRunner",
    "reset_hooks",
]
