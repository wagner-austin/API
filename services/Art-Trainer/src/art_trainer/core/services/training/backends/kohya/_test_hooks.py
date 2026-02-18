"""Test hooks for Kohya backend.

Follows the covenant pattern: production code sets hooks to real implementations,
tests set hooks to fakes for isolation.
"""

from __future__ import annotations

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


class Hooks:
    """Container for test hooks.

    Production code sets these to real implementations.
    Tests set these to fakes for isolation.
    """

    subprocess_runner: SubprocessRunner | None = None
    config_writer: ConfigWriter | None = None


def reset_hooks() -> None:
    """Reset all hooks to None (for test cleanup)."""
    Hooks.subprocess_runner = None
    Hooks.config_writer = None


__all__ = [
    "ConfigWriter",
    "Hooks",
    "SubprocessResult",
    "SubprocessRunner",
    "reset_hooks",
]
