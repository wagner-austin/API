"""Test fakes for Kohya backend testing.

This module provides fake implementations of Kohya backend dependencies
for use in tests.
"""

from __future__ import annotations

from pathlib import Path

from art_trainer.core.services.training.backends.kohya.config import KohyaConfig


class FakeSubprocessResult:
    """Fake subprocess result for testing.

    Simulates subprocess execution results without running actual processes.
    """

    _returncode: int
    _stdout: str
    _stderr: str

    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        """Initialize fake subprocess result.

        Args:
            returncode: Return code to simulate.
            stdout: Standard output to simulate.
            stderr: Standard error to simulate.
        """
        self._returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    @property
    def returncode(self) -> int:
        """Return code from subprocess."""
        return self._returncode

    @property
    def stdout(self) -> str:
        """Standard output from subprocess."""
        return self._stdout

    @property
    def stderr(self) -> str:
        """Standard error from subprocess."""
        return self._stderr


class FakeKohyaRunner:
    """Fake subprocess runner for Kohya testing.

    Records all subprocess calls and returns configurable results.
    """

    calls: list[tuple[list[str], Path | None]]
    _should_succeed: bool
    _final_loss: float | None
    _returncode: int
    _stderr: str

    def __init__(
        self,
        *,
        should_succeed: bool = True,
        final_loss: float | None = 0.05,
        returncode: int | None = None,
        stderr: str = "",
    ) -> None:
        """Initialize fake Kohya runner.

        Args:
            should_succeed: Whether training should succeed.
            final_loss: Final loss value to report, or None for no loss output.
            returncode: Override return code (default: 0 if success, 1 otherwise).
            stderr: Standard error to return.
        """
        self.calls = []
        self._should_succeed = should_succeed
        self._final_loss = final_loss
        self._returncode = returncode if returncode is not None else (0 if should_succeed else 1)
        self._stderr = stderr

    def __call__(
        self,
        args: list[str],
        *,
        cwd: Path | None = None,
        timeout: int | None = None,
    ) -> FakeSubprocessResult:
        """Simulate running a subprocess.

        Args:
            args: Command and arguments.
            cwd: Working directory.
            timeout: Timeout in seconds.

        Returns:
            Fake subprocess result.
        """
        self.calls.append((args, cwd))

        if self._should_succeed:
            if self._final_loss is not None:
                stdout = f"Training complete. loss={self._final_loss}"
            else:
                stdout = "Training complete."
        else:
            stdout = "Training failed"

        return FakeSubprocessResult(
            returncode=self._returncode,
            stdout=stdout,
            stderr=self._stderr,
        )


class FakeConfigWriter:
    """Fake config writer for Kohya testing.

    Records all config writes without writing to disk.
    """

    written_configs: list[tuple[KohyaConfig, Path]]

    def __init__(self) -> None:
        """Initialize fake config writer."""
        self.written_configs = []

    def __call__(self, config: KohyaConfig, path: Path) -> None:
        """Record config write.

        Args:
            config: Kohya configuration.
            path: Path where config would be written.
        """
        self.written_configs.append((config, path))


__all__ = [
    "FakeConfigWriter",
    "FakeKohyaRunner",
    "FakeSubprocessResult",
]
