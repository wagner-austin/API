"""Test-hook surface for the display-capture pipeline.

Production code uses the real implementations; tests rebind these
module-level symbols to drive the capture lifecycle with processes and
clocks they control. The processes tests inject are REAL processes
(``sys.executable -c`` children), never fakes — the seam exists so a
test can choose WHICH process, not so it can pretend one ran.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Protocol


class CaptureProcessProtocol(Protocol):
    """The slice of :class:`subprocess.Popen` the capture lifecycle uses."""

    @property
    def pid(self) -> int:
        """The process id."""
        ...

    def poll(self) -> int | None:
        """Return the exit code, or ``None`` while still running."""
        ...

    def terminate(self) -> None:
        """Ask the process to exit (SIGTERM)."""
        ...

    def kill(self) -> None:
        """End the process unconditionally (SIGKILL)."""
        ...

    def wait(self, timeout: float) -> int:
        """Block until exit and return the exit code.

        Args:
            timeout: Seconds to wait before giving up.

        Returns:
            The exit code.

        Raises:
            subprocess.TimeoutExpired: Still running after ``timeout``.
        """
        ...


class SpawnCaptureProcessProtocol(Protocol):
    """Spawns one capture helper process (Xvfb or ffmpeg)."""

    def __call__(self, command: list[str], log_path: Path) -> CaptureProcessProtocol:
        """Start the process with its console appended to a log file.

        Args:
            command: Full argv, program first.
            log_path: File both stdout and stderr append to.

        Returns:
            The spawned process handle.

        Raises:
            OSError: The program is missing or cannot be started.
        """
        ...


def _real_spawn_capture_process(command: list[str], log_path: Path) -> CaptureProcessProtocol:
    """Spawn a capture helper with its console captured to a file.

    A file rather than ``DEVNULL`` for the same reason the fleet
    spawner records child consoles: the one line that explains a
    dead encoder ("Cannot open display") is printed to stderr as the
    process dies, after which nothing else will ever say why.

    Args:
        command: Full argv, program first.
        log_path: File both stdout and stderr append to.

    Returns:
        The spawned process handle.

    Raises:
        OSError: The program is missing or cannot be started.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as console:
        return subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=console,
            stderr=subprocess.STDOUT,
        )


class MonotonicSecondsProtocol(Protocol):
    """Reads a monotonic clock."""

    def __call__(self) -> float:
        """Return monotonic seconds."""
        ...


class SleepSecondsProtocol(Protocol):
    """Blocks the calling thread for a while."""

    def __call__(self, seconds: float) -> None:
        """Sleep.

        Args:
            seconds: How long to block for.
        """
        ...


#: Spawns Xvfb / ffmpeg. Tests rebind to spawn processes they control.
spawn_capture_process: SpawnCaptureProcessProtocol = _real_spawn_capture_process


def _real_sleep_seconds(seconds: float) -> None:
    """Production sleep — :func:`time.sleep` behind the protocol's
    keyword-capable signature (the stdlib's parameter is positional-only).

    Args:
        seconds: How long to block for.
    """
    time.sleep(seconds)


def _real_monotonic_seconds() -> float:
    """Production clock — :func:`time.monotonic`.

    Returns:
        Monotonic seconds.
    """
    return time.monotonic()


#: Monotonic clock the display-readiness deadline is measured on.
monotonic_seconds: MonotonicSecondsProtocol = _real_monotonic_seconds

#: Poll-loop sleep. Tests rebind to advance their injected clock.
sleep_seconds: SleepSecondsProtocol = _real_sleep_seconds


__all__ = [
    "CaptureProcessProtocol",
    "MonotonicSecondsProtocol",
    "SleepSecondsProtocol",
    "SpawnCaptureProcessProtocol",
    "_real_monotonic_seconds",
    "_real_sleep_seconds",
    "_real_spawn_capture_process",
    "monotonic_seconds",
    "sleep_seconds",
    "spawn_capture_process",
]
