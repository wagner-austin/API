"""Shared stand-ins for driving capture helpers with real child processes.

The DI seam chooses WHICH process runs, never whether one does: every
handle these hand the capture code wraps a real ``sys.executable``
child, so terminate/kill/poll semantics are the operating system's own.
Shared by the display, encoder and sound-server tests, which drive the
same spawn seam.
"""

from __future__ import annotations

import sys
from pathlib import Path

from tankpit_bot.stream import _test_hooks as stream_hooks
from tankpit_bot.stream.types import StreamConfigDict
from tests._capture_process import PatientCaptureProcess


def stream_config(hls_dir: Path) -> StreamConfigDict:
    """Build one capture configuration rooted in a test directory.

    Args:
        hls_dir: Where the encoder would write.

    Returns:
        The configuration.
    """
    return StreamConfigDict(
        display=91,
        width=704,
        height=544,
        scale=2,
        fps=30,
        bitrate_kbps=1000,
        segment_seconds=2,
        hls_dir=str(hls_dir),
    )


def sleeper_argv() -> list[str]:
    """A child that runs until terminated.

    Returns:
        Argv for a 60-second sleeper.
    """
    return [sys.executable, "-c", "import time; time.sleep(60)"]


class SubstitutingSpawner:
    """Spawn REAL children while recording the commands asked for.

    The capture code asks for ``Xvfb``/``pulseaudio``/``ffmpeg``, which
    do not exist on the test host; this seam records that request and
    runs a ``sys.executable`` stand-in through the REAL spawner, so
    log-file plumbing and process semantics stay production code. Each
    handle is a :class:`PatientCaptureProcess`, so the bound of every
    wait is recorded for the test to assert.
    """

    def __init__(self, argv_per_call: list[list[str]]) -> None:
        """Bind the substitute argv for each successive call.

        Args:
            argv_per_call: What to actually run, call by call.
        """
        self._argv_per_call = argv_per_call
        self.commands: list[list[str]] = []
        self.log_paths: list[Path] = []
        self.processes: list[PatientCaptureProcess] = []

    def __call__(self, command: list[str], log_path: Path) -> stream_hooks.CaptureProcessProtocol:
        """Record the request and spawn the substitute.

        Args:
            command: What the capture code wanted to run.
            log_path: Where it wanted the console.

        Returns:
            The substitute process handle.
        """
        self.commands.append(command)
        self.log_paths.append(log_path)
        process = PatientCaptureProcess(
            stream_hooks._real_spawn_capture_process(
                self._argv_per_call[len(self.processes)], log_path
            )
        )
        self.processes.append(process)
        return process


class SteppingClock:
    """Monotonic clock that advances a fixed step per read."""

    def __init__(self, step: float) -> None:
        """Start at zero, advancing ``step`` per read.

        Args:
            step: Seconds each read advances.
        """
        self._now = 0.0
        self._step = step

    def __call__(self) -> float:
        """Read and advance the clock.

        Returns:
            The pre-advance reading.
        """
        now = self._now
        self._now += self._step
        return now


def noop_sleep(seconds: float) -> None:
    """Sleep hook that spends no wall clock.

    Args:
        seconds: Ignored.
    """
    del seconds


__all__ = [
    "SteppingClock",
    "SubstitutingSpawner",
    "noop_sleep",
    "sleeper_argv",
    "stream_config",
]
