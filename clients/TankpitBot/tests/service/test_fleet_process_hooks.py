"""The production process seams, against real OS processes.

Adoption's whole claim is about processes the manager did not fork:
that it can find one by pid, refuse a pid that now belongs to
something else, and still read the exit code after it dies. None of
that is testable against a double -- the behaviour under test IS the
operating system's -- so these start real children and watch them.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Generator

import psutil
import pytest

from tankpit_bot.service._test_hooks import (
    SpawnedProcessProtocol,
    _real_open_adopted_process,
    _real_process_identity,
    _real_sleep_seconds,
)

_SLEEPER = "import time; time.sleep(120)"


def _identity_of(pid: int) -> float:
    """Return a live pid's creation time, failing loudly when absent.

    Args:
        pid: Process id expected to be running.

    Returns:
        The creation time in epoch seconds.

    Raises:
        AssertionError: If the seam reports no such process.
    """
    created_at = _real_process_identity(pid)
    if created_at is None:
        raise AssertionError(f"pid {pid} is running but reported no creation time")
    return created_at


def _adopt(pid: int, created_at: float) -> SpawnedProcessProtocol:
    """Adopt a process, failing loudly when the seam refuses.

    Args:
        pid: Process id to adopt.
        created_at: Its recorded creation time.

    Returns:
        The adopted handle.

    Raises:
        AssertionError: If the seam declined to adopt.
    """
    adopted = _real_open_adopted_process(pid, created_at)
    if adopted is None:
        raise AssertionError(f"pid {pid} is running but was not adopted")
    return adopted


@pytest.fixture()
def sleeper() -> Generator[subprocess.Popen[bytes], None, None]:
    """Start a real child process that idles until killed.

    Yields:
        The running child.
    """
    process = subprocess.Popen([sys.executable, "-c", _SLEEPER])
    try:
        yield process
    finally:
        process.kill()
        process.wait(timeout=30)


def test_identity_reads_a_live_process_creation_time(
    sleeper: subprocess.Popen[bytes],
) -> None:
    """A running pid has a creation time, and it is stable."""
    first = _identity_of(sleeper.pid)
    second = _identity_of(sleeper.pid)

    assert first == second
    assert first > 0.0


def test_identity_of_a_pid_that_is_not_running_is_absent() -> None:
    """A child that died before it could be recorded has no identity.

    That is the case the spawn path treats as "nothing to adopt
    later", rather than as a failure to record.
    """
    process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(0)"])
    process.wait(timeout=30)

    assert _real_process_identity(process.pid) is None


def test_a_live_process_is_adopted_and_reports_itself_running(
    sleeper: subprocess.Popen[bytes],
) -> None:
    """The handle satisfies the same surface a spawned child does."""
    adopted = _adopt(sleeper.pid, _identity_of(sleeper.pid))

    assert adopted.pid == sleeper.pid
    assert adopted.is_running() is True
    assert adopted.exit_code() is None


def test_an_adopted_process_reports_its_exit_code_after_it_dies(
    sleeper: subprocess.Popen[bytes],
) -> None:
    """The handle outlives the process, which is why the code survives.

    A pid looked up fresh at this point would already be gone; holding
    the handle is what keeps the exit observable.
    """
    adopted = _adopt(sleeper.pid, _identity_of(sleeper.pid))

    sleeper.kill()
    expected = sleeper.wait(timeout=30)

    # The adopted handle agrees with the parent's own handle, and keeps
    # agreeing: neither answer is a one-shot reading.
    assert adopted.is_running() is False
    assert adopted.exit_code() == expected
    assert adopted.is_running() is False
    assert adopted.exit_code() == expected


def test_liveness_never_depends_on_recovering_an_exit_code(
    sleeper: subprocess.Popen[bytes],
) -> None:
    """A bot that has ended reads as not-running even with no code.

    The first live drain wedged on exactly this: the manager read a
    bot that had already landed as still running and waited forever,
    because "alive" was derived from "no exit code yet". The two
    questions are asked separately now, and this pins that a missing
    code can never masquerade as a running tank.
    """
    adopted = _adopt(sleeper.pid, _identity_of(sleeper.pid))
    assert adopted.is_running() is True

    # ONE handle, resolved while the process is still alive. Looking the
    # pid up a second time after the kill is a race the test loses at
    # random: the child can be fully reaped between the two calls, and
    # ``psutil.Process`` on a gone pid raises NoSuchProcess — a red build
    # describing nothing about the code under test. An existing handle
    # waits on an already-exited process without complaint.
    victim = psutil.Process(sleeper.pid)
    victim.kill()
    victim.wait(timeout=30)

    assert adopted.is_running() is False
    # Whatever the code turns out to be -- recoverable or not -- it is
    # never allowed to change the liveness answer.
    _ = adopted.exit_code()
    assert adopted.is_running() is False


def test_a_pid_that_is_not_running_cannot_be_adopted() -> None:
    """Nothing under the pid means the bot finished unsupervised."""
    process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(0)"])
    process.wait(timeout=30)

    assert _real_open_adopted_process(process.pid, 1.0) is None


def test_a_recycled_pid_is_refused(sleeper: subprocess.Popen[bytes]) -> None:
    """A live pid whose creation time differs is not our bot.

    Windows reuses pids. Without this check a restarted manager would
    adopt whatever inherited the number, then refuse to restart the
    instance forever because its imaginary bot never exits.
    """
    created_at = _identity_of(sleeper.pid)

    assert _real_open_adopted_process(sleeper.pid, created_at + 1.0) is None


def test_the_real_sleep_returns_after_waiting() -> None:
    """The production sleep is a real, if very short, wait."""
    _real_sleep_seconds(0.01)
