"""A candidate's loader workers die with it, whichever way it ends.

The decision logic runs against recording fakes for both platforms; the
real-process cases spawn a candidate that claims ownership the way the
runner's child does and start a descendant inside it, then check the
descendant is gone after the candidate is closed, on whichever platform
this suite runs.
"""

from __future__ import annotations

import multiprocessing
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest
from platform_core.testing_process_group import FakeGroupApi
from platform_core.testing_process_job import FakeJobApi

from handwriting_ai import _test_hooks
from handwriting_ai.training.calibration.process_lifetime import (
    EXIT_TIMEOUT_SECONDS,
    close_candidate,
    isolate_candidate,
)


class _Process:
    """A started candidate whose kill takes effect unless it is stuck."""

    pid = 123

    def __init__(self, stuck: bool = False) -> None:
        self.alive = True
        self.stuck = stuck
        self.joined: list[float | None] = []

    def is_alive(self) -> bool:
        return self.alive

    def kill(self) -> None:
        self.alive = self.stuck

    def join(self, timeout: float | None = None) -> None:
        self.joined.append(timeout)


def test_a_posix_candidate_takes_its_own_group() -> None:
    """On POSIX, isolation is one setsid call."""
    api = FakeGroupApi()
    _test_hooks.os_name = "posix"
    _test_hooks.group_api = lambda: api
    isolate_candidate()
    assert api.calls == ["setsid"]


def test_a_windows_candidate_joins_a_kill_on_close_job() -> None:
    """Off POSIX, isolation is the job sequence."""
    api = FakeJobApi()
    _test_hooks.os_name = "nt"
    _test_hooks.job_api = lambda: api
    isolate_candidate()
    assert api.calls == ["create", "set", "assign"]


def test_a_refused_group_is_not_silently_ignored() -> None:
    """A setsid failure names the errno and stops the candidate."""
    api = FakeGroupApi(session=-1, code=1)
    _test_hooks.os_name = "posix"
    _test_hooks.group_api = lambda: api
    with pytest.raises(RuntimeError, match=r"CALIBRATION_OWNERSHIP: setsid failed \(errno 1\)"):
        isolate_candidate()


def test_a_refused_job_is_not_silently_ignored() -> None:
    """A job assignment failure names the Win32 error and stops the candidate."""
    api = FakeJobApi(assign_ok=False, error=5)
    _test_hooks.os_name = "nt"
    _test_hooks.job_api = lambda: api
    with pytest.raises(RuntimeError, match=r"CALIBRATION_OWNERSHIP.*win32 5"):
        isolate_candidate()
    assert api.calls == ["create", "set", "assign", "error"]


@pytest.mark.parametrize("already_gone", [False, True])
def test_posix_cleanup_kills_joins_and_sweeps_the_group(already_gone: bool) -> None:
    """The candidate is killed, joined, and its group swept; an empty group is a clean sweep."""
    api = FakeGroupApi(kill_result=-1 if already_gone else 0, code=3)
    process = _Process()
    _test_hooks.os_name = "posix"
    _test_hooks.group_api = lambda: api
    close_candidate(process)
    assert not process.alive
    assert process.joined == [EXIT_TIMEOUT_SECONDS]
    assert api.signalled == [(123, 9)]


def test_posix_cleanup_reports_a_sweep_the_kernel_refused() -> None:
    """An errno other than ESRCH is a failed sweep, named."""
    api = FakeGroupApi(kill_result=-1, code=1)
    _test_hooks.os_name = "posix"
    _test_hooks.group_api = lambda: api
    with pytest.raises(RuntimeError, match=r"CALIBRATION_GROUP_SWEEP: killpg failed \(errno 1\)"):
        close_candidate(_Process())


def test_windows_cleanup_kills_and_joins_without_a_sweep() -> None:
    """Off POSIX the job closed with the candidate, so nothing else is signalled."""
    api = FakeGroupApi()
    process = _Process()
    _test_hooks.os_name = "nt"
    _test_hooks.group_api = lambda: api
    close_candidate(process)
    assert not process.alive
    assert process.joined == [EXIT_TIMEOUT_SECONDS]
    assert api.calls == []


def test_an_already_exited_candidate_is_joined_not_killed() -> None:
    """A candidate that finished on its own is reaped, not killed."""
    process = _Process()
    process.alive = False
    _test_hooks.os_name = "nt"
    close_candidate(process)
    assert process.joined == [EXIT_TIMEOUT_SECONDS]


def test_cleanup_reports_a_process_that_did_not_exit() -> None:
    """A candidate alive after the bounded join is named, not forgotten."""
    process = _Process(stuck=True)
    _test_hooks.os_name = "nt"
    with pytest.raises(RuntimeError, match="CALIBRATION_EXIT_TIMEOUT"):
        close_candidate(process)
    assert process.joined == [EXIT_TIMEOUT_SECONDS]


def test_unstarted_candidate_is_rejected() -> None:
    """A process that never started has no pid and nothing to close."""
    process = multiprocessing.get_context("spawn").Process()
    with pytest.raises(RuntimeError, match="CALIBRATION_NOT_STARTED"):
        close_candidate(process)
    process.close()


def _candidate(path: str, exit_after_ready: bool) -> None:
    """The spawned candidate: claim ownership, start a descendant, then wait or exit."""
    isolate_candidate()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    Path(path).write_text(str(child.pid), encoding="utf-8")
    if not exit_after_ready:
        time.sleep(120)


def _descendant_is_gone(descendant: psutil.Process) -> bool:
    """Whether a process has exited (a zombie has, and awaits its reaper)."""
    if not descendant.is_running():
        return True
    return descendant.status() == psutil.STATUS_ZOMBIE


@pytest.mark.parametrize("exit_after_ready", [False, True])
def test_candidate_close_removes_a_real_descendant(tmp_path: Path, exit_after_ready: bool) -> None:
    """A real descendant of a real candidate is gone once the candidate is closed.

    Covers both endings: the candidate killed while it waits, and the
    candidate that exited on its own leaving the descendant behind.
    """
    ready = tmp_path / "descendant.pid"
    process = multiprocessing.get_context("spawn").Process(
        target=_candidate, args=(str(ready), exit_after_ready)
    )
    process.start()
    try:
        deadline = time.monotonic() + 60
        while not ready.exists() and time.monotonic() < deadline:
            assert process.is_alive(), f"candidate exited before ready: {process.exitcode}"
            time.sleep(0.02)
        assert ready.exists(), "candidate never created its descendant"
        descendant = psutil.Process(int(ready.read_text(encoding="utf-8")))
        if exit_after_ready:
            process.join(timeout=EXIT_TIMEOUT_SECONDS)
            assert not process.is_alive()
        close_candidate(process)
        assert not process.is_alive()
        deadline = time.monotonic() + 5
        while not _descendant_is_gone(descendant) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert _descendant_is_gone(descendant), f"descendant {descendant.pid} outlived its owner"
    finally:
        close_candidate(process)
        process.close()
