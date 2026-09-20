"""Shared fakes and the hook reset that keeps tests independent.

Every fake implements the same Protocol the production binding does and
RECORDS what it was asked, so an assertion is about the commands this
package builds and the processes it kills rather than about a patching
library. Nothing here patches anything: the hooks in
:mod:`maketools._test_hooks` are module-level names, and the ``world``
fixture rebinds them for one test and restores the defaults afterwards.

HOOKS ARE RESET AFTER EVERY TEST. A rebinding that leaked would produce a
test that fails only when it runs after a specific other one, and
``-n auto`` reorders freely, so the symptom would be an intermittent
failure whose cause is invisible in the failing test.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from pathlib import Path
from typing import TypedDict

import pytest

from maketools import _test_hooks
from maketools.commands import CommandResult
from maketools.job import JobApi
from maketools.processes import ProcessRow


class InheritingCall(TypedDict):
    """One recorded :func:`maketools._test_hooks.run_inheriting` call.

    Attributes:
        argv: The command.
        cwd: Its directory.
        env: Its environment.
        new_session: Whether a new session was asked for.
    """

    argv: tuple[str, ...]
    cwd: Path
    env: dict[str, str]
    new_session: bool


def ok(stdout: str) -> CommandResult:
    """A successful captured result.

    Args:
        stdout: What it printed.

    Returns:
        The result.
    """
    return CommandResult(returncode=0, stdout=stdout, stderr="")


def failed(returncode: int, stderr: str) -> CommandResult:
    """A failed captured result.

    Args:
        returncode: The status.
        stderr: What it complained.

    Returns:
        The result.
    """
    return CommandResult(returncode=returncode, stdout="", stderr=stderr)


def row(
    pid: int,
    parent_pid: int,
    name: str = "python.exe",
    command_line: str = "python -m pytest",
    *,
    executable: str = "",
    created_unix: float = 0.0,
    cpu_seconds: float = 0.0,
    commit_mb: float = 100.0,
) -> ProcessRow:
    """A process row with defaults that read as a pytest process.

    Args:
        pid: The pid.
        parent_pid: The parent.
        name: The executable name.
        command_line: The command line.
        executable: The executable path.
        created_unix: Start time.
        cpu_seconds: CPU so far.
        commit_mb: Commit.

    Returns:
        The row.
    """
    return ProcessRow(
        pid=pid,
        parent_pid=parent_pid,
        name=name,
        command_line=command_line,
        executable=executable,
        created_unix=created_unix,
        cpu_seconds=cpu_seconds,
        commit_mb=commit_mb,
    )


class FakeJobApi:
    """A job API whose calls succeed or fail by configuration.

    Attributes:
        handle: What ``create_job_object`` answers.
        set_ok: What ``set_kill_on_close`` answers.
        assign_ok: What ``assign_current_process`` answers.
        error: What ``last_error`` answers.
        calls: The method names, in order.
    """

    def __init__(
        self, *, handle: int = 42, set_ok: bool = True, assign_ok: bool = True, error: int = 0
    ) -> None:
        """Configure the answers.

        Args:
            handle: The job handle, 0 for a failed creation.
            set_ok: Whether setting the limit succeeds.
            assign_ok: Whether assignment succeeds.
            error: The Win32 error code.
        """
        self.handle = handle
        self.set_ok = set_ok
        self.assign_ok = assign_ok
        self.error = error
        self.calls: list[str] = []
        self.handles: list[int] = []

    def create_job_object(self) -> int:
        self.calls.append("create")
        return self.handle

    def set_kill_on_close(self, handle: int) -> bool:
        self.calls.append("set")
        self.handles.append(handle)
        return self.set_ok

    def assign_current_process(self, handle: int) -> bool:
        self.calls.append("assign")
        self.handles.append(handle)
        return self.assign_ok

    def last_error(self) -> int:
        self.calls.append("error")
        return self.error


class World:
    """Every hook, faked and recording.

    Attributes:
        inheriting_calls: Every ``run_inheriting`` call.
        inheriting_code: What ``run_inheriting`` returns.
        capturing_answers: ``run_capturing`` answers, by argv.
        captured: Every ``run_capturing`` argv, in order.
        now_value: What the clock answers.
        slept: Every ``sleep`` duration.
        lines: Every stdout line.
        errors: Every stderr line.
        environment: What ``environ`` answers.
        platform_name: What ``platform`` answers.
        tables: Process snapshots, consumed one per ``process_table`` call;
            the last one repeats.
        killed: Every pid ``kill`` received.
        refuse_kill: Pids ``kill`` raises ``OSError`` for.
        alive: Pids ``process_alive`` answers True for.
        removed_trees: Every ``remove_tree`` path.
        removed_files: Every ``remove_file`` path.
        pid: What ``process_id`` answers.
        token_value: What ``token`` answers.
        tracked: What ``tracked_files`` answers.
        job: What ``job_api`` answers.
    """

    def __init__(self) -> None:
        """Start with a quiet world."""
        self.inheriting_calls: list[InheritingCall] = []
        self.inheriting_code = 0
        self.inheriting_codes: list[int] = []
        self.capturing_answers: dict[tuple[str, ...], CommandResult] = {}
        self.captured: list[tuple[str, ...]] = []
        self.now_value = 1_000_000.0
        self.slept: list[float] = []
        self.lines: list[str] = []
        self.errors: list[str] = []
        self.environment: dict[str, str] = {"PATH": "/bin"}
        self.platform_name = "linux"
        self.tables: list[list[ProcessRow]] = [[]]
        self.killed: list[int] = []
        self.refuse_kill: set[int] = set()
        self.alive: set[int] = set()
        self.removed_trees: list[Path] = []
        self.removed_files: list[Path] = []
        self.pid = 4242
        self.token_value = "deadbeef"
        self.drawn = 27777
        self.draws: list[tuple[int, int]] = []
        self.tracked: list[Path] = []
        self.job = FakeJobApi()

    def run_inheriting(
        self, argv: Sequence[str], *, cwd: Path, env: Mapping[str, str], new_session: bool
    ) -> int:
        self.inheriting_calls.append(
            InheritingCall(argv=tuple(argv), cwd=cwd, env=dict(env), new_session=new_session)
        )
        if self.inheriting_codes:
            return self.inheriting_codes.pop(0)
        return self.inheriting_code

    def run_capturing(self, argv: Sequence[str], *, cwd: Path) -> CommandResult:
        key = tuple(argv)
        self.captured.append(key)
        if key not in self.capturing_answers:
            raise AssertionError(f"no scripted answer for {key} in {cwd}")
        return self.capturing_answers[key]

    def now(self) -> float:
        return self.now_value

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)

    def write_line(self, line: str) -> None:
        self.lines.append(line)

    def write_error(self, line: str) -> None:
        self.errors.append(line)

    def environ(self) -> dict[str, str]:
        return dict(self.environment)

    def platform(self) -> str:
        return self.platform_name

    def process_table(self) -> Sequence[ProcessRow]:
        if len(self.tables) > 1:
            return self.tables.pop(0)
        return self.tables[0]

    def kill(self, pid: int) -> None:
        self.killed.append(pid)
        if pid in self.refuse_kill:
            raise OSError(f"refused to kill {pid}")

    def process_alive(self, pid: int) -> bool:
        return pid in self.alive

    def remove_tree(self, path: Path) -> None:
        self.removed_trees.append(path)

    def remove_file(self, path: Path) -> None:
        self.removed_files.append(path)
        path.unlink()

    def process_id(self) -> int:
        return self.pid

    def token(self) -> str:
        return self.token_value

    def draw(self, low: int, high: int) -> int:
        self.draws.append((low, high))
        return self.drawn

    def tracked_files(self, repo_root: Path, pathspec: str) -> Sequence[Path]:
        return [repo_root / path for path in self.tracked]

    def job_api(self) -> JobApi:
        return self.job

    def bind(self) -> None:
        """Point every hook at this world."""
        _test_hooks.run_inheriting = self.run_inheriting
        _test_hooks.run_capturing = self.run_capturing
        _test_hooks.now = self.now
        _test_hooks.sleep = self.sleep
        _test_hooks.write_line = self.write_line
        _test_hooks.write_error = self.write_error
        _test_hooks.environ = self.environ
        _test_hooks.platform = self.platform
        _test_hooks.process_table = self.process_table
        _test_hooks.kill = self.kill
        _test_hooks.process_alive = self.process_alive
        _test_hooks.remove_tree = self.remove_tree
        _test_hooks.remove_file = self.remove_file
        _test_hooks.process_id = self.process_id
        _test_hooks.token = self.token
        _test_hooks.draw = self.draw
        _test_hooks.tracked_files = self.tracked_files
        _test_hooks.job_api = self.job_api


def restore_defaults() -> None:
    """Point every hook back at its real implementation."""
    _test_hooks.run_inheriting = _test_hooks._default_run_inheriting
    _test_hooks.run_capturing = _test_hooks._default_run_capturing
    _test_hooks.now = _test_hooks._default_now
    _test_hooks.sleep = _test_hooks._default_sleep
    _test_hooks.write_line = _test_hooks._default_write_line
    _test_hooks.write_error = _test_hooks._default_write_error
    _test_hooks.environ = _test_hooks._default_environ
    _test_hooks.platform = _test_hooks._default_platform
    _test_hooks.process_table = _test_hooks._default_process_table
    _test_hooks.kill = _test_hooks._default_kill
    _test_hooks.process_alive = _test_hooks._default_process_alive
    _test_hooks.remove_tree = _test_hooks._default_remove_tree
    _test_hooks.remove_file = _test_hooks._default_remove_file
    _test_hooks.process_id = _test_hooks._default_process_id
    _test_hooks.token = _test_hooks._default_token
    _test_hooks.draw = _test_hooks._default_draw
    _test_hooks.tracked_files = _test_hooks._default_tracked_files
    _test_hooks.job_api = _test_hooks._default_job_api


@pytest.fixture()
def world() -> Generator[World, None, None]:
    """A recording world bound for the test's duration."""
    fake = World()
    fake.bind()
    yield fake
    restore_defaults()
