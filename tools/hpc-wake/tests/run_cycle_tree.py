"""The scheduled entry point's test world: a staged package tree, a recording
process runner, a movable clock and readers for the health record a tick
writes.

Shared by ``test_run_cycle.py`` and ``test_run_cycle_health.py``; the
``runner`` and ``clock`` fixtures are registered in ``conftest.py`` from the
generators here. The one fake is the process runner rebound through
``scripts._test_hooks``, which is how the suites cover the entry point
without posting to the board.
"""

from __future__ import annotations

import datetime
import pathlib
from collections.abc import Generator, Mapping, Sequence

from scripts import _test_hooks, run_cycle
from scripts.pump_health import HEALTH_FILENAME, PublisherHealth, read_health


class Completed:
    """The three fields the entry point reads, as plain data."""

    def __init__(self, stdout: str, stderr: str, returncode: int) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class RecordingRunner:
    """A process runner that records calls and returns scripted results in order."""

    def __init__(self, results: Sequence[Completed]) -> None:
        self.results = list(results)
        self.calls: list[tuple[Sequence[str], pathlib.Path, dict[str, str]]] = []

    def __call__(
        self,
        args: Sequence[str],
        *,
        cwd: pathlib.Path,
        env: Mapping[str, str],
        timeout: int,
    ) -> Completed:
        assert timeout == run_cycle.PUBLISHER_WALL_SECONDS
        self.calls.append((args, cwd, dict(env)))
        return self.results[len(self.calls) - 1]


def make_runner() -> Generator[RecordingRunner, None, None]:
    """Install a recording runner, and put the real one back afterwards.

    Scripted with one result per publisher, distinguishable by content so
    ordering assertions mean something.
    """
    fake = RecordingRunner(
        [
            Completed("hpc-out\n", "hpc-err\n", 0),
            Completed("ci-out\n", "ci-err\n", 0),
            Completed("lock-out\n", "lock-err\n", 0),
            Completed("health-out\n", "health-err\n", 0),
            Completed("tunnel-out\n", "tunnel-err\n", 0),
            Completed("diphtheria-out\n", "diphtheria-err\n", 0),
            Completed("actions-out\n", "actions-err\n", 0),
            Completed("rcs-out\n", "rcs-err\n", 0),
            Completed("login-out\n", "login-err\n", 0),
            Completed("search-out\n", "search-err\n", 0),
        ]
    )
    _test_hooks.run_process = fake
    yield fake
    _test_hooks.run_process = _test_hooks._default_run_process


class FrozenClock:
    """A clock a test can move, standing in for ``_test_hooks.now``."""

    def __init__(self, instant: datetime.datetime) -> None:
        self.instant = instant

    def __call__(self) -> datetime.datetime:
        """Return the instant this clock is currently set to.

        Returns:
            The instant.
        """
        return self.instant


def make_clock() -> Generator[FrozenClock, None, None]:
    """Freeze the tick's clock, and put the real one back afterwards.

    The header stamp and the health record both read it, so a test that
    asserted either against the wall clock would assert nothing.
    """
    frozen = FrozenClock(datetime.datetime(2026, 9, 21, 20, 43, 15, tzinfo=datetime.UTC))
    _test_hooks.now = frozen
    yield frozen
    _test_hooks.now = _test_hooks._utc_now


def health_path(root: pathlib.Path) -> pathlib.Path:
    """Where the tick writes its health record.

    Args:
        root: The staged package root.

    Returns:
        The record's path.
    """
    return root / "runs" / HEALTH_FILENAME


def publisher(root: pathlib.Path, name: str) -> PublisherHealth:
    """One publisher's row out of the record the tick just wrote.

    Read through the module's own reader rather than by splitting the file
    here, so a rendering this reader cannot read back fails the test.

    Args:
        root: The staged package root.
        name: The publisher's marker.

    Returns:
        Its entry.
    """
    return read_health(health_path(root))[name]


def written_line(root: pathlib.Path) -> str:
    """The record's ``written`` field, as the file spells it.

    Args:
        root: The staged package root.

    Returns:
        The instant.
    """
    lines = health_path(root).read_text(encoding="utf-8").splitlines()
    return next(line.split("\t")[1] for line in lines if line.startswith("written\t"))


def staged_root(tmp_path: pathlib.Path, env_body: str) -> pathlib.Path:
    """Create a package-shaped tree holding one credentials file."""
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "env.ps1").write_text(env_body, encoding="utf-8")
    return tmp_path


GOOD_ENV = "\n".join(
    [
        "# hpc-wake credentials. UNTRACKED.",
        "$env:TASKBOARD_MCP_API_KEY = 'key-value'",
        "",
        "$env:CORVIS_TENANT_ID = 'tenant-value'",
        "$env:HPC_WAKE_TASK_ID = 'task-value'",
    ]
)
