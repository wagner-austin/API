"""The scheduled entry point's test world: a staged package tree and a
recording process runner.

Shared by ``test_run_cycle.py`` and ``test_run_cycle_recorder.py``; the
``runner`` fixture is registered in ``conftest.py`` from the generator here.
The one fake is the process runner rebound through ``scripts._test_hooks``,
which is how the suites cover the entry point without posting to the board.
"""

from __future__ import annotations

import pathlib
from collections.abc import Generator, Mapping, Sequence

from scripts import _test_hooks, run_cycle


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

    Scripted with one result per publisher and one for the recorder row
    after them, distinguishable by content so ordering assertions mean
    something.
    """
    fake = RecordingRunner(
        [
            Completed("hpc-out\n", "hpc-err\n", 0),
            Completed("ci-out\n", "ci-err\n", 0),
            Completed("lock-out\n", "lock-err\n", 0),
            Completed("health-out\n", "health-err\n", 0),
            Completed("ziggurat-out\n", "ziggurat-err\n", 0),
            Completed("actions-out\n", "actions-err\n", 0),
            Completed("login-out\n", "login-err\n", 0),
            Completed("search-out\n", "search-err\n", 0),
            Completed("ticks-out\n", "ticks-err\n", 0),
        ]
    )
    _test_hooks.run_process = fake
    yield fake
    _test_hooks.run_process = _test_hooks._default_run_process


def staged_root(tmp_path: pathlib.Path, env_body: str) -> pathlib.Path:
    """Create a package-shaped tree holding one credentials file."""
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "env.ps1").write_text(env_body, encoding="utf-8")
    return tmp_path


#: The MCPs checkout and the interpreter the hub runner's row names, as
#: diphtheria fills them.
WORKSPACE = pathlib.Path("/home/corvis/PROJECTS/MCPs")
PYTHON = "/usr/bin/python3"


def cycle_argv(root: pathlib.Path) -> list[str]:
    """The arguments the hub runner's ``hpc-wake`` row passes for this tree.

    Args:
        root: The staged package root.

    Returns:
        The three flags with their values.
    """
    return ["--package-root", str(root), "--workspace", str(WORKSPACE), "--python", PYTHON]


GOOD_ENV = "\n".join(
    [
        "# hpc-wake credentials. UNTRACKED.",
        "$env:TASKBOARD_MCP_API_KEY = 'key-value'",
        "",
        "$env:CORVIS_TENANT_ID = 'tenant-value'",
        "$env:HPC_WAKE_TASK_ID = 'task-value'",
    ]
)
