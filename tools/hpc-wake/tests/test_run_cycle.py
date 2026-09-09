"""The scheduled entry point, exercised against a temporary package tree.

Real parsing, real log rotation, real file writes; the one fake is the
process runner rebound through ``scripts._test_hooks``, which is how the
suite covers the entry point without posting to the board.
"""

from __future__ import annotations

import pathlib
import subprocess
from collections.abc import Generator, Mapping, Sequence

import pytest
from scripts import _test_hooks, run_cycle


class _Completed:
    """The three fields the entry point reads, as plain data."""

    def __init__(self, stdout: str, stderr: str, returncode: int) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class _RecordingRunner:
    """A process runner that records its call and returns scripted output."""

    def __init__(self, result: _Completed) -> None:
        self.result = result
        self.calls: list[tuple[Sequence[str], pathlib.Path, dict[str, str]]] = []

    def __call__(
        self,
        args: Sequence[str],
        *,
        cwd: pathlib.Path,
        env: Mapping[str, str],
        capture_output: bool,
        text: bool,
    ) -> _Completed:
        assert capture_output is True
        assert text is True
        self.calls.append((args, cwd, dict(env)))
        return self.result


@pytest.fixture(name="runner")
def _runner() -> Generator[_RecordingRunner, None, None]:
    """Install a recording runner, and put the real one back afterwards."""
    fake = _RecordingRunner(_Completed("out-line\n", "err-line\n", 0))
    _test_hooks.run_process = fake
    yield fake
    _test_hooks.run_process = subprocess.run


def _staged_root(tmp_path: pathlib.Path, env_body: str) -> pathlib.Path:
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


class TestLoadEnvAssignments:
    def test_parses_assignments_and_skips_comments_and_blanks(self, tmp_path: pathlib.Path) -> None:
        root = _staged_root(tmp_path, GOOD_ENV)
        assignments = run_cycle.load_env_assignments(root / "runs" / "env.ps1")
        assert assignments == {
            "TASKBOARD_MCP_API_KEY": "key-value",
            "CORVIS_TENANT_ID": "tenant-value",
            "HPC_WAKE_TASK_ID": "task-value",
        }

    def test_swallows_a_byte_order_mark(self, tmp_path: pathlib.Path) -> None:
        # PowerShell writes UTF-8 files with a BOM; a parser that read it
        # literally would refuse the first assignment as unparseable.
        path = tmp_path / "env.ps1"
        path.write_bytes(b"\xef\xbb\xbf$env:ONLY = 'value'\n")
        assert run_cycle.load_env_assignments(path) == {"ONLY": "value"}

    def test_refuses_a_line_it_cannot_parse(self, tmp_path: pathlib.Path) -> None:
        # A skipped credential surfaces later as an unauthenticated cycle;
        # the refusal is the whole point of the strict parse.
        path = tmp_path / "env.ps1"
        path.write_text('$env:DOUBLE = "quoted"\n', encoding="utf-8")
        with pytest.raises(ValueError, match="unparseable line"):
            run_cycle.load_env_assignments(path)


class TestMain:
    def test_runs_the_cycle_and_appends_header_stdout_then_stderr(
        self, tmp_path: pathlib.Path, runner: _RecordingRunner
    ) -> None:
        root = _staged_root(tmp_path, GOOD_ENV)

        code = run_cycle.main(root)

        assert code == 0
        content = (root / "runs" / "cycle.log").read_text(encoding="utf-8")
        header, out, err = content.splitlines()
        assert header.startswith("== 20") and header.endswith("Z")
        assert out == "out-line"
        assert err == "err-line"

    def test_hands_the_runner_the_command_the_tree_and_a_merged_env(
        self, tmp_path: pathlib.Path, runner: _RecordingRunner
    ) -> None:
        root = _staged_root(tmp_path, GOOD_ENV)

        run_cycle.main(root)

        args, cwd, env = runner.calls[0]
        assert list(args) == [
            "poetry",
            "run",
            "hpc-wake",
            "--config",
            "..\\hpc3\\runs\\hpc3.json",
        ]
        assert cwd == root
        assert env["TASKBOARD_MCP_API_KEY"] == "key-value"
        assert env["HPC_WAKE_TASK_ID"] == "task-value"
        # The parent environment rides along — a scheduled task still needs
        # PATH and friends to find poetry at all. pytest's own marker is the
        # witness: the framework sets PYTEST_CURRENT_TEST in THIS process's
        # environment, it is not in the credentials file, and it can only
        # reach the recorded env by inheritance.
        assert env["PYTEST_CURRENT_TEST"].endswith("(call)")
        assert len(env) > 3

    def test_propagates_the_cycles_own_failure(
        self, tmp_path: pathlib.Path, runner: _RecordingRunner
    ) -> None:
        runner.result = _Completed("", "cluster unreachable\n", 3)
        root = _staged_root(tmp_path, GOOD_ENV)

        assert run_cycle.main(root) == 3

    def test_truncates_an_oversized_log_and_keeps_a_small_one(
        self, tmp_path: pathlib.Path, runner: _RecordingRunner
    ) -> None:
        root = _staged_root(tmp_path, GOOD_ENV)
        log = root / "runs" / "cycle.log"
        log.write_text("old\n" * 300_000, encoding="utf-8")
        assert log.stat().st_size > 1_000_000

        run_cycle.main(root)
        rotated = log.read_text(encoding="utf-8")
        assert "old" not in rotated and rotated.startswith("== ")

        run_cycle.main(root)
        kept = log.read_text(encoding="utf-8")
        assert kept.startswith(rotated)
        assert kept.count("== 20") == 2


class _RaisingRunner:
    """A runner that refuses, recording the call — the ``__main__`` probe.

    Raising is what keeps the probe side-effect-free: ``main`` appends to
    the cycle log only AFTER the process call, so a runner that raises
    proves the entry block invoked ``main`` against the real package tree
    without the test writing a fake entry into the operational log.
    """

    def __init__(self) -> None:
        self.calls: list[pathlib.Path] = []

    def __call__(
        self,
        args: Sequence[str],
        *,
        cwd: pathlib.Path,
        env: Mapping[str, str],
        capture_output: bool,
        text: bool,
    ) -> _Completed:
        self.calls.append(cwd)
        raise RuntimeError("main-block probe")


class TestMainBlock:
    def test_running_as_a_module_reaches_the_real_package_tree(self) -> None:
        """The half that silently goes missing without an ``if __name__``
        block — and the scheduler invokes exactly this form."""
        import runpy
        import sys

        probe = _RaisingRunner()
        _test_hooks.run_process = probe
        module_name = "scripts.run_cycle"
        saved_module = sys.modules.pop(module_name, None)
        try:
            with pytest.raises(RuntimeError, match="main-block probe"):
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            _test_hooks.run_process = subprocess.run
            if saved_module is not None:
                sys.modules[module_name] = saved_module
        assert probe.calls == [pathlib.Path(__file__).resolve().parent.parent]


class TestHookDefault:
    def test_the_production_runner_is_the_real_one(self) -> None:
        expected: _test_hooks.RunProcess = subprocess.run
        assert _test_hooks.run_process is expected
