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
    """A process runner that records calls and returns scripted results in order."""

    def __init__(self, results: Sequence[_Completed]) -> None:
        self.results = list(results)
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
        return self.results[len(self.calls) - 1]


@pytest.fixture(name="runner")
def _runner() -> Generator[_RecordingRunner, None, None]:
    """Install a recording runner, and put the real one back afterwards.

    Scripted with one result per publisher, distinguishable by content so
    ordering assertions mean something.
    """
    fake = _RecordingRunner(
        [
            _Completed("hpc-out\n", "hpc-err\n", 0),
            _Completed("ci-out\n", "ci-err\n", 0),
            _Completed("lock-out\n", "lock-err\n", 0),
        ]
    )
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


class TestPackageRootFrom:
    def test_parses_the_one_flag(self, tmp_path: pathlib.Path) -> None:
        assert run_cycle.package_root_from(["--package-root", str(tmp_path)]) == tmp_path

    def test_refuses_no_arguments(self) -> None:
        # A defaulted root would reach for the file's own tree — the
        # untestable dependence the flag exists to remove.
        with pytest.raises(ValueError, match="usage:"):
            run_cycle.package_root_from([])

    def test_refuses_an_unknown_flag(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="usage:"):
            run_cycle.package_root_from(["--root", str(tmp_path)])


class TestMain:
    def test_runs_every_publisher_and_appends_marked_sections_in_order(
        self, tmp_path: pathlib.Path, runner: _RecordingRunner
    ) -> None:
        root = _staged_root(tmp_path, GOOD_ENV)

        code = run_cycle.main(["--package-root", str(root)])

        assert code == 0
        content = (root / "runs" / "cycle.log").read_text(encoding="utf-8")
        header, mark1, out1, err1, mark2, out2, err2, mark3, out3, err3 = content.splitlines()
        assert header.startswith("== 20") and header.endswith("Z")
        assert mark1 == "-- hpc-wake"
        assert (out1, err1) == ("hpc-out", "hpc-err")
        assert mark2 == "-- ci-wake"
        assert (out2, err2) == ("ci-out", "ci-err")
        assert mark3 == "-- lock-wake"
        assert (out3, err3) == ("lock-out", "lock-err")

    def test_hands_each_publisher_its_command_its_cwd_and_a_merged_env(
        self, tmp_path: pathlib.Path, runner: _RecordingRunner
    ) -> None:
        root = _staged_root(tmp_path, GOOD_ENV)

        run_cycle.main(["--package-root", str(root)])

        args, cwd, env = runner.calls[0]
        assert list(args) == [
            "poetry",
            "run",
            "hpc-wake",
            "--config",
            "..\\hpc3\\runs\\hpc3.json",
        ]
        assert cwd == root.resolve()
        ci_args, ci_cwd, ci_env = runner.calls[1]
        assert list(ci_args) == [
            "poetry",
            "run",
            "ci-wake",
            "--enrolment",
            "runs\\pushes.jsonl",
        ]
        assert ci_cwd == (root / "..\\ci-wake").resolve()
        lock_args, lock_cwd, lock_env = runner.calls[2]
        assert list(lock_args) == [
            "poetry",
            "run",
            "lock-wake",
            "--journal",
            "C:\\Users\\Test\\PROJECTS\\MCPs\\.fleet-events.jsonl",
        ]
        assert lock_cwd == (root / "..\\lock-wake").resolve()
        assert lock_env == env
        assert env["TASKBOARD_MCP_API_KEY"] == "key-value"
        assert env["HPC_WAKE_TASK_ID"] == "task-value"
        assert ci_env == env
        # The parent environment rides along — a scheduled task still needs
        # PATH and friends to find poetry at all. pytest's own marker is the
        # witness: the framework sets PYTEST_CURRENT_TEST in THIS process's
        # environment, it is not in the credentials file, and it can only
        # reach the recorded env by inheritance.
        assert env["PYTEST_CURRENT_TEST"].endswith("(call)")
        assert len(env) > 3

    def test_a_failing_first_publisher_does_not_stop_the_second(
        self, tmp_path: pathlib.Path, runner: _RecordingRunner
    ) -> None:
        runner.results[0] = _Completed("", "cluster unreachable\n", 3)
        root = _staged_root(tmp_path, GOOD_ENV)

        assert run_cycle.main(["--package-root", str(root)]) == 3
        assert len(runner.calls) == 3

    def test_a_failing_second_publisher_reddens_the_tick(
        self, tmp_path: pathlib.Path, runner: _RecordingRunner
    ) -> None:
        runner.results[1] = _Completed("", "actions api refused\n", 7)
        root = _staged_root(tmp_path, GOOD_ENV)

        assert run_cycle.main(["--package-root", str(root)]) == 7

    def test_the_first_failure_names_the_tick_when_both_fail(
        self, tmp_path: pathlib.Path, runner: _RecordingRunner
    ) -> None:
        runner.results[0] = _Completed("", "a\n", 3)
        runner.results[1] = _Completed("", "b\n", 7)
        root = _staged_root(tmp_path, GOOD_ENV)

        assert run_cycle.main(["--package-root", str(root)]) == 3

    def test_truncates_an_oversized_log_and_keeps_a_small_one(
        self, tmp_path: pathlib.Path, runner: _RecordingRunner
    ) -> None:
        runner.results.extend(runner.results)
        root = _staged_root(tmp_path, GOOD_ENV)
        log = root / "runs" / "cycle.log"
        log.write_text("old\n" * 300_000, encoding="utf-8")
        assert log.stat().st_size > 1_000_000

        run_cycle.main(["--package-root", str(root)])
        rotated = log.read_text(encoding="utf-8")
        assert "old" not in rotated and rotated.startswith("== ")

        run_cycle.main(["--package-root", str(root)])
        kept = log.read_text(encoding="utf-8")
        assert kept.startswith(rotated)
        assert kept.count("== 20") == 2


class TestPublishers:
    def test_the_inventory_is_the_three_bridges_in_publication_order(self) -> None:
        """Pinned as data: a publisher added or removed shows up HERE, and
        board task 9406cfd9's rule -- publishers join this table, never
        become sibling scheduled tasks -- has a diff to point at."""
        assert [p["name"] for p in run_cycle.PUBLISHERS] == ["hpc-wake", "ci-wake", "lock-wake"]
        assert run_cycle.PUBLISHERS[0]["cwd"] == "."
        assert run_cycle.PUBLISHERS[1]["cwd"] == "..\\ci-wake"
        assert run_cycle.PUBLISHERS[1]["args"][2:] == (
            "ci-wake",
            "--enrolment",
            "runs\\pushes.jsonl",
        )
        assert run_cycle.PUBLISHERS[2]["cwd"] == "..\\lock-wake"
        assert run_cycle.PUBLISHERS[2]["args"][2:] == (
            "lock-wake",
            "--journal",
            "C:\\Users\\Test\\PROJECTS\\MCPs\\.fleet-events.jsonl",
        )


class TestMainBlock:
    def test_running_as_a_module_actually_runs_a_cycle(
        self, tmp_path: pathlib.Path, runner: _RecordingRunner
    ) -> None:
        """The half that silently goes missing without an ``if __name__``
        block — and the scheduler invokes exactly this form. Hermetic
        against a temporary tree: the root arrives through argv, so this
        probe never reads the operating machine's untracked ``runs/`` state
        — the dependence that shipped this job red in CI while green
        locally (five runs, five failures, board 2026-09-09 20:39Z)."""
        import runpy
        import sys

        root = _staged_root(tmp_path, GOOD_ENV)
        module_name = "scripts.run_cycle"
        saved_argv = list(sys.argv)
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = ["run-cycle", "--package-root", str(root)]
        try:
            with pytest.raises(SystemExit) as caught:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv[:] = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module
        assert caught.value.code == 0
        assert runner.calls[0][1] == root.resolve()
        assert len(runner.calls) == len(run_cycle.PUBLISHERS)
        assert (root / "runs" / "cycle.log").read_text(encoding="utf-8").startswith("== ")


class TestHookDefault:
    def test_the_production_runner_is_the_real_one(self) -> None:
        expected: _test_hooks.RunProcess = subprocess.run
        assert _test_hooks.run_process is expected
