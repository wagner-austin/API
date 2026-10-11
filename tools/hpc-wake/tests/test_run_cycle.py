"""The scheduled entry point, exercised against a temporary package tree.

Real parsing, real log rotation, real file writes; the one fake is the
process runner rebound through ``scripts._test_hooks``, which is how the
suite covers the entry point without posting to the board. The tree, the
runner and the clock are :mod:`tests.run_cycle_tree`'s, shared with the
health record's suite, ``test_run_cycle_health.py``.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, ErrorCode
from scripts import _test_hooks, run_cycle

from tests.run_cycle_tree import GOOD_ENV, Completed, RecordingRunner, staged_root


class TestTheCredentialsFile:
    def test_a_line_the_shared_parser_refuses_stops_the_cycle_before_any_publisher(
        self, tmp_path: pathlib.Path
    ) -> None:
        # The grammar is platform_core.env_assignments' and tested there;
        # what this package owns is that main reads runs/env.ps1 through it
        # and that a refusal lands before a publisher runs or a log is cut.
        root = staged_root(tmp_path, GOOD_ENV + '\n$env:DOUBLE = "quoted"\n')
        with pytest.raises(AppError) as caught:
            run_cycle.main(["--package-root", str(root)])
        assert caught.value.code is ErrorCode.CONFIG_ERROR
        assert caught.value.message.startswith(f"line 6 of {root / 'runs' / 'env.ps1'} ")
        assert not (root / "runs" / "cycle.log").exists()


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
        self, tmp_path: pathlib.Path, runner: RecordingRunner
    ) -> None:
        root = staged_root(tmp_path, GOOD_ENV)

        code = run_cycle.main(["--package-root", str(root)])

        assert code == 0
        content = (root / "runs" / "cycle.log").read_text(encoding="utf-8")
        header, *sections = content.splitlines()
        assert header.startswith("== 20") and header.endswith("Z")
        # Each row's marker, then its stdout, then its stderr, in table order.
        assert [tuple(sections[i : i + 3]) for i in range(0, len(sections), 3)] == [
            ("-- hpc-wake", "hpc-out", "hpc-err"),
            ("-- ci-wake", "ci-out", "ci-err"),
            ("-- lock-wake", "lock-out", "lock-err"),
            ("-- fleet-health-wake", "health-out", "health-err"),
            ("-- hub-tunnel-wake", "tunnel-out", "tunnel-err"),
            ("-- lock-wake-diphtheria", "diphtheria-out", "diphtheria-err"),
            ("-- lock-wake-diphtheria-actions", "actions-out", "actions-err"),
            ("-- rcs-pair-watch", "rcs-out", "rcs-err"),
            ("-- login-probe", "login-out", "login-err"),
            ("-- search-probe", "search-out", "search-err"),
        ]

    def test_hands_each_publisher_its_command_its_cwd_and_a_merged_env(
        self, tmp_path: pathlib.Path, runner: RecordingRunner
    ) -> None:
        root = staged_root(tmp_path, GOOD_ENV)

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
            "--check-journal",
            "C:\\Users\\Test\\PROJECTS\\MCPs\\.check-events.jsonl",
            "--action-journal",
            "C:\\Users\\Test\\PROJECTS\\MCPs\\.action-events.jsonl",
        ]
        assert lock_cwd == (root / "..\\lock-wake").resolve()
        assert lock_env == env
        health_args, health_cwd, health_env = runner.calls[3]
        assert list(health_args) == [
            "poetry",
            "run",
            "fleet-health-wake",
            "--journal",
            "C:\\Users\\Test\\PROJECTS\\MCPs\\fleet-mcp\\state\\health-events.jsonl",
        ]
        assert health_cwd == (root / "..\\fleet-health-wake").resolve()
        assert health_env == env
        tunnel_args, tunnel_cwd, tunnel_env = runner.calls[4]
        assert list(tunnel_args) == [
            "poetry",
            "run",
            "fleet-health-wake",
            "--journal",
            "C:\\Users\\Test\\PROJECTS\\MCPs\\logs\\hub-tunnel\\health-events.jsonl",
        ]
        assert tunnel_cwd == (root / "..\\fleet-health-wake").resolve()
        assert tunnel_env == env
        remote_args, remote_cwd, remote_env = runner.calls[5]
        assert list(remote_args) == [
            "poetry",
            "run",
            "lock-wake",
            "--remote-journal",
            "diphtheria:/home/corvis/PROJECTS/MCPs/.fleet-events.jsonl",
            "--cursor-dir",
            "C:\\Users\\Test\\PROJECTS\\MCPs",
        ]
        assert remote_cwd == (root / "..\\lock-wake").resolve()
        assert remote_env == env
        actions_args, actions_cwd, actions_env = runner.calls[6]
        assert list(actions_args) == [
            "poetry",
            "run",
            "lock-wake",
            "--remote-action-journal",
            "diphtheria:/home/corvis/PROJECTS/MCPs/.action-events.jsonl",
            "--cursor-dir",
            "C:\\Users\\Test\\PROJECTS\\MCPs",
        ]
        assert actions_cwd == (root / "..\\lock-wake").resolve()
        assert actions_env == env
        rcs_args, rcs_cwd, rcs_env = runner.calls[7]
        assert list(rcs_args) == [
            "C:\\Users\\Test\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
            "-B",
            "packages\\maketools\\scripts\\run.py",
            "rcs-pair-watch",
            "--workspace",
            "C:\\Users\\Test\\PROJECTS\\MCPs",
        ]
        # Resolved from the package root as the entry point resolves every
        # row's cwd: on Windows the hub checkout's absolute path replaces the
        # root, and on a Linux runner the same text is a relative name under
        # it, so an expectation resolved from the bare text alone failed
        # there (API CI run 38014391403 at c9f4e74a4, 2026-10-10 01:45Z).
        assert rcs_cwd == (root / "C:\\Users\\Test\\PROJECTS\\MCPs").resolve()
        assert rcs_env == env
        login_args, login_cwd, login_env = runner.calls[8]
        assert list(login_args) == [
            "C:\\Users\\Test\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
            "-B",
            "packages\\maketools\\scripts\\run.py",
            "login-probe",
            "--workspace",
            "C:\\Users\\Test\\PROJECTS\\MCPs",
        ]
        assert login_cwd == (root / "C:\\Users\\Test\\PROJECTS\\MCPs").resolve()
        assert login_env == env
        search_args, search_cwd, search_env = runner.calls[9]
        assert list(search_args) == [
            "C:\\Users\\Test\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
            "-B",
            "packages\\maketools\\scripts\\run.py",
            "search-probe",
            "--workspace",
            "C:\\Users\\Test\\PROJECTS\\MCPs",
        ]
        assert search_cwd == (root / "C:\\Users\\Test\\PROJECTS\\MCPs").resolve()
        assert search_env == env
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
        self, tmp_path: pathlib.Path, runner: RecordingRunner
    ) -> None:
        runner.results[0] = Completed("", "cluster unreachable\n", 3)
        root = staged_root(tmp_path, GOOD_ENV)

        assert run_cycle.main(["--package-root", str(root)]) == 3
        assert len(runner.calls) == len(run_cycle.PUBLISHERS) == 10

    def test_a_failing_second_publisher_reddens_the_tick(
        self, tmp_path: pathlib.Path, runner: RecordingRunner
    ) -> None:
        runner.results[1] = Completed("", "actions api refused\n", 7)
        root = staged_root(tmp_path, GOOD_ENV)

        assert run_cycle.main(["--package-root", str(root)]) == 7

    def test_the_first_failure_names_the_tick_when_both_fail(
        self, tmp_path: pathlib.Path, runner: RecordingRunner
    ) -> None:
        runner.results[0] = Completed("", "a\n", 3)
        runner.results[1] = Completed("", "b\n", 7)
        root = staged_root(tmp_path, GOOD_ENV)

        assert run_cycle.main(["--package-root", str(root)]) == 3

    def test_truncates_an_oversized_log_and_keeps_a_small_one(
        self, tmp_path: pathlib.Path, runner: RecordingRunner
    ) -> None:
        runner.results.extend(runner.results)
        root = staged_root(tmp_path, GOOD_ENV)
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
    def test_the_inventory_is_the_ten_rows_in_publication_order(self) -> None:
        """Pinned as data: a publisher added or removed shows up HERE, and
        board task 9406cfd9's rule -- publishers join this table, never
        become sibling scheduled tasks -- has a diff to point at."""
        assert [p["name"] for p in run_cycle.PUBLISHERS] == [
            "hpc-wake",
            "ci-wake",
            "lock-wake",
            "fleet-health-wake",
            "hub-tunnel-wake",
            "lock-wake-diphtheria",
            "lock-wake-diphtheria-actions",
            "rcs-pair-watch",
            "login-probe",
            "search-probe",
        ]
        # The public search's probe (MCPs board task d84d37e3) is last, so a
        # slow site delays no other row's post.
        assert run_cycle.PUBLISHERS[-1]["cwd"] == "C:\\Users\\Test\\PROJECTS\\MCPs"
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
            "--check-journal",
            "C:\\Users\\Test\\PROJECTS\\MCPs\\.check-events.jsonl",
            "--action-journal",
            "C:\\Users\\Test\\PROJECTS\\MCPs\\.action-events.jsonl",
        )
        assert run_cycle.PUBLISHERS[3]["cwd"] == "..\\fleet-health-wake"
        assert run_cycle.PUBLISHERS[3]["args"][2:] == (
            "fleet-health-wake",
            "--journal",
            "C:\\Users\\Test\\PROJECTS\\MCPs\\fleet-mcp\\state\\health-events.jsonl",
        )
        assert run_cycle.PUBLISHERS[4]["cwd"] == "..\\fleet-health-wake"
        assert run_cycle.PUBLISHERS[4]["args"][2:] == (
            "fleet-health-wake",
            "--journal",
            "C:\\Users\\Test\\PROJECTS\\MCPs\\logs\\hub-tunnel\\health-events.jsonl",
        )
        # The same flags tools/lock-wake's test_remote_cycle.py drives
        # against a journal in diphtheria's form (MCPs board task 03590bf9).
        assert run_cycle.PUBLISHERS[5]["cwd"] == "..\\lock-wake"
        assert run_cycle.PUBLISHERS[5]["args"][2:] == (
            "lock-wake",
            "--remote-journal",
            "diphtheria:/home/corvis/PROJECTS/MCPs/.fleet-events.jsonl",
            "--cursor-dir",
            "C:\\Users\\Test\\PROJECTS\\MCPs",
        )
        # The remote form's other flag, which tools/lock-wake's
        # test_remote_cycle.py drives as an action journal (MCPs board task
        # b00ab472), with the cursor in the same clone root.
        assert run_cycle.PUBLISHERS[6]["cwd"] == "..\\lock-wake"
        assert run_cycle.PUBLISHERS[6]["args"][2:] == (
            "lock-wake",
            "--remote-action-journal",
            "diphtheria:/home/corvis/PROJECTS/MCPs/.action-events.jsonl",
            "--cursor-dir",
            "C:\\Users\\Test\\PROJECTS\\MCPs",
        )
        # MCPs' maketools command (board task 21366226), run in the hub
        # checkout like MCPs' own scheduled maketools tasks.
        assert run_cycle.PUBLISHERS[7]["cwd"] == "C:\\Users\\Test\\PROJECTS\\MCPs"
        assert run_cycle.PUBLISHERS[7]["args"][1:] == (
            "-B",
            "packages\\maketools\\scripts\\run.py",
            "rcs-pair-watch",
            "--workspace",
            "C:\\Users\\Test\\PROJECTS\\MCPs",
        )


class TestMainBlock:
    def test_running_as_a_module_actually_runs_a_cycle(
        self, tmp_path: pathlib.Path, runner: RecordingRunner
    ) -> None:
        """The half that silently goes missing without an ``if __name__``
        block — and the scheduler invokes exactly this form. Hermetic
        against a temporary tree: the root arrives through argv, so this
        probe never reads the operating machine's untracked ``runs/`` state
        — the dependence that shipped this job red in CI while green
        locally (five runs, five failures, board 2026-09-09 20:39Z)."""
        import runpy
        import sys

        root = staged_root(tmp_path, GOOD_ENV)
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
        assert _test_hooks.run_process is _test_hooks._default_run_process
