"""The scheduled entry point, exercised against a temporary package tree.

Real parsing, real log rotation, real file writes; the one fake is the
process runner rebound through ``scripts._test_hooks``, which is how the
suite covers the entry point without posting to the board. The tree and the
runner are :mod:`tests.run_cycle_tree`'s, shared with the recorder row's
suite, ``test_run_cycle_recorder.py``.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, ErrorCode
from scripts import _test_hooks, run_cycle

from tests.run_cycle_tree import (
    GOOD_ENV,
    PYTHON,
    WORKSPACE,
    Completed,
    RecordingRunner,
    cycle_argv,
    staged_root,
)

#: The rows of one tick: every publisher, then the recorder.
ROWS = 9


def maketools(command: str) -> list[str]:
    """One of MCPs' maketools commands as the pump runs it in the hub checkout.

    Args:
        command: The maketools command.

    Returns:
        The argv.
    """
    return [PYTHON, "-B", run_cycle.MAKETOOLS_RUN, command, "--workspace", str(WORKSPACE)]


class TestTheCredentialsFile:
    def test_a_line_the_shared_parser_refuses_stops_the_cycle_before_any_publisher(
        self, tmp_path: pathlib.Path
    ) -> None:
        # The grammar is platform_core.env_assignments' and tested there;
        # what this package owns is that main reads runs/env.ps1 through it
        # and that a refusal lands before a publisher runs or a log is cut.
        root = staged_root(tmp_path, GOOD_ENV + '\n$env:DOUBLE = "quoted"\n')
        with pytest.raises(AppError) as caught:
            run_cycle.main(cycle_argv(root))
        assert caught.value.code is ErrorCode.CONFIG_ERROR
        assert caught.value.message.startswith(f"line 6 of {root / 'runs' / 'env.ps1'} ")
        assert not (root / "runs" / "cycle.log").exists()


class TestArgumentsFrom:
    def test_parses_the_three_flags_in_any_order(self, tmp_path: pathlib.Path) -> None:
        expected: run_cycle.PumpArguments = {
            "package_root": tmp_path,
            "workspace": WORKSPACE,
            "python": PYTHON,
        }
        assert run_cycle.arguments_from(cycle_argv(tmp_path)) == expected
        reordered = ["--python", PYTHON, "--workspace", str(WORKSPACE)]
        assert run_cycle.arguments_from([*reordered, "--package-root", str(tmp_path)]) == expected

    @pytest.mark.parametrize(
        "tokens",
        [
            # A defaulted root would reach for the file's own tree — the
            # untestable dependence the flag exists to remove.
            [],
            # The root alone, the one flag austinpc's row passed: the
            # checkout and the interpreter would be one machine's literals.
            ["--package-root", "/r"],
            ["--package-root", "/r", "--workspace", "/w", "--root", "/x"],
            ["--package-root", "/r", "--package-root", "/r", "--python", "p"],
            ["--package-root", "/r", "--workspace", "/w", "--python"],
        ],
    )
    def test_refuses_anything_but_each_flag_once_with_its_value(self, tokens: list[str]) -> None:
        with pytest.raises(ValueError, match="usage:"):
            run_cycle.arguments_from(tokens)


class TestMain:
    def test_runs_every_publisher_and_appends_marked_sections_in_order(
        self, tmp_path: pathlib.Path, runner: RecordingRunner
    ) -> None:
        root = staged_root(tmp_path, GOOD_ENV)

        code = run_cycle.main(cycle_argv(root))

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
            ("-- lock-wake-ziggurat", "ziggurat-out", "ziggurat-err"),
            ("-- lock-wake-ziggurat-actions", "actions-out", "actions-err"),
            ("-- login-probe", "login-out", "login-err"),
            ("-- search-probe", "search-out", "search-err"),
            ("-- publisher-ticks", "ticks-out", "ticks-err"),
        ]

    def test_hands_each_publisher_its_command_its_cwd_and_a_merged_env(
        self, tmp_path: pathlib.Path, runner: RecordingRunner
    ) -> None:
        root = staged_root(tmp_path, GOOD_ENV)

        run_cycle.main(cycle_argv(root))

        commands = [(list(args), cwd) for args, cwd, _ in runner.calls]
        assert commands[:-1] == [
            (["poetry", "run", "hpc-wake", "--config", "../hpc3/runs/hpc3.json"], root.resolve()),
            (
                ["poetry", "run", "ci-wake", "--enrolment", "runs/pushes.jsonl"],
                (root / "../ci-wake").resolve(),
            ),
            (
                [
                    "poetry",
                    "run",
                    "lock-wake",
                    "--journal",
                    f"{WORKSPACE}/.fleet-events.jsonl",
                    "--check-journal",
                    f"{WORKSPACE}/.check-events.jsonl",
                    "--action-journal",
                    f"{WORKSPACE}/.action-events.jsonl",
                ],
                (root / "../lock-wake").resolve(),
            ),
            (
                [
                    "poetry",
                    "run",
                    "fleet-health-wake",
                    "--journal",
                    f"{WORKSPACE}/fleet-mcp/state/health-events.jsonl",
                ],
                (root / "../fleet-health-wake").resolve(),
            ),
            # The same flags tools/lock-wake's test_remote_cycle.py drives
            # against a journal in a Linux host's form (MCPs board task
            # 03590bf9), here the session host's.
            (
                [
                    "poetry",
                    "run",
                    "lock-wake",
                    "--remote-journal",
                    "ziggurat:/home/corvis/PROJECTS/MCPs/.fleet-events.jsonl",
                    "--cursor-dir",
                    str(WORKSPACE),
                ],
                (root / "../lock-wake").resolve(),
            ),
            (
                [
                    "poetry",
                    "run",
                    "lock-wake",
                    "--remote-action-journal",
                    "ziggurat:/home/corvis/PROJECTS/MCPs/.action-events.jsonl",
                    "--cursor-dir",
                    str(WORKSPACE),
                ],
                (root / "../lock-wake").resolve(),
            ),
            # MCPs' maketools commands, run in the hub checkout under the
            # interpreter MCPs' own maketools rows use. The cwd is resolved
            # from the package root as the entry point resolves every row's.
            (maketools("login-probe"), (root / WORKSPACE).resolve()),
            (maketools("search-probe"), (root / WORKSPACE).resolve()),
        ]
        env = runner.calls[0][2]
        assert all(call_env == env for _, _, call_env in runner.calls)
        assert env["TASKBOARD_MCP_API_KEY"] == "key-value"
        assert env["HPC_WAKE_TASK_ID"] == "task-value"
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

        assert run_cycle.main(cycle_argv(root)) == 3
        assert len(runner.calls) == ROWS

    def test_a_failing_second_publisher_reddens_the_tick(
        self, tmp_path: pathlib.Path, runner: RecordingRunner
    ) -> None:
        runner.results[1] = Completed("", "actions api refused\n", 7)
        root = staged_root(tmp_path, GOOD_ENV)

        assert run_cycle.main(cycle_argv(root)) == 7

    def test_the_first_failure_names_the_tick_when_both_fail(
        self, tmp_path: pathlib.Path, runner: RecordingRunner
    ) -> None:
        runner.results[0] = Completed("", "a\n", 3)
        runner.results[1] = Completed("", "b\n", 7)
        root = staged_root(tmp_path, GOOD_ENV)

        assert run_cycle.main(cycle_argv(root)) == 3

    def test_truncates_an_oversized_log_and_keeps_a_small_one(
        self, tmp_path: pathlib.Path, runner: RecordingRunner
    ) -> None:
        runner.results.extend(runner.results)
        root = staged_root(tmp_path, GOOD_ENV)
        log = root / "runs" / "cycle.log"
        log.write_text("old\n" * 300_000, encoding="utf-8")
        assert log.stat().st_size > 1_000_000

        run_cycle.main(cycle_argv(root))
        rotated = log.read_text(encoding="utf-8")
        assert "old" not in rotated and rotated.startswith("== ")

        run_cycle.main(cycle_argv(root))
        kept = log.read_text(encoding="utf-8")
        assert kept.startswith(rotated)
        assert kept.count("== 20") == 2


class TestPublishers:
    def test_the_inventory_is_the_eight_rows_in_publication_order(self) -> None:
        """Pinned as data: a publisher added or removed shows up HERE, and
        board task 9406cfd9's rule -- publishers join this table, never
        become sibling scheduled tasks -- has a diff to point at."""
        rows = run_cycle.publishers(WORKSPACE, PYTHON)
        assert [p["name"] for p in rows] == [
            "hpc-wake",
            "ci-wake",
            "lock-wake",
            "fleet-health-wake",
            "lock-wake-ziggurat",
            "lock-wake-ziggurat-actions",
            "login-probe",
            "search-probe",
        ]
        # The public search's probe (MCPs board task d84d37e3) is last, so a
        # slow site delays no other row's post.
        assert rows[-1]["args"][3] == "search-probe"

    def test_no_row_names_a_windows_path(self) -> None:
        """MCPs board task c39ee056: the table spelled austinpc's paths, and on
        diphtheria its first tick looked for ``..\\ci-wake`` as one name. The
        workspace is one name here, which both platforms render alike, so a
        backslash could come only from the table's own text."""
        rows = (*run_cycle.publishers(pathlib.Path("MCPs"), "python3"), run_cycle.RECORDER)
        words = [word for row in rows for word in (*row["args"], row["cwd"])]
        assert [word for word in words if "\\" in word or "C:" in word] == []


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
        sys.argv = ["run-cycle", *cycle_argv(root)]
        try:
            with pytest.raises(SystemExit) as caught:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv[:] = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module
        assert caught.value.code == 0
        assert runner.calls[0][1] == root.resolve()
        assert len(runner.calls) == ROWS
        assert (root / "runs" / "cycle.log").read_text(encoding="utf-8").startswith("== ")


class TestHookDefault:
    def test_the_production_runner_is_the_real_one(self) -> None:
        assert _test_hooks.run_process is _test_hooks._default_run_process
