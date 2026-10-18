"""The pump's recorder row: what each publisher ran, handed to fleet (MCPs board task 9f1367d5).

The row runs ``fleet.cli.publisher_ticks`` in the fleet package with one
``<publisher>=<exit code>=<directory>`` token per publisher; that module's
own suite (``tools/fleet/tests/test_cli_publisher_ticks.py``) reads real
checkouts and sends the queue call. What this package owns is that the row
runs after every publisher, carries each one's exit and resolved directory,
and is recorded like any row.
"""

from __future__ import annotations

import pathlib

from scripts import run_cycle

from tests.run_cycle_tree import (
    GOOD_ENV,
    PYTHON,
    WORKSPACE,
    Completed,
    FrozenClock,
    RecordingRunner,
    cycle_argv,
    publisher,
    staged_root,
)


class TestTheRecorderRow:
    def test_it_is_pinned_as_data(self) -> None:
        assert run_cycle.RECORDER == {
            "name": "publisher-ticks",
            "args": ("poetry", "run", "--", "python", "-m", "fleet.cli.publisher_ticks"),
            "cwd": "../fleet",
        }

    def test_it_runs_last_with_each_publisher_s_exit_and_directory(
        self, tmp_path: pathlib.Path, runner: RecordingRunner
    ) -> None:
        runner.results[1] = Completed("", "actions api refused\n", 7)
        root = staged_root(tmp_path, GOOD_ENV)

        run_cycle.main(cycle_argv(root))

        args, cwd, env = runner.calls[-1]
        assert cwd == (root / "../fleet").resolve()
        assert list(args) == [
            *run_cycle.RECORDER["args"],
            *(
                f"{row['name']}={7 if row['name'] == 'ci-wake' else 0}="
                f"{(root / row['cwd']).resolve()}"
                for row in run_cycle.publishers(WORKSPACE, PYTHON)
            ),
        ]
        # The pump's credentials reach it as they reach every publisher.
        assert env == runner.calls[0][2]

    def test_a_record_that_failed_is_its_own_red_row(
        self, tmp_path: pathlib.Path, runner: RecordingRunner, clock: FrozenClock
    ) -> None:
        runner.results[-1] = Completed("", "QUEUE_UNANSWERED\n", 1)
        root = staged_root(tmp_path, GOOD_ENV)

        assert run_cycle.main(cycle_argv(root)) == 1

        failing = publisher(root, "publisher-ticks")
        assert (failing["exit_code"], failing["consecutive_failures"]) == (1, 1)
        assert publisher(root, "hpc-wake")["last_ok"] == "2026-09-21T20:43:15Z"
        log = (root / "runs" / "cycle.log").read_text(encoding="utf-8")
        assert log.endswith("-- publisher-ticks\nQUEUE_UNANSWERED\n")
