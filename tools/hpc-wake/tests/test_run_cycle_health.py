"""The health record every tick of the scheduled entry point writes.

Real file writes against a staged package tree, read back through
``scripts.pump_health``'s own reader; the process runner and the clock are
the fakes from :mod:`tests.run_cycle_tree`.
"""

from __future__ import annotations

import datetime
import pathlib

from scripts import run_cycle
from scripts.pump_health import read_health

from tests.run_cycle_tree import (
    GOOD_ENV,
    PYTHON,
    WORKSPACE,
    Completed,
    FrozenClock,
    RecordingRunner,
    cycle_argv,
    health_path,
    publisher,
    staged_root,
    written_line,
)


class TestHealthRecord:
    """Every tick writes ``runs/pump-health.json``, and that file is the
    only thing outside this machine's scheduler that can tell whether the
    wake system is publishing. Between 2026-09-16 and 2026-09-21 all three
    publishers failed on every tick and the only trace was a traceback in
    cycle.log, which is truncated past 1 MB and had kept 80 minutes of it."""

    def test_a_green_tick_records_every_publisher_as_succeeding(
        self, tmp_path: pathlib.Path, runner: RecordingRunner, clock: FrozenClock
    ) -> None:
        root = staged_root(tmp_path, GOOD_ENV)

        run_cycle.main(cycle_argv(root))

        assert written_line(root) == "2026-09-21T20:43:15Z"
        recorded = read_health(health_path(root))
        assert set(recorded) == {
            "hpc-wake",
            "ci-wake",
            "lock-wake",
            "fleet-health-wake",
            "lock-wake-ziggurat",
            "lock-wake-ziggurat-actions",
            "login-probe",
            "search-probe",
            "publisher-ticks",
        }
        assert all(entry["consecutive_failures"] == 0 for entry in recorded.values())
        assert all(entry["last_ok"] == "2026-09-21T20:43:15Z" for entry in recorded.values())

    def test_the_session_host_row_is_recorded_published(
        self, tmp_path: pathlib.Path, runner: RecordingRunner, clock: FrozenClock
    ) -> None:
        """MCPs board tasks 03590bf9 and c39ee056: the row reading the
        session host's fleet-lock journal is a publisher like the others, so
        its tick's exit 0 lands in the health record as published."""
        root = staged_root(tmp_path, GOOD_ENV)

        run_cycle.main(cycle_argv(root))

        entry = publisher(root, "lock-wake-ziggurat")
        assert entry["exit_code"] == 0
        assert entry["consecutive_failures"] == 0
        assert entry["last_ok"] == "2026-09-21T20:43:15Z"

    def test_a_failing_session_host_actions_row_is_recorded_against_itself_alone(
        self, tmp_path: pathlib.Path, runner: RecordingRunner, clock: FrozenClock
    ) -> None:
        """MCPs board task b00ab472: the row reading the session host's action
        journal has its own health entry, so an unreachable host or a
        refused key reddens it and not the fleet-lock row beside it."""
        runner.results[5] = Completed("", "LOCK_WAKE_REMOTE_JOURNAL_UNREADABLE\n", 1)
        root = staged_root(tmp_path, GOOD_ENV)

        assert run_cycle.main(cycle_argv(root)) == 1

        failing = publisher(root, "lock-wake-ziggurat-actions")
        assert (failing["exit_code"], failing["consecutive_failures"]) == (1, 1)
        assert publisher(root, "lock-wake-ziggurat")["consecutive_failures"] == 0
        assert publisher(root, "login-probe")["last_ok"] == "2026-09-21T20:43:15Z"

    def test_a_publisher_failing_twice_is_recorded_as_a_streak_of_two(
        self, tmp_path: pathlib.Path, runner: RecordingRunner, clock: FrozenClock
    ) -> None:
        """THE OUTAGE'S SHAPE, reproduced: ci-wake refused on consecutive
        ticks while the pump itself went on running. The streak is what
        tells a reader this is an outage rather than one bad minute."""
        green = Completed("", "", 0)
        runner.results = [
            green,
            Completed("", "TASK_SESSION_UNLEDGERED\n", 1),
            # The rest of the publishers, then the recorder row after them.
            *[green] * (len(run_cycle.publishers(WORKSPACE, PYTHON)) - 1),
        ] * 2
        root = staged_root(tmp_path, GOOD_ENV)

        run_cycle.main(cycle_argv(root))
        clock.instant = datetime.datetime(2026, 9, 21, 20, 46, 15, tzinfo=datetime.UTC)
        run_cycle.main(cycle_argv(root))

        failing = publisher(root, "ci-wake")
        assert failing["consecutive_failures"] == 2
        assert failing["exit_code"] == 1
        assert failing["last_ok"] == ""
        assert publisher(root, "lock-wake")["last_ok"] == "2026-09-21T20:46:15Z"
