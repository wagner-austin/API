"""Bringing a result back, which nothing used to do.

THE FIRST TEST HERE IS THE ONE THAT WOULD HAVE CAUGHT THE HOLE.
:meth:`TestTheWiring.test_a_dispatched_run_can_reach_a_terminal_state` runs a
dispatch and then collects it, and asserts the feed ends on ``passed``. Before
``fleet-collect`` existed that was unreachable: ``dispatch.finish`` and
``launch.result_script`` were both fully covered by tests that called them
directly, and NO command called either, so every dispatch stayed ``running``
until its lease lapsed. A hundred per cent of statements and branches, and the
feature did not exist.

The lesson is in ``memory/feedback_registered_is_not_invoked.md`` and this file
is the discipline: assert the PATH from one command's effect to another's, not
that each function works when called by a test.
"""

from __future__ import annotations

import pathlib
import runpy
import sys

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.cli import _config, collect, run, watch
from fleet.contracts.node import NodeConfig
from fleet.core import _test_hooks, leases, records, staging
from fleet.core import collect as core_collect
from tests.conftest import (
    DEMO_NOW,
    DEMO_PROJECT,
    DEMO_RUN_ID,
    FakeClock,
    FakeRun,
    dispatch_replies,
    ok,
    prebuilt_archive,
)

#: When the node says a demo build finished: inside its lease, which the demo
#: project's five declared minutes at :data:`~fleet.core.dispatch.LEASE_SLACK`
#: put 600 seconds after the dispatch.
DEMO_FINISHED = DEMO_NOW + 120


def _node() -> NodeConfig:
    """The one node the demo workspace declares.

    Returns:
        Its declaration.
    """
    return NodeConfig(
        host="lavender",
        stage_root="C:/fleet/stage",
        logical_cores=16,
        ram_gb=32.0,
        gpu=None,
        budget={
            "reserved_cores": 2,
            "reserved_ram_gb": 4.0,
            "worker_ram_gb": 1.1,
            "max_concurrent_runs": 2,
            "max_disk_gb": 20.0,
        },
    )


def _dispatch(config_path: pathlib.Path, repo: pathlib.Path) -> None:
    """Run one dispatch to completion of its launch.

    Args:
        config_path: The workspace document.
        repo: The synthetic monorepo root.
    """
    payload = prebuilt_archive(config_path, repo)
    _test_hooks.run = FakeRun(dispatch_replies(staging.digest(payload)))
    run.main(
        [
            _config.CONFIG_FLAG,
            str(config_path),
            run.PROJECT_FLAG,
            DEMO_PROJECT,
            run.AGENT_FLAG,
            "opus-fleet-0904",
            run.SESSION_FLAG,
            "acc774c0-3bc3-4cce-9dda-c7a12fb99519",
            run.ROOT_FLAG,
            str(repo),
        ]
    )


class TestPollResult:
    def test_an_empty_answer_means_still_running(self) -> None:
        """Absence is the signal. The build writes the file last, so it
        cannot exist while make is still going."""
        _test_hooks.run = FakeRun([ok(""), ok("   \n")])

        assert core_collect.poll_result(_node(), run_id=DEMO_RUN_ID) is None

    def test_a_zero_is_a_result_and_not_an_absence(self) -> None:
        """The distinction the empty-string signal exists to protect."""
        _test_hooks.run = FakeRun([ok(""), ok(f"0 {DEMO_FINISHED}\n")])

        assert core_collect.poll_result(_node(), run_id=DEMO_RUN_ID) == {
            "exit_code": 0,
            "finished_unix": DEMO_FINISHED,
        }

    def test_a_failing_status_is_read(self) -> None:
        _test_hooks.run = FakeRun([ok(""), ok(f"2 {DEMO_FINISHED}\r\n")])

        assert core_collect.poll_result(_node(), run_id=DEMO_RUN_ID) == {
            "exit_code": 2,
            "finished_unix": DEMO_FINISHED,
        }

    def test_a_negative_status_is_read(self) -> None:
        """PowerShell reports an unhandled exception as a large negative."""
        _test_hooks.run = FakeRun([ok(""), ok(f"-1073741819 {DEMO_FINISHED}\n")])

        assert core_collect.poll_result(_node(), run_id=DEMO_RUN_ID) == {
            "exit_code": -1073741819,
            "finished_unix": DEMO_FINISHED,
        }

    def test_an_unreadable_answer_is_refused_rather_than_called_unfinished(self) -> None:
        """Treating it as running would hold the node's budget forever."""
        _test_hooks.run = FakeRun([ok(""), ok("Access is denied.\n")])

        with pytest.raises(AppError) as refusal:
            core_collect.poll_result(_node(), run_id=DEMO_RUN_ID)

        assert refusal.value.code is FleetErrorCode.RUN_RESULT_UNREADABLE
        assert "Access is denied." in refusal.value.message

    def test_a_status_without_a_timestamp_is_refused(self) -> None:
        """The old single-field answer, which a stale node could still give."""
        _test_hooks.run = FakeRun([ok(""), ok("0\n")])

        with pytest.raises(AppError) as refusal:
            core_collect.poll_result(_node(), run_id=DEMO_RUN_ID)

        assert refusal.value.code is FleetErrorCode.RUN_RESULT_UNREADABLE

    def test_a_timestamp_that_is_not_a_number_is_refused(self) -> None:
        _test_hooks.run = FakeRun([ok(""), ok("0 yesterday\n")])

        with pytest.raises(AppError) as refusal:
            core_collect.poll_result(_node(), run_id=DEMO_RUN_ID)

        assert refusal.value.code is FleetErrorCode.RUN_RESULT_UNREADABLE


class TestOutcome:
    def test_zero_passed(self) -> None:
        assert core_collect.outcome_for(0) == "passed"

    def test_anything_else_failed(self) -> None:
        assert core_collect.outcome_for(2) == "failed"

    def test_the_detail_names_the_log_a_reader_would_open(self) -> None:
        detail = core_collect.describe(_node(), run_id=DEMO_RUN_ID, exit_code=2)

        assert "lavender:C:/fleet/stage" in detail
        assert DEMO_RUN_ID in detail
        assert "result.txt.log" in detail


class TestTheWiring:
    def test_a_dispatched_run_can_reach_a_terminal_state(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """THE REGRESSION. Nothing connected a finished suite to the feed.

        Before fleet-collect there was no code path at all from a node's exit
        status back to the ledger, so this assertion could not have been made
        about any dispatch the package performed.
        """
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        _dispatch(config_path, repo)
        _test_hooks.run = FakeRun([ok(""), ok(f"0 {DEMO_FINISHED}\n")])

        assert collect.main([_config.CONFIG_FLAG, str(config_path)]) == 0

        rows = records.read_ledger(loaded.ledger)
        assert [row["outcome"] for row in rows] == ["running", "passed"]
        assert rows[-1]["exit_code"] == 0
        assert [event["kind"] for event in records.read_feed(loaded.feed)] == [
            "leased",
            "staged",
            "started",
            "passed",
        ]

    def test_collection_gives_the_lease_back(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """Until it does, the project's environment stays claimed and the
        next dispatch of it is refused for a run that already finished."""
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        _dispatch(config_path, repo)
        _test_hooks.run = FakeRun([ok(""), ok(f"0 {DEMO_FINISHED}\n")])

        collect.main([_config.CONFIG_FLAG, str(config_path)])

        assert leases.find_by_run(loaded.leases, run_id=DEMO_RUN_ID, now_unix=DEMO_NOW) is None

    def test_a_failing_suite_is_recorded_and_still_exits_zero(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """Collection worked; the work failed. A shell loop must not stop on
        the first red build -- that is the moment somebody wants reporting."""
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        _dispatch(config_path, repo)
        _test_hooks.run = FakeRun([ok(""), ok(f"2 {DEMO_FINISHED}\n")])

        assert collect.main([_config.CONFIG_FLAG, str(config_path)]) == 0

        assert records.read_ledger(loaded.ledger)[-1]["outcome"] == "failed"
        assert records.read_feed(loaded.feed)[-1]["kind"] == "failed"

    def test_an_unfinished_run_is_left_alone(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        _dispatch(config_path, repo)
        _test_hooks.run = FakeRun([ok(""), ok("")])

        assert collect.main([_config.CONFIG_FLAG, str(config_path)]) == 0

        assert [row["outcome"] for row in records.read_ledger(loaded.ledger)] == ["running"]
        held = leases.find_by_run(loaded.leases, run_id=DEMO_RUN_ID, now_unix=DEMO_NOW)
        assert held == {
            "node": "lavender",
            "project": DEMO_PROJECT,
            "run_id": DEMO_RUN_ID,
            "agent": "opus-fleet-0904",
            "session_id": "acc774c0-3bc3-4cce-9dda-c7a12fb99519",
            "acquired_unix": DEMO_NOW,
            "expires_unix": DEMO_NOW + 600,
        }

    def test_collecting_twice_closes_the_run_once(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """What makes the shell loop safe to run every thirty seconds."""
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        _dispatch(config_path, repo)
        _test_hooks.run = FakeRun([ok(""), ok(f"0 {DEMO_FINISHED}\n")])
        collect.main([_config.CONFIG_FLAG, str(config_path)])

        # No replies scripted: a second collection must reach no node at all.
        _test_hooks.run = FakeRun([])

        assert collect.main([_config.CONFIG_FLAG, str(config_path)]) == 0
        assert len(records.read_ledger(loaded.ledger)) == 2

    def test_an_empty_workspace_collects_nothing(self, config_path: pathlib.Path) -> None:
        _test_hooks.run = FakeRun([])

        assert collect.main([_config.CONFIG_FLAG, str(config_path)]) == 0

    def test_one_run_can_be_named(self, config_path: pathlib.Path, repo: pathlib.Path) -> None:
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        _dispatch(config_path, repo)
        _test_hooks.run = FakeRun([])

        assert (
            collect.main(
                [_config.CONFIG_FLAG, str(config_path), collect.RUN_FLAG, "some-other-run"]
            )
            == 0
        )
        assert [row["outcome"] for row in records.read_ledger(loaded.ledger)] == ["running"]

    def test_a_run_that_outlived_its_lease_is_refused_rather_than_recorded(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """The environment was unprotected while the build was still writing,
        so the result may have been produced by two suites interfering.
        Recording it tidily would be the last chance anybody had to notice."""
        _dispatch(config_path, repo)
        # The lease is 5 declared minutes at slack 2, so it lapses at +600.
        _test_hooks.run = FakeRun([ok(""), ok(f"0 {DEMO_NOW + 900}\n")])

        with pytest.raises(AppError) as refusal:
            collect.main([_config.CONFIG_FLAG, str(config_path)])

        assert refusal.value.code is FleetErrorCode.LEASE_NOT_HELD
        assert "300s after its lease lapsed" in refusal.value.message
        assert "fleet-cancel" in refusal.value.message

    def test_collecting_late_is_not_the_same_as_running_unprotected(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """THE REGRESSION, measured 2026-09-04 on the real fleet.

        A run finished three minutes inside its window and was refused twenty
        minutes later, because collection asked "is a lease held now" -- a
        question about how promptly somebody came to look -- instead of "did
        the lease cover the run". Here the clock is a day past the lease and
        the run still closes cleanly, because it FINISHED inside it.
        """
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        _dispatch(config_path, repo)
        _test_hooks.now = FakeClock(DEMO_NOW + 86_400)
        _test_hooks.run = FakeRun([ok(""), ok(f"0 {DEMO_FINISHED}\n")])

        assert collect.main([_config.CONFIG_FLAG, str(config_path)]) == 0

        rows = records.read_ledger(loaded.ledger)
        assert [row["outcome"] for row in rows] == ["running", "passed"]
        assert records.read_feed(loaded.feed)[-1]["kind"] == "passed"

    def test_a_run_finishing_exactly_on_its_deadline_is_still_protected(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """The boundary is inclusive: the lease was live for that second."""
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        _dispatch(config_path, repo)
        _test_hooks.run = FakeRun([ok(""), ok(f"0 {DEMO_NOW + 600}\n")])

        assert collect.main([_config.CONFIG_FLAG, str(config_path)]) == 0

        assert records.read_ledger(loaded.ledger)[-1]["outcome"] == "passed"


class TestEntryPoints:
    def test_the_entrypoint_exits_zero(self, config_path: pathlib.Path, repo: pathlib.Path) -> None:
        _dispatch(config_path, repo)
        _test_hooks.run = FakeRun([ok(""), ok(f"0 {DEMO_FINISHED}\n")])
        saved = sys.argv
        sys.argv = ["fleet-collect", _config.CONFIG_FLAG, str(config_path)]
        try:
            with pytest.raises(SystemExit) as raised:
                collect.entrypoint()
        finally:
            sys.argv = saved

        assert raised.value.code == 0

    def test_running_as_a_module_actually_collects(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """Without the `if __name__` block this imports, runs nothing and
        exits 0 -- which reads as "nothing has finished" and would leave every
        dispatch open while looking like a clean collection."""
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        _dispatch(config_path, repo)
        _test_hooks.run = FakeRun([ok(""), ok(f"0 {DEMO_FINISHED}\n")])
        saved_argv = sys.argv
        saved_module = sys.modules.pop("fleet.cli.collect", None)
        sys.argv = ["x", _config.CONFIG_FLAG, str(config_path)]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module("fleet.cli.collect", run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules["fleet.cli.collect"] = saved_module

        assert raised.value.code == 0
        assert records.read_ledger(loaded.ledger)[-1]["outcome"] == "passed"


class TestLiveRows:
    def test_the_last_row_for_an_id_decides(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """The ledger is append-only, so a closed run still has a running row
        in it. Taking every row whose outcome is `running` would re-collect
        dispatches that ended hours ago."""
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        _dispatch(config_path, repo)
        _test_hooks.run = FakeRun([ok(""), ok(f"0 {DEMO_FINISHED}\n")])
        collect.main([_config.CONFIG_FLAG, str(config_path)])

        assert collect.live_rows(loaded, run_id=None) == ()

    def test_a_finished_run_stops_counting_against_its_node(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """THE SECOND-DISPATCH REGRESSION, measured 2026-09-04.

        `records.live_runs` counted every row whose outcome was `running`,
        superseded ones included. sedona declares max_concurrent_runs: 1, so
        having run and closed exactly one dispatch it refused every future one
        as already full -- a node's live count could only go up.
        """
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        _dispatch(config_path, repo)
        assert records.live_runs(loaded.ledger, node="lavender") == 1
        _test_hooks.run = FakeRun([ok(""), ok(f"0 {DEMO_FINISHED}\n")])

        collect.main([_config.CONFIG_FLAG, str(config_path)])

        assert records.live_runs(loaded.ledger, node="lavender") == 0

    def test_a_finished_run_is_not_reported_lost(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """The same defect seen from the watcher: a closed run holds no
        lease, so reading the raw ledger would call it wedged forever."""
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        _dispatch(config_path, repo)
        _test_hooks.run = FakeRun([ok(""), ok(f"0 {DEMO_FINISHED}\n")])
        collect.main([_config.CONFIG_FLAG, str(config_path)])

        assert watch.lost_runs(loaded, now_unix=DEMO_NOW) == ()
        assert not any(
            "LOST" in line for line in watch.lines_for(loaded, run_id=None, now_unix=DEMO_NOW)
        )
