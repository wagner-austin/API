"""The runs a node runner's tick stops (MCPs board task fd5cabfa).

Two defects, each measured on sedona before this was written. A queue cancel
never reached the node: job 61bd1cbc was cancelled at 2026-09-22 20:22Z and
its run closed on the hub only at 00:49Z, after its processes had been killed
by hand, because the collect pass listed only live jobs and a cancelled one is
not live. And a hung build renewed its claim forever, because a renewal had
no deadline. Each test here drives one whole tick through ``node_agent.main``
against the fake queue and the fake ssh runner, and asserts what reached the
node, the queue and the ledger.

The lease the demo project gets is ``expected_minutes`` 5 at the slack of 2,
so a run launched at ``DEMO_NOW`` is past its lease from ``DEMO_NOW + 601``.
"""

from __future__ import annotations

import pathlib

from platform_core.json_utils import JSONObject, dump_json_str, narrow_json_to_str

from fleet.cli import _config, node_agent, node_collect
from fleet.contracts.ledger import LedgerEntry
from fleet.core import _test_hooks, dialect, leases, names, queue, records
from tests._node_agent_fixtures import (
    PASSING_TAIL,
    PROBED,
    VERDICT_TASK,
    _credentials_in_env,
    _sourced_config,
    held_answer,
    launch,
    node_argv,
)
from tests._queue_fakes import DEFAULT_JOB_ID, FakeQueue, queue_job
from tests.conftest import DEMO_NOW, DEMO_RUN_ID, FakeClock, FakeRun, ok

__all__ = ["_credentials_in_env", "_sourced_config"]

#: The first second the demo run is past its lease.
PAST_THE_LEASE = DEMO_NOW + 601

#: The job a cancel left behind: terminal, still naming this runner and run.
CANCELLED_JOB: JSONObject = queue_job(
    status="cancelled", node="lavender", runId=DEMO_RUN_ID, claimedBy="fleet-node-lavender"
)


def _stop_body(config_path: pathlib.Path) -> str:
    """The stop script lavender is sent for the demo run.

    Args:
        config_path: The workspace document.

    Returns:
        The script's text, as the dialect renders it.
    """
    loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
    node = loaded.workspace["nodes"]["lavender"]
    return dialect.for_platform(node["platform"]).stop_script(
        target=names.dispatch_directory(node["stage_root"], DEMO_RUN_ID), run_id=DEMO_RUN_ID
    )


def _cancelled_page(jobs: list[JSONObject], next_offset: int | None) -> str:
    """One page of the cancelled listing.

    Args:
        jobs: The page's wire rows.
        next_offset: Where the next page begins, or None on the last.

    Returns:
        The rendered ``dispatch_list`` answer.
    """
    return dump_json_str({"jobs": jobs, "pagination": {"nextOffset": next_offset}})


def _ledger(config_path: pathlib.Path) -> tuple[LedgerEntry, ...]:
    """Every ledger row the workspace holds.

    Args:
        config_path: The workspace document.

    Returns:
        The rows, oldest first.
    """
    loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
    return records.read_ledger(loaded.ledger)


def _lease_released(config_path: pathlib.Path, *, now_unix: int) -> bool:
    """Whether the demo run holds no lease at a given moment.

    Args:
        config_path: The workspace document.
        now_unix: The moment asked about. At ``DEMO_NOW`` the lease taken at
            launch is unexpired, so None there means it was released.

    Returns:
        True when no lease names the run.
    """
    loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
    return leases.find_by_run(loaded.leases, run_id=DEMO_RUN_ID, now_unix=now_unix) is None


class TestPastItsLease:
    def test_a_build_still_running_past_its_lease_is_stopped_and_closed_failed(
        self, sourced_config: pathlib.Path
    ) -> None:
        launch(sourced_config)
        _test_hooks.now = FakeClock(PAST_THE_LEASE)
        runner = FakeRun([ok(""), ok(""), ok(""), ok("stopped"), ok(""), ok(PASSING_TAIL), *PROBED])
        _test_hooks.run = runner
        endpoint = FakeQueue(
            [
                held_answer(taskId=VERDICT_TASK),
                "posted",
                dump_json_str({"job": queue_job(status="failed")}),
                dump_json_str({"claimed": None}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert node_agent.main(node_argv(sourced_config)) == 0

        assert runner.stdin[2] == _stop_body(sourced_config).encode("utf-8")
        assert endpoint.tools == ["dispatch_list", "task_post", "dispatch_report", "dispatch_claim"]
        reason = (
            "LEASE_NOT_HELD: still running 1s past its lease deadline 1757000600, so the runner "
            "ended its process tree; raise libs/demo's expected_minutes if the suite needs longer"
        )
        line = narrow_json_to_str(endpoint.arguments[1]["body"])
        assert line.startswith(f"FLEET-CHECK {DEFAULT_JOB_ID[:8]} libs/demo ")
        assert " exit=124 " in line
        assert line.endswith(f" stopped: {reason}")
        closed = endpoint.arguments[2]
        assert closed["status"] == "failed"
        assert closed["exitCode"] == node_collect.TIMED_OUT_EXIT_CODE
        assert closed["detail"] == line
        last = _ledger(sourced_config)[-1]
        assert last["outcome"] == "failed"
        assert last["exit_code"] == node_collect.TIMED_OUT_EXIT_CODE
        assert last["detail"] == reason
        assert _lease_released(sourced_config, now_unix=PAST_THE_LEASE)

    def test_a_build_at_its_deadline_is_renewed_not_stopped(
        self, sourced_config: pathlib.Path
    ) -> None:
        """The deadline is the last second the lease covers, so the stop
        begins one second after it and not on it."""
        launch(sourced_config)
        _test_hooks.now = FakeClock(PAST_THE_LEASE - 1)
        _test_hooks.run = FakeRun([ok(""), ok(""), *PROBED])
        endpoint = FakeQueue(
            [
                held_answer(),
                dump_json_str({"job": queue_job(status="running", node="lavender")}),
                dump_json_str({"claimed": None}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert node_agent.main(node_argv(sourced_config)) == 0

        assert endpoint.tools == ["dispatch_list", "dispatch_report", "dispatch_claim"]
        assert endpoint.arguments[1]["action"] == "progress"
        assert _ledger(sourced_config)[-1]["outcome"] == "running"


class TestCancelledUnderIt:
    def test_a_run_whose_job_was_cancelled_is_stopped_and_closed_cancelled(
        self, sourced_config: pathlib.Path
    ) -> None:
        launch(sourced_config)
        runner = FakeRun([ok(""), ok("stopped"), *PROBED])
        _test_hooks.run = runner
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                _cancelled_page([CANCELLED_JOB], None),
                dump_json_str({"claimed": None}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert node_agent.main(node_argv(sourced_config)) == 0

        assert endpoint.tools == ["dispatch_list", "dispatch_list", "dispatch_claim"]
        assert endpoint.arguments[1] == {
            "claimedBy": "fleet-node-lavender",
            "status": "cancelled",
            "offset": 0,
            "limit": queue.LISTING_PAGE_LIMIT,
        }
        assert runner.stdin[0] == _stop_body(sourced_config).encode("utf-8")
        last = _ledger(sourced_config)[-1]
        assert last["outcome"] == "cancelled"
        assert last["detail"] == (
            f"queue job {DEFAULT_JOB_ID} was cancelled while it ran; stopped by "
            "fleet-node-lavender; was dispatched by opus-dispatch-0905"
        )
        assert _lease_released(sourced_config, now_unix=DEMO_NOW)

    def test_the_listing_is_paged_until_the_run_is_found(
        self, sourced_config: pathlib.Path
    ) -> None:
        launch(sourced_config)
        _test_hooks.run = FakeRun([ok(""), ok("stopped"), *PROBED])
        older = queue_job(status="cancelled", node="lavender", runId="libs-demo-1756000000")
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                _cancelled_page([older], 100),
                _cancelled_page([CANCELLED_JOB], None),
                dump_json_str({"claimed": None}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert node_agent.main(node_argv(sourced_config)) == 0

        assert [arguments.get("offset") for arguments in endpoint.arguments[1:3]] == [0, 100]
        assert _ledger(sourced_config)[-1]["outcome"] == "cancelled"

    def test_paging_stops_once_every_candidate_is_accounted_for(
        self, sourced_config: pathlib.Path
    ) -> None:
        """A further page is not asked for when nothing is left to find,
        however long the runner's cancelled history is."""
        launch(sourced_config)
        _test_hooks.run = FakeRun([ok(""), ok("stopped"), *PROBED])
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                _cancelled_page([CANCELLED_JOB], 100),
                dump_json_str({"claimed": None}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert node_agent.main(node_argv(sourced_config)) == 0

        assert endpoint.tools == ["dispatch_list", "dispatch_list", "dispatch_claim"]

    def test_a_live_run_no_cancelled_job_names_is_left_running(
        self, sourced_config: pathlib.Path
    ) -> None:
        """A run this runner cannot account for, one dispatched by hand with
        fleet-run say, is not this runner's to stop."""
        launch(sourced_config)
        runner = FakeRun(list(PROBED))
        _test_hooks.run = runner
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                _cancelled_page([], None),
                dump_json_str({"claimed": None}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert node_agent.main(node_argv(sourced_config)) == 0

        assert len(runner.calls) == len(PROBED)
        assert _ledger(sourced_config)[-1]["outcome"] == "running"
        assert not _lease_released(sourced_config, now_unix=DEMO_NOW)

    def test_a_tick_with_no_live_run_asks_the_queue_nothing_about_cancels(
        self, sourced_config: pathlib.Path
    ) -> None:
        _test_hooks.run = FakeRun(list(PROBED))
        endpoint = FakeQueue([dump_json_str({"jobs": []}), dump_json_str({"claimed": None})])
        _test_hooks.http_post = endpoint

        assert node_agent.main(node_argv(sourced_config)) == 0

        assert endpoint.tools == ["dispatch_list", "dispatch_claim"]
