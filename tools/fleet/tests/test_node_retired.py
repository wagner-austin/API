"""A held run whose project the roll removed is settled, and the pass goes on.

MCPs board task 4ec65b8f.

Measured 2026-10-25: loki's node runner held job 45dc05d6 for MCPs/execution
when the roll of API 48bbcd716 removed that project, its build ended exit 0
at 19:00:47Z, and from 19:02:55Z every tick of fleet-node-loki ended on
``WORKSPACE_PROJECT_UNKNOWN`` for that row and claimed nothing. Each case runs
the real collect pass against the fake queue and the fake ssh runner, on the
demo run lavender launched, with a workspace that no longer declares the
demo project, as the rolled one no longer declared MCPs/execution; the pass
holds a second running job after it, which it must still reach.
"""

from __future__ import annotations

import logging
import pathlib

import pytest
from platform_core.json_utils import dump_json_str, narrow_json_to_str

from fleet.cli import _config, node_collect
from fleet.contracts.ledger import NO_EXIT_CODE
from fleet.contracts.workspace import FleetWorkspace
from fleet.core import _test_hooks, leases, queue, records
from tests._holder_fakes import RecordingHolder
from tests._node_agent_fixtures import (
    PASSING_TAIL,
    VERDICT_TASK,
    _credentials_in_env,
    _sourced_config,
    launch,
)
from tests._queue_fakes import DEFAULT_JOB_ID, FakeQueue, queue_job
from tests.conftest import DEMO_NOW, DEMO_PROJECT, DEMO_RUN_ID, FakeRun, ok, retire_replies

__all__ = ["_credentials_in_env", "_sourced_config"]

#: This node's runner.
LAVENDER = "fleet-node-lavender"

#: The second job the pass holds, running under lavender with no live row here.
SECOND_JOB_ID = "bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb"
SECOND_RUN_ID = f"libs-demo-lavender-{DEMO_NOW + 5}"

#: Well past the demo project's lease, which ends at ``DEMO_NOW + 600``.
PAST_THE_LEASE = DEMO_NOW + 900

#: What the demo run's row opens with once its project is gone.
RETIRED = (
    "WORKSPACE_PROJECT_UNKNOWN: libs/demo is not in the rolled workspace, "
    "so its lease was not judged"
)


def _held() -> str:
    """The queue's listing of the two jobs lavender holds: the demo run's, then the second.

    Returns:
        The rendered ``dispatch_list`` answer.
    """
    return dump_json_str(
        {
            "jobs": [
                queue_job(
                    status="running",
                    node="lavender",
                    runId=DEMO_RUN_ID,
                    claimedBy=LAVENDER,
                    taskId=VERDICT_TASK,
                ),
                queue_job(
                    id=SECOND_JOB_ID,
                    status="running",
                    node="lavender",
                    runId=SECOND_RUN_ID,
                    claimedBy=LAVENDER,
                    taskId=VERDICT_TASK,
                ),
            ]
        }
    )


def _retired(config_path: pathlib.Path) -> _config.LoadedWorkspace:
    """The workspace as a roll that removed the demo project left it.

    Args:
        config_path: The workspace document.

    Returns:
        The loaded workspace, every project but the demo one still declared.
    """
    loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
    projects = {
        path: project
        for path, project in loaded.workspace["projects"].items()
        if path != DEMO_PROJECT
    }
    workspace: FleetWorkspace = {**loaded.workspace, "projects": projects}
    return _config.LoadedWorkspace(workspace, loaded.registry, loaded.directory)


def _pass(loaded: _config.LoadedWorkspace) -> RecordingHolder:
    """Run lavender's collect pass once.

    Args:
        loaded: The workspace the pass reads.

    Returns:
        The watch it handed runs to.
    """
    holder = RecordingHolder()
    node_collect.collect_pass(
        loaded,
        queue.load_credentials(),
        queue.load_credentials(),
        {},
        agent=LAVENDER,
        alias="lavender",
        holder=holder,
        launching=frozenset(),
    )
    return holder


def _reached_the_second(caplog: pytest.LogCaptureFixture) -> bool:
    """Whether the pass collected the second job after the demo one.

    Args:
        caplog: The pass's log.

    Returns:
        True when its line says it found no live run of it here.
    """
    return any(
        f"run={SECOND_RUN_ID}: no live run on this machine, leaving it" in record.getMessage()
        for record in caplog.records
    )


class TestARunTheRollRetired:
    def test_that_finished_past_its_lease_closes_with_its_own_exit_and_the_pass_goes_on(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """45dc05d6 on loki: finished exit 0, and every tick raised on it."""
        launch(sourced_config)
        loaded = _retired(sourced_config)
        _test_hooks.run = FakeRun(
            [ok(""), ok(f"0 {PAST_THE_LEASE}"), ok(""), ok(PASSING_TAIL), *retire_replies()]
        )
        endpoint = FakeQueue([_held(), dump_json_str({"job": queue_job(status="passed")})])
        _test_hooks.http_post = endpoint

        with caplog.at_level(logging.INFO):
            holder = _pass(loaded)

        assert endpoint.tools == ["dispatch_list", "dispatch_report"]
        closed = endpoint.arguments[1]
        assert closed["action"] == "close"
        assert closed["status"] == "passed"
        assert closed["exitCode"] == 0
        assert narrow_json_to_str(closed["detail"]).startswith(
            f"FLEET-CHECK {DEFAULT_JOB_ID[:8]} libs/demo "
        )
        last = records.read_ledger(loaded.ledger)[-1]
        assert last["run_id"] == DEMO_RUN_ID
        assert last["outcome"] == "passed"
        assert last["exit_code"] == 0
        assert last["detail"].startswith(f"{RETIRED}; ")
        assert leases.find_by_run(loaded.leases, run_id=DEMO_RUN_ID, now_unix=DEMO_NOW) is None
        assert holder.held == [frozenset({DEMO_RUN_ID, SECOND_RUN_ID})]
        assert _reached_the_second(caplog)

    def test_still_running_is_given_back_then_stopped_and_closed_lost(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        launch(sourced_config)
        loaded = _retired(sourced_config)
        runner = FakeRun([ok(""), ok(""), ok(""), ok("stopped"), *retire_replies()])
        _test_hooks.run = runner
        endpoint = FakeQueue([_held(), dump_json_str({"job": queue_job(status="queued")})])
        _test_hooks.http_post = endpoint

        with caplog.at_level(logging.INFO):
            _pass(loaded)

        detail = (
            f"{RETIRED}; {DEMO_RUN_ID} was still running on lavender, so its job was given back "
            "and the runner ended its process tree"
        )
        # Given back before the stop: no exit status is invented for it.
        assert endpoint.tools == ["dispatch_list", "dispatch_report"]
        report = endpoint.arguments[1]
        assert report["action"] == "lost"
        assert "exitCode" not in report
        assert report["detail"] == detail
        assert "stop-" in " ".join(runner.calls[2])
        last = records.read_ledger(loaded.ledger)[-1]
        assert last["outcome"] == "lost"
        assert last["exit_code"] == NO_EXIT_CODE
        assert last["detail"] == detail
        assert any(
            record.getMessage().endswith(f": {detail}; the queue left it queued")
            for record in caplog.records
        )
        assert _reached_the_second(caplog)
