"""The collect half of a node runner's tick, its ``--announce`` registration
and its entry points (MCPs board task fd5cabfa, A3).

A later tick finds the job it launched among ``held_by``, reads the result
and the transcript's tail off the node, composes the verdict line, posts it
to the submitting task's thread (or the submitter's feed when the job names
no task) and closes the job on both sides. The claim half is
``test_node_agent.py``; the fixtures both use are ``_node_agent_fixtures.py``.
"""

from __future__ import annotations

import pathlib
import runpy
import sys

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import dump_json_str, narrow_json_to_str

from fleet.cli import node_agent, node_collect
from fleet.core import _test_hooks
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
from tests._queue_fakes import DEFAULT_JOB_ID, DEFAULT_SHA, FakeQueue, queue_job
from tests.conftest import DEMO_PROJECT, DEMO_RUN_ID, FakeRun, ok

__all__ = ["_credentials_in_env", "_sourced_config"]


class TestCollecting:
    def test_a_finished_suite_posts_its_verdict_to_the_task_and_closes_both_sides(
        self, sourced_config: pathlib.Path
    ) -> None:
        launch(sourced_config)
        _test_hooks.run = FakeRun([ok(""), ok("0 1757000060"), ok(""), ok(PASSING_TAIL), *PROBED])
        endpoint = FakeQueue(
            [
                held_answer(taskId=VERDICT_TASK),
                "posted",
                dump_json_str({"job": queue_job(status="passed")}),
                dump_json_str({"claimed": None}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert node_agent.main(node_argv(sourced_config)) == 0

        assert endpoint.tools == ["dispatch_list", "task_post", "dispatch_report", "dispatch_claim"]
        posted = endpoint.arguments[1]
        assert posted["kind"] == "note"
        assert posted["taskId"] == VERDICT_TASK
        assert posted["agent"] == "fleet-node-lavender"
        line = narrow_json_to_str(posted["body"])
        assert line == (
            f"FLEET-CHECK {DEFAULT_JOB_ID[:8]} {DEMO_PROJECT} sha={DEFAULT_SHA} node=lavender "
            "exit=0 banner=yes tests=887p/0f coverage=statements=100% branches=100% "
            f"log=lavender:C:/fleet/stage/{DEMO_RUN_ID}/result.txt.log run={DEMO_RUN_ID}"
        )
        closed = endpoint.arguments[2]
        assert closed["action"] == "close"
        assert closed["status"] == "passed"
        assert closed["exitCode"] == 0
        assert closed["detail"] == line
        ledger = (sourced_config.parent / "runs" / "ledger.jsonl").read_text(encoding="utf-8")
        assert "passed" in ledger

    def test_a_job_with_no_task_is_addressed_to_the_submitter_in_the_fleet_room(
        self, sourced_config: pathlib.Path
    ) -> None:
        launch(sourced_config)
        _test_hooks.run = FakeRun(
            [ok(""), ok("2 1757000060"), ok(""), ok("3 failed, 100 passed\n"), *PROBED]
        )
        endpoint = FakeQueue(
            [
                held_answer(),
                "posted",
                dump_json_str({"job": queue_job(status="failed")}),
                dump_json_str({"claimed": None}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert node_agent.main(node_argv(sourced_config)) == 0

        posted = endpoint.arguments[1]
        assert "taskId" not in posted
        assert posted["room"] == "fleet"
        body = narrow_json_to_str(posted["body"])
        assert body.startswith("@opus-dispatch-0905 FLEET-CHECK ")
        assert "exit=2 banner=no tests=100p/3f coverage=unread" in body
        assert endpoint.arguments[2]["status"] == "failed"
        assert endpoint.arguments[2]["exitCode"] == 2

    def test_a_suite_still_running_has_its_lease_renewed(
        self, sourced_config: pathlib.Path
    ) -> None:
        launch(sourced_config)
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
        renewed = endpoint.arguments[1]
        assert renewed["action"] == "progress"
        assert renewed["leaseSeconds"] == node_collect.CLAIM_LEASE_SECONDS
        assert renewed["note"] == f"still running on lavender as {DEMO_RUN_ID}"

    def test_a_job_whose_run_this_machine_never_had_is_left_alone(
        self, sourced_config: pathlib.Path
    ) -> None:
        _test_hooks.run = FakeRun(PROBED)
        endpoint = FakeQueue(
            [
                dump_json_str(
                    {
                        "jobs": [
                            queue_job(status="running", node="lavender", runId="somebody-elses-run")
                        ]
                    }
                ),
                dump_json_str({"claimed": None}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert node_agent.main(node_argv(sourced_config)) == 0

        assert endpoint.tools == ["dispatch_list", "dispatch_claim"]

    def test_a_held_job_that_has_not_started_is_skipped(self, sourced_config: pathlib.Path) -> None:
        _test_hooks.run = FakeRun(PROBED)
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": [queue_job(status="claimed")]}),
                dump_json_str({"claimed": None}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert node_agent.main(node_argv(sourced_config)) == 0

        assert endpoint.tools == ["dispatch_list", "dispatch_claim"]

    def test_a_build_that_outlived_its_lease_stops_the_tick(
        self, sourced_config: pathlib.Path
    ) -> None:
        launch(sourced_config)
        _test_hooks.run = FakeRun([ok(""), ok("0 1757003600")])
        _test_hooks.http_post = FakeQueue([held_answer()])

        with pytest.raises(AppError) as raised:
            node_agent.main(node_argv(sourced_config))

        assert raised.value.code is FleetErrorCode.LEASE_NOT_HELD

    def test_a_held_running_row_without_a_sha_is_a_contract_fault(
        self, sourced_config: pathlib.Path
    ) -> None:
        launch(sourced_config)
        _test_hooks.run = FakeRun([ok(""), ok("0 1757000060")])
        _test_hooks.http_post = FakeQueue([held_answer(sha=None)])

        with pytest.raises(AppError) as raised:
            node_agent.main(node_argv(sourced_config))

        assert raised.value.code is FleetErrorCode.QUEUE_ANSWER_MALFORMED


class TestAnnounceAndEntryPoints:
    def test_announce_registers_the_runner_on_the_ledger_and_claims_nothing(
        self, sourced_config: pathlib.Path
    ) -> None:
        _test_hooks.hostname = lambda: "austinpc"
        endpoint = FakeQueue(["checked in"])
        _test_hooks.http_post = endpoint

        assert node_agent.main([*node_argv(sourced_config), node_agent.ANNOUNCE_FLAG]) == 0

        assert endpoint.tools == ["task_post"]
        checkin = endpoint.arguments[0]
        assert checkin["kind"] == "checkin"
        assert checkin["harness"] == "fleet-agent"
        assert checkin["machine"] == f"{sys.platform}:austinpc"
        assert checkin["room"] == "fleet"
        assert checkin["agent"] == "fleet-node-lavender"
        assert "carrying windows" in narrow_json_to_str(checkin["body"])

    def test_an_undeclared_node_is_refused_before_any_call(
        self, sourced_config: pathlib.Path
    ) -> None:
        endpoint = FakeQueue([])
        _test_hooks.http_post = endpoint

        with pytest.raises(AppError) as raised:
            node_agent.main(["--config", str(sourced_config), node_agent.NODE_FLAG, "nowhere"])

        assert raised.value.code is FleetErrorCode.WORKSPACE_NODE_UNKNOWN
        assert endpoint.tools == []

    def test_the_console_entry_point_exits_zero(self, sourced_config: pathlib.Path) -> None:
        _test_hooks.run = FakeRun(PROBED)
        _test_hooks.http_post = FakeQueue(
            [dump_json_str({"jobs": []}), dump_json_str({"claimed": None})]
        )
        saved = sys.argv
        sys.argv = ["fleet-node-agent", *node_argv(sourced_config)]
        try:
            with pytest.raises(SystemExit) as raised:
                node_agent.entrypoint()
        finally:
            sys.argv = saved

        assert raised.value.code == 0

    def test_running_as_a_module_actually_runs(self, sourced_config: pathlib.Path) -> None:
        """Without the ``if __name__`` block ``python -m`` imports the module,
        runs nothing and exits 0, which reads as a tick that found nothing."""
        _test_hooks.run = FakeRun(PROBED)
        _test_hooks.http_post = FakeQueue(
            [dump_json_str({"jobs": []}), dump_json_str({"claimed": None})]
        )
        saved_argv = sys.argv
        saved_module = sys.modules.pop("fleet.cli.node_agent", None)
        sys.argv = ["x", *node_argv(sourced_config)]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module("fleet.cli.node_agent", run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules["fleet.cli.node_agent"] = saved_module

        assert raised.value.code == 0
