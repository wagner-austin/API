"""One tick of the node runner, against a faked queue and a faked fleet.

The queue speaks the real JSON-RPC-over-SSE shape and the node speaks through
the same command hook every other dispatch test uses, so what is exercised is
the real engine -- the real staging, the real capacity check, the real ledger
-- with only the two boundaries this machine cannot cross in a test faked.

WHAT EACH TEST IS REALLY ASKING. A tick has two passes and three outcomes per
pass, and the ones that matter are the unhappy ones: a job that no node can
take must come back to the submitter as ``refused`` rather than vanishing,
and a finished suite must be closed on BOTH sides or the queue and this
machine's ledger disagree about a run that is over.
"""

from __future__ import annotations

import pathlib
import runpy
import sys

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import dump_json_str, narrow_json_to_str

from fleet.cli import agent
from fleet.core import _test_hooks, queue, staging
from tests._queue_fakes import FakeEnv, FakeQueue, queue_job
from tests.conftest import (
    DEMO_PROJECT,
    DEMO_RUN_ID,
    FakeRun,
    dispatch_replies,
    ok,
    prebuilt_archive,
)

JOB_ID = "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa"


@pytest.fixture(name="credentials_in_env", autouse=True)
def _credentials_in_env() -> None:
    """Give every test the two variables the agent refuses to run without."""
    _test_hooks.env = FakeEnv(
        {queue.API_KEY_VARIABLE: "test-key", queue.TENANT_ID_VARIABLE: "tenant"}
    )


def argv(config_path: pathlib.Path, repo: pathlib.Path) -> list[str]:
    """Build the agent's arguments for a tick.

    Args:
        config_path: The workspace document.
        repo: The monorepo root on this machine.

    Returns:
        The argument list.
    """
    return [
        "--config",
        str(config_path),
        agent.AGENT_FLAG,
        "fleet-runner-austinpc",
        agent.SESSION_FLAG,
        "33333333-cccc-4ccc-8ccc-333333333333",
        agent.ROOT_FLAG,
        str(repo),
    ]


class TestAnEmptyQueue:
    def test_a_tick_with_nothing_to_do_succeeds(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """The outcome of most ticks, and it must be exit 0 -- a scheduling
        loop that stopped on an empty queue would stop immediately."""
        endpoint = FakeQueue([dump_json_str({"jobs": []}), dump_json_str({"claimed": None})])
        _test_hooks.http_post = endpoint

        assert agent.main(argv(config_path, repo)) == 0
        assert endpoint.tools == ["dispatch_list", "dispatch_claim"]

    def test_collection_runs_before_the_claim(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """Order matters: a tick that claimed first would leave a finished
        result unreported for a whole interval."""
        endpoint = FakeQueue([dump_json_str({"jobs": []}), dump_json_str({"claimed": None})])
        _test_hooks.http_post = endpoint

        agent.main(argv(config_path, repo))

        assert endpoint.tools[0] == "dispatch_list"


class TestClaiming:
    def test_a_claimed_job_is_staged_launched_and_reported_started(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        payload = prebuilt_archive(config_path, repo)
        _test_hooks.run = FakeRun(dispatch_replies(staging.digest(payload)))
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": queue_job(status="claimed")}),
                dump_json_str({"job": queue_job(status="running", node="lavender")}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert agent.main(argv(config_path, repo)) == 0

        assert endpoint.tools == ["dispatch_list", "dispatch_claim", "dispatch_report"]
        started = endpoint.arguments[2]
        assert started["action"] == "start"
        assert started["node"] == "lavender"
        assert started["runId"] == DEMO_RUN_ID

    def test_the_ledger_row_names_the_submitting_session_not_the_runner(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """Acceptance criterion 4 of the task this package was built for: a
        ledger row carries WHO dispatched it. A row stamped with the runner's
        own label would record only that the runner ran something."""
        payload = prebuilt_archive(config_path, repo)
        _test_hooks.run = FakeRun(dispatch_replies(staging.digest(payload)))
        _test_hooks.http_post = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str(
                    {
                        "claimed": queue_job(
                            status="claimed",
                            submittedBy="opus-weight-injection-0902",
                            sessionId="acc774c0-3bc3-4cce-9dda-c7a12fb99519",
                        )
                    }
                ),
                dump_json_str({"job": queue_job(status="running", node="lavender")}),
            ]
        )

        agent.main(argv(config_path, repo))

        ledger = (config_path.parent / "runs" / "ledger.jsonl").read_text(encoding="utf-8")
        assert "opus-weight-injection-0902" in ledger
        assert "acc774c0-3bc3-4cce-9dda-c7a12fb99519" in ledger

    def test_a_job_no_node_can_take_comes_back_as_refused(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """The outcome the queue exists to report. Left unreported, the job
        would sit claimed until its lease lapsed and then be claimed again,
        forever, while the submitter watched a status that never moved."""
        _test_hooks.run = FakeRun([ok(""), ok("free_ram_gb=0.2\nfree_disk_gb=860.0\n")])
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": queue_job(status="claimed")}),
                dump_json_str({"job": queue_job(status="refused")}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert agent.main(argv(config_path, repo)) == 0

        closed = endpoint.arguments[2]
        assert closed["action"] == "close"
        assert closed["status"] == "refused"
        assert "exitCode" not in closed

    def test_the_refusal_carries_the_engine_s_own_code_and_message(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """Transport, not softening: the local error is handed to the queue
        verbatim so the submitter reads what this machine actually said."""
        _test_hooks.run = FakeRun([ok(""), ok("free_ram_gb=0.2\nfree_disk_gb=860.0\n")])
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": queue_job(status="claimed")}),
                dump_json_str({"job": queue_job(status="refused")}),
            ]
        )
        _test_hooks.http_post = endpoint

        agent.main(argv(config_path, repo))

        detail = narrow_json_to_str(endpoint.arguments[2]["detail"])
        assert FleetErrorCode.NODE_MEMORY_EXHAUSTED in detail

    def test_an_unknown_project_is_refused_rather_than_crashing_the_loop(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """A queue job can name a project this workspace has never heard of --
        the fleet.json here is a different file from whatever the submitter
        was looking at. That is a config gap, reported, not a crash."""
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": queue_job(status="claimed", project="libs/absent")}),
                dump_json_str({"job": queue_job(status="refused")}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert agent.main(argv(config_path, repo)) == 0

        detail = narrow_json_to_str(endpoint.arguments[2]["detail"])
        assert FleetErrorCode.WORKSPACE_PROJECT_UNKNOWN in detail


class TestCollecting:
    def _launch(self, config_path: pathlib.Path, repo: pathlib.Path) -> None:
        """Run one tick that claims and launches, leaving a live ledger row.

        Args:
            config_path: The workspace document.
            repo: The monorepo root.
        """
        payload = prebuilt_archive(config_path, repo)
        _test_hooks.run = FakeRun(dispatch_replies(staging.digest(payload)))
        _test_hooks.http_post = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": queue_job(status="claimed")}),
                dump_json_str({"job": queue_job(status="running", node="lavender")}),
            ]
        )
        agent.main(argv(config_path, repo))

    def test_a_finished_suite_is_closed_on_both_sides(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """The queue and this machine's ledger must agree that a run is over.
        Closing one and not the other is how a job stays 'running' forever
        while its node is idle."""
        self._launch(config_path, repo)
        _test_hooks.run = FakeRun([ok(""), ok("0 1757000060")])
        endpoint = FakeQueue(
            [
                dump_json_str(
                    {"jobs": [queue_job(status="running", node="lavender", runId=DEMO_RUN_ID)]}
                ),
                dump_json_str({"job": queue_job(status="passed")}),
                dump_json_str({"claimed": None}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert agent.main(argv(config_path, repo)) == 0

        closed = endpoint.arguments[1]
        assert closed["action"] == "close"
        assert closed["status"] == "passed"
        assert closed["exitCode"] == 0
        ledger = (config_path.parent / "runs" / "ledger.jsonl").read_text(encoding="utf-8")
        assert "passed" in ledger

    def test_a_failing_suite_closes_as_failed_with_its_exit_code(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        self._launch(config_path, repo)
        _test_hooks.run = FakeRun([ok(""), ok("2 1757000060")])
        endpoint = FakeQueue(
            [
                dump_json_str(
                    {"jobs": [queue_job(status="running", node="lavender", runId=DEMO_RUN_ID)]}
                ),
                dump_json_str({"job": queue_job(status="failed")}),
                dump_json_str({"claimed": None}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert agent.main(argv(config_path, repo)) == 0

        assert endpoint.arguments[1]["status"] == "failed"
        assert endpoint.arguments[1]["exitCode"] == 2

    def test_a_suite_still_running_is_left_exactly_as_it_was(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        self._launch(config_path, repo)
        # An empty answer is the node saying "still going" -- the build has
        # not written its result file yet.
        _test_hooks.run = FakeRun([ok(""), ok("")])
        endpoint = FakeQueue(
            [
                dump_json_str(
                    {"jobs": [queue_job(status="running", node="lavender", runId=DEMO_RUN_ID)]}
                ),
                dump_json_str({"claimed": None}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert agent.main(argv(config_path, repo)) == 0

        assert endpoint.tools == ["dispatch_list", "dispatch_claim"]

    def test_a_job_whose_run_this_machine_never_had_is_left_alone(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """What a job claimed by a runner on a DIFFERENT machine looks like
        from here. Reporting on it would be one runner overwriting another's
        verdict; refusing to would stop this runner's whole loop."""
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

        assert agent.main(argv(config_path, repo)) == 0

        assert endpoint.tools == ["dispatch_list", "dispatch_claim"]

    def test_a_held_job_that_has_not_started_is_skipped(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """Claimed but not running: there is no node to ask yet."""
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": [queue_job(status="claimed")]}),
                dump_json_str({"claimed": None}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert agent.main(argv(config_path, repo)) == 0

        assert endpoint.tools == ["dispatch_list", "dispatch_claim"]

    def test_a_build_that_outlived_its_lease_stops_the_tick(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """NOT caught and reported as an outcome. While the lease was gone and
        the build was still writing, a second dispatch could have been
        admitted into the same environment -- so this result may have been
        produced by two suites interfering, and recording it tidily would be
        the last chance anybody had to notice."""
        self._launch(config_path, repo)
        _test_hooks.run = FakeRun([ok(""), ok("0 1757003600")])
        _test_hooks.http_post = FakeQueue(
            [
                dump_json_str(
                    {"jobs": [queue_job(status="running", node="lavender", runId=DEMO_RUN_ID)]}
                )
            ]
        )

        with pytest.raises(AppError) as raised:
            agent.main(argv(config_path, repo))

        assert raised.value.code is FleetErrorCode.LEASE_NOT_HELD


class TestCredentialsAndEntryPoint:
    def test_a_missing_credential_stops_the_tick_before_any_call(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        _test_hooks.env = FakeEnv({})
        endpoint = FakeQueue([])
        _test_hooks.http_post = endpoint

        with pytest.raises(AppError) as raised:
            agent.main(argv(config_path, repo))

        assert raised.value.code is FleetErrorCode.QUEUE_CREDENTIALS_MISSING
        assert endpoint.tools == []

    def test_a_node_scoped_agent_passes_its_node_to_the_claim(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        endpoint = FakeQueue([dump_json_str({"jobs": []}), dump_json_str({"claimed": None})])
        _test_hooks.http_post = endpoint

        agent.main([*argv(config_path, repo), agent.NODE_FLAG, "lavender"])

        assert endpoint.arguments[1]["node"] == "lavender"

    def test_the_console_entry_point_exits_zero(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        _test_hooks.http_post = FakeQueue(
            [dump_json_str({"jobs": []}), dump_json_str({"claimed": None})]
        )
        saved = sys.argv
        sys.argv = ["fleet-agent", *argv(config_path, repo)]
        try:
            with pytest.raises(SystemExit) as raised:
                agent.entrypoint()
        finally:
            sys.argv = saved

        assert raised.value.code == 0

    def test_running_as_a_module_actually_runs(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """The half that silently goes missing without an `if __name__` block:
        `python -m` would import the module, run nothing, and exit 0 -- which
        reads as a tick that found an empty queue, on a runner that never
        asked."""
        _test_hooks.http_post = FakeQueue(
            [dump_json_str({"jobs": []}), dump_json_str({"claimed": None})]
        )
        saved_argv = sys.argv
        saved_module = sys.modules.pop("fleet.cli.agent", None)
        sys.argv = ["x", *argv(config_path, repo)]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module("fleet.cli.agent", run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules["fleet.cli.agent"] = saved_module

        assert raised.value.code == 0

    def test_the_project_is_the_one_the_queue_named(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """A runner that built its OWN idea of the project would ignore the
        queue entirely, and every test above would still pass."""
        payload = prebuilt_archive(config_path, repo)
        _test_hooks.run = FakeRun(dispatch_replies(staging.digest(payload)))
        _test_hooks.http_post = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": queue_job(status="claimed", project=DEMO_PROJECT)}),
                dump_json_str({"job": queue_job(status="running", node="lavender")}),
            ]
        )

        agent.main(argv(config_path, repo))

        ledger = (config_path.parent / "runs" / "ledger.jsonl").read_text(encoding="utf-8")
        assert DEMO_PROJECT in ledger
