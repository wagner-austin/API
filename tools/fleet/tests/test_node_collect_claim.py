"""Claims an earlier tick left without a start (MCPs board task 5a4f9b3e).

Measured on 2026-09-29: serendipity claimed job 0a15353c at 09:00:11Z,
launched its run, and died at 09:00:58Z when report_start met a refused
connection. The queue row stayed claimed with no run id, the ledger row
stayed live, and nothing ever looked at either again, so the node counted
the orphan against its one run slot until 10:18Z. Each test here drives
whole ticks through ``node_agent.main`` against the fake queue and the fake
ssh runner, starting from that state or from the cancel that followed it.
"""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime
from urllib.error import URLError

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import dump_json_str
from platform_core.mcp_client import McpHttpResponse

from fleet.cli import _config, node_agent, node_collect
from fleet.contracts.ledger import LedgerEntry
from fleet.core import _test_hooks, leases, records, staging
from tests._node_agent_fixtures import (
    PROBED,
    VERDICT_TASK,
    _credentials_in_env,
    _sourced_config,
    claim_replies,
    launch,
    node_argv,
    prebuilt_export,
)
from tests._queue_fakes import DEFAULT_JOB_ID, FakeQueue, queue_job
from tests.conftest import DEMO_NOW, DEMO_RUN_ID, FakeRun, ok, retire_replies

__all__ = ["_credentials_in_env", "_sourced_config"]


def _instant(unix: int) -> str:
    """An instant as the queue renders one.

    Args:
        unix: Whole seconds since the epoch.

    Returns:
        JavaScript's ``toISOString`` of it.
    """
    return datetime.fromtimestamp(unix, UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")


#: The demo job as the queue holds it after its start report was lost:
#: claimed by lavender's runner at the moment the run launched, no run id.
ORPHANED = queue_job(
    status="claimed",
    claimedBy="fleet-node-lavender",
    claimedAt=_instant(DEMO_NOW),
    taskId=VERDICT_TASK,
)

#: The queue's answer to the adopting start report.
STARTED = dump_json_str({"job": queue_job(status="running", node="lavender", runId=DEMO_RUN_ID)})

#: An empty page of the cancelled listing.
NO_CANCELS = dump_json_str({"jobs": [], "pagination": {"nextOffset": None}})


class RefusedAt:
    """A queue endpoint that answers from a script until one call, which the
    network refuses, as diphtheria's did while its services were recreated.

    Attributes:
        queue: The scripted endpoint answering every other call.
    """

    def __init__(self, queue: FakeQueue, *, refused_call: int) -> None:
        """Build it.

        Args:
            queue: What answers the calls before the refused one.
            refused_call: The zero-based index of the call that is refused.
        """
        self.queue = queue
        self._refused_call = refused_call
        self._calls = 0

    def __call__(
        self, url: str, *, headers: dict[str, str], body: bytes, timeout_seconds: int
    ) -> McpHttpResponse:
        """Answer the call, or refuse it.

        Args:
            url: Absolute URL posted to.
            headers: Every request header.
            body: The encoded JSON-RPC body.
            timeout_seconds: The caller's timeout.

        Returns:
            The scripted answer.

        Raises:
            URLError: On the refused call, as urllib raises a refused connection.
        """
        call = self._calls
        self._calls += 1
        if call == self._refused_call:
            raise URLError(ConnectionRefusedError(10061, "the target machine actively refused it"))
        return self.queue(url, headers=headers, body=body, timeout_seconds=timeout_seconds)


def _ledger(config_path: pathlib.Path) -> tuple[LedgerEntry, ...]:
    """Every ledger row the workspace holds, oldest first."""
    loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
    return records.read_ledger(loaded.ledger)


def _live_runs(config_path: pathlib.Path) -> int:
    """How many runs lavender's capacity check counts."""
    loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
    return records.live_runs(loaded.ledger, node="lavender")


class TestTheFailure:
    def test_a_refused_start_report_ends_the_tick_with_the_run_launched_and_live(
        self, sourced_config: pathlib.Path
    ) -> None:
        payload = prebuilt_export(sourced_config)
        _test_hooks.run = FakeRun(claim_replies(staging.digest(payload), commit_present=True))
        claimed = queue_job(status="claimed", taskId=VERDICT_TASK)
        queue = FakeQueue([dump_json_str({"jobs": []}), dump_json_str({"claimed": claimed})])
        _test_hooks.http_post = RefusedAt(queue, refused_call=2)

        with pytest.raises(URLError):
            node_agent.main(node_argv(sourced_config))

        assert queue.tools == ["dispatch_list", "dispatch_claim"]
        assert _ledger(sourced_config)[-1]["outcome"] == "running"
        assert _live_runs(sourced_config) == 1


class TestTheNextTick:
    def test_adopts_the_run_its_claim_launched_by_reporting_its_start(
        self, sourced_config: pathlib.Path
    ) -> None:
        launch(sourced_config)
        _test_hooks.run = FakeRun(list(PROBED))
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": [ORPHANED]}),
                STARTED,
                STARTED,
                NO_CANCELS,
                dump_json_str({"claimed": None}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert node_agent.main(node_argv(sourced_config)) == 0

        assert endpoint.tools == [
            "dispatch_list",
            "dispatch_report",
            "dispatch_report",
            "dispatch_list",
            "dispatch_claim",
        ]
        # The trail says why the start came a tick late (55f2cb0b, A2).
        assert endpoint.arguments[2]["action"] == "progress"
        assert endpoint.arguments[2]["note"] == (
            f"adopted on lavender: {DEMO_RUN_ID} was launched by the claiming tick, whose start "
            "report never reached the queue"
        )
        assert endpoint.arguments[1] == {
            "jobId": DEFAULT_JOB_ID,
            "action": "start",
            "node": "lavender",
            "runId": DEMO_RUN_ID,
            "leaseSeconds": node_collect.CLAIM_LEASE_SECONDS,
            "agent": "fleet-node-lavender",
            "sessionId": endpoint.arguments[1]["sessionId"],
            "cwd": endpoint.arguments[1]["cwd"],
        }
        # The run is adopted, not stopped: it still counts against the slot it
        # is using, and the next tick collects it as the running job it now is.
        assert _ledger(sourced_config)[-1]["outcome"] == "running"
        assert _live_runs(sourced_config) == 1

    def test_a_run_that_began_just_before_the_queue_stamped_the_claim_is_still_its_run(
        self, sourced_config: pathlib.Path
    ) -> None:
        """The claim is stamped by the database host and the row by the hub,
        so a row up to a minute older than the claim still matches."""
        launch(sourced_config)
        _test_hooks.run = FakeRun(list(PROBED))
        late = _instant(DEMO_NOW + node_collect.CLAIM_CLOCK_SLACK_SECONDS)
        stamped_late = {**ORPHANED, "claimedAt": late}
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": [stamped_late]}),
                STARTED,
                STARTED,
                NO_CANCELS,
                dump_json_str({"claimed": None}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert node_agent.main(node_argv(sourced_config)) == 0

        assert endpoint.arguments[1]["runId"] == DEMO_RUN_ID

    def test_refuses_a_claim_that_launched_nothing_and_claims_again_in_the_same_tick(
        self, sourced_config: pathlib.Path
    ) -> None:
        _test_hooks.run = FakeRun(list(PROBED))
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": [ORPHANED]}),
                dump_json_str({"job": queue_job(status="refused")}),
                dump_json_str({"claimed": None}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert node_agent.main(node_argv(sourced_config)) == 0

        assert endpoint.tools == ["dispatch_list", "dispatch_report", "dispatch_claim"]
        assert endpoint.arguments[1]["action"] == "close"
        assert endpoint.arguments[1]["status"] == "refused"
        assert endpoint.arguments[1]["detail"] == (
            f"DISPATCH_NOT_LAUNCHED: {DEFAULT_JOB_ID} was claimed on lavender, its start report "
            "never reached the queue, and nothing was launched; refused so the submitter can "
            "resubmit it"
        )

    def test_refuses_rather_than_guesses_between_two_runs_either_claim_could_have_launched(
        self, sourced_config: pathlib.Path
    ) -> None:
        launch(sourced_config)
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(sourced_config)})
        twin: LedgerEntry = {**_ledger(sourced_config)[-1], "run_id": f"{DEMO_RUN_ID}-twin"}
        records.append_ledger(loaded.ledger, twin)
        _test_hooks.run = FakeRun(list(PROBED))
        _test_hooks.http_post = FakeQueue([dump_json_str({"jobs": [ORPHANED]})])

        with pytest.raises(AppError) as raised:
            node_agent.main(node_argv(sourced_config))

        assert raised.value.code is FleetErrorCode.DISPATCH_CLAIM_AMBIGUOUS
        assert raised.value.message == (
            f"{DEFAULT_JOB_ID} claimed make check libs/demo at 4e3c6bc1d9f0 @any node never "
            f"reported a start, and 2 live runs on lavender could be the one it launched: "
            f"{DEMO_RUN_ID}, {DEMO_RUN_ID}-twin"
        )


class TestACancelBeforeTheStart:
    def test_reaches_the_run_the_claim_launched_and_frees_the_slot_in_the_same_tick(
        self, sourced_config: pathlib.Path
    ) -> None:
        launch(sourced_config)
        _test_hooks.run = FakeRun([ok(""), ok("stopped"), *retire_replies(), *PROBED])
        cancelled = {**ORPHANED, "status": "cancelled"}
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str({"jobs": [cancelled], "pagination": {"nextOffset": None}}),
                dump_json_str({"claimed": None}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert node_agent.main(node_argv(sourced_config)) == 0

        # The slot is free by the claim pass of the same tick, which asks.
        assert endpoint.tools == ["dispatch_list", "dispatch_list", "dispatch_claim"]
        assert _ledger(sourced_config)[-1]["outcome"] == "cancelled"
        assert _live_runs(sourced_config) == 0
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(sourced_config)})
        assert leases.find_by_run(loaded.leases, run_id=DEMO_RUN_ID, now_unix=DEMO_NOW) is None

    @pytest.mark.parametrize(
        "claimed_at",
        [
            pytest.param(
                _instant(DEMO_NOW + 2 * node_collect.CLAIM_CLOCK_SLACK_SECONDS),
                id="claimed-after-it-began",
            ),
            pytest.param(
                _instant(DEMO_NOW - node_collect.CLAIM_LEASE_SECONDS - 1), id="an-older-claim"
            ),
            pytest.param(None, id="never-claimed"),
        ],
    )
    def test_never_stops_a_run_its_claim_cannot_have_launched(
        self, sourced_config: pathlib.Path, claimed_at: str | None
    ) -> None:
        """An old cancelled job of the same submitter, session and project is
        not the orphan's: the run began outside that claim's window."""
        launch(sourced_config)
        runner = FakeRun(list(PROBED))
        _test_hooks.run = runner
        cancelled = {**ORPHANED, "status": "cancelled", "claimedAt": claimed_at}
        _test_hooks.http_post = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str({"jobs": [cancelled], "pagination": {"nextOffset": None}}),
                dump_json_str({"claimed": None}),
            ]
        )

        assert node_agent.main(node_argv(sourced_config)) == 0

        assert len(runner.calls) == len(PROBED)
        assert _ledger(sourced_config)[-1]["outcome"] == "running"
