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

import logging
import pathlib
from urllib.error import URLError

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import dump_json_str
from platform_core.mcp_client import McpHttpResponse

from fleet.cli import _config, node_agent, node_collect
from fleet.contracts.dispatch import decode_job, encode_job_line
from fleet.contracts.feed import FeedKind
from fleet.contracts.host_claim import HostClaim
from fleet.contracts.lease import Lease
from fleet.contracts.ledger import LedgerEntry
from fleet.contracts.node import LiveLoad
from fleet.core import _test_hooks, claim_window, host_claims, leases, queue, records, staging
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
from tests._queue_fakes import (
    DEFAULT_JOB_ID,
    FakeQueue,
    listing_page,
    queue_instant,
    queue_job,
    trail_answer,
)
from tests.conftest import DEMO_NOW, DEMO_RUN_ID, FakeRun, ok, retire_replies

__all__ = ["_credentials_in_env", "_sourced_config"]


#: The demo job as the queue holds it after its start report was lost:
#: claimed by lavender's runner at the moment the run launched, no run id.
ORPHANED = queue_job(
    status="claimed",
    claimedBy="fleet-node-lavender",
    claimedAt=queue_instant(DEMO_NOW),
    taskId=VERDICT_TASK,
)

#: The session that submitted it, which a lease its claim took names.
ORPHANED_SUBMITTER = "opus-dispatch-0905"
ORPHANED_SESSION = "11111111-aaaa-4aaa-8aaa-111111111111"

#: The job once adopted: running under lavender's runner, naming the run.
ADOPTED = queue_job(
    status="running", node="lavender", runId=DEMO_RUN_ID, claimedBy="fleet-node-lavender"
)

#: The queue's answer to the adopting start report.
STARTED = dump_json_str({"job": ADOPTED})


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


def _plant_lease(
    config_path: pathlib.Path,
    *,
    agent: str = ORPHANED_SUBMITTER,
    session_id: str = ORPHANED_SESSION,
    acquired_unix: int = DEMO_NOW,
) -> Lease:
    """Hold the demo project on lavender as a claiming tick's take leaves it.

    Args:
        config_path: The workspace document.
        agent: The session label the lease names.
        session_id: That session's UUID.
        acquired_unix: When it was taken.

    Returns:
        The lease, as acquired.
    """
    loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
    lease = Lease(
        node="lavender",
        project="libs/demo",
        run_id=f"libs-demo-lavender-{acquired_unix}",
        agent=agent,
        session_id=session_id,
        acquired_unix=acquired_unix,
        expires_unix=DEMO_NOW + 1200,
        resources=(),
    )
    leases.acquire(loaded.leases, lease, now_unix=DEMO_NOW)
    return lease


def _live_runs(config_path: pathlib.Path) -> int:
    """How many runs lavender's capacity check counts."""
    loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
    live = records.live_load(loaded.ledger, node="lavender", projects=loaded.workspace["projects"])
    return live["runs"]


class TestTheFailure:
    def test_a_refused_start_report_ends_the_tick_with_the_run_launched_and_live(
        self, sourced_config: pathlib.Path
    ) -> None:
        payload = prebuilt_export(sourced_config)
        _test_hooks.run = FakeRun(
            claim_replies(staging.digest(payload), registry=sourced_config, commit_present=True)
        )
        claimed = queue_job(status="claimed", taskId=VERDICT_TASK)
        queue = FakeQueue([dump_json_str({"jobs": []}), dump_json_str({"claimed": claimed})])
        # Calls 0 to 2 are the list, the claim and the tick's record; 3 is
        # the start report.
        _test_hooks.http_post = RefusedAt(queue, refused_call=3)

        with pytest.raises(URLError):
            node_agent.main(node_argv(sourced_config))

        assert queue.tools == ["dispatch_list", "dispatch_claim"]
        assert len(queue.ticks) == 1
        assert queue.ticks[0]["claiming"] is True
        job = decode_job(claimed, answer=dump_json_str(claimed))
        assert queue.ticks[0]["verdict"] == f"claimed {encode_job_line(job)}"
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
            ]
        )
        _test_hooks.http_post = endpoint

        assert node_agent.main(node_argv(sourced_config)) == 0

        # No dispatch_claim: the adopted run holds the one project's lease on
        # lavender, so the claim pass leaves it out (MCPs board task 939ec5c7).
        # No cancelled listing and no trail either: the adopted run is the
        # pass's own, so no other live run is left to account for (MCPs
        # board task c1d48330).
        assert endpoint.tools == ["dispatch_list", "dispatch_report", "dispatch_report"]
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
        late = DEMO_NOW + claim_window.CLAIM_CLOCK_SLACK_SECONDS
        stamped_late = {**ORPHANED, "claimedAt": queue_instant(late)}
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": [stamped_late]}),
                STARTED,
                STARTED,
            ]
        )
        _test_hooks.http_post = endpoint

        assert node_agent.main(node_argv(sourced_config)) == 0

        assert endpoint.arguments[1]["runId"] == DEMO_RUN_ID

    @pytest.mark.parametrize("left", ["queued", "failed"])
    def test_gives_back_a_claim_that_launched_nothing_and_claims_again_in_the_same_tick(
        self, sourced_config: pathlib.Path, left: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """MCPs board task 0866f132: a claim whose tick a fleet roll cut off
        ran nothing, so it goes back through the queue's lost report, never a
        refusal; the queue requeues it the first time and fails it the
        second, and the log says which."""
        _test_hooks.run = FakeRun(list(PROBED))
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": [ORPHANED]}),
                dump_json_str({"job": queue_job(status=left)}),
                dump_json_str({"claimed": None}),
            ]
        )
        _test_hooks.http_post = endpoint

        with caplog.at_level(logging.INFO):
            assert node_agent.main(node_argv(sourced_config)) == 0

        assert endpoint.tools == ["dispatch_list", "dispatch_report", "dispatch_claim"]
        assert endpoint.arguments[1]["action"] == "lost"
        assert "status" not in endpoint.arguments[1]
        assert endpoint.arguments[1]["detail"] == (
            f"DISPATCH_NOT_LAUNCHED: {DEFAULT_JOB_ID} was claimed on lavender, its start report "
            "never reached the queue, and nothing was launched, so the job is given back"
        )
        assert (
            f"launched nothing, so it is given back and the queue left it {left}; its host "
            "claim on lavender is discharged"
        ) in caplog.text

    def test_a_given_back_claim_stops_charging_its_host_in_the_tick_that_gives_it_back(
        self, sourced_config: pathlib.Path
    ) -> None:
        """MCPs board task 3f7d40fe. A tick killed mid-launch leaves the
        claim file its launch would have removed, as the hub runner's restart
        at 2026-10-16T07:10:08Z left six; until the collect removed it, the
        host was charged for it until its lease lapsed an hour later. The
        give-back of board task 0866f132 removes it in the same tick."""
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(sourced_config)})
        host_claims.record(
            loaded.host_claims,
            alias="lavender",
            claim=HostClaim(
                job_id=DEFAULT_JOB_ID,
                runner="fleet-node-lavender",
                project="libs/demo",
                workers=2,
                ram_gb=2.0,
                until_unix=DEMO_NOW + claim_window.CLAIM_LEASE_SECONDS,
            ),
        )
        _test_hooks.run = FakeRun(list(PROBED))
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": [ORPHANED]}),
                dump_json_str({"job": queue_job(status="queued")}),
                dump_json_str({"claimed": None}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert node_agent.main(node_argv(sourced_config)) == 0

        assert endpoint.arguments[1]["action"] == "lost"
        claims = host_claims.claims_directory(loaded.host_claims, alias="lavender")
        assert list(claims.iterdir()) == []
        assert host_claims.host_load(
            host_claims.live_claims(loaded.host_claims, alias="lavender")
        ) == LiveLoad(runs=0, workers=0, ram_gb=0.0)

    def test_a_given_back_claim_abandons_the_lease_it_took_and_the_lane_is_claimed_again(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """MCPs board task 31d1627d, A1. The claiming tick took the project's
        lease and died before it launched, as demure's did for job c1e06efc
        at 2026-10-30T23:39:57Z; the give-back abandons that lease, so the
        claim pass of the same tick asks for the lane it had excluded."""
        _plant_lease(sourced_config)
        _test_hooks.run = FakeRun(list(PROBED))
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": [ORPHANED]}),
                dump_json_str({"job": queue_job(status="queued")}),
                dump_json_str({"claimed": None}),
            ]
        )
        _test_hooks.http_post = endpoint

        with caplog.at_level(logging.INFO):
            assert node_agent.main(node_argv(sourced_config)) == 0

        assert endpoint.tools == ["dispatch_list", "dispatch_report", "dispatch_claim"]
        assert endpoint.arguments[2]["projects"] == ["libs/demo"]
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(sourced_config)})
        assert (
            leases.find_holder(
                loaded.leases, node="lavender", project="libs/demo", now_unix=DEMO_NOW
            )
            is None
        )
        refused = records.read_feed(loaded.feed)[-1]
        assert refused["kind"] is FeedKind.REFUSED
        assert refused["run_id"] == DEMO_RUN_ID
        assert refused["detail"] == endpoint.arguments[1]["detail"]
        assert f"its lease {DEMO_RUN_ID} is abandoned: lease released" in caplog.text

    @pytest.mark.parametrize(
        ("agent", "session_id", "acquired_unix"),
        [
            pytest.param("opus-fleet-w4-1003", ORPHANED_SESSION, DEMO_NOW, id="another-submitter"),
            pytest.param(ORPHANED_SUBMITTER, "s", DEMO_NOW, id="another-session"),
            pytest.param(
                ORPHANED_SUBMITTER,
                ORPHANED_SESSION,
                DEMO_NOW - claim_window.CLAIM_CLOCK_SLACK_SECONDS - 1,
                id="taken-before-the-claim",
            ),
        ],
    )
    def test_a_given_back_claim_leaves_a_lease_another_dispatch_took_held(
        self,
        sourced_config: pathlib.Path,
        caplog: pytest.LogCaptureFixture,
        agent: str,
        session_id: str,
        acquired_unix: int,
    ) -> None:
        """MCPs board task 31d1627d, A2. A lease carries no job id, so one the
        dead claim cannot have taken is another dispatch's environment."""
        planted = _plant_lease(
            sourced_config, agent=agent, session_id=session_id, acquired_unix=acquired_unix
        )
        _test_hooks.run = FakeRun(list(PROBED))
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": [ORPHANED]}),
                dump_json_str({"job": queue_job(status="queued")}),
                dump_json_str({"claimed": None}),
            ]
        )
        _test_hooks.http_post = endpoint

        with caplog.at_level(logging.INFO):
            assert node_agent.main(node_argv(sourced_config)) == 0

        assert endpoint.tools[:2] == ["dispatch_list", "dispatch_report"]
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(sourced_config)})
        assert leases.held_leases(loaded.leases, now_unix=DEMO_NOW) == (planted,)
        assert [event["kind"] for event in records.read_feed(loaded.feed)] == []
        assert (
            f"{planted['run_id']} holding libs/demo on lavender is not its claim's" in caplog.text
        )

    def test_a_given_back_claim_leaves_the_lease_of_a_run_still_going_held(
        self, sourced_config: pathlib.Path
    ) -> None:
        """MCPs board task 31d1627d, A2. A second job of the same submitter,
        session and project, claimed in the window of a run another job holds
        running, matches that run's lease on all three, and only the ledger
        says the lease is a launched run's."""
        launch(sourced_config)
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(sourced_config)})
        second = {**ORPHANED, "id": "22222222-bbbb-4bbb-8bbb-222222222222"}
        _test_hooks.http_post = FakeQueue([dump_json_str({"job": queue_job(status="queued")})])

        reconciled = node_collect.reconcile_claim(
            loaded,
            queue.load_credentials(),
            decode_job(second, answer=dump_json_str(second)),
            {},
            alias="lavender",
            running=frozenset({DEMO_RUN_ID}),
        )

        assert reconciled["adopted"] is None
        assert reconciled["line"].endswith(
            f"{DEMO_RUN_ID} holding libs/demo on lavender is not its claim's, so it stays held"
        )
        assert [
            lease["run_id"] for lease in leases.held_leases(loaded.leases, now_unix=DEMO_NOW)
        ] == [DEMO_RUN_ID]
        assert _ledger(sourced_config)[-1]["outcome"] == "running"

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
                DEMO_NOW + 2 * claim_window.CLAIM_CLOCK_SLACK_SECONDS,
                id="claimed-after-it-began",
            ),
            pytest.param(DEMO_NOW - claim_window.CLAIM_LEASE_SECONDS - 1, id="an-older-claim"),
            pytest.param(None, id="never-claimed"),
        ],
    )
    def test_never_stops_a_run_its_claim_cannot_have_launched(
        self, sourced_config: pathlib.Path, claimed_at: int | None
    ) -> None:
        """An old cancelled job of the same submitter, session and project is
        not the orphan's: the run began outside that claim's window, and the
        lost-run pass reads the same window off the job's trail."""
        launch(sourced_config)
        runner = FakeRun(list(PROBED))
        _test_hooks.run = runner
        instant = None if claimed_at is None else queue_instant(claimed_at)
        cancelled = {**ORPHANED, "status": "cancelled", "claimedAt": instant}
        claims = [] if claimed_at is None else [("fleet-node-lavender", claimed_at)]
        _test_hooks.http_post = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                listing_page([cancelled], None),
                listing_page([cancelled], None),
                trail_answer(cancelled, claims),
            ]
        )

        assert node_agent.main(node_argv(sourced_config)) == 0

        assert len(runner.calls) == len(PROBED)
        assert _ledger(sourced_config)[-1]["outcome"] == "running"
