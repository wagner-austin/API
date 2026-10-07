"""What one collect did, and what the collect pass hands the watch (MCPs board task c1d48330).

Measured between 2026-10-06T20:16Z and 2026-10-07T03:48Z on the rolled
runner: nine lease renewals came under 60 s after the one before, from
collect passes rerun while the queue recovered; lavender-wsl's tail read at
02:54:10Z raised NODE_UNREACHABLE and ended the serve; and serendipity's
adopted run 220c2a9e went unwatched until the next fire. Each case runs the
real collect against the fake queue and the fake ssh runner, on the demo run
lavender launched.
"""

from __future__ import annotations

import pathlib

from platform_core.json_utils import dump_json_str, narrow_json_to_str

from fleet.cli import _config, node_collect
from fleet.cli.node_collected import RENEW_AFTER_SECONDS, Collected, CollectOutcome, lease_age
from fleet.contracts.dispatch import decode_job, decode_listing
from fleet.core import _test_hooks, claim_window, queue
from tests._holder_fakes import RecordingHolder
from tests._node_agent_fixtures import (
    VERDICT_TASK,
    _credentials_in_env,
    _sourced_config,
    launch,
)
from tests._queue_fakes import FakeQueue, queue_instant, queue_job
from tests.conftest import DEMO_NOW, DEMO_RUN_ID, FakeRun, failed, ok, pin_clock

__all__ = ["_credentials_in_env", "_sourced_config"]

#: This node's runner.
LAVENDER = "fleet-node-lavender"

#: One read of the result that finds none: the collect script sent, then run.
STILL_RUNNING = (ok(""), ok(""))

#: What ssh answers a node that stopped answering, as lavender-wsl's did.
TIMED_OUT = failed(255, "Connection timed out during banner exchange")


def _running_job(lease_set: int | None, *, now: int) -> str:
    """The queue's listing of the demo run's job, running under lavender.

    Args:
        lease_set: How long ago its lease was set, or None for no lease.
        now: Whole seconds since the epoch.

    Returns:
        The rendered ``dispatch_list`` answer.
    """
    expires = (
        None
        if lease_set is None
        else queue_instant(now + claim_window.CLAIM_LEASE_SECONDS - lease_set)
    )
    return dump_json_str(
        {
            "jobs": [
                queue_job(
                    status="running",
                    node="lavender",
                    runId=DEMO_RUN_ID,
                    claimedBy=LAVENDER,
                    taskId=VERDICT_TASK,
                    leaseExpiresAt=expires,
                )
            ]
        }
    )


def _collect(
    config_path: pathlib.Path, listing: str, node: list[_test_hooks.CommandResult]
) -> tuple[Collected, FakeQueue]:
    """Collect the demo job once, as the listing has it.

    Args:
        config_path: The workspace document.
        listing: The queue's listing holding the job.
        node: The node's answers.

    Returns:
        What the collect did, and the queue for its calls.
    """
    loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
    _test_hooks.run = FakeRun(node)
    endpoint = FakeQueue([dump_json_str({"job": queue_job(status="running")})])
    _test_hooks.http_post = endpoint
    job = decode_listing(listing)[0]
    collected = node_collect.collect_one_job(
        loaded, queue.load_credentials(), queue.load_credentials(), job, {}
    )
    return collected, endpoint


class TestARunStillGoing:
    def test_whose_lease_was_set_under_a_minute_ago_writes_nothing(
        self, sourced_config: pathlib.Path
    ) -> None:
        launch(sourced_config)
        now = _test_hooks.now()
        pin_clock(now)

        collected, endpoint = _collect(
            sourced_config, _running_job(RENEW_AFTER_SECONDS - 1, now=now), list(STILL_RUNNING)
        )

        assert collected["outcome"] is CollectOutcome.RUNNING
        assert collected["line"].endswith(
            f"run={DEMO_RUN_ID}: still running, lease set {RENEW_AFTER_SECONDS - 1} s ago"
        )
        assert endpoint.tools == []

    def test_whose_lease_was_set_a_minute_ago_is_renewed(
        self, sourced_config: pathlib.Path
    ) -> None:
        launch(sourced_config)
        now = _test_hooks.now()
        pin_clock(now)

        collected, endpoint = _collect(
            sourced_config, _running_job(RENEW_AFTER_SECONDS, now=now), list(STILL_RUNNING)
        )

        assert collected["outcome"] is CollectOutcome.RENEWED
        assert endpoint.tools == ["dispatch_report"]
        assert narrow_json_to_str(endpoint.arguments[0]["action"]) == "progress"

    def test_carrying_no_lease_is_renewed(self, sourced_config: pathlib.Path) -> None:
        launch(sourced_config)
        now = _test_hooks.now()
        pin_clock(now)

        collected, endpoint = _collect(
            sourced_config, _running_job(None, now=now), list(STILL_RUNNING)
        )

        assert collected["outcome"] is CollectOutcome.RENEWED
        assert endpoint.tools == ["dispatch_report"]


class TestLeaseAge:
    def test_is_the_lease_length_less_what_remains_and_none_without_a_lease(self) -> None:
        now = 1_791_341_600
        running = decode_listing(_running_job(42, now=now))[0]
        unleased = decode_job(queue_job(status="running"), answer="")

        assert lease_age(running, now=now) == 42
        assert lease_age(unleased, now=now) is None


class TestANodeThatMissesTheTailRead:
    def test_changes_nothing_and_answers_unreachable(self, sourced_config: pathlib.Path) -> None:
        """Row 1d5d0a1a on lavender-wsl, 2026-10-07: its result was read at
        02:54:10Z, the tail read then timed out, and the serve ended."""
        launch(sourced_config)
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(sourced_config)})
        ledger = loaded.ledger.read_text(encoding="utf-8")

        collected, endpoint = _collect(
            sourced_config, _running_job(None, now=0), [ok(""), ok("0 1757000060"), TIMED_OUT]
        )

        assert collected["outcome"] is CollectOutcome.UNREACHABLE
        assert collected["line"].endswith(
            f"run={DEMO_RUN_ID}: did not answer the read of its transcript's tail: ssh to "
            "lavender failed while sending "
            f"C:/fleet/stage/{DEMO_RUN_ID}/log-tail.ps1: Connection timed out during banner "
            "exchange"
        )
        assert endpoint.tools == []
        assert loaded.ledger.read_text(encoding="utf-8") == ledger


class TestTheCollectPass:
    def test_owes_the_watch_a_renewal_its_node_did_not_answer(
        self, sourced_config: pathlib.Path
    ) -> None:
        """374f0656 on lavender-wsl, 2026-10-07: the 03:00Z pass could not read
        it, and its lease went unrenewed from 02:57:56Z to 03:02:54Z."""
        launch(sourced_config)
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(sourced_config)})
        _test_hooks.run = FakeRun([TIMED_OUT])
        endpoint = FakeQueue([_running_job(None, now=0)])
        _test_hooks.http_post = endpoint
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

        assert holder.held == [frozenset({DEMO_RUN_ID})]
        assert holder.owed == [frozenset({DEMO_RUN_ID})]
        assert endpoint.tools == ["dispatch_list"]

    def test_hands_the_watch_the_run_it_adopts(self, sourced_config: pathlib.Path) -> None:
        """Row 220c2a9e on serendipity, 2026-10-07: adopted at 01:52:25Z,
        unwatched, ended 01:53:06Z and closed only at the 01:54Z fire."""
        launch(sourced_config)
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(sourced_config)})
        orphaned = queue_job(
            status="claimed",
            claimedBy=LAVENDER,
            claimedAt=queue_instant(DEMO_NOW),
            taskId=VERDICT_TASK,
        )
        started = dump_json_str(
            {"job": queue_job(status="running", node="lavender", runId=DEMO_RUN_ID)}
        )
        _test_hooks.run = FakeRun([])
        endpoint = FakeQueue([dump_json_str({"jobs": [orphaned]}), started, started])
        _test_hooks.http_post = endpoint
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

        assert holder.held == [frozenset(), frozenset({DEMO_RUN_ID})]
        assert holder.owed == []
        # The adopted run is the pass's own: no cancelled listing, no lost-run ask.
        assert endpoint.tools == ["dispatch_list", "dispatch_report", "dispatch_report"]
