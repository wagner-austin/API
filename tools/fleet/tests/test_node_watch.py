"""The watch beside a serving node runner (MCPs board tasks c1d48330 and 8993c306).

Each case runs the real :class:`fleet.cli.node_watch.RunWatch` on a thread
of its own against a launched run, with the node's ssh answers and the
settle faked: a run that ends is settled on the first poll after it ends,
and a handover asked for while it settles is refused until the settle is
over; a watch closed before a run arrived, or between finding a run ended
and settling it, settles nothing; a settle that raises leaves the watch
closable and the error goes on. The polls wait on the watch's condition for
the workspace's ``node_poll_seconds``, so each case declares a one-second
poll and lasts about a second per poll.
"""

from __future__ import annotations

import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import dump_json_str, narrow_json_to_str

from fleet.cli import _config, node_watch
from fleet.cli.node_collected import Collected, CollectOutcome
from fleet.core import _test_hooks, queue
from tests._node_agent_fixtures import (
    PASSING_TAIL,
    VERDICT_TASK,
    _credentials_in_env,
    _sourced_config,
    held_answer,
    launch,
)
from tests._queue_fakes import FakeQueue, queue_job
from tests._thread_fakes import await_event
from tests._watch_fixtures import (
    ENDED,
    STILL_RUNNING,
    AnsweringThen,
    collected,
    lavender_watch,
    never_settles,
    poll_every_second,
)
from tests.conftest import DEMO_RUN_ID, FakeRun, failed, ok, retire_replies

__all__ = ["_credentials_in_env", "_sourced_config"]


class TestARunThatEnds:
    def test_is_settled_on_the_first_poll_after_it_ends_and_no_handover_cuts_it_short(
        self, sourced_config: pathlib.Path
    ) -> None:
        """The run is still going at the first poll and has ended at the
        second; a handover asked for while its settle is under way is
        refused, and the watch then closes holding nothing."""
        launch(sourced_config)
        loaded = poll_every_second(sourced_config)
        node = FakeRun([*STILL_RUNNING, *ENDED])
        _test_hooks.run = node
        settling = threading.Event()
        release = threading.Event()
        settled: list[str] = []

        def settle(*, run_id: str) -> Collected:
            settling.set()
            await_event(release, what="the case's go-ahead to finish the settle")
            settled.append(run_id)
            return collected(run_id, CollectOutcome.SETTLED)

        watch = lavender_watch(loaded, settle)
        with ThreadPoolExecutor(max_workers=1) as pool:
            watching = pool.submit(watch.watch)
            with watch:
                watch.hold(frozenset({DEMO_RUN_ID}))
                await_event(settling, what="the settle to begin")
                refused = watch.close_if_idle()
                release.set()
            watching.result()

        assert refused is False
        assert settled == [DEMO_RUN_ID]
        assert watch.polls == 2
        assert watch.closed() == 1
        assert len(node.calls) == len(STILL_RUNNING) + len(ENDED)

    def test_found_ended_after_the_handover_is_left_for_the_next_start(
        self, sourced_config: pathlib.Path
    ) -> None:
        """The handover lands between reading the result and settling: the
        watch settles nothing, and the next start's collect pass reads it."""
        launch(sourced_config)
        loaded = poll_every_second(sourced_config)
        watch = lavender_watch(loaded, never_settles)
        node = AnsweringThen(list(ENDED), after=1, then=watch.close_if_idle)
        _test_hooks.run = node

        with ThreadPoolExecutor(max_workers=1) as pool:
            watching = pool.submit(watch.watch)
            watch.hold(frozenset({DEMO_RUN_ID}))
            watching.result()

        assert watch.polls == 1
        assert watch.closed() == 0
        assert watch.still_watched() == 1
        assert len(node.runner.calls) == len(ENDED)


class TestAWatchHoldingNothing:
    def test_reads_nothing_and_ends_when_it_is_closed(self, sourced_config: pathlib.Path) -> None:
        loaded = poll_every_second(sourced_config)
        node = FakeRun([])
        _test_hooks.run = node
        watch = lavender_watch(loaded, never_settles)

        with ThreadPoolExecutor(max_workers=1) as pool:
            watching = pool.submit(watch.watch)
            with watch:
                watch.hold(frozenset({"a-run-the-ledger-never-had"}))
            watching.result()

        assert watch.polls == 0
        assert watch.still_watched() == 0
        assert node.calls == []
        assert watch.close_if_idle() is True


class TestARunTheLedgerNoLongerCallsRunning:
    def test_is_not_read_off_the_node(self, sourced_config: pathlib.Path) -> None:
        """A run the collect pass settled or stopped while the watch held
        it: the read would send its script into the directory the retire
        removes, as at 07:00Z on 2026-10-05 on lavender-wsl, so it is not
        made (MCPs board task 8993c306)."""
        loaded = poll_every_second(sourced_config)
        node = FakeRun([])
        _test_hooks.run = node
        watch = lavender_watch(loaded, never_settles)

        assert watch.read("a-run-the-ledger-closed") is node_watch.RunRead.GONE
        assert node.calls == []

    def test_is_read_while_it_is_live(self, sourced_config: pathlib.Path) -> None:
        launch(sourced_config)
        loaded = poll_every_second(sourced_config)
        node = FakeRun([*STILL_RUNNING, *ENDED])
        _test_hooks.run = node
        watch = lavender_watch(loaded, never_settles)

        assert watch.read(DEMO_RUN_ID) is node_watch.RunRead.RUNNING
        assert watch.read(DEMO_RUN_ID) is node_watch.RunRead.ENDED
        assert len(node.calls) == len(STILL_RUNNING) + len(ENDED)


class TestANodeThatMissesARead:
    def test_is_read_again_at_the_next_poll_and_the_run_settled_then(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """At 09:26:24Z on 2026-10-05 lavender-wsl's sshd timed out a read's
        banner exchange with its host down to 2.6 GB free, and the watch's
        error ended the serve; now the read is retried at the next poll."""
        launch(sourced_config)
        loaded = poll_every_second(sourced_config)
        node = FakeRun([failed(255, "Connection timed out during banner exchange"), *ENDED])
        _test_hooks.run = node
        settled: list[str] = []
        settling = threading.Event()

        def settle(*, run_id: str) -> Collected:
            settled.append(run_id)
            settling.set()
            return collected(run_id, CollectOutcome.SETTLED)

        watch = lavender_watch(loaded, settle)
        with caplog.at_level("INFO"), ThreadPoolExecutor(max_workers=1) as pool:
            watching = pool.submit(watch.watch)
            with watch:
                watch.hold(frozenset({DEMO_RUN_ID}))
                await_event(settling, what="the run to be settled")
            watching.result()

        assert settled == [DEMO_RUN_ID]
        assert watch.closed() == 1
        assert watch.polls == 2
        assert len(node.calls) == 1 + len(ENDED)
        assert any(
            record.getMessage().startswith(
                f"lavender did not answer the read of {DEMO_RUN_ID}; it is read again at the "
                "next poll: ssh to lavender failed while sending"
            )
            for record in caplog.records
        )


class TestASettleTheQueueDidNotAnswer:
    def test_keeps_the_run_held_and_settles_it_at_the_next_poll(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """At 11:55:31Z on 2026-10-05 the settle of diphtheria's run met the
        refusal a deploy's recreate of mcp-fleet answered every call with,
        and its error ended the watch; now the run is settled again at the
        next poll (MCPs board task 8993c306)."""
        launch(sourced_config)
        loaded = poll_every_second(sourced_config)
        node = FakeRun([*ENDED, *ENDED])
        _test_hooks.run = node
        attempts: list[str] = []
        settling = threading.Event()
        refused = "http://127.0.0.1:8035/mcp did not answer: URLError: refused"

        def settle(*, run_id: str) -> Collected:
            attempts.append(run_id)
            if len(attempts) == 1:
                raise AppError(code=FleetErrorCode.QUEUE_UNANSWERED, message=refused)
            settling.set()
            return collected(run_id, CollectOutcome.SETTLED)

        watch = lavender_watch(loaded, settle)
        with caplog.at_level("INFO"), ThreadPoolExecutor(max_workers=1) as pool:
            watching = pool.submit(watch.watch)
            with watch:
                watch.hold(frozenset({DEMO_RUN_ID}))
                await_event(settling, what="the run to be settled")
            watching.result()

        assert attempts == [DEMO_RUN_ID, DEMO_RUN_ID]
        assert watch.closed() == 1
        assert watch.polls == 2
        assert len(node.calls) == 2 * len(ENDED)
        assert (
            f"lavender: the queue did not answer the settle of {DEMO_RUN_ID}; it is settled "
            f"again at the next poll: {refused}"
        ) in [record.getMessage() for record in caplog.records]


class TestASettleThatRaises:
    def test_leaves_the_watch_closable_and_the_error_goes_on(
        self, sourced_config: pathlib.Path
    ) -> None:
        launch(sourced_config)
        loaded = poll_every_second(sourced_config)
        _test_hooks.run = FakeRun(list(ENDED))

        def settle(*, run_id: str) -> Collected:
            raise AppError(FleetErrorCode.QUEUE_ANSWER_MALFORMED, f"{run_id}: nonsense")

        watch = lavender_watch(loaded, settle)
        with ThreadPoolExecutor(max_workers=1) as pool:
            watching = pool.submit(watch.watch)
            watch.hold(frozenset({DEMO_RUN_ID}))
            with pytest.raises(AppError, match="nonsense"):
                watching.result()

        assert watch.close_if_idle() is True
        assert watch.closed() == 0


class TestCollectEnded:
    def test_a_run_its_runner_still_holds_running_is_settled_and_closed(
        self, sourced_config: pathlib.Path
    ) -> None:
        launch(sourced_config)
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(sourced_config)})
        credentials = queue.load_credentials()
        _test_hooks.run = FakeRun([*ENDED, ok(""), ok(PASSING_TAIL), *retire_replies()])
        endpoint = FakeQueue(
            [held_answer(taskId=VERDICT_TASK), dump_json_str({"job": queue_job(status="passed")})]
        )
        _test_hooks.http_post = endpoint

        collected = node_watch.collect_ended(
            loaded, credentials, credentials, {}, agent="fleet-node-lavender", run_id=DEMO_RUN_ID
        )

        assert collected["outcome"] is CollectOutcome.SETTLED
        assert collected["line"].endswith(f"run={DEMO_RUN_ID}")
        assert endpoint.tools == ["dispatch_list", "dispatch_report"]
        assert narrow_json_to_str(endpoint.arguments[1]["action"]) == "close"

    def test_a_run_no_running_job_names_is_left_for_the_next_start(
        self, sourced_config: pathlib.Path
    ) -> None:
        """A run cancelled or taken over between its launch and its end is
        not this watch's to close; the next fire's collect pass stops it."""
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(sourced_config)})
        credentials = queue.load_credentials()
        another = queue_job(status="running", node="lavender", runId="another-run")
        endpoint = FakeQueue([dump_json_str({"jobs": [another]})])
        _test_hooks.http_post = endpoint

        collected = node_watch.collect_ended(
            loaded, credentials, credentials, {}, agent="fleet-node-lavender", run_id=DEMO_RUN_ID
        )

        assert collected == Collected(
            outcome=CollectOutcome.NOT_HELD,
            line=(
                f"{DEMO_RUN_ID}: no running job of fleet-node-lavender names it now; "
                "the next fire's collect reads it"
            ),
        )
        assert endpoint.tools == ["dispatch_list"]
