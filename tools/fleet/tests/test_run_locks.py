"""A lock per run, and the stops that re-read the ledger under it (MCPs board task 8993c306).

:class:`fleet.cli.run_locks.RunLocks` holds one run back from a second
holder and lets every other run through, which is what lets a serving
runner settle two runs at once; a holder of one run takes it again on the
same thread, as the watch does when the settle it starts takes the lock it
holds. The two stops a collect pass makes, of a cancelled run and of a lost
one, run beside the watch's settles now, so each re-reads the ledger under
the run's lock and leaves alone a run settled meanwhile, sending the node
nothing.
"""

from __future__ import annotations

import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor

from platform_core.mcp_client import McpHttpResponse

from fleet.cli import _config, node_collect, node_lost
from fleet.cli.run_locks import RunLocks
from fleet.contracts.dispatch import decode_job
from fleet.contracts.ledger import LedgerEntry, LedgerOutcome
from fleet.core import _test_hooks, queue, records
from tests._node_agent_fixtures import _credentials_in_env, _sourced_config, launch
from tests._queue_fakes import FakeQueue, listing_page, queue_job, trail_answer
from tests._thread_fakes import await_event
from tests.conftest import DEMO_NOW, DEMO_RUN_ID, FakeRun

__all__ = ["_credentials_in_env", "_sourced_config"]

#: This node's runner, the one that launched the demo run.
LAVENDER = "fleet-node-lavender"


class TestRunLocks:
    def test_hold_one_run_back_and_let_another_through(self) -> None:
        locks = RunLocks()
        holding = threading.Event()
        release = threading.Event()
        order: list[str] = []

        def hold_first() -> None:
            with locks.holding("run-a"):
                holding.set()
                await_event(release, what="the case's go-ahead to release run-a")
                order.append("run-a released")

        def take_first_again() -> None:
            with locks.holding("run-a"):
                order.append("run-a taken again")

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(hold_first)
            await_event(holding, what="run-a to be held")
            with locks.holding("run-b"):
                order.append("run-b taken")
            second = pool.submit(take_first_again)
            release.set()
            first.result()
            second.result()

        assert order == ["run-b taken", "run-a released", "run-a taken again"]

    def test_a_holder_takes_its_run_again_on_the_same_thread(self) -> None:
        locks = RunLocks()
        with locks.holding("run-a"), locks.holding("run-a"):
            taken = True
        assert taken


def _settled_meanwhile(config_path: pathlib.Path) -> tuple[_config.LoadedWorkspace, LedgerEntry]:
    """The demo run's row, renamed for a run the ledger no longer calls live.

    Args:
        config_path: The workspace document, after a launch.

    Returns:
        The workspace and the row.
    """
    loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
    row = records.read_ledger(loaded.ledger)[-1].copy()
    row["run_id"] = "libs-demo-lavender-settled-meanwhile"
    return loaded, row


class SettledWhileAsked:
    """A queue asked while the watch settles the demo run beside the pass.

    Satisfies :class:`~platform_core.mcp_client.McpPostProtocol`. Before it
    answers it records the run finished in the ledger, as the watch's settle
    does, so the pass read the run as live and meets it settled.
    """

    def __init__(self, ledger: pathlib.Path, answers: FakeQueue) -> None:
        """Bind the ledger and the queue's answers.

        Args:
            ledger: The workspace's ledger file.
            answers: The queue's scripted answers.
        """
        self._ledger = ledger
        self._answers = answers

    def __call__(
        self, url: str, *, headers: dict[str, str], body: bytes, timeout_seconds: int
    ) -> McpHttpResponse:
        """Finish the run in the ledger, then answer.

        Args:
            url: Absolute URL posted to.
            headers: Every request header.
            body: The encoded JSON-RPC body.
            timeout_seconds: The caller's timeout.

        Returns:
            The queue's next scripted answer.
        """
        finished = records.read_ledger(self._ledger)[-1].copy()
        finished["outcome"] = LedgerOutcome.PASSED
        records.append_ledger(self._ledger, finished)
        return self._answers(url, headers=headers, body=body, timeout_seconds=timeout_seconds)


class TestAStopOfARunSettledMeanwhile:
    def test_counts_no_stop_when_the_cancelled_run_was_settled_while_the_queue_was_asked(
        self, sourced_config: pathlib.Path
    ) -> None:
        launch(sourced_config)
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(sourced_config)})
        node = FakeRun([])
        _test_hooks.run = node
        cancelled = queue_job(
            status="cancelled", node="lavender", runId=DEMO_RUN_ID, claimedBy=LAVENDER
        )
        _test_hooks.http_post = SettledWhileAsked(
            loaded.ledger, FakeQueue([listing_page([cancelled], None)])
        )

        stopped = node_collect.stop_cancelled(
            loaded, queue.load_credentials(), agent=LAVENDER, alias="lavender", held=frozenset()
        )

        assert stopped == 0
        assert node.calls == []

    def test_for_a_cancel_sends_the_node_nothing(self, sourced_config: pathlib.Path) -> None:
        launch(sourced_config)
        loaded, row = _settled_meanwhile(sourced_config)
        node = FakeRun([])
        _test_hooks.run = node
        cancelled = decode_job(queue_job(status="cancelled"), answer="dispatch_list")

        stopped = node_collect.stop_cancelled_run(
            loaded,
            node=loaded.workspace["nodes"]["lavender"],
            row=row,
            job=cancelled,
            agent=LAVENDER,
        )

        assert stopped is False
        assert node.calls == []

    def test_for_a_takeover_sends_the_node_nothing(self, sourced_config: pathlib.Path) -> None:
        launch(sourced_config)
        loaded, row = _settled_meanwhile(sourced_config)
        node = FakeRun([])
        _test_hooks.run = node
        taken_over = queue_job(status="passed", claimedBy="fleet-node-diphtheria")
        _test_hooks.http_post = FakeQueue(
            [
                listing_page([taken_over], None),
                trail_answer(taken_over, [(LAVENDER, DEMO_NOW)]),
            ]
        )

        stopped = node_lost.stop_lost(
            loaded,
            queue.load_credentials(),
            node=loaded.workspace["nodes"]["lavender"],
            rows=(row,),
            agent=LAVENDER,
        )

        assert stopped == 0
        assert node.calls == []
