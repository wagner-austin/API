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
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import dump_json_str, narrow_json_to_str

from fleet.cli import _config, node_watch
from fleet.core import _test_hooks, queue
from tests._node_agent_fixtures import (
    NPM_CI,
    PASSING_TAIL,
    VERDICT_TASK,
    _credentials_in_env,
    _sourced_config,
    held_answer,
    launch,
    sourced_document,
)
from tests._queue_fakes import FakeQueue, queue_job
from tests._thread_fakes import await_event
from tests.conftest import DEMO_NOW, DEMO_RUN_ID, FakeRun, ok, retire_replies

__all__ = ["_credentials_in_env", "_sourced_config"]

#: One read of a run's result that finds none: the collect script sent, then
#: run, printing nothing.
STILL_RUNNING = (ok(""), ok(""))

#: The same read once the build has written its result.
ENDED = (ok(""), ok(f"0 {DEMO_NOW + 72}"))


def _poll_every_second(config_path: pathlib.Path) -> _config.LoadedWorkspace:
    """Declare a one-second poll and load the workspace as the runner does.

    Args:
        config_path: The workspace document.

    Returns:
        It, loaded.
    """
    document = sourced_document((NPM_CI,))
    document["node_poll_seconds"] = 1
    config_path.write_text(dump_json_str(document), encoding="utf-8")
    return _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})


def _watch(loaded: _config.LoadedWorkspace, settle: node_watch.Settle) -> node_watch.RunWatch:
    """Lavender's watch.

    Args:
        loaded: The workspace.
        settle: What it settles an ended run with.

    Returns:
        The watch, holding nothing.
    """
    node = loaded.workspace["nodes"]["lavender"]
    return node_watch.RunWatch(loaded, alias="lavender", node=node, settle=settle)


def _never_settles(*, run_id: str) -> str:
    """A settle a case asserts is never reached.

    Args:
        run_id: The run.

    Raises:
        AssertionError: Always.
    """
    raise AssertionError(f"settled {run_id}")


class AnsweringThen:
    """A node that answers from a script and does something once a given call is answered.

    Satisfies :class:`~fleet.core._test_hooks.RunProtocol`.

    Attributes:
        runner: The scripted node, whose calls a case reads.
    """

    runner: FakeRun

    def __init__(
        self,
        replies: Sequence[_test_hooks.CommandResult],
        *,
        after: int,
        then: Callable[[], bool],
    ) -> None:
        """Bind the script and the action.

        Args:
            replies: One result per expected call.
            after: The index, from zero, of the call the action follows.
            then: The action.
        """
        self.runner = FakeRun(replies)
        self._after = after
        self._then = then

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: int,
        stdin_bytes: bytes | None = None,
        unset_env: Sequence[str] = (),
        set_env: Sequence[tuple[str, str]] = (),
    ) -> _test_hooks.CommandResult:
        """Answer the call, then act if it is the one named.

        Args:
            argv: The command.
            timeout_seconds: The deadline the caller chose.
            stdin_bytes: Its standard input, or None.
            unset_env: The variables the caller withheld from the child.
            set_env: The variables the caller set in the child.

        Returns:
            The scripted result.
        """
        index = len(self.runner.calls)
        answer = self.runner(
            argv,
            timeout_seconds=timeout_seconds,
            stdin_bytes=stdin_bytes,
            unset_env=unset_env,
            set_env=set_env,
        )
        if index == self._after:
            self._then()
        return answer


class TestARunThatEnds:
    def test_is_settled_on_the_first_poll_after_it_ends_and_no_handover_cuts_it_short(
        self, sourced_config: pathlib.Path
    ) -> None:
        """The run is still going at the first poll and has ended at the
        second; a handover asked for while its settle is under way is
        refused, and the watch then closes holding nothing."""
        launch(sourced_config)
        loaded = _poll_every_second(sourced_config)
        node = FakeRun([*STILL_RUNNING, *ENDED])
        _test_hooks.run = node
        settling = threading.Event()
        release = threading.Event()
        settled: list[str] = []

        def settle(*, run_id: str) -> str:
            settling.set()
            await_event(release, what="the case's go-ahead to finish the settle")
            settled.append(run_id)
            return f"{run_id}: settled"

        watch = _watch(loaded, settle)
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
        loaded = _poll_every_second(sourced_config)
        watch = _watch(loaded, _never_settles)
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
        loaded = _poll_every_second(sourced_config)
        node = FakeRun([])
        _test_hooks.run = node
        watch = _watch(loaded, _never_settles)

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
        loaded = _poll_every_second(sourced_config)
        node = FakeRun([])
        _test_hooks.run = node
        watch = _watch(loaded, _never_settles)

        assert watch.ended("a-run-the-ledger-closed") is False
        assert node.calls == []

    def test_is_read_while_it_is_live(self, sourced_config: pathlib.Path) -> None:
        launch(sourced_config)
        loaded = _poll_every_second(sourced_config)
        node = FakeRun([*STILL_RUNNING, *ENDED])
        _test_hooks.run = node
        watch = _watch(loaded, _never_settles)

        assert watch.ended(DEMO_RUN_ID) is False
        assert watch.ended(DEMO_RUN_ID) is True
        assert len(node.calls) == len(STILL_RUNNING) + len(ENDED)


class TestASettleThatRaises:
    def test_leaves_the_watch_closable_and_the_error_goes_on(
        self, sourced_config: pathlib.Path
    ) -> None:
        launch(sourced_config)
        loaded = _poll_every_second(sourced_config)
        _test_hooks.run = FakeRun(list(ENDED))

        def settle(*, run_id: str) -> str:
            raise AppError(FleetErrorCode.QUEUE_ANSWER_MALFORMED, f"{run_id}: nonsense")

        watch = _watch(loaded, settle)
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

        line = node_watch.collect_ended(
            loaded, credentials, credentials, {}, agent="fleet-node-lavender", run_id=DEMO_RUN_ID
        )

        assert line.endswith(f"run={DEMO_RUN_ID}")
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

        line = node_watch.collect_ended(
            loaded, credentials, credentials, {}, agent="fleet-node-lavender", run_id=DEMO_RUN_ID
        )

        assert line == (
            f"{DEMO_RUN_ID}: no running job of fleet-node-lavender names it now; "
            "the next fire's collect reads it"
        )
        assert endpoint.tools == ["dispatch_list"]
