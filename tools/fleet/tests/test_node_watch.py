"""The watch beside a node runner's tick (MCPs board task c1d48330).

The tick cases run the real tick through :func:`fleet.cli.node_agent.main`
against a workspace declaring a 100-second watch, as ``fleet.json`` does,
with the node's ssh answers, the queue's answers and the clock faked, each
thread answered by fakes of its own (:mod:`tests._thread_fakes`): a run that
ends while the fill pass is still going is closed by the watch before that
pass's claim is answered, a runner holding nothing makes no call past its
passes and never waits, and a watch whose run keeps going reads the node
until the window closes and writes nothing to the queue. The rest drive
:func:`fleet.cli.node_watch.run_tick` and its parts directly.
"""

from __future__ import annotations

import pathlib
import threading

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import (
    JSONTypeError,
    dump_json_str,
    load_json_str,
    narrow_json_to_str,
)

from fleet.cli import _config, node_agent, node_watch
from fleet.contracts.workspace import NODE_WATCH_CEILING_SECONDS, decode_fleet_workspace
from fleet.core import _test_hooks, queue
from tests._node_agent_fixtures import (
    NO_WATCH,
    NPM_CI,
    PASSING_TAIL,
    PROBED,
    VERDICT_TASK,
    _credentials_in_env,
    _sourced_config,
    held_answer,
    launch,
    node_argv,
    sourced_document,
)
from tests._queue_fakes import FakeQueue, queue_job
from tests._thread_fakes import QueueByThread, RunByThread
from tests.conftest import (
    DEMO_NOW,
    DEMO_RUN_ID,
    FakeClock,
    FakeRun,
    FakeSleep,
    failed,
    ok,
    retire_replies,
)

__all__ = ["_credentials_in_env", "_sourced_config"]

#: The window ``fleet.json`` declares.
WATCH_SECONDS = 100

#: When the watching tick starts: a minute after the launch, the run going.
TICK_START = DEMO_NOW + 60

#: When the node says the build wrote its result: 12 s into the tick, which
#: the watch's third poll, at 15 s, is the first to see.
CHECK_ENDED = TICK_START + 12

#: One read of a run's result that finds none: the collect script sent, then
#: run, printing nothing.
STILL_RUNNING = (ok(""), ok(""))


def _watching(config_path: pathlib.Path) -> FakeSleep:
    """Declare the watch in the workspace and put the tick's clock in place.

    Args:
        config_path: The workspace document.

    Returns:
        The sleep the watch waits through, its clock at :data:`TICK_START`.
    """
    document = sourced_document((NPM_CI,))
    document["node_watch_seconds"] = WATCH_SECONDS
    config_path.write_text(dump_json_str(document), encoding="utf-8")
    clock = FakeClock(TICK_START)
    _test_hooks.now = clock
    sleep = FakeSleep(clock)
    _test_hooks.sleep = sleep
    return sleep


def _messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Every message the tick logged.

    Args:
        caplog: The test's log capture.

    Returns:
        The messages, in order.
    """
    return [record.getMessage() for record in caplog.records]


def _loaded(config_path: pathlib.Path) -> _config.LoadedWorkspace:
    """The workspace as the runner loads it.

    Args:
        config_path: The workspace document.

    Returns:
        It, loaded.
    """
    return _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})


def _never_settles(run_id: str) -> str:
    """A settle a case asserts is never reached.

    Args:
        run_id: The run.

    Raises:
        AssertionError: Always.
    """
    raise AssertionError(f"settled {run_id}")


def _unsettling_watch(loaded: _config.LoadedWorkspace) -> node_watch.RunWatch:
    """Lavender's watch, with a settle the case asserts is never reached.

    Args:
        loaded: The workspace, its window declared.

    Returns:
        The watch, its window starting now.
    """
    node = loaded.workspace["nodes"]["lavender"]
    return node_watch.RunWatch(loaded, alias="lavender", node=node, settle=_never_settles)


class TestARunEndingWhileTheFillPassRuns:
    def test_is_closed_by_the_watch_while_the_fill_pass_waits_on_the_node(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """33fdd720 ended 04:06:14Z while diphtheria's passes ran and closed
        34 s later, once they were over; now the watch closes it while the
        fill pass is still going, on the first poll after it ends. The fill
        pass's first probe is held until then, so the same pass then finds
        the room the run freed and asks the lane for work."""
        launch(sourced_config)
        sleep = _watching(sourced_config)
        renewed = threading.Event()
        closed = threading.Event()
        ended = ok(f"0 {CHECK_ENDED}")
        main_replies = [*STILL_RUNNING, *PROBED]  # the collect pass's read, the fill's probe
        watch_replies = [
            *STILL_RUNNING,  # the poll at 5 s
            *STILL_RUNNING,  # the poll at 10 s
            ok(""),  # the poll at 15 s: the collect script sent
            ended,  # and run, reading the result
            ok(""),  # the settle reads it again
            ended,
            ok(""),  # the transcript's tail, sent
            ok(PASSING_TAIL),  # and run
            *retire_replies(),
        ]
        node = RunByThread(
            FakeRun(main_replies),
            FakeRun(watch_replies),
            watch_after=renewed,
            main_waits={len(STILL_RUNNING): closed},
        )
        _test_hooks.run = node
        endpoint = QueueByThread(
            FakeQueue(
                [
                    held_answer(taskId=VERDICT_TASK),
                    dump_json_str({"job": queue_job(status="running", node="lavender")}),
                    dump_json_str({"claimed": None}),
                ]
            ),
            FakeQueue(
                [
                    held_answer(taskId=VERDICT_TASK),
                    dump_json_str({"job": queue_job(status="passed")}),
                ]
            ),
            signals={"main:dispatch_report": renewed, "watch:dispatch_report": closed},
        )
        _test_hooks.http_post = endpoint

        with caplog.at_level("INFO"):
            assert node_agent.main(node_argv(sourced_config)) == 0

        assert sleep.slept == [node_watch.POLL_SECONDS] * 3
        assert sleep.clock.seconds - CHECK_ENDED == 3
        assert endpoint.order.index("watch:dispatch_report") < endpoint.order.index(
            "main:dispatch_claim"
        )
        assert endpoint.main.tools == ["dispatch_list", "dispatch_report", "dispatch_claim"]
        assert endpoint.watch.tools == ["dispatch_list", "dispatch_report"]
        assert endpoint.watch.arguments[1]["action"] == "close"
        assert endpoint.watch.arguments[1]["status"] == "passed"
        closing = narrow_json_to_str(endpoint.watch.arguments[1]["detail"])
        assert closing.endswith(f"run={DEMO_RUN_ID}")
        assert " ended=2025-09-04T15:34:32Z " in closing  # CHECK_ENDED, for closed_at to measure
        assert len(node.main.calls) == len(main_replies)
        assert len(node.watch.calls) == len(watch_replies)
        messages = _messages(caplog)
        assert f"lavender: {DEMO_RUN_ID} has ended; collecting it now" in messages
        assert messages[-1] == (
            "lavender watch until 2025-09-04T15:36:00+00:00: 3 poll(s), 1 run(s) closed, "
            "0 run(s) still watched"
        )


class TestARunnerHoldingNothing:
    def test_makes_no_call_past_its_passes_and_never_waits(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        sleep = _watching(sourced_config)
        node = FakeRun(PROBED)
        _test_hooks.run = node
        endpoint = FakeQueue([dump_json_str({"jobs": []}), dump_json_str({"claimed": None})])
        _test_hooks.http_post = endpoint

        with caplog.at_level("INFO"):
            assert node_agent.main(node_argv(sourced_config)) == 0

        assert sleep.slept == []
        assert len(node.calls) == len(PROBED)
        assert endpoint.tools == ["dispatch_list", "dispatch_claim"]
        assert _messages(caplog)[-1] == NO_WATCH


class TestARunStillGoingWhenTheWindowCloses:
    def test_is_read_every_poll_and_nothing_is_written_to_the_queue(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Twenty polls fit a 100-second window at five seconds each; the
        last lands on the deadline, none of them writes anywhere, and no
        pass runs again."""
        launch(sourced_config)
        sleep = _watching(sourced_config)
        renewed = threading.Event()
        main_replies = [*STILL_RUNNING, *PROBED]
        watch_replies = [*STILL_RUNNING] * 20
        node = RunByThread(
            FakeRun(main_replies), FakeRun(watch_replies), watch_after=renewed, main_waits={}
        )
        _test_hooks.run = node
        endpoint = QueueByThread(
            FakeQueue(
                [
                    held_answer(),
                    dump_json_str({"job": queue_job(status="running", node="lavender")}),
                ]
            ),
            FakeQueue([]),
            signals={"main:dispatch_report": renewed},
        )
        _test_hooks.http_post = endpoint

        with caplog.at_level("INFO"):
            assert node_agent.main(node_argv(sourced_config)) == 0

        assert sleep.slept == [node_watch.POLL_SECONDS] * 20
        assert sleep.clock.seconds == TICK_START + WATCH_SECONDS
        # The run's lease holds lavender's one project, so the fill pass asks for nothing.
        assert endpoint.main.tools == ["dispatch_list", "dispatch_report"]
        assert endpoint.watch.tools == []
        assert len(node.main.calls) == len(main_replies)
        assert len(node.watch.calls) == len(watch_replies)
        assert _messages(caplog)[-1] == (
            "lavender watch until 2025-09-04T15:36:00+00:00: 20 poll(s), 0 run(s) closed, "
            "1 run(s) still watched"
        )


class TestRunTick:
    def test_a_run_this_machine_does_not_call_running_is_not_watched(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A run id the passes hold that the ledger has closed, or never
        had, starts no watch and costs no ssh read."""
        sleep = _watching(sourced_config)
        loaded = _loaded(sourced_config)
        _test_hooks.run = FakeRun([])
        watch = _unsettling_watch(loaded)

        def passes() -> None:
            watch.hold(frozenset({"a-run-the-ledger-never-had"}))

        with caplog.at_level("INFO"):
            node_watch.run_tick(watch, alias="lavender", passes=passes)

        assert sleep.slept == []
        assert _messages(caplog) == [NO_WATCH]

    def test_passes_that_raise_end_the_watch_and_the_error_goes_on(
        self, sourced_config: pathlib.Path
    ) -> None:
        """An idle watch waits on its condition; the passes' exit wakes it,
        so the tick ends with their error instead of hanging."""
        _watching(sourced_config)
        loaded = _loaded(sourced_config)
        _test_hooks.run = FakeRun([])
        watch = _unsettling_watch(loaded)

        def passes() -> None:
            raise AppError(FleetErrorCode.QUEUE_ANSWER_MALFORMED, "the queue answered nonsense")

        with pytest.raises(AppError, match="the queue answered nonsense"):
            node_watch.run_tick(watch, alias="lavender", passes=passes)

    def test_a_poll_that_fails_is_raised_once_the_passes_are_over(
        self, sourced_config: pathlib.Path
    ) -> None:
        launch(sourced_config)
        sleep = _watching(sourced_config)
        loaded = _loaded(sourced_config)
        node = FakeRun([failed(255, "ssh: connect to host lavender port 22: timed out")])
        _test_hooks.run = node
        watch = _unsettling_watch(loaded)

        def passes() -> None:
            watch.hold(frozenset({DEMO_RUN_ID}))

        with pytest.raises(AppError):
            node_watch.run_tick(watch, alias="lavender", passes=passes)
        assert sleep.slept == [node_watch.POLL_SECONDS]
        assert len(node.calls) == 1


class TestCollectEnded:
    def test_a_run_no_running_job_names_is_left_for_the_next_tick(
        self, sourced_config: pathlib.Path
    ) -> None:
        """A run cancelled or taken over between its launch and its end is
        not this watch's to close; the next tick's collect pass stops it."""
        loaded = _loaded(sourced_config)
        credentials = queue.load_credentials()
        another = queue_job(status="running", node="lavender", runId="another-run")
        endpoint = FakeQueue([dump_json_str({"jobs": [another]})])
        _test_hooks.http_post = endpoint

        line = node_watch.collect_ended(
            loaded, credentials, credentials, {}, agent="fleet-node-lavender", run_id=DEMO_RUN_ID
        )

        assert line == (
            f"{DEMO_RUN_ID}: no running job of fleet-node-lavender names it now; "
            "the next tick's collect reads it"
        )
        assert endpoint.tools == ["dispatch_list"]


class TestTheDeclaredWindow:
    def test_fleet_json_declares_one_inside_the_ceiling(self) -> None:
        document = pathlib.Path(__file__).parent.parent / "fleet.json"
        workspace = decode_fleet_workspace(load_json_str(document.read_text(encoding="utf-8")))
        assert workspace["node_watch_seconds"] == WATCH_SECONDS
        assert WATCH_SECONDS <= NODE_WATCH_CEILING_SECONDS

    @pytest.mark.parametrize("seconds", [-1, NODE_WATCH_CEILING_SECONDS + 1])
    def test_a_window_outside_its_bounds_is_refused(self, seconds: int) -> None:
        document = sourced_document((NPM_CI,))
        document["node_watch_seconds"] = seconds
        with pytest.raises(JSONTypeError, match=f"node_watch_seconds is {seconds}"):
            decode_fleet_workspace(document)

    def test_a_workspace_without_one_is_refused(self) -> None:
        document = sourced_document((NPM_CI,))
        del document["node_watch_seconds"]
        with pytest.raises(JSONTypeError, match="node_watch_seconds"):
            decode_fleet_workspace(document)
