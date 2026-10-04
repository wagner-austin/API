"""The watch that ends a node runner's tick (MCPs board task c1d48330).

Each case runs the real tick through :func:`fleet.cli.node_agent.main`
against a workspace declaring a 100-second watch, as ``fleet.json`` does,
with the node's ssh answers, the queue's answers and the clock faked: a run
that ends inside the window is closed on the poll that first sees its
result, a runner holding nothing makes no call past its passes, and a watch
whose runs keep going reads the node and writes nothing to the queue.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import (
    JSONTypeError,
    dump_json_str,
    load_json_str,
    narrow_json_to_str,
)

from fleet.cli import _config, node_agent, node_watch
from fleet.contracts.workspace import NODE_WATCH_CEILING_SECONDS, decode_fleet_workspace
from fleet.core import _test_hooks
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
from tests.conftest import (
    DEMO_NOW,
    DEMO_RUN_ID,
    FakeClock,
    FakeRun,
    FakeSleep,
    ok,
    retire_replies,
)

__all__ = ["_credentials_in_env", "_sourced_config"]

#: The window ``fleet.json`` declares.
WATCH_SECONDS = 100

#: When the watching tick starts: a minute after the launch, the run going.
TICK_START = DEMO_NOW + 60

#: When the node says the build wrote its result: 75 s into the tick, which
#: the watch's poll at 80 s is the first to see.
CHECK_ENDED = TICK_START + 75

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


class TestARunEndingInsideTheWindow:
    def test_is_closed_on_the_poll_that_first_sees_its_result(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """serendipity's osm-mcp row of 2026-10-04 ended 19:27:05Z and closed
        19:30:21Z, on the next tick; with the watch it closes on the first
        poll after it ends, inside ten seconds."""
        launch(sourced_config)
        sleep = _watching(sourced_config)
        ended = ok(f"0 {CHECK_ENDED}")
        replies = [
            *STILL_RUNNING,  # the tick's collect pass: still going
            *PROBED,  # its fill pass: the run's lease holds the one project
            *([*STILL_RUNNING] * 7),  # the watch's polls at 10 s to 70 s
            ok(""),  # the poll at 80 s: the collect script sent
            ended,  # and run, reading the result
            ok(""),  # the rerun collect pass reads it again
            ended,
            ok(""),  # the transcript's tail, sent
            ok(PASSING_TAIL),  # and run
            *retire_replies(),
            *PROBED,  # the rerun fill pass: room again, nothing queued
        ]
        node = FakeRun(replies)
        _test_hooks.run = node
        endpoint = FakeQueue(
            [
                held_answer(taskId=VERDICT_TASK),
                dump_json_str({"job": queue_job(status="running", node="lavender")}),
                held_answer(taskId=VERDICT_TASK),
                dump_json_str({"job": queue_job(status="passed")}),
                dump_json_str({"claimed": None}),
            ]
        )
        _test_hooks.http_post = endpoint

        with caplog.at_level("INFO"):
            assert node_agent.main(node_argv(sourced_config)) == 0

        assert sleep.slept == [node_watch.POLL_SECONDS] * 8
        closed_at = sleep.clock.seconds
        assert closed_at - CHECK_ENDED == 5
        assert endpoint.tools == [
            "dispatch_list",
            "dispatch_report",
            "dispatch_list",
            "dispatch_report",
            "dispatch_claim",
        ]
        assert endpoint.arguments[3]["action"] == "close"
        assert endpoint.arguments[3]["status"] == "passed"
        assert narrow_json_to_str(endpoint.arguments[3]["detail"]).endswith(f"run={DEMO_RUN_ID}")
        assert len(node.calls) == len(replies)
        messages = _messages(caplog)
        assert f"lavender: {DEMO_RUN_ID} has ended; collecting it now" in messages
        assert messages[-1] == (
            "lavender watch until 2025-09-04T15:36:00+00:00: 8 poll(s), 1 pass(es) rerun, "
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
        """Ten polls fit a 100-second window at ten seconds each; the last
        lands on the deadline, and none of them writes anywhere."""
        launch(sourced_config)
        sleep = _watching(sourced_config)
        replies = [*STILL_RUNNING, *PROBED, *([*STILL_RUNNING] * 10)]
        node = FakeRun(replies)
        _test_hooks.run = node
        endpoint = FakeQueue(
            [
                held_answer(),
                dump_json_str({"job": queue_job(status="running", node="lavender")}),
            ]
        )
        _test_hooks.http_post = endpoint

        with caplog.at_level("INFO"):
            assert node_agent.main(node_argv(sourced_config)) == 0

        assert sleep.slept == [node_watch.POLL_SECONDS] * 10
        assert sleep.clock.seconds == TICK_START + WATCH_SECONDS
        assert endpoint.tools == ["dispatch_list", "dispatch_report"]
        assert len(node.calls) == len(replies)
        assert _messages(caplog)[-1] == (
            "lavender watch until 2025-09-04T15:36:00+00:00: 10 poll(s), 0 pass(es) rerun, "
            "1 run(s) still watched"
        )


class TestTheWatchedSet:
    def test_a_run_this_machine_no_longer_calls_running_is_not_watched(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A run id the passes name that the ledger has closed, or never
        had, starts no watch and costs no ssh read."""
        _watching(sourced_config)
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(sourced_config)})
        node = loaded.workspace["nodes"]["lavender"]
        _test_hooks.run = FakeRun([])

        def passes() -> frozenset[str]:
            return frozenset({"a-run-the-ledger-never-had"})

        with caplog.at_level("INFO"):
            reruns = node_watch.run_tick(loaded, alias="lavender", node=node, passes=passes)

        assert reruns == 0
        assert _messages(caplog) == [NO_WATCH]


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
