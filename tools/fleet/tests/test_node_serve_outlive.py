"""A serve whose loop fails keeps its watch to the handover (MCPs board task 8993c306).

At 07:24:56Z on 2026-10-05 the dispatch endpoint refused one queue listing,
every serving runner ended with it, and lavender-wsl's run that ended 16 s
later closed at the next start, 120.9 s after its check (row 62734702). The
:func:`fleet.cli.node_serve.serve_on` cases drive the wait with a pinned
clock and the scripted watch of ``test_node_serve.py``: it closes the watch
10 s before the first fire boundary after the failure, later while a settle
is under way, at once inside those last 10 s, and stops waiting the moment
the watch thread has ended. The end-to-end case runs
:func:`fleet.cli.node_serve.serve` with the real watch on a launched run
whose loop's first queue listing fails: the run that ends afterwards is
settled, and only then does the failure end the serve. A listing the queue
did not answer at all no longer fails the loop
(``test_node_serve_claim.py``), so the failure here is an answer the
runner cannot read.
"""

from __future__ import annotations

import pathlib
import threading
from concurrent.futures import Future

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import dump_json_str

from fleet.cli import _config, node_serve, node_watch
from fleet.core import _test_hooks
from tests._node_agent_fixtures import (
    NPM_CI,
    _credentials_in_env,
    _sourced_config,
    launch,
    sourced_document,
)
from tests._thread_fakes import await_event
from tests.conftest import DEMO_NOW, DEMO_RUN_ID, FakeClock, FakeRun, ok, pin_clock
from tests.test_node_serve import FIRST_FIRE, ScriptedWatch

__all__ = ["_credentials_in_env", "_sourced_config"]

#: When :func:`fleet.cli.node_serve.serve_on` closes the watch after a
#: failure at DEMO_NOW, 20 s past a fire boundary.
HANDOVER = FIRST_FIRE - node_serve.HANDOVER_SECONDS

#: What the scripted listing's answer fails with.
MALFORMED = "dispatch_list answered a job with no id"


def _still_watching() -> Future[None]:
    """A watch thread that has not ended.

    Returns:
        A future nothing has completed.
    """
    return Future()


class TestServingOnAfterTheLoopFailed:
    def test_closes_the_watch_ten_seconds_before_the_next_fire(self) -> None:
        pin_clock(DEMO_NOW)
        watch = ScriptedWatch([True])

        handed_over = node_serve.serve_on(watch, _still_watching(), poll_seconds=5)

        assert handed_over == HANDOVER
        assert watch.asked == 1

    def test_waits_a_poll_while_a_settle_is_under_way(self) -> None:
        pin_clock(DEMO_NOW)
        watch = ScriptedWatch([False, True])

        handed_over = node_serve.serve_on(watch, _still_watching(), poll_seconds=5)

        assert handed_over == HANDOVER + 5
        assert watch.asked == 2

    def test_hands_over_at_once_inside_the_last_seconds_before_a_fire(self) -> None:
        pin_clock(FIRST_FIRE - 3)
        watch = ScriptedWatch([True])

        handed_over = node_serve.serve_on(watch, _still_watching(), poll_seconds=5)

        assert handed_over == FIRST_FIRE - 3
        assert watch.asked == 1

    def test_stops_waiting_once_the_watch_thread_has_ended(self) -> None:
        clock = pin_clock(DEMO_NOW)
        ended: Future[None] = Future()
        ended.set_result(None)
        watch = ScriptedWatch([])

        handed_over = node_serve.serve_on(watch, ended, poll_seconds=5)

        assert handed_over == DEMO_NOW
        assert watch.asked == 0
        assert clock.seconds == DEMO_NOW


class CountingWatch(node_watch.RunWatch):
    """The real watch, which also says when it has counted a settled run.

    The settle's own callable returns before the watch marks the settle over
    and counts the run, so an event set inside it would let the main thread
    ask for the handover in between, refused or granted by the scheduler's
    whim (fleet job ed284162 on loki read 0 runs closed). The count is the
    watch's last step for a settled run, so the event set after it marks the
    watch idle with the run counted.

    Attributes:
        counted: Set once a settled run is counted.
    """

    counted: threading.Event

    def __init__(self, loaded: _config.LoadedWorkspace, settle: node_watch.Settle) -> None:
        """Bind lavender's watch.

        Args:
            loaded: The workspace.
            settle: What it settles an ended run with.
        """
        node = loaded.workspace["nodes"]["lavender"]
        super().__init__(loaded, alias="lavender", node=node, settle=settle)
        self.counted = threading.Event()

    def _settled(self, run_id: str) -> None:
        """Count the run as the watch does, then say so.

        Args:
            run_id: The run.
        """
        super()._settled(run_id)
        self.counted.set()


class SleepAfterTheSettle:
    """A sleep that moves the clock, the main thread's only once a run is counted.

    Satisfies :class:`~fleet.core._test_hooks.SleepProtocol`. The serving
    loop sleeps on its own thread and moves the clock at once; the main
    thread's sleeps are :func:`fleet.cli.node_serve.serve_on`'s, and the
    first of them waits for the watch thread to settle and count the run,
    standing for the seconds a real sleep lets the watch poll, so the case
    fixes the order a real serve takes rather than racing the watch to the
    handover.
    """

    def __init__(self, clock: FakeClock, settled: threading.Event) -> None:
        """Bind the clock and the count's event.

        Args:
            clock: The clock each sleep advances.
            settled: Set once the watch has settled and counted the run.
        """
        self._clock = clock
        self._settled = settled

    def __call__(self, seconds: int) -> None:
        """Wait for the settle when on the main thread, then advance the clock.

        Args:
            seconds: How long the caller asked to wait.
        """
        if threading.current_thread() is threading.main_thread():
            await_event(self._settled, what="the watch to settle the run")
        self._clock.seconds += seconds


def _malformed() -> frozenset[str] | None:
    """A queue listing the endpoint answered with something no job decodes from.

    A refusal no longer ends the loop (``QUEUE_UNANSWERED``,
    :mod:`fleet.cli.node_serve_claim`); an answer the runner cannot read
    still does, and is what this case fails the loop with.

    Raises:
        AppError: ``QUEUE_ANSWER_MALFORMED``, always.
    """
    raise AppError(code=FleetErrorCode.QUEUE_ANSWER_MALFORMED, message=MALFORMED)


def _fills_nothing() -> None:
    """A fill pass that launches nothing."""


def _rolled() -> str:
    """The roll, which never moves here.

    Returns:
        Its commit.
    """
    return "roll-1"


def _none_launching() -> int:
    """The launches under way, of which this serve starts none.

    Returns:
        None.
    """
    return 0


def _none_unreported() -> bool:
    """Whether a start report went unanswered, which none here does.

    Returns:
        False.
    """
    return False


class TestAServeWhoseQueueListingFails:
    def test_settles_the_run_that_ends_afterwards_then_raises_the_failure(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        launch(sourced_config)
        document = sourced_document((NPM_CI,))
        document["node_poll_seconds"] = 1
        sourced_config.write_text(dump_json_str(document), encoding="utf-8")
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(sourced_config)})
        clock = pin_clock(DEMO_NOW)
        # Read once still going, then once with its result written.
        _test_hooks.run = FakeRun([ok(""), ok(""), ok(""), ok(f"0 {DEMO_NOW + 72}")])
        closed: list[str] = []

        def settle(*, run_id: str) -> str:
            closed.append(run_id)
            return f"{run_id}: settled"

        watch = CountingWatch(loaded, settle)
        _test_hooks.sleep = SleepAfterTheSettle(clock, watch.counted)

        def collect() -> None:
            watch.hold(frozenset({DEMO_RUN_ID}))

        steps = node_serve.ServeSteps(
            collect=collect,
            fill=_fills_nothing,
            queued=_malformed,
            rolled=_rolled,
            launching=_none_launching,
            start_unreported=_none_unreported,
        )

        with caplog.at_level("INFO"), pytest.raises(AppError, match=MALFORMED):
            node_serve.serve(
                watch,
                alias="lavender",
                started=DEMO_NOW,
                serve_seconds=60,
                poll_seconds=1,
                steps=steps,
            )

        assert closed == [DEMO_RUN_ID]
        assert watch.closed() == 1
        assert clock.seconds == HANDOVER
        assert caplog.records[-1].getMessage() == (
            "lavender: its serving loop failed at 2025-09-04T15:33:21+00:00 (AppError: "
            f"{MALFORMED}); its watch served on to "
            "2025-09-04T15:35:50+00:00 with 1 run(s) closed, 0 still watched, and the "
            "failure ends this start"
        )
