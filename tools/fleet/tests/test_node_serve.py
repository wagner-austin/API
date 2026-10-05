"""A node runner's serve across its scheduled starts (MCPs board task 8993c306).

The loop cases drive :func:`fleet.cli.node_serve.serve_loop` with a pinned
clock whose sleeps move it, the passes and the queue listing recorded by
:class:`Steps`, and the two questions it asks its watch answered by
:class:`ScriptedWatch`: a serve of zero seconds runs its opening passes and
hands over 10 s before the first fire boundary with no listing and no read
of the roll; a longer one fills again when a job arrives or the watch
settles a run, runs the passes at each boundary, and hands over once it has
served its time or the roll has moved; a handover refused while a settle
runs, or a pass that runs past the fire, moves it to the next boundary. The
end-to-end case runs :func:`fleet.cli.node_agent.main` with a 60-second
serve against the node's and the queue's scripted answers, and the rest pin
the workspace's two declarations and the schedule the boundaries come from.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable, Sequence
from concurrent.futures import Future

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONTypeError, dump_json_str, load_json_str

from fleet.cli import node_agent, node_serve
from fleet.contracts.workspace import (
    NODE_POLL_CEILING_SECONDS,
    NODE_SERVE_CEILING_SECONDS,
    decode_fleet_workspace,
)
from fleet.core import _test_hooks, queue
from tests._node_agent_fixtures import (
    NPM_CI,
    PROBED,
    _credentials_in_env,
    _sourced_config,
    node_argv,
    sourced_document,
)
from tests._queue_fakes import DEFAULT_SHA, FakeQueue, queue_job
from tests.conftest import DEMO_NOW, FakeClock, FakeRun, ok, pin_clock

__all__ = ["_credentials_in_env", "_sourced_config"]

#: The fire boundary after DEMO_NOW, which is 20 s past one.
FIRST_FIRE = DEMO_NOW + 160

#: The boundary after it.
SECOND_FIRE = FIRST_FIRE + node_serve.TICK_SECONDS


class ScriptedWatch:
    """The serving loop's view of its watch, answered from a script.

    Satisfies :class:`fleet.cli.node_serve.Watched`.

    Attributes:
        settled: What :meth:`closed` answers; a case moves it to stand for a
            settle the watch thread made.
        asked: How many handovers were asked for.
    """

    settled: int
    asked: int

    def __init__(self, idle: Sequence[bool]) -> None:
        """Bind the answers to the handovers asked for.

        Args:
            idle: One per handover asked for, in order: True when the watch
                was idle and is now closed.
        """
        self.settled = 0
        self.asked = 0
        self._idle = list(idle)

    def closed(self) -> int:
        """How many runs the watch has settled.

        Returns:
            :attr:`settled`.
        """
        return self.settled

    def close_if_idle(self) -> bool:
        """The next scripted answer.

        Returns:
            Whether the watch is now closed.
        """
        self.asked += 1
        return self._idle.pop(0)


class Steps:
    """A serve's passes and questions, recorded with the clock's reading.

    Attributes:
        calls: ``<step>@<seconds after DEMO_NOW>`` for every step taken.
        clock: The pinned clock the steps read.
    """

    calls: list[str]
    clock: FakeClock

    def __init__(
        self,
        clock: FakeClock,
        *,
        queued: Callable[[int], frozenset[str]],
        rolled: Sequence[str],
        fill: Callable[[int], None],
    ) -> None:
        """Bind the answers.

        Args:
            clock: The pinned clock.
            queued: The queued job ids at a moment, seconds after DEMO_NOW.
            rolled: What the roll reads as, one per read.
            fill: What a fill pass does at a moment, beside being recorded.
        """
        self.calls = []
        self.clock = clock
        self._queued = queued
        self._rolled = list(rolled)
        self._fill = fill

    def _at(self) -> int:
        """Seconds after DEMO_NOW.

        Returns:
            The clock's reading less DEMO_NOW.
        """
        return self.clock.seconds - DEMO_NOW

    def collect(self) -> None:
        """Record a collect pass."""
        self.calls.append(f"collect@{self._at()}")

    def fill(self) -> None:
        """Record a fill pass and run the case's action."""
        self.calls.append(f"fill@{self._at()}")
        self._fill(self._at())

    def queued(self) -> frozenset[str]:
        """Record a queue listing.

        Returns:
            The case's queued job ids now.
        """
        self.calls.append(f"queued@{self._at()}")
        return self._queued(self._at())

    def rolled(self) -> str:
        """Record a read of the roll.

        Returns:
            The next scripted reading.
        """
        self.calls.append(f"rolled@{self._at()}")
        return self._rolled.pop(0)

    def bound(self) -> node_serve.ServeSteps:
        """The steps as the loop takes them.

        Returns:
            Them.
        """
        return node_serve.ServeSteps(
            collect=self.collect, fill=self.fill, queued=self.queued, rolled=self.rolled
        )


def _nothing_queued(at: int) -> frozenset[str]:
    """A queue that never holds a job.

    Args:
        at: Seconds after DEMO_NOW.

    Returns:
        Nothing.
    """
    return frozenset()


def _fill_does_nothing(at: int) -> None:
    """A fill pass that launches nothing.

    Args:
        at: Seconds after DEMO_NOW.
    """


def _serve(
    steps: Steps, watch: ScriptedWatch, *, serve_seconds: int, watching: Future[None] | None = None
) -> node_serve.Served:
    """Run the loop from DEMO_NOW with a five-second poll.

    Args:
        steps: The recorded steps.
        watch: The scripted watch.
        serve_seconds: The serve.
        watching: The watch thread's future; a fresh one still running
            when None.

    Returns:
        What the serve did.
    """
    return node_serve.serve_loop(
        watch,
        Future() if watching is None else watching,
        started=DEMO_NOW,
        serve_seconds=serve_seconds,
        poll_seconds=5,
        steps=steps.bound(),
    )


class TestAServeOfZeroSeconds:
    def test_runs_its_opening_passes_and_hands_over_before_the_first_fire(self) -> None:
        clock = pin_clock(DEMO_NOW)
        steps = Steps(clock, queued=_nothing_queued, rolled=(), fill=_fill_does_nothing)
        watch = ScriptedWatch([True])

        served = _serve(steps, watch, serve_seconds=0)

        assert steps.calls == ["collect@0", "fill@0"]
        assert served == node_serve.Served(
            started=DEMO_NOW,
            handed_over=FIRST_FIRE - node_serve.HANDOVER_SECONDS,
            fire=FIRST_FIRE,
            fires=0,
            fills=1,
            reason="its node_serve_seconds is 0",
        )

    def test_refused_while_a_settle_runs_renews_at_the_fire_and_hands_over_at_the_next(
        self,
    ) -> None:
        clock = pin_clock(DEMO_NOW)
        steps = Steps(clock, queued=_nothing_queued, rolled=(), fill=_fill_does_nothing)
        watch = ScriptedWatch([False, True])

        served = _serve(steps, watch, serve_seconds=0)

        # A serve past its time renews at the boundary and claims nothing more.
        assert steps.calls == ["collect@0", "fill@0", "collect@150"]
        assert served["handed_over"] == SECOND_FIRE - node_serve.HANDOVER_SECONDS
        assert served["fire"] == SECOND_FIRE
        assert served["fires"] == 1
        assert watch.asked == 2

    def test_whose_pass_runs_past_the_fire_moves_to_the_next_boundary(self) -> None:
        clock = pin_clock(DEMO_NOW)

        def past_the_fire(at: int) -> None:
            clock.seconds = FIRST_FIRE + 5

        steps = Steps(clock, queued=_nothing_queued, rolled=(), fill=past_the_fire)
        watch = ScriptedWatch([True])

        served = _serve(steps, watch, serve_seconds=0)

        assert steps.calls == ["collect@0", "fill@0", "collect@165"]
        assert served["fire"] == SECOND_FIRE
        assert served["fires"] == 1
        assert watch.asked == 1


class TestALongerServe:
    def test_fills_on_an_arrival_or_a_settle_and_renews_at_each_boundary(self) -> None:
        clock = pin_clock(DEMO_NOW)
        watch = ScriptedWatch([True])

        def arrivals(at: int) -> frozenset[str]:
            if at == 100:
                watch.settled = 1  # the watch settled a run before this poll
            if at >= 60:
                return frozenset({"job-a", "job-b"})
            if at >= 30:
                return frozenset({"job-a"})
            return frozenset()

        steps = Steps(clock, queued=arrivals, rolled=("roll-1", "roll-1"), fill=_fill_does_nothing)

        served = _serve(steps, watch, serve_seconds=200)

        fills = [call for call in steps.calls if call.startswith("fill@")]
        assert fills == ["fill@0", "fill@30", "fill@60", "fill@100", "fill@150"]
        assert [call for call in steps.calls if not call.startswith(("fill@", "queued@"))] == [
            "rolled@0",
            "collect@0",
            "rolled@150",
            "collect@150",
        ]
        assert served["reason"] == "served 330 s of its 200 s"
        assert served["fires"] == 1
        assert served["fills"] == len(fills)

    def test_hands_over_at_the_first_boundary_once_the_roll_has_moved(self) -> None:
        clock = pin_clock(DEMO_NOW)
        steps = Steps(
            clock, queued=_nothing_queued, rolled=("roll-1", "roll-2"), fill=_fill_does_nothing
        )

        served = _serve(steps, ScriptedWatch([True]), serve_seconds=NODE_SERVE_CEILING_SECONDS)

        assert served["reason"] == "refs/fleet/rolled moved from roll-1 to roll-2"
        assert served["handed_over"] == FIRST_FIRE - node_serve.HANDOVER_SECONDS
        # Thirty polls of five seconds to the handover, each a listing.
        assert sum(call.startswith("queued@") for call in steps.calls) == 30

    def test_whose_watch_thread_ended_stops_at_once(self) -> None:
        clock = pin_clock(DEMO_NOW)
        steps = Steps(clock, queued=_nothing_queued, rolled=("roll-1",), fill=_fill_does_nothing)
        ended: Future[None] = Future()
        ended.set_exception(AppError(FleetErrorCode.NODE_UNREACHABLE, "lavender went away"))

        served = _serve(steps, ScriptedWatch([]), serve_seconds=60, watching=ended)

        assert served["reason"] == "its watch thread ended"
        assert served["handed_over"] == DEMO_NOW
        assert steps.calls == ["rolled@0", "collect@0", "fill@0"]


class TestTheQueuedJobsAServeWatches:
    def test_are_those_naming_this_node_or_none(self, sourced_config: pathlib.Path) -> None:
        endpoint = FakeQueue(
            [
                dump_json_str(
                    {
                        "jobs": [
                            queue_job(id="11111111-1111-4111-8111-111111111111"),
                            queue_job(
                                id="22222222-2222-4222-8222-222222222222", requestedNode="lavender"
                            ),
                            queue_job(
                                id="33333333-3333-4333-8333-333333333333", requestedNode="loki"
                            ),
                        ]
                    }
                )
            ]
        )
        _test_hooks.http_post = endpoint

        ids = node_serve.queued_here(queue.load_credentials(), alias="lavender")

        assert ids == frozenset(
            {"11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222"}
        )
        assert endpoint.tools == ["dispatch_list"]
        assert endpoint.arguments[0] == {"status": "queued", "limit": queue.LISTING_PAGE_LIMIT}


class TestAServingRunner:
    def test_reads_the_roll_lists_the_queue_and_hands_over_once_its_time_is_served(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        document = sourced_document((NPM_CI,))
        document["node_serve_seconds"] = 60
        sourced_config.write_text(dump_json_str(document), encoding="utf-8")
        node = FakeRun([ok(f"{DEFAULT_SHA}\n"), *PROBED])
        _test_hooks.run = node
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": []}),  # the collect pass: nothing held
                dump_json_str({"claimed": None}),  # the fill pass: nothing matched
                dump_json_str({"jobs": []}),  # the poll at the handover: nothing queued
            ]
        )
        _test_hooks.http_post = endpoint

        with caplog.at_level("INFO"):
            assert node_agent.main(node_argv(sourced_config)) == 0

        # Read from the records directory, which is the document's own here,
        # resolved as the workspace loader resolves it.
        assert node.calls[0] == (
            "git",
            "-C",
            str(sourced_config.resolve().parent),
            "rev-parse",
            "--verify",
            "refs/fleet/rolled^{commit}",
        )
        assert len(node.calls) == 1 + len(PROBED)
        assert endpoint.tools == ["dispatch_list", "dispatch_claim", "dispatch_list"]
        assert endpoint.arguments[2] == {"status": "queued", "limit": queue.LISTING_PAGE_LIMIT}
        assert caplog.records[-1].getMessage() == (
            "lavender served 150 s from 2025-09-04T15:33:20+00:00: 0 fire(s), 1 fill pass(es), "
            "0 poll(s), 0 run(s) closed, 0 still watched; handed over at "
            "2025-09-04T15:35:50+00:00 before the 2025-09-04T15:36:00+00:00 fire: "
            "served 150 s of its 60 s"
        )


class TestTheDeclarations:
    def test_fleet_json_serves_twenty_minutes_and_polls_every_five_seconds(self) -> None:
        document = pathlib.Path(__file__).parent.parent / "fleet.json"
        workspace = decode_fleet_workspace(load_json_str(document.read_text(encoding="utf-8")))
        assert workspace["node_serve_seconds"] == 1200
        assert workspace["node_poll_seconds"] == 5

    @pytest.mark.parametrize(
        ("field", "seconds"),
        [
            ("node_serve_seconds", -1),
            ("node_serve_seconds", NODE_SERVE_CEILING_SECONDS + 1),
            ("node_poll_seconds", 0),
            ("node_poll_seconds", NODE_POLL_CEILING_SECONDS + 1),
        ],
    )
    def test_a_value_outside_its_bounds_is_refused(self, field: str, seconds: int) -> None:
        document = sourced_document((NPM_CI,))
        document[field] = seconds
        with pytest.raises(JSONTypeError, match=f"{field} is {seconds}"):
            decode_fleet_workspace(document)

    @pytest.mark.parametrize("field", ["node_serve_seconds", "node_poll_seconds"])
    def test_a_workspace_without_one_is_refused(self, field: str) -> None:
        document = sourced_document((NPM_CI,))
        del document[field]
        with pytest.raises(JSONTypeError, match=field):
            decode_fleet_workspace(document)

    def test_the_boundaries_are_the_schedules_three_minutes(self) -> None:
        """The serve's boundaries are the scheduled task's fires, registered
        at local midnight repeating every 3 minutes."""
        schedule = pathlib.Path(__file__).parent.parent / "scripts" / "FleetSchedule.ps1"
        text = schedule.read_text(encoding="utf-8")
        assert "-At (Get-Date).Date -RepetitionInterval (New-TimeSpan -Minutes 3)" in text
        assert node_serve.TICK_SECONDS == 3 * 60
        assert node_serve.next_fire(DEMO_NOW) == FIRST_FIRE
        assert node_serve.next_fire(FIRST_FIRE) == SECOND_FIRE
