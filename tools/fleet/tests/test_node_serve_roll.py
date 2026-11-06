"""A serve whose claim found the roll moved hands over at once (MCPs tasks 1fd01f09, 007bfc7e).

MCPs board task 1fd01f09: pendragon's serve claimed job 0fa14fab 16 s
before its boundary, on the registry before the roll. A claim pass reads the
ref (:mod:`fleet.cli.node_roll`); once it finds it moved, the serve lists
and fills nothing more.

MCPs board task 007bfc7e: and it hands over at the first poll at which it
holds no running run and launches none, instead of at the next boundary. On
2026-11-06 the roll from 870a207 to dbf5c71 left all seven gate checks of
MCPs d1f64219f queued from 23:10:50 to 23:11:56Z while no node claimed
anything, each serve waiting for its boundary. A serve still watching a run
settles it first, and one launching a job finishes the launch. The loop is
driven as ``test_node_serve.py`` drives it.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Future

from fleet.cli import node_serve
from fleet.contracts.workspace import NODE_SERVE_CEILING_SECONDS
from tests.conftest import DEMO_NOW, FakeClock, pin_clock
from tests.test_node_serve import FIRST_FIRE, SECOND_FIRE, ScriptedWatch, Steps

#: The reason a serve the roll moved hands over with.
MOVED = "refs/fleet/rolled moved from roll-1 to roll-2, nothing left to settle"


def nothing_queued(at: int) -> frozenset[str]:
    """A queue that never holds a job.

    Args:
        at: Seconds after DEMO_NOW.

    Returns:
        Nothing.
    """
    return frozenset()


def serve(steps: Steps, watch: ScriptedWatch) -> node_serve.Served:
    """Run the loop from DEMO_NOW with a five-second poll and the longest serve.

    Args:
        steps: The recorded steps.
        watch: The scripted watch.

    Returns:
        What the serve did.
    """
    return node_serve.serve_loop(
        watch,
        Future(),
        alias="pendragon",
        started=DEMO_NOW,
        serve_seconds=NODE_SERVE_CEILING_SECONDS,
        poll_seconds=5,
        steps=steps.bound(),
    )


class LaunchingSteps(Steps):
    """Steps whose opening fill finds the roll moved, with launches scripted per ask."""

    def __init__(self, clock: FakeClock, *, under_way: Sequence[int]) -> None:
        """Bind the counts.

        Args:
            clock: The pinned clock.
            under_way: One count per time the loop asks.
        """
        super().__init__(clock, queued=nothing_queued, rolled=("roll-2",), fill=self.claims)
        self._under_way = list(under_way)

    def claims(self, at: int) -> None:
        """A fill whose claim reads the roll moved.

        Args:
            at: Seconds after DEMO_NOW.
        """
        self.roll.moved()

    def launching(self) -> int:
        """The next scripted count.

        Returns:
            How many launches are under way.
        """
        return self._under_way.pop(0)


class TestAServeWhoseClaimFoundTheRollMoved:
    def test_hands_over_at_the_poll_whose_fill_found_it(self) -> None:
        clock = pin_clock(DEMO_NOW)
        watch = ScriptedWatch([True])

        def arrivals(at: int) -> frozenset[str]:
            return frozenset({"job-a"}) if at >= 30 else frozenset()

        def claims(at: int) -> None:
            if at == 30:
                steps.roll.moved()

        steps = Steps(clock, queued=arrivals, rolled=("roll-2",), fill=claims)

        served = serve(steps, watch)

        assert steps.calls[-3:] == ["queued@30", "fill@30", "rolled@30"]
        assert served["reason"] == MOVED
        # Not the 160 s boundary's 150 s: the poll that found it.
        assert served["handed_over"] == DEMO_NOW + 30
        assert served["fills"] == 2

    def test_a_move_found_by_the_opening_fill_lists_nothing_and_hands_over_at_once(self) -> None:
        clock = pin_clock(DEMO_NOW)

        def claims(at: int) -> None:
            steps.roll.moved()

        steps = Steps(clock, queued=nothing_queued, rolled=("roll-2",), fill=claims)

        served = serve(steps, ScriptedWatch([True]))

        assert steps.calls == ["collect@0", "fill@0", "rolled@0"]
        assert served["handed_over"] == DEMO_NOW

    def test_a_move_found_at_a_boundarys_fill_hands_over_there(self) -> None:
        """The boundary reads the ref unmoved and renews; the fill it runs
        there reads it moved, and the serve hands over without waiting for
        the boundary after."""
        clock = pin_clock(DEMO_NOW)

        def claims(at: int) -> None:
            if at == 150:
                steps.roll.moved()

        steps = Steps(clock, queued=nothing_queued, rolled=("roll-1", "roll-2"), fill=claims)

        served = serve(steps, ScriptedWatch([True]))

        assert [call for call in steps.calls if not call.startswith("queued@")] == [
            "collect@0",
            "fill@0",
            "rolled@150",
            "collect@150",
            "fill@150",
            "rolled@150",
        ]
        assert served["reason"] == MOVED
        assert served["handed_over"] == FIRST_FIRE - node_serve.HANDOVER_SECONDS
        assert served["fire"] == SECOND_FIRE
        assert served["fills"] == 2

    def test_still_watching_a_run_it_settles_it_and_hands_over_on_that_poll(self) -> None:
        clock = pin_clock(DEMO_NOW)

        def claims(at: int) -> None:
            steps.roll.moved()

        steps = Steps(clock, queued=nothing_queued, rolled=("roll-2",), fill=claims)
        # Its run is still running at 0 and 5 s and settled by 10 s.
        watch = ScriptedWatch([True], running=[1, 1, 0])

        served = serve(steps, watch)

        assert served["handed_over"] == DEMO_NOW + 10
        assert watch.asked == 1
        # Nothing was listed or claimed while it waited.
        assert steps.calls == ["collect@0", "fill@0", "rolled@0"]

    def test_a_settle_under_way_or_a_launch_holds_it_a_poll_each(self) -> None:
        clock = pin_clock(DEMO_NOW)
        steps = LaunchingSteps(clock, under_way=[1, 0, 0])
        watch = ScriptedWatch([False, True])

        served = serve(steps, watch)

        # A launch at 0 s, a settle refusing the close at 5 s, closed at 10 s.
        assert served["handed_over"] == DEMO_NOW + 10
        assert watch.asked == 2
