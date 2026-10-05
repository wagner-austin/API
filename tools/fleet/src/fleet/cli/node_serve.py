"""A node runner SERVES across its scheduled starts instead of ticking once.

WHY (MCPs board task 8993c306). Each node's runner is a scheduled task that
starts every 3 minutes under IgnoreNew (``scripts/FleetSchedule.ps1``). Run
once per start, it claimed a job only on a start and closed a finished one
only on a start, so a row's wall time carried up to two waits of 3 minutes
beside its work. Measured on 2026-10-04 (board task 74b13c20, lavender-wsl):
job 3308c976 did 20 s of work in a 182 s row, closed 148 s after its check
ended, after 150 s queued on an idle node. The 100-second watch that followed
(board task c1d48330) covered 100 s of each 180 s, and a fill that ran past
the next fire skipped it: loki's 77fee544 ended 05:53:35Z on 2026-10-05 and
closed 05:57:44Z, 249 s later.

WHAT A SERVE DOES. It runs a start's two passes as every start did
(:func:`fleet.cli.node_collect.collect_pass`, then
:func:`fleet.cli.node_agent.fill_pass`) and then keeps going: its watch
thread (:mod:`fleet.cli.node_watch`) reads each held run every
``node_poll_seconds`` of the workspace and settles one the moment it has
ended, and at the same pace the loop's thread lists the queued jobs (one
``dispatch_list`` call, :func:`fleet.core.queue.queued_for`) and runs the
fill pass again when a job naming this node or no node has arrived since
its last fill, or the watch has settled a run since then. The
fill pass is the one every start runs, so the claim still goes through the
node's probe, its tags, its leases, its owner reservation and the elevated
yield, unchanged. At every fire boundary it runs the collect pass again,
which renews each running job's queue lease as each start did.

WHEN IT HANDS OVER. Only :data:`HANDOVER_SECONDS` before a fire boundary,
and only while no settle is under way (:meth:`fleet.cli.node_watch.RunWatch.close_if_idle`)
and no claimed job is still being launched (:mod:`fleet.cli.node_launch`),
once it has served the workspace's ``node_serve_seconds`` or
``refs/fleet/rolled`` has moved off the commit it runs, so a roll takes
effect at the next boundary as it did before. The scheduler's fires that
arrive while it serves are skipped by IgnoreNew, as intended, and the
next fire after the handover starts the next serve: the node is unwatched
only for those seconds and the next start's few, once a serve rather than
for 80 s of every start. A serve past its time claims nothing more, so it
reaches an idle boundary within a fire or two.

WHEN THE LOOP FAILS, THE WATCH SERVES ON TO THE BOUNDARY. The loop runs on
a thread of its own beside the watch's. Until 2026-10-05 it ran on the main
thread and its error closed the watch at once: at 07:24:56Z that day the
dispatch endpoint refused one ``dispatch_list`` (URLError, WinError 10061),
all 7 serving runners ended, and lavender-wsl's run that ended 16 s later
closed at the 07:27 start, 120.9 s after its check (row 62734702). Now an
error from the loop ends the claiming and nothing else: the watch goes on
settling what this runner holds until :data:`HANDOVER_SECONDS` before the
next fire boundary (:func:`serve_on`), and only then is the loop's error
raised, unchanged, so the start still fails with it and the next fire
starts the next serve, as it would have anyway.

A QUEUE THAT DID NOT ANSWER IS NOT SUCH AN ERROR. The watch above still
ended at the first settle the outage refused: at 11:54:58Z on 2026-10-05
the listing was refused while a deploy recreated mcp-fleet, row bde22e57's
run ended at 11:55:29Z, its settle met the same refusal, and it closed at
the 11:57 fire, 105 s after its check. A call nothing answered now raises
``QUEUE_UNANSWERED`` (:mod:`fleet.core.queue_transport`), and the loop
(:func:`serve_loop`) and the watch each ask again at their next poll, so a
run ending in an outage closes, and a job queued in one is claimed, within
a poll of the queue answering again.

THE BOUNDARIES ARE THE SCHEDULE'S. FleetSchedule.ps1 registers each
runner's task once at local midnight, repeating every 3 minutes; local
midnight is a whole multiple of 180 s since the epoch wherever the UTC
offset is a whole number of 3-minute steps, which every offset in use is,
so a fire boundary is a multiple of :data:`TICK_SECONDS` since the epoch.
The hub's logs of 2026-10-05 show each start 2 to 4 s after one.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Final, Protocol, TypedDict

from platform_core.logging import get_logger

from fleet.cli.node_serve_claim import Claiming
from fleet.cli.node_watch import THREAD_PREFIX, RunWatch
from fleet.core import _test_hooks
from fleet.core.rolled import ROLLED_REF

_log = get_logger(__name__)

#: The prefix of the serving loop's thread's name; the executor appends ``_0``.
LOOP_PREFIX: Final = "fleet-serve"

#: The scheduled tasks' repetition, in seconds (``scripts/FleetSchedule.ps1``).
TICK_SECONDS: Final[int] = 180

#: How long before a fire boundary a serve decides to hand over: enough for
#: the watch to finish the sleep it is in and the launcher to remove its
#: extraction and write the start's log, so the process has exited when
#: the scheduler fires.
HANDOVER_SECONDS: Final[int] = 10


class Watched(Protocol):
    """The two questions the serving loop asks its watch
    (:class:`fleet.cli.node_watch.RunWatch`)."""

    def closed(self) -> int:
        """How many runs the watch has settled so far.

        Returns:
            Their count.
        """

    def close_if_idle(self) -> bool:
        """Close the watch unless a settle is under way.

        Returns:
            Whether it is now closed.
        """


class ServeSteps(TypedDict):
    """What a serve runs, bound to its node by :func:`fleet.cli.node_agent.main`.

    Attributes:
        collect: The collect pass: settles, renews and reconciles what this
            runner holds, and stops what was cancelled.
        fill: The fill pass: claims and launches until the node has no room
            or the lane nothing it fits.
        queued: The ids of the queued jobs naming this node or no node, or
            None when the queue did not answer
            (:func:`fleet.cli.node_serve_claim.queued_here`).
        rolled: What ``refs/fleet/rolled`` names now
            (:func:`fleet.core.rolled.rolled_state`).
        launching: How many claimed jobs are still being launched
            (:mod:`fleet.cli.node_launch`); a serve does not hand over while
            any is, since its process would outlive the fire boundary
            draining it, the scheduler would skip that fire, and the run
            would go unwatched until the next (job 6c568ecb on diphtheria,
            2026-10-05: launched 17 s after a 10:05:50Z handover, ended
            10:06:43Z, closed 10:09:14Z).
        start_unreported: Whether a launch's start report went unanswered
            since the last ask (:meth:`fleet.cli.node_launch.Launcher.take_unreported`),
            which the loop recovers from as from any unanswered pass.
    """

    collect: Callable[[], None]
    fill: Callable[[], None]
    queued: Callable[[], frozenset[str] | None]
    rolled: Callable[[], str]
    launching: Callable[[], int]
    start_unreported: Callable[[], bool]


class Handover(TypedDict):
    """When a serve hands over, fixed at its start by :func:`handover_policy`.

    Attributes:
        at_start: Why the serve is due to hand over from its very start, or
            None when it serves on.
        due: Why it should hand over at a boundary reached at the given
            moment, or None to keep serving.
    """

    at_start: str | None
    due: Callable[[int], str | None]


class Served(TypedDict):
    """What one serve did, for the line it ends with.

    Attributes:
        started: When it started, whole seconds since the epoch.
        handed_over: When it handed over.
        fire: The fire boundary it handed over before.
        fires: How many fire boundaries it ran the passes at, after its first.
        fills: How many fill passes it ran, its first included.
        reason: Why it handed over.
    """

    started: int
    handed_over: int
    fire: int
    fires: int
    fills: int
    reason: str


def next_fire(now: int) -> int:
    """The first fire boundary after a moment.

    Args:
        now: Whole seconds since the epoch.

    Returns:
        The next multiple of :data:`TICK_SECONDS` strictly after it.
    """
    return (now // TICK_SECONDS + 1) * TICK_SECONDS


def handover_policy(*, started: int, serve_seconds: int, rolled: Callable[[], str]) -> Handover:
    """Decide, at a serve's start, how it will decide to hand over.

    A serve of ZERO seconds is due from its start: it runs the opening
    passes, claims nothing after them and hands over at the first boundary,
    the shape of one scheduled start, so it never reads the roll, since
    nothing it read could change when it hands over. Any longer serve reads
    the roll now and hands over at the first boundary at which it has served
    its time or the roll has moved, so a roll takes effect at the next
    boundary, as it did when every start was a new process. A roll made in
    the seconds between the launcher's extraction and this read keeps the
    serve on the older code until its time is up.

    Args:
        started: When the serve started.
        serve_seconds: The workspace's ``node_serve_seconds``.
        rolled: Reads what ``refs/fleet/rolled`` names now.

    Returns:
        The policy.
    """
    if serve_seconds == 0:
        return Handover(at_start="its node_serve_seconds is 0", due=lambda now: None)
    at_start = rolled()

    def due(now: int) -> str | None:
        served = now - started
        if served >= serve_seconds:
            return f"served {served} s of its {serve_seconds} s"
        current = rolled()
        if current == at_start:
            return None
        return f"{ROLLED_REF} moved from {at_start} to {current}"

    return Handover(at_start=None, due=due)


def serve_loop(
    watch: Watched,
    watching: Future[None],
    *,
    alias: str,
    started: int,
    serve_seconds: int,
    poll_seconds: int,
    steps: ServeSteps,
) -> Served:
    """Run the passes, then claim and renew until a handover or the watch's end.

    Between boundaries each poll is :meth:`fleet.cli.node_serve_claim.Claiming.poll`,
    and A QUEUE THAT DID NOT ANSWER costs a poll there, not the serve (MCPs
    board task 8993c306). The watch rides the same outage on its own,
    settling a run that ended in it at the first poll the queue answers
    (:mod:`fleet.cli.node_watch`).

    Args:
        watch: The watch the passes hand their runs to.
        watching: The watch thread, whose end before a handover means it
            raised.
        alias: This node's workspace name, for the log.
        started: When the serve started.
        serve_seconds: The workspace's ``node_serve_seconds``.
        poll_seconds: The workspace's ``node_poll_seconds``.
        steps: The serve's passes and questions.

    Returns:
        What the serve did; when the watch thread ended first, the reason
        says so and :func:`serve` raises the thread's error.

    Raises:
        AppError: From the passes and the queue listing, but for
            ``QUEUE_UNANSWERED``. Not caught: the next start meets the same
            runs.
    """
    policy = handover_policy(started=started, serve_seconds=serve_seconds, rolled=steps["rolled"])
    fire = next_fire(started)
    claiming = Claiming(
        alias=alias,
        collect=steps["collect"],
        fill=steps["fill"],
        queued=steps["queued"],
        start_unreported=steps["start_unreported"],
        closed=watch.closed,
    )
    claiming.passes(fill=True)
    fires = 0
    reason = policy["at_start"]
    while True:
        now = _test_hooks.now()
        if watching.done():
            return Served(
                started=started,
                handed_over=now,
                fire=fire,
                fires=fires,
                fills=claiming.fills,
                reason="its watch thread ended",
            )
        handover = fire - HANDOVER_SECONDS
        if now < handover:
            _test_hooks.sleep(min(poll_seconds, handover - now))
            if reason is None:
                claiming.poll()
            continue
        if reason is None:
            reason = policy["due"](now)
        if (
            reason is not None
            and now < fire
            and steps["launching"]() == 0
            and watch.close_if_idle()
        ):
            return Served(
                started=started,
                handed_over=now,
                fire=fire,
                fires=fires,
                fills=claiming.fills,
                reason=reason,
            )
        fire = next_fire(max(now, fire))
        fires += 1
        claiming.passes(fill=reason is None)


def serve_on(watch: Watched, watching: Future[None], *, poll_seconds: int) -> int:
    """Keep the watch settling after the loop has failed, until the handover.

    The handover is :data:`HANDOVER_SECONDS` before the first fire boundary
    after the failure, the moment a serving loop would have handed over at,
    and it waits, as the loop's does, until no settle is under way. A
    failure inside those last seconds hands over at once.

    Args:
        watch: The watch the loop handed its runs to.
        watching: The watch thread, whose end stops the wait at once.
        poll_seconds: The workspace's ``node_poll_seconds``, the pace at
            which the thread's end and the handover are looked for.

    Returns:
        When the watch was closed, or when its thread was found ended.
    """
    handover = next_fire(_test_hooks.now()) - HANDOVER_SECONDS
    while True:
        now = _test_hooks.now()
        if watching.done():
            return now
        if now >= handover and watch.close_if_idle():
            return now
        _test_hooks.sleep(min(poll_seconds, handover - now) if now < handover else poll_seconds)


def _utc(seconds: int) -> str:
    """A moment as the serve's line prints it.

    Args:
        seconds: Whole seconds since the epoch.

    Returns:
        Its ISO 8601 form in UTC.
    """
    return datetime.datetime.fromtimestamp(seconds, tz=datetime.UTC).isoformat()


def serve(
    watch: RunWatch,
    *,
    alias: str,
    started: int,
    serve_seconds: int,
    poll_seconds: int,
    steps: ServeSteps,
) -> Served:
    """Serve one node with the watch and the loop each on a thread, then say what was done.

    Args:
        watch: The watch the passes hand their runs to.
        alias: This node's workspace name, which names both threads.
        started: When the serve started.
        serve_seconds: The workspace's ``node_serve_seconds``.
        poll_seconds: The workspace's ``node_poll_seconds``.
        steps: The serve's passes and questions.

    Returns:
        What the serve did.

    Raises:
        AppError: From the loop, once the watch has served on to the
            handover (:func:`serve_on`) and ended; or from the watch, once
            the loop has stopped. Neither is caught: the next start meets
            the same runs. Any other error the loop raised is raised the
            same way; a queue that did not answer is not one
            (:func:`serve_loop`).
    """
    with (
        ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"{THREAD_PREFIX}-{alias}"
        ) as watch_pool,
        ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{LOOP_PREFIX}-{alias}") as loop_pool,
    ):
        watching = watch_pool.submit(watch.watch)
        with watch:
            looping = loop_pool.submit(
                serve_loop,
                watch,
                watching,
                alias=alias,
                started=started,
                serve_seconds=serve_seconds,
                poll_seconds=poll_seconds,
                steps=steps,
            )
            failure = looping.exception()
            failed = _test_hooks.now()
            handed_over = (
                failed if failure is None else serve_on(watch, watching, poll_seconds=poll_seconds)
            )
    # Both threads have ended here, so a settle the watch finished before it
    # was closed is in its count.
    if failure is not None:
        _log.info(
            "%s: its serving loop failed at %s (%s: %s); its watch served on to %s "
            "with %d run(s) closed, %d still watched, and the failure ends this start",
            alias,
            _utc(failed),
            type(failure).__name__,
            failure,
            _utc(handed_over),
            watch.closed(),
            watch.still_watched(),
        )
    served = looping.result()
    watching.result()
    _log.info(
        "%s served %d s from %s: %d fire(s), %d fill pass(es), %d poll(s), %d run(s) closed, "
        "%d still watched; handed over at %s before the %s fire: %s",
        alias,
        served["handed_over"] - served["started"],
        _utc(served["started"]),
        served["fires"],
        served["fills"],
        watch.polls,
        watch.closed(),
        watch.still_watched(),
        _utc(served["handed_over"]),
        _utc(served["fire"]),
        served["reason"],
    )
    return served


__all__ = [
    "HANDOVER_SECONDS",
    "LOOP_PREFIX",
    "TICK_SECONDS",
    "Handover",
    "ServeSteps",
    "Served",
    "Watched",
    "handover_policy",
    "next_fire",
    "serve",
    "serve_loop",
    "serve_on",
]
