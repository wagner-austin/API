"""Tests for the shared serve loop both HTTP surfaces run under.

Lifted out of ``test_service_main.py`` 2026-09-01 alongside the
function itself, when the fleet manager grew a drain and needed the
same start / wait / cleanup shape the bot service already had.
"""

from __future__ import annotations

import asyncio

import pytest

from tankpit_bot.service.serving import cancel_and_await, run_until_stopped
from tests.service._service_main_harness import _CancellingSite, _RecordingSite


class TestRunUntilStopped:
    """The blocking serve loop that binds a site to a stop signal."""

    async def test_starts_the_site_and_waits_on_the_stop_event(self) -> None:
        """Site is started, teardown waits until the stop event fires."""
        site = _RecordingSite()
        stop_event = asyncio.Event()

        async def flip_after_first_tick() -> None:
            await asyncio.sleep(0)
            stop_event.set()

        await asyncio.gather(
            run_until_stopped(site, stop_event, name="Test surface"),
            flip_after_first_tick(),
        )

        assert site.start_calls == 1
        assert site.cleanup_calls == 1

    async def test_cleanup_runs_even_when_stop_event_is_pre_set(self) -> None:
        """A stop event set before the wait still tears the site down."""
        site = _RecordingSite()
        stop_event = asyncio.Event()
        stop_event.set()

        await run_until_stopped(site, stop_event, name="Test surface")

        assert site.start_calls == 1
        assert site.cleanup_calls == 1

    async def test_cleanup_runs_when_start_raises(self) -> None:
        """A site that fails to start is still torn down before propagating.

        The ``AppRunner`` may have opened the socket before ``start``
        raised, so the ``finally`` is what stops a failed boot from
        leaking a bound port.
        """
        site = _CancellingSite()
        stop_event = asyncio.Event()

        with pytest.raises(asyncio.CancelledError):
            await run_until_stopped(site, stop_event, name="Test surface")

        assert site.start_calls == 1
        assert site.cleanup_calls == 1


class TestCancelAndAwait:
    """Cancelling a background task is not the same as ending it."""

    async def test_a_cancelled_task_runs_its_finally_before_this_returns(self) -> None:
        """THE POINT. ``cancel`` schedules; only the await completes it.

        Both surfaces cancel a task whose ``finally`` does real work --
        the bot service's autostart calls ``on_finished`` from one, and
        that is what stops a fleet child once its session ends. A
        caller that cancelled and returned left those blocks to run, or
        not run, after the loop had already moved on.
        """
        ran: list[str] = []

        async def sleeper() -> None:
            try:
                await asyncio.sleep(3600)
            finally:
                ran.append("cleaned up")

        task = asyncio.create_task(sleeper())
        await asyncio.sleep(0)

        await cancel_and_await(task)

        assert ran == ["cleaned up"]
        assert task.cancelled()

    async def test_an_already_finished_task_is_accepted(self) -> None:
        """Cancelling a completed task is a no-op, and must stay one.

        The autostart task normally finishes on its own -- that is what
        SETS the stop event -- so by the time the shutdown path cancels
        it, it is usually already done.
        """
        done: list[str] = []

        async def quick() -> None:
            done.append("finished")

        task = asyncio.create_task(quick())
        await task

        await cancel_and_await(task)

        assert done == ["finished"]
        assert not task.cancelled()

    async def test_a_task_that_failed_on_its_own_still_reports(self) -> None:
        """Only ``CancelledError`` is swallowed; a real failure is raised.

        A background task that died of its own accord is the one thing
        here worth hearing about, and this is the last place anyone is
        listening. Swallowing everything would turn a crashed idle
        monitor into a silent service.
        """

        async def broken() -> None:
            raise RuntimeError("idle monitor fell over")

        task = asyncio.create_task(broken())
        await asyncio.sleep(0)

        with pytest.raises(RuntimeError, match="idle monitor fell over"):
            await cancel_and_await(task)

    async def test_every_task_is_cancelled_before_any_is_awaited(self) -> None:
        """Order matters: cancel all, then wait.

        Cancelling and awaiting one at a time makes the second task run
        on for however long the first takes to unwind, which on a
        shutdown path is exactly the delay this exists to avoid.
        """
        started = asyncio.Event()
        ran: list[str] = []

        async def slow_to_unwind() -> None:
            try:
                started.set()
                await asyncio.sleep(3600)
            finally:
                ran.append("slow")

        async def other() -> None:
            try:
                await asyncio.sleep(3600)
            finally:
                ran.append("other")

        first = asyncio.create_task(slow_to_unwind())
        second = asyncio.create_task(other())
        await started.wait()
        await asyncio.sleep(0)

        await cancel_and_await(first, second)

        assert sorted(ran) == ["other", "slow"]
