"""Serve one aiohttp site until its stop signal fires.

Shared by the two long-running HTTP surfaces in this package: the bot
service (:mod:`tankpit_bot.service.service_main`) and the fleet
manager (:mod:`tankpit_bot.service.fleet`). Both want the same three
things -- start the site, wait for a stop signal, tear the site down
even when start raised -- so the loop lives here instead of once per
surface.

Lifted out of ``service_main`` 2026-09-01, when the fleet grew a
drain. The fleet had been blocking in :func:`aiohttp.web.run_app`,
which owns the whole process: there is no way to ask it to stop, and
therefore no way to stop the fleet's bots *before* the socket closes.
A stop :class:`asyncio.Event` is the whole difference between a
manager that can drain its children and one whose only exit is
abandoning them.
"""

from __future__ import annotations

import asyncio

from platform_core.logging import get_logger

from tankpit_bot.service._test_hooks import SiteRunnerProtocol

log = get_logger(__name__)


async def run_until_stopped(
    site: SiteRunnerProtocol,
    stop_event: asyncio.Event,
    *,
    name: str,
) -> None:
    """Serve until ``stop_event`` is set, then tear the site down.

    ``site.cleanup`` runs even when ``site.start`` raises -- the
    site's ``AppRunner`` may have partially set up (opened the socket,
    wired handlers) before start failed, and cleanup is idempotent.
    The ``finally`` wraps both stages for that reason.

    Args:
        site: The aiohttp site backing the HTTP surface.
        stop_event: Signal the caller sets to request shutdown --
            wired to a drain monitor, a signal handler, or a test
            harness that just calls ``stop_event.set()``.
        name: Human name of the surface, used in the ready and
            shutdown log lines so two surfaces in one log are
            distinguishable.

    Returns:
        None. Returns once the site has been torn down.
    """
    try:
        await site.start()
        log.info("%s ready", name)
        await stop_event.wait()
    finally:
        log.info("%s shutting down", name)
        await site.cleanup()


async def cancel_and_await(*tasks: asyncio.Task[None]) -> None:
    """Cancel background tasks and WAIT for each one to actually finish.

    ``Task.cancel`` only schedules the cancellation: it raises
    ``CancelledError`` inside the coroutine the next time the loop runs
    it. A caller that cancels and returns is asking the loop to shut
    down while those coroutines are still between their cancel and
    their unwinding, so their ``finally`` blocks may never execute and
    the loop closes with pending tasks -- which asyncio reports, after
    the fact and out of context, as "Task was destroyed but it is
    pending!".

    That is not cosmetic here. Both surfaces that call this cancel a
    task whose ``finally`` does real work:
    :func:`~tankpit_bot.service.service_main._autostart_session` calls
    ``on_finished`` from one, and it is the thing that stops the
    service when a fleet child's session ends.

    Cancellation is the expected outcome, so ``CancelledError`` is
    swallowed -- and ONLY ``CancelledError``. Anything else a task
    raised on its way out is re-raised, because a background task that
    failed for its own reasons has something to say and this is the
    last place anyone is listening.

    Args:
        tasks: The tasks to cancel and await. Already-finished tasks
            are fine: cancelling one is a no-op and awaiting it returns
            immediately.

    Raises:
        BaseException: Whatever a task raised, other than
            ``CancelledError``.
    """
    for task in tasks:
        task.cancel()
    for task in tasks:
        try:
            await task
        except asyncio.CancelledError:
            log.debug("Background task %r cancelled during shutdown", task.get_name())


__all__ = [
    "cancel_and_await",
    "run_until_stopped",
]
