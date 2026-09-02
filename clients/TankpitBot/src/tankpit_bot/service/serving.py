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


__all__ = [
    "run_until_stopped",
]
