"""Serving seams: how a surface gets an aiohttp site and runs it.

The bot service and the fleet manager both bind a port, both hand
their app to :func:`~tankpit_bot.service.serving.run_until_stopped`,
and both need tests to exercise that wiring without opening a socket.
Those seams live here.
"""

from __future__ import annotations

import asyncio
from typing import Protocol

from aiohttp import web

from tankpit_bot.service.probe import default_probe_existing_instance


class ProbeExistingInstanceProtocol(Protocol):
    """Detects whether another :func:`main` invocation is already serving.

    :func:`main` calls this before :data:`serve` so a double-invocation
    (double-tap of the phone's SERVER button, a second ``make service``
    while the first is still up) is idempotent — the second instance
    exits cleanly instead of crash-looping while the outer respawn loop
    fights the port-bind conflict every 5 s.
    """

    def __call__(self) -> bool:
        """Return True when a tankpit-bot-service instance already responds.

        Returns:
            True when ``GET http://127.0.0.1:27100/health`` returns
            ``200 ok``; False when the port is idle OR another
            (non-us) process holds the port and does not answer
            ``/health`` with the expected body.
        """
        ...


class ServeProtocol(Protocol):
    """Runs the bot service until it exits (defaults to :func:`asyncio.run`).

    Nullary by design: the hook does not accept a coroutine argument
    because a strict-typed :class:`Awaitable` / :class:`Coroutine`
    signature cannot be passed to :func:`asyncio.run` without leaking
    ``Any`` through mypy's generic constraints. Production wires the
    hook to a factory that builds + drives the coroutine internally.
    """

    def __call__(self) -> None:
        """Run the bot service to completion or fail loudly."""
        ...


class SiteRunnerProtocol(Protocol):
    """Aiohttp lifecycle handle :func:`run_until_stopped` consumes."""

    async def start(self) -> None:
        """Begin serving. Non-blocking; the event loop keeps running."""
        ...

    async def cleanup(self) -> None:
        """Tear the site + underlying application runner down cleanly."""
        ...


class SiteFactoryProtocol(Protocol):
    """Async factory that produces a started-ready site for an app."""

    async def __call__(
        self,
        app: web.Application,
        host: str,
        port: int,
    ) -> SiteRunnerProtocol:
        """Set up aiohttp and return the merged ``SiteRunnerProtocol``.

        Args:
            app: Routed :class:`web.Application` to serve.
            host: Loopback host to bind to.
            port: TCP port to bind to.

        Returns:
            A :class:`SiteRunnerProtocol` whose ``start`` / ``cleanup``
            :func:`run_until_stopped` drives.
        """
        ...


class _AppRunnerCleanupProtocol(Protocol):
    """Minimum surface :class:`_AiohttpSite` needs from :class:`web.AppRunner`."""

    async def cleanup(self) -> None:
        """Tear the application runner down."""
        ...


class _TCPSiteStartProtocol(Protocol):
    """Minimum surface :class:`_AiohttpSite` needs from :class:`web.TCPSite`."""

    async def start(self) -> None:
        """Begin serving on the site's host + port."""
        ...


class _AiohttpSite:
    """Adapter matching :class:`SiteRunnerProtocol` for aiohttp's split API.

    aiohttp splits the lifecycle across :class:`web.AppRunner` (setup +
    cleanup) and :class:`web.TCPSite` (start + stop). The service main's
    :class:`SiteRunnerProtocol` merges the two so
    :func:`run_until_stopped` sees a single start / cleanup surface.
    """

    def __init__(
        self,
        aiohttp_runner: _AppRunnerCleanupProtocol,
        site: _TCPSiteStartProtocol,
    ) -> None:
        """Store both aiohttp handles for the merged lifecycle.

        Args:
            aiohttp_runner: The aiohttp application runner, already
                ``setup``-called by the caller.
            site: The TCP site pinning the app to a host + port.
        """
        self._aiohttp_runner = aiohttp_runner
        self._site = site

    async def start(self) -> None:
        """Begin serving on the site's host + port."""
        await self._site.start()

    async def cleanup(self) -> None:
        """Tear down the site and its underlying application runner."""
        await self._aiohttp_runner.cleanup()


async def _real_build_site(
    app: web.Application,
    host: str,
    port: int,
) -> SiteRunnerProtocol:
    """Production site factory — wires aiohttp's split AppRunner + TCPSite.

    ``reuse_address=True`` is passed to :class:`web.TCPSite` so a
    restart within Windows' ~120 s ``TIME_WAIT`` window succeeds
    instead of failing with ``WinError 10048`` (EADDRINUSE). Every
    Ctrl+C exit + immediate ``make service`` restart pattern hits
    this if the flag is missing; the flag maps to the underlying
    ``SO_REUSEADDR`` socket option, which is the standard fix.

    Args:
        app: Routed :class:`web.Application` to serve.
        host: Loopback host to bind to.
        port: TCP port to bind to.

    Returns:
        The aiohttp AppRunner + TCPSite pair merged into a
        :class:`SiteRunnerProtocol` via :class:`_AiohttpSite`.
    """
    aiohttp_runner = web.AppRunner(app)
    await aiohttp_runner.setup()
    return _AiohttpSite(
        aiohttp_runner,
        web.TCPSite(aiohttp_runner, host, port, reuse_address=True),
    )


def _real_serve() -> None:
    """Production entry point — drives ``_async_main`` under :func:`asyncio.run`.

    Kept inside the hook so the coroutine object never crosses a
    Protocol boundary and mypy sees a plain nullary callable.
    """
    from tankpit_bot.service.service_main import _async_main

    asyncio.run(_async_main())


def _real_serve_fleet() -> None:
    """Production entry point — drives the fleet's ``_async_main``.

    Kept inside the hook so the coroutine object never crosses a
    Protocol boundary and mypy sees a plain nullary callable, exactly
    as :func:`_real_serve` does for the bot service.
    """
    from tankpit_bot.service.fleet import _async_main

    asyncio.run(_async_main())


#: Site-construction hook — production wires aiohttp; tests inject a
#: fake that returns a recording :class:`SiteRunnerProtocol` without
#: opening a socket.
build_site: SiteFactoryProtocol = _real_build_site

#: Existence-probe hook — production sends a live HTTP GET to the
#: expected health endpoint (see :mod:`tankpit_bot.service.probe`);
#: tests inject a scriptable double that returns True or False to
#: exercise the short-circuit + normal paths of :func:`main` without
#: touching the network.
probe_existing_instance: ProbeExistingInstanceProtocol = default_probe_existing_instance

#: Service-run hook — production drives ``_async_main`` under
#: :func:`asyncio.run`; tests replace with a fake that either returns
#: normally or raises :class:`KeyboardInterrupt` to exercise
#: :func:`main`'s interrupt branch without a real event loop.
serve: ServeProtocol = _real_serve

#: Fleet-manager serve seam — production drives the fleet's
#: ``_async_main`` under :func:`asyncio.run`; tests replace it with a
#: fake so ``fleet.main`` is exercised without opening a socket.
serve_fleet: ServeProtocol = _real_serve_fleet
