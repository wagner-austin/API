"""Tests for :mod:`tankpit_bot.service.probe`.

The probe is exercised against a fixture-owned aiohttp test server
bound to a random free port — never the production 27100 — so the
tests stay stable regardless of whether a real service happens to be
running on the developer's machine.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from tankpit_bot.service.constants import health_url, resolve_service_port
from tankpit_bot.service.probe import (
    default_probe_existing_instance,
    probe_health_url,
)


class TestProbeHealthURL:
    """The stdlib :mod:`http.client` core of the existence probe."""

    async def _run_test_server(
        self,
        handler: Callable[[web.Request], Awaitable[web.Response]],
        *,
        path: str = "/health",
    ) -> TestServer:
        """Start an aiohttp :class:`TestServer` on a random port.

        Args:
            handler: aiohttp request handler serving ``path``.
            path: Route path the handler is registered under.

        Returns:
            The started :class:`TestServer` — the caller must invoke
            ``await server.close()`` when done. Exposes ``.port`` as
            a typed ``int``.
        """
        app = web.Application()
        app.router.add_get(path, handler)
        server = TestServer(app, host="127.0.0.1", port=0)
        await server.start_server()
        return server

    async def _probe_in_thread(self, url: str) -> bool:
        """Run the sync :func:`probe_health_url` off the event loop.

        The probe uses stdlib :mod:`http.client`, which blocks the
        thread it runs on. Called from an async test whose aiohttp
        :class:`TestServer` shares the SAME event loop, a synchronous
        call would deadlock: the loop can't accept the incoming TCP
        connection while blocked on the probe. :func:`asyncio.to_thread`
        moves the probe to a worker thread so the aiohttp side keeps
        spinning.
        """
        return await asyncio.to_thread(probe_health_url, url)

    async def test_returns_true_when_service_answers_with_ok(self) -> None:
        """A ``200 ok`` response is the signal we own an existing instance."""

        async def health_ok(_request: web.Request) -> web.Response:
            return web.Response(text="ok")

        server = await self._run_test_server(health_ok)
        try:
            result = await self._probe_in_thread(
                f"http://127.0.0.1:{server.port}/health",
            )
            assert result is True
        finally:
            await server.close()

    async def test_returns_false_when_body_is_not_ok(self) -> None:
        """A ``200`` response with any other body is another server, not us."""

        async def health_wrong_body(_request: web.Request) -> web.Response:
            return web.Response(text="hello")

        server = await self._run_test_server(health_wrong_body)
        try:
            result = await self._probe_in_thread(
                f"http://127.0.0.1:{server.port}/health",
            )
            assert result is False
        finally:
            await server.close()

    async def test_returns_false_on_non_200_status(self) -> None:
        """A non-200 status means the peer is not answering the health contract."""

        async def health_500(_request: web.Request) -> web.Response:
            return web.Response(status=500, text="boom")

        server = await self._run_test_server(health_500)
        try:
            result = await self._probe_in_thread(
                f"http://127.0.0.1:{server.port}/health",
            )
            assert result is False
        finally:
            await server.close()

    def test_returns_false_when_nothing_is_listening(self) -> None:
        """A connection refused (nothing on the port) is the "no instance" signal."""
        # Port 1 is the reserved TCP port for tcpmux, guaranteed by the
        # OS to reject a userland connect() — a stable "nothing is
        # listening" fixture without needing to open + close a real
        # socket to find a free port.
        assert probe_health_url("http://127.0.0.1:1/health") is False

    def test_returns_false_when_url_has_no_host(self) -> None:
        """A URL with no host component raises ``ValueError`` (drift catch)."""
        with pytest.raises(ValueError, match="probe URL missing host"):
            probe_health_url("http:///health")

    async def test_probe_uses_root_path_when_url_path_is_empty(self) -> None:
        """A URL with an empty path defaults to ``/`` — every HTTP server accepts it.

        Exercises the ``path if parsed.path != "" else "/"`` branch in
        :func:`probe_health_url`.
        """

        async def health_ok(_request: web.Request) -> web.Response:
            return web.Response(text="ok")

        server = await self._run_test_server(health_ok, path="/")
        try:
            result = await self._probe_in_thread(f"http://127.0.0.1:{server.port}")
            assert result is True
        finally:
            await server.close()


class TestDefaultProbeExistingInstance:
    """The production wrapper wired to the fixed loopback URL."""

    def test_default_probe_delegates_to_probe_health_url_on_fixed_url(self) -> None:
        """The wrapper is exactly ``probe_health_url`` on the resolved URL.

        The exact truthy/falsy outcome depends on whether a real
        :program:`tankpit-bot-service` happens to be running on the
        developer's machine at test time — but both calls hit the
        same target with the same 1-second timeout, so their results
        MUST agree. That equality proves the wrapper is a faithful
        delegate without depending on the environment.
        """
        assert default_probe_existing_instance() == probe_health_url(
            health_url(resolve_service_port())
        )
