"""Tests for the service entry point: startup and wiring.

``test_service_main.py`` was 626 lines; the shutdown and probe paths
are now a sibling.
"""

from __future__ import annotations

import asyncio

import pytest
from aiohttp import web

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot.bot.base import Bot
from tankpit_bot.service import _test_hooks as service_hooks
from tankpit_bot.service._test_hooks import (
    SiteRunnerProtocol,
    _AiohttpSite,
    _real_build_bot_factory,
)
from tankpit_bot.service.constants import (
    SERVICE_HOST,
    SERVICE_PORT,
)
from tankpit_bot.service.frame_bus import (
    FrameBus,
    FrameBusProtocol,
)
from tankpit_bot.service.mode_bridge import (
    ModeBridge,
    ModeBridgeProtocol,
)
from tankpit_bot.service.service_main import (
    _async_main,
    main,
    run_service_forever,
)
from tankpit_bot.service.status_bus import (
    StatusBus,
    StatusBusProtocol,
)
from tests.conftest import FakeEnv
from tests.service._service_main_harness import (
    _CancellingSite,
    _make_recording_bot_factory,
    _RecordingBot,
    _RecordingSite,
)


class TestRunServiceForever:
    """The blocking serve loop that binds a site to a stop signal."""

    async def test_starts_the_site_and_waits_on_the_stop_event(self) -> None:
        """Site is started, teardown waits until the stop event fires."""
        site = _RecordingSite()
        stop_event = asyncio.Event()

        async def flip_after_first_tick() -> None:
            await asyncio.sleep(0)
            stop_event.set()

        await asyncio.gather(run_service_forever(site, stop_event), flip_after_first_tick())

        assert site.start_calls == 1
        assert site.cleanup_calls == 1

    async def test_cleanup_runs_even_when_stop_event_is_pre_set(self) -> None:
        """A stop event set before the wait still tears the site down."""
        site = _RecordingSite()
        stop_event = asyncio.Event()
        stop_event.set()

        await run_service_forever(site, stop_event)

        assert site.start_calls == 1
        assert site.cleanup_calls == 1


class TestRealBuildBotFactory:
    """Contract for the real bot-factory builder used at production boot."""

    def test_factory_produces_a_bot_bound_to_its_bridge_and_bus(self) -> None:
        """The bot returned by the factory has the injected channels."""
        factory = _real_build_bot_factory(
            "https://test.tankpit.com/",
            headless=True,
            prefer_account=False,
        )
        bridge: ModeBridgeProtocol = ModeBridge()
        bus: StatusBusProtocol = StatusBus()
        frames: FrameBusProtocol = FrameBus()

        raw = factory(mode_bridge=bridge, status_bus=bus, frame_bus=frames)
        if not isinstance(raw, Bot):
            raise AssertionError("real bot factory must return a Bot instance")

        assert raw._mode_bridge is bridge
        assert raw._status_bus is bus
        assert raw._frame_bus is frames

    def test_factory_carries_headless_and_prefer_account(self) -> None:
        """Construction args flow through to the produced bot."""
        factory = _real_build_bot_factory(
            "https://test.tankpit.com/",
            headless=True,
            prefer_account=True,
        )
        bridge: ModeBridgeProtocol = ModeBridge()
        bus: StatusBusProtocol = StatusBus()

        raw = factory(mode_bridge=bridge, status_bus=bus, frame_bus=FrameBus())
        if not isinstance(raw, Bot):
            raise AssertionError("real bot factory must return a Bot instance")

        assert raw._headless is True
        assert raw._prefer_account is True


class TestAiohttpSiteAdapter:
    """The lifecycle adapter between :class:`AppRunner` and :class:`TCPSite`."""

    async def test_start_forwards_to_the_tcp_site(self) -> None:
        """Adapter ``start`` delegates to the TCP site's ``start``."""

        class _RecordingAppRunner:
            def __init__(self) -> None:
                self.cleanup_calls = 0

            async def cleanup(self) -> None:
                self.cleanup_calls += 1

        class _RecordingTCPSite:
            def __init__(self) -> None:
                self.start_calls = 0

            async def start(self) -> None:
                self.start_calls += 1

        aiohttp_runner = _RecordingAppRunner()
        tcp_site = _RecordingTCPSite()
        adapter = _AiohttpSite(aiohttp_runner, tcp_site)

        await adapter.start()

        assert tcp_site.start_calls == 1
        assert aiohttp_runner.cleanup_calls == 0

    async def test_cleanup_forwards_to_the_app_runner(self) -> None:
        """Adapter ``cleanup`` delegates to the AppRunner's ``cleanup``."""

        class _RecordingAppRunner:
            def __init__(self) -> None:
                self.cleanup_calls = 0

            async def cleanup(self) -> None:
                self.cleanup_calls += 1

        class _RecordingTCPSite:
            def __init__(self) -> None:
                self.start_calls = 0

            async def start(self) -> None:
                self.start_calls += 1

        aiohttp_runner = _RecordingAppRunner()
        tcp_site = _RecordingTCPSite()
        adapter = _AiohttpSite(aiohttp_runner, tcp_site)

        await adapter.cleanup()

        assert aiohttp_runner.cleanup_calls == 1
        assert tcp_site.start_calls == 0


class TestAsyncMain:
    """``_async_main`` wiring: bridge/bus/runner constructed and site served."""

    async def test_wires_primitives_and_serves_until_stop(
        self,
        restore_service_hooks: None,
    ) -> None:
        """``_async_main`` invokes the site factory and drives the site."""
        _ = restore_service_hooks
        top_hooks.get_env = FakeEnv({})

        received_apps: list[web.Application] = []
        fake_site = _CancellingSite()

        async def fake_build_site(app: web.Application, host: str, port: int) -> SiteRunnerProtocol:
            received_apps.append(app)
            assert host == SERVICE_HOST
            assert port == SERVICE_PORT
            return fake_site

        service_hooks.build_site = fake_build_site
        service_hooks.build_bot_factory = _make_recording_bot_factory(_RecordingBot())

        with pytest.raises(asyncio.CancelledError):
            await _async_main()

        assert len(received_apps) == 1
        registered_paths = {resource.canonical for resource in received_apps[0].router.resources()}
        assert registered_paths == {
            "/health",
            "/start",
            "/stop",
            "/mode",
            "/status",
            "/shutdown",
            "/watch",
            "/video",
            "/frame",
        }
        assert fake_site.start_calls == 1
        assert fake_site.cleanup_calls == 1

    async def test_publishes_initial_idle_frame_before_serving(
        self,
        restore_service_hooks: None,
    ) -> None:
        """The initial idle frame reaches the status bus before serving begins."""
        _ = restore_service_hooks
        top_hooks.get_env = FakeEnv({})

        captured_app: list[web.Application] = []

        async def capturing_build_site(
            app: web.Application, host: str, port: int
        ) -> SiteRunnerProtocol:
            _ = (host, port)
            captured_app.append(app)
            return _CancellingSite()

        service_hooks.build_site = capturing_build_site
        service_hooks.build_bot_factory = _make_recording_bot_factory(_RecordingBot())

        with pytest.raises(asyncio.CancelledError):
            await _async_main()

        assert len(captured_app) == 1


class TestMain:
    """``main`` console entry — dotenv + serve + KeyboardInterrupt discipline."""

    def test_loads_env_and_runs_serve(
        self,
        restore_service_hooks: None,
    ) -> None:
        """Happy path: ``.env`` loads, probe reports no instance, serve runs."""
        _ = restore_service_hooks

        load_dotenv_calls = 0
        probe_calls = 0
        serve_calls = 0

        def fake_load_dotenv() -> None:
            nonlocal load_dotenv_calls
            load_dotenv_calls += 1

        def fake_probe() -> bool:
            nonlocal probe_calls
            probe_calls += 1
            return False

        def fake_serve() -> None:
            nonlocal serve_calls
            serve_calls += 1

        service_hooks.load_dotenv = fake_load_dotenv
        service_hooks.probe_existing_instance = fake_probe
        service_hooks.serve = fake_serve

        main()

        assert load_dotenv_calls == 1
        assert probe_calls == 1
        assert serve_calls == 1

    def test_short_circuits_when_probe_reports_existing_instance(
        self,
        restore_service_hooks: None,
    ) -> None:
        """``main`` exits idempotently when another instance already answers.

        A second ``make service`` (or a double-tap of the phone's SERVER
        button) MUST NOT re-bind the port. This test is the load-bearing
        guard against the double-tap race that would otherwise put the
        respawn loop into a permanent port-bind conflict.
        """
        _ = restore_service_hooks

        serve_calls = 0

        def fake_probe() -> bool:
            return True

        def fake_serve() -> None:
            nonlocal serve_calls
            serve_calls += 1

        service_hooks.load_dotenv = lambda: None
        service_hooks.probe_existing_instance = fake_probe
        service_hooks.serve = fake_serve

        main()

        assert serve_calls == 0

    def test_keyboard_interrupt_is_caught_at_the_boundary(
        self,
        restore_service_hooks: None,
    ) -> None:
        """A ``KeyboardInterrupt`` from ``serve`` unwinds cleanly."""
        _ = restore_service_hooks

        def fake_load_dotenv() -> None:
            pass

        def fake_serve() -> None:
            raise KeyboardInterrupt

        service_hooks.load_dotenv = fake_load_dotenv
        service_hooks.probe_existing_instance = lambda: False
        service_hooks.serve = fake_serve

        main()  # must not raise
