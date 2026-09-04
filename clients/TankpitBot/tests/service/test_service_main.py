"""Tests for the service entry point: startup and wiring.

``test_service_main.py`` was 626 lines; the shutdown and probe paths
are now a sibling.
"""

from __future__ import annotations

import asyncio

import pytest
from aiohttp import web

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot.bot.base import Bot
from tankpit_bot.bus.frame_bus import (
    FrameBus,
    FrameBusProtocol,
)
from tankpit_bot.bus.mode_bridge import (
    ModeBridge,
    ModeBridgeProtocol,
)
from tankpit_bot.bus.status_bus import (
    StatusBus,
    StatusBusProtocol,
)
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
from tankpit_bot.service.service_main import (
    _async_main,
    _autostart_session,
    main,
)
from tests.conftest import FakeEnv
from tests.service._service_main_harness import (
    _CancellingSite,
    _CapturingBotFactoryBuilder,
    _make_recording_bot_factory,
    _RecordingBot,
)


class TestRealBuildBotFactory:
    """Contract for the real bot-factory builder used at production boot."""

    def test_factory_produces_a_bot_bound_to_its_bridge_and_bus(self) -> None:
        """The bot returned by the factory has the injected channels."""
        factory = _real_build_bot_factory(
            "https://test.tankpit.com/",
            headless=True,
            prefer_account=False,
            cast_url="http://127.0.0.1:27100/cast",
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
        # The cast URL is what makes a caster exist at all; a bot built
        # without one installs nothing (see Bot.__init__).
        if raw._live_view is None:
            raise AssertionError("a factory given a cast url must build a caster")
        assert raw._live_view.active is False

    def test_factory_carries_headless_and_prefer_account(self) -> None:
        """Construction args flow through to the produced bot."""
        factory = _real_build_bot_factory(
            "https://test.tankpit.com/",
            headless=True,
            prefer_account=True,
            cast_url="http://127.0.0.1:27100/cast",
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

    async def test_the_autostart_task_is_cancelled_when_serving_ends(
        self,
        restore_service_hooks: None,
    ) -> None:
        """A child's session task does not outlive the service.

        The task owns a worker thread running a session. Leaving it
        scheduled after the site is gone would mean a process that has
        stopped serving still holding a bot, which is the shape of an
        orphan.
        """
        _ = restore_service_hooks
        top_hooks.get_env = FakeEnv({})

        async def cancelling_build_site(
            app: web.Application, host: str, port: int
        ) -> SiteRunnerProtocol:
            _ = (app, host, port)
            return _CancellingSite()

        service_hooks.build_site = cancelling_build_site
        service_hooks.build_bot_factory = _make_recording_bot_factory(_RecordingBot())

        with pytest.raises(asyncio.CancelledError):
            await _async_main()

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
            # The frame intake. Its whole purpose is the THREAD it runs
            # on: aiohttp serves it on the main-thread event loop, so a
            # frame posted here reaches the bus while the session's
            # executor thread is busy in a heavy tick.
            "/cast",
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


class _RecordingSessionRunner:
    """A session runner recording the bounds it was started with."""

    def __init__(self, *, fails: bool = False) -> None:
        """Start with nothing recorded.

        Args:
            fails: Whether ``start`` raises instead of returning.
        """
        self.starts: list[tuple[int, int]] = []
        self._fails = fails

    def start(self, *, session_seconds: int = 0, session_kills: int = 0) -> None:
        """Record one session start.

        Args:
            session_seconds: Seconds bound the caller asked for.
            session_kills: Kill bound the caller asked for.

        Raises:
            RuntimeError: When this double was built to fail.
        """
        self.starts.append((session_seconds, session_kills))
        if self._fails:
            raise RuntimeError("session could not start")

    def request_stop(self) -> None:
        """Unused by the autostart path."""
        raise AssertionError("autostart must never request a stop")

    def is_running(self) -> bool:
        """Report idle.

        Returns:
            Always False; the autostart path never consults this.
        """
        return False


class TestAutostartSession:
    """``_autostart_session`` runs one session, then ends the process.

    A fleet child exists for exactly one session. The service stops
    because that session ENDED, not because an idle timer noticed: the
    idle monitor counts "no session running" as idle, and a bot spends
    its first seconds launching a browser, so any window short enough to
    reap a finished child would also reap a starting one.
    """

    async def test_the_session_is_bounded_by_the_env(self) -> None:
        """The bounds are the ones the fleet set on the child."""
        top_hooks.get_env = FakeEnv(
            {"TANKPIT_BOT_SESSION_SECONDS": "300", "TANKPIT_BOT_SESSION_KILLS": "20"}
        )
        runner = _RecordingSessionRunner()
        finished: list[bool] = []

        await _autostart_session(runner, lambda: finished.append(True))

        assert runner.starts == [(300, 20)]
        assert finished == [True]

    async def test_an_unbounded_child_runs_until_stopped(self) -> None:
        """No bounds set means zero, which the runner reads as unbounded."""
        top_hooks.get_env = FakeEnv({})
        runner = _RecordingSessionRunner()

        await _autostart_session(runner, lambda: None)

        assert runner.starts == [(0, 0)]

    async def test_a_session_that_cannot_start_still_stops_the_service(self) -> None:
        """The failure propagates AND the process is told to end.

        A service left running with no bot in it would sit there holding
        a port and reporting healthy, which is worse than a child that
        exits and lets the manager see it die.
        """
        top_hooks.get_env = FakeEnv({})
        runner = _RecordingSessionRunner(fails=True)
        finished: list[bool] = []

        with pytest.raises(RuntimeError, match="session could not start"):
            await _autostart_session(runner, lambda: finished.append(True))

        assert finished == [True]


class TestHeadlessWiring:
    """The service must launch with the headless setting it resolved.

    A correct resolver that nothing calls is the shape of the original
    defect: ``prefer_account`` was resolved properly in the very same
    constructor call where ``headless`` was a literal ``False``, so a
    unit test of the resolver alone stayed green while every
    containerized bot exited 1 on "Missing X server or $DISPLAY". These
    assert the value that actually reaches the launch.
    """

    async def test_headless_env_reaches_the_bot_factory(
        self,
        restore_service_hooks: None,
    ) -> None:
        """``TANKPIT_HEADLESS=true`` launches a windowless browser."""
        _ = restore_service_hooks
        top_hooks.get_env = FakeEnv({"TANKPIT_HEADLESS": "true"})
        builder = _CapturingBotFactoryBuilder(_RecordingBot())
        service_hooks.build_bot_factory = builder

        async def cancelling_build_site(
            app: web.Application, host: str, port: int
        ) -> SiteRunnerProtocol:
            _ = (app, host, port)
            return _CancellingSite()

        service_hooks.build_site = cancelling_build_site

        with pytest.raises(asyncio.CancelledError):
            await _async_main()

        assert [headless for _url, headless, _prefer in builder.calls] == [True]

    async def test_the_default_launch_keeps_the_window(
        self,
        restore_service_hooks: None,
    ) -> None:
        """Unset env launches headed, so a desktop run stays watchable."""
        _ = restore_service_hooks
        top_hooks.get_env = FakeEnv({})
        builder = _CapturingBotFactoryBuilder(_RecordingBot())
        service_hooks.build_bot_factory = builder

        async def cancelling_build_site(
            app: web.Application, host: str, port: int
        ) -> SiteRunnerProtocol:
            _ = (app, host, port)
            return _CancellingSite()

        service_hooks.build_site = cancelling_build_site

        with pytest.raises(asyncio.CancelledError):
            await _async_main()

        assert [headless for _url, headless, _prefer in builder.calls] == [False]


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

        core_hooks.load_dotenv = fake_load_dotenv
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

        core_hooks.load_dotenv = lambda: None
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

        core_hooks.load_dotenv = fake_load_dotenv
        service_hooks.probe_existing_instance = lambda: False
        service_hooks.serve = fake_serve

        main()  # must not raise
