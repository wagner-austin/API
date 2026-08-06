"""Tests for :mod:`tankpit_bot.service.service_main` and its hooks.

Every non-pure operation the service main relies on lives in
:mod:`tankpit_bot.service._test_hooks` — the tests below install fake
implementations for the duration of each test and restore the real
defaults in teardown. No mocks, no monkey-patching of methods:
recording fakes match the Protocol structurally.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from pathlib import Path

import pytest
from aiohttp import web

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot.bot.base import Bot
from tankpit_bot.service import _test_hooks as service_hooks
from tankpit_bot.service._test_hooks import (
    SiteRunnerProtocol,
    _AiohttpSite,
    _real_build_bot_factory,
    _real_build_site,
    _real_load_dotenv,
    _real_serve,
)
from tankpit_bot.service.constants import SERVICE_HOST, SERVICE_PORT
from tankpit_bot.service.frame_bus import FrameBus, FrameBusProtocol
from tankpit_bot.service.mode_bridge import ModeBridge, ModeBridgeProtocol
from tankpit_bot.service.service_main import (
    _async_main,
    exit_when_idle,
    main,
    run_service_forever,
)
from tankpit_bot.service.session_runner import BotFactoryProtocol, RunnableBotProtocol
from tankpit_bot.service.status_bus import StatusBus, StatusBusProtocol
from tests.conftest import FakeEnv


@pytest.fixture()
def restore_service_hooks() -> Generator[None, None, None]:
    """Snapshot + restore ``service._test_hooks`` symbols around a test.

    Yields:
        Nothing — the fixture exists solely for its side-effect on the
        module-level hook symbols.
    """
    original_build_site = service_hooks.build_site
    original_load_dotenv = service_hooks.load_dotenv
    original_serve = service_hooks.serve
    original_build_bot_factory = service_hooks.build_bot_factory
    original_probe_existing_instance = service_hooks.probe_existing_instance
    try:
        yield
    finally:
        service_hooks.build_site = original_build_site
        service_hooks.load_dotenv = original_load_dotenv
        service_hooks.serve = original_serve
        service_hooks.build_bot_factory = original_build_bot_factory
        service_hooks.probe_existing_instance = original_probe_existing_instance


class _RecordingSite:
    """SiteRunnerProtocol stand-in that captures its lifecycle calls."""

    def __init__(self) -> None:
        """Initialise the call counters."""
        self.start_calls = 0
        self.cleanup_calls = 0

    async def start(self) -> None:
        """Record one ``start`` invocation."""
        self.start_calls += 1

    async def cleanup(self) -> None:
        """Record one ``cleanup`` invocation."""
        self.cleanup_calls += 1


class _CancellingSite:
    """SiteRunnerProtocol stand-in that cancels the caller on ``start``.

    Raises :class:`asyncio.CancelledError` synchronously from ``start``
    so :func:`run_service_forever`'s ``finally`` runs the site's
    ``cleanup`` before the exception propagates. Simpler and stricter-
    typed than reaching for ``asyncio.current_task().cancel``, which
    leaks ``Any`` through mypy's strict rules.
    """

    def __init__(self) -> None:
        """Initialise the call counters."""
        self.start_calls = 0
        self.cleanup_calls = 0

    async def start(self) -> None:
        """Cancel the calling task by raising :class:`asyncio.CancelledError`."""
        self.start_calls += 1
        raise asyncio.CancelledError

    async def cleanup(self) -> None:
        """Record one ``cleanup`` invocation."""
        self.cleanup_calls += 1


class _RecordingBot:
    """Runnable bot stand-in for the default-bot-factory test."""

    def __init__(self) -> None:
        """Initialise the call log."""
        self.runs: list[tuple[int, Path]] = []

    def run(
        self,
        *,
        session_seconds: int,
        session_kills: int = 0,
        stop_file_path: Path,
    ) -> None:
        """Record one ``run`` invocation."""
        self.runs.append((session_seconds, stop_file_path))


def _make_recording_bot_factory(
    recording_bot: _RecordingBot,
) -> service_hooks.BotFactoryBuilderProtocol:
    """Return a builder that ignores its args and hands back ``recording_bot``.

    Args:
        recording_bot: Bot the produced factory will return per call.

    Returns:
        A :class:`BotFactoryBuilderProtocol`-compatible callable.
    """

    def builder(target_url: str, *, headless: bool, prefer_account: bool) -> BotFactoryProtocol:
        _ = (target_url, headless, prefer_account)

        def factory(
            *,
            mode_bridge: ModeBridgeProtocol,
            status_bus: StatusBusProtocol,
            frame_bus: FrameBusProtocol,
        ) -> RunnableBotProtocol:
            _ = (mode_bridge, status_bus, frame_bus)
            return recording_bot

        return factory

    return builder


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


class TestRealHookImplementations:
    """The production defaults every service module boot uses."""

    def test_real_load_dotenv_calls_dotenv_module(self) -> None:
        """``_real_load_dotenv`` shells out to :mod:`dotenv` without error."""
        _real_load_dotenv()

    def test_real_serve_drives_async_main_under_asyncio_run(
        self,
        restore_service_hooks: None,
    ) -> None:
        """``_real_serve`` calls :func:`asyncio.run` around :func:`_async_main`."""
        _ = restore_service_hooks
        top_hooks.get_env = FakeEnv({})

        async def cancel_site(app: web.Application, host: str, port: int) -> SiteRunnerProtocol:
            _ = (app, host, port)
            return _CancellingSite()

        service_hooks.build_site = cancel_site
        service_hooks.build_bot_factory = _make_recording_bot_factory(_RecordingBot())

        with pytest.raises(asyncio.CancelledError):
            _real_serve()

    async def test_real_build_site_returns_a_site_runner(self) -> None:
        """``_real_build_site`` sets up the aiohttp AppRunner + TCPSite pair."""
        app = web.Application()

        site = await _real_build_site(app, "127.0.0.1", 0)

        # We do not call ``start`` — that would open a socket. But the
        # cleanup exercise proves the AppRunner setup ran.
        await site.cleanup()


class _IdleProbeRunner:
    """``is_running`` stub whose answer the test flips at will."""

    def __init__(self, *, running: bool = False) -> None:
        """Start with the given running answer."""
        self.running = running

    def start(self, *, session_seconds: int = 0, session_kills: int = 0) -> None:
        """Unused — the idle monitor never starts sessions."""
        raise AssertionError("exit_when_idle must never call start()")

    def request_stop(self) -> None:
        """Unused — the idle monitor never stops sessions."""
        raise AssertionError("exit_when_idle must never call request_stop()")

    def is_running(self) -> bool:
        return self.running


class TestExitWhenIdle:
    """Idle self-exit contract (2026-07-18 lifecycle pass)."""

    @pytest.mark.asyncio
    async def test_sets_stop_event_after_sustained_idleness(self) -> None:
        """No session + no subscriber long enough → shutdown signal fires."""
        stop_event = asyncio.Event()

        await asyncio.wait_for(
            exit_when_idle(
                _IdleProbeRunner(),
                StatusBus(),
                FrameBus(),
                stop_event,
                idle_exit_seconds=0.03,
                poll_seconds=0.01,
            ),
            timeout=2.0,
        )

        assert stop_event.is_set()

    @pytest.mark.asyncio
    async def test_an_sse_subscriber_keeps_the_service_alive(self) -> None:
        """A connected viewer resets the idle clock every poll."""
        stop_event = asyncio.Event()
        bus = StatusBus()
        subscriber = bus.subscribe()
        task = asyncio.create_task(
            exit_when_idle(
                _IdleProbeRunner(),
                bus,
                FrameBus(),
                stop_event,
                idle_exit_seconds=0.03,
                poll_seconds=0.01,
            )
        )

        await asyncio.sleep(0.15)
        assert not stop_event.is_set()

        # Viewer leaves → the clock finally runs out.
        bus.unsubscribe(subscriber)
        await asyncio.wait_for(task, timeout=2.0)
        assert stop_event.is_set()

    @pytest.mark.asyncio
    async def test_a_video_viewer_keeps_the_service_alive(self) -> None:
        """An open ``/video`` connection resets the idle clock every poll."""
        stop_event = asyncio.Event()
        frames = FrameBus()
        subscriber = frames.subscribe()
        task = asyncio.create_task(
            exit_when_idle(
                _IdleProbeRunner(),
                StatusBus(),
                frames,
                stop_event,
                idle_exit_seconds=0.03,
                poll_seconds=0.01,
            )
        )

        await asyncio.sleep(0.15)
        assert not stop_event.is_set()

        frames.unsubscribe(subscriber)
        await asyncio.wait_for(task, timeout=2.0)
        assert stop_event.is_set()

    @pytest.mark.asyncio
    async def test_a_running_session_keeps_the_service_alive(self) -> None:
        """An active session resets the idle clock every poll."""
        stop_event = asyncio.Event()
        runner = _IdleProbeRunner(running=True)
        task = asyncio.create_task(
            exit_when_idle(
                runner,
                StatusBus(),
                FrameBus(),
                stop_event,
                idle_exit_seconds=0.03,
                poll_seconds=0.01,
            )
        )

        await asyncio.sleep(0.15)
        assert not stop_event.is_set()

        runner.running = False
        await asyncio.wait_for(task, timeout=2.0)
        assert stop_event.is_set()

    @pytest.mark.asyncio
    async def test_non_positive_threshold_disables_the_monitor(self) -> None:
        """A 0 threshold returns immediately without ever firing the stop.

        The always-on deployment (2026-07-29): the SPA's tankpit
        video is served by this process, so the startup launcher
        disables the idle self-exit via
        ``TANKPIT_BOT_SERVICE_IDLE_EXIT_SECONDS=0``.
        """
        stop_event = asyncio.Event()

        await asyncio.wait_for(
            exit_when_idle(
                _IdleProbeRunner(),
                StatusBus(),
                FrameBus(),
                stop_event,
                idle_exit_seconds=0.0,
                poll_seconds=0.01,
            ),
            timeout=1.0,
        )

        assert not stop_event.is_set()

    @pytest.mark.asyncio
    async def test_returns_promptly_when_stop_event_already_set(self) -> None:
        """An externally-fired shutdown ends the monitor without a full wait."""
        stop_event = asyncio.Event()
        stop_event.set()

        await asyncio.wait_for(
            exit_when_idle(
                _IdleProbeRunner(),
                StatusBus(),
                FrameBus(),
                stop_event,
                idle_exit_seconds=10.0,
                poll_seconds=0.01,
            ),
            timeout=2.0,
        )
