"""Tests for service shutdown and instance probing."""

from __future__ import annotations

import asyncio

import pytest
from aiohttp import web

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot._test_hooks import _real_load_dotenv
from tankpit_bot.bus.status_bus import (
    StatusBus,
)
from tankpit_bot.service import _test_hooks as service_hooks
from tankpit_bot.service._test_hooks import (
    SiteRunnerProtocol,
    _real_build_site,
    _real_serve,
)
from tankpit_bot.service.service_main import (
    exit_when_idle,
)
from tests.conftest import FakeEnv
from tests.service._service_main_harness import (
    _CancellingSite,
    _IdleProbeRunner,
    _make_recording_bot_factory,
    _RecordingBot,
)


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
                stop_event,
                idle_exit_seconds=0.03,
                poll_seconds=0.01,
            )
        )

        await asyncio.sleep(0.15)
        assert not stop_event.is_set()

        # Viewer leaves → the clock finally runs out. Awaiting the
        # task is also what pins the shutdown exit: the monitor's
        # ``return`` after ``stop_event.set()`` is unobservable from
        # outside, so what matters is asserted here -- the task
        # completes rather than polling forever.
        bus.unsubscribe(subscriber)
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
                stop_event,
                idle_exit_seconds=10.0,
                poll_seconds=0.01,
            ),
            timeout=2.0,
        )
