"""The fleet manager's lifecycle: boot, adopt, drain, exit.

Replaces the old ``test_fleet_shutdown.py``, whose whole subject was
that Ctrl+C printed a log line instead of a traceback. Ctrl+C now
means something else entirely: it starts a drain, and the manager
keeps serving until its last bot has torn down, because exiting while
a tank is still playing is how the fleet used to strand them.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

import pytest
from aiohttp import web

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot.service import _test_hooks as service_hooks
from tankpit_bot.service._test_hooks import SiteRunnerProtocol, _real_serve_fleet
from tankpit_bot.service.fleet import (
    FLEET_HOST_DEFAULT,
    _async_main,
    drain_on_interrupt,
    exit_when_drained,
    main,
    resolve_fleet_host,
)
from tankpit_bot.service.fleet_manager import FleetManager
from tests.conftest import FakeEnv
from tests.service._fleet_fixtures import (
    FakeRecordStore,
    _FakeSpawner,
    _restore_account_hooks,
    _with_configured_accounts,
)
from tests.service._service_main_harness import _CancellingSite, _RecordingSite


def _ignore_handler(on_interrupt: Callable[[], None]) -> None:
    """Accept a signal handler and do nothing with it.

    Args:
        on_interrupt: The handler the manager wanted to install.

    Returns:
        None.
    """
    _ = on_interrupt


def _spawn_two(spawner: _FakeSpawner) -> FleetManager:
    """Build a manager holding two live bots.

    Args:
        spawner: The recording spawner fixture.

    Returns:
        The manager, with ``alpha`` and ``bravo`` registered.
    """
    _ = spawner
    originals = _with_configured_accounts()
    try:
        manager = FleetManager()
        manager.spawn(instance="alpha", account="artax", kills=0, seconds=0)
        manager.spawn(instance="bravo", account="second", kills=0, seconds=0)
    finally:
        _restore_account_hooks(originals)
    return manager


class TestDrain:
    """Asking every bot to stop, and knowing when they have."""

    def test_drain_writes_a_stop_sentinel_for_every_live_bot(
        self,
        spawner: _FakeSpawner,
    ) -> None:
        """The drain is the same sentinel a single stop writes, for all."""
        written: list[str] = []

        def record_write(path: Path, content: str) -> None:
            _ = content
            written.append(str(path).replace("\\", "/"))

        manager = _spawn_two(spawner)
        original_write = top_hooks.write_text
        top_hooks.write_text = record_write
        try:
            draining = manager.request_drain()
        finally:
            top_hooks.write_text = original_write

        assert draining == ["alpha", "bravo"]
        assert manager.draining() is True
        assert written == ["runs/bot/alpha/STOP", "runs/bot/bravo/STOP"]

    def test_a_dead_bot_is_not_asked_to_stop_again(
        self,
        spawner: _FakeSpawner,
    ) -> None:
        """Only live bots are drained; a finished one needs nothing."""
        manager = _spawn_two(spawner)
        spawner.processes[0].returncode = 0

        original_write = top_hooks.write_text
        top_hooks.write_text = lambda path, content: None
        try:
            draining = manager.request_drain()
        finally:
            top_hooks.write_text = original_write

        assert draining == ["bravo"]
        assert manager.live_instances() == ["bravo"]

    def test_draining_is_false_until_asked(self, spawner: _FakeSpawner) -> None:
        """A manager just serving is not draining."""
        assert _spawn_two(spawner).draining() is False


class TestExitWhenDrained:
    """The monitor that ends the process once the last bot lands."""

    async def test_waits_while_a_bot_is_still_tearing_down(
        self,
        spawner: _FakeSpawner,
    ) -> None:
        """The stop event stays clear until every bot has exited."""
        manager = _spawn_two(spawner)
        original_write = top_hooks.write_text
        top_hooks.write_text = lambda path, content: None
        try:
            manager.request_drain()
        finally:
            top_hooks.write_text = original_write
        stop_event = asyncio.Event()

        monitor = asyncio.create_task(exit_when_drained(manager, stop_event, poll_seconds=0.0))
        for _ in range(5):
            await asyncio.sleep(0)
        still_waiting = not stop_event.is_set()

        spawner.processes[0].returncode = 0
        spawner.processes[1].returncode = 0
        await monitor

        assert still_waiting is True
        assert stop_event.is_set() is True

    async def test_never_stops_a_manager_that_was_not_asked_to_drain(
        self,
        spawner: _FakeSpawner,
    ) -> None:
        """No drain requested means the monitor never sets the event."""
        manager = _spawn_two(spawner)
        spawner.processes[0].returncode = 0
        spawner.processes[1].returncode = 0
        stop_event = asyncio.Event()

        monitor = asyncio.create_task(exit_when_drained(manager, stop_event, poll_seconds=0.0))
        for _ in range(5):
            await asyncio.sleep(0)
        monitor.cancel()

        assert stop_event.is_set() is False

    async def test_a_pre_set_stop_event_ends_the_monitor(
        self,
        spawner: _FakeSpawner,
    ) -> None:
        """An already-stopping manager needs no further monitoring."""
        manager = _spawn_two(spawner)
        stop_event = asyncio.Event()
        stop_event.set()

        await exit_when_drained(manager, stop_event, poll_seconds=0.0)

        assert manager.draining() is False


class TestDrainOnInterrupt:
    """Ctrl+C starts the drain instead of abandoning the bots."""

    def test_first_interrupt_requests_the_drain(self, spawner: _FakeSpawner) -> None:
        """One interrupt asks every live bot to stop."""
        manager = _spawn_two(spawner)
        original_write = top_hooks.write_text
        top_hooks.write_text = lambda path, content: None
        try:
            drain_on_interrupt(manager)()
        finally:
            top_hooks.write_text = original_write

        assert manager.draining() is True

    def test_a_second_interrupt_does_not_restart_the_drain(
        self,
        spawner: _FakeSpawner,
    ) -> None:
        """Interrupting a draining manager reports, it does not re-ask.

        Re-writing the sentinels would be harmless, but the operator
        needs to know the manager is already on its way out and what
        it is waiting for.
        """
        manager = _spawn_two(spawner)
        writes: list[str] = []

        def count_write(path: Path, content: str) -> None:
            _ = content
            writes.append(str(path))

        original_write = top_hooks.write_text
        top_hooks.write_text = count_write
        try:
            handle = drain_on_interrupt(manager)
            handle()
            after_first = len(writes)
            handle()
        finally:
            top_hooks.write_text = original_write

        assert after_first == 2
        assert len(writes) == 2


class TestAsyncMain:
    """Composition: adopt, route, serve, and hand the drain a monitor."""

    async def test_serves_the_fleet_routes_on_the_resolved_port(
        self,
        restore_service_hooks: None,
        records: FakeRecordStore,
        spawner: _FakeSpawner,
    ) -> None:
        """``_async_main`` builds the site on the configured port."""
        _ = (restore_service_hooks, records, spawner)
        top_hooks.get_env = FakeEnv({"TANKPIT_FLEET_PORT": "27311"})
        served: list[tuple[web.Application, str, int]] = []
        handlers: list[Callable[[], None]] = []

        def capture_handler(on_interrupt: Callable[[], None]) -> None:
            handlers.append(on_interrupt)

        async def fake_build_site(app: web.Application, host: str, port: int) -> SiteRunnerProtocol:
            served.append((app, host, port))
            return _CancellingSite()

        service_hooks.build_site = fake_build_site
        core_hooks.install_signal_handlers = capture_handler

        with pytest.raises(asyncio.CancelledError):
            await _async_main()

        assert len(served) == 1
        app, host, port = served[0]
        assert (host, port) == (FLEET_HOST_DEFAULT, 27311)
        assert len(handlers) == 1
        canonical = {resource.canonical for resource in app.router.resources()}
        assert canonical == {
            "/",
            "/accounts",
            "/rooms",
            "/troops",
            "/doctrines",
            "/tanks",
            "/bots",
            "/bots/{instance}/stats",
            "/bots/{instance}/hud",
            "/bots/{instance}/activity",
            "/bots/{instance}/video",
            "/bots/{instance}/stop",
            "/bots/{instance}/restart",
            "/bots/{instance}",
            "/shutdown",
        }

    async def test_the_drain_monitor_is_cancelled_when_serving_ends(
        self,
        restore_service_hooks: None,
        records: FakeRecordStore,
        spawner: _FakeSpawner,
    ) -> None:
        """A stopped manager leaves no monitor task behind."""
        _ = (restore_service_hooks, records, spawner)
        top_hooks.get_env = FakeEnv({})
        site = _RecordingSite()

        async def fake_build_site(app: web.Application, host: str, port: int) -> SiteRunnerProtocol:
            _ = (app, host, port)
            return site

        service_hooks.build_site = fake_build_site
        core_hooks.install_signal_handlers = _ignore_handler

        served = asyncio.create_task(_async_main())
        await asyncio.sleep(0)
        served.cancel()
        with pytest.raises(asyncio.CancelledError):
            await served
        await asyncio.sleep(0)

        assert site.start_calls == 1
        assert site.cleanup_calls == 1


class TestRealServeFleet:
    """The production default ``fleet.main`` drives."""

    def test_real_serve_fleet_runs_async_main(
        self,
        restore_service_hooks: None,
        records: FakeRecordStore,
        spawner: _FakeSpawner,
    ) -> None:
        """``_real_serve_fleet`` drives ``_async_main`` under asyncio.run."""
        _ = (restore_service_hooks, records, spawner)
        top_hooks.get_env = FakeEnv({})

        async def cancel_site(app: web.Application, host: str, port: int) -> SiteRunnerProtocol:
            _ = (app, host, port)
            return _CancellingSite()

        service_hooks.build_site = cancel_site
        core_hooks.install_signal_handlers = _ignore_handler

        with pytest.raises(asyncio.CancelledError):
            _real_serve_fleet()


class TestMain:
    """The ``tankpit-fleet`` console entry point."""

    def test_loads_env_then_serves(self, restore_service_hooks: None) -> None:
        """Happy path: dotenv loads, then the fleet serves."""
        _ = restore_service_hooks
        calls: list[str] = []

        core_hooks.load_dotenv = lambda: calls.append("dotenv")
        service_hooks.serve_fleet = lambda: calls.append("serve")

        main()

        assert calls == ["dotenv", "serve"]

    def test_an_interrupt_before_serving_is_a_log_line_not_a_traceback(
        self,
        restore_service_hooks: None,
    ) -> None:
        """Ctrl+C in the sliver before the drain handler exists returns."""
        _ = restore_service_hooks

        def interrupted() -> None:
            raise KeyboardInterrupt

        core_hooks.load_dotenv = lambda: None
        service_hooks.serve_fleet = interrupted

        main()


def test_fleet_host_resolves_from_the_environment(fake_env: FakeEnv) -> None:
    """The container's 0.0.0.0 bind is an explicit env choice.

    Unset and empty both keep the loopback default — off the
    container, nothing should ever widen the bind by accident.
    """
    assert resolve_fleet_host() == FLEET_HOST_DEFAULT
    fake_env.set("TANKPIT_FLEET_HOST", "")
    assert resolve_fleet_host() == FLEET_HOST_DEFAULT
    fake_env.set("TANKPIT_FLEET_HOST", "0.0.0.0")
    assert resolve_fleet_host() == "0.0.0.0"


def test_a_child_that_exits_before_the_identity_read_leaves_no_record(
    records: FakeRecordStore,
    spawner: _FakeSpawner,
) -> None:
    """A spawn whose child died instantly records nothing to adopt.

    The identity seam answers None for a pid nothing runs under, and
    the manager treats that as "the run is already over" rather than
    a failure to record. Until 2026-09-03 this branch was covered only
    when a REAL spawned child happened to exit before the identity
    read — the same nondeterminism class as the kill-cleanup race —
    so the gate flickered between 100% and a miss with the timing.
    """
    _ = spawner
    records.identities.clear()
    manager = FleetManager()

    row = manager.spawn(instance="alpha", account="", kills=0, seconds=60)

    assert row["instance"] == "alpha"
    assert records.files == {}
