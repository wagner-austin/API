"""Tests for the bot service's control routes.

Health, start, stop, shutdown, and mode. ``test_http_server.py`` was
1,080 lines; the two streaming surfaces are now siblings.
"""

from __future__ import annotations

import threading

import pytest
from aiohttp import (
    web,
)
from aiohttp.test_utils import (
    TestClient,
    TestServer,
)
from platform_core.json_utils import (
    dump_json_str,
)

from tankpit_bot.service.frame_bus import (
    FrameBus,
)
from tankpit_bot.service.http_server import make_app
from tankpit_bot.service.mode_bridge import ModeBridge
from tankpit_bot.service.status_bus import (
    StatusBus,
)
from tests.service._http_fixtures import (
    _noop_shutdown,
    _RecordingRunner,
)


class TestHealthRoute:
    """``GET /health`` contract."""

    @pytest.mark.asyncio
    async def test_returns_ok(self, client: TestClient[web.Request, web.Application]) -> None:
        """Health probe returns 200 with a stable body."""
        response = await client.get("/health")
        assert response.status == 200
        body = await response.text()
        assert body == "ok"


class TestStartRoute:
    """``POST /start`` contract."""

    @pytest.mark.asyncio
    async def test_accept_when_idle(
        self,
        client: TestClient[web.Request, web.Application],
        runner: _RecordingRunner,
    ) -> None:
        """A first start when idle returns 202 and enqueues the run."""
        on_start = threading.Event()
        runner._on_start = on_start

        response = await client.post("/start")

        assert response.status == 202
        # The executor thread runs start() asynchronously; wait for it.
        assert on_start.wait(timeout=1.0), "start() never invoked"
        assert runner.start_calls == 1
        assert runner.last_session_seconds == 0
        assert runner.last_session_kills == 0

    @pytest.mark.asyncio
    async def test_json_body_sets_session_bounds(
        self,
        client: TestClient[web.Request, web.Application],
        runner: _RecordingRunner,
    ) -> None:
        """``{"seconds": 2700, "kills": 30}`` reaches the runner verbatim."""
        on_start = threading.Event()
        runner._on_start = on_start

        payload: dict[str, int] = {"seconds": 2700, "kills": 30}
        response = await client.post("/start", json=payload)

        assert response.status == 202
        assert on_start.wait(timeout=1.0), "start() never invoked"
        assert runner.last_session_seconds == 2700
        assert runner.last_session_kills == 30

    @pytest.mark.asyncio
    async def test_partial_body_defaults_the_missing_bound(
        self,
        client: TestClient[web.Request, web.Application],
        runner: _RecordingRunner,
    ) -> None:
        """Either key may be omitted; the other defaults to unbounded."""
        on_start = threading.Event()
        runner._on_start = on_start

        payload: dict[str, int] = {"kills": 29}
        response = await client.post("/start", json=payload)

        assert response.status == 202
        assert on_start.wait(timeout=1.0), "start() never invoked"
        assert runner.last_session_seconds == 0
        assert runner.last_session_kills == 29

    @pytest.mark.asyncio
    async def test_bad_bounds_are_a_400_not_a_session(
        self,
        client: TestClient[web.Request, web.Application],
        runner: _RecordingRunner,
    ) -> None:
        """Non-integer and negative bounds reject without touching the runner."""
        bad_payload: dict[str, str] = {"kills": "many"}
        negative_payload: dict[str, int] = {"seconds": -5}
        bad_type = await client.post("/start", json=bad_payload)
        negative = await client.post("/start", json=negative_payload)

        assert bad_type.status == 400
        assert negative.status == 400
        assert runner.start_calls == 0

    @pytest.mark.asyncio
    async def test_conflict_when_already_running(
        self,
        bridge: ModeBridge,
        bus: StatusBus,
    ) -> None:
        """A start while already running returns 409 without touching runner.start."""
        runner = _RecordingRunner(already_running=True)
        app = make_app(runner, bridge, bus, FrameBus(), _noop_shutdown)
        server = TestServer(app)
        async with TestClient(server) as tc:
            response = await tc.post("/start")
            assert response.status == 409
            assert runner.start_calls == 0

    @pytest.mark.asyncio
    async def test_race_between_precheck_and_start_is_logged_not_500(
        self,
        bridge: ModeBridge,
        bus: StatusBus,
    ) -> None:
        """A ``SessionAlreadyRunningError`` from the executor is swallowed at WARN."""
        on_start = threading.Event()
        runner = _RecordingRunner(starts_reject=True, on_start=on_start)
        app = make_app(runner, bridge, bus, FrameBus(), _noop_shutdown)
        server = TestServer(app)
        async with TestClient(server) as tc:
            response = await tc.post("/start")

            assert response.status == 202  # pre-check passed
            assert on_start.wait(timeout=1.0)
            # The executor swallowed SessionAlreadyRunningError so
            # nothing surfaces as a client-visible error. Fine.
            assert runner.start_calls == 1


class TestStopRoute:
    """``POST /stop`` contract."""

    @pytest.mark.asyncio
    async def test_stop_calls_request_stop_and_returns_202(
        self,
        client: TestClient[web.Request, web.Application],
        runner: _RecordingRunner,
    ) -> None:
        """A stop request always returns 202 and forwards to the runner."""
        response = await client.post("/stop")
        assert response.status == 202
        assert runner.stop_calls == 1


class TestExecutorCrashLogging:
    """Session crashes on the executor thread must be logged, not swallowed."""

    def test_unexpected_crash_is_logged_and_reraised(self) -> None:
        """A non-rejection exception logs with traceback and re-raises.

        The executor future this wrapper runs under is never awaited,
        so without the explicit log a crash vanishes — observed
        2026-07-19: two ``POST /start`` → 202 with the session dead
        before its run log existed and no trace anywhere.
        """

        from tankpit_bot.service.http_server import _run_session_and_log_rejection

        class _CrashingRunner:
            def start(self, *, session_seconds: int = 0, session_kills: int = 0) -> None:
                _ = (session_seconds, session_kills)
                raise ValueError("simulated pre-log crash")

            def request_stop(self) -> None:
                raise AssertionError("never called")

            def is_running(self) -> bool:
                return False

        with pytest.raises(ValueError, match="simulated pre-log crash"):
            _run_session_and_log_rejection(_CrashingRunner(), 0, 0)


class TestShutdownRoute:
    """``POST /shutdown`` contract (2026-07-18 lifecycle pass)."""

    @pytest.mark.asyncio
    async def test_shutdown_stops_session_then_fires_signal(
        self,
        bridge: ModeBridge,
        bus: StatusBus,
    ) -> None:
        """The route requests session stop, fires ``on_shutdown``, returns 202."""
        runner = _RecordingRunner()
        fired: list[bool] = []
        app = make_app(runner, bridge, bus, FrameBus(), lambda: fired.append(True))
        server = TestServer(app)
        async with TestClient(server) as tc:
            response = await tc.post("/shutdown")
        assert response.status == 202
        assert runner.stop_calls == 1
        assert fired == [True]


class TestModeRoute:
    """``POST /mode`` contract."""

    @pytest.mark.asyncio
    async def test_valid_hunt_submits_to_bridge(
        self,
        client: TestClient[web.Request, web.Application],
        bridge: ModeBridge,
    ) -> None:
        """A valid ``HUNT`` payload lands on the bridge as ``"HUNT"``."""
        response = await client.post("/mode", data=dump_json_str({"manual_mode": "HUNT"}))
        assert response.status == 204
        assert bridge.drain() == "HUNT"

    @pytest.mark.asyncio
    async def test_valid_auto_submits_to_bridge(
        self,
        client: TestClient[web.Request, web.Application],
        bridge: ModeBridge,
    ) -> None:
        """A valid ``AUTO`` payload lands on the bridge as ``"AUTO"``."""
        response = await client.post("/mode", data=dump_json_str({"manual_mode": "AUTO"}))
        assert response.status == 204
        assert bridge.drain() == "AUTO"

    @pytest.mark.asyncio
    async def test_invalid_mode_string_surfaces_500(
        self,
        client: TestClient[web.Request, web.Application],
        bridge: ModeBridge,
    ) -> None:
        """An unknown mode literal raises out of the handler."""
        response = await client.post("/mode", data=dump_json_str({"manual_mode": "PATROL"}))
        assert response.status == 500
        # Nothing landed on the bridge — the raise happened during decode.
        assert bridge.drain() is None


class TestRunSessionAndLogRejectionHelper:
    """Unit contract for the executor-side runner invoker."""

    def test_normal_start_is_called(self) -> None:
        """A runner that starts cleanly is not intercepted."""
        from tankpit_bot.service.http_server import _run_session_and_log_rejection

        runner = _RecordingRunner()

        _run_session_and_log_rejection(runner, 0, 0)

        assert runner.start_calls == 1

    def test_session_already_running_error_is_swallowed(self) -> None:
        """The specific rejection error does not propagate to the caller."""
        from tankpit_bot.service.http_server import _run_session_and_log_rejection

        runner = _RecordingRunner(starts_reject=True)

        _run_session_and_log_rejection(runner, 0, 0)

        assert runner.start_calls == 1
