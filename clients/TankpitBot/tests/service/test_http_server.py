"""Tests for :mod:`tankpit_bot.service.http_server`.

Uses aiohttp's :class:`~aiohttp.test_utils.TestServer` and
:class:`~aiohttp.test_utils.TestClient` to drive the app against a
real loopback socket. The :class:`SessionRunner` is replaced with a
:class:`_RecordingRunner` that captures ``start`` / ``stop``
invocations, and the real :class:`ModeBridge` / :class:`StatusBus`
back the mode + status routes so cross-thread submits reach the
handlers the same way a live tick loop would.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator

import pytest
from aiohttp import ClientResponse, ClientSession, web
from aiohttp.test_utils import TestClient, TestServer
from platform_core.json_utils import dump_json_str, load_json_str, narrow_json_to_dict

from tankpit_bot.service.http_server import make_app
from tankpit_bot.service.mode_bridge import ModeBridge
from tankpit_bot.service.session_runner import SessionAlreadyRunningError
from tankpit_bot.service.status_bus import StatusBus, StatusSubscriberProtocol
from tankpit_bot.service.types import (
    SessionStatusDict,
    idle_session_status,
    make_live_stats,
    make_session_status,
)
from tankpit_bot.service.types_codecs import decode_session_status


class _RecordingRunner:
    """SessionRunner stand-in that records lifecycle calls."""

    def __init__(
        self,
        *,
        starts_reject: bool = False,
        already_running: bool = False,
        on_start: threading.Event | None = None,
    ) -> None:
        """Configure the fake runner's behaviour.

        Args:
            starts_reject: When True, ``start`` raises
                :class:`SessionAlreadyRunningError`. Simulates the
                race between two concurrent ``POST /start`` calls
                after the pre-check but before the state lock.
            already_running: When True, ``is_running`` returns True —
                the ``POST /start`` pre-check trips before ``start``
                is even called.
            on_start: Optional threading.Event set by ``start`` so the
                calling test can wait for the executor thread to run.
        """
        self.start_calls: int = 0
        self.stop_calls: int = 0
        self._starts_reject = starts_reject
        self._already_running = already_running
        self._on_start = on_start

    def is_running(self) -> bool:
        return self._already_running

    def start(self) -> None:
        self.start_calls += 1
        if self._on_start is not None:
            self._on_start.set()
        if self._starts_reject:
            raise SessionAlreadyRunningError("simulated race")

    def request_stop(self) -> None:
        self.stop_calls += 1


@pytest.fixture()
async def bus() -> StatusBus:
    """Fresh :class:`StatusBus` per test."""
    return StatusBus()


@pytest.fixture()
async def bridge() -> ModeBridge:
    """Fresh :class:`ModeBridge` per test."""
    return ModeBridge()


@pytest.fixture()
async def runner() -> _RecordingRunner:
    """Recording runner in the idle-not-rejecting default."""
    return _RecordingRunner()


@pytest.fixture()
async def client(
    runner: _RecordingRunner,
    bridge: ModeBridge,
    bus: StatusBus,
) -> AsyncIterator[TestClient[web.Request, web.Application]]:
    """aiohttp TestClient bound to a real app."""
    app = make_app(runner, bridge, bus)
    server = TestServer(app)
    async with TestClient(server) as tc:
        yield tc


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

    @pytest.mark.asyncio
    async def test_conflict_when_already_running(
        self,
        bridge: ModeBridge,
        bus: StatusBus,
    ) -> None:
        """A start while already running returns 409 without touching runner.start."""
        runner = _RecordingRunner(already_running=True)
        app = make_app(runner, bridge, bus)
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
        app = make_app(runner, bridge, bus)
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


class TestStatusRoute:
    """``GET /status`` SSE contract."""

    @pytest.mark.asyncio
    async def test_publishes_frames_to_the_client(
        self,
        client: TestClient[web.Request, web.Application],
        bus: StatusBus,
    ) -> None:
        """A frame the tick loop publishes reaches the client's SSE stream."""

        async def read_first_data_line(response: ClientResponse) -> str:
            async for raw in response.content:
                text = raw.decode()
                if text.startswith("data:"):
                    return text
            raise AssertionError("no data frame arrived on the SSE stream")

        stats = make_live_stats(kills=1, hits=2, misses=3, radars_used=4, teleports=5)
        frame: SessionStatusDict = make_session_status(
            running=True,
            manual_mode="HUNT",
            active_mode="HUNT",
            active_mode_state="ACQUIRE",
            session_started_ms=1000,
            tick_timestamp_ms=1200,
            stats=stats,
        )

        async with client.get("/status") as response:
            assert response.status == 200
            # Bus publish must land AFTER the subscriber is registered.
            # Give aiohttp one loop tick to reach ``subscribe``.
            await asyncio.sleep(0.05)
            bus.publish(frame)

            line = await asyncio.wait_for(read_first_data_line(response), timeout=2.0)

        assert line.startswith("data: ")
        payload = line[len("data: ") :].strip()
        parsed = narrow_json_to_dict(load_json_str(payload))
        decoded = decode_session_status(parsed)
        assert decoded == frame

    @pytest.mark.asyncio
    async def test_late_subscriber_sees_cached_frame(
        self,
        client: TestClient[web.Request, web.Application],
        bus: StatusBus,
    ) -> None:
        """A frame published before the SSE subscription is still delivered."""
        stats = make_live_stats(kills=9, hits=0, misses=0, radars_used=0, teleports=0)
        frame: SessionStatusDict = make_session_status(
            running=False,
            manual_mode="AUTO",
            active_mode="UNSET",
            active_mode_state="",
            session_started_ms=0,
            tick_timestamp_ms=17,
            stats=stats,
        )
        bus.publish(frame)

        async def wait_for_data(cs: ClientSession) -> str:
            async with cs.get(str(client.make_url("/status"))) as response:
                assert response.status == 200
                async for raw in response.content:
                    text = raw.decode()
                    if text.startswith("data:"):
                        return text
                raise AssertionError("no data frame arrived")

        session = client.session
        if session is None:
            raise AssertionError("TestClient.session should be a live ClientSession")
        line = await asyncio.wait_for(wait_for_data(session), timeout=2.0)
        payload = line[len("data: ") :].strip()
        parsed = narrow_json_to_dict(load_json_str(payload))
        decoded = decode_session_status(parsed)
        assert decoded == frame

    @pytest.mark.asyncio
    async def test_idle_status_round_trips_over_sse(
        self,
        client: TestClient[web.Request, web.Application],
        bus: StatusBus,
    ) -> None:
        """The idle-session helper produces a decodable SSE frame."""
        bus.publish(idle_session_status(tick_timestamp_ms=1))
        async with client.get("/status") as response:
            async for raw in response.content:
                text = raw.decode()
                if text.startswith("data:"):
                    payload = text[len("data: ") :].strip()
                    decoded = decode_session_status(narrow_json_to_dict(load_json_str(payload)))
                    assert decoded["running"] is False
                    assert decoded["active_mode"] == "UNSET"
                    return
        raise AssertionError("no idle status frame arrived")

    @pytest.mark.asyncio
    async def test_handler_closes_cleanly_when_bus_closes_subscriber(
        self,
        runner: _RecordingRunner,
        bridge: ModeBridge,
    ) -> None:
        """Server-side subscriber close drains, exits, and the handler returns.

        Uses an ``_ImmediateCloseBus`` whose subscriber is closed
        after the first ``next_frame`` returns ``None`` — the drain
        loop then hits ``if subscriber.closed: return``, unwinds
        through the finally, and the handler executes ``return
        response``. That final line is the coverage target — client
        cancellation on the other SSE tests raises inside the drain
        and skips it.
        """
        immediate_bus = _ImmediateCloseBus()
        app = make_app(runner, bridge, immediate_bus)
        server = TestServer(app)
        async with TestClient(server) as tc, tc.get("/status") as response:
            # Read the entire response body — the server closes
            # the SSE stream once the drain returns, so this is
            # bounded, not indefinite.
            await response.read()
            assert response.status == 200
        assert immediate_bus.unsubscribed >= 1


class _ImmediateCloseSubscriber:
    """Subscriber whose first ``next_frame`` returns ``None`` after closing.

    Ordering:

    1. Handler subscribes → ``closed`` is False (loop entered).
    2. Handler calls ``next_frame`` → subscriber closes itself and
       returns ``None``.
    3. Loop re-checks ``closed`` — True — and returns.
    """

    def __init__(self) -> None:
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def push(self, status: SessionStatusDict) -> None:
        _ = status

    def next_frame(self, timeout: float | None = None) -> SessionStatusDict | None:
        _ = timeout
        self._closed = True
        return None

    def close(self) -> None:
        self._closed = True


class _ImmediateCloseBus:
    """StatusBus stand-in whose subscriber closes on the first poll.

    Covers the ``return response`` line at the end of the ``status``
    handler without relying on client-side cancellation timing.
    """

    def __init__(self) -> None:
        self.unsubscribed: int = 0

    def publish(self, status: SessionStatusDict) -> None:
        _ = status

    def subscribe(self) -> StatusSubscriberProtocol:
        return _ImmediateCloseSubscriber()

    def unsubscribe(self, subscriber: StatusSubscriberProtocol) -> None:
        subscriber.close()
        self.unsubscribed += 1

    def subscriber_count(self) -> int:
        return 0


class _StubResponse:
    """Minimal ``aiohttp.web.StreamResponse`` stand-in for drain tests."""

    def __init__(self) -> None:
        self.writes: list[bytes] = []

    async def write(self, data: bytes) -> None:
        self.writes.append(data)


class _StubSubscriber:
    """Sync subscriber whose closed/next_frame behaviour tests configure."""

    def __init__(
        self,
        *,
        closed: bool = False,
        frames: list[SessionStatusDict | None] | None = None,
        close_after_none: bool = True,
    ) -> None:
        self._closed = closed
        self._frames: list[SessionStatusDict | None] = list(frames or [None])
        self._close_after_none = close_after_none

    @property
    def closed(self) -> bool:
        return self._closed

    def push(self, status: SessionStatusDict) -> None:
        self._frames.append(status)

    def next_frame(self, timeout: float | None = None) -> SessionStatusDict | None:
        _ = timeout
        if not self._frames:
            self._closed = True
            return None
        frame = self._frames.pop(0)
        if frame is None and self._close_after_none:
            self._closed = True
        return frame

    def close(self) -> None:
        self._closed = True


class _HeartbeatSubscriber:
    """Subscriber that returns None-then-closes so the drain hits heartbeat + exit.

    Kept as a separate class from :class:`_StubSubscriber` so the
    subscriber-state transitions don't need runtime overrides.
    """

    def __init__(self) -> None:
        self._closed = False
        self._calls = 0

    @property
    def closed(self) -> bool:
        return self._closed

    def push(self, status: SessionStatusDict) -> None:
        _ = status

    def next_frame(self, timeout: float | None = None) -> SessionStatusDict | None:
        _ = timeout
        self._calls += 1
        if self._calls >= 2:
            self._closed = True
        return None

    def close(self) -> None:
        self._closed = True


class _StubBus:
    """Status bus stand-in that hands back a preconfigured subscriber."""

    def __init__(self, subscriber: StatusSubscriberProtocol) -> None:
        self._subscriber = subscriber
        self.unsubscribed = 0

    def publish(self, status: SessionStatusDict) -> None:
        _ = status

    def subscribe(self) -> StatusSubscriberProtocol:
        return self._subscriber

    def unsubscribe(self, subscriber: StatusSubscriberProtocol) -> None:
        _ = subscriber
        self.unsubscribed += 1

    def subscriber_count(self) -> int:
        return 1


class TestDrainStatusBusToResponseHelper:
    """Unit contract for the SSE-drain helper.

    Exercises the two internal branches ``TestStatusRoute`` cannot
    hit through the wire in a reasonable time: the ``subscriber.closed``
    early-return + the heartbeat write when ``next_frame`` returns
    ``None``. Everything runs against lightweight stubs so the tests
    never open a socket or wait a heartbeat interval.
    """

    @pytest.mark.asyncio
    async def test_close_before_first_frame_returns_without_writing(self) -> None:
        """An already-closed subscriber exits the loop before any write."""
        from tankpit_bot.service.http_server import _drain_status_bus_to_response

        subscriber = _StubSubscriber(closed=True)
        bus = _StubBus(subscriber)
        response = _StubResponse()

        await _drain_status_bus_to_response(bus, response)

        assert response.writes == []
        assert bus.unsubscribed == 1

    @pytest.mark.asyncio
    async def test_close_between_wait_and_recheck_returns_before_write(self) -> None:
        """A close observed AFTER ``next_frame`` exits the loop pre-write."""
        from tankpit_bot.service.http_server import _drain_status_bus_to_response

        # Subscriber enters loop open (loop condition True), returns
        # a frame, then closes so the post-wait ``if subscriber.closed``
        # check trips before the write branch.
        subscriber = _StubSubscriber(closed=False, frames=[None], close_after_none=True)
        bus = _StubBus(subscriber)
        response = _StubResponse()

        await _drain_status_bus_to_response(bus, response)

        assert response.writes == []
        assert bus.unsubscribed == 1

    @pytest.mark.asyncio
    async def test_heartbeat_written_when_next_frame_times_out(self) -> None:
        """A ``None`` from ``next_frame`` triggers a heartbeat comment.

        The subscriber returns ``None`` on the first ``next_frame`` call
        (heartbeat branch) then closes so the loop exits. Uses
        ``_HeartbeatSubscriber`` — a dedicated stub whose state
        transitions serialise cleanly for coverage.
        """
        from tankpit_bot.service.http_server import _drain_status_bus_to_response

        subscriber = _HeartbeatSubscriber()
        bus = _StubBus(subscriber)
        response = _StubResponse()

        await _drain_status_bus_to_response(bus, response)

        assert response.writes == [b": heartbeat\n\n"]
        assert bus.unsubscribed == 1

    @pytest.mark.asyncio
    async def test_frame_writes_then_close_exits_loop(self) -> None:
        """A published frame emits a ``data:`` line, then close halts the loop."""
        from tankpit_bot.service.http_server import _drain_status_bus_to_response
        from tankpit_bot.service.types import idle_session_status

        frame = idle_session_status(tick_timestamp_ms=42)
        subscriber = _StubSubscriber(closed=False, frames=[frame])
        bus = _StubBus(subscriber)
        response = _StubResponse()

        await _drain_status_bus_to_response(bus, response)

        assert len(response.writes) == 1
        assert response.writes[0].startswith(b"data: ")
        assert bus.unsubscribed == 1


class TestRunSessionAndLogRejectionHelper:
    """Unit contract for the executor-side runner invoker."""

    def test_normal_start_is_called(self) -> None:
        """A runner that starts cleanly is not intercepted."""
        from tankpit_bot.service.http_server import _run_session_and_log_rejection

        runner = _RecordingRunner()

        _run_session_and_log_rejection(runner)

        assert runner.start_calls == 1

    def test_session_already_running_error_is_swallowed(self) -> None:
        """The specific rejection error does not propagate to the caller."""
        from tankpit_bot.service.http_server import _run_session_and_log_rejection

        runner = _RecordingRunner(starts_reject=True)

        _run_session_and_log_rejection(runner)

        assert runner.start_calls == 1
