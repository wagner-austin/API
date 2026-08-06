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

from tankpit_bot.service.frame_bus import FrameBus, FrameSubscriberProtocol
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
        self.last_session_seconds: int = -1
        self.last_session_kills: int = -1
        self._starts_reject = starts_reject
        self._already_running = already_running
        self._on_start = on_start

    def is_running(self) -> bool:
        return self._already_running

    def start(self, *, session_seconds: int = 0, session_kills: int = 0) -> None:
        self.start_calls += 1
        self.last_session_seconds = session_seconds
        self.last_session_kills = session_kills
        if self._on_start is not None:
            self._on_start.set()
        if self._starts_reject:
            raise SessionAlreadyRunningError("simulated race")

    def request_stop(self) -> None:
        self.stop_calls += 1


def _noop_shutdown() -> None:
    """Placeholder ``on_shutdown`` for routes that never fire it."""


@pytest.fixture()
async def bus() -> StatusBus:
    """Fresh :class:`StatusBus` per test."""
    return StatusBus()


@pytest.fixture()
async def fbus() -> FrameBus:
    """Fresh :class:`FrameBus` per test."""
    return FrameBus()


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
    fbus: FrameBus,
) -> AsyncIterator[TestClient[web.Request, web.Application]]:
    """aiohttp TestClient bound to a real app."""
    app = make_app(runner, bridge, bus, fbus, _noop_shutdown)
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
        app = make_app(runner, bridge, immediate_bus, FrameBus(), _noop_shutdown)
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


class TestWatchRoute:
    """``GET /watch`` contract (2026-07-28 fiesta-free watch page)."""

    @pytest.mark.asyncio
    async def test_serves_the_watch_page(
        self, client: TestClient[web.Request, web.Application]
    ) -> None:
        """The page arrives as HTML with the relative asset URLs intact."""
        response = await client.get("/watch")
        assert response.status == 200
        assert response.content_type == "text/html"
        body = await response.text()
        # Relative (not absolute) route references are the load-bearing
        # detail: they keep the page working both direct (:27100/watch)
        # and behind nginx's /api/tankbot/ prefix strip.
        assert "video?t=" in body
        assert 'new EventSource("status")' in body
        assert 'post("start")' in body
        assert 'post("stop")' in body


class TestFrameRoute:
    """``GET /frame`` snapshot contract."""

    @pytest.mark.asyncio
    async def test_returns_cached_frame_as_jpeg(
        self,
        client: TestClient[web.Request, web.Application],
        fbus: FrameBus,
    ) -> None:
        """A published frame is served back with the image content type."""
        fbus.publish(b"\xff\xd8jpegbytes")
        response = await client.get("/frame")
        assert response.status == 200
        assert response.content_type == "image/jpeg"
        body = await response.read()
        assert body == b"\xff\xd8jpegbytes"

    @pytest.mark.asyncio
    async def test_404_when_no_frame_ever_published(
        self,
        runner: _RecordingRunner,
        bridge: ModeBridge,
        bus: StatusBus,
    ) -> None:
        """An empty bus yields 404 once the demand wait times out."""
        empty_bus = _ImmediateTimeoutFrameBus()
        app = make_app(runner, bridge, bus, empty_bus, _noop_shutdown)
        server = TestServer(app)
        async with TestClient(server) as tc:
            response = await tc.get("/frame")
            assert response.status == 404
        assert empty_bus.unsubscribed == 1


class TestVideoRoute:
    """``GET /video`` MJPEG contract."""

    @pytest.mark.asyncio
    async def test_streams_published_frames_as_multipart(
        self,
        client: TestClient[web.Request, web.Application],
        fbus: FrameBus,
    ) -> None:
        """A published frame reaches the client inside a multipart part."""
        async with client.get("/video") as response:
            assert response.status == 200
            assert response.content_type == "multipart/x-mixed-replace"
            await asyncio.sleep(0.05)
            fbus.publish(b"\xff\xd8frame-one")
            chunk = await asyncio.wait_for(response.content.read(1024), timeout=2.0)
        assert b"--tankpitbotframe" in chunk
        assert b"Content-Type: image/jpeg" in chunk
        assert b"\xff\xd8frame-one" in chunk

    @pytest.mark.asyncio
    async def test_handler_closes_cleanly_when_bus_closes_subscriber(
        self,
        runner: _RecordingRunner,
        bridge: ModeBridge,
        bus: StatusBus,
    ) -> None:
        """Server-side subscriber close drains, exits, and the handler returns.

        Uses ``_ImmediateTimeoutFrameBus`` — its subscriber times out
        instantly and the drain closes it via the ``frames``-exhausted
        path — so the ``return response`` line at the end of the
        ``video`` handler executes. Client cancellation on the
        streaming test above raises inside the drain and skips it.
        """
        closing_bus = _ImmediateCloseFrameBus()
        app = make_app(runner, bridge, bus, closing_bus, _noop_shutdown)
        server = TestServer(app)
        async with TestClient(server) as tc, tc.get("/video") as response:
            await response.read()
            assert response.status == 200
        assert closing_bus.unsubscribed >= 1


class _ImmediateCloseFrameSubscriber:
    """Frame subscriber whose first ``next_frame`` closes itself, returns None.

    Ordering mirrors ``_ImmediateCloseSubscriber``: the ``video``
    handler's drain enters its loop, the first wait closes the
    subscriber, and the post-wait ``closed`` check exits the drain so
    the handler's final ``return response`` executes.
    """

    def __init__(self) -> None:
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def push(self, frame: bytes) -> None:
        _ = frame

    def next_frame(self, timeout: float | None = None) -> bytes | None:
        _ = timeout
        self._closed = True
        return None

    def close(self) -> None:
        self._closed = True


class _ImmediateCloseFrameBus:
    """FrameBus stand-in whose subscriber closes on the first poll."""

    def __init__(self) -> None:
        self.unsubscribed: int = 0

    def publish(self, frame: bytes) -> None:
        _ = frame

    def subscribe(self) -> FrameSubscriberProtocol:
        return _ImmediateCloseFrameSubscriber()

    def unsubscribe(self, subscriber: FrameSubscriberProtocol) -> None:
        subscriber.close()
        self.unsubscribed += 1

    def subscriber_count(self) -> int:
        return 0

    def latest(self) -> bytes | None:
        return None


class _ImmediateTimeoutFrameSubscriber:
    """Frame subscriber whose waits time out instantly and never yield."""

    def __init__(self) -> None:
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def push(self, frame: bytes) -> None:
        _ = frame

    def next_frame(self, timeout: float | None = None) -> bytes | None:
        _ = timeout
        return None

    def close(self) -> None:
        self._closed = True


class _ImmediateTimeoutFrameBus:
    """FrameBus stand-in with no cache whose subscriber always times out."""

    def __init__(self) -> None:
        self.unsubscribed: int = 0

    def publish(self, frame: bytes) -> None:
        _ = frame

    def subscribe(self) -> FrameSubscriberProtocol:
        return _ImmediateTimeoutFrameSubscriber()

    def unsubscribe(self, subscriber: FrameSubscriberProtocol) -> None:
        subscriber.close()
        self.unsubscribed += 1

    def subscriber_count(self) -> int:
        return 0

    def latest(self) -> bytes | None:
        return None


class _StubFrameSubscriber:
    """Sync frame subscriber whose frame sequence tests configure.

    A ``None`` entry models a wait timeout; the subscriber closes when
    the sequence is exhausted so the drain loop terminates.
    """

    def __init__(self, frames: list[bytes | None]) -> None:
        self._frames = list(frames)
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def push(self, frame: bytes) -> None:
        self._frames.append(frame)

    def next_frame(self, timeout: float | None = None) -> bytes | None:
        _ = timeout
        if not self._frames:
            self._closed = True
            return None
        return self._frames.pop(0)

    def close(self) -> None:
        self._closed = True


class _StubFrameBus:
    """Frame bus stand-in that hands back a preconfigured subscriber."""

    def __init__(self, subscriber: FrameSubscriberProtocol) -> None:
        self._subscriber = subscriber
        self.unsubscribed = 0

    def publish(self, frame: bytes) -> None:
        _ = frame

    def subscribe(self) -> FrameSubscriberProtocol:
        return self._subscriber

    def unsubscribe(self, subscriber: FrameSubscriberProtocol) -> None:
        _ = subscriber
        self.unsubscribed += 1

    def subscriber_count(self) -> int:
        return 1

    def latest(self) -> bytes | None:
        return None


class _CloseAfterFirstWaitFrameSubscriber:
    """Frame subscriber that closes itself during the first wait.

    Covers the post-wait ``if subscriber.closed: return`` branch of
    the MJPEG drain: the wait returns a frame, but the close observed
    afterwards must win over the write.
    """

    def __init__(self) -> None:
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def push(self, frame: bytes) -> None:
        _ = frame

    def next_frame(self, timeout: float | None = None) -> bytes | None:
        _ = timeout
        self._closed = True
        return b"late-frame"

    def close(self) -> None:
        self._closed = True


class TestDrainFrameBusToResponseHelper:
    """Unit contract for the MJPEG-drain helper."""

    @pytest.mark.asyncio
    async def test_frame_writes_multipart_part_then_close_exits(self) -> None:
        """A frame becomes one boundary-delimited JPEG part."""
        from tankpit_bot.service.http_server import _drain_frame_bus_to_response

        subscriber = _StubFrameSubscriber(frames=[b"AB"])
        bus = _StubFrameBus(subscriber)
        response = _StubResponse()

        await _drain_frame_bus_to_response(bus, response)

        assert response.writes == [
            b"--tankpitbotframe\r\nContent-Type: image/jpeg\r\nContent-Length: 2\r\n\r\nAB\r\n"
        ]
        assert bus.unsubscribed == 1

    @pytest.mark.asyncio
    async def test_timeout_before_any_frame_writes_nothing(self) -> None:
        """A timeout with no prior frame loops silently (no keepalive to send)."""
        from tankpit_bot.service.http_server import _drain_frame_bus_to_response

        subscriber = _StubFrameSubscriber(frames=[None])
        bus = _StubFrameBus(subscriber)
        response = _StubResponse()

        await _drain_frame_bus_to_response(bus, response)

        assert response.writes == []
        assert bus.unsubscribed == 1

    @pytest.mark.asyncio
    async def test_timeout_after_a_frame_resends_it_as_keepalive(self) -> None:
        """A timeout after a frame re-sends that frame (MJPEG keepalive)."""
        from tankpit_bot.service.http_server import _drain_frame_bus_to_response

        subscriber = _StubFrameSubscriber(frames=[b"XY", None])
        bus = _StubFrameBus(subscriber)
        response = _StubResponse()

        await _drain_frame_bus_to_response(bus, response)

        part = b"--tankpitbotframe\r\nContent-Type: image/jpeg\r\nContent-Length: 2\r\n\r\nXY\r\n"
        assert response.writes == [part, part]
        assert bus.unsubscribed == 1

    @pytest.mark.asyncio
    async def test_close_before_first_wait_returns_without_writing(self) -> None:
        """An already-closed subscriber exits before any write."""
        from tankpit_bot.service.http_server import _drain_frame_bus_to_response

        subscriber = _StubFrameSubscriber(frames=[])
        subscriber.close()
        bus = _StubFrameBus(subscriber)
        response = _StubResponse()

        await _drain_frame_bus_to_response(bus, response)

        assert response.writes == []
        assert bus.unsubscribed == 1

    @pytest.mark.asyncio
    async def test_close_observed_after_wait_wins_over_the_frame(self) -> None:
        """A close during the wait suppresses the just-returned frame."""
        from tankpit_bot.service.http_server import _drain_frame_bus_to_response

        subscriber = _CloseAfterFirstWaitFrameSubscriber()
        bus = _StubFrameBus(subscriber)
        response = _StubResponse()

        await _drain_frame_bus_to_response(bus, response)

        assert response.writes == []
        assert bus.unsubscribed == 1


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
