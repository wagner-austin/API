"""Tests for the video frame stream.

The watch, frame, and video routes plus the frame-bus drain helper,
with the subscriber stubs for each close and timeout path.
"""

from __future__ import annotations

import asyncio

import pytest
from aiohttp import (
    web,
)
from aiohttp.test_utils import (
    TestClient,
    TestServer,
)

from tankpit_bot.service.frame_bus import (
    FrameBus,
    FrameSubscriberProtocol,
)
from tankpit_bot.service.http_server import make_app
from tankpit_bot.service.mode_bridge import ModeBridge
from tankpit_bot.service.status_bus import (
    StatusBus,
)
from tests.service._http_fixtures import (
    _noop_shutdown,
    _RecordingRunner,
    _StubResponse,
)


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
