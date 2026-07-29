"""Tests for :class:`tankpit_bot.browser.screencast.ScreencastService`.

Covers the start/stop idempotency contract, the once-per-CDP-session
handler registration, the ack-then-publish frame relay, and the
loud-failure invariants (missing CDP session, malformed event). The
CDP session is a recording fake matching
:class:`~tankpit_bot._test_hooks.CDPSessionProtocol` structurally.
"""

from __future__ import annotations

import base64
from collections.abc import Callable

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.browser.screencast import (
    SCREENCAST_MAX_DIMENSION,
    SCREENCAST_QUALITY,
    ScreencastService,
)


class _RecordingCDP:
    """CDP-session fake that records sends and handler registrations."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, JSONObject | None]] = []
        self.registrations: list[str] = []
        self.handlers: dict[str, Callable[[JSONObject], None]] = {}

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        self.sent.append((method, params))
        return {}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        self.registrations.append(event)
        self.handlers[event] = handler

    def detach(self) -> None:
        raise AssertionError("the screencast service never detaches the session")


class _FrameSink:
    """Records every frame the service publishes."""

    def __init__(self) -> None:
        self.frames: list[bytes] = []

    def __call__(self, frame: bytes) -> None:
        self.frames.append(frame)


class TestStartStop:
    """Lifecycle contract: demand-driven start/stop, idempotent both ways."""

    def test_start_registers_handler_and_begins_the_cast(self) -> None:
        """First start wires the frame handler and sends startScreencast."""
        sink = _FrameSink()
        service = ScreencastService(publish=sink)
        cdp = _RecordingCDP()

        service.start(cdp)

        assert service.active is True
        assert cdp.registrations == ["Page.screencastFrame"]
        assert cdp.sent == [
            (
                "Page.startScreencast",
                {
                    "format": "jpeg",
                    "quality": SCREENCAST_QUALITY,
                    "maxWidth": SCREENCAST_MAX_DIMENSION,
                    "maxHeight": SCREENCAST_MAX_DIMENSION,
                    "everyNthFrame": 1,
                },
            )
        ]

    def test_start_while_active_is_a_noop(self) -> None:
        """A second start does not double-send or re-register."""
        service = ScreencastService(publish=_FrameSink())
        cdp = _RecordingCDP()
        service.start(cdp)

        service.start(cdp)

        assert len(cdp.sent) == 1
        assert cdp.registrations == ["Page.screencastFrame"]

    def test_stop_ends_the_cast(self) -> None:
        """Stop sends stopScreencast and flips ``active`` off."""
        service = ScreencastService(publish=_FrameSink())
        cdp = _RecordingCDP()
        service.start(cdp)

        service.stop(cdp)

        assert service.active is False
        assert cdp.sent[-1] == ("Page.stopScreencast", None)

    def test_stop_while_inactive_is_a_noop(self) -> None:
        """A stop with no active cast sends nothing."""
        service = ScreencastService(publish=_FrameSink())
        cdp = _RecordingCDP()

        service.stop(cdp)

        assert cdp.sent == []

    def test_restart_on_same_session_reuses_the_registration(self) -> None:
        """start → stop → start keeps exactly one handler registration."""
        service = ScreencastService(publish=_FrameSink())
        cdp = _RecordingCDP()
        service.start(cdp)
        service.stop(cdp)

        service.start(cdp)

        assert service.active is True
        assert cdp.registrations == ["Page.screencastFrame"]
        start_sends = [method for method, _ in cdp.sent if method == "Page.startScreencast"]
        assert len(start_sends) == 2

    def test_fresh_session_gets_a_fresh_registration(self) -> None:
        """A NEW CDP session (next game session) is wired independently."""
        service = ScreencastService(publish=_FrameSink())
        first = _RecordingCDP()
        service.start(first)
        service.stop(first)

        second = _RecordingCDP()
        service.start(second)

        assert second.registrations == ["Page.screencastFrame"]
        assert service.active is True


class TestFrameRelay:
    """The ack-then-publish contract for ``Page.screencastFrame`` events."""

    def test_frame_is_acked_then_published_decoded(self) -> None:
        """The handler acks the sessionId and publishes the decoded JPEG."""
        sink = _FrameSink()
        service = ScreencastService(publish=sink)
        cdp = _RecordingCDP()
        service.start(cdp)
        handler = cdp.handlers["Page.screencastFrame"]

        payload = base64.b64encode(b"\xff\xd8jpeg-bytes").decode()
        handler({"sessionId": 7, "data": payload, "metadata": {}})

        assert ("Page.screencastFrameAck", {"sessionId": 7}) in cdp.sent
        assert sink.frames == [b"\xff\xd8jpeg-bytes"]

    def test_ack_happens_before_publish(self) -> None:
        """Chrome's stream stays unthrottled: the ack precedes the publish."""
        order: list[str] = []

        class _OrderCDP(_RecordingCDP):
            def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
                order.append(method)
                return super().send(method, params)

        def publish(frame: bytes) -> None:
            _ = frame
            order.append("publish")

        service = ScreencastService(publish=publish)
        cdp = _OrderCDP()
        service.start(cdp)
        handler = cdp.handlers["Page.screencastFrame"]

        handler({"sessionId": 1, "data": base64.b64encode(b"x").decode()})

        assert order == ["Page.startScreencast", "Page.screencastFrameAck", "publish"]

    def test_frame_without_attached_session_raises(self) -> None:
        """A frame with no stored CDP session is an invariant violation."""
        service = ScreencastService(publish=_FrameSink())

        with pytest.raises(RuntimeError, match="no CDP session attached"):
            service._on_frame({"sessionId": 1, "data": base64.b64encode(b"x").decode()})

    def test_malformed_event_fails_loudly(self) -> None:
        """A frame event missing ``data`` raises instead of freezing the stream."""
        service = ScreencastService(publish=_FrameSink())
        cdp = _RecordingCDP()
        service.start(cdp)
        handler = cdp.handlers["Page.screencastFrame"]

        with pytest.raises(JSONTypeError):
            handler({"sessionId": 3})
