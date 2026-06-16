"""Tests for CDPService."""

from __future__ import annotations

from collections.abc import Callable

from platform_core.json_utils import JSONObject

from tankpit_bot.browser.cdp_service import CDPService
from tankpit_bot.types import CapturedMessage


class _FakeCDP:
    """Minimal CDP fake for CDPService tests."""

    def __init__(self, send_result: str = "SENT_5_BYTES via wss://test") -> None:
        self._send_result = send_result
        self.sent_expressions: list[str] = []

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Record and return canned send result."""
        if method == "Runtime.evaluate" and params is not None:
            expr = params.get("expression", "")
            if isinstance(expr, str):
                self.sent_expressions.append(expr[:50])
            return {"result": {"value": self._send_result}}
        return {"result": {"value": None}}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Ignore event registration."""
        _ = (event, handler)

    def detach(self) -> None:
        """Ignore detach."""


def test_send_bytes_delegates_to_send_websocket_bytes() -> None:
    """CDPService.send_bytes calls through to send_websocket_bytes."""
    service = CDPService()
    cdp = _FakeCDP()
    result = service.send_bytes(cdp, b"\x01\x02\x03", "test_label")
    assert result == "SENT_5_BYTES via wss://test"


def test_extract_magic_without_callbacks() -> None:
    """Magic extraction works when no callbacks are set."""
    import base64

    service = CDPService()
    body = "%AUTH !be session|hash|ts my_magic_key_here"
    body_bytes = body.encode("utf-8")
    length_prefix = len(body_bytes).to_bytes(2, "little")
    payload = base64.b64encode(length_prefix + body_bytes).decode("ascii")

    msg = CapturedMessage(
        timestamp_ms=1000,
        direction="sent",
        payload=payload,
        ws_url="wss://test",
    )
    service._extract_magic_and_notify(msg)
    assert service.magic == "my_magic_key_here"


def test_extract_magic_with_callbacks() -> None:
    """Magic extraction invokes both callbacks."""
    import base64

    captured_messages: list[CapturedMessage] = []
    captured_magic: list[str] = []

    service = CDPService()
    service.set_callbacks(
        on_message_captured=lambda msg: captured_messages.append(msg),
        on_magic_captured=lambda magic: captured_magic.append(magic),
    )

    body = "%AUTH !be session|hash|ts callback_magic"
    body_bytes = body.encode("utf-8")
    length_prefix = len(body_bytes).to_bytes(2, "little")
    payload = base64.b64encode(length_prefix + body_bytes).decode("ascii")

    msg = CapturedMessage(
        timestamp_ms=1000,
        direction="sent",
        payload=payload,
        ws_url="wss://test",
    )
    service._extract_magic_and_notify(msg)
    assert service.magic == "callback_magic"
    assert len(captured_messages) == 1
    assert captured_magic == ["callback_magic"]
