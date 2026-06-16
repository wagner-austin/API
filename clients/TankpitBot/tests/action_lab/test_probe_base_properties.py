"""Tests for ProbeBase property delegation and service wiring."""

from __future__ import annotations

from tankpit_bot.action_lab.probe_base import ProbeBase
from tankpit_bot.action_lab.probe_factory import create_probe
from tankpit_bot.types import CapturedMessage


def test_factory_creates_probe_with_injected_services() -> None:
    probe = create_probe(ProbeBase, "https://test.com", headless=True)
    assert probe._target_url == "https://test.com"
    assert probe._headless is True
    assert type(probe._cdp_service).__name__ == "CDPService"
    assert type(probe._commands).__name__ == "CommandService"


def test_messages_property_delegates_to_cdp_service() -> None:
    probe = ProbeBase("https://test.com")
    assert probe.messages == []
    msg = CapturedMessage(timestamp_ms=1, direction="received", payload="x", ws_url="wss://t")
    probe._cdp_service.messages.append(msg)
    assert len(probe.messages) == 1


def test_messages_setter_delegates_to_cdp_service() -> None:
    probe = ProbeBase("https://test.com")
    msg = CapturedMessage(timestamp_ms=1, direction="received", payload="x", ws_url="wss://t")
    probe._messages = [msg]
    assert probe._cdp_service.messages == [msg]


def test_ws_urls_property_delegates_to_cdp_service() -> None:
    probe = ProbeBase("https://test.com")
    assert probe._ws_urls == {}
    probe._ws_urls = {"r1": "wss://test"}
    assert probe._ws_urls == {"r1": "wss://test"}
    assert probe._cdp_service.ws_urls == {"r1": "wss://test"}


def test_magic_property_delegates_to_cdp_service() -> None:
    probe = ProbeBase("https://test.com")
    assert probe._magic is None
    probe._magic = "test_magic"
    assert probe._magic == "test_magic"
    assert probe._cdp_service.magic == "test_magic"
    assert probe.magic == "test_magic"


def test_captured_message_count() -> None:
    probe = ProbeBase("https://test.com")
    assert probe.captured_message_count() == 0
    probe._cdp_service.messages.append(
        CapturedMessage(timestamp_ms=1, direction="received", payload="x", ws_url="wss://t"),
    )
    assert probe.captured_message_count() == 1


def test_setup_methods_delegate_to_cdp_service() -> None:
    from collections.abc import Callable

    from platform_core.json_utils import JSONObject

    calls: list[str] = []

    class _TrackingCDP:
        def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
            calls.append(f"send:{method}")
            return {"result": {"value": None}}

        def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
            calls.append(f"on:{event}")

        def detach(self) -> None:
            pass

    probe = ProbeBase("https://test.com")
    cdp = _TrackingCDP()
    probe._setup_cdp_handlers(cdp)
    assert "send:Page.enable" in calls

    calls.clear()
    probe._setup_console_listener(cdp)
    assert "send:Runtime.enable" in calls


def test_send_websocket_bytes_delegates() -> None:
    from collections.abc import Callable

    from platform_core.json_utils import JSONObject

    class _SendCDP:
        def __init__(self) -> None:
            self.expressions: list[str] = []

        def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
            if method == "Runtime.evaluate" and params is not None:
                expr = params.get("expression", "")
                if isinstance(expr, str):
                    self.expressions.append(expr[:30])
                return {"result": {"value": "SENT_3_BYTES via wss://test"}}
            return {"result": {"value": None}}

        def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
            _ = (event, handler)

        def detach(self) -> None:
            pass

    probe = ProbeBase("https://test.com")
    cdp = _SendCDP()
    result = probe._send_websocket_bytes(cdp, b"\x01\x02\x03", "test")
    assert result == "SENT_3_BYTES via wss://test"
