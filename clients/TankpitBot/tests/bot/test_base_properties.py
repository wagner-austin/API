"""Tests for Bot property delegation and service wiring via composition."""

from __future__ import annotations

from tankpit_bot.bot.base import Bot
from tankpit_bot.types import CapturedMessage


def test_bot_messages_property_delegates_to_cdp_service() -> None:
    """Bot._messages delegates to CDPService.messages."""
    bot = Bot("https://test.com")
    assert bot._messages == []
    msg = CapturedMessage(timestamp_ms=1, direction="received", payload="x", ws_url="wss://t")
    bot._cdp_service.messages.append(msg)
    assert len(bot._messages) == 1


def test_bot_messages_setter_delegates_to_cdp_service() -> None:
    """Bot._messages setter delegates to CDPService.messages."""
    bot = Bot("https://test.com")
    msg = CapturedMessage(timestamp_ms=1, direction="received", payload="x", ws_url="wss://t")
    bot._messages = [msg]
    assert bot._cdp_service.messages == [msg]


def test_bot_ws_urls_property_delegates_to_cdp_service() -> None:
    """Bot._ws_urls delegates to CDPService.ws_urls."""
    bot = Bot("https://test.com")
    assert bot._ws_urls == {}
    bot._ws_urls = {"r1": "wss://test"}
    assert bot._ws_urls == {"r1": "wss://test"}
    assert bot._cdp_service.ws_urls == {"r1": "wss://test"}


def test_bot_magic_property_delegates_to_cdp_service() -> None:
    """Bot._magic delegates to CDPService.magic."""
    bot = Bot("https://test.com")
    assert bot._magic is None
    bot._magic = "test_magic"
    assert bot._magic == "test_magic"
    assert bot._cdp_service.magic == "test_magic"


def test_bot_captured_message_count_delegates() -> None:
    """Bot.captured_message_count delegates to CDPService message list."""
    bot = Bot("https://test.com")
    assert bot.captured_message_count() == 0
    bot._cdp_service.messages.append(
        CapturedMessage(timestamp_ms=1, direction="received", payload="x", ws_url="wss://t"),
    )
    assert bot.captured_message_count() == 1


def test_bot_owns_cdp_service_directly() -> None:
    """Bot owns CDPService via composition, not inheritance."""
    bot = Bot("https://test.com")
    assert type(bot._cdp_service).__name__ == "CDPService"
    assert type(bot._commands).__name__ == "CommandService"


def test_bot_accepts_injected_services() -> None:
    """Bot accepts cdp_service and command_service kwargs for DI."""
    from tankpit_bot.bot.command_service import CommandService
    from tankpit_bot.browser.cdp_service import CDPService
    from tankpit_bot.browser.cdp_utils import send_websocket_bytes

    cdp_svc = CDPService()
    cmd_svc = CommandService(send_ws_bytes=send_websocket_bytes)
    bot = Bot(
        "https://test.com",
        cdp_service=cdp_svc,
        command_service=cmd_svc,
    )
    assert bot._cdp_service is cdp_svc
    assert bot._commands is cmd_svc


def test_bot_setup_methods_delegate_to_cdp_service() -> None:
    """Bot._setup_cdp_handlers and _setup_console_listener delegate to CDPService."""
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

    bot = Bot("https://test.com")
    cdp = _TrackingCDP()
    bot._setup_cdp_handlers(cdp)
    assert "send:Page.enable" in calls

    calls.clear()
    bot._setup_console_listener(cdp)
    assert "send:Runtime.enable" in calls


def test_bot_send_websocket_bytes_delegates() -> None:
    """Bot._send_websocket_bytes delegates to send_websocket_bytes."""
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

    bot = Bot("https://test.com")
    cdp = _SendCDP()
    result = bot._send_websocket_bytes(cdp, b"\x01\x02\x03", "test")
    assert result == "SENT_3_BYTES via wss://test"


def test_bot_static_key_set_from_gather_intel() -> None:
    """Bot._static_key is set when gather_intel returns a key."""
    bot = Bot("https://test.com")
    assert bot._static_key is None
    bot._static_key = "test_static_key"
    assert bot._static_key == "test_static_key"
