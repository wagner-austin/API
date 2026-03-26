"""Tests for world_sync module: install_ws_hook and drain_js_messages."""

from __future__ import annotations

from collections.abc import Callable

from platform_core.json_utils import JSONObject, dump_json_str

from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.world_sync import drain_js_messages, install_ws_hook
from tankpit_bot.sniffer.world_state import reset_world_state
from tests.conftest import FakeEnv


class FakeCDPWithResponse:
    """CDP fake that returns configurable values for Runtime.evaluate."""

    def __init__(self, response_value: str = "") -> None:
        """Initialize with a response value for Runtime.evaluate.

        Args:
            response_value: The value string returned by CDP evaluate calls.
        """
        self._response_value = response_value
        self._sent_methods: list[str] = []
        self._call_count = 0

    def send(
        self,
        method: str,
        params: JSONObject | None = None,
    ) -> JSONObject:
        """Record method call and return configured response."""
        _ = params
        self._sent_methods.append(method)
        self._call_count += 1
        return {"result": {"value": self._response_value}}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Register event handler (no-op)."""
        _ = (event, handler)

    def detach(self) -> None:
        """Detach session (no-op)."""


class FakeCDPNonDictResult:
    """CDP fake that returns a non-dict inner result."""

    def __init__(self) -> None:
        """Initialize fake CDP."""
        self._sent_methods: list[str] = []

    def send(
        self,
        method: str,
        params: JSONObject | None = None,
    ) -> JSONObject:
        """Return a response where 'result' is a string, not a dict."""
        _ = params
        self._sent_methods.append(method)
        return {"result": "not_a_dict"}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Register event handler (no-op)."""
        _ = (event, handler)

    def detach(self) -> None:
        """Detach session (no-op)."""


class TestInstallWsHook:
    """Tests for install_ws_hook."""

    def test_sends_runtime_evaluate(self, fake_env: FakeEnv) -> None:
        """install_ws_hook sends Runtime.evaluate to inject JS hook."""
        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        cdp = FakeCDPWithResponse('{"runtime_hook":"installed"}')
        bot._cdp = cdp
        install_ws_hook(bot)
        assert cdp._sent_methods == ["Runtime.evaluate"]

    def test_noop_without_cdp(self, fake_env: FakeEnv) -> None:
        """install_ws_hook does nothing when CDP is None."""
        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        # bot._cdp is None by default
        install_ws_hook(bot)  # should not raise


class TestDrainJsMessages:
    """Tests for drain_js_messages."""

    def test_returns_zero_without_cdp(self, fake_env: FakeEnv) -> None:
        """drain_js_messages returns 0 when CDP is None."""
        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        count = drain_js_messages(bot)
        assert count == 0

    def test_returns_zero_with_empty_response(self, fake_env: FakeEnv) -> None:
        """drain_js_messages returns 0 when JS returns empty string."""
        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        cdp = FakeCDPWithResponse("")
        bot._cdp = cdp
        count = drain_js_messages(bot)
        assert count == 0

    def test_returns_zero_with_empty_msgs(self, fake_env: FakeEnv) -> None:
        """drain_js_messages returns 0 when msgs array is empty."""
        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        response = dump_json_str({"count": 0, "msgs": []})
        cdp = FakeCDPWithResponse(response)
        bot._cdp = cdp
        count = drain_js_messages(bot)
        assert count == 0

    def test_returns_zero_with_non_dict_inner(self, fake_env: FakeEnv) -> None:
        """drain_js_messages returns 0 when inner result is not a dict."""
        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        cdp = FakeCDPNonDictResult()
        bot._cdp = cdp
        count = drain_js_messages(bot)
        assert count == 0

    def test_returns_zero_with_non_list_msgs(self, fake_env: FakeEnv) -> None:
        """drain_js_messages returns 0 when msgs is not a list."""
        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        response = dump_json_str({"count": 0, "msgs": "not_a_list"})
        cdp = FakeCDPWithResponse(response)
        bot._cdp = cdp
        count = drain_js_messages(bot)
        assert count == 0

    def test_skips_non_string_entries(self, fake_env: FakeEnv) -> None:
        """drain_js_messages skips non-string entries in msgs."""
        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        response = dump_json_str({"count": 1, "msgs": [123, None]})
        cdp = FakeCDPWithResponse(response)
        bot._cdp = cdp
        count = drain_js_messages(bot)
        assert count == 0

    def test_processes_valid_base64_messages(self, fake_env: FakeEnv) -> None:
        """drain_js_messages decodes and processes valid base64 messages.

        Uses a minimal binary WebSocket frame (a Sync 0x3F message) to verify
        the full drain → decode → process_received_message pipeline.
        """
        import base64

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        # Construct a minimal binary frame: 1-byte type header 0x3F (sync)
        # process_received_message handles raw binary via base64
        raw_frame = bytes([0x3F])
        b64_msg = base64.b64encode(raw_frame).decode("ascii")
        response = dump_json_str({"count": 1, "msgs": [b64_msg]})
        cdp = FakeCDPWithResponse(response)
        bot._cdp = cdp
        count = drain_js_messages(bot)
        assert count == 1

    def test_reports_count_for_multiple_messages(self, fake_env: FakeEnv) -> None:
        """drain_js_messages returns count of processed string messages."""
        import base64

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        raw_frame = bytes([0x3F])
        b64_msg = base64.b64encode(raw_frame).decode("ascii")
        response = dump_json_str({"count": 3, "msgs": [b64_msg, b64_msg, b64_msg]})
        cdp = FakeCDPWithResponse(response)
        bot._cdp = cdp
        count = drain_js_messages(bot)
        assert count == 3
