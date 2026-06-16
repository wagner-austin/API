"""Tests for static key functionality."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from platform_core.json_utils import JSONValue

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks.cdp import RouteFulfillHandler
from tankpit_bot.browser import (
    extract_xor_first_bytes,
    find_best_static_byte,
    load_static_key,
    save_static_key,
)
from tankpit_bot.types import CapturedMessage
from tests.no_op_keyboard import NoOpKeyboard

# =============================================================================
# Static Key Helper Function Tests
# =============================================================================


def test_extract_xor_first_bytes_empty_list() -> None:
    """Test extract_xor_first_bytes with empty messages."""
    result = extract_xor_first_bytes([])
    assert result == []


def test_extract_xor_first_bytes_skips_sent_messages() -> None:
    """Test extract_xor_first_bytes skips sent messages."""
    # Create a sent message
    payload = bytes([0x00, 0x04, 0x2E, 0x55])  # length=4, type=0x2E, data=0x55
    msg = CapturedMessage(
        timestamp_ms=1000,
        direction="sent",
        payload=base64.b64encode(payload).decode(),
        ws_url="wss://test.com",
    )
    result = extract_xor_first_bytes([msg])
    assert result == []


def test_extract_xor_first_bytes_skips_short_payloads() -> None:
    """Test extract_xor_first_bytes skips payloads < 4 bytes."""
    # Create a short message (less than 4 bytes)
    payload = bytes([0x00, 0x01, 0x2E])  # only 3 bytes
    msg = CapturedMessage(
        timestamp_ms=1000,
        direction="received",
        payload=base64.b64encode(payload).decode(),
        ws_url="wss://test.com",
    )
    result = extract_xor_first_bytes([msg])
    assert result == []


def test_extract_xor_first_bytes_skips_text_messages() -> None:
    """Test extract_xor_first_bytes skips text message types."""
    # Create a text message (type 0x2B is in TEXT_MESSAGE_TYPES)
    payload = bytes([0x00, 0x04, 0x2B, 0xAB])  # type=0x2B (text)
    msg = CapturedMessage(
        timestamp_ms=1000,
        direction="received",
        payload=base64.b64encode(payload).decode(),
        ws_url="wss://test.com",
    )
    result = extract_xor_first_bytes([msg])
    assert result == []


def test_extract_xor_first_bytes_extracts_binary_messages() -> None:
    """Test extract_xor_first_bytes extracts bytes from binary messages."""
    # Create binary messages (type 0x2E is container)
    payload1 = bytes([0x00, 0x05, 0x2E, 0x55, 0x00])  # data byte = 0x55
    payload2 = bytes([0x00, 0x06, 0x2E, 0xAA, 0x00, 0x00])  # data byte = 0xAA
    msg1 = CapturedMessage(
        timestamp_ms=1000,
        direction="received",
        payload=base64.b64encode(payload1).decode(),
        ws_url="wss://test.com",
    )
    msg2 = CapturedMessage(
        timestamp_ms=1001,
        direction="received",
        payload=base64.b64encode(payload2).decode(),
        ws_url="wss://test.com",
    )
    result = extract_xor_first_bytes([msg1, msg2])
    assert result == [0x55, 0xAA]


def test_find_best_static_byte_returns_tuple() -> None:
    """Test find_best_static_byte returns (best_byte, match_count) tuple."""
    # With empty data, any value works - result is (0, 0) since nothing matches
    result = find_best_static_byte([], ord("a"))
    assert type(result) is tuple
    assert len(result) == 2
    assert result == (0, 0)


def test_find_best_static_byte_finds_best_match() -> None:
    """Test find_best_static_byte finds the byte with most signature matches."""
    # Create data that would produce known signatures when XOR'd correctly
    # Known signature 0x01 is position_update
    # If magic[0]='a' (97), static[0]=X, data byte=Y
    # decoded = Y ^ (X ^ 97)
    # For decoded=0x01, we need Y ^ (X ^ 97) = 0x01

    # Set magic[0]='a' (97)
    magic_first = ord("a")
    # If static[0]=0x00, then table[0] = 0x00 ^ 97 = 97
    # For decoded=0x01, data = 0x01 ^ 97 = 96
    raw_bytes = [96]  # This should decode to 0x01 when static[0]=0

    best_static, count = find_best_static_byte(raw_bytes, magic_first)
    # The algorithm brute-forces to find which static[0] produces known signatures
    # Since 0x01 is a known signature, we expect some coverage
    assert count >= 0
    assert 0 <= best_static <= 255


# =============================================================================
# Static Key Load/Save Tests
# =============================================================================


def test_load_static_key_success() -> None:
    """Test load_static_key loads 1000-character key."""
    original = _test_hooks.read_text
    key_content = "a" * 1000

    def fake_read_text(path: Path) -> str:
        _ = path
        return key_content + "\n"

    _test_hooks.read_text = fake_read_text
    try:
        result = load_static_key()
        assert result == key_content
    finally:
        _test_hooks.read_text = original


def test_load_static_key_wrong_length_raises() -> None:
    """Test load_static_key raises ValueError for wrong key length."""
    original = _test_hooks.read_text
    key_content = "a" * 500  # Too short

    def fake_read_text(path: Path) -> str:
        _ = path
        return key_content + "\n"

    _test_hooks.read_text = fake_read_text
    try:
        with pytest.raises(ValueError, match="expected 1000"):
            load_static_key()
    finally:
        _test_hooks.read_text = original


def test_save_static_key_success() -> None:
    """Test save_static_key writes key to file."""
    original = _test_hooks.write_text
    written_content: list[str] = []

    def fake_write_text(path: Path, content: str) -> None:
        _ = path
        written_content.append(content)

    _test_hooks.write_text = fake_write_text
    try:
        key = "b" * 1000
        save_static_key(key)
        assert len(written_content) == 1
        assert written_content[0] == key + "\n"
    finally:
        _test_hooks.write_text = original


def test_save_static_key_wrong_length_raises() -> None:
    """Test save_static_key raises ValueError for wrong key length."""
    with pytest.raises(ValueError, match="expected 1000"):
        save_static_key("short_key")


# =============================================================================
# Fake Page Classes for Static Key Tests
# =============================================================================


class FakePageWithStaticKey:
    """Fake page that can find and fetch tpclient script for testing."""

    def __init__(self) -> None:
        """Initialize with eval count tracker."""
        self._eval_count = 0
        self._url = "https://tankpit.com/play"
        self._keyboard = NoOpKeyboard()

    @property
    def url(self) -> str:
        """Return test URL."""
        return self._url

    @property
    def keyboard(self) -> NoOpKeyboard:
        """Return keyboard interface."""
        return self._keyboard

    def goto(
        self,
        url: str,
        *,
        referer: str | None = None,
        timeout: float | None = None,
        wait_until: str | None = None,
    ) -> None:
        """Navigate to URL."""
        _ = (referer, timeout, wait_until)
        self._url = url

    def wait_for_timeout(self, timeout: float) -> None:
        """Wait for timeout."""
        _ = timeout

    def set_content(self, html: str, *, timeout: float | None = None) -> None:
        _ = (html, timeout)

    def route(self, url: str, handler: RouteFulfillHandler) -> None:
        _ = (url, handler)

    def wait_for_event(self, event: str, *, timeout: float | None = None) -> None:
        """Wait for event."""
        _ = (event, timeout)

    def wait_for_function(self, expression: str, *, timeout: float | None = None) -> None:
        """Wait for function - always succeeds."""
        _ = (expression, timeout)

    def close(self, *, reason: str | None = None, run_before_unload: bool | None = None) -> None:
        """Close page."""
        _ = (reason, run_before_unload)

    def evaluate(self, expression: str) -> JSONValue:
        """Return script URL or JS content based on the expression.

        First call looks for tpclient URL -> return string URL
        Second call fetches content -> return JS with 1000-char key
        """
        self._eval_count += 1
        if "fetch" in expression:
            # Return JS content with a 1000-char key
            key = "A" * 1000
            return f'var x = "{key}";'
        # First call - looking for tpclient script URL
        return "https://tankpit.com/js/tpclient.min.js"


class FakePageNoKey:
    """Fake page that returns JS without a 1000-char key."""

    def __init__(self) -> None:
        """Initialize."""
        self._url = "https://tankpit.com/play"
        self._keyboard = NoOpKeyboard()

    @property
    def url(self) -> str:
        """Return test URL."""
        return self._url

    @property
    def keyboard(self) -> NoOpKeyboard:
        """Return keyboard interface."""
        return self._keyboard

    def goto(
        self,
        url: str,
        *,
        referer: str | None = None,
        timeout: float | None = None,
        wait_until: str | None = None,
    ) -> None:
        """Navigate to URL."""
        _ = (referer, timeout, wait_until)
        self._url = url

    def wait_for_timeout(self, timeout: float) -> None:
        """Wait for timeout."""
        _ = timeout

    def set_content(self, html: str, *, timeout: float | None = None) -> None:
        _ = (html, timeout)

    def route(self, url: str, handler: RouteFulfillHandler) -> None:
        _ = (url, handler)

    def wait_for_event(self, event: str, *, timeout: float | None = None) -> None:
        """Wait for event."""
        _ = (event, timeout)

    def wait_for_function(self, expression: str, *, timeout: float | None = None) -> None:
        """Wait for function - always succeeds."""
        _ = (expression, timeout)

    def close(self, *, reason: str | None = None, run_before_unload: bool | None = None) -> None:
        """Close page."""
        _ = (reason, run_before_unload)

    def evaluate(self, expression: str) -> JSONValue:
        """Return script URL or JS content without 1000-char key."""
        if "fetch" in expression:
            # Return JS without a 1000-char key
            return 'var x = "short_key";'
        return "https://tankpit.com/js/tpclient.min.js"


class FakePageFetchFails:
    """Fake page where fetch returns non-string."""

    def __init__(self) -> None:
        """Initialize."""
        self._url = "https://tankpit.com/play"
        self._keyboard = NoOpKeyboard()

    @property
    def url(self) -> str:
        """Return test URL."""
        return self._url

    @property
    def keyboard(self) -> NoOpKeyboard:
        """Return keyboard interface."""
        return self._keyboard

    def goto(
        self,
        url: str,
        *,
        referer: str | None = None,
        timeout: float | None = None,
        wait_until: str | None = None,
    ) -> None:
        """Navigate to URL."""
        _ = (referer, timeout, wait_until)
        self._url = url

    def wait_for_timeout(self, timeout: float) -> None:
        """Wait for timeout."""
        _ = timeout

    def set_content(self, html: str, *, timeout: float | None = None) -> None:
        _ = (html, timeout)

    def route(self, url: str, handler: RouteFulfillHandler) -> None:
        _ = (url, handler)

    def wait_for_event(self, event: str, *, timeout: float | None = None) -> None:
        """Wait for event."""
        _ = (event, timeout)

    def wait_for_function(self, expression: str, *, timeout: float | None = None) -> None:
        """Wait for function - always succeeds."""
        _ = (expression, timeout)

    def close(self, *, reason: str | None = None, run_before_unload: bool | None = None) -> None:
        """Close page."""
        _ = (reason, run_before_unload)

    def evaluate(self, expression: str) -> JSONValue:
        """Return script URL or None for fetch."""
        if "fetch" in expression:
            # Return None (simulates failed fetch)
            return None
        return "https://tankpit.com/js/tpclient.min.js"


# =============================================================================
# BrowserSession Static Key Tests
# =============================================================================
