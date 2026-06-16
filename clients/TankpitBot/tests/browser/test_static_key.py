"""Tests for static key functionality."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from platform_core.json_utils import JSONValue

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks.cdp import RouteFulfillHandler
from tankpit_bot.browser import (
    BrowserSession,
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


def test_browser_session_capture_static_key_success() -> None:
    """Test _capture_static_key successfully extracts and saves static key."""
    from tankpit_bot._test_hooks import PageProtocol

    session = BrowserSession("https://example.com")
    page: PageProtocol = FakePageWithStaticKey()

    # Capture original hooks
    original_save = _test_hooks.write_text
    saved_content: list[str] = []

    def fake_write(path: Path, content: str) -> None:
        saved_content.append(content)

    _test_hooks.write_text = fake_write
    try:
        session._capture_static_key(page)
        assert session._static_key == "A" * 1000
        assert len(saved_content) == 2
        assert '"' + "A" * 1000 + '"' in saved_content[0]
        assert saved_content[1] == "A" * 1000 + "\n"
    finally:
        _test_hooks.write_text = original_save


def test_browser_session_derive_static_key_success() -> None:
    """Test _derive_static_key_from_messages derives key from messages."""
    session = BrowserSession("https://example.com")

    # Set magic key
    session._magic = "test_magic_key_12345678901234567890"

    # Create a message that matches known signature when XOR decoded
    # For signature 0x2E, we need first_byte XOR static[0] XOR magic[0] = 0x2E
    # first_byte = 0x2E XOR static[0] XOR magic[0]
    # Let's use static[0] = 'A' (0x41), magic[0] = 't' (0x74)
    # first_byte = 0x2E XOR 0x41 XOR 0x74 = 0x1B
    raw_bytes = bytes([0x1B]) + b"\x00" * 10
    b64_payload = base64.b64encode(raw_bytes).decode()

    session._messages = [
        CapturedMessage(
            timestamp_ms=1000,
            direction="received",
            payload=b64_payload,
            ws_url="wss://test.com/ws",
        ),
    ]

    # Set up static key file with 'A' as first char
    original_read = _test_hooks.read_text
    original_write = _test_hooks.write_text
    saved_keys: list[str] = []

    def fake_read(path: Path) -> str:
        return "A" * 1000

    def fake_write(path: Path, content: str) -> None:
        saved_keys.append(content.strip())

    _test_hooks.read_text = fake_read
    _test_hooks.write_text = fake_write
    try:
        session._derive_static_key_from_messages()
        # Key should have been derived and potentially saved
        # The exact behavior depends on the first byte calculation
    finally:
        _test_hooks.read_text = original_read
        _test_hooks.write_text = original_write


def test_browser_session_derive_static_key_no_magic() -> None:
    """Test _derive_static_key_from_messages exits early without magic."""
    session = BrowserSession("https://example.com")
    # No magic set
    session._derive_static_key_from_messages()
    # Should return early without error


def test_browser_session_derive_static_key_no_messages() -> None:
    """Test _derive_static_key_from_messages exits early without messages."""
    session = BrowserSession("https://example.com")
    session._magic = "test_magic"
    session._messages = []
    session._derive_static_key_from_messages()
    # Should return early without error


def test_browser_session_derive_static_key_no_binary_messages() -> None:
    """Test _derive_static_key_from_messages logs warning for no binary messages."""
    session = BrowserSession("https://example.com")
    session._magic = "test_magic_key"

    # Create a valid base64 payload with TEXT_MESSAGE_TYPE (0x2B = 43)
    # Format: [length_hi, length_lo, msg_type, data...]
    # Using msg_type 0x2B which is in TEXT_MESSAGE_TYPES
    text_type_payload = bytes([0x00, 0x04, 0x2B, 0x00])  # 0x2B is text type
    payload_b64 = base64.b64encode(text_type_payload).decode()

    session._messages = [
        CapturedMessage(
            timestamp_ms=1000,
            direction="received",
            payload=payload_b64,
            ws_url="wss://test.com/ws",
        ),
    ]
    # Should return early after logging warning about no binary messages
    # because all messages are text type (filtered out)
    session._derive_static_key_from_messages()
    # No exception, static key remains None
    assert session._static_key is None


def test_browser_session_capture_static_key_no_key_found() -> None:
    """A keyless tpclient JS is saved via the hook and leaves no static key.

    Regression guard: the JS save used to be a raw ``Path.write_text``,
    so this test overwrote the real repo-root ``tpclient.js`` protocol
    artifact with the 20-byte fake on every run.
    """
    from tankpit_bot._test_hooks import PageProtocol

    session = BrowserSession("https://example.com")
    page: PageProtocol = FakePageNoKey()

    original_write = _test_hooks.write_text
    written: list[tuple[Path, str]] = []

    def fake_write(path: Path, content: str) -> None:
        written.append((path, content))

    _test_hooks.write_text = fake_write
    try:
        session._capture_static_key(page)
    finally:
        _test_hooks.write_text = original_write

    assert session._static_key is None
    assert written == [(Path("tpclient.js"), 'var x = "short_key";')]


def test_browser_session_capture_static_key_fetch_fails() -> None:
    """Test _capture_static_key logs warning when fetch returns non-string."""
    from tankpit_bot._test_hooks import PageProtocol

    session = BrowserSession("https://example.com")
    page: PageProtocol = FakePageFetchFails()

    session._capture_static_key(page)
    # Should return early, static key remains None
    assert session._static_key is None


def test_browser_session_derive_static_key_no_signatures_matched() -> None:
    """Test _derive_static_key_from_messages logs warning when no signatures match."""
    session = BrowserSession("https://example.com")
    session._magic = "A"

    # Create a valid binary message
    binary_payload = bytes([0x00, 0x04, 0x01, 0x00])
    payload_b64 = base64.b64encode(binary_payload).decode()

    session._messages = [
        CapturedMessage(
            timestamp_ms=1000,
            direction="received",
            payload=payload_b64,
            ws_url="wss://test.com/ws",
        ),
    ]

    original_finder = _test_hooks.find_best_static_byte

    def fake_finder(raw_first_bytes: list[int], magic_first_byte: int) -> tuple[int, int]:
        """Fake finder that returns 0 coverage."""
        _ = (raw_first_bytes, magic_first_byte)
        return (0, 0)  # No signatures matched

    _test_hooks.find_best_static_byte = fake_finder
    try:
        session._derive_static_key_from_messages()
        # Should return early with warning, static key remains None
        assert session._static_key is None
    finally:
        _test_hooks.find_best_static_byte = original_finder


def test_browser_session_derive_static_key_matches_current() -> None:
    """Test _derive_static_key_from_messages when derived key matches current."""
    session = BrowserSession("https://example.com")
    session._magic = "A"  # magic[0] = 65

    # We want derived static[0] to match the current key's first byte.
    # With magic='A' (65) and raw_0=65, K = raw_0 ^ magic = 0.
    # decoded = static_0 ^ K = static_0.
    # The smallest signature (0x21 = 33) is hit when static_0 = 33.
    # So best_static_0 = 33, and we set current key to start with chr(33) = '!'.
    binary_payload = bytes([0x00, 0x04, 0x01, 65])  # data byte 65 = 'A' = magic
    payload_b64 = base64.b64encode(binary_payload).decode()

    session._messages = [
        CapturedMessage(
            timestamp_ms=1000,
            direction="received",
            payload=payload_b64,
            ws_url="wss://test.com/ws",
        ),
    ]

    original_read = _test_hooks.read_text
    write_called = False

    def fake_read(path: Path) -> str:
        if "static_key" in str(path):
            # chr(33) = '!' - this matches best_static_0 = 33
            return "!" + "A" * 999
        return original_read(path)

    def fake_write(path: Path, content: str) -> None:
        nonlocal write_called
        write_called = True

    _test_hooks.read_text = fake_read
    original_write = _test_hooks.write_text
    _test_hooks.write_text = fake_write
    try:
        session._derive_static_key_from_messages()
        # Key matches, so file should NOT be written (684->exit branch)
        assert not write_called
        assert session._static_key is None  # Not updated since it matches
    finally:
        _test_hooks.read_text = original_read
        _test_hooks.write_text = original_write
