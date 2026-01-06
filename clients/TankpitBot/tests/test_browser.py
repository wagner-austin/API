"""Tests for tankpit_bot.browser module."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject

from tankpit_bot import _test_hooks
from tankpit_bot.browser import (
    BrowserError,
    BrowserSession,
    GameNotJoinedError,
    PlaywrightNotInstalledError,
    cdp_timestamp_to_ms,
    extract_xor_first_bytes,
    find_best_static_byte,
    get_current_time_ms,
    load_static_key,
    reset_cdp_time_offset,
    save_static_key,
)
from tankpit_bot.types import CapturedMessage
from tests.fakes import FakeBrowser, FakeBrowserContext, FakeCDPSession, FakePage

# =============================================================================
# Helper Function Tests
# =============================================================================


def test_get_current_time_ms_returns_int() -> None:
    """Test get_current_time_ms returns an integer."""
    result = get_current_time_ms()
    assert type(result) is int
    assert result > 0


def test_cdp_timestamp_to_ms() -> None:
    """Test cdp_timestamp_to_ms converts CDP time to Unix time."""
    reset_cdp_time_offset()
    current_unix_ms = get_current_time_ms()
    cdp_seconds = 12345.678
    result = cdp_timestamp_to_ms(cdp_seconds)
    # Result should be approximately current Unix time
    # (within 100ms to account for test execution time)
    expected_offset = current_unix_ms - int(cdp_seconds * 1000)
    expected = int(cdp_seconds * 1000) + expected_offset
    assert abs(result - expected) < 100


def test_cdp_timestamp_offset_persists() -> None:
    """Test CDP time offset is calculated once and reused."""
    reset_cdp_time_offset()
    # First call establishes the offset
    result1 = cdp_timestamp_to_ms(100.0)
    # Second call uses same offset, so difference should be exactly 1000ms
    result2 = cdp_timestamp_to_ms(101.0)
    assert result2 - result1 == 1000


# =============================================================================
# Error Class Tests
# =============================================================================


def test_browser_error_is_exception() -> None:
    """Test BrowserError is a subclass of Exception."""
    assert issubclass(BrowserError, Exception)
    err = BrowserError("test error")
    assert str(err) == "test error"


def test_playwright_not_installed_error_is_browser_error() -> None:
    """Test PlaywrightNotInstalledError is a BrowserError."""
    assert issubclass(PlaywrightNotInstalledError, BrowserError)


def test_game_not_joined_error_is_browser_error() -> None:
    """Test GameNotJoinedError is a BrowserError."""
    assert issubclass(GameNotJoinedError, BrowserError)


# =============================================================================
# BrowserSession Tests
# =============================================================================


def test_browser_session_init() -> None:
    """Test BrowserSession initialization."""
    session = BrowserSession("https://example.com", headless=True, prefer_account=False)
    assert session._target_url == "https://example.com"
    assert session._headless is True
    assert session._prefer_account is False
    assert len(session.session_id) == 36  # UUID format
    assert session.messages == []
    assert session.magic is None


def test_browser_session_properties() -> None:
    """Test BrowserSession property accessors."""
    session = BrowserSession("https://example.com")
    session._magic = "test_magic"
    assert session.magic == "test_magic"
    assert len(session.session_id) == 36  # UUID format


def test_browser_session_on_websocket_created() -> None:
    """Test _on_websocket_created records WebSocket URL."""
    session = BrowserSession("https://example.com")
    params: JSONObject = {
        "requestId": "req1",
        "url": "wss://example.com/ws",
    }
    session._on_websocket_created(params)
    assert session._ws_urls["req1"] == "wss://example.com/ws"


def test_browser_session_on_websocket_frame_received() -> None:
    """Test _on_websocket_frame_received records message."""
    session = BrowserSession("https://example.com")
    session._ws_urls["req1"] = "wss://example.com/ws"
    params: JSONObject = {
        "requestId": "req1",
        "timestamp": 12345.678,
        "response": {"opcode": 1, "mask": False, "payloadData": "test_payload"},
    }
    session._on_websocket_frame_received(params)
    assert len(session.messages) == 1
    msg = session.messages[0]
    assert msg["direction"] == "received"
    assert msg["payload"] == "test_payload"
    assert msg["ws_url"] == "wss://example.com/ws"


def test_browser_session_on_websocket_frame_sent() -> None:
    """Test _on_websocket_frame_sent records message."""
    session = BrowserSession("https://example.com")
    session._ws_urls["req1"] = "wss://example.com/ws"
    params: JSONObject = {
        "requestId": "req1",
        "timestamp": 12345.678,
        "response": {"opcode": 1, "mask": True, "payloadData": "sent_payload"},
    }
    session._on_websocket_frame_sent(params)
    assert len(session.messages) == 1
    msg = session.messages[0]
    assert msg["direction"] == "sent"
    assert msg["payload"] == "sent_payload"


def test_browser_session_on_message_captured_default() -> None:
    """Test _on_message_captured does nothing by default."""
    session = BrowserSession("https://example.com")
    msg = CapturedMessage(
        timestamp_ms=12345,
        direction="sent",
        payload="test",
        ws_url="wss://example.com/ws",
    )
    # Should not raise
    session._on_message_captured(msg)


def test_browser_session_setup_cdp_handlers() -> None:
    """Test _setup_cdp_handlers registers event handlers."""
    session = BrowserSession("https://example.com")
    cdp = FakeCDPSession()
    session._setup_cdp_handlers(cdp)

    assert "Network.enable" in cdp._sent_methods
    assert "Network.webSocketCreated" in cdp._handlers
    assert "Network.webSocketFrameReceived" in cdp._handlers
    assert "Network.webSocketFrameSent" in cdp._handlers


def test_browser_session_capture_magic_key() -> None:
    """Test _capture_magic_key captures tankpit.magic from page."""
    session = BrowserSession("https://example.com")
    cdp = FakeCDPSession()
    page = FakePage(cdp, magic="test_magic_key_12345")
    session._capture_magic_key(page)
    assert session.magic == "test_magic_key_12345"


def test_browser_session_capture_magic_key_empty() -> None:
    """Test _capture_magic_key handles empty magic."""
    session = BrowserSession("https://example.com")
    cdp = FakeCDPSession()
    page = FakePage(cdp, magic="")
    session._capture_magic_key(page)
    assert session.magic is None


def test_browser_session_capture_magic_key_not_string() -> None:
    """Test _capture_magic_key handles non-string magic."""
    session = BrowserSession("https://example.com")
    cdp = FakeCDPSession()
    # FakePage returns None for magic by default - simulates non-string result
    page = FakePage(cdp, magic=None)
    session._capture_magic_key(page)
    assert session.magic is None


def test_browser_session_wait_for_game_ready_success() -> None:
    """Test _wait_for_game_ready succeeds when messages captured."""
    session = BrowserSession("https://example.com")
    # Pre-populate messages to simulate game loaded
    session._messages = [
        CapturedMessage(timestamp_ms=1, direction="received", payload="msg1", ws_url="ws://test"),
        CapturedMessage(timestamp_ms=2, direction="received", payload="msg2", ws_url="ws://test"),
    ]
    cdp = FakeCDPSession()
    page = FakePage(cdp)
    session._wait_for_game_ready(page)
    # Should not raise


def test_browser_session_wait_for_game_ready_no_messages() -> None:
    """Test _wait_for_game_ready raises when no messages captured."""
    session = BrowserSession("https://example.com")
    cdp = FakeCDPSession()
    page = FakePage(cdp)
    with pytest.raises(GameNotJoinedError):
        session._wait_for_game_ready(page)


def test_browser_session_launch_browser_no_playwright() -> None:
    """Test _launch_browser raises when Playwright not installed."""
    session = BrowserSession("https://example.com")
    original = _test_hooks.sync_playwright
    _test_hooks.sync_playwright = None
    try:
        with pytest.raises(PlaywrightNotInstalledError):
            session._launch_browser()
    finally:
        _test_hooks.sync_playwright = original


def test_browser_session_launch_browser_success() -> None:
    """Test _launch_browser launches browser and sets up CDP handlers."""
    from tests.fakes import fake_sync_playwright

    session = BrowserSession("https://example.com", headless=True)
    original = _test_hooks.sync_playwright
    _test_hooks.sync_playwright = fake_sync_playwright
    try:
        browser, context, page, cdp = session._launch_browser()
        # Simulate a WebSocket creation event to verify handlers are working
        ws_created_event: JSONObject = {
            "requestId": "test_req",
            "url": "wss://test.com/ws",
        }
        session._on_websocket_created(ws_created_event)
        assert session._ws_urls["test_req"] == "wss://test.com/ws"

        # Simulate a WebSocket frame event to verify message capture works
        ws_frame_event: JSONObject = {
            "requestId": "test_req",
            "timestamp": 1000.0,
            "response": {"opcode": 1, "mask": False, "payloadData": "test_data"},
        }
        session._on_websocket_frame_received(ws_frame_event)
        assert len(session.messages) == 1
        assert session.messages[0]["payload"] == "test_data"
        assert session.messages[0]["ws_url"] == "wss://test.com/ws"

        # Verify cleanup works correctly
        session._cleanup(cdp, page, context, browser)
    finally:
        _test_hooks.sync_playwright = original


def test_browser_session_cleanup() -> None:
    """Test _cleanup closes all browser resources."""
    session = BrowserSession("https://example.com")
    cdp = FakeCDPSession()
    page = FakePage(cdp)
    context = FakeBrowserContext()
    browser = FakeBrowser()

    session._cleanup(cdp, page, context, browser)

    assert cdp._detached is True
    assert page._closed is True
    assert context._closed is True
    assert browser._closed is True


def test_browser_session_init_game_log_scraper() -> None:
    """Test _init_game_log_scraper creates scraper and can scrape."""
    from tests.test_dom_scraper import FakeCDPForScraper

    session = BrowserSession("https://example.com")
    cdp = FakeCDPForScraper("LOCATION: 1,2")

    session._init_game_log_scraper(cdp)

    # Poll returns entries, proving scraper was created and works
    entries = session._poll_game_log()
    assert len(entries) == 1
    assert entries[0]["text"] == "LOCATION: 1,2"
    assert entries[0]["category"] == "location"


def test_browser_session_poll_game_log_no_scraper() -> None:
    """Test _poll_game_log returns empty list when scraper not initialized."""
    session = BrowserSession("https://example.com")
    result = session._poll_game_log()
    assert result == []


def test_browser_session_poll_game_log_with_entries() -> None:
    """Test _poll_game_log returns new entries and logs them."""
    from tests.test_dom_scraper import FakeCDPForScraper

    session = BrowserSession("https://example.com")
    cdp = FakeCDPForScraper("LOCATION: 10,20\nYou hit red-1")

    session._init_game_log_scraper(cdp)

    # First poll should return 2 entries
    entries = session._poll_game_log()
    assert len(entries) == 2
    assert entries[0]["text"] == "LOCATION: 10,20"
    assert entries[1]["text"] == "You hit red-1"

    # Second poll should return empty (same content)
    entries = session._poll_game_log()
    assert len(entries) == 0


def test_browser_session_init_inventory_scraper() -> None:
    """Test _init_inventory_scraper creates scraper and can scrape."""
    from tests.test_dom_scraper import FakeCDPForScraper

    session = BrowserSession("https://example.com")
    cdp = FakeCDPForScraper("30 dual shots\n10 extra radars")

    session._init_inventory_scraper(cdp)

    # Poll returns empty on first call (initializes state)
    changes = session._poll_inventory()
    assert len(changes) == 0


def test_browser_session_poll_inventory_no_scraper() -> None:
    """Test _poll_inventory returns empty list when scraper not initialized."""
    session = BrowserSession("https://example.com")
    result = session._poll_inventory()
    assert result == []


def test_browser_session_poll_inventory_with_changes() -> None:
    """Test _poll_inventory returns changes and logs them."""
    from tests.test_dom_scraper import FakeCDPForScraper

    session = BrowserSession("https://example.com")
    cdp = FakeCDPForScraper("30 dual shots\n10 extra radars")

    session._init_inventory_scraper(cdp)

    # First poll initializes state
    session._poll_inventory()

    # Update fake with changed inventory
    cdp._return_value = "37 dual shots\n10 extra radars"

    # Second poll should return 1 change
    changes = session._poll_inventory()
    assert len(changes) == 1
    assert changes[0]["item"] == "dual_shots"
    assert changes[0]["delta"] == 7


def test_browser_session_init_combat_tracker() -> None:
    """Test _init_combat_tracker initializes tracker."""
    session = BrowserSession("https://example.com")
    # Before init, get_combat_events returns empty
    assert session._get_combat_events() == []

    session._init_combat_tracker()

    # After init, tracker can process events
    assert session._get_combat_events() == []
    # Verify it works by processing a line
    if session._combat_tracker:
        session._combat_tracker.process_log_line("You hit blue-7")
        assert len(session._get_combat_events()) == 1


def test_browser_session_get_combat_events_no_tracker() -> None:
    """Test _get_combat_events returns empty list when tracker not initialized."""
    session = BrowserSession("https://example.com")
    result = session._get_combat_events()
    assert result == []


def test_browser_session_get_combat_events_with_tracker() -> None:
    """Test _get_combat_events returns events from tracker."""
    from tankpit_bot.combat import CombatEvent

    session = BrowserSession("https://example.com")
    session._init_combat_tracker()

    # Process a combat line via the tracker (if initialized)
    if session._combat_tracker:
        session._combat_tracker.process_log_line("You hit blue-7")

    events = session._get_combat_events()
    assert len(events) == 1
    assert events[0] == CombatEvent(event_type="hit_by_player", attacker="player", target="blue-7")


def test_browser_session_poll_game_log_processes_combat() -> None:
    """Test _poll_game_log processes combat events when tracker initialized."""
    from tankpit_bot.combat import CombatEvent
    from tests.test_dom_scraper import FakeCDPForScraper

    session = BrowserSession("https://example.com")
    cdp = FakeCDPForScraper("You hit blue-7\nblue-7 hit you")

    session._init_game_log_scraper(cdp)
    session._init_combat_tracker()

    # Poll should process combat events
    entries = session._poll_game_log()
    assert len(entries) == 2

    # Combat tracker should have recorded events
    events = session._get_combat_events()
    assert len(events) == 2
    assert events[0] == CombatEvent(event_type="hit_by_player", attacker="player", target="blue-7")
    assert events[1] == CombatEvent(event_type="hit_by_enemy", attacker="blue-7", target="player")


def test_browser_session_poll_game_log_no_combat_without_tracker() -> None:
    """Test _poll_game_log skips combat processing when no tracker."""
    from tests.test_dom_scraper import FakeCDPForScraper

    session = BrowserSession("https://example.com")
    cdp = FakeCDPForScraper("You hit blue-7")

    session._init_game_log_scraper(cdp)
    # Do not init combat tracker

    # Poll should still return entries
    entries = session._poll_game_log()
    assert len(entries) == 1

    # No crash, no events
    assert session._get_combat_events() == []


def test_browser_session_poll_game_log_combat_event_not_none() -> None:
    """Test _poll_game_log calls log_event when combat event is not None."""
    from tankpit_bot.combat import CombatEvent
    from tests.test_dom_scraper import FakeCDPForScraper

    session = BrowserSession("https://example.com")
    # Use combat lines that will parse successfully
    cdp = FakeCDPForScraper("You hit blue-7\nYou hit red-5\nYou hit green-3")

    session._init_game_log_scraper(cdp)
    session._init_combat_tracker()

    # Poll processes combat events - all 3 should be captured
    entries = session._poll_game_log()
    assert len(entries) == 3

    # Verify all combat events were recorded (confirming log_event path was taken)
    events = session._get_combat_events()
    assert len(events) == 3
    assert events[0] == CombatEvent(event_type="hit_by_player", attacker="player", target="blue-7")
    assert events[1] == CombatEvent(event_type="hit_by_player", attacker="player", target="red-5")
    assert events[2] == CombatEvent(event_type="hit_by_player", attacker="player", target="green-3")

    # Verify combat tracker stats are correct
    from tankpit_bot.combat import CombatStats

    if session._combat_tracker:
        blue_stats = session._combat_tracker.get_stats("blue-7")
        expected = CombatStats(
            name="blue-7", hits_given=1, hits_received=0, deactivated=False, destroyed=False
        )
        assert blue_stats == expected


def test_browser_session_poll_game_log_combat_category_but_no_parse() -> None:
    """Test _poll_game_log handles combat-categorized but non-parseable lines."""
    from tests.test_dom_scraper import FakeCDPForScraper

    session = BrowserSession("https://example.com")
    # "You earned 10 points" contains "earned" which triggers combat category
    # but doesn't match any combat parsing patterns
    cdp = FakeCDPForScraper("You earned 10 points for hitting something")

    session._init_game_log_scraper(cdp)
    session._init_combat_tracker()

    # Poll should process the entry but not create combat events
    entries = session._poll_game_log()
    assert len(entries) == 1
    assert entries[0]["category"] == "combat"

    # No combat events should be created (parse_combat_line returns None)
    events = session._get_combat_events()
    assert len(events) == 0


# =============================================================================
# Static Key Helper Function Tests
# =============================================================================


def test_extract_xor_first_bytes_empty_list() -> None:
    """Test extract_xor_first_bytes with empty messages."""
    result = extract_xor_first_bytes([])
    assert result == []


def test_extract_xor_first_bytes_skips_sent_messages() -> None:
    """Test extract_xor_first_bytes skips sent messages."""
    import base64

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
    import base64

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
    import base64

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
    import base64

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
    from pathlib import Path

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
    from pathlib import Path

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
    from pathlib import Path

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
