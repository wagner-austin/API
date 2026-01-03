"""Tests for dom_scraper module."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.dom_scraper import (
    GameLogEntry,
    GameLogScraper,
    GameLogState,
    categorize_log_line,
    decode_game_log_entry,
    decode_game_log_state,
    encode_game_log_entry,
    encode_game_log_state,
    parse_game_log,
    scrape_game_log_text,
    validate_log_category,
)


class FakeCDPForScraper:
    """Fake CDP session for scraper tests.

    Configurable to return different values for Runtime.evaluate.
    """

    def __init__(self, return_value: str = "") -> None:
        """Initialize fake with return value.

        Args:
            return_value: Value to return from Runtime.evaluate.
        """
        self._return_value = return_value
        self.calls: list[tuple[str, JSONObject | None]] = []
        self._handlers: dict[str, list[Callable[[JSONObject], None]]] = {}

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Send CDP command.

        Args:
            method: CDP method name.
            params: Optional parameters.

        Returns:
            Fake result with configured value.
        """
        self.calls.append((method, params))
        return {"result": {"value": self._return_value}}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Register event handler.

        Args:
            event: Event name.
            handler: Event handler function.
        """
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)

    def detach(self) -> None:
        """Detach CDP session."""


class FakeCDPEmptyResult:
    """Fake CDP session that returns empty result (no 'result' key)."""

    def __init__(self) -> None:
        """Initialize fake."""
        self._handlers: dict[str, list[Callable[[JSONObject], None]]] = {}

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Return empty dict (missing 'result' key)."""
        _ = method
        _ = params
        return {}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Register event handler."""
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)

    def detach(self) -> None:
        """Detach CDP session."""


class FakeCDPNonDictResult:
    """Fake CDP session that returns non-dict result."""

    def __init__(self) -> None:
        """Initialize fake."""
        self._handlers: dict[str, list[Callable[[JSONObject], None]]] = {}

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Return result that is not a dict."""
        _ = method
        _ = params
        return {"result": "not a dict"}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Register event handler."""
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)

    def detach(self) -> None:
        """Detach CDP session."""


class FakeCDPNumericValue:
    """Fake CDP session that returns numeric value instead of string."""

    def __init__(self) -> None:
        """Initialize fake."""
        self._handlers: dict[str, list[Callable[[JSONObject], None]]] = {}

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Return result with numeric value."""
        _ = method
        _ = params
        return {"result": {"value": 12345}}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Register event handler."""
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)

    def detach(self) -> None:
        """Detach CDP session."""


def test_scrape_game_log_text_returns_value() -> None:
    """Test scrape_game_log_text extracts value from CDP result."""
    cdp = FakeCDPForScraper("LOCATION: 123,456\nUsing armor shield")
    result = scrape_game_log_text(cdp)
    assert result == "LOCATION: 123,456\nUsing armor shield"
    assert len(cdp.calls) == 1
    assert cdp.calls[0][0] == "Runtime.evaluate"


def test_scrape_game_log_text_handles_empty() -> None:
    """Test scrape_game_log_text returns empty string when no value."""
    cdp = FakeCDPForScraper("")
    result = scrape_game_log_text(cdp)
    assert result == ""


def test_scrape_game_log_text_handles_missing_result() -> None:
    """Test scrape_game_log_text handles missing result object."""
    cdp = FakeCDPEmptyResult()
    result = scrape_game_log_text(cdp)
    assert result == ""


def test_scrape_game_log_text_handles_non_dict_result() -> None:
    """Test scrape_game_log_text handles non-dict result."""
    cdp = FakeCDPNonDictResult()
    result = scrape_game_log_text(cdp)
    assert result == ""


def test_scrape_game_log_text_handles_non_string_value() -> None:
    """Test scrape_game_log_text handles non-string value."""
    cdp = FakeCDPNumericValue()
    result = scrape_game_log_text(cdp)
    assert result == ""


def test_categorize_log_line_location() -> None:
    """Test categorize_log_line identifies LOCATION entries."""
    assert categorize_log_line("LOCATION: 123,456") == "location"


def test_categorize_log_line_combat_hit() -> None:
    """Test categorize_log_line identifies combat hit entries."""
    assert categorize_log_line("You hit blue-7") == "combat"
    assert categorize_log_line("red-1 hit you") == "combat"


def test_categorize_log_line_combat_deactivated() -> None:
    """Test categorize_log_line identifies deactivation entries."""
    assert categorize_log_line("Tank deactivated") == "combat"


def test_categorize_log_line_combat_destroyed() -> None:
    """Test categorize_log_line identifies destruction entries."""
    assert categorize_log_line("Enemy destroyed") == "combat"


def test_categorize_log_line_equipment_using_enabled() -> None:
    """Test categorize_log_line identifies equipment enabled entries."""
    assert categorize_log_line("Using armor shield enabled") == "equipment"


def test_categorize_log_line_equipment_gained() -> None:
    """Test categorize_log_line identifies equipment gain entries."""
    assert categorize_log_line("6 dual shots gained") == "equipment"


def test_categorize_log_line_equipment_disabled() -> None:
    """Test categorize_log_line identifies disabled entries."""
    assert categorize_log_line("Using extra radar disabled") == "equipment"


def test_categorize_log_line_tip() -> None:
    """Test categorize_log_line identifies tip entries."""
    assert categorize_log_line("Tip (press N): Missiles fly over rocks") == "tip"


def test_categorize_log_line_action_autoscroll() -> None:
    """Test categorize_log_line identifies autoscroll entries."""
    assert categorize_log_line("Autoscroll: ON") == "action"


def test_categorize_log_line_action_radar_detected() -> None:
    """Test categorize_log_line identifies radar detection entries."""
    assert categorize_log_line("blue-7 (private) detected to W [57,135]") == "action"


def test_categorize_log_line_teleport() -> None:
    """Test categorize_log_line identifies teleport entries."""
    assert categorize_log_line("Teleporting to [117,136]") == "teleport"


def test_categorize_log_line_inventory_full() -> None:
    """Test categorize_log_line identifies inventory full as equipment."""
    assert categorize_log_line("Inventory full") == "equipment"


def test_categorize_log_line_you_earned() -> None:
    """Test categorize_log_line identifies earned points as combat."""
    assert categorize_log_line("You earned extra points") == "combat"


def test_categorize_log_line_you_fire() -> None:
    """Test categorize_log_line identifies fire action."""
    assert categorize_log_line("You fire") == "action"


def test_categorize_log_line_extend_view() -> None:
    """Test categorize_log_line identifies extend view as action."""
    assert categorize_log_line("Extend view N") == "action"
    assert categorize_log_line("Extend view W") == "action"


def test_categorize_log_line_zoom() -> None:
    """Test categorize_log_line identifies zoom as action."""
    assert categorize_log_line("Zoom in") == "action"


def test_categorize_log_line_obstacle() -> None:
    """Test categorize_log_line identifies obstacle actions."""
    assert categorize_log_line("Obstacle picked up") == "action"
    assert categorize_log_line("Obstacle dropped") == "action"


def test_categorize_log_line_fuel() -> None:
    """Test categorize_log_line identifies fuel deposit as action."""
    assert categorize_log_line("Fuel deposited") == "action"


def test_categorize_log_line_cant_go() -> None:
    """Test categorize_log_line identifies movement feedback as action."""
    assert categorize_log_line("You can't go there!") == "action"
    assert categorize_log_line("You are already there!") == "action"


def test_categorize_log_line_other() -> None:
    """Test categorize_log_line returns other for unknown entries."""
    assert categorize_log_line("Some random text") == "other"


def test_parse_game_log_empty() -> None:
    """Test parse_game_log handles empty input."""
    state = parse_game_log("")
    assert state["raw_text"] == ""
    assert state["entries"] == []
    assert state["location"] == ""


def test_parse_game_log_extracts_location() -> None:
    """Test parse_game_log extracts location from text."""
    raw = "LOCATION: 123,456\nUsing armor"
    state = parse_game_log(raw)
    assert state["location"] == "123,456"


def test_parse_game_log_parses_entries() -> None:
    """Test parse_game_log creates entries for each line."""
    raw = "LOCATION: 100,200\nYou hit red-1\n6 dual shots gained"
    state = parse_game_log(raw)
    assert len(state["entries"]) == 3
    assert state["entries"][0]["text"] == "LOCATION: 100,200"
    assert state["entries"][0]["category"] == "location"
    assert state["entries"][1]["text"] == "You hit red-1"
    assert state["entries"][1]["category"] == "combat"
    assert state["entries"][2]["text"] == "6 dual shots gained"
    assert state["entries"][2]["category"] == "equipment"


def test_parse_game_log_skips_empty_lines() -> None:
    """Test parse_game_log skips empty lines."""
    raw = "LOCATION: 50,60\n\n\nUsing armor"
    state = parse_game_log(raw)
    assert len(state["entries"]) == 2


def test_game_log_scraper_scrape() -> None:
    """Test GameLogScraper.scrape returns current state."""
    cdp = FakeCDPForScraper("LOCATION: 10,20\nTest entry")
    scraper = GameLogScraper(cdp)
    state = scraper.scrape()
    assert state["location"] == "10,20"
    assert len(state["entries"]) == 2


def test_game_log_scraper_get_new_entries_first_call() -> None:
    """Test GameLogScraper.get_new_entries returns all on first call."""
    cdp = FakeCDPForScraper("LOCATION: 1,2\nEntry one")
    scraper = GameLogScraper(cdp)
    new_entries = scraper.get_new_entries()
    assert len(new_entries) == 2


def test_game_log_scraper_get_new_entries_deduplicates() -> None:
    """Test GameLogScraper.get_new_entries doesn't repeat entries."""
    cdp = FakeCDPForScraper("LOCATION: 1,2\nEntry one")
    scraper = GameLogScraper(cdp)

    # First call gets all
    first = scraper.get_new_entries()
    assert len(first) == 2

    # Second call with same data gets none
    second = scraper.get_new_entries()
    assert len(second) == 0


def test_game_log_scraper_get_new_entries_detects_new() -> None:
    """Test GameLogScraper.get_new_entries detects new entries."""
    cdp = FakeCDPForScraper("LOCATION: 1,2")
    scraper = GameLogScraper(cdp)

    # First call
    scraper.get_new_entries()

    # Update fake with new content
    cdp._return_value = "LOCATION: 1,2\nNew entry"
    new_entries = scraper.get_new_entries()
    assert len(new_entries) == 1
    assert new_entries[0]["text"] == "New entry"


def test_game_log_scraper_log_new_entries() -> None:
    """Test GameLogScraper.log_new_entries logs entries."""
    cdp = FakeCDPForScraper("LOCATION: 5,5")
    scraper = GameLogScraper(cdp)
    # Just verify it doesn't raise
    scraper.log_new_entries()


def test_encode_game_log_entry() -> None:
    """Test encode_game_log_entry creates correct dict."""
    entry: GameLogEntry = {"text": "Test text", "category": "combat"}
    encoded = encode_game_log_entry(entry)
    assert encoded["text"] == "Test text"
    assert encoded["category"] == "combat"


def test_encode_game_log_state() -> None:
    """Test encode_game_log_state creates correct dict."""
    entry: GameLogEntry = {"text": "Line one", "category": "action"}
    state: GameLogState = {
        "raw_text": "Line one",
        "entries": [entry],
        "location": "99,88",
    }
    encoded = encode_game_log_state(state)
    assert encoded["raw_text"] == "Line one"
    assert encoded["location"] == "99,88"
    # Verify round-trip via decode
    decoded = decode_game_log_state(encoded)
    assert decoded["raw_text"] == "Line one"
    assert decoded["location"] == "99,88"
    assert len(decoded["entries"]) == 1
    assert decoded["entries"][0]["text"] == "Line one"
    assert decoded["entries"][0]["category"] == "action"


def test_decode_game_log_entry_success() -> None:
    """Test decode_game_log_entry decodes valid entry."""
    obj: JSONObject = {"text": "Test message", "category": "combat"}
    entry = decode_game_log_entry(obj)
    assert entry["text"] == "Test message"
    assert entry["category"] == "combat"


def test_decode_game_log_entry_missing_text() -> None:
    """Test decode_game_log_entry raises on missing text."""
    obj: JSONObject = {"category": "combat"}
    with pytest.raises(JSONTypeError, match="Missing required field 'text'"):
        decode_game_log_entry(obj)


def test_decode_game_log_entry_missing_category() -> None:
    """Test decode_game_log_entry raises on missing category."""
    obj: JSONObject = {"text": "Some text"}
    with pytest.raises(JSONTypeError, match="Missing required field 'category'"):
        decode_game_log_entry(obj)


def test_decode_game_log_entry_invalid_category() -> None:
    """Test decode_game_log_entry raises on invalid category."""
    obj: JSONObject = {"text": "Some text", "category": "invalid_cat"}
    with pytest.raises(ValueError, match="Invalid category"):
        decode_game_log_entry(obj)


def test_decode_game_log_state_success() -> None:
    """Test decode_game_log_state decodes valid state."""
    obj: JSONObject = {
        "raw_text": "LOCATION: 1,2",
        "location": "1,2",
        "entries": [{"text": "LOCATION: 1,2", "category": "location"}],
    }
    state = decode_game_log_state(obj)
    assert state["raw_text"] == "LOCATION: 1,2"
    assert state["location"] == "1,2"
    assert len(state["entries"]) == 1


def test_decode_game_log_state_missing_raw_text() -> None:
    """Test decode_game_log_state raises on missing raw_text."""
    obj: JSONObject = {"location": "1,2", "entries": []}
    with pytest.raises(JSONTypeError, match="Missing required field 'raw_text'"):
        decode_game_log_state(obj)


def test_decode_game_log_state_entry_not_dict() -> None:
    """Test decode_game_log_state raises when entry is not a dict."""
    obj: JSONObject = {
        "raw_text": "test",
        "location": "1,2",
        "entries": ["not a dict"],
    }
    with pytest.raises(ValueError, match="Entry at index 0 must be a dict"):
        decode_game_log_state(obj)


def test_parse_game_log_location_empty_value() -> None:
    """Test parse_game_log handles LOCATION: with no value after it."""
    raw = "LOCATION:\nUsing armor"
    state = parse_game_log(raw)
    assert state["location"] == ""  # Empty because no coordinates after LOCATION:
    assert len(state["entries"]) == 2


def test_decode_game_log_entry_all_categories() -> None:
    """Test decode_game_log_entry handles all category values."""
    # Test all categories to ensure full coverage
    assert validate_log_category("location") == "location"
    assert validate_log_category("action") == "action"
    assert validate_log_category("combat") == "combat"
    assert validate_log_category("equipment") == "equipment"
    assert validate_log_category("teleport") == "teleport"
    assert validate_log_category("tip") == "tip"
    assert validate_log_category("other") == "other"

    # Test via decode to cover entry creation
    all_cats = ["location", "action", "combat", "equipment", "teleport", "tip", "other"]
    for cat in all_cats:
        obj: JSONObject = {"text": "Test", "category": cat}
        entry = decode_game_log_entry(obj)
        assert entry["category"] == cat
