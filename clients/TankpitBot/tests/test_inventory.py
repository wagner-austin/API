"""Tests for tankpit_bot.inventory module."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.inventory import (
    InventoryChange,
    InventoryItem,
    InventoryScraper,
    InventoryState,
    decode_inventory_change,
    decode_inventory_item,
    decode_inventory_state,
    diff_inventory,
    encode_inventory_change,
    encode_inventory_item,
    encode_inventory_state,
    parse_inventory,
    scrape_inventory_text,
    validate_item_type,
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


# =============================================================================
# Inventory Scraping Tests
# =============================================================================


def test_scrape_inventory_text_returns_value() -> None:
    """Test scrape_inventory_text extracts value from CDP result."""
    cdp = FakeCDPForScraper("30 armor shields (disabled)\n30 dual shots")
    result = scrape_inventory_text(cdp)
    assert result == "30 armor shields (disabled)\n30 dual shots"


def test_scrape_inventory_text_handles_empty() -> None:
    """Test scrape_inventory_text returns empty string when no value."""
    cdp = FakeCDPForScraper("")
    result = scrape_inventory_text(cdp)
    assert result == ""


def test_scrape_inventory_text_handles_missing_result() -> None:
    """Test scrape_inventory_text handles missing result object."""
    cdp = FakeCDPEmptyResult()
    result = scrape_inventory_text(cdp)
    assert result == ""


def test_scrape_inventory_text_handles_non_dict_result() -> None:
    """Test scrape_inventory_text handles non-dict result."""
    cdp = FakeCDPNonDictResult()
    result = scrape_inventory_text(cdp)
    assert result == ""


def test_scrape_inventory_text_handles_non_string_value() -> None:
    """Test scrape_inventory_text handles non-string value."""
    cdp = FakeCDPNumericValue()
    result = scrape_inventory_text(cdp)
    assert result == ""


def test_parse_inventory_empty() -> None:
    """Test parse_inventory handles empty input."""
    state = parse_inventory("")
    assert state["armor_shields"]["count"] == 0
    assert state["dual_shots"]["count"] == 0
    assert state["missile_shots"]["count"] == 0
    assert state["homing_shots"]["count"] == 0
    assert state["extra_radars"]["count"] == 0


def test_parse_inventory_all_items() -> None:
    """Test parse_inventory parses all inventory items."""
    raw = """30 armor shields (disabled)
25 dual shots
20 missile shots (disabled)
15 homing shots
10 extra radars"""
    state = parse_inventory(raw)
    assert state["armor_shields"]["count"] == 30
    assert state["armor_shields"]["enabled"] is False
    assert state["dual_shots"]["count"] == 25
    assert state["dual_shots"]["enabled"] is True
    assert state["missile_shots"]["count"] == 20
    assert state["missile_shots"]["enabled"] is False
    assert state["homing_shots"]["count"] == 15
    assert state["homing_shots"]["enabled"] is True
    assert state["extra_radars"]["count"] == 10
    assert state["extra_radars"]["enabled"] is True


def test_parse_inventory_skips_unknown_items() -> None:
    """Test parse_inventory skips unknown item names."""
    raw = "30 dual shots\n99 unknown items\n10 extra radars"
    state = parse_inventory(raw)
    assert state["dual_shots"]["count"] == 30
    assert state["extra_radars"]["count"] == 10
    # Unknown items are ignored, armor_shields stays at default
    assert state["armor_shields"]["count"] == 0


def test_parse_inventory_skips_invalid_count() -> None:
    """Test parse_inventory skips lines with non-numeric count."""
    raw = "XX dual shots\n10 extra radars"
    state = parse_inventory(raw)
    assert state["dual_shots"]["count"] == 0  # Stays at default
    assert state["extra_radars"]["count"] == 10


def test_parse_inventory_skips_single_word_lines() -> None:
    """Test parse_inventory skips lines without item name."""
    raw = "30\n10 extra radars"
    state = parse_inventory(raw)
    assert state["extra_radars"]["count"] == 10


def test_parse_inventory_extra_radars_not_last() -> None:
    """Test parse_inventory handles extra_radars when not last item."""
    # extra_radars before homing_shots (covers branch 313->300)
    # After processing extra_radars, loop continues to process homing_shots
    raw = "10 extra radars\n15 homing shots\n30 armor shields"
    state = parse_inventory(raw)
    assert state["extra_radars"]["count"] == 10
    assert state["homing_shots"]["count"] == 15
    assert state["armor_shields"]["count"] == 30


def test_diff_inventory_no_changes() -> None:
    """Test diff_inventory returns empty list when no changes."""
    old: InventoryState = {
        "armor_shields": InventoryItem(count=30, enabled=True),
        "dual_shots": InventoryItem(count=25, enabled=True),
        "missile_shots": InventoryItem(count=20, enabled=True),
        "homing_shots": InventoryItem(count=15, enabled=True),
        "extra_radars": InventoryItem(count=10, enabled=True),
    }
    new: InventoryState = {
        "armor_shields": InventoryItem(count=30, enabled=True),
        "dual_shots": InventoryItem(count=25, enabled=True),
        "missile_shots": InventoryItem(count=20, enabled=True),
        "homing_shots": InventoryItem(count=15, enabled=True),
        "extra_radars": InventoryItem(count=10, enabled=True),
    }
    changes = diff_inventory(old, new)
    assert len(changes) == 0


def test_diff_inventory_count_change() -> None:
    """Test diff_inventory detects count changes."""
    old: InventoryState = {
        "armor_shields": InventoryItem(count=30, enabled=True),
        "dual_shots": InventoryItem(count=25, enabled=True),
        "missile_shots": InventoryItem(count=20, enabled=True),
        "homing_shots": InventoryItem(count=15, enabled=True),
        "extra_radars": InventoryItem(count=10, enabled=True),
    }
    new: InventoryState = {
        "armor_shields": InventoryItem(count=30, enabled=True),
        "dual_shots": InventoryItem(count=32, enabled=True),  # +7
        "missile_shots": InventoryItem(count=20, enabled=True),
        "homing_shots": InventoryItem(count=15, enabled=True),
        "extra_radars": InventoryItem(count=9, enabled=True),  # -1
    }
    changes = diff_inventory(old, new)
    assert len(changes) == 2

    dual_change = next(c for c in changes if c["item"] == "dual_shots")
    assert dual_change["old_count"] == 25
    assert dual_change["new_count"] == 32
    assert dual_change["delta"] == 7
    assert dual_change["enabled_changed"] is False

    radar_change = next(c for c in changes if c["item"] == "extra_radars")
    assert radar_change["delta"] == -1


def test_diff_inventory_enabled_change() -> None:
    """Test diff_inventory detects enabled state changes."""
    old: InventoryState = {
        "armor_shields": InventoryItem(count=30, enabled=True),
        "dual_shots": InventoryItem(count=25, enabled=True),
        "missile_shots": InventoryItem(count=20, enabled=True),
        "homing_shots": InventoryItem(count=15, enabled=True),
        "extra_radars": InventoryItem(count=10, enabled=True),
    }
    new: InventoryState = {
        "armor_shields": InventoryItem(count=30, enabled=False),  # disabled
        "dual_shots": InventoryItem(count=25, enabled=True),
        "missile_shots": InventoryItem(count=20, enabled=True),
        "homing_shots": InventoryItem(count=15, enabled=True),
        "extra_radars": InventoryItem(count=10, enabled=True),
    }
    changes = diff_inventory(old, new)
    assert len(changes) == 1
    assert changes[0]["item"] == "armor_shields"
    assert changes[0]["enabled_changed"] is True
    assert changes[0]["now_enabled"] is False
    assert changes[0]["delta"] == 0


# =============================================================================
# InventoryScraper Tests
# =============================================================================


def test_inventory_scraper_scrape() -> None:
    """Test InventoryScraper.scrape returns current state."""
    cdp = FakeCDPForScraper("30 dual shots\n10 extra radars")
    scraper = InventoryScraper(cdp)
    state = scraper.scrape()
    assert state["dual_shots"]["count"] == 30
    assert state["extra_radars"]["count"] == 10


def test_inventory_scraper_get_changes_first_call() -> None:
    """Test InventoryScraper.get_changes returns empty on first call."""
    cdp = FakeCDPForScraper("30 dual shots")
    scraper = InventoryScraper(cdp)
    changes = scraper.get_changes()
    assert len(changes) == 0  # First call initializes state


def test_inventory_scraper_get_changes_detects_change() -> None:
    """Test InventoryScraper.get_changes detects changes."""
    cdp = FakeCDPForScraper("30 dual shots\n10 extra radars")
    scraper = InventoryScraper(cdp)

    # First call initializes
    scraper.get_changes()

    # Update fake with changed inventory
    cdp._return_value = "37 dual shots\n10 extra radars"
    changes = scraper.get_changes()
    assert len(changes) == 1
    assert changes[0]["item"] == "dual_shots"
    assert changes[0]["delta"] == 7


def test_inventory_scraper_log_changes_gained() -> None:
    """Test InventoryScraper.log_changes logs gained items."""
    cdp = FakeCDPForScraper("30 dual shots")
    scraper = InventoryScraper(cdp)
    scraper.get_changes()  # Initialize

    cdp._return_value = "35 dual shots"
    changes = scraper.log_changes()
    assert len(changes) == 1
    assert changes[0]["delta"] == 5


def test_inventory_scraper_log_changes_used() -> None:
    """Test InventoryScraper.log_changes logs used items."""
    cdp = FakeCDPForScraper("30 dual shots")
    scraper = InventoryScraper(cdp)
    scraper.get_changes()  # Initialize

    cdp._return_value = "28 dual shots"
    changes = scraper.log_changes()
    assert len(changes) == 1
    assert changes[0]["delta"] == -2


def test_inventory_scraper_log_changes_toggle() -> None:
    """Test InventoryScraper.log_changes logs toggle changes."""
    cdp = FakeCDPForScraper("30 armor shields")
    scraper = InventoryScraper(cdp)
    scraper.get_changes()  # Initialize

    cdp._return_value = "30 armor shields (disabled)"
    changes = scraper.log_changes()
    assert len(changes) == 1
    assert changes[0]["enabled_changed"] is True
    assert changes[0]["now_enabled"] is False


def test_inventory_scraper_log_changes_toggle_enabled() -> None:
    """Test InventoryScraper.log_changes logs re-enable changes."""
    cdp = FakeCDPForScraper("30 armor shields (disabled)")
    scraper = InventoryScraper(cdp)
    scraper.get_changes()  # Initialize

    cdp._return_value = "30 armor shields"
    changes = scraper.log_changes()
    assert len(changes) == 1
    assert changes[0]["enabled_changed"] is True
    assert changes[0]["now_enabled"] is True


# =============================================================================
# Inventory Encode/Decode Tests
# =============================================================================


def test_validate_item_type_all_values() -> None:
    """Test validate_item_type handles all item types."""
    assert validate_item_type("armor_shields") == "armor_shields"
    assert validate_item_type("dual_shots") == "dual_shots"
    assert validate_item_type("missile_shots") == "missile_shots"
    assert validate_item_type("homing_shots") == "homing_shots"
    assert validate_item_type("extra_radars") == "extra_radars"


def test_validate_item_type_invalid() -> None:
    """Test validate_item_type raises on invalid type."""
    with pytest.raises(ValueError, match="Invalid item type"):
        validate_item_type("invalid_item")


def test_encode_inventory_item() -> None:
    """Test encode_inventory_item creates correct dict."""
    item: InventoryItem = {"count": 30, "enabled": True}
    encoded = encode_inventory_item(item)
    assert encoded["count"] == 30
    assert encoded["enabled"] is True


def test_decode_inventory_item() -> None:
    """Test decode_inventory_item decodes valid item."""
    obj: JSONObject = {"count": 25, "enabled": False}
    item = decode_inventory_item(obj)
    assert item["count"] == 25
    assert item["enabled"] is False


def test_decode_inventory_item_missing_count() -> None:
    """Test decode_inventory_item raises on missing count."""
    obj: JSONObject = {"enabled": True}
    with pytest.raises(JSONTypeError, match="Missing required field 'count'"):
        decode_inventory_item(obj)


def test_decode_inventory_item_missing_enabled() -> None:
    """Test decode_inventory_item raises on missing enabled."""
    obj: JSONObject = {"count": 10}
    with pytest.raises(JSONTypeError, match="Missing required field 'enabled'"):
        decode_inventory_item(obj)


def test_encode_inventory_state() -> None:
    """Test encode_inventory_state creates correct dict."""
    state: InventoryState = {
        "armor_shields": InventoryItem(count=30, enabled=False),
        "dual_shots": InventoryItem(count=25, enabled=True),
        "missile_shots": InventoryItem(count=20, enabled=False),
        "homing_shots": InventoryItem(count=15, enabled=True),
        "extra_radars": InventoryItem(count=10, enabled=True),
    }
    encoded = encode_inventory_state(state)
    # Verify by decoding back and checking values
    decoded = decode_inventory_state(encoded)
    assert decoded["armor_shields"]["count"] == 30
    assert decoded["dual_shots"]["enabled"] is True


def test_decode_inventory_state() -> None:
    """Test decode_inventory_state decodes valid state."""
    obj: JSONObject = {
        "armor_shields": {"count": 30, "enabled": False},
        "dual_shots": {"count": 25, "enabled": True},
        "missile_shots": {"count": 20, "enabled": False},
        "homing_shots": {"count": 15, "enabled": True},
        "extra_radars": {"count": 10, "enabled": True},
    }
    state = decode_inventory_state(obj)
    assert state["armor_shields"]["count"] == 30
    assert state["armor_shields"]["enabled"] is False
    assert state["dual_shots"]["count"] == 25


def test_decode_inventory_state_missing_item() -> None:
    """Test decode_inventory_state raises on missing item."""
    obj: JSONObject = {
        "armor_shields": {"count": 30, "enabled": False},
        "dual_shots": {"count": 25, "enabled": True},
        # Missing missile_shots, homing_shots, extra_radars
    }
    with pytest.raises(ValueError, match="missile_shots must be a dict"):
        decode_inventory_state(obj)


def test_decode_inventory_state_non_dict_armor() -> None:
    """Test decode_inventory_state raises on non-dict armor_shields."""
    obj: JSONObject = {
        "armor_shields": "not a dict",
        "dual_shots": {"count": 25, "enabled": True},
        "missile_shots": {"count": 20, "enabled": False},
        "homing_shots": {"count": 15, "enabled": True},
        "extra_radars": {"count": 10, "enabled": True},
    }
    with pytest.raises(ValueError, match="armor_shields must be a dict"):
        decode_inventory_state(obj)


def test_decode_inventory_state_non_dict_dual() -> None:
    """Test decode_inventory_state raises on non-dict dual_shots."""
    obj: JSONObject = {
        "armor_shields": {"count": 30, "enabled": False},
        "dual_shots": "not a dict",
        "missile_shots": {"count": 20, "enabled": False},
        "homing_shots": {"count": 15, "enabled": True},
        "extra_radars": {"count": 10, "enabled": True},
    }
    with pytest.raises(ValueError, match="dual_shots must be a dict"):
        decode_inventory_state(obj)


def test_decode_inventory_state_non_dict_homing() -> None:
    """Test decode_inventory_state raises on non-dict homing_shots."""
    obj: JSONObject = {
        "armor_shields": {"count": 30, "enabled": False},
        "dual_shots": {"count": 25, "enabled": True},
        "missile_shots": {"count": 20, "enabled": False},
        "homing_shots": "not a dict",
        "extra_radars": {"count": 10, "enabled": True},
    }
    with pytest.raises(ValueError, match="homing_shots must be a dict"):
        decode_inventory_state(obj)


def test_decode_inventory_state_non_dict_radar() -> None:
    """Test decode_inventory_state raises on non-dict extra_radars."""
    obj: JSONObject = {
        "armor_shields": {"count": 30, "enabled": False},
        "dual_shots": {"count": 25, "enabled": True},
        "missile_shots": {"count": 20, "enabled": False},
        "homing_shots": {"count": 15, "enabled": True},
        "extra_radars": "not a dict",
    }
    with pytest.raises(ValueError, match="extra_radars must be a dict"):
        decode_inventory_state(obj)


def test_encode_inventory_change() -> None:
    """Test encode_inventory_change creates correct dict."""
    change: InventoryChange = {
        "item": "dual_shots",
        "old_count": 25,
        "new_count": 32,
        "delta": 7,
        "enabled_changed": False,
        "now_enabled": True,
    }
    encoded = encode_inventory_change(change)
    assert encoded["item"] == "dual_shots"
    assert encoded["delta"] == 7


def test_decode_inventory_change() -> None:
    """Test decode_inventory_change decodes valid change."""
    obj: JSONObject = {
        "item": "extra_radars",
        "old_count": 10,
        "new_count": 9,
        "delta": -1,
        "enabled_changed": False,
        "now_enabled": True,
    }
    change = decode_inventory_change(obj)
    assert change["item"] == "extra_radars"
    assert change["delta"] == -1


def test_decode_inventory_change_invalid_item() -> None:
    """Test decode_inventory_change raises on invalid item type."""
    obj: JSONObject = {
        "item": "invalid_item",
        "old_count": 10,
        "new_count": 9,
        "delta": -1,
        "enabled_changed": False,
        "now_enabled": True,
    }
    with pytest.raises(ValueError, match="Invalid item type"):
        decode_inventory_change(obj)


def test_inventory_state_encode_decode_roundtrip() -> None:
    """Test encode/decode roundtrip preserves InventoryState."""
    original: InventoryState = {
        "armor_shields": InventoryItem(count=30, enabled=False),
        "dual_shots": InventoryItem(count=25, enabled=True),
        "missile_shots": InventoryItem(count=20, enabled=False),
        "homing_shots": InventoryItem(count=15, enabled=True),
        "extra_radars": InventoryItem(count=10, enabled=True),
    }
    encoded = encode_inventory_state(original)
    decoded = decode_inventory_state(encoded)
    assert decoded == original


def test_inventory_change_encode_decode_roundtrip() -> None:
    """Test encode/decode roundtrip preserves InventoryChange."""
    original: InventoryChange = {
        "item": "homing_shots",
        "old_count": 15,
        "new_count": 20,
        "delta": 5,
        "enabled_changed": True,
        "now_enabled": False,
    }
    encoded = encode_inventory_change(original)
    decoded = decode_inventory_change(encoded)
    assert decoded == original
