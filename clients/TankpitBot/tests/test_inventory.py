"""Tests for tankpit_bot.inventory module.

The DOM-scraping path (``InventoryScraper``, ``parse_inventory``,
``scrape_inventory_text``, ``SCRAPE_INVENTORY_JS``, ``ITEM_NAME_MAP``)
was deleted 2026-06-19 along with its ~250-line test suite; wire-decoded
0x49 / 0x67 / 0x74 messages are now the authoritative inventory source.
Remaining coverage in this file exercises ``diff_inventory`` and the
inventory TypedDict codecs.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.inventory import (
    InventoryChange,
    InventoryItem,
    InventoryState,
    decode_inventory_change,
    decode_inventory_item,
    decode_inventory_state,
    diff_inventory,
    encode_inventory_change,
    encode_inventory_item,
    encode_inventory_state,
    validate_item_type,
)

# =============================================================================
# diff_inventory tests
# =============================================================================


def test_diff_inventory_no_changes() -> None:
    """``diff_inventory`` returns an empty list when no fields changed."""
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
    """``diff_inventory`` detects per-item count changes with signed deltas."""
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
    """``diff_inventory`` detects enabled-flag changes with zero delta."""
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
# validate_item_type tests
# =============================================================================


def test_validate_item_type_all_values() -> None:
    """``validate_item_type`` accepts every wire-known item label."""
    assert validate_item_type("armor_shields") == "armor_shields"
    assert validate_item_type("dual_shots") == "dual_shots"
    assert validate_item_type("missile_shots") == "missile_shots"
    assert validate_item_type("homing_shots") == "homing_shots"
    assert validate_item_type("extra_radars") == "extra_radars"


def test_validate_item_type_invalid() -> None:
    """``validate_item_type`` raises ``ValueError`` on unknown labels."""
    with pytest.raises(ValueError, match="Invalid item type"):
        validate_item_type("invalid_item")


# =============================================================================
# Codec tests
# =============================================================================


def test_encode_inventory_item() -> None:
    """``encode_inventory_item`` produces a JSON-serializable dict."""
    item: InventoryItem = {"count": 30, "enabled": True}
    encoded = encode_inventory_item(item)
    assert encoded["count"] == 30
    assert encoded["enabled"] is True


def test_decode_inventory_item() -> None:
    """``decode_inventory_item`` returns a validated ``InventoryItem``."""
    obj: JSONObject = {"count": 25, "enabled": False}
    item = decode_inventory_item(obj)
    assert item["count"] == 25
    assert item["enabled"] is False


def test_decode_inventory_item_missing_count() -> None:
    """``decode_inventory_item`` raises ``JSONTypeError`` on missing count."""
    obj: JSONObject = {"enabled": True}
    with pytest.raises(JSONTypeError, match="Missing required field 'count'"):
        decode_inventory_item(obj)


def test_decode_inventory_item_missing_enabled() -> None:
    """``decode_inventory_item`` raises ``JSONTypeError`` on missing enabled."""
    obj: JSONObject = {"count": 10}
    with pytest.raises(JSONTypeError, match="Missing required field 'enabled'"):
        decode_inventory_item(obj)


def test_encode_inventory_state() -> None:
    """``encode_inventory_state`` round-trips through ``decode_inventory_state``."""
    state: InventoryState = {
        "armor_shields": InventoryItem(count=30, enabled=False),
        "dual_shots": InventoryItem(count=25, enabled=True),
        "missile_shots": InventoryItem(count=20, enabled=False),
        "homing_shots": InventoryItem(count=15, enabled=True),
        "extra_radars": InventoryItem(count=10, enabled=True),
    }
    encoded = encode_inventory_state(state)
    decoded = decode_inventory_state(encoded)
    assert decoded["armor_shields"]["count"] == 30
    assert decoded["dual_shots"]["enabled"] is True


def test_decode_inventory_state() -> None:
    """``decode_inventory_state`` validates and decodes a complete state."""
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
    """A missing item in the state JSON raises ``ValueError``."""
    obj: JSONObject = {
        "armor_shields": {"count": 30, "enabled": False},
        "dual_shots": {"count": 25, "enabled": True},
        # Missing missile_shots, homing_shots, extra_radars
    }
    with pytest.raises(ValueError, match="missile_shots must be a dict"):
        decode_inventory_state(obj)


def test_decode_inventory_state_non_dict_armor() -> None:
    """A non-dict ``armor_shields`` field raises ``ValueError``."""
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
    """A non-dict ``dual_shots`` field raises ``ValueError``."""
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
    """A non-dict ``homing_shots`` field raises ``ValueError``."""
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
    """A non-dict ``extra_radars`` field raises ``ValueError``."""
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
    """``encode_inventory_change`` produces a JSON-serializable dict."""
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
    """``decode_inventory_change`` returns a validated ``InventoryChange``."""
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
    """``decode_inventory_change`` rejects an unknown item label."""
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
    """``encode_inventory_state`` -> ``decode_inventory_state`` is identity."""
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
    """``encode_inventory_change`` -> ``decode_inventory_change`` is identity."""
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
