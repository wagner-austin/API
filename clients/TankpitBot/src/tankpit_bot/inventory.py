"""Inventory TypedDicts, diffing, and codecs.

The DOM-scraping path (``InventoryScraper``, ``parse_inventory``,
``scrape_inventory_text``, ``SCRAPE_INVENTORY_JS``, ``ITEM_NAME_MAP``)
was deleted 2026-06-19. Wire-decoded 0x49 / 0x67 / 0x74 messages are the
authoritative inventory source; DOM scraping reads stale UI and was
never wired into the production tick loop.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.json_utils import (
    JSONObject,
    require_bool,
    require_int,
    require_str,
)
from platform_core.logging import get_logger

log = get_logger(__name__)


# =============================================================================
# TypedDicts
# =============================================================================


# Known inventory item types
ItemType = Literal["armor_shields", "dual_shots", "missile_shots", "homing_shots", "extra_radars"]


class InventoryItem(TypedDict):
    """A single inventory item with count and enabled state.

    Attributes:
        count: Number of this item in inventory.
        enabled: Whether the item is currently enabled (True if no "(disabled)" suffix).
    """

    count: int
    enabled: bool


class InventoryState(TypedDict):
    """Current inventory state.

    Attributes:
        armor_shields: Armor shield count and state.
        dual_shots: Dual shot count and state.
        missile_shots: Missile shot count and state.
        homing_shots: Homing shot count and state.
        extra_radars: Extra radar count and state.
    """

    armor_shields: InventoryItem
    dual_shots: InventoryItem
    missile_shots: InventoryItem
    homing_shots: InventoryItem
    extra_radars: InventoryItem


class InventoryChange(TypedDict):
    """A change detected in inventory.

    Attributes:
        item: The item type that changed.
        old_count: Previous count.
        new_count: New count.
        delta: Change amount (positive = gained, negative = lost).
        enabled_changed: Whether enabled state changed.
        now_enabled: Current enabled state (if enabled_changed is True).
    """

    item: ItemType
    old_count: int
    new_count: int
    delta: int
    enabled_changed: bool
    now_enabled: bool


def inventory_all_full(state: InventoryState, capacity: int) -> bool:
    """Report whether every inventory slot is at capacity.

    User mechanic (2026-07-18, verbatim): equipment containers "fill
    whatever is empty. you will only get a full inventory message if
    all your items are full." So all-slots-full is exactly the state
    in which an equipment pickup can gain nothing and the server would
    refuse it with 0x52 code 7.

    Args:
        state: Current inventory state.
        capacity: Per-slot capacity for the current rank.

    Returns:
        True when every slot's count is at or above capacity.
    """
    return (
        state["armor_shields"]["count"] >= capacity
        and state["dual_shots"]["count"] >= capacity
        and state["missile_shots"]["count"] >= capacity
        and state["homing_shots"]["count"] >= capacity
        and state["extra_radars"]["count"] >= capacity
    )


def replace_inventory_slot(
    state: InventoryState,
    slot: ItemType,
    item: InventoryItem,
) -> InventoryState:
    """Return a new ``InventoryState`` with one slot replaced.

    Used when only one item changed (e.g. ammo consumed by a hit). The
    other four slots are preserved by reference.

    Args:
        state: Current inventory state.
        slot: Which slot to replace.
        item: New value for that slot.

    Returns:
        Fresh ``InventoryState`` with ``slot`` set to ``item`` and
        every other slot preserved from ``state``.
    """
    return InventoryState(
        armor_shields=item if slot == "armor_shields" else state["armor_shields"],
        dual_shots=item if slot == "dual_shots" else state["dual_shots"],
        missile_shots=item if slot == "missile_shots" else state["missile_shots"],
        homing_shots=item if slot == "homing_shots" else state["homing_shots"],
        extra_radars=item if slot == "extra_radars" else state["extra_radars"],
    )


def diff_inventory(old: InventoryState, new: InventoryState) -> list[InventoryChange]:
    """Compare two inventory states and return list of changes.

    Args:
        old: Previous inventory state.
        new: Current inventory state.

    Returns:
        List of InventoryChange for each item that changed.
    """
    changes: list[InventoryChange] = []
    item_types: list[ItemType] = [
        "armor_shields",
        "dual_shots",
        "missile_shots",
        "homing_shots",
        "extra_radars",
    ]

    for item_type in item_types:
        old_item = old[item_type]
        new_item = new[item_type]

        count_changed = old_item["count"] != new_item["count"]
        enabled_changed = old_item["enabled"] != new_item["enabled"]

        if count_changed or enabled_changed:
            changes.append(
                InventoryChange(
                    item=item_type,
                    old_count=old_item["count"],
                    new_count=new_item["count"],
                    delta=new_item["count"] - old_item["count"],
                    enabled_changed=enabled_changed,
                    now_enabled=new_item["enabled"],
                )
            )

    return changes


# =============================================================================
# Encode/Decode Functions
# =============================================================================


VALID_ITEM_TYPES: frozenset[str] = frozenset(
    ["armor_shields", "dual_shots", "missile_shots", "homing_shots", "extra_radars"]
)


def validate_item_type(value: str) -> ItemType:
    """Validate and narrow a string to an ItemType literal.

    Args:
        value: String value to validate.

    Returns:
        The validated item type as a Literal type.

    Raises:
        ValueError: If value is not a valid item type.
    """
    if value == "armor_shields":
        return "armor_shields"
    if value == "dual_shots":
        return "dual_shots"
    if value == "missile_shots":
        return "missile_shots"
    if value == "homing_shots":
        return "homing_shots"
    if value == "extra_radars":
        return "extra_radars"
    raise ValueError(f"Invalid item type '{value}', must be one of {VALID_ITEM_TYPES}")


def encode_inventory_item(item: InventoryItem) -> JSONObject:
    """Encode InventoryItem to JSON-serializable dict.

    Args:
        item: Item to encode.

    Returns:
        JSON-serializable dict.
    """
    return {
        "count": item["count"],
        "enabled": item["enabled"],
    }


def decode_inventory_item(obj: JSONObject) -> InventoryItem:
    """Decode JSON object to InventoryItem.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated InventoryItem.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    count = require_int(obj, "count")
    enabled = require_bool(obj, "enabled")
    return InventoryItem(count=count, enabled=enabled)


def encode_inventory_state(state: InventoryState) -> JSONObject:
    """Encode InventoryState to JSON-serializable dict.

    Args:
        state: State to encode.

    Returns:
        JSON-serializable dict.
    """
    return {
        "armor_shields": encode_inventory_item(state["armor_shields"]),
        "dual_shots": encode_inventory_item(state["dual_shots"]),
        "missile_shots": encode_inventory_item(state["missile_shots"]),
        "homing_shots": encode_inventory_item(state["homing_shots"]),
        "extra_radars": encode_inventory_item(state["extra_radars"]),
    }


def decode_inventory_state(obj: JSONObject) -> InventoryState:
    """Decode JSON object to InventoryState.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated InventoryState.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    armor_obj = obj.get("armor_shields")
    dual_obj = obj.get("dual_shots")
    missile_obj = obj.get("missile_shots")
    homing_obj = obj.get("homing_shots")
    radar_obj = obj.get("extra_radars")

    if not isinstance(armor_obj, dict):
        raise ValueError("armor_shields must be a dict")
    if not isinstance(dual_obj, dict):
        raise ValueError("dual_shots must be a dict")
    if not isinstance(missile_obj, dict):
        raise ValueError("missile_shots must be a dict")
    if not isinstance(homing_obj, dict):
        raise ValueError("homing_shots must be a dict")
    if not isinstance(radar_obj, dict):
        raise ValueError("extra_radars must be a dict")

    return InventoryState(
        armor_shields=decode_inventory_item(armor_obj),
        dual_shots=decode_inventory_item(dual_obj),
        missile_shots=decode_inventory_item(missile_obj),
        homing_shots=decode_inventory_item(homing_obj),
        extra_radars=decode_inventory_item(radar_obj),
    )


def encode_inventory_change(change: InventoryChange) -> JSONObject:
    """Encode InventoryChange to JSON-serializable dict.

    Args:
        change: Change to encode.

    Returns:
        JSON-serializable dict.
    """
    return {
        "item": change["item"],
        "old_count": change["old_count"],
        "new_count": change["new_count"],
        "delta": change["delta"],
        "enabled_changed": change["enabled_changed"],
        "now_enabled": change["now_enabled"],
    }


def decode_inventory_change(obj: JSONObject) -> InventoryChange:
    """Decode JSON object to InventoryChange.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated InventoryChange.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
        ValueError: If item type is invalid.
    """
    item_str = require_str(obj, "item")
    item = validate_item_type(item_str)
    old_count = require_int(obj, "old_count")
    new_count = require_int(obj, "new_count")
    delta = require_int(obj, "delta")
    enabled_changed = require_bool(obj, "enabled_changed")
    now_enabled = require_bool(obj, "now_enabled")
    return InventoryChange(
        item=item,
        old_count=old_count,
        new_count=new_count,
        delta=delta,
        enabled_changed=enabled_changed,
        now_enabled=now_enabled,
    )


__all__ = [
    "VALID_ITEM_TYPES",
    "InventoryChange",
    "InventoryItem",
    "InventoryState",
    "ItemType",
    "decode_inventory_change",
    "decode_inventory_item",
    "decode_inventory_state",
    "diff_inventory",
    "encode_inventory_change",
    "encode_inventory_item",
    "encode_inventory_state",
    "inventory_all_full",
    "replace_inventory_slot",
    "validate_item_type",
]
