"""Inventory tracking for Tankpit game.

Provides utilities to parse and track inventory items from the game DOM.
Detects changes in item counts and enabled states for correlation with
WebSocket messages.
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

from tankpit_bot._test_hooks import CDPSessionProtocol

log = get_logger(__name__)


# =============================================================================
# TypedDicts
# =============================================================================


# Known inventory item types
ItemType = Literal["armor_shields", "dual_shots", "missile_shots", "homing_shots", "extra_radars"]

# Mapping from display names to item types
ITEM_NAME_MAP: dict[str, ItemType] = {
    "armor shields": "armor_shields",
    "dual shots": "dual_shots",
    "missile shots": "missile_shots",
    "homing shots": "homing_shots",
    "extra radars": "extra_radars",
}


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


# =============================================================================
# JavaScript for DOM scraping
# =============================================================================


SCRAPE_INVENTORY_JS = """
(() => {
    const body = document.body;
    if (!body) return '';

    const text = body.innerText || '';

    // Find "Inventory:" header
    const invStart = text.indexOf('Inventory:');
    if (invStart < 0) return '';

    // Find the dashed line after Inventory:
    const dashStart = text.indexOf('----', invStart);
    if (dashStart < 0) return '';

    // Find newline after dashes to get item list start
    let itemStart = text.indexOf('\\n', dashStart);
    if (itemStart < 0) return '';

    // Find closing stars (end of inventory section)
    const endStars = text.indexOf('****', itemStart);
    if (endStars < 0) return '';

    return text.substring(itemStart, endStars).trim();
})()
"""


# =============================================================================
# Scraping Functions
# =============================================================================


def scrape_inventory_text(cdp: CDPSessionProtocol) -> str:
    """Scrape the raw inventory text from the DOM.

    Uses CDP Runtime.evaluate to execute JavaScript that extracts
    the inventory section from the page body.

    Args:
        cdp: CDP session for executing JavaScript.

    Returns:
        Raw inventory text, or empty string if not found.
    """
    result: JSONObject = cdp.send(
        "Runtime.evaluate",
        {"expression": SCRAPE_INVENTORY_JS, "returnByValue": True},
    )
    result_obj = result.get("result")
    if isinstance(result_obj, dict):
        value = result_obj.get("value")
        if isinstance(value, str):
            return value
    return ""


def _make_empty_inventory() -> InventoryState:
    """Create an empty inventory state with all items at zero.

    Returns:
        InventoryState with all counts at 0 and enabled True.
    """
    return InventoryState(
        armor_shields=InventoryItem(count=0, enabled=True),
        dual_shots=InventoryItem(count=0, enabled=True),
        missile_shots=InventoryItem(count=0, enabled=True),
        homing_shots=InventoryItem(count=0, enabled=True),
        extra_radars=InventoryItem(count=0, enabled=True),
    )


def _parse_inventory_line(line: str) -> tuple[ItemType, InventoryItem] | None:
    """Parse a single inventory line into item type and state.

    Expected format: "30 armor shields (disabled)" or "30 dual shots"

    Args:
        line: A single line from the inventory section.

    Returns:
        Tuple of (item_type, InventoryItem) if parsed successfully, None otherwise.
    """
    stripped = line.strip()
    if not stripped:
        return None

    # Check for (disabled) suffix
    enabled = True
    if stripped.endswith("(disabled)"):
        enabled = False
        stripped = stripped.replace("(disabled)", "").strip()

    # Split into count and item name
    parts = stripped.split(None, 1)
    if len(parts) != 2:
        return None

    count_str, item_name = parts

    # Parse count
    if not count_str.isdigit():
        return None
    count = int(count_str)

    # Look up item type
    item_name_lower = item_name.lower()
    item_type = ITEM_NAME_MAP.get(item_name_lower)
    if item_type is None:
        return None

    return (item_type, InventoryItem(count=count, enabled=enabled))


def parse_inventory(raw_text: str) -> InventoryState:
    """Parse raw inventory text into structured state.

    Args:
        raw_text: Raw text scraped from the inventory section.

    Returns:
        Parsed InventoryState with all item counts and enabled states.
    """
    state = _make_empty_inventory()

    for line in raw_text.split("\n"):
        parsed = _parse_inventory_line(line)
        if parsed is not None:
            item_type, item = parsed
            # Use explicit assignment for type safety (Literal narrowing)
            state[item_type] = item

    return state


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
# InventoryScraper Class
# =============================================================================


class InventoryScraper:
    """Tracks inventory changes over time.

    Maintains previous inventory state to detect changes when scraping
    the DOM repeatedly. Reports changes with deltas for correlation
    with WebSocket messages.
    """

    def __init__(self, cdp: CDPSessionProtocol) -> None:
        """Initialize the scraper.

        Args:
            cdp: CDP session for DOM access.
        """
        self._cdp = cdp
        self._previous_state: InventoryState | None = None

    def scrape(self) -> InventoryState:
        """Scrape current inventory state.

        Returns:
            Current inventory state.
        """
        raw_text = scrape_inventory_text(self._cdp)
        return parse_inventory(raw_text)

    def get_changes(self) -> list[InventoryChange]:
        """Get inventory changes since last call.

        Compares current inventory with previous state and returns
        list of changes. On first call, returns empty list.

        Returns:
            List of inventory changes.
        """
        current = self.scrape()

        if self._previous_state is None:
            self._previous_state = current
            return []

        changes = diff_inventory(self._previous_state, current)
        self._previous_state = current
        return changes

    def log_changes(self) -> list[InventoryChange]:
        """Log any inventory changes to the logger.

        Checks for changes and logs them with appropriate messages.

        Returns:
            List of inventory changes that were logged.
        """
        changes = self.get_changes()
        for change in changes:
            item_display = change["item"].replace("_", " ")
            if change["delta"] != 0:
                if change["delta"] > 0:
                    log.info(
                        "[INV:GAINED] %s: +%d (%d->%d)",
                        item_display,
                        change["delta"],
                        change["old_count"],
                        change["new_count"],
                    )
                else:
                    log.info(
                        "[INV:USED] %s: %d (%d->%d)",
                        item_display,
                        change["delta"],
                        change["old_count"],
                        change["new_count"],
                    )
            if change["enabled_changed"]:
                state_str = "enabled" if change["now_enabled"] else "disabled"
                log.info("[INV:TOGGLE] %s: %s", item_display, state_str)
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
    "ITEM_NAME_MAP",
    "VALID_ITEM_TYPES",
    "InventoryChange",
    "InventoryItem",
    "InventoryScraper",
    "InventoryState",
    "ItemType",
    "decode_inventory_change",
    "decode_inventory_item",
    "decode_inventory_state",
    "diff_inventory",
    "encode_inventory_change",
    "encode_inventory_item",
    "encode_inventory_state",
    "parse_inventory",
    "scrape_inventory_text",
    "validate_item_type",
]
