"""DOM scraping for in-game log capture.

Provides utilities to extract game log text from the Tankpit game DOM
via Chrome DevTools Protocol (CDP). Used to correlate WebSocket events
with visible in-game messages.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.json_utils import (
    JSONObject,
    load_json_str,
    narrow_json_to_dict,
    require_bool,
    require_dict,
    require_int,
    require_list,
    require_str,
)
from platform_core.logging import get_logger

from tankpit_bot._test_hooks import CDPSessionProtocol

log = get_logger(__name__)


# =============================================================================
# TypedDicts
# =============================================================================


LogCategory = Literal[
    "location", "action", "combat", "equipment", "teleport", "tip", "fuel", "other"
]


class GameLogEntry(TypedDict):
    """A single entry from the game log.

    Attributes:
        text: The log message text.
        category: Category of the log entry.
    """

    text: str
    category: LogCategory


class GameLogState(TypedDict):
    """Current state of the game log.

    Attributes:
        raw_text: The raw scraped text from the DOM.
        entries: Parsed log entries.
        location: Current tank location as "x,y" or empty if unknown.
    """

    raw_text: str
    entries: list[GameLogEntry]
    location: str


# =============================================================================
# JavaScript for DOM scraping
# =============================================================================

SCRAPE_GAME_LOG_JS = """
(() => {
    const body = document.body;
    if (!body) return '';

    const text = body.innerText || '';

    // Game log is between inventory (ends with "extra radars") and chat ("Attack the")
    const inventoryEnd = text.lastIndexOf('extra radars');
    if (inventoryEnd < 0) return '';

    // Find newline after inventory to get log start
    let logStart = text.indexOf('\\n', inventoryEnd);
    if (logStart < 0) logStart = inventoryEnd;

    const afterInventory = text.substring(logStart);

    // Find where chat buttons start
    const chatIndex = afterInventory.indexOf('Attack the');
    const logSection = chatIndex > 0 ? afterInventory.substring(0, chatIndex) : afterInventory;

    return logSection.trim();
})()
"""


# =============================================================================
# Scraping Functions
# =============================================================================


def scrape_page_text(cdp: CDPSessionProtocol) -> str:
    """Scrape the full rendered page text from the DOM.

    Used for panels that live outside the game-log section, e.g. the
    ``C`` statistics panel (play time, destroyed enemies, promotion
    points).

    Args:
        cdp: CDP session for executing JavaScript.

    Returns:
        Full ``document.body.innerText``, or empty string when the body
        is unavailable.
    """
    result: JSONObject = cdp.send(
        "Runtime.evaluate",
        {
            "expression": "document.body ? document.body.innerText : ''",
            "returnByValue": True,
        },
    )
    result_obj = result.get("result")
    if isinstance(result_obj, dict):
        value = result_obj.get("value")
        if isinstance(value, str):
            return value
    return ""


def scrape_game_log_text(cdp: CDPSessionProtocol) -> str:
    """Scrape the raw game log text from the DOM.

    Uses CDP Runtime.evaluate to execute JavaScript that extracts
    the game log section from the page body.

    Args:
        cdp: CDP session for executing JavaScript.

    Returns:
        Raw game log text, or empty string if not found.
    """
    result: JSONObject = cdp.send(
        "Runtime.evaluate",
        {"expression": SCRAPE_GAME_LOG_JS, "returnByValue": True},
    )
    result_obj = result.get("result")
    if isinstance(result_obj, dict):
        value = result_obj.get("value")
        if isinstance(value, str):
            return value
    return ""


SCRAPE_LOG_HEALTH_JS = """
(() => {
    const body = document.body;
    if (!body) return JSON.stringify(
        {bodyLength: 0, hasInventoryAnchor: false, hasChatAnchor: false});
    const text = body.innerText || '';
    return JSON.stringify({
        bodyLength: text.length,
        hasInventoryAnchor: text.lastIndexOf('extra radars') >= 0,
        hasChatAnchor: text.indexOf('Attack the') >= 0,
    });
})()
"""


class GameLogHealthDict(TypedDict):
    """Anchor-level health probe of the game-log scrape.

    Attributes:
        body_length: Length of ``document.body.innerText``.
        has_inventory_anchor: Whether the ``extra radars`` inventory
            anchor the scrape starts from exists in the body text.
        has_chat_anchor: Whether the ``Attack the`` chat anchor the
            scrape ends at exists in the body text.
    """

    body_length: int
    has_inventory_anchor: bool
    has_chat_anchor: bool


def scrape_game_log_health(cdp: CDPSessionProtocol) -> GameLogHealthDict:
    """Probe why the game-log scrape might be returning empty text.

    The scrape silently anchors on page text markers; when an anchor
    disappears the scrape returns ``""`` forever and every downstream
    consumer (kill detection, hit/miss feedback, tank-full learning)
    goes blind without any signal (live 2026-06-12: silent for 8+
    hours across 5 runs). This probe reports the anchors directly.

    Args:
        cdp: CDP session for executing JavaScript.

    Returns:
        Anchor-level health of the scrape.
    """
    result: JSONObject = cdp.send(
        "Runtime.evaluate",
        {"expression": SCRAPE_LOG_HEALTH_JS, "returnByValue": True},
    )
    result_obj = require_dict(result, "result")
    raw = require_str(result_obj, "value")
    data = narrow_json_to_dict(load_json_str(raw))
    return GameLogHealthDict(
        body_length=require_int(data, "bodyLength"),
        has_inventory_anchor=require_bool(data, "hasInventoryAnchor"),
        has_chat_anchor=require_bool(data, "hasChatAnchor"),
    )


# Pattern lists for categorization (checked in order)
_EQUIPMENT_PATTERNS = ("enabled", "disabled", "gained", "inventory full")
_COMBAT_PATTERNS = ("hit", "deactivated", "destroyed", "you earned")
_ACTION_PATTERNS = (
    "autoscroll",
    "detected to",
    "you fire",
    "extend view",
    "zoom in",
    "zoom out",
    "obstacle picked up",
    "obstacle dropped",
    "fuel deposited",
    "you can't go there",
    "you are already there",
)


def categorize_log_line(line: str) -> LogCategory:
    """Categorize a log line based on its content.

    Args:
        line: A single log line.

    Returns:
        Category string for the log entry.
    """
    # Check prefix-based categories first
    if line.startswith("LOCATION:"):
        return "location"
    if line.startswith("Teleporting to"):
        return "teleport"
    if line.startswith("Tip"):
        return "tip"

    # Pattern-based matching (case-insensitive)
    line_lower = line.lower()

    if any(p in line_lower for p in _EQUIPMENT_PATTERNS):
        return "equipment"
    if any(p in line_lower for p in _COMBAT_PATTERNS):
        return "combat"
    if any(p in line_lower for p in _ACTION_PATTERNS):
        return "action"

    return "other"


def parse_game_log(raw_text: str) -> GameLogState:
    """Parse raw game log text into structured entries.

    Args:
        raw_text: Raw text scraped from the DOM.

    Returns:
        Parsed GameLogState with entries and location.
    """
    entries: list[GameLogEntry] = []
    location = ""

    lines = raw_text.split("\n")
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Extract location
        if stripped.startswith("LOCATION:"):
            loc_part = stripped.replace("LOCATION:", "").strip()
            if loc_part:
                location = loc_part

        category = categorize_log_line(stripped)
        entries.append(GameLogEntry(text=stripped, category=category))

    return GameLogState(raw_text=raw_text, entries=entries, location=location)


# =============================================================================
# GameLogScraper Class
# =============================================================================


class GameLogScraper:
    """Tracks game log changes over time.

    Maintains per-text occurrence counts of the previous scrape so that
    repeated identical lines (``Empty container``, ``Tank full``, a
    second kill banner for the same enemy) are each detected as new.
    A forever-set of seen texts would report each distinct line exactly
    once per session and silently swallow every repeat.
    """

    def __init__(self, cdp: CDPSessionProtocol) -> None:
        """Initialize the scraper.

        Args:
            cdp: CDP session for DOM access.
        """
        self._cdp = cdp
        self._previous_counts: dict[str, int] = {}

    def scrape(self) -> GameLogState:
        """Scrape current game log state.

        Returns:
            Current game log state.
        """
        raw_text = scrape_game_log_text(self._cdp)
        return parse_game_log(raw_text)

    def get_new_entries(self) -> list[GameLogEntry]:
        """Get new log entries since last call.

        Compares per-text occurrence counts between the previous and
        current scrape: the Nth occurrence of a text in the current
        window is new when the previous window held fewer than N of it.
        In-place line mutations (equip-bar counters) therefore surface
        once as a new line but never re-emit the rest of the window.

        Returns:
            List of new log entries.
        """
        state = self.scrape()
        current_counts: dict[str, int] = {}
        new_entries: list[GameLogEntry] = []

        for entry in state["entries"]:
            text = entry["text"]
            current_counts[text] = current_counts.get(text, 0) + 1
            if current_counts[text] > self._previous_counts.get(text, 0):
                new_entries.append(entry)

        self._previous_counts = current_counts
        return new_entries

    def log_new_entries(self) -> None:
        """Log any new entries to the logger.

        Checks for new entries and logs them with appropriate prefixes.
        """
        new_entries = self.get_new_entries()
        for entry in new_entries:
            prefix = f"[GAME:{entry['category'].upper()}]"
            log.info("%s %s", prefix, entry["text"])


# =============================================================================
# Encode/Decode Functions
# =============================================================================


def encode_game_log_entry(entry: GameLogEntry) -> JSONObject:
    """Encode GameLogEntry to JSON-serializable dict.

    Args:
        entry: Entry to encode.

    Returns:
        JSON-serializable dict.
    """
    return {
        "text": entry["text"],
        "category": entry["category"],
    }


def encode_game_log_state(state: GameLogState) -> JSONObject:
    """Encode GameLogState to JSON-serializable dict.

    Args:
        state: State to encode.

    Returns:
        JSON-serializable dict.
    """
    return {
        "raw_text": state["raw_text"],
        "entries": [encode_game_log_entry(e) for e in state["entries"]],
        "location": state["location"],
    }


VALID_CATEGORIES: frozenset[str] = frozenset(
    ["location", "action", "combat", "equipment", "teleport", "tip", "other"]
)


def validate_log_category(value: str) -> LogCategory:
    """Validate and narrow a string to a LogCategory literal.

    Args:
        value: String value to validate.

    Returns:
        The validated category as a Literal type.

    Raises:
        ValueError: If value is not a valid category.
    """
    if value == "location":
        return "location"
    if value == "action":
        return "action"
    if value == "combat":
        return "combat"
    if value == "equipment":
        return "equipment"
    if value == "teleport":
        return "teleport"
    if value == "tip":
        return "tip"
    if value == "other":
        return "other"
    raise ValueError(f"Invalid category '{value}', must be one of {VALID_CATEGORIES}")


def decode_game_log_entry(obj: JSONObject) -> GameLogEntry:
    """Decode JSON object to GameLogEntry.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated GameLogEntry.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
        ValueError: If category is not a valid value.
    """
    text = require_str(obj, "text")
    category_str = require_str(obj, "category")
    category = validate_log_category(category_str)
    return GameLogEntry(text=text, category=category)


def decode_game_log_state(obj: JSONObject) -> GameLogState:
    """Decode JSON object to GameLogState.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated GameLogState.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
        ValueError: If entry categories are invalid.
    """
    raw_text = require_str(obj, "raw_text")
    location = require_str(obj, "location")
    entries_raw = require_list(obj, "entries")
    entries: list[GameLogEntry] = []
    for i, item in enumerate(entries_raw):
        if not isinstance(item, dict):
            raise ValueError(f"Entry at index {i} must be a dict, got {type(item).__name__}")
        entries.append(decode_game_log_entry(item))
    return GameLogState(raw_text=raw_text, entries=entries, location=location)


__all__ = [
    "VALID_CATEGORIES",
    "GameLogEntry",
    "GameLogScraper",
    "GameLogState",
    "LogCategory",
    "categorize_log_line",
    "decode_game_log_entry",
    "decode_game_log_state",
    "encode_game_log_entry",
    "encode_game_log_state",
    "parse_game_log",
    "scrape_game_log_text",
    "scrape_page_text",
    "validate_log_category",
]
