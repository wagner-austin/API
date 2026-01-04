"""DOM scraping for in-game log capture.

Provides utilities to extract game log text from the Tankpit game DOM
via Chrome DevTools Protocol (CDP). Used to correlate WebSocket events
with visible in-game messages.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.json_utils import (
    JSONObject,
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

    Maintains state of the previous log to detect new entries
    when scraping the DOM repeatedly.
    """

    def __init__(self, cdp: CDPSessionProtocol) -> None:
        """Initialize the scraper.

        Args:
            cdp: CDP session for DOM access.
        """
        self._cdp = cdp
        self._previous_entries: set[str] = set()

    def scrape(self) -> GameLogState:
        """Scrape current game log state.

        Returns:
            Current game log state.
        """
        raw_text = scrape_game_log_text(self._cdp)
        return parse_game_log(raw_text)

    def get_new_entries(self) -> list[GameLogEntry]:
        """Get new log entries since last call.

        Compares current log with previous state and returns
        only entries that weren't seen before.

        Returns:
            List of new log entries.
        """
        state = self.scrape()
        new_entries: list[GameLogEntry] = []

        for entry in state["entries"]:
            if entry["text"] not in self._previous_entries:
                new_entries.append(entry)
                self._previous_entries.add(entry["text"])

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
    "validate_log_category",
]
