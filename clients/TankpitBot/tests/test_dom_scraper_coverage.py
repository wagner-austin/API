"""Coverage tests for browser/dom_scraper.py: scrape_page_text fallback and
scrape_game_log_health body."""

from __future__ import annotations

from collections.abc import Callable

from platform_core.json_utils import JSONObject, dump_json_str

from tankpit_bot.browser.dom_scraper import (
    GameLogHealthDict,
    scrape_game_log_health,
    scrape_page_text,
)


class _FakeCDPPageTextEmpty:
    """Fake CDP session that returns an empty result for scrape_page_text.

    Exercises the ``return ""`` fallback at line 126 -- the result object
    has a ``result`` dict but the ``value`` is not a string.
    """

    def __init__(self) -> None:
        """Initialize fake."""
        self._handlers: dict[str, list[Callable[[JSONObject], None]]] = {}

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Return result with numeric value (not a string)."""
        return {"result": {"value": 42}}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Register event handler."""
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)

    def detach(self) -> None:
        """Detach CDP session."""


class _FakeCDPPageTextMissingResult:
    """Fake CDP session where result is missing entirely."""

    def __init__(self) -> None:
        """Initialize fake."""
        self._handlers: dict[str, list[Callable[[JSONObject], None]]] = {}

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Return empty dict (no result key)."""
        return {}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Register event handler."""
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)

    def detach(self) -> None:
        """Detach CDP session."""


class _FakeCDPGameLogHealth:
    """Fake CDP session for scrape_game_log_health that returns valid JSON."""

    def __init__(self, body_length: int, has_inventory: bool, has_chat: bool) -> None:
        """Initialize with health values."""
        self._response = dump_json_str(
            {
                "bodyLength": body_length,
                "hasInventoryAnchor": has_inventory,
                "hasChatAnchor": has_chat,
            }
        )
        self._handlers: dict[str, list[Callable[[JSONObject], None]]] = {}

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Return result with JSON-encoded health dict."""
        return {"result": {"value": self._response}}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Register event handler."""
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)

    def detach(self) -> None:
        """Detach CDP session."""


def test_scrape_page_text_non_string_value_returns_empty() -> None:
    """scrape_page_text returns "" when the result value is not a string."""
    cdp = _FakeCDPPageTextEmpty()
    result = scrape_page_text(cdp)
    assert result == ""


def test_scrape_page_text_missing_result_returns_empty() -> None:
    """scrape_page_text returns "" when no result key in CDP response."""
    cdp = _FakeCDPPageTextMissingResult()
    result = scrape_page_text(cdp)
    assert result == ""


def test_scrape_game_log_health_healthy_page() -> None:
    """scrape_game_log_health returns correct dict for a healthy page."""
    cdp = _FakeCDPGameLogHealth(
        body_length=5000,
        has_inventory=True,
        has_chat=True,
    )
    result = scrape_game_log_health(cdp)
    expected = GameLogHealthDict(
        body_length=5000,
        has_inventory_anchor=True,
        has_chat_anchor=True,
    )
    assert result == expected


def test_scrape_game_log_health_missing_anchors() -> None:
    """scrape_game_log_health reports missing anchors correctly."""
    cdp = _FakeCDPGameLogHealth(
        body_length=0,
        has_inventory=False,
        has_chat=False,
    )
    result = scrape_game_log_health(cdp)
    assert result["body_length"] == 0
    assert result["has_inventory_anchor"] is False
    assert result["has_chat_anchor"] is False
