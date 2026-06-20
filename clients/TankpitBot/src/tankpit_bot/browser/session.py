"""Browser session management for WebSocket capture.

Provides a base class that handles:
- Playwright browser launch and CDP setup
- WebSocket event handlers and message capture
- Magic key capture for XOR decoding
- Login flow integration
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot._test_hooks import (
    CDPSessionProtocol,
    PageProtocol,
)
from tankpit_bot.browser.cdp_utils import (
    get_current_time_ms,
    reset_cdp_time_offset,
)
from tankpit_bot.browser.dom_scraper import (
    GameLogEntry,
    GameLogScraper,
)
from tankpit_bot.browser.session_base import SessionBase
from tankpit_bot.types import (
    CapturedMessage,
)

log = get_logger(__name__)

# Teardown bound: artifacts are saved before cleanup starts, so a
# teardown that outlives this is converted into a recorded forced exit
# instead of an eternal hang (runs 20260611-083908/092159 each sat 10+


class BrowserSession(SessionBase):
    """Base class for browser-based WebSocket capture.

    Inherits CDPService composition from SessionBase. Adds sniffer-specific
    scrapers (game log, combat, inventory, fuel), browser lifecycle methods,
    and intel gathering.
    """

    def __init__(
        self,
        target_url: str,
        *,
        headless: bool = False,
        prefer_account: bool = False,
    ) -> None:
        """Initialize browser session.

        Args:
            target_url: URL to navigate to.
            headless: Whether to run browser in headless mode.
            prefer_account: Skip guest login and use account credentials.
        """
        super().__init__(target_url, headless=headless, prefer_account=prefer_account)
        self._page: PageProtocol | None = None
        self._game_log_scraper: GameLogScraper | None = None

    @property
    def session_id(self) -> str:
        """Get session ID."""
        return self._session_id

    @property
    def messages(self) -> list[CapturedMessage]:
        """Get captured messages."""
        return self._messages

    @property
    def magic(self) -> str | None:
        """Get captured magic key for XOR decoding."""
        return self._magic

    @property
    def static_key(self) -> str | None:
        """Get captured static XOR key from game JS."""
        return self._static_key

    def _init_game_log_scraper(self, cdp: CDPSessionProtocol) -> None:
        """Initialize the game log scraper.

        Args:
            cdp: CDP session for DOM access.
        """
        self._game_log_scraper = GameLogScraper(cdp)
        log.info("Game log scraper initialized")

    def _poll_game_log(self) -> list[GameLogEntry]:
        """Poll for new game log entries since the last call.

        Each new entry is logged at INFO via ``_process_game_log_entry``;
        subclasses override that hook (or override this method) to add
        per-entry handling such as combat tracking or capture-session
        persistence.

        Returns:
            List of new entries found since last poll (in arrival order).
        """
        if self._game_log_scraper is None:
            return []
        new_entries = self._game_log_scraper.get_new_entries()
        for entry in new_entries:
            self._process_game_log_entry(entry)
        return new_entries

    def _process_game_log_entry(self, entry: GameLogEntry) -> None:
        """Process a single game log entry.

        Default behavior logs the entry at INFO with a category prefix.
        Subclasses override to add session-specific handling (e.g. the
        sniffer subclass records entries to the capture session and
        feeds combat-category lines to its CombatTracker).

        Args:
            entry: The game log entry to process.
        """
        prefix = f"[GAME:{entry['category'].upper()}]"
        log.info("%s %s", prefix, entry["text"])


__all__ = [
    "BrowserSession",
    "get_current_time_ms",
    "reset_cdp_time_offset",
]
