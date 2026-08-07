"""Game-log polling shared by every session that owns a page.

The in-game DOM log is the only place the server states some outcomes
in words ("you destroyed X", "container empty"), so both the bot and
the sniffer scrape it. Both had their own copy of the scraper setup and
the poll guard; this module is the single owner, and each session holds
only the scraper handle.
"""

from __future__ import annotations

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot.browser.cdp_utils import get_current_time_ms
from tankpit_bot.browser.dom_scraper import GameLogEntry, GameLogScraper
from tankpit_bot.types import GameLogEntryWithTimestamp


def make_game_log_scraper(cdp: CDPSessionProtocol) -> GameLogScraper:
    """Create the game-log scraper for one session.

    Args:
        cdp: Active CDP session for DOM access.

    Returns:
        A scraper bound to that session.
    """
    return GameLogScraper(cdp)


def poll_game_log(scraper: GameLogScraper | None) -> list[GameLogEntry]:
    """Return the game-log entries added since the last poll.

    Args:
        scraper: The session's scraper, or ``None`` before the page is
            ready -- a session polls every tick, including before the
            scraper exists.

    Returns:
        New entries in arrival order; empty when there is no scraper.
    """
    if scraper is None:
        return []
    return scraper.get_new_entries()


def timestamp_game_log_entries(entries: list[GameLogEntry]) -> list[GameLogEntryWithTimestamp]:
    """Stamp new game-log entries with the current wall clock.

    The capture session stores the witness list so a replay can line
    the server's words up against the wire.

    Args:
        entries: New entries from this tick's poll, in order.

    Returns:
        The same entries, each carrying the poll timestamp.
    """
    now = get_current_time_ms()
    return [
        GameLogEntryWithTimestamp(
            timestamp_ms=now,
            text=entry["text"],
            category=entry["category"],
        )
        for entry in entries
    ]


__all__ = [
    "make_game_log_scraper",
    "poll_game_log",
    "timestamp_game_log_entries",
]
