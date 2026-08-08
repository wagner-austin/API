"""The DOM game log as a witness, plus the one-shot account-stats read.

The bot inherits ``SessionBase`` — a parallel hierarchy from the
``BrowserSession`` the standalone sniffer uses — so it owns its own
game-log scraper hooks. The DOM log is a WITNESS, not an actor: every
line it renders is the client's presentation of a wire message the bot
already decodes (0x41 Deactivation for kills, 0x52 error codes for
rejections — capture replay 2026-07-19, see [[deactivation-format]]).
The tick loop polls it each tick and records the entries into the
capture artifact so the analyzer can diff the client's rendering
against the wire; nothing in the bot acts on them.

The account-stats capture lives here for the same reason: it is a
second read of what the CLIENT shows about the account, not something
the bot's decisions consume.

Attributes are DECLARED here and assigned by :class:`Bot`, so this
mixin adds behaviour without taking ownership of the session's state
([[session-state-deglobalisation]]).
"""

from __future__ import annotations

from tankpit_bot._test_hooks import AutoscrollPageProtocol, CDPSessionProtocol
from tankpit_bot.bot.account_stats_capture import capture_account_stats
from tankpit_bot.browser.dom_scraper import GameLogEntry, GameLogScraper
from tankpit_bot.browser.game_log import (
    make_game_log_scraper,
    poll_game_log,
    timestamp_game_log_entries,
)
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.types import GameLogEntryWithTimestamp

# The first-tick keypress itself can be swallowed by the client (run
# 20260611-013801: panel never opened across a full poll budget), so
# the startup capture retries on later ticks.
_ACCOUNT_STATS_MAX_CAPTURE_ATTEMPTS = 3


class GameLogWitnessMixin:
    """Polls the DOM game log and captures account stats once."""

    _game_log_scraper: GameLogScraper | None
    _game_log_witness: list[GameLogEntryWithTimestamp]
    _cdp: CDPSessionProtocol | None
    _page: AutoscrollPageProtocol | None
    world: WorldService
    _account_stats_captured: bool
    _account_stats_attempts: int

    def _init_game_log_scraper(self, cdp: CDPSessionProtocol) -> None:
        """Create the game log scraper for server feedback visibility.

        Args:
            cdp: Active CDP session for DOM access.
        """
        self._game_log_scraper = make_game_log_scraper(cdp)

    def _poll_game_log(self) -> list[GameLogEntry]:
        """Poll the game log for new entries since the last scrape.

        Returns:
            New log entries (kills, hits, empty containers, etc.).
        """
        return poll_game_log(self._game_log_scraper)

    def _record_game_log_witness(self, entries: list[GameLogEntry]) -> None:
        """Timestamp new game-log entries into the capture witness list.

        Args:
            entries: New log entries from this tick's poll, in order.
        """
        self._game_log_witness.extend(timestamp_game_log_entries(entries))

    def maybe_capture_account_stats_once(self) -> None:
        """Capture account stats on the first healthy tick, with bounded retries.

        The C-panel hotkey can be swallowed by the game client (run
        20260611-013801), so failed attempts retry on later ticks up to
        a bounded maximum.
        """
        if self._account_stats_captured:
            return
        if self._account_stats_attempts >= _ACCOUNT_STATS_MAX_CAPTURE_ATTEMPTS:
            return
        self._account_stats_attempts += 1
        capture_account_stats(self.world, self._cdp, self._page, "startup")
        self._account_stats_captured = True


__all__ = [
    "GameLogWitnessMixin",
]
