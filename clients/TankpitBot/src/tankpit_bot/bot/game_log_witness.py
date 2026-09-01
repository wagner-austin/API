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

from tankpit_bot._test_hooks import CDPSessionProtocol, GamePageProtocol
from tankpit_bot.bot.account_stats_capture import capture_account_stats
from tankpit_bot.browser.dom_scraper import GameLogEntry, GameLogScraper
from tankpit_bot.browser.game_log import (
    make_game_log_scraper,
    poll_game_log,
    timestamp_game_log_entries,
)
from tankpit_bot.runtime_logging import emit_diagnostic
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.types import GameLogEntryWithTimestamp

# The first-tick keypress itself can be swallowed by the client (run
# 20260611-013801: panel never opened across a full poll budget), so
# the startup capture retries on later ticks.
_ACCOUNT_STATS_MAX_CAPTURE_ATTEMPTS = 3

# The kill banner renders as five DOM lines (archived capture, run
# bot-20260828-212653): a star rule, the victim's name alone, this
# exact line, a closing star rule, then — on World only — the points
# verdict. The name is NEVER on the deactivation line itself.
_KILL_BANNER_LINE = "has been deactivated by you"
_BANNER_RULE_PREFIX = "****"
_POINTS_EXTRA_LINE = "You earned extra points"
_POINTS_TOO_LOW_PREFIX = "Enemy's rank was too low"


def _rank_of_named_tank(world: WorldService, name: str) -> int:
    """Look a tank's rank up by name in the live registry.

    Args:
        world: The session's world service.
        name: Exact tank name as the banner rendered it.

    Returns:
        The registry rank, or -1 when no tank carries the name (the
        0x58 cleanup can outrun the banner poll).
    """
    for tank in world.world_state["tanks"].values():
        if tank["name"] == name:
            return tank["rank"]
    return -1


class GameLogWitnessMixin:
    """Polls the DOM game log and captures account stats once."""

    _game_log_scraper: GameLogScraper | None
    _game_log_witness: list[GameLogEntryWithTimestamp]
    _cdp: CDPSessionProtocol | None
    _page: GamePageProtocol | None
    world: WorldService
    _account_stats_captured: bool
    _account_stats_attempts: int
    _kill_banner_victim: str
    _last_game_log_line: str

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
        for entry in entries:
            self._witness_points_outcome(entry["text"].strip())

    def _witness_points_outcome(self, text: str) -> None:
        """Surface the World points verdict as a diagnostic, still a witness.

        The points-floor survey (operator flags 5/6/8/12/13,
        2026-09-01): World kills carry a verdict line — "You earned
        extra points" or "Enemy's rank was too low" — that only ever
        reached the capture artifact, so no ledger row paired a
        verdict with the victim rank it judged. This pairs them live:
        the banner names the victim on its own line one before the
        deactivation line, and the verdict follows the closing star
        rule. Nothing acts on the diagnostic — the witness law of this
        module holds.

        Args:
            text: One stripped game-log line, in arrival order.
        """
        if text.startswith(_BANNER_RULE_PREFIX):
            return
        if text == _KILL_BANNER_LINE:
            self._kill_banner_victim = self._last_game_log_line
            self._last_game_log_line = text
            return
        outcome = ""
        if text == _POINTS_EXTRA_LINE:
            outcome = "extra_points"
        elif text.startswith(_POINTS_TOO_LOW_PREFIX):
            outcome = "rank_too_low"
        if outcome and self._kill_banner_victim:
            self_state = self.world.world_state["self_state"]
            emit_diagnostic(
                diagnostic_kind="kill_points_outcome",
                victim_name=self._kill_banner_victim,
                victim_rank=_rank_of_named_tank(self.world, self._kill_banner_victim),
                self_rank=-1 if self_state is None else self_state["rank"],
                outcome=outcome,
            )
            self._kill_banner_victim = ""
        self._last_game_log_line = text

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
