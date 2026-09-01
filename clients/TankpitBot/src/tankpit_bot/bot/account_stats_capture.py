"""Account statistics capture from the in-game ``C`` panel.

The panel carries account-wide ground truth the wire never sends --
lifetime play time, kills, deactivations, promotion points. Lives
outside :mod:`tankpit_bot.bot.base` because it needs only a CDP
session and a page, not a whole bot.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot._test_hooks import CDPSessionProtocol, PageWaitProtocol
from tankpit_bot.browser.cdp_utils import get_current_time_ms
from tankpit_bot.browser.dom_scraper import scrape_page_text
from tankpit_bot.diagnostics.account_stats import (
    emit_account_stats_sample,
    parse_account_stats,
)
from tankpit_bot.sniffer.world_service import WorldService

log = get_logger(__name__)

# The C statistics panel paints incrementally: the "Statistics:" header
# can be in the DOM before the stat lines (a single 1500ms timed read
# landed in that gap and crashed sessions 20260611-004251/004405/012807).
# Poll the parse predicate instead of trusting one timed read.
_ACCOUNT_STATS_POLL_INTERVAL_MS = 300

_ACCOUNT_STATS_POLL_ATTEMPTS = 10

# Total wait budget for a single timed panel read (used by the simple
# capture path; equals one full poll budget).
_ACCOUNT_STATS_PANEL_RENDER_MS = _ACCOUNT_STATS_POLL_INTERVAL_MS * _ACCOUNT_STATS_POLL_ATTEMPTS


def capture_account_stats(
    ws: WorldService,
    cdp: CDPSessionProtocol | None,
    page: PageWaitProtocol | None,
    phase: str,
) -> None:
    """Sample the in-game ``C`` statistics panel and emit it.

    The panel carries account-wide ground truth the wire never
    sends (lifetime play time, kills, deactivations, promotion
    points); the startup sample baselines every run so consecutive
    runs' deltas verify the wire 0x41 kill detection. The ``C`` key
    does not toggle a stateful panel -- each keypress emits a
    fresh ``Statistics:`` block into the in-game DOM log -- so a
    single press is enough to scrape, and a second press would
    only duplicate the block in the log without ``closing``
    anything.

    Args:
        phase: Capture point label (e.g. ``startup``).
    """
    if cdp is None or page is None:
        return
    for event_type in ("keyDown", "keyUp"):
        cdp.send(
            "Input.dispatchKeyEvent",
            {
                "type": event_type,
                "key": "c",
                "code": "KeyC",
                "windowsVirtualKeyCode": ord("C"),
                "nativeVirtualKeyCode": ord("C"),
            },
        )
    page.wait_for_timeout(_ACCOUNT_STATS_PANEL_RENDER_MS)
    page_text = scrape_page_text(cdp)
    stats = parse_account_stats(page_text)
    emit_account_stats_sample(stats, phase=phase)
    if stats is not None:
        # Canonical account model (state/types/self_account.py):
        # runtime features read this instead of re-fishing the
        # diagnostic stream.
        ws.record_account_stats(
            rank_name=stats["rank_name"],
            leaderboard_position=stats["leaderboard_position"],
            promotion_points=stats["promotion_points"],
            destroyed_enemies=stats["destroyed_enemies"],
            deactivated_total=stats["deactivated"],
            play_time_s=stats["play_time_s"],
            timestamp_ms=get_current_time_ms(),
        )


__all__ = ["capture_account_stats"]
