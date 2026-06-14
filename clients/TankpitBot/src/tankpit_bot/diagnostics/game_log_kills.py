"""Register kills from the in-game text log.

Live runs 20260610-005248 and 20260610-011x proved the wire 0x41
Deactivation message never arrives for the bot's own kills: the
``tank_deactivated`` diagnostics on both wire decode paths stayed at
zero across two on-screen kills. The authoritative kill signal the
client actually renders is the game-log banner::

    ********************************************
     red-8
     has been deactivated by you
    ********************************************

The DOM game-log scraper has parsed this text all along -- but only the
sniffer ever polled it; the bot initialized the scraper and never read
it. This module is the bot-side consumer: each tick's new log entries
are scanned for the deactivation banner (one-line and two-line shapes),
the victim name is resolved to a tank id through the bot's tracked
tanks, and the kill is registered exactly where wire-decoded kills
would land.
"""

from __future__ import annotations

import re

from tankpit_bot.browser.dom_scraper import GameLogEntry
from tankpit_bot.runtime_logging import emit_diagnostic
from tankpit_bot.sniffer.world_state_combat import mark_tank_killed
from tankpit_bot.state.types import WorldStateDict

_PLAYER_KILL_PATTERN = re.compile(r"^(.+?)\s+has been deactivated by you$")
_BANNER_SUFFIX = "has been deactivated by you"


def _extract_victim_name(text: str, previous_text: str) -> str | None:
    """Return the victim name when a log line completes a kill banner.

    Args:
        text: Current log line text (stripped by the scraper).
        previous_text: The immediately preceding log line text, used for
            the two-line banner shape where the victim name and the
            "has been deactivated by you" suffix arrive as separate
            lines.

    Returns:
        Victim name, or ``None`` when the line is not a kill banner.
    """
    matched = _PLAYER_KILL_PATTERN.match(text)
    if matched is not None:
        # Scraper entries are pre-stripped and the lazy group ends before
        # the whitespace preceding the suffix, so the name needs no trim.
        return matched.group(1)
    if text == _BANNER_SUFFIX and previous_text:
        return previous_text
    return None


def _resolve_tank_id(world: WorldStateDict, victim_name: str) -> int:
    """Return the tracked tank id for a victim name, or ``-1`` when unknown.

    Args:
        world: Current world state with tracked tanks.
        victim_name: Display name from the kill banner (e.g. ``red-8``).

    Returns:
        Tank id, or ``-1`` when no tracked tank carries that name.
    """
    for tank in world["tanks"].values():
        if tank["name"] == victim_name:
            return tank["tank_id"]
    return -1


def register_kills_from_game_log(
    entries: list[GameLogEntry],
    world: WorldStateDict,
) -> int:
    """Register every kill banner found in new game-log entries.

    Each kill is emitted as a ``tank_deactivated`` DIAGNOSTIC
    (``origin="game_log"``) regardless of id resolution so unresolved
    names stay visible in the artifact; only resolved ids are marked
    killed in world state.

    Args:
        entries: New game-log entries from this tick's poll, in order.
        world: Current world state used for name-to-id resolution.

    Returns:
        Number of kill banners registered.
    """
    kills = 0
    previous_text = ""
    for entry in entries:
        victim_name = _extract_victim_name(entry["text"], previous_text)
        previous_text = entry["text"]
        if victim_name is None:
            continue
        victim_id = _resolve_tank_id(world, victim_name)
        if victim_id > 0:
            mark_tank_killed(victim_id)
        emit_diagnostic(
            diagnostic_kind="tank_deactivated",
            origin="game_log",
            victim_name=victim_name,
            victim_id=victim_id,
            killer_id=-1,
        )
        kills += 1
    return kills


__all__ = [
    "register_kills_from_game_log",
]
