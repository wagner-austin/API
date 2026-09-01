"""Parse and emit the in-game ``C`` statistics panel.

The panel is account-wide ground truth the wire never carries: total
play time, lifetime destroyed enemies, lifetime deactivations, and
promotion points (probe 20260610-2348 captured ``Play time: 41:25:23``
/ ``Destroyed enemies: 65`` / ``Deactivated: 0`` /
``Promotion points: 121314``). Sampling it at bot startup gives every
run a baseline; the delta between consecutive runs' samples verifies
the per-run game-log kill detection and yields promotion points per
run, an effectiveness number no other channel provides.
"""

from __future__ import annotations

import re

from platform_core.json_utils import JSONObject, require_int, require_str
from typing_extensions import TypedDict

from tankpit_bot.runtime_logging import emit_diagnostic

_STATS_MARKER = "Statistics:"
# The game does NOT zero-pad minutes/seconds ("Play time: 42:3:10" in
# run 20260611-093904), so the fields are 1-2 digits, not exactly 2.
_PLAY_TIME_PATTERN = re.compile(r"Play time:\s*(\d+):(\d{1,2}):(\d{1,2})")
_DESTROYED_PATTERN = re.compile(r"Destroyed enemies:\s*(\d+)")
_DEACTIVATED_PATTERN = re.compile(r"Deactivated:\s*(\d+)")
_PROMOTION_PATTERN = re.compile(r"Promotion points:\s*(\d+)")
_RANK_PATTERN = re.compile(r"Rank:\s*([a-z ]+?)\s*\((\d+)\)")


class AccountStatsDict(TypedDict):
    """Account-wide statistics from the in-game ``C`` panel.

    Attributes:
        play_time_s: Lifetime play time in whole seconds.
        destroyed_enemies: Lifetime kill count.
        deactivated: Lifetime own-deactivation count.
        promotion_points: Lifetime promotion points.
        rank_name: Current rank label (e.g. ``private``).
        leaderboard_position: The tank's PLACE in the room's standings,
            printed in parentheses after the rank -- "Rank: private
            (18)". 1 is the top. Descends as the tank accumulates
            (user, 2026-08-05: "im currently rank 26. as we get more
            kills ill move down to 25, 24, ..., and eventually 1").

            Called ``rank_number`` and documented as a promotion
            COUNTDOWN until 2026-09-01, when the archive settled it:
            two tanks at 0 kills and 0 promotion points read 28946 and
            28952, seventeen seconds apart. A countdown to the next
            rank is a function of rank and points alone, so identical
            state MUST yield an identical number; a place in a
            standings table must not. Confirmed from the other end --
            one tank went 148 kills -> 12055 and then 149 kills ->
            12060, the number worsening as it scored, which positional
            drift explains and a countdown cannot.

            Per TANK, not per account: one account's colours read 18,
            4562 and 28946 on the same day. NOT a points total -- the
            points live in ``promotion_points``.
    """

    play_time_s: int
    destroyed_enemies: int
    deactivated: int
    promotion_points: int
    rank_name: str
    leaderboard_position: int


def encode_account_stats(stats: AccountStatsDict) -> JSONObject:
    """Encode account stats to a JSON-serializable dict.

    Args:
        stats: Account stats to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "play_time_s": stats["play_time_s"],
        "destroyed_enemies": stats["destroyed_enemies"],
        "deactivated": stats["deactivated"],
        "promotion_points": stats["promotion_points"],
        "rank_name": stats["rank_name"],
        "leaderboard_position": stats["leaderboard_position"],
    }


def decode_account_stats(data: JSONObject) -> AccountStatsDict:
    """Decode account stats with strict validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated account stats.

    Raises:
        JSONTypeError: If required fields are missing or mistyped.
    """
    return AccountStatsDict(
        play_time_s=require_int(data, "play_time_s"),
        destroyed_enemies=require_int(data, "destroyed_enemies"),
        deactivated=require_int(data, "deactivated"),
        promotion_points=require_int(data, "promotion_points"),
        rank_name=require_str(data, "rank_name"),
        leaderboard_position=require_int(data, "leaderboard_position"),
    )


def parse_account_stats(page_text: str) -> AccountStatsDict | None:
    """Parse the ``C`` statistics panel out of rendered page text.

    Args:
        page_text: Full ``document.body.innerText`` captured while the
            panel is expected to be open.

    Returns:
        Parsed account stats, or ``None`` when the panel is not
        readable in this scrape: the ``Statistics:`` marker is absent
        (the scrape raced the panel toggle) or the stats/rank lines
        have not rendered yet. Mid-render text is a normal observable
        state of the panel, not corrupt input -- treating it as an
        error crashed sessions 20260611-004251/004405/012807 whenever
        a single timed read landed between the marker painting and the
        stat lines painting. Callers poll until non-``None`` or their
        deadline.
    """
    if _STATS_MARKER not in page_text:
        return None
    play_time = _PLAY_TIME_PATTERN.search(page_text)
    destroyed = _DESTROYED_PATTERN.search(page_text)
    deactivated = _DEACTIVATED_PATTERN.search(page_text)
    promotion = _PROMOTION_PATTERN.search(page_text)
    rank = _RANK_PATTERN.search(page_text)
    if play_time is None or destroyed is None or deactivated is None or promotion is None:
        return None
    if rank is None:
        return None
    hours, minutes, seconds = (
        int(play_time.group(1)),
        int(play_time.group(2)),
        int(play_time.group(3)),
    )
    return AccountStatsDict(
        play_time_s=hours * 3600 + minutes * 60 + seconds,
        destroyed_enemies=int(destroyed.group(1)),
        deactivated=int(deactivated.group(1)),
        promotion_points=int(promotion.group(1)),
        rank_name=rank.group(1),
        leaderboard_position=int(rank.group(2)),
    )


def stats_marker_present(page_text: str) -> bool:
    """Report whether the panel header has painted in the scrape.

    Distinguishes the two failure modes of a capture attempt: the panel
    never opened (marker absent -- the keypress was swallowed) versus
    the stat lines never finished painting (marker present). Callers
    record this with failed captures so artifacts explain themselves.

    Args:
        page_text: Full ``document.body.innerText`` from the scrape.

    Returns:
        True when the ``Statistics:`` marker is in the text.
    """
    return _STATS_MARKER in page_text


def emit_account_stats_sample(
    stats: AccountStatsDict | None,
    *,
    phase: str,
    marker_present: bool = False,
    scrape_chars: int = 0,
) -> None:
    """Emit the panel sample (or its absence) as a DIAGNOSTIC.

    Args:
        stats: Parsed panel stats, or ``None`` when the panel was not
            readable in any scrape of the attempt.
        phase: Capture point label (e.g. ``startup``).
        marker_present: For failed attempts, whether the final scrape
            contained the panel header (ignored when ``stats`` is set).
        scrape_chars: For failed attempts, the final scrape's length
            (ignored when ``stats`` is set).
    """
    if stats is None:
        emit_diagnostic(
            diagnostic_kind="session_account_stats",
            phase=phase,
            panel_visible=False,
            marker_present=marker_present,
            scrape_chars=scrape_chars,
        )
        return
    emit_diagnostic(
        diagnostic_kind="session_account_stats",
        phase=phase,
        panel_visible=True,
        play_time_s=stats["play_time_s"],
        destroyed_enemies=stats["destroyed_enemies"],
        deactivated=stats["deactivated"],
        promotion_points=stats["promotion_points"],
        rank_name=stats["rank_name"],
        leaderboard_position=stats["leaderboard_position"],
    )


__all__ = [
    "AccountStatsDict",
    "decode_account_stats",
    "emit_account_stats_sample",
    "encode_account_stats",
    "parse_account_stats",
    "stats_marker_present",
]
