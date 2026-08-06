"""Human-vs-practice-bot classification and target priority tiers.

The wire never announces "this is a human" (the registry's ``is_bot``
field has no populating decoder), but the practice room's bots are
perfectly fingerprinted by NAME: every one is ``<team color>-<n>``
(``orange-1``, ``red-6``, ...), while humans carry account names or
``guest``. User doctrine (2026-07-28): farm bots normally, but any
human who logs in outranks every bot -- and an explicitly configured
priority account name outranks even other humans.

The tier feeds the shared threat sort (`bot/ai/threats.py`), so the
priority flows through every selection path unchanged: visible-threat
picks, map-level acquisition, and the dot-relay travel target (the
yellow-dot hop chain that closes distance on an unaffordable enemy
refuels on the way there).
"""

from __future__ import annotations

from tankpit_bot.protocol.naming import is_human_name

PRIORITY_NAMED = 0
"""Tier for the configured priority account (outranks everything)."""

PRIORITY_HUMAN = 1
"""Tier for any human-classified enemy (outranks all bots)."""

PRIORITY_BOT = 2
"""Tier for practice bots and unnamed/unknown tanks."""


DEFAULT_HUMAN_MIN_RANK = 1
"""Default floor: privates and up are targetable, recruits are not
(user ruling 2026-07-28: "we dont target recruits")."""

DEFAULT_HUMAN_MAX_RANK = 8
"""Default ceiling: general -- no rank is protected from above unless
configured (the "captains and generals out of respect" knob)."""


def is_human_rank_protected(
    name: str,
    rank: int,
    *,
    min_rank: int,
    max_rank: int,
) -> bool:
    """Return whether a tank is a human outside the targetable rank window.

    User ruling 2026-07-28: rank-aware HUMAN targeting only -- practice
    bots are farmed at any rank. The window is a configurable
    ``[min_rank, max_rank]`` (ranks are plain integers, 0 recruit ..
    8 general): a practice-room bot runs the default (ignore recruits,
    engage everyone else), a main-map bot might run min 4 (lieutenant
    and higher only) or cap the top out of respect for high ranks.
    Fail-safe note: an unsynced tank's rank defaults to 0, so a human
    whose rank has not yet ridden the wire is briefly spared rather
    than briefly attacked (rank arrives with the early 0x21/0x28 wire).

    Args:
        name: Tank display name from the registry.
        rank: Registry rank (0 = recruit .. 8 = general).
        min_rank: Lowest human rank the bot may target.
        max_rank: Highest human rank the bot may target.

    Returns:
        True when the tank is human-classified and outside the window.
    """
    return is_human_name(name) and (rank < min_rank or rank > max_rank)


def threat_priority_tier(name: str, priority_target_name: str) -> int:
    """Return the target-priority tier for a tank name.

    Args:
        name: Tank display name from the registry.
        priority_target_name: Configured priority account name
            (case-insensitive), or ``""`` when none is configured.

    Returns:
        :data:`PRIORITY_NAMED`, :data:`PRIORITY_HUMAN`, or
        :data:`PRIORITY_BOT`.
    """
    if priority_target_name != "" and name.casefold() == priority_target_name.casefold():
        return PRIORITY_NAMED
    if is_human_name(name):
        return PRIORITY_HUMAN
    return PRIORITY_BOT


__all__ = [
    "DEFAULT_HUMAN_MAX_RANK",
    "DEFAULT_HUMAN_MIN_RANK",
    "PRIORITY_BOT",
    "PRIORITY_HUMAN",
    "PRIORITY_NAMED",
    "is_human_rank_protected",
    "threat_priority_tier",
]
