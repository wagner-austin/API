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

import re

_BOT_NAME_PATTERN = re.compile(r"^(?:red|purple|blue|orange)-\d+$")

PRIORITY_NAMED = 0
"""Tier for the configured priority account (outranks everything)."""

PRIORITY_HUMAN = 1
"""Tier for any human-classified enemy (outranks all bots)."""

PRIORITY_BOT = 2
"""Tier for practice bots and unnamed/unknown tanks."""


def is_practice_bot_name(name: str) -> bool:
    """Return whether a tank name matches the practice-bot pattern.

    Args:
        name: Tank display name from the registry.

    Returns:
        True for ``<team color>-<n>`` names (``orange-1``, ``red-6``).
    """
    return _BOT_NAME_PATTERN.match(name) is not None


def is_human_name(name: str) -> bool:
    """Return whether a tank name classifies as a human player.

    An empty name is UNKNOWN (an unsynced tank), never human -- the
    priority must not chase phantoms.

    Args:
        name: Tank display name from the registry.

    Returns:
        True for any non-empty name outside the practice-bot pattern
        (account names, ``guest``).
    """
    return name != "" and not is_practice_bot_name(name)


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
    "PRIORITY_BOT",
    "PRIORITY_HUMAN",
    "PRIORITY_NAMED",
    "is_human_name",
    "is_practice_bot_name",
    "threat_priority_tier",
]
