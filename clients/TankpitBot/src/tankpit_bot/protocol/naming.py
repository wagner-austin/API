"""Tank-name classification: practice bot versus human.

The wire never announces "this is a human" (the registry's ``is_bot``
field has no populating decoder), but the practice room's bots are
perfectly fingerprinted by NAME: the client's ``sd()`` initializer
builds every one as ``<team color>-<n>`` (``orange-1``, ``red-6``,
...), while humans carry account names or ``guest``.

Single home for the classification (lifted 2026-08-03): the bot AI's
priority tiers, the sim's reactive-ghost roster selection, and the
validate layer's shadow laws and fight timelines all consume these
two predicates — previously ``validate/shadow_bot_laws.py`` kept its
own copy of the regex.
"""

from __future__ import annotations

import re

PRACTICE_BOT_NAME_PATTERN = re.compile(r"^(red|purple|blue|orange)-\d+$")
"""Practice-bot naming from the client ``sd()`` initializer: team-N.

Group 1 captures the color — part of the API: the shadow laws map it
to the team id (join-roster ground truth: red-1 arrives team 0)."""


def is_practice_bot_name(name: str) -> bool:
    """Return whether a tank name matches the practice-bot pattern.

    Args:
        name: Tank display name from the registry.

    Returns:
        True for ``<team color>-<n>`` names (``orange-1``, ``red-6``).
    """
    return PRACTICE_BOT_NAME_PATTERN.match(name) is not None


def is_human_name(name: str) -> bool:
    """Return whether a tank name classifies as a human player.

    An empty name is UNKNOWN (an unsynced tank), never human -- the
    consumers must not chase phantoms.

    Args:
        name: Tank display name from the registry.

    Returns:
        True for any non-empty name outside the practice-bot pattern
        (account names, ``guest``).
    """
    return name != "" and not is_practice_bot_name(name)


__all__ = [
    "PRACTICE_BOT_NAME_PATTERN",
    "is_human_name",
    "is_practice_bot_name",
]
