"""The engine's own verdict on a match, rather than a proxy for it.

Every figure the run report has carried so far measures the bot's *activity*.
Three of them cannot answer "did this help", and the A/B that motivated this
module would have been read off them:

- ``engaged gone`` counts targets ordered against that are no longer visible. A
  hostile that retreated into fog reads identically to a dead one.
- ``enemies seen`` counts **visible** hostiles, so it rises when our own army
  walks somewhere new. It measures our vision as much as their army.
- ``army`` at the end is real, but three runs of identical code gave 3, 6 and 14
  ([[policy-economy]]). The variance swamps most effects worth measuring.

The engine keeps the answer itself and names it. A player carries a "was
defeated" flag and a "has been wiped out" flag, and the world knows how many
players remain -- the engine ends the match when that count reaches one
([[perception-visibility]]). None of the three can be inflated by re-targeting,
by scouting, or by a lucky sample.

Pure: a sample goes in and a verdict comes out.
"""

from __future__ import annotations

from rw_bot.wire.state import Sample

#: One player remains, and it is not us.
GRADE_WON = "won"

#: Still playing when the sample budget ran out. Not a win and not a loss --
#: the honest reading of a match that was stopped rather than decided.
GRADE_SURVIVED = "survived"

#: The engine set the player's "was defeated" flag.
GRADE_DEFEATED = "defeated"

#: The engine set the player's "has been wiped out" flag, which is the stronger
#: of the two and is reported in preference to it.
GRADE_WIPED = "wiped"


def grade(sample: Sample) -> str:
    """Return the engine's verdict on the player, as of this sample.

    Order matters. Losing is checked before winning because both can be true of
    the same observation: when we are eliminated the survivor count falls
    towards one as well, and reading that as a victory would grade a loss as a
    win. Wiped is checked before defeated for the same reason -- it is the
    stronger statement about the same event.

    Args:
        sample: One observation of the world.

    Returns:
        One of :data:`GRADE_WIPED`, :data:`GRADE_DEFEATED`, :data:`GRADE_WON`
        or :data:`GRADE_SURVIVED`.
    """
    if sample["wiped"]:
        return GRADE_WIPED
    if sample["defeated"]:
        return GRADE_DEFEATED
    if sample["players_left"] <= 1:
        return GRADE_WON
    return GRADE_SURVIVED


def eliminated(started_with: int, left_now: int) -> int:
    """Return how many players have gone since the phase opened.

    Clamped at zero. A player count that rose is not something the engine
    should produce, and reporting a negative elimination would be a stranger
    claim than reporting none -- so the anomaly is flattened here and shows up
    as a phase that eliminated nobody.

    Args:
        started_with: Players remaining when the phase opened.
        left_now: Players remaining now.

    Returns:
        How many players were eliminated, never below zero.
    """
    return max(0, started_with - left_now)


__all__ = [
    "GRADE_DEFEATED",
    "GRADE_SURVIVED",
    "GRADE_WIPED",
    "GRADE_WON",
    "eliminated",
    "grade",
]
