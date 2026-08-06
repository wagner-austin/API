"""The shared assessment layer: what this tick's world amounts to.

:mod:`rw_bot.policy.situation` proved the pattern -- scoreboard figures rode
the wire all campaign, used for one report line, until the closer needed
them as decisions ([[policy-situation]]). This module is where that pattern
grows into a layer: per-tick conclusions drawn once by the campaign and
handed to the channels that need them, instead of each channel re-reading
raw samples and keeping a private memory of what it saw.

The first tenant is the air watch, moved here from the expander's private
state: whether the opponent has ever shown aircraft is a fact about the
MATCH, not about the expansion channel that happened to latch it -- the
production tilt and the AA arms need the same answer, and two private
latches would be two chances to disagree. Coalition totals (the 1v3
lesson: "best rival" undercounts a three-AI seating threefold) and the
engageability census (154 enemies seen, 0 engageable) arrive next, each
with its first consumer in the same change.
"""

from __future__ import annotations

from rw_bot.wire.state import Sample


class AirWatch:
    """Latches the first hostile aircraft sighting, for the whole match.

    Latched, not sampled: aircraft leave the viewport and come back, and an
    answer that stands down between sorties arms anti-air that is never
    finished when the sortie arrives. The latch was measured into existence
    on the expander -- AA that re-derived the sighting per tick never stood
    ([[policy-holding-ground]]) -- and lives here now so every consumer
    reads one answer.
    """

    def __init__(self) -> None:
        self._seen = False

    def observe(self, sample: Sample) -> None:
        """Record whether this observation shows a hostile aircraft.

        Args:
            sample: One observation of the world.
        """
        if not self._seen:
            self._seen = any(
                entity["hostile"] and entity["flying"] for entity in sample["entities"]
            )

    def seen(self) -> bool:
        """Report whether the opponent has ever shown aircraft.

        Returns:
            True from the first sighting, forever after.
        """
        return self._seen


__all__ = ["AirWatch"]
