"""Remembering which pools were taken from us, so the walk back can wait.

The economy claims pools and the survey re-offers a razed one the moment
nothing stands on it -- correctly, because occupancy is what the survey
judges. What no layer remembered is that the pool is razed rather than
merely free: the builder's walk back goes into the same fire that razed
it, and at Impossible the wave that took the extractor is usually still
there. The 96-scorecard autopsy that named the economy problem shows the
shape exactly -- owned peaks of five extractors razed to zero, `works
lost to c_tank x5, hoverTank x4` -- built, then razed, then re-walked
into the same wave ([[impossible-economy-problem]]).

This module holds the memory. Which pools count as OURS-AND-LOST is a
diff of owned extractor positions across observations; whether the walk
back may start is the expander's decision, gated on the same wave-break
signal the strike release reads ([[policy-situation]]) -- a razed pool is
reclaimable when the wave that is the reason it fell has itself broken.
"""

from __future__ import annotations

from rw_bot.mechanics.upgrades import TIER_CHAINS
from rw_bot.wire.state import Sample

#: Every type whose loss marks its pool as razed: the extractor line,
#: read from the same tier chains the upgrade policy climbs, so a tier
#: added there is tracked here without a second roster to forget.
EXTRACTOR_TYPES: frozenset[str] = frozenset(
    type_name for chain in TIER_CHAINS for type_name in chain
)


def _tile(x: float, y: float) -> tuple[int, int]:
    """Round a world position to the tile identity a structure keeps.

    Extractors never move, but the stream carries float coordinates and a
    float equality across samples is a promise the wire format never made.
    Rounding to the nearest whole unit gives a stable identity; the survey
    side compares with :func:`rw_bot.policy.siting.is_refused`'s one-unit
    tolerance against the original coordinates, which this class keeps.
    """
    return (round(x), round(y))


class Razed:
    """Watches owned extractors and remembers where one used to stand.

    Cross-sample memory with one owner, like :class:`~rw_bot.policy.situation.Momentum`:
    the campaign observes every sample, the expander reads the positions.

    A position leaves the memory the moment an owned extractor stands
    there again -- however it got there -- because the memory answers
    "is this pool razed", not "was it ever".

    Attributes are internal: the last observed owned tiles, and the razed
    tiles mapped to the world coordinates the survey compares against.
    """

    def __init__(self) -> None:
        self._owned: dict[tuple[int, int], tuple[float, float]] = {}
        self._razed: dict[tuple[int, int], tuple[float, float]] = {}

    def observe(self, sample: Sample) -> None:
        """Record this observation's owned extractors and diff the last.

        Args:
            sample: One observation of the world.
        """
        owned: dict[tuple[int, int], tuple[float, float]] = {}
        for entity in sample["entities"]:
            if entity["mine"] and entity["type_name"] in EXTRACTOR_TYPES:
                owned[_tile(entity["x"], entity["y"])] = (entity["x"], entity["y"])
        for tile, position in self._owned.items():
            if tile not in owned:
                self._razed[tile] = position
        for tile in owned:
            self._razed.pop(tile, None)
        self._owned = owned

    def positions(self) -> tuple[tuple[float, float], ...]:
        """Return where razed extractors stood, oldest-claim order not promised.

        Returns:
            The world coordinates, sorted for determinism.
        """
        return tuple(sorted(self._razed.values()))


def embargoed(
    razed: tuple[tuple[float, float], ...], wave_drop: int, rebuild: int
) -> tuple[tuple[float, float], ...]:
    """Decide which razed pools the survey must not offer this observation.

    Args:
        razed: Where razed extractors stood, from :meth:`Razed.positions`.
        wave_drop: How far the rival's army value sits below its recent
            peak, from :meth:`~rw_bot.policy.situation.Momentum.drop`.
        rebuild: The doctrine's threshold: the drop required before a
            razed pool may be re-claimed, zero for off.

    Returns:
        The positions to withhold from the pool survey. Empty when the
        knob is off -- today's behaviour, the walk back starts at once --
        and empty when the rival's wave has broken by at least the
        threshold, which is the door the walk goes through.
    """
    if rebuild <= 0 or wave_drop >= rebuild:
        return ()
    return razed
