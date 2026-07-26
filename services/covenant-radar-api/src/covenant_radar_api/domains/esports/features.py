"""Match state feature extraction for esports win prediction.

Derives per-snapshot features describing which side is ahead and by how
much. Every feature is a pure function of a single MatchEventV1, so
extraction is stateless and needs no fitted state, unlike weather.

Features produced per event:
- kill_diff, gold_diff, tower_diff, dragon_diff, baron_diff: blue minus red
- gold_diff_per_minute: gold lead normalised by elapsed time, so an early
  lead is not read as equivalent to the same lead late
- blue_kill_ratio, blue_gold_ratio: blue's share, in [0, 1]
- game_time_minutes: elapsed time, which scales how decisive a lead is
- blue_objectives, red_objectives, objective_diff: towers, dragons and
  barons summed per side

Strict typing: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .schemas import MatchEventV1

ESPORTS_FEATURE_NAMES: tuple[str, ...] = (
    "kill_diff",
    "gold_diff",
    "gold_diff_per_minute",
    "tower_diff",
    "dragon_diff",
    "baron_diff",
    "blue_kill_ratio",
    "blue_gold_ratio",
    "game_time_minutes",
    "blue_objectives",
    "red_objectives",
    "objective_diff",
)

# A share when neither side has scored. Zero would read as "red holds
# everything", which is the opposite of what an empty scoreline means; even
# is the only reading consistent with the ratio's definition.
_EVEN_SHARE = 0.5

# Gold per minute before any time has elapsed. No rate exists yet, and any
# non-zero value would assert a trend from a single instant.
_NO_RATE = 0.0

_SECONDS_PER_MINUTE = 60.0


def _share(blue: int, red: int) -> float:
    """Return blue's share of a two-sided total.

    Args:
        blue: Blue side's count.
        red: Red side's count.

    Returns:
        Blue's fraction of the total, or an even share when the total is
        zero, which is the state at the start of every game.
    """
    total = blue + red
    if total == 0:
        return _EVEN_SHARE
    return float(blue) / float(total)


class EsportsFeatureExtractor:
    """Extract win-probability features from match state snapshots.

    Stateless: every feature is computed from the event alone, so no fitted
    state is injected and the extractor is safe to share across matches.

    Attributes:
        feature_names: Ordered tuple of feature names matching extract output.
    """

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return the ordered feature names this extractor produces."""
        return ESPORTS_FEATURE_NAMES

    def extract(self, event: MatchEventV1) -> NDArray[np.float64]:
        """Extract the feature vector from a single match snapshot.

        Args:
            event: Match state snapshot with per-side kills, gold and
                objective counts.

        Returns:
            Feature vector of shape (12,) with dtype float64, ordered to
            match ESPORTS_FEATURE_NAMES.
        """
        kill_diff = float(event["blue_kills"] - event["red_kills"])
        gold_diff = float(event["blue_gold"] - event["red_gold"])
        tower_diff = float(event["blue_towers"] - event["red_towers"])
        dragon_diff = float(event["blue_dragons"] - event["red_dragons"])
        baron_diff = float(event["blue_barons"] - event["red_barons"])

        game_time_minutes = float(event["game_time_seconds"]) / _SECONDS_PER_MINUTE
        gold_diff_per_minute = (
            _NO_RATE if game_time_minutes == 0.0 else gold_diff / game_time_minutes
        )

        blue_objectives = float(event["blue_towers"] + event["blue_dragons"] + event["blue_barons"])
        red_objectives = float(event["red_towers"] + event["red_dragons"] + event["red_barons"])

        result: NDArray[np.float64] = np.zeros(len(ESPORTS_FEATURE_NAMES), dtype=np.float64)
        result[0] = kill_diff
        result[1] = gold_diff
        result[2] = gold_diff_per_minute
        result[3] = tower_diff
        result[4] = dragon_diff
        result[5] = baron_diff
        result[6] = _share(event["blue_kills"], event["red_kills"])
        result[7] = _share(event["blue_gold"], event["red_gold"])
        result[8] = game_time_minutes
        result[9] = blue_objectives
        result[10] = red_objectives
        result[11] = blue_objectives - red_objectives
        return result


__all__ = [
    "ESPORTS_FEATURE_NAMES",
    "EsportsFeatureExtractor",
]
