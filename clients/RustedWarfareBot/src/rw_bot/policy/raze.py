"""Predicting the economy's razing from the match's own recent shape.

The Impossible autopsy named the death: the economy is built, then razed,
and every static answer measured flat -- denial, rebuild timing, the
strike release, each with its mechanism proven firing
([[impossible-economy-problem]]). What no static rule can express is a
CONTEXTUAL pivot: play normally while the economy is safe, and change
spending posture when this match's own shape says the razing is coming.
That trigger is learnable exactly the way fleet doom was
([[policy-exact-timing]]): a standardized logistic head over shape
statistics, deployed through :mod:`rw_bot.policy.head`'s one model format
and one scoring rule.

Two things distinguish this watch from the doom watch, and both are the
prediction's nature rather than new machinery. The window SLIDES: razing
happens mid-match at a time that varies by seed, so the features are a
photograph of the last ``window`` samples, recomputed as the match moves,
where doom's photograph of the opening is taken once. And the momentum
column rides in: ``rival_army`` and its within-window drop are inputs,
because the wave that razes announces itself on the scoreboard first
(imptr12 measured the signal's range; log 2026-09-04).

The latch still arms ONCE (law eight): the brace it drives reshapes the
match, so the decision is one decision. Train/serve parity is by
construction -- the exporter that fits the model computes features
through this same class.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Final

from rw_bot import RwBotError
from rw_bot.policy.head import HeadModel, score_features

_BAD_FEATURE = "RW-RAZE-001"

#: The per-sample figures the watch consumes, in feed order -- the trace's
#: economy-and-pressure columns, exactly ([[policy-trace]]). The spatial
#: trio is here because an extractor standing inside hostile reach is the
#: most literal "about to be razed" a column can say, and ``rival_army``
#: because the wave that does the razing is scoreboard-visible first.
COLUMNS: Final = (
    "army",
    "credits",
    "extractors",
    "lost",
    "workers",
    "income",
    "worth",
    "rival",
    "rival_income",
    "rival_army",
    "eco_covered",
    "own_covered",
    "foe_covered",
)


class RazeError(RwBotError):
    """The raze watch was fed or read wrongly.

    Args:
        code: ``RW-RAZE-001``: a wrong figure count, or features read
            before the window fills. Model-file errors are
            :class:`~rw_bot.policy.head.HeadError`'s, with the decoder.
        message: Human-readable description of what was malformed.
    """


class RazeWatch:
    """A sliding photograph of the last ``window`` samples' shape.

    Per column: the window mean, the latest value, the window max and
    min, and the second-half-minus-first-half slope -- the doom feature
    vocabulary over a moving window. Plus the drop pair the calibration
    probe measured: ``rival_army_drop`` (window max minus latest, the
    exact momentum figure) and ``extractors_drop`` (the razing already
    under way).
    """

    def __init__(self, window: int) -> None:
        """Open a watch.

        Args:
            window: Samples the photograph spans, at least two -- the
                half-split slope needs both halves populated. The oldest
                falls out as each new one arrives.

        Raises:
            RazeError: ``RW-RAZE-001`` on a window below two.
        """
        if window < 2:
            raise RazeError(
                _BAD_FEATURE,
                f"the sliding window needs both slope halves, got {window}",
            )
        self._window = window
        self._rows: deque[tuple[float, ...]] = deque(maxlen=window)

    def feed(self, values: Sequence[int]) -> None:
        """Read one sample's figures, in :data:`COLUMNS` order.

        Args:
            values: One figure per column.

        Raises:
            RazeError: ``RW-RAZE-001`` when the figure count disagrees
                with the column list -- a wiring bug, not data.
        """
        if len(values) != len(COLUMNS):
            raise RazeError(
                _BAD_FEATURE,
                f"the watch is fed {len(COLUMNS)} figures, got {len(values)}",
            )
        self._rows.append(tuple(float(value) for value in values))

    def full(self) -> bool:
        """Report whether the window has filled and the features stand."""
        return len(self._rows) >= self._window

    def features(self) -> dict[str, float]:
        """Return the training features for the current window.

        Returns:
            Feature name to value, exactly the dataset builder's set.

        Raises:
            RazeError: ``RW-RAZE-001`` before the window fills -- a
                partial window would leak match age into every mean, the
                same trap the doom watch refuses.
        """
        if not self.full():
            raise RazeError(
                _BAD_FEATURE,
                f"the window holds {len(self._rows)} of {self._window} samples",
            )
        n = float(self._window)
        half = self._window // 2
        feats: dict[str, float] = {}
        for j, name in enumerate(COLUMNS):
            column = [row[j] for row in self._rows]
            mean = sum(column) / n
            feats[f"{name}_mean"] = mean
            feats[f"{name}_last"] = column[-1]
            feats[f"{name}_max"] = max(column)
            feats[f"{name}_min"] = min(column)
            first = sum(column[:half]) / float(half)
            second = sum(column[half:]) / float(self._window - half)
            feats[f"{name}_slope"] = second - first
        feats["rival_army_drop"] = feats["rival_army_max"] - feats["rival_army_last"]
        feats["extractors_drop"] = feats["extractors_max"] - feats["extractors_last"]
        return feats

    def score(self, model: HeadModel) -> float:
        """Return the razing probability under one model.

        Args:
            model: The fitted model to score against.

        Returns:
            The logistic probability of razing.

        Raises:
            RazeError: ``RW-RAZE-001`` before the window fills.
            HeadError: ``RW-HEAD-002`` through
                :func:`~rw_bot.policy.head.score_features` when the model
                names a feature the watch does not compute.
        """
        return score_features(model, self.features())


class BraceLatch:
    """The sliding watch and its verdict, latched: arm once, hold.

    The loop's whole interface to the prediction -- feed it every
    sample's figures and read ``armed``. Scoring happens every sample
    once the window fills, but arming happens ONCE: the brace it drives
    reshapes the match, so the decision is one decision (law eight),
    exactly the doom latch's discipline on a moving photograph.
    """

    def __init__(self, model: HeadModel) -> None:
        """Open a latch over one model.

        Args:
            model: The fitted model, scored at its own window size.
        """
        self._model = model
        self._watch = RazeWatch(model["window"])
        self.armed = False

    def feed(self, values: Sequence[int]) -> None:
        """Read one sample's figures; score and latch when it clears.

        Args:
            values: One figure per :data:`COLUMNS` entry.

        Raises:
            RazeError: ``RW-RAZE-001`` on a wrong figure count.
        """
        self._watch.feed(values)
        if not self.armed and self._watch.full():
            self.armed = self._watch.score(self._model) >= self._model["threshold"]


__all__ = ["COLUMNS", "BraceLatch", "RazeError", "RazeWatch"]
