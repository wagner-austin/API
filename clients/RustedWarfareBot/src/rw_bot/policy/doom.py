"""Predicting fleet doom from the match's own early shape.

The three-calibration arc proved no present-tense scalar can arm a
match-reshaping response ([[policy-exact-timing]]; log 2026-08-08): naval
contact is nearly universal, naval doom is rare, and every gate traded
rescues for re-rolled wins at net -2. Law eight named the requirement --
prediction -- and the sighted trace columns made it learnable: a
standardized logistic model over the first 2,000 samples' shape statistics
reads a fresh tree's matches at AUC 0.75 with precision saturating near
0.7 (log 2026-08-09, the replication verdict).

This module is that model's watch: the class that recomputes the training
features in-match, and the latch that scores once at the window. The file
format and the scoring rule moved to :mod:`rw_bot.policy.head` when the
second head arrived -- one decoder for every head, so two deployments
cannot drift into parallel copies of the same arithmetic. **Train/serve
parity is by construction, not by discipline**: the exporter that fits the
model computes its features through this same class, so the arithmetic
cannot drift between the dataset and the loop.

Pure throughout: values in, a score out. The loop feeds the watch the same
figures it hands the recorder, which are the same figures the trace files
carry, which are what the model was fitted on.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

from rw_bot import RwBotError
from rw_bot.policy.head import HeadModel, score_features

_BAD_FEATURE = "RW-DOOM-002"

#: The per-sample figures the watch consumes, in feed order -- the trace's
#: numeric columns, exactly ([[policy-trace]]).
COLUMNS: Final = (
    "army",
    "credits",
    "enemies",
    "extractors",
    "lost",
    "producers",
    "idle",
    "orders",
    "refused",
    "worth",
    "rival",
    "income",
    "rival_income",
    "workers",
    "navy_seen",
    "air_seen",
    "navy_blood",
)


class DoomError(RwBotError):
    """The doom watch was fed or read wrongly.

    Args:
        code: ``RW-DOOM-002``, the module's one remaining code: a wrong
            figure count, or features read before the window closes.
            Model-file errors moved to :class:`~rw_bot.policy.head.HeadError`
            with the decoder.
        message: Human-readable description of what was malformed.
    """


class DoomWatch:
    """Accumulates the early window's shape and scores it once, at the window.

    The training features, recomputed live: per column the mean, the value
    at the window's close, the max, the min, and the second-half-minus-
    first-half slope; the two rival ratios; and the naval timing trio --
    first sight, first blood, mean pressure. The exporter fits on features
    from this same class, so what the model sees in the loop is what it
    saw in training, by construction.
    """

    def __init__(self, window: int) -> None:
        """Open a watch.

        Args:
            window: Samples to accumulate before the features close.
        """
        self._window = window
        self._half = window // 2
        self._seen = 0
        self._sums = [0.0] * len(COLUMNS)
        self._first_half_sums = [0.0] * len(COLUMNS)
        self._mins = [math.inf] * len(COLUMNS)
        self._maxes = [-math.inf] * len(COLUMNS)
        self._lasts = [0.0] * len(COLUMNS)
        self._first_navy_contact = window
        self._first_navy_blood = window

    def feed(self, values: Sequence[int]) -> None:
        """Read one sample's figures, in :data:`COLUMNS` order.

        Samples past the window are ignored: the features are a photograph
        of the early game, and the model's moment does not move.

        Args:
            values: One figure per column.

        Raises:
            DoomError: ``RW-DOOM-002`` when the figure count disagrees
                with the column list -- a wiring bug, not data.
        """
        if len(values) != len(COLUMNS):
            raise DoomError(
                _BAD_FEATURE,
                f"the watch is fed {len(COLUMNS)} figures, got {len(values)}",
            )
        if self._seen >= self._window:
            return
        index = self._seen
        self._seen += 1
        for j, value in enumerate(values):
            figure = float(value)
            self._sums[j] += figure
            if index < self._half:
                self._first_half_sums[j] += figure
            if figure < self._mins[j]:
                self._mins[j] = figure
            if figure > self._maxes[j]:
                self._maxes[j] = figure
            self._lasts[j] = figure
        navy = float(values[COLUMNS.index("navy_seen")])
        blood = float(values[COLUMNS.index("navy_blood")])
        if navy > 0 and self._first_navy_contact == self._window:
            self._first_navy_contact = index
        if blood > 0 and self._first_navy_blood == self._window:
            self._first_navy_blood = index

    def closed(self) -> bool:
        """Report whether the window has filled and the features stand."""
        return self._seen >= self._window

    def features(self) -> dict[str, float]:
        """Return the training features for the accumulated window.

        Returns:
            Feature name to value, exactly the dataset builder's set.

        Raises:
            DoomError: ``RW-DOOM-002`` before the window closes -- a
                partial window would leak match length into every mean,
                the same trap the dataset builder refuses.
        """
        if not self.closed():
            raise DoomError(
                _BAD_FEATURE,
                f"the window holds {self._seen} of {self._window} samples",
            )
        n = float(self._window)
        half = float(self._half)
        feats: dict[str, float] = {}
        for j, name in enumerate(COLUMNS):
            mean = self._sums[j] / n
            feats[f"{name}_mean"] = mean
            feats[f"{name}_last"] = self._lasts[j]
            feats[f"{name}_max"] = self._maxes[j]
            feats[f"{name}_min"] = self._mins[j]
            second = (self._sums[j] - self._first_half_sums[j]) / (n - half)
            feats[f"{name}_slope"] = second - self._first_half_sums[j] / half
        feats["rival_income_ratio"] = feats["rival_income_mean"] / max(feats["income_mean"], 1.0)
        feats["rival_worth_ratio"] = feats["rival_mean"] / max(feats["worth_mean"], 1.0)
        feats["first_navy_contact"] = float(self._first_navy_contact)
        feats["first_navy_blood"] = float(self._first_navy_blood)
        feats["navy_pressure"] = self._sums[COLUMNS.index("navy_seen")] / n
        return feats

    def score(self, model: HeadModel) -> float:
        """Return the doom probability under one model.

        Args:
            model: The fitted model to score against.

        Returns:
            The logistic probability of doom.

        Raises:
            DoomError: ``RW-DOOM-002`` before the window closes.
            HeadError: ``RW-HEAD-002`` through
                :func:`~rw_bot.policy.head.score_features` when the model
                names a feature the watch does not compute.
        """
        return score_features(model, self.features())


class DoomLatch:
    """The watch and its verdict, latched: score once at the window, hold.

    The loop's whole interface to the prediction -- feed it every sample's
    figures and read ``armed``. Scoring happens exactly once, the sample
    the window closes; the answer then stands for the match, because a
    prediction that flaps is a scalar gate wearing a model's coat
    (law eight: the response it arms reshapes the match, so the decision
    must be one decision).
    """

    def __init__(self, model: HeadModel) -> None:
        """Open a latch over one model.

        Args:
            model: The fitted model to score with at its own window.
        """
        self._model = model
        self._watch = DoomWatch(model["window"])
        self._scored = False
        self.armed = False

    def feed(self, values: Sequence[int]) -> None:
        """Read one sample's figures; score and latch at the window.

        Args:
            values: One figure per :data:`COLUMNS` entry.

        Raises:
            DoomError: ``RW-DOOM-002`` on a wrong figure count.
        """
        self._watch.feed(values)
        if not self._scored and self._watch.closed():
            self._scored = True
            self.armed = self._watch.score(self._model) >= self._model["threshold"]


__all__ = [
    "COLUMNS",
    "DoomError",
    "DoomLatch",
    "DoomWatch",
]
