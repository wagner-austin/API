"""The model shape every corpus-trained head deploys through.

One file format, one decoder, one scoring rule -- extracted from the
fleet-doom deployment when the second head arrived, so the two could not
drift into parallel copies of the same arithmetic ([[policy-exact-timing]]
for the first head's story; [[impossible-step-three-design]] for why a
second exists). A head is a standardized logistic model: NDJSON whose
first line carries the scalars and every later line one feature's
standardization and weight. What differs between heads is the WATCH --
which figures are photographed, over what window, scored when -- and that
stays in each head's own module.

Train/serve parity is the whole point of the shape: the exporter that
fits a head computes its features through the same watch class the loop
scores with, and both sides score through :func:`score_features`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TypedDict

from rw_bot import RwBotError
from rw_bot.validation import require_finite_float, require_int, require_non_empty_str
from rw_bot.wire.ndjson import parse_object

_BAD_SHAPE = "RW-HEAD-001"
_BAD_FEATURE = "RW-HEAD-002"


class HeadError(RwBotError):
    """A head model file could not be read, or a feature disagreed.

    Args:
        code: Stable machine-readable identifier -- ``RW-HEAD-001`` for a
            malformed model file, ``RW-HEAD-002`` when a model names a
            feature the watch did not compute.
        message: Human-readable description of what was malformed.
    """


class HeadModel(TypedDict):
    """One fitted standardized-logistic model, complete.

    Attributes:
        window: Samples the head's watch accumulates before scoring --
            the prediction moment the model was fitted at.
        threshold: Probability at or above which the prediction arms.
        intercept: The logistic intercept.
        features: Feature name -> ``(mean, std, coefficient)``, the
            standardization and weight for each input. Order never
            matters; features join by name.
    """

    window: int
    threshold: float
    intercept: float
    features: Mapping[str, tuple[float, float, float]]


def decode_head_model(lines: Sequence[str]) -> HeadModel:
    """Decode a head model from its NDJSON lines.

    The first line carries the scalars, every later line one feature.

    Args:
        lines: The model file's lines, without newline terminators.

    Returns:
        The validated model.

    Raises:
        HeadError: ``RW-HEAD-001`` when the head line is malformed or no
            feature lines follow.
    """
    if not lines:
        raise HeadError(_BAD_SHAPE, "a head model file cannot be empty")
    head = parse_object(lines[0])
    window = require_int(head, "window")
    if window <= 0:
        raise HeadError(_BAD_SHAPE, f"the window must be positive, got {window}")
    threshold = require_finite_float(head, "threshold")
    intercept = require_finite_float(head, "intercept")
    features: dict[str, tuple[float, float, float]] = {}
    for line in lines[1:]:
        record = parse_object(line)
        name = require_non_empty_str(record, "name")
        std = require_finite_float(record, "std")
        if std <= 0.0:
            raise HeadError(_BAD_SHAPE, f"feature {name} carries a non-positive std: {std}")
        features[name] = (
            require_finite_float(record, "mean"),
            std,
            require_finite_float(record, "coef"),
        )
    if not features:
        raise HeadError(_BAD_SHAPE, "a head model carries at least one feature line")
    return HeadModel(
        window=window, threshold=float(threshold), intercept=float(intercept), features=features
    )


def score_features(model: HeadModel, feats: Mapping[str, float]) -> float:
    """Return the logistic probability of one feature set under one model.

    Args:
        model: The fitted model to score against.
        feats: Feature name to value, from the head's own watch.

    Returns:
        The logistic probability.

    Raises:
        HeadError: ``RW-HEAD-002`` when the model names a feature the
            watch did not compute -- a train/serve drift, not data.
    """
    z = model["intercept"]
    for name, (mean, std, coef) in model["features"].items():
        if name not in feats:
            raise HeadError(_BAD_FEATURE, f"the model names an unknown feature: {name}")
        z += coef * ((feats[name] - mean) / std)
    return 1.0 / (1.0 + math.exp(-z))


__all__ = ["HeadError", "HeadModel", "decode_head_model", "score_features"]
