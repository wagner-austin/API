"""Earlier-only tuning and calibration; holdout labels never enter a fitted model."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_float,
    require_list,
    require_str,
)
from typing_extensions import ReadOnly, TypedDict

from ..calibration.calibrator import Calibrator, create_platt_calibrator
from ..calibration.types import decode_calibrator_state, encode_calibrator_state
from ..metrics import compute_brier_score
from .warning_folds import WarningFold, require_warning_study
from .warning_learning import (
    WarningLearnerFactory,
    predict_warning_probabilities,
    require_warning_probabilities,
    warning_features,
    warning_labels,
)


class WarningForecast(TypedDict):
    """Immutable held-out forecasts with a portable calibration state."""

    raw: ReadOnly[tuple[float, ...]]
    calibrated: ReadOnly[tuple[float, ...]]
    calibration_json: ReadOnly[str]


def decode_warning_forecast(document: JSONObject) -> WarningForecast:
    """Validate aligned probabilities and the existing calibration state codec.

    Args:
        document: Archived predictions and serialized calibration state.

    Returns:
        WarningForecast: Immutable probability vectors in input row order.

    Raises:
        TypeError: Required fields have invalid native types.
        ValueError: WARN-019 for empty, misaligned or invalid probabilities;
            calibration-state decoding errors propagate.
    """
    raw = tuple(require_float({"p": v}, "p") for v in require_list(document, "raw"))
    calibrated = tuple(require_float({"p": v}, "p") for v in require_list(document, "calibrated"))
    if not raw or len(raw) != len(calibrated):
        raise ValueError("[WARN-019] raw and calibrated predictions must align and be nonempty")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in (*raw, *calibrated)):
        raise ValueError("[WARN-019] archived probabilities must be finite and in [0, 1]")
    state = require_str(document, "calibration_json")
    decode_calibrator_state(narrow_json_to_dict(load_json_str(state)))
    result = WarningForecast(raw=raw, calibrated=calibrated, calibration_json=state)
    if replay_warning_calibration(result) != calibrated:
        raise ValueError("[WARN-019] archived calibrated values disagree with their fitted state")
    return result


def encode_warning_forecast(forecast: WarningForecast) -> JSONObject:
    """Encode probabilities and calibration state without mutable aliases.

    Args:
        forecast: Validated held-out predictions.

    Returns:
        JSONObject: Portable forecast representation.
    """
    return {
        "raw": list(forecast["raw"]),
        "calibrated": list(forecast["calibrated"]),
        "calibration_json": forecast["calibration_json"],
    }


def _fit_warning_forecast(factory: WarningLearnerFactory, fold: WarningFold) -> WarningForecast:
    """Fit and calibrate on explicit earlier partitions, then predict from features only.

    Args:
        factory: Fresh learner constructor receiving only fitting features and labels.
        fold: Chronology-validated fitting, calibration and evaluation partitions.

    Returns:
        WarningForecast: Raw and calibrated predictions for the evaluation rows.

    Raises:
        ValueError: WARN-020 if calibration lacks either class; learner and
            probability-contract failures propagate. No recovery or clipping occurs.
    """
    fitting = fold["fitting"]
    calibration = fold["calibration"]
    evaluation = fold["validation"]
    labels = warning_labels(calibration)
    classes = np.unique(labels)
    expected = np.asarray((0, 1), dtype=np.int64)
    if not bool(np.array_equal(classes, expected)):
        raise ValueError("[WARN-020] calibration requires both observed binary classes")
    model = factory(
        warning_features(tuple(row["disclosure"] for row in fitting)), warning_labels(fitting)
    )
    calibration_p = predict_warning_probabilities(
        model, tuple(row["disclosure"] for row in calibration)
    )
    calibrator = create_platt_calibrator(clip_proba=False)
    calibrator.fit(labels, calibration_p)
    raw = predict_warning_probabilities(model, tuple(row["disclosure"] for row in evaluation))
    calibrated = require_warning_probabilities(
        calibrator.transform(raw)["calibrated_proba"], len(evaluation)
    )
    return WarningForecast(
        raw=tuple(float(raw.flat[i]) for i in range(len(raw))),
        calibrated=tuple(float(calibrated.flat[i]) for i in range(len(calibrated))),
        calibration_json=dump_json_str(encode_calibrator_state(calibrator.get_state())),
    )


def run_warning_study(
    candidates: Mapping[str, WarningLearnerFactory],
    folds: Sequence[WarningFold],
    final: WarningFold,
    as_of_ms: int,
) -> tuple[str, dict[str, float], dict[str, WarningForecast]]:
    """Select by pooled development Brier and evaluate every frozen arm once later.

    Args:
        candidates: Named frozen learner specifications, including simple baselines.
        folds: Explicit chronological development folds, with separate calibration.
        final: Final fitting, untouched calibration and untouched later holdout.
        as_of_ms: Timestamp at which all scored labels must be available.

    Returns:
        tuple[str, dict[str, float], dict[str, WarningForecast]]: Selected name,
        development Brier by candidate, and raw/calibrated holdout probabilities.
        Selection uses equal weight per validation row and lexical name ties.
        Holdout labels are checked for availability but never used to select or fit.

    Raises:
        ValueError: WARN-021 for insufficient or invalid candidate names; study,
            learner and calibration errors propagate before results are returned.
    """
    if len(candidates) < 2 or any(not name or name != name.strip() for name in candidates):
        raise ValueError("[WARN-021] supply at least two uniquely named frozen candidates")
    require_warning_study(folds, final, as_of_ms)
    validation_scores: dict[str, float] = {}
    total = sum(len(fold["validation"]) for fold in folds)
    for name, factory in candidates.items():
        loss = 0.0
        for fold in folds:
            forecast = _fit_warning_forecast(factory, fold)
            probabilities = np.asarray(forecast["calibrated"], dtype=np.float64)
            loss += len(fold["validation"]) * compute_brier_score(
                warning_labels(fold["validation"]), probabilities
            )
        validation_scores[name] = loss / total
    selected = min((score, name) for name, score in validation_scores.items())[1]
    predictions = {
        name: _fit_warning_forecast(factory, final) for name, factory in candidates.items()
    }
    return selected, validation_scores, predictions


def replay_warning_calibration(forecast: WarningForecast) -> tuple[float, ...]:
    """Recompute archived calibration through the real calibrator implementation.

    Args:
        forecast: Decoded forecasts carrying their fitted calibration state.

    Returns:
        tuple[float, ...]: Recomputed calibrated values for exact replay comparison.
    """
    state = decode_calibrator_state(
        narrow_json_to_dict(load_json_str(forecast["calibration_json"]))
    )
    values = Calibrator.from_state(state).transform(np.asarray(forecast["raw"], dtype=np.float64))
    calibrated = values["calibrated_proba"]
    return tuple(float(calibrated.flat[i]) for i in range(len(calibrated)))


__all__ = [
    "WarningForecast",
    "decode_warning_forecast",
    "encode_warning_forecast",
    "replay_warning_calibration",
    "run_warning_study",
]
