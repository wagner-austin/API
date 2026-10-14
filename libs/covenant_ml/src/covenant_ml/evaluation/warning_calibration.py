"""Fixed-bin reliability summaries on held-out warning probabilities."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from platform_core.json_utils import JSONObject, require_float, require_int, require_list
from typing_extensions import ReadOnly, TypedDict

from ..metrics import compute_brier_score
from .warning_learning import require_warning_probabilities, warning_labels
from .warning_panel import WarningRow


class WarningCalibration(TypedDict):
    """Brier and sufficient bin counts for an explicitly equal-width reliability plot."""

    brier: ReadOnly[float]
    counts: ReadOnly[tuple[int, ...]]
    probability_sums: ReadOnly[tuple[float, ...]]
    positives: ReadOnly[tuple[int, ...]]


def decode_warning_calibration(document: JSONObject) -> WarningCalibration:
    """Require valid aligned reliability statistics, retaining empty bins.

    Args:
        document: Serialized calibration summary.

    Returns:
        WarningCalibration: Immutable fixed-bin statistics.

    Raises:
        TypeError: Fields or array elements have invalid types.
        ValueError: WARN-026 for impossible counts or nonfinite statistics.
    """
    brier = require_float(document, "brier")
    counts = tuple(require_int({"n": v}, "n") for v in require_list(document, "counts"))
    sums = tuple(require_float({"p": v}, "p") for v in require_list(document, "probability_sums"))
    positives = tuple(require_int({"n": v}, "n") for v in require_list(document, "positives"))
    if not math.isfinite(brier) or not 0 <= brier <= 1:
        raise ValueError("[WARN-026] Brier score must be finite and in [0, 1]")
    if not counts or len(counts) != len(sums) or len(sums) != len(positives) or sum(counts) <= 0:
        raise ValueError("[WARN-026] nonempty aligned reliability bins are required")
    for count, total, events in zip(counts, sums, positives, strict=True):
        if not 0 <= events <= count or not math.isfinite(total) or not 0 <= total <= count:
            raise ValueError("[WARN-026] bin counts and probability sums are inconsistent")
    return WarningCalibration(
        brier=brier, counts=counts, probability_sums=sums, positives=positives
    )


def encode_warning_calibration(result: WarningCalibration) -> JSONObject:
    """Encode reliability sufficient statistics without inventing values for empty bins.

    Args:
        result: Validated calibration summary.

    Returns:
        JSONObject: Independent serializable statistics.
    """
    return {
        "brier": result["brier"],
        "counts": list(result["counts"]),
        "probability_sums": list(result["probability_sums"]),
        "positives": list(result["positives"]),
    }


def summarize_warning_calibration(
    rows: Sequence[WarningRow], probabilities: Sequence[float], bins: int
) -> WarningCalibration:
    """Compute held-out Brier and equal-width bins using a predeclared bin count.

    Args:
        rows: Validated mature labeled evaluation opportunities.
        probabilities: Aligned held-out probabilities, not arbitrary ranking scores.
        bins: Positive integer count fixed before evaluation; final bin includes one.

    Returns:
        WarningCalibration: Counts, summed probabilities and positives in each bin.
        Divide each nonempty bin's sums by its count for reliability coordinates.

    Raises:
        ValueError: WARN-027 for absent rows or invalid bin count; probability and
            censored-label contract failures propagate.
    """
    if not rows or isinstance(bins, bool) or bins <= 0:
        raise ValueError("[WARN-027] nonempty rows and a positive integer bin count are required")
    values = require_warning_probabilities(np.asarray(probabilities, dtype=np.float64), len(rows))
    labels = warning_labels(rows)
    counts = [0] * bins
    sums = [0.0] * bins
    positives = [0] * bins
    for index in range(len(rows)):
        p = float(values.flat[index])
        bucket = min(bins - 1, int(p * bins))
        counts[bucket] += 1
        sums[bucket] += p
        positives[bucket] += int(labels.flat[index])
    return WarningCalibration(
        brier=compute_brier_score(labels, values),
        counts=tuple(counts),
        probability_sums=tuple(sums),
        positives=tuple(positives),
    )


def warning_calibration_error(result: WarningCalibration) -> float:
    """Derive the weighted absolute bin calibration error for the declared bins.

    Args:
        result: Validated nonempty reliability statistics.

    Returns:
        float: Bin-dependent descriptive error, not a significance test.
    """
    return sum(
        abs(p - y) for p, y in zip(result["probability_sums"], result["positives"], strict=True)
    ) / sum(result["counts"])


__all__ = [
    "WarningCalibration",
    "decode_warning_calibration",
    "encode_warning_calibration",
    "summarize_warning_calibration",
    "warning_calibration_error",
]
