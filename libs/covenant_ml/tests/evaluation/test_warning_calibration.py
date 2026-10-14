"""Reliability checks with independently hand-calculated and library reference scores."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pytest
from numpy.typing import NDArray
from platform_core.json_utils import JSONValue

from covenant_ml.evaluation.warning_calibration import (
    decode_warning_calibration,
    encode_warning_calibration,
    summarize_warning_calibration,
    warning_calibration_error,
)

from .warning_testing import cohort


class BrierReference(Protocol):
    """Minimal positional sklearn Brier scorer signature."""

    def __call__(self, y_true: NDArray[np.int64], y_proba: NDArray[np.float64], /) -> float: ...


def test_hand_calculation_and_independent_sklearn_agree() -> None:
    rows = cohort("A", 0)
    probabilities = (0.0, 0.25, 0.75, 1.0)
    result = summarize_warning_calibration(rows, probabilities, 4)
    assert encode_warning_calibration(result) == {
        "brier": 0.03125,
        "counts": [1, 1, 0, 2],
        "probability_sums": [0.0, 0.25, 0.0, 1.75],
        "positives": [0, 0, 0, 2],
    }
    assert warning_calibration_error(result) == 0.125
    assert decode_warning_calibration(encode_warning_calibration(result)) == result
    reference: BrierReference = __import__(
        "sklearn.metrics", fromlist=["brier_score_loss"]
    ).brier_score_loss
    labels: NDArray[np.int64] = np.asarray((0, 0, 1, 1), dtype=np.int64)
    p: NDArray[np.float64] = np.asarray(probabilities, dtype=np.float64)
    assert result["brier"] == reference(labels, p)
    perfect = summarize_warning_calibration(rows, (0.0, 0.0, 1.0, 1.0), 4)
    inverted = summarize_warning_calibration(rows, (1.0, 1.0, 0.0, 0.0), 4)
    assert perfect["brier"] == warning_calibration_error(perfect) == 0.0
    assert inverted["brier"] == warning_calibration_error(inverted) == 1.0


@pytest.mark.parametrize("bins", [0, -1, True])
def test_invalid_bins(bins: int) -> None:
    with pytest.raises(ValueError, match="WARN-027"):
        summarize_warning_calibration(cohort("A", 0), (0.0, 0.0, 1.0, 1.0), bins)


def test_empty_cohort() -> None:
    with pytest.raises(ValueError, match="WARN-027"):
        summarize_warning_calibration((), (), 4)


@pytest.mark.parametrize(
    "key,value",
    [
        ("brier", -0.1),
        ("brier", float("nan")),
        ("counts", []),
        ("counts", [0, 0]),
        ("probability_sums", [0.0]),
        ("positives", [0]),
        ("positives", [3, 1]),
        ("probability_sums", [float("nan"), 1.0]),
        ("probability_sums", [3.0, 1.0]),
    ],
)
def test_invalid_statistics(key: str, value: JSONValue) -> None:
    result = summarize_warning_calibration(cohort("A", 0), (0.0, 0.25, 0.75, 1.0), 2)
    document = encode_warning_calibration(result)
    document[key] = value
    with pytest.raises(ValueError, match="WARN-026"):
        decode_warning_calibration(document)
