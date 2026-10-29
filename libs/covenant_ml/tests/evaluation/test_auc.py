"""Tests for the rank AUC, DeLong's paired test and the weighted AUC.

DeLong's variance is checked against a brute-force implementation written
from the definition (every positive-negative pair compared), so the fast
midrank form is verified by an independent computation, not by itself.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.evaluation.auc import (
    bootstrap_auc_difference,
    delong_auc_difference,
    score_bins,
    weighted_auc,
)
from covenant_ml.evaluation.resampling import build_resampling
from covenant_ml.evaluation.types import (
    ERR_DEGENERATE_RESAMPLE,
    ERR_SINGLE_CLASS,
    IntervalMethod,
    ResamplingUnit,
)
from covenant_ml.metrics import compute_auc

from ._fixtures import floats, ints

LABELS = [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
SCORES_A = [0.9, 0.3, 0.8, 0.45, 0.5, 0.1, 0.7, 0.2, 0.65, 0.55, 0.35, 0.05, 0.6, 0.15]
SCORES_B = [0.6, 0.4, 0.9, 0.2, 0.3, 0.25, 0.5, 0.1, 0.7, 0.8, 0.45, 0.35, 0.55, 0.05]


def _psi(positive: float, negative: float) -> float:
    if positive > negative:
        return 1.0
    if positive == negative:
        return 0.5
    return 0.0


def _brute_force_components(
    labels: list[int], scores: list[float]
) -> tuple[float, list[float], list[float]]:
    positives = [s for s, y in zip(scores, labels, strict=True) if y == 1]
    negatives = [s for s, y in zip(scores, labels, strict=True) if y == 0]
    per_positive = [sum(_psi(p, n) for n in negatives) / len(negatives) for p in positives]
    per_negative = [sum(_psi(p, n) for p in positives) / len(positives) for n in negatives]
    return sum(per_positive) / len(per_positive), per_positive, per_negative


def _variance(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / (len(values) - 1)


def test_delong_auc_matches_headline_auc_on_untied_scores() -> None:
    result = delong_auc_difference(ints(LABELS), floats(SCORES_A), floats(SCORES_B), 0.95)
    assert result.auc_arm == pytest.approx(compute_auc(ints(LABELS), floats(SCORES_A)))
    assert result.auc_reference == pytest.approx(compute_auc(ints(LABELS), floats(SCORES_B)))


def test_delong_variance_matches_the_pairwise_definition() -> None:
    auc_a, pos_a, neg_a = _brute_force_components(LABELS, SCORES_A)
    auc_b, pos_b, neg_b = _brute_force_components(LABELS, SCORES_B)
    variance = _variance([a - b for a, b in zip(pos_a, pos_b, strict=True)]) / len(pos_a) + (
        _variance([a - b for a, b in zip(neg_a, neg_b, strict=True)]) / len(neg_a)
    )
    result = delong_auc_difference(ints(LABELS), floats(SCORES_A), floats(SCORES_B), 0.95)
    assert result.difference["estimate"] == pytest.approx(auc_a - auc_b)
    assert result.difference["std_error"] == pytest.approx(math.sqrt(variance))
    assert result.difference["method"] is IntervalMethod.DELONG
    assert result.difference["n_units"] == len(LABELS)


def test_delong_handles_ties_as_halves() -> None:
    tied = [0.5, 0.5, 0.9, 0.1, 0.5, 0.2, 0.9, 0.5, 0.3, 0.5, 0.5, 0.1, 0.9, 0.2]
    expected, _, _ = _brute_force_components(LABELS, tied)
    result = delong_auc_difference(ints(LABELS), floats(tied), floats(SCORES_B), 0.95)
    assert result.auc_arm == pytest.approx(expected)
    assert compute_auc(ints(LABELS), floats(tied)) == pytest.approx(expected)
    reversed_order: list[int] = list(reversed(range(len(LABELS))))
    rotated_order: list[int] = [*range(1, len(LABELS)), 0]
    for order in (reversed_order, rotated_order):
        labels = ints([LABELS[index] for index in order])
        scores = floats([tied[index] for index in order])
        assert compute_auc(labels, scores) == pytest.approx(expected)


def test_identical_models_differ_by_exactly_nothing() -> None:
    result = delong_auc_difference(ints(LABELS), floats(SCORES_A), floats(SCORES_A), 0.95)
    assert result.difference["estimate"] == 0.0
    assert result.difference["std_error"] == 0.0
    assert result.difference["p_value"] == 1.0


def test_delong_needs_two_rows_of_each_class() -> None:
    labels = ints([1, 0, 0, 0])
    scores = floats([0.9, 0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match=ERR_SINGLE_CLASS):
        delong_auc_difference(labels, scores, scores, 0.95)


def test_delong_refuses_misaligned_scores() -> None:
    with pytest.raises(ValueError, match="CVML-METRIC-001"):
        delong_auc_difference(ints(LABELS), floats(SCORES_A[:-1]), floats(SCORES_B), 0.95)


def test_weighted_auc_with_unit_weights_is_the_rank_auc() -> None:
    tied = [0.5, 0.5, 0.9, 0.1, 0.5, 0.2, 0.9, 0.5, 0.3, 0.5, 0.5, 0.1, 0.9, 0.2]
    expected, _, _ = _brute_force_components(LABELS, tied)
    ones: NDArray[np.float64] = np.ones(len(LABELS), dtype=np.float64)
    assert weighted_auc(score_bins(floats(tied)), ints(LABELS), ones) == pytest.approx(expected)


def test_integer_weights_equal_repeating_the_rows() -> None:
    weights = [2, 0, 1, 3, 1, 0, 2, 1, 1, 0, 2, 1, 1, 3]
    repeated_labels = [y for y, w in zip(LABELS, weights, strict=True) for _ in range(w)]
    repeated_scores = [s for s, w in zip(SCORES_A, weights, strict=True) for _ in range(w)]
    expected, _, _ = _brute_force_components(repeated_labels, repeated_scores)
    weight_array = floats([float(w) for w in weights])
    got = weighted_auc(score_bins(floats(SCORES_A)), ints(LABELS), weight_array)
    assert got == pytest.approx(expected)


def test_weights_that_empty_a_class_are_refused() -> None:
    weights = floats([0.0 if y == 1 else 1.0 for y in LABELS])
    with pytest.raises(ValueError, match=ERR_SINGLE_CLASS):
        weighted_auc(score_bins(floats(SCORES_A)), ints(LABELS), weights)


def _clusters() -> NDArray[np.int64]:
    return ints([index // 2 for index in range(len(LABELS))])


def test_bootstrap_reports_the_full_sample_aucs_and_a_unit_interval() -> None:
    resampling = build_resampling(ResamplingUnit.CLUSTER, len(LABELS), _clusters(), 1)
    boot = bootstrap_auc_difference(
        ints(LABELS), floats(SCORES_A), floats(SCORES_B), resampling, 60, 5, 0.9
    )
    delong = delong_auc_difference(ints(LABELS), floats(SCORES_A), floats(SCORES_B), 0.9)
    assert boot.auc_arm == pytest.approx(delong.auc_arm)
    assert boot.auc_reference == pytest.approx(delong.auc_reference)
    assert boot.difference["method"] is IntervalMethod.UNIT_BOOTSTRAP
    assert boot.difference["n_units"] == 7
    assert boot.difference["ci_low"] <= boot.difference["ci_high"]


def test_bootstrap_is_a_function_of_its_seed() -> None:
    resampling = build_resampling(ResamplingUnit.CLUSTER, len(LABELS), _clusters(), 1)
    first = bootstrap_auc_difference(
        ints(LABELS), floats(SCORES_A), floats(SCORES_B), resampling, 30, 9, 0.9
    )
    again = bootstrap_auc_difference(
        ints(LABELS), floats(SCORES_A), floats(SCORES_B), resampling, 30, 9, 0.9
    )
    assert first.difference == again.difference


def test_a_resample_of_one_class_is_refused_not_skipped() -> None:
    # Two clusters, one all positive and one all negative: any resample that
    # draws the same cluster twice holds one class only.
    labels = ints([1, 1, 0, 0])
    scores = floats([0.9, 0.8, 0.2, 0.1])
    resampling = build_resampling(ResamplingUnit.CLUSTER, 4, ints([0, 0, 1, 1]), 1)
    with pytest.raises(ValueError, match=ERR_DEGENERATE_RESAMPLE):
        bootstrap_auc_difference(labels, scores, scores, resampling, 40, 0, 0.9)


def test_bootstrap_refuses_misaligned_scores() -> None:
    resampling = build_resampling(ResamplingUnit.CLUSTER, len(LABELS), _clusters(), 1)
    with pytest.raises(ValueError, match="CVML-METRIC-001"):
        bootstrap_auc_difference(
            ints(LABELS), floats(SCORES_A), floats(SCORES_B[:-1]), resampling, 10, 0, 0.9
        )
