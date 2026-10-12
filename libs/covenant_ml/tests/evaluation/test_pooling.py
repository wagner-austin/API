"""Tests for pooling within-seed comparisons, and the observations built from them."""

from __future__ import annotations

import math

import pytest

from covenant_ml.evaluation.intervals import pool_across_seeds
from covenant_ml.evaluation.pooling import held_out_ratio, pool_seed_comparisons
from covenant_ml.evaluation.provenance import policy_observations, pooled_observations
from covenant_ml.evaluation.types import (
    ERR_TOO_FEW_SEEDS,
    ComparisonMetric,
    IntervalEstimate,
    IntervalMethod,
    PairedComparison,
    PolicyOutcome,
    SeedComparisons,
)


def _interval(estimate: float, std_error: float) -> IntervalEstimate:
    return {
        "method": IntervalMethod.PAIRED_T,
        "estimate": estimate,
        "std_error": std_error,
        "ci_low": estimate - 2 * std_error,
        "ci_high": estimate + 2 * std_error,
        "confidence": 0.95,
        "p_value": 0.5,
        "n_units": 10,
    }


def _comparison(arm: str, metric: ComparisonMetric, estimate: float) -> PairedComparison:
    return {
        "metric": metric,
        "arm": arm,
        "reference": "ref",
        "value_arm": 0.0,
        "value_reference": 0.0,
        "difference": _interval(estimate, 0.001),
        "n_rows": 40,
    }


def _policy(arm: str, value: float) -> PolicyOutcome:
    return {"arm": arm, "action_counts": [30, 10], "value": _interval(value, 0.1)}


def _seed(seed: int, n_train: int, n_test: int, shift: float) -> SeedComparisons:
    return {
        "seed": seed,
        "n_train": n_train,
        "n_test": n_test,
        "n_test_units": 10,
        "comparisons": [
            _comparison("a", ComparisonMetric.AUC, 0.02 + shift),
            _comparison("b", ComparisonMetric.BRIER, -0.01 + shift),
        ],
        "policies": [_policy("ref", 1.0 + shift), _policy("a", 2.0 + shift)],
    }


def test_held_out_ratio_averages_each_seeds_ratio() -> None:
    seeds = [_seed(1, 100, 20, 0.0), _seed(2, 100, 30, 0.0)]
    assert held_out_ratio(seeds) == pytest.approx(0.25)


def test_one_seed_has_nothing_to_pool() -> None:
    with pytest.raises(ValueError, match=ERR_TOO_FEW_SEEDS):
        pool_seed_comparisons([_seed(1, 100, 20, 0.0)], 0.95)


def test_each_arm_metric_pair_is_pooled_in_first_seen_order() -> None:
    seeds = [_seed(1, 100, 25, 0.0), _seed(2, 100, 25, 0.01), _seed(3, 100, 25, 0.02)]
    pooled = pool_seed_comparisons(seeds, 0.95)
    assert [(p["arm"], p["metric"]) for p in pooled] == [
        ("a", ComparisonMetric.AUC),
        ("b", ComparisonMetric.BRIER),
    ]
    per_seed: list[float] = [0.02, 0.03, 0.04]
    assert pooled[0]["per_seed"] == pytest.approx(per_seed)
    expected = pool_across_seeds([0.02, 0.03, 0.04], [0.001, 0.001, 0.001], 0.25, 0.95)
    assert pooled[0]["difference"] == expected
    assert pooled[0]["reference"] == "ref"


def test_pooled_observations_name_the_difference_its_bounds_and_its_p() -> None:
    seeds = [_seed(1, 100, 25, 0.0), _seed(2, 100, 25, 0.01)]
    pooled = pool_seed_comparisons(seeds, 0.95)
    observations = pooled_observations(pooled)
    names = [o["name"] for o in observations]
    assert names[:4] == [
        "pooled_a_vs_ref.auc.difference",
        "pooled_a_vs_ref.auc.ci_low",
        "pooled_a_vs_ref.auc.ci_high",
        "pooled_a_vs_ref.auc.p_value",
    ]
    assert len(observations) == 8
    assert observations[0]["value"] == pytest.approx(0.025)
    assert observations[3]["value"] == pooled[0]["difference"]["p_value"]


def test_policy_observations_average_each_policy_over_seeds() -> None:
    seeds = [_seed(1, 100, 25, 0.0), _seed(2, 100, 25, 0.5)]
    observations = policy_observations(seeds)
    assert [o["name"] for o in observations] == [
        "mean_policy_value.ref",
        "mean_policy_value.a",
    ]
    assert observations[0]["value"] == pytest.approx(1.25)
    assert math.isclose(observations[1]["value"], 2.25)


def test_no_policies_give_no_observations() -> None:
    entry = _seed(1, 100, 25, 0.0)
    entry["policies"] = []
    assert policy_observations([entry]) == ()
