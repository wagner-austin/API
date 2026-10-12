"""Tests for the paired evaluation runner, driven by the real learners."""

from __future__ import annotations

import numpy as np
import pytest

from covenant_ml.benchmarking.factory import make_baseline_trainers
from covenant_ml.evaluation.runner import (
    NEVER_ALERT,
    evaluate_seed,
    require_arms,
    require_seeds,
    run_paired_evaluation,
)
from covenant_ml.evaluation.splits import CompanyDisjointGroupedSplits
from covenant_ml.evaluation.types import (
    ERR_REFERENCE_ARM,
    ERR_TOO_FEW_SEEDS,
    EVALUATION_SCHEMA_VERSION,
    ComparisonMetric,
    IntervalMethod,
    ResamplingUnit,
)

from ._fixtures import COSTS, DATASET, FINGERPRINT, grouped_dataset, settings, tiny_config


def test_every_arm_is_compared_with_the_reference_at_every_seed() -> None:
    features, labels, codes = grouped_dataset()
    manifest = run_paired_evaluation(
        make_baseline_trainers(tiny_config()),
        CompanyDisjointGroupedSplits(features, labels, codes),
        [1, 2],
        "lightgbm",
        tiny_config(),
        DATASET,
        settings(),
        COSTS,
        FINGERPRINT,
    )
    assert manifest["schema_version"] == EVALUATION_SCHEMA_VERSION
    assert manifest["action_names"] == ["hold", "alert"]
    assert manifest["reference"] == "lightgbm"
    assert {(r["arm"], r["seed"]) for r in manifest["results"]} == {
        (arm, seed) for arm in ("cleargbm", "lightgbm", "xgboost") for seed in (1, 2)
    }
    for entry in manifest["seed_comparisons"]:
        pairs = {(c["arm"], c["metric"]) for c in entry["comparisons"]}
        assert pairs == {(arm, m) for arm in ("cleargbm", "xgboost") for m in ComparisonMetric}
        assert [p["arm"] for p in entry["policies"]] == [
            NEVER_ALERT,
            "lightgbm",
            "cleargbm",
            "xgboost",
        ]
        assert entry["n_train"] + entry["n_test"] < len(labels)
        assert entry["n_test_units"] < entry["n_test"]
    assert len(manifest["pooled"]) == 8
    assert all(p["difference"]["method"] is IntervalMethod.ACROSS_SEEDS for p in manifest["pooled"])
    assert all(len(p["per_seed"]) == 2 for p in manifest["pooled"])
    assert manifest["fingerprint"] == FINGERPRINT


def test_independent_rows_get_delong_for_auc() -> None:
    features, labels, codes = grouped_dataset()
    trainers = make_baseline_trainers(tiny_config())
    grouped = CompanyDisjointGroupedSplits(features, labels, codes)(seed=3)
    results, entry = evaluate_seed(
        trainers, grouped, 3, "xgboost", settings(ResamplingUnit.ROW), COSTS
    )
    assert len(results) == 3
    auc = [c for c in entry["comparisons"] if c["metric"] is ComparisonMetric.AUC]
    assert all(c["difference"]["method"] is IntervalMethod.DELONG for c in auc)
    assert entry["n_test_units"] == entry["n_test"]


def test_the_never_alert_policy_pays_the_missed_cost_on_every_positive() -> None:
    features, labels, codes = grouped_dataset()
    grouped = CompanyDisjointGroupedSplits(features, labels, codes)(seed=3)
    _, entry = evaluate_seed(
        make_baseline_trainers(tiny_config()), grouped, 3, "lightgbm", settings(), COSTS
    )
    never = entry["policies"][0]
    positives = int(np.sum(grouped.split.y_test))
    assert never["action_counts"] == [entry["n_test"], 0]
    assert never["value"]["estimate"] == pytest.approx(-5.0 * positives / entry["n_test"])


def test_fewer_than_two_arms_is_refused() -> None:
    with pytest.raises(ValueError, match=ERR_REFERENCE_ARM):
        require_arms(["lightgbm"], "lightgbm")


def test_two_arms_sharing_a_name_is_refused() -> None:
    with pytest.raises(ValueError, match=ERR_REFERENCE_ARM):
        require_arms(["lightgbm", "lightgbm"], "lightgbm")


def test_a_reference_that_is_not_an_arm_is_refused() -> None:
    with pytest.raises(ValueError, match=ERR_REFERENCE_ARM):
        require_arms(["lightgbm", "xgboost"], "cleargbm")


def test_one_seed_cannot_be_pooled() -> None:
    with pytest.raises(ValueError, match=ERR_TOO_FEW_SEEDS):
        require_seeds([42])
