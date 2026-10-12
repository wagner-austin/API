"""Tests for feature ablation, driven by a real learner."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.benchmarking.adapters import LightGbmTrainer
from covenant_ml.evaluation.ablation import (
    FULL_ARM,
    ablated_arm,
    drop_columns,
    keep_mask,
    run_feature_ablation,
)
from covenant_ml.evaluation.splits import CompanyDisjointGroupedSplits
from covenant_ml.evaluation.types import (
    ERR_EMPTY_ABLATION,
    ERR_INVALID_FRACTION,
    ERR_TOO_FEW_SEEDS,
    ERR_UNKNOWN_FEATURE,
    EVALUATION_SCHEMA_VERSION,
    AblationGroup,
    ComparisonMetric,
)

from ._fixtures import (
    DATASET,
    FEATURE_NAMES,
    FINGERPRINT,
    grouped_dataset,
    settings,
    tiny_config,
)

SIGNAL: AblationGroup = {"name": "signal", "features": ["signal"]}
NOISE: AblationGroup = {"name": "noise", "features": ["noise_a", "noise_b"]}


def test_keep_mask_drops_exactly_the_named_features() -> None:
    mask = keep_mask(FEATURE_NAMES, NOISE)
    kept: list[bool] = [True, False, False]
    expected: NDArray[np.bool_] = np.asarray(kept, dtype=np.bool_)
    assert np.array_equal(mask, expected)


def test_an_unknown_feature_is_refused_by_name() -> None:
    group: AblationGroup = {"name": "typo", "features": ["signall"]}
    with pytest.raises(ValueError, match=ERR_UNKNOWN_FEATURE):
        keep_mask(FEATURE_NAMES, group)


def test_a_group_that_removes_nothing_is_refused() -> None:
    group: AblationGroup = {"name": "empty", "features": []}
    with pytest.raises(ValueError, match=ERR_EMPTY_ABLATION):
        keep_mask(FEATURE_NAMES, group)


def test_a_group_that_removes_everything_is_refused() -> None:
    group: AblationGroup = {"name": "all", "features": list(FEATURE_NAMES)}
    with pytest.raises(ValueError, match=ERR_EMPTY_ABLATION):
        keep_mask(FEATURE_NAMES, group)


def test_drop_columns_keeps_rows_and_labels() -> None:
    features, labels, codes = grouped_dataset()
    split = CompanyDisjointGroupedSplits(features, labels, codes)(seed=1).split
    reduced = drop_columns(split, keep_mask(FEATURE_NAMES, SIGNAL))
    assert reduced.x_train.shape == (len(split.y_train), 2)
    assert reduced.x_test.shape == (len(split.y_test), 2)
    assert np.array_equal(reduced.y_val, split.y_val)
    noise_columns: NDArray[np.float64] = split.x_test[:, 1:]
    assert np.array_equal(reduced.x_test, noise_columns)
    assert np.array_equal(reduced.y_train, split.y_train)


def test_ablated_arms_are_named_for_their_group() -> None:
    assert ablated_arm(NOISE) == "without:noise"


def test_removing_the_signal_costs_auc_and_removing_noise_does_not() -> None:
    features, labels, codes = grouped_dataset()
    manifest = run_feature_ablation(
        LightGbmTrainer(tiny_config()),
        CompanyDisjointGroupedSplits(features, labels, codes),
        [1, 2],
        FEATURE_NAMES,
        [SIGNAL, NOISE],
        [0.5, 1.0],
        tiny_config(),
        DATASET,
        settings(),
        FINGERPRINT,
    )
    assert manifest["schema_version"] == EVALUATION_SCHEMA_VERSION
    assert manifest["model"] == "lightgbm"
    assert {r["arm"] for r in manifest["results"]} == {FULL_ARM, "without:signal", "without:noise"}
    pooled_auc = {
        p["arm"]: p["difference"]["estimate"]
        for p in manifest["pooled"]
        if p["metric"] is ComparisonMetric.AUC
    }
    assert pooled_auc["without:signal"] < -0.1
    assert abs(pooled_auc["without:noise"]) < 0.05
    assert all(entry["policies"] == [] for entry in manifest["seed_comparisons"])
    fractions = [(p["seed"], p["train_fraction"]) for p in manifest["learning_curve"]]
    assert fractions == [(1, 0.5), (1, 1.0), (2, 0.5), (2, 1.0)]


def test_groups_must_be_present_and_distinct() -> None:
    features, labels, codes = grouped_dataset()
    factory = CompanyDisjointGroupedSplits(features, labels, codes)
    for groups in ([], [SIGNAL, SIGNAL]):
        with pytest.raises(ValueError, match=ERR_EMPTY_ABLATION):
            run_feature_ablation(
                LightGbmTrainer(tiny_config()),
                factory,
                [1, 2],
                FEATURE_NAMES,
                groups,
                [1.0],
                tiny_config(),
                DATASET,
                settings(),
                FINGERPRINT,
            )


def test_seeds_and_fractions_are_checked_before_any_fit() -> None:
    features, labels, codes = grouped_dataset()
    factory = CompanyDisjointGroupedSplits(features, labels, codes)
    trainer = LightGbmTrainer(tiny_config())
    with pytest.raises(ValueError, match=ERR_TOO_FEW_SEEDS):
        run_feature_ablation(
            trainer,
            factory,
            [1],
            FEATURE_NAMES,
            [SIGNAL],
            [1.0],
            tiny_config(),
            DATASET,
            settings(),
            FINGERPRINT,
        )
    with pytest.raises(ValueError, match=ERR_INVALID_FRACTION):
        run_feature_ablation(
            trainer,
            factory,
            [1, 2],
            FEATURE_NAMES,
            [SIGNAL],
            [1.5],
            tiny_config(),
            DATASET,
            settings(),
            FINGERPRINT,
        )
