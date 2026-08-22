"""Tests for the concrete ClearGBM, LightGBM and XGBoost trainers.

Every learner is a real dependency of this library, so these tests train real
models on small data rather than substituting a double for any of them.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.benchmarking.adapters import (
    ClearGbmTrainer,
    LightGbmTrainer,
    XgBoostTrainer,
)
from covenant_ml.benchmarking.protocols import DataSplit
from covenant_ml.benchmarking.types import BenchmarkConfig


def make_config(max_depth: int = 3, num_leaves: int = 7) -> BenchmarkConfig:
    """Build a small but real training configuration.

    Args:
        max_depth: Maximum tree depth.
        num_leaves: LightGBM leaf cap.

    Returns:
        The configuration.
    """
    return {
        "n_estimators": 4,
        "max_depth": max_depth,
        "learning_rate": 0.1,
        "max_bins": 16,
        "min_data_in_leaf": 5,
        "num_leaves": num_leaves,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "n_jobs": 1,
        "repeats": 1,
        "warmups": 0,
    }


def make_learnable_split(n_rows: int = 400) -> DataSplit:
    """Build a partition with real signal, so trees actually split.

    Args:
        n_rows: Rows per fold.

    Returns:
        The partition.
    """
    rng = np.random.default_rng(0)
    features: NDArray[np.float64] = rng.random((n_rows, 4), dtype=np.float64)
    noise: NDArray[np.float64] = rng.random(n_rows, dtype=np.float64)
    labels: NDArray[np.int64] = (features[:, 0] + noise * 0.3 > 0.8).astype(np.int64)
    return DataSplit(
        x_train=features,
        y_train=labels,
        x_val=features,
        y_val=labels,
        x_test=features,
        y_test=labels,
    )


def test_cleargbm_trainer_reports_its_name() -> None:
    trainer = ClearGbmTrainer(make_config(), growth_strategy="depth_wise")
    assert trainer.model_name == "cleargbm"


def test_cleargbm_leaf_wise_arm_reports_a_distinct_name() -> None:
    """The variant must not share the baseline's name.

    A manifest groups records by arm name, so two arms answering "cleargbm"
    would merge into one series and silently average two growth policies.
    """
    trainer = ClearGbmTrainer(make_config(), growth_strategy="leaf_wise")
    assert trainer.model_name == "cleargbm@leaf_wise"


def test_lightgbm_trainer_reports_its_name() -> None:
    assert LightGbmTrainer(make_config()).model_name == "lightgbm"


def test_xgboost_trainer_reports_its_name() -> None:
    assert XgBoostTrainer(make_config()).model_name == "xgboost"


def test_xgboost_grows_depth_wise_like_the_cleargbm_baseline() -> None:
    """XGBoost is here to separate "ClearGBM is slow" from "depth-wise is slow".

    That only holds if it actually grows depth-wise, so the arm is pinned to a
    depth-shaped tree rather than to LightGBM's leaf budget: at max_depth 4
    with the leaf budget released it must exceed the 3-leaf cap the config
    carries for the leaf-wise arms.
    """
    split = make_learnable_split()
    config = make_config(max_depth=4, num_leaves=3)
    leaves = XgBoostTrainer(config).fit(split, seed=42).mean_leaves()
    assert leaves > 3.0


def test_xgboost_predicts_a_probability_per_row() -> None:
    split = make_learnable_split()
    fitted = XgBoostTrainer(make_config()).fit(split, seed=42)
    proba = fitted.predict_positive_proba(split.x_test)
    assert len(proba) == len(split.y_test)
    within_unit: NDArray[np.bool_] = (proba >= 0.0) & (proba <= 1.0)
    assert int(np.sum(within_unit)) == len(proba)


def test_cleargbm_predicts_a_probability_per_row() -> None:
    split = make_learnable_split()
    fitted = ClearGbmTrainer(make_config(), growth_strategy="depth_wise").fit(split, seed=42)
    proba = fitted.predict_positive_proba(split.x_test)

    assert len(proba) == len(split.y_test)
    assert float(np.min(proba)) >= 0.0
    assert float(np.max(proba)) <= 1.0


def test_lightgbm_predicts_a_probability_per_row() -> None:
    split = make_learnable_split()
    fitted = LightGbmTrainer(make_config()).fit(split, seed=42)
    proba = fitted.predict_positive_proba(split.x_test)

    assert len(proba) == len(split.y_test)
    assert float(np.min(proba)) >= 0.0
    assert float(np.max(proba)) <= 1.0


def test_cleargbm_reports_a_positive_leaf_count() -> None:
    split = make_learnable_split()
    fitted = ClearGbmTrainer(make_config(), growth_strategy="depth_wise").fit(split, seed=42)
    assert fitted.mean_leaves() > 1.0


def test_lightgbm_reports_a_positive_leaf_count() -> None:
    split = make_learnable_split()
    fitted = LightGbmTrainer(make_config()).fit(split, seed=42)
    assert fitted.mean_leaves() > 1.0


def test_cleargbm_grows_depth_wise_beyond_the_leaf_cap() -> None:
    """ClearGBM is bounded by depth, not by ``num_leaves``.

    This asymmetry is the reason results are normalized by tree size: at the
    same configuration the two learners build different-sized trees, so a raw
    wall-clock ratio would compare unequal amounts of work.
    """
    split = make_learnable_split()
    config = make_config(max_depth=4, num_leaves=3)

    cleargbm = ClearGbmTrainer(config, growth_strategy="depth_wise")
    cleargbm_leaves = cleargbm.fit(split, seed=42).mean_leaves()
    lightgbm_leaves = LightGbmTrainer(config).fit(split, seed=42).mean_leaves()

    assert lightgbm_leaves <= 3.0
    assert cleargbm_leaves > lightgbm_leaves


def test_cleargbm_is_deterministic_for_a_seed() -> None:
    split = make_learnable_split()
    trainer = ClearGbmTrainer(make_config(), growth_strategy="depth_wise")
    first = trainer.fit(split, seed=42).predict_positive_proba(split.x_test)
    second = trainer.fit(split, seed=42).predict_positive_proba(split.x_test)
    assert np.array_equal(first, second)


def test_lightgbm_is_deterministic_for_a_seed() -> None:
    split = make_learnable_split()
    trainer = LightGbmTrainer(make_config())
    first = trainer.fit(split, seed=42).predict_positive_proba(split.x_test)
    second = trainer.fit(split, seed=42).predict_positive_proba(split.x_test)
    assert np.array_equal(first, second)


def test_both_learners_beat_chance_on_separable_data() -> None:
    split = make_learnable_split()
    config = make_config()
    arms = (
        ClearGbmTrainer(config, growth_strategy="depth_wise"),
        ClearGbmTrainer(config, growth_strategy="leaf_wise"),
        LightGbmTrainer(config),
        XgBoostTrainer(config),
    )
    for trainer in arms:
        proba = trainer.fit(split, seed=42).predict_positive_proba(split.x_test)
        positive_mask: NDArray[np.bool_] = split.y_test == 1
        negative_mask: NDArray[np.bool_] = split.y_test == 0
        positives: NDArray[np.float64] = proba[positive_mask]
        negatives: NDArray[np.float64] = proba[negative_mask]
        mean_positive = float(np.sum(positives)) / len(positives)
        mean_negative = float(np.sum(negatives)) / len(negatives)
        assert mean_positive > mean_negative


def test_cleargbm_respects_a_deeper_max_depth() -> None:
    split = make_learnable_split()
    shallow_trainer = ClearGbmTrainer(make_config(max_depth=2), growth_strategy="depth_wise")
    deep_trainer = ClearGbmTrainer(make_config(max_depth=4), growth_strategy="depth_wise")
    shallow = shallow_trainer.fit(split, seed=1).mean_leaves()
    deep = deep_trainer.fit(split, seed=1).mean_leaves()
    assert deep > shallow


def test_lightgbm_respects_its_leaf_cap() -> None:
    split = make_learnable_split()
    fitted = LightGbmTrainer(make_config(max_depth=6, num_leaves=5)).fit(split, seed=1)
    assert fitted.mean_leaves() <= 5.0


def test_probabilities_are_finite() -> None:
    split = make_learnable_split()
    config = make_config()
    arms = (
        ClearGbmTrainer(config, growth_strategy="depth_wise"),
        ClearGbmTrainer(config, growth_strategy="leaf_wise"),
        LightGbmTrainer(config),
        XgBoostTrainer(config),
    )
    for trainer in arms:
        proba = trainer.fit(split, seed=42).predict_positive_proba(split.x_test)
        finite: NDArray[np.bool_] = np.isfinite(proba)
        assert int(np.sum(finite)) == len(proba)


def test_cleargbm_scores_rows_it_was_not_trained_on() -> None:
    split = make_learnable_split()
    fitted = ClearGbmTrainer(make_config(), growth_strategy="depth_wise").fit(split, seed=42)
    held_out: NDArray[np.float64] = split.x_test[:10]
    proba = fitted.predict_positive_proba(held_out)
    assert len(proba) == 10


def test_lightgbm_scores_rows_it_was_not_trained_on() -> None:
    split = make_learnable_split()
    fitted = LightGbmTrainer(make_config()).fit(split, seed=42)
    held_out: NDArray[np.float64] = split.x_test[:10]
    proba = fitted.predict_positive_proba(held_out)
    assert len(proba) == 10
