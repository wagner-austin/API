"""Tests for covenant_ml.testing module utilities.

Tests the public testing utility functions.
"""

from __future__ import annotations

from covenant_ml.testing import (
    make_logreg_config,
    make_random_forest_config,
    make_train_config,
)


def test_make_train_config_defaults() -> None:
    """make_train_config creates TrainConfig with defaults."""
    config = make_train_config()

    assert config["device"] == "cpu"
    assert config["learning_rate"] == 0.1
    assert config["max_depth"] == 3
    assert config["n_estimators"] == 10
    assert config["random_state"] == 42


def test_make_train_config_custom_values() -> None:
    """make_train_config accepts custom values."""
    config = make_train_config(
        device="cuda",
        learning_rate=0.05,
        max_depth=5,
        n_estimators=20,
        random_state=123,
    )

    assert config["device"] == "cuda"
    assert config["learning_rate"] == 0.05
    assert config["max_depth"] == 5
    assert config["n_estimators"] == 20
    assert config["random_state"] == 123


def test_make_train_config_scale_pos_weight() -> None:
    """make_train_config includes scale_pos_weight when provided."""
    config = make_train_config(scale_pos_weight=2.5)

    assert config["scale_pos_weight"] == 2.5


def test_make_logreg_config_defaults() -> None:
    """make_logreg_config creates LogRegConfig with defaults."""
    config = make_logreg_config()

    assert config["solver"] == "lbfgs"
    assert config["penalty"] == "l2"
    assert config["C"] == 1.0
    assert config["max_iter"] == 100
    assert config["class_weight_balanced"] is True
    assert config["random_state"] == 42


def test_make_logreg_config_custom_values() -> None:
    """make_logreg_config accepts custom values."""
    config = make_logreg_config(
        solver="saga",
        penalty="l1",
        inverse_reg_strength=0.5,
        max_iter=500,
        class_weight_balanced=False,
        random_state=99,
    )

    assert config["solver"] == "saga"
    assert config["penalty"] == "l1"
    assert config["C"] == 0.5
    assert config["max_iter"] == 500
    assert config["class_weight_balanced"] is False
    assert config["random_state"] == 99


def test_make_logreg_config_elasticnet() -> None:
    """make_logreg_config works with elasticnet penalty."""
    config = make_logreg_config(
        solver="saga",
        penalty="elasticnet",
        l1_ratio=0.7,
    )

    assert config["penalty"] == "elasticnet"
    assert config["l1_ratio"] == 0.7


def test_make_random_forest_config_defaults() -> None:
    """make_random_forest_config creates RandomForestConfig with defaults."""
    config = make_random_forest_config()

    assert config["n_estimators"] == 10
    assert config["max_depth"] == 5
    assert config["min_samples_split"] == 2
    assert config["min_samples_leaf"] == 1
    assert config["max_features"] == "sqrt"
    assert config["bootstrap"] is True
    assert config["class_weight_balanced"] is True
    assert config["random_state"] == 42


def test_make_random_forest_config_custom_values() -> None:
    """make_random_forest_config accepts custom values."""
    config = make_random_forest_config(
        n_estimators=50,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="log2",
        bootstrap=False,
        class_weight_balanced=False,
        random_state=123,
        oob_score=True,
    )

    assert config["n_estimators"] == 50
    assert config["max_depth"] == 10
    assert config["min_samples_split"] == 5
    assert config["min_samples_leaf"] == 2
    assert config["max_features"] == "log2"
    assert config["bootstrap"] is False
    assert config["class_weight_balanced"] is False
    assert config["random_state"] == 123
    assert config["oob_score"] is True


def test_make_random_forest_config_no_max_depth() -> None:
    """make_random_forest_config works with None max_depth."""
    config = make_random_forest_config(max_depth=None)

    assert config["max_depth"] is None
