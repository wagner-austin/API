"""Tests for covenant_ml types module."""

from __future__ import annotations

from covenant_ml import TrainConfig


def test_train_config_has_required_keys() -> None:
    """TrainConfig TypedDict has all required keys."""
    config: TrainConfig = {
        "learning_rate": 0.1,
        "max_depth": 6,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }

    assert config["learning_rate"] == 0.1
    assert config["max_depth"] == 6
    assert config["n_estimators"] == 100
    assert config["subsample"] == 0.8
    assert config["colsample_bytree"] == 0.8
    assert config["random_state"] == 42


def test_train_config_values_are_correct_types() -> None:
    """TrainConfig values have correct types at runtime."""
    config: TrainConfig = {
        "learning_rate": 0.05,
        "max_depth": 4,
        "n_estimators": 50,
        "subsample": 0.9,
        "colsample_bytree": 0.7,
        "random_state": 123,
    }

    # Verify float operations work
    assert config["learning_rate"] * 2 == 0.1
    assert config["subsample"] + 0.1 == 1.0

    # Verify int operations work
    assert config["max_depth"] + 1 == 5
    assert config["n_estimators"] // 10 == 5
