"""Tests for optimizer search space factory functions."""

from __future__ import annotations

from covenant_ml.optimizer.search_spaces import (
    make_default_optimization_config,
    make_xgboost_categorical_space,
    make_xgboost_default_space,
    make_xgboost_focused_space,
)


def test_make_xgboost_default_space_returns_complete_space() -> None:
    """make_xgboost_default_space returns space with all required parameters."""
    space = make_xgboost_default_space()

    # Verify all required keys exist
    assert "max_depth" in space
    assert "n_estimators" in space
    assert "learning_rate" in space
    assert "reg_alpha" in space
    assert "reg_lambda" in space
    assert "subsample" in space
    assert "colsample_bytree" in space


def test_make_xgboost_default_space_param_types() -> None:
    """make_xgboost_default_space uses correct param types."""
    space = make_xgboost_default_space()

    # Integer parameters
    assert space["max_depth"]["param_type"] == "int"
    assert space["n_estimators"]["param_type"] == "int"

    # Float parameters
    assert space["learning_rate"]["param_type"] == "float"
    assert space["reg_alpha"]["param_type"] == "float"
    assert space["reg_lambda"]["param_type"] == "float"
    assert space["subsample"]["param_type"] == "float"
    assert space["colsample_bytree"]["param_type"] == "float"


def test_make_xgboost_default_space_ranges() -> None:
    """make_xgboost_default_space has sensible default ranges."""
    space = make_xgboost_default_space()

    # Check max_depth range
    max_depth = space["max_depth"]
    assert max_depth["param_type"] == "int"
    if max_depth["param_type"] == "int":
        assert max_depth["low"] == 3
        assert max_depth["high"] == 10

    # Check n_estimators range
    n_estimators = space["n_estimators"]
    assert n_estimators["param_type"] == "int"
    if n_estimators["param_type"] == "int":
        assert n_estimators["low"] == 50
        assert n_estimators["high"] == 300

    # Check learning_rate uses log scale
    lr = space["learning_rate"]
    assert lr["param_type"] == "float"
    if lr["param_type"] == "float":
        assert lr["log_scale"] is True
        assert lr["low"] == 0.01
        assert lr["high"] == 0.3


def test_make_xgboost_default_space_reg_alpha_allows_zero() -> None:
    """reg_alpha default space allows zero (no L1 regularization)."""
    space = make_xgboost_default_space()

    reg_alpha = space["reg_alpha"]
    assert reg_alpha["param_type"] == "float"
    if reg_alpha["param_type"] == "float":
        assert reg_alpha["low"] == 0.0


def test_make_xgboost_focused_space_narrows_around_best() -> None:
    """make_xgboost_focused_space creates narrower ranges around best values."""
    space = make_xgboost_focused_space(best_max_depth=6, best_learning_rate=0.1)

    # Check max_depth is narrowed around 6
    max_depth = space["max_depth"]
    assert max_depth["param_type"] == "int"
    if max_depth["param_type"] == "int":
        # Should be ~4 to ~8 (±2 from best)
        assert max_depth["low"] == 4
        assert max_depth["high"] == 8

    # Check learning_rate is narrowed (0.5x to 2x)
    lr = space["learning_rate"]
    assert lr["param_type"] == "float"
    if lr["param_type"] == "float":
        assert lr["low"] == 0.05  # 0.1 * 0.5
        assert lr["high"] == 0.2  # 0.1 * 2.0


def test_make_xgboost_focused_space_clamps_depth() -> None:
    """make_xgboost_focused_space clamps depth to valid range."""
    # Test clamping at low end
    space_low = make_xgboost_focused_space(best_max_depth=2, best_learning_rate=0.1)
    max_depth_low = space_low["max_depth"]
    if max_depth_low["param_type"] == "int":
        assert max_depth_low["low"] == 2  # max(2, 2-2) = 2

    # Test clamping at high end
    space_high = make_xgboost_focused_space(best_max_depth=14, best_learning_rate=0.1)
    max_depth_high = space_high["max_depth"]
    if max_depth_high["param_type"] == "int":
        assert max_depth_high["high"] == 15  # min(15, 14+2) = 15


def test_make_xgboost_focused_space_clamps_learning_rate() -> None:
    """make_xgboost_focused_space clamps learning rate to valid range."""
    # Test clamping at low end
    space_low = make_xgboost_focused_space(best_max_depth=5, best_learning_rate=0.001)
    lr_low = space_low["learning_rate"]
    if lr_low["param_type"] == "float":
        assert lr_low["low"] == 0.001  # max(0.001, 0.001*0.5) = 0.001

    # Test clamping at high end
    space_high = make_xgboost_focused_space(best_max_depth=5, best_learning_rate=0.4)
    lr_high = space_high["learning_rate"]
    if lr_high["param_type"] == "float":
        assert lr_high["high"] == 0.5  # min(0.5, 0.4*2.0) = 0.5


def test_make_xgboost_categorical_space_uses_discrete_values() -> None:
    """make_xgboost_categorical_space uses categorical specs."""
    space = make_xgboost_categorical_space()

    # Check all parameters use categorical types
    assert space["max_depth"]["param_type"] == "categorical_int"
    assert space["n_estimators"]["param_type"] == "categorical_int"
    assert space["learning_rate"]["param_type"] == "categorical_float"
    assert space["reg_alpha"]["param_type"] == "categorical_float"
    assert space["reg_lambda"]["param_type"] == "categorical_float"
    assert space["subsample"]["param_type"] == "categorical_float"
    assert space["colsample_bytree"]["param_type"] == "categorical_float"


def test_make_xgboost_categorical_space_choices() -> None:
    """make_xgboost_categorical_space has expected choices."""
    space = make_xgboost_categorical_space()

    # Check max_depth choices
    max_depth = space["max_depth"]
    if max_depth["param_type"] == "categorical_int":
        assert max_depth["choices"] == (3, 4, 5, 6, 7, 8)

    # Check learning_rate choices
    lr = space["learning_rate"]
    if lr["param_type"] == "categorical_float":
        assert lr["choices"] == (0.01, 0.05, 0.1, 0.2, 0.3)


def test_make_default_optimization_config_defaults() -> None:
    """make_default_optimization_config has sensible defaults."""
    config = make_default_optimization_config()

    assert config["n_trials"] == 100
    assert config["timeout_seconds"] is None
    assert config["n_startup_trials"] == 10
    assert config["random_state"] == 42
    assert config["direction"] == "maximize"
    assert config["pruning_enabled"] is True
    assert config["train_ratio"] == 0.7
    assert config["val_ratio"] == 0.15
    assert config["test_ratio"] == 0.15


def test_make_default_optimization_config_custom_trials() -> None:
    """make_default_optimization_config accepts custom n_trials."""
    config = make_default_optimization_config(n_trials=50)

    assert config["n_trials"] == 50
    assert config["timeout_seconds"] is None


def test_make_default_optimization_config_custom_timeout() -> None:
    """make_default_optimization_config accepts custom timeout."""
    config = make_default_optimization_config(timeout_seconds=600)

    assert config["timeout_seconds"] == 600


def test_make_default_optimization_config_custom_seed() -> None:
    """make_default_optimization_config accepts custom random_state."""
    config = make_default_optimization_config(random_state=123)

    assert config["random_state"] == 123


def test_make_default_optimization_config_all_custom() -> None:
    """make_default_optimization_config accepts all custom parameters."""
    config = make_default_optimization_config(
        n_trials=25,
        timeout_seconds=300,
        random_state=999,
    )

    assert config["n_trials"] == 25
    assert config["timeout_seconds"] == 300
    assert config["random_state"] == 999
    # Other defaults preserved
    assert config["n_startup_trials"] == 10
    assert config["direction"] == "maximize"
