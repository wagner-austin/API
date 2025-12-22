"""Tests for optimizer search space factory functions."""

from __future__ import annotations

from covenant_ml.optimizer.search_spaces import (
    make_default_optimization_config,
    make_lightgbm_default_space,
    make_lightgbm_focused_space,
    make_lstm_default_space,
    make_lstm_focused_space,
    make_mlp_default_space,
    make_mlp_focused_space,
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

    # Verify DART params are included
    assert "booster" in space
    assert "rate_drop" in space
    assert "skip_drop" in space


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

    # DART parameters
    assert space["booster"]["param_type"] == "categorical_str"
    assert space["rate_drop"]["param_type"] == "float"
    assert space["skip_drop"]["param_type"] == "float"


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


def test_make_xgboost_default_space_dart_params() -> None:
    """make_xgboost_default_space includes DART booster with correct ranges.

    DART (Dropouts meet Multiple Additive Regression Trees) applies dropout
    regularization during boosting. The booster param allows Optuna to explore
    both gbtree and dart configurations.
    """
    space = make_xgboost_default_space()

    # Booster choices
    booster = space["booster"]
    assert booster["param_type"] == "categorical_str"
    if booster["param_type"] == "categorical_str":
        assert booster["choices"] == ("gbtree", "dart")

    # rate_drop range (0.0-0.5)
    rate_drop = space["rate_drop"]
    assert rate_drop["param_type"] == "float"
    if rate_drop["param_type"] == "float":
        assert rate_drop["low"] == 0.0
        assert rate_drop["high"] == 0.5
        assert rate_drop["log_scale"] is False

    # skip_drop range (0.0-0.5)
    skip_drop = space["skip_drop"]
    assert skip_drop["param_type"] == "float"
    if skip_drop["param_type"] == "float":
        assert skip_drop["low"] == 0.0
        assert skip_drop["high"] == 0.5
        assert skip_drop["log_scale"] is False


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


# =============================================================================
# MLP Search Space Tests
# =============================================================================


def test_make_mlp_default_space_returns_complete_space() -> None:
    """make_mlp_default_space returns space with all required parameters."""
    space = make_mlp_default_space()

    assert "n_layers" in space
    assert "hidden_size" in space
    assert "learning_rate" in space
    assert "dropout" in space
    assert "batch_size" in space


def test_make_mlp_default_space_param_types() -> None:
    """make_mlp_default_space uses correct param types."""
    space = make_mlp_default_space()

    assert space["n_layers"]["param_type"] == "int"
    assert space["hidden_size"]["param_type"] == "categorical_int"
    assert space["learning_rate"]["param_type"] == "float"
    assert space["dropout"]["param_type"] == "float"
    assert space["batch_size"]["param_type"] == "categorical_int"


def test_make_mlp_default_space_ranges() -> None:
    """make_mlp_default_space has sensible default ranges."""
    space = make_mlp_default_space()

    n_layers = space["n_layers"]
    if n_layers["param_type"] == "int":
        assert n_layers["low"] == 1
        assert n_layers["high"] == 4

    hidden_size = space["hidden_size"]
    if hidden_size["param_type"] == "categorical_int":
        assert hidden_size["choices"] == (64, 128, 256, 512)

    lr = space["learning_rate"]
    if lr["param_type"] == "float":
        assert lr["log_scale"] is True
        assert lr["low"] == 1e-5
        assert lr["high"] == 1e-2

    dropout = space["dropout"]
    if dropout["param_type"] == "float":
        assert dropout["low"] == 0.0
        assert dropout["high"] == 0.5


def test_make_mlp_focused_space_narrows_around_best() -> None:
    """make_mlp_focused_space creates narrower ranges around best values."""
    space = make_mlp_focused_space(best_n_layers=2, best_hidden_size=128, best_learning_rate=1e-3)

    n_layers = space["n_layers"]
    if n_layers["param_type"] == "int":
        assert n_layers["low"] == 1  # max(1, 2-1)
        assert n_layers["high"] == 3  # min(5, 2+1)

    lr = space["learning_rate"]
    if lr["param_type"] == "float":
        assert lr["low"] == 1e-4  # 1e-3 * 0.1
        assert lr["high"] == 1e-2  # 1e-3 * 10.0


def test_make_mlp_focused_space_clamps_layers() -> None:
    """make_mlp_focused_space clamps layers to valid range."""
    space_low = make_mlp_focused_space(
        best_n_layers=1, best_hidden_size=64, best_learning_rate=1e-3
    )
    n_layers_low = space_low["n_layers"]
    if n_layers_low["param_type"] == "int":
        assert n_layers_low["low"] == 1  # max(1, 1-1) = 1

    space_high = make_mlp_focused_space(
        best_n_layers=5, best_hidden_size=64, best_learning_rate=1e-3
    )
    n_layers_high = space_high["n_layers"]
    if n_layers_high["param_type"] == "int":
        assert n_layers_high["high"] == 5  # min(5, 5+1) = 5


def test_make_mlp_focused_space_clamps_learning_rate() -> None:
    """make_mlp_focused_space clamps learning rate to valid range."""
    space_low = make_mlp_focused_space(
        best_n_layers=2, best_hidden_size=128, best_learning_rate=1e-7
    )
    lr_low = space_low["learning_rate"]
    if lr_low["param_type"] == "float":
        assert lr_low["low"] == 1e-6  # max(1e-6, 1e-7*0.1) = 1e-6

    space_high = make_mlp_focused_space(
        best_n_layers=2, best_hidden_size=128, best_learning_rate=0.5
    )
    lr_high = space_high["learning_rate"]
    if lr_high["param_type"] == "float":
        assert lr_high["high"] == 0.1  # min(0.1, 0.5*10) = 0.1


def test_make_mlp_focused_space_hidden_size_fallback() -> None:
    """make_mlp_focused_space falls back to best hidden size when no matches."""
    # Using a very small value (10) ensures no sizes pass the abs(size - best) <= best check
    space = make_mlp_focused_space(best_n_layers=2, best_hidden_size=10, best_learning_rate=1e-3)
    hidden = space["hidden_size"]
    if hidden["param_type"] == "categorical_int":
        assert hidden["choices"] == (10,)


# =============================================================================
# LSTM Search Space Tests
# =============================================================================


def test_make_lstm_default_space_returns_complete_space() -> None:
    """make_lstm_default_space returns space with all required parameters."""
    space = make_lstm_default_space()

    assert "hidden_size" in space
    assert "num_layers" in space
    assert "dropout" in space
    assert "learning_rate" in space
    assert "batch_size" in space


def test_make_lstm_default_space_param_types() -> None:
    """make_lstm_default_space uses correct param types."""
    space = make_lstm_default_space()

    assert space["hidden_size"]["param_type"] == "categorical_int"
    assert space["num_layers"]["param_type"] == "int"
    assert space["dropout"]["param_type"] == "float"
    assert space["learning_rate"]["param_type"] == "float"
    assert space["batch_size"]["param_type"] == "categorical_int"


def test_make_lstm_default_space_ranges() -> None:
    """make_lstm_default_space has sensible default ranges."""
    space = make_lstm_default_space()

    hidden = space["hidden_size"]
    if hidden["param_type"] == "categorical_int":
        assert hidden["choices"] == (64, 128, 256)

    num_layers = space["num_layers"]
    if num_layers["param_type"] == "int":
        assert num_layers["low"] == 1
        assert num_layers["high"] == 3

    lr = space["learning_rate"]
    if lr["param_type"] == "float":
        assert lr["log_scale"] is True
        assert lr["low"] == 1e-5
        assert lr["high"] == 1e-2

    batch = space["batch_size"]
    if batch["param_type"] == "categorical_int":
        assert batch["choices"] == (16, 32, 64)


def test_make_lstm_focused_space_narrows_around_best() -> None:
    """make_lstm_focused_space creates narrower ranges around best values."""
    space = make_lstm_focused_space(
        best_hidden_size=128, best_num_layers=2, best_learning_rate=1e-3
    )

    num_layers = space["num_layers"]
    if num_layers["param_type"] == "int":
        assert num_layers["low"] == 1  # max(1, 2-1)
        assert num_layers["high"] == 3  # min(4, 2+1)

    lr = space["learning_rate"]
    if lr["param_type"] == "float":
        assert lr["low"] == 1e-4  # 1e-3 * 0.1
        assert lr["high"] == 1e-2  # 1e-3 * 10.0


def test_make_lstm_focused_space_clamps_layers() -> None:
    """make_lstm_focused_space clamps num_layers to valid range."""
    space_low = make_lstm_focused_space(
        best_hidden_size=64, best_num_layers=1, best_learning_rate=1e-3
    )
    layers_low = space_low["num_layers"]
    if layers_low["param_type"] == "int":
        assert layers_low["low"] == 1  # max(1, 1-1) = 1

    space_high = make_lstm_focused_space(
        best_hidden_size=64, best_num_layers=4, best_learning_rate=1e-3
    )
    layers_high = space_high["num_layers"]
    if layers_high["param_type"] == "int":
        assert layers_high["high"] == 4  # min(4, 4+1) = 4


def test_make_lstm_focused_space_clamps_learning_rate() -> None:
    """make_lstm_focused_space clamps learning rate to valid range."""
    space_low = make_lstm_focused_space(
        best_hidden_size=64, best_num_layers=2, best_learning_rate=1e-7
    )
    lr_low = space_low["learning_rate"]
    if lr_low["param_type"] == "float":
        assert lr_low["low"] == 1e-6  # max(1e-6, 1e-7*0.1) = 1e-6

    space_high = make_lstm_focused_space(
        best_hidden_size=64, best_num_layers=2, best_learning_rate=0.5
    )
    lr_high = space_high["learning_rate"]
    if lr_high["param_type"] == "float":
        assert lr_high["high"] == 0.1  # min(0.1, 0.5*10) = 0.1


def test_make_lstm_focused_space_hidden_size_fallback() -> None:
    """make_lstm_focused_space falls back to best hidden size when no matches."""
    # Using a very small value (10) ensures no sizes pass the abs(size - best) <= best check
    space = make_lstm_focused_space(best_hidden_size=10, best_num_layers=2, best_learning_rate=1e-3)
    hidden = space["hidden_size"]
    if hidden["param_type"] == "categorical_int":
        assert hidden["choices"] == (10,)


# =============================================================================
# LightGBM Search Space Tests
# =============================================================================


def test_make_lightgbm_default_space_returns_complete_space() -> None:
    """make_lightgbm_default_space returns space with all required parameters.

    Note: max_depth is intentionally excluded. LightGBM uses leaf-wise growth
    where num_leaves is the primary complexity control. The optimizer uses
    max_depth=-1 (unlimited) to avoid constraint conflicts.
    """
    space = make_lightgbm_default_space()

    # max_depth is NOT in the search space - it's fixed at -1
    assert "max_depth" not in space
    assert "n_estimators" in space
    assert "num_leaves" in space
    assert "learning_rate" in space
    assert "subsample" in space
    assert "colsample_bytree" in space
    assert "reg_alpha" in space
    assert "reg_lambda" in space

    # Verify DART params are included
    assert "boosting_type" in space
    assert "drop_rate" in space
    assert "skip_drop" in space
    assert "feature_fraction" in space


def test_make_lightgbm_default_space_param_types() -> None:
    """make_lightgbm_default_space uses correct param types."""
    space = make_lightgbm_default_space()

    assert space["n_estimators"]["param_type"] == "int"
    assert space["num_leaves"]["param_type"] == "int"
    assert space["learning_rate"]["param_type"] == "float"
    assert space["subsample"]["param_type"] == "float"
    assert space["colsample_bytree"]["param_type"] == "float"
    assert space["reg_alpha"]["param_type"] == "float"
    assert space["reg_lambda"]["param_type"] == "float"

    # DART parameters
    assert space["boosting_type"]["param_type"] == "categorical_str"
    assert space["drop_rate"]["param_type"] == "float"
    assert space["skip_drop"]["param_type"] == "float"
    assert space["feature_fraction"]["param_type"] == "float"


def test_make_lightgbm_default_space_ranges() -> None:
    """make_lightgbm_default_space has sensible default ranges."""
    space = make_lightgbm_default_space()

    n_estimators = space["n_estimators"]
    if n_estimators["param_type"] == "int":
        assert n_estimators["low"] == 50
        assert n_estimators["high"] == 500

    num_leaves = space["num_leaves"]
    if num_leaves["param_type"] == "int":
        assert num_leaves["low"] == 20
        assert num_leaves["high"] == 100

    lr = space["learning_rate"]
    if lr["param_type"] == "float":
        assert lr["log_scale"] is True
        assert lr["low"] == 0.01
        assert lr["high"] == 0.3


def test_make_lightgbm_default_space_reg_alpha_allows_zero() -> None:
    """reg_alpha default space allows zero (no L1 regularization)."""
    space = make_lightgbm_default_space()

    reg_alpha = space["reg_alpha"]
    if reg_alpha["param_type"] == "float":
        assert reg_alpha["low"] == 0.0


def test_make_lightgbm_default_space_dart_params() -> None:
    """make_lightgbm_default_space includes DART boosting with correct ranges.

    DART (Dropouts meet Multiple Additive Regression Trees) applies dropout
    regularization during boosting. The boosting_type param allows Optuna to
    explore both gbdt and dart configurations.
    """
    space = make_lightgbm_default_space()

    # Boosting type choices
    boosting_type = space["boosting_type"]
    assert boosting_type["param_type"] == "categorical_str"
    if boosting_type["param_type"] == "categorical_str":
        assert boosting_type["choices"] == ("gbdt", "dart")

    # drop_rate range (0.0-0.5)
    drop_rate = space["drop_rate"]
    assert drop_rate["param_type"] == "float"
    if drop_rate["param_type"] == "float":
        assert drop_rate["low"] == 0.0
        assert drop_rate["high"] == 0.5
        assert drop_rate["log_scale"] is False

    # skip_drop range (0.0-0.5)
    skip_drop = space["skip_drop"]
    assert skip_drop["param_type"] == "float"
    if skip_drop["param_type"] == "float":
        assert skip_drop["low"] == 0.0
        assert skip_drop["high"] == 0.5
        assert skip_drop["log_scale"] is False

    # feature_fraction range (0.02-0.1) - Phase 6 DART-specific
    feature_fraction = space["feature_fraction"]
    assert feature_fraction["param_type"] == "float"
    if feature_fraction["param_type"] == "float":
        assert feature_fraction["low"] == 0.02
        assert feature_fraction["high"] == 0.1
        assert feature_fraction["log_scale"] is False


def test_make_lightgbm_default_space_reg_lambda_range() -> None:
    """make_lightgbm_default_space has higher reg_lambda range for DART regularization."""
    space = make_lightgbm_default_space()

    # reg_lambda range extended to 50.0 for Phase 6 DART regularization
    reg_lambda = space["reg_lambda"]
    assert reg_lambda["param_type"] == "float"
    if reg_lambda["param_type"] == "float":
        assert reg_lambda["low"] == 0.1
        assert reg_lambda["high"] == 50.0  # Extended from 10.0 for DART
        assert reg_lambda["log_scale"] is True


def test_make_lightgbm_focused_space_narrows_around_best() -> None:
    """make_lightgbm_focused_space creates narrower ranges around best values."""
    space = make_lightgbm_focused_space(best_num_leaves=50, best_learning_rate=0.1)

    num_leaves = space["num_leaves"]
    if num_leaves["param_type"] == "int":
        assert num_leaves["low"] == 30  # max(10, 50-20)
        assert num_leaves["high"] == 70  # min(150, 50+20)

    lr = space["learning_rate"]
    if lr["param_type"] == "float":
        assert lr["low"] == 0.05  # max(0.001, 0.1*0.5)
        assert lr["high"] == 0.2  # min(0.5, 0.1*2.0)


def test_make_lightgbm_focused_space_clamps_num_leaves() -> None:
    """make_lightgbm_focused_space clamps num_leaves to valid range."""
    space_low = make_lightgbm_focused_space(best_num_leaves=15, best_learning_rate=0.1)
    leaves_low = space_low["num_leaves"]
    if leaves_low["param_type"] == "int":
        assert leaves_low["low"] == 10  # max(10, 15-20) = 10

    space_high = make_lightgbm_focused_space(best_num_leaves=145, best_learning_rate=0.1)
    leaves_high = space_high["num_leaves"]
    if leaves_high["param_type"] == "int":
        assert leaves_high["high"] == 150  # min(150, 145+20) = 150


def test_make_lightgbm_focused_space_clamps_learning_rate() -> None:
    """make_lightgbm_focused_space clamps learning rate to valid range."""
    space_low = make_lightgbm_focused_space(best_num_leaves=50, best_learning_rate=0.001)
    lr_low = space_low["learning_rate"]
    if lr_low["param_type"] == "float":
        assert lr_low["low"] == 0.001  # max(0.001, 0.001*0.5) = 0.001

    space_high = make_lightgbm_focused_space(best_num_leaves=50, best_learning_rate=0.4)
    lr_high = space_high["learning_rate"]
    if lr_high["param_type"] == "float":
        assert lr_high["high"] == 0.5  # min(0.5, 0.4*2.0) = 0.5
