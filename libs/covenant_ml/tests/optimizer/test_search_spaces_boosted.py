"""Tests for optimizer search space factory functions."""

from __future__ import annotations

from covenant_ml.optimizer.search_spaces import (
    make_cleargbm_default_space,
    make_cleargbm_focused_space,
    make_lightgbm_default_space,
    make_lightgbm_focused_space,
)


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


def test_make_cleargbm_default_space_returns_complete_space() -> None:
    """make_cleargbm_default_space returns space with all required parameters."""
    space = make_cleargbm_default_space()

    # Verify all required keys exist
    assert "n_estimators" in space
    assert "max_depth" in space
    assert "learning_rate" in space
    assert "min_samples_split" in space
    assert "min_samples_leaf" in space
    assert "max_bins" in space
    assert "subsample" in space
    assert "min_data_in_bin_denom" in space


def test_make_cleargbm_default_space_param_types() -> None:
    """make_cleargbm_default_space uses correct param types."""
    space = make_cleargbm_default_space()

    # Integer parameters
    assert space["n_estimators"]["param_type"] == "int"
    assert space["max_depth"]["param_type"] == "int"
    assert space["min_samples_split"]["param_type"] == "int"
    assert space["min_samples_leaf"]["param_type"] == "int"

    # Float parameters
    assert space["learning_rate"]["param_type"] == "float"
    assert space["subsample"]["param_type"] == "float"

    # Categorical int
    assert space["max_bins"]["param_type"] == "categorical_int"
    assert space["min_data_in_bin_denom"]["param_type"] == "categorical_int"


def test_cleargbm_coarseness_divisor_menu_includes_off() -> None:
    """Both spaces sample the divisor menu with 1 (no floor) present —
    the tuner must always be able to choose the unfloored baseline."""
    for space in (
        make_cleargbm_default_space(),
        make_cleargbm_focused_space(best_max_depth=5, best_learning_rate=0.1),
    ):
        spec = space["min_data_in_bin_denom"]
        assert spec["param_type"] == "categorical_int"
        if spec["param_type"] == "categorical_int":
            assert spec["choices"] == (1, 256, 64, 16, 4)


def test_make_cleargbm_default_space_ranges() -> None:
    """make_cleargbm_default_space has sensible default ranges."""
    space = make_cleargbm_default_space()

    # Check n_estimators range
    n_estimators = space["n_estimators"]
    assert n_estimators["param_type"] == "int"
    if n_estimators["param_type"] == "int":
        assert n_estimators["low"] == 50
        assert n_estimators["high"] == 300

    # Check max_depth range
    max_depth = space["max_depth"]
    assert max_depth["param_type"] == "int"
    if max_depth["param_type"] == "int":
        assert max_depth["low"] == 3
        assert max_depth["high"] == 10

    # Check learning_rate uses log scale
    lr = space["learning_rate"]
    assert lr["param_type"] == "float"
    if lr["param_type"] == "float":
        assert lr["log_scale"] is True
        assert lr["low"] == 0.01
        assert lr["high"] == 0.3


def test_make_cleargbm_default_space_min_samples_ranges() -> None:
    """make_cleargbm_default_space has correct min_samples ranges."""
    space = make_cleargbm_default_space()

    min_samples_split = space["min_samples_split"]
    if min_samples_split["param_type"] == "int":
        assert min_samples_split["low"] == 5
        assert min_samples_split["high"] == 50

    min_samples_leaf = space["min_samples_leaf"]
    if min_samples_leaf["param_type"] == "int":
        assert min_samples_leaf["low"] == 2
        assert min_samples_leaf["high"] == 20


def test_make_cleargbm_default_space_max_bins_choices() -> None:
    """make_cleargbm_default_space has correct max_bins choices."""
    space = make_cleargbm_default_space()

    max_bins = space["max_bins"]
    assert max_bins["param_type"] == "categorical_int"
    if max_bins["param_type"] == "categorical_int":
        assert max_bins["choices"] == (32, 64, 128)


def test_make_cleargbm_default_space_subsample_range() -> None:
    """make_cleargbm_default_space has correct subsample range."""
    space = make_cleargbm_default_space()

    subsample = space["subsample"]
    assert subsample["param_type"] == "float"
    if subsample["param_type"] == "float":
        assert subsample["low"] == 0.6
        assert subsample["high"] == 1.0
        assert subsample["log_scale"] is False


def test_make_cleargbm_focused_space_narrows_around_best() -> None:
    """make_cleargbm_focused_space creates narrower ranges around best values."""
    space = make_cleargbm_focused_space(best_max_depth=5, best_learning_rate=0.1)

    max_depth = space["max_depth"]
    if max_depth["param_type"] == "int":
        assert max_depth["low"] == 3  # max(2, 5-2)
        assert max_depth["high"] == 7  # min(15, 5+2)

    lr = space["learning_rate"]
    if lr["param_type"] == "float":
        assert lr["low"] == 0.05  # max(0.001, 0.1*0.5)
        assert lr["high"] == 0.2  # min(0.5, 0.1*2.0)


def test_make_cleargbm_focused_space_clamps_max_depth() -> None:
    """make_cleargbm_focused_space clamps max_depth to valid range."""
    space_low = make_cleargbm_focused_space(best_max_depth=2, best_learning_rate=0.1)
    depth_low = space_low["max_depth"]
    if depth_low["param_type"] == "int":
        assert depth_low["low"] == 2  # max(2, 2-2) = 2

    space_high = make_cleargbm_focused_space(best_max_depth=14, best_learning_rate=0.1)
    depth_high = space_high["max_depth"]
    if depth_high["param_type"] == "int":
        assert depth_high["high"] == 15  # min(15, 14+2) = 15


def test_make_cleargbm_focused_space_clamps_learning_rate() -> None:
    """make_cleargbm_focused_space clamps learning rate to valid range."""
    space_low = make_cleargbm_focused_space(best_max_depth=5, best_learning_rate=0.001)
    lr_low = space_low["learning_rate"]
    if lr_low["param_type"] == "float":
        assert lr_low["low"] == 0.001  # max(0.001, 0.001*0.5) = 0.001

    space_high = make_cleargbm_focused_space(best_max_depth=5, best_learning_rate=0.4)
    lr_high = space_high["learning_rate"]
    if lr_high["param_type"] == "float":
        assert lr_high["high"] == 0.5  # min(0.5, 0.4*2.0) = 0.5


def test_make_cleargbm_focused_space_narrower_ranges() -> None:
    """make_cleargbm_focused_space has narrower secondary parameter ranges."""
    space = make_cleargbm_focused_space(best_max_depth=5, best_learning_rate=0.1)

    # n_estimators narrower (75-200 vs 50-300)
    n_estimators = space["n_estimators"]
    if n_estimators["param_type"] == "int":
        assert n_estimators["low"] == 75
        assert n_estimators["high"] == 200

    # min_samples_split narrower (5-30 vs 5-50)
    min_samples_split = space["min_samples_split"]
    if min_samples_split["param_type"] == "int":
        assert min_samples_split["low"] == 5
        assert min_samples_split["high"] == 30

    # min_samples_leaf narrower (2-15 vs 2-20)
    min_samples_leaf = space["min_samples_leaf"]
    if min_samples_leaf["param_type"] == "int":
        assert min_samples_leaf["low"] == 2
        assert min_samples_leaf["high"] == 15

    # max_bins reduced choices (64,128 vs 32,64,128)
    max_bins = space["max_bins"]
    if max_bins["param_type"] == "categorical_int":
        assert max_bins["choices"] == (64, 128)

    # subsample narrower (0.7-1.0 vs 0.6-1.0)
    subsample = space["subsample"]
    if subsample["param_type"] == "float":
        assert subsample["low"] == 0.7
        assert subsample["high"] == 1.0
