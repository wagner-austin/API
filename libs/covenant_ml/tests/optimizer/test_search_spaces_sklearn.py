"""Tests for optimizer search space factory functions."""

from __future__ import annotations

from covenant_ml.optimizer.search_spaces import (
    make_logreg_default_space,
    make_logreg_focused_space,
    make_random_forest_default_space,
    make_random_forest_focused_space,
)


def test_make_logreg_default_space_returns_complete_space() -> None:
    """make_logreg_default_space returns space with all required parameters."""
    space = make_logreg_default_space()

    assert "C" in space
    assert "max_iter" in space
    assert "tol" in space
    assert "penalty" in space
    assert "solver" in space
    assert "l1_ratio" in space


def test_make_logreg_default_space_param_types() -> None:
    """make_logreg_default_space uses correct param types."""
    space = make_logreg_default_space()

    assert space["C"]["param_type"] == "float"
    assert space["max_iter"]["param_type"] == "int"
    assert space["tol"]["param_type"] == "float"
    assert space["penalty"]["param_type"] == "categorical_str"
    assert space["solver"]["param_type"] == "categorical_str"
    assert space["l1_ratio"]["param_type"] == "float"


def test_make_logreg_default_space_ranges() -> None:
    """make_logreg_default_space has sensible default ranges."""
    space = make_logreg_default_space()

    c_spec = space["C"]
    if c_spec["param_type"] == "float":
        assert c_spec["low"] == 1e-4
        assert c_spec["high"] == 1e4
        assert c_spec["log_scale"] is True

    max_iter = space["max_iter"]
    if max_iter["param_type"] == "int":
        assert max_iter["low"] == 100
        assert max_iter["high"] == 1000

    tol_spec = space["tol"]
    if tol_spec["param_type"] == "float":
        assert tol_spec["low"] == 1e-6
        assert tol_spec["high"] == 1e-3
        assert tol_spec["log_scale"] is True


def test_make_logreg_default_space_penalty_choices() -> None:
    """make_logreg_default_space includes l2 and l1 penalties."""
    space = make_logreg_default_space()
    penalty = space["penalty"]
    assert penalty["param_type"] == "categorical_str"
    if penalty["param_type"] == "categorical_str":
        assert "l2" in penalty["choices"]
        assert "l1" in penalty["choices"]


def test_make_logreg_default_space_solver_choices() -> None:
    """make_logreg_default_space uses saga solver."""
    space = make_logreg_default_space()
    solver = space["solver"]
    assert solver["param_type"] == "categorical_str"
    if solver["param_type"] == "categorical_str":
        assert "saga" in solver["choices"]


def test_make_logreg_focused_space_narrows_around_best() -> None:
    """make_logreg_focused_space narrows C and tol around best values."""
    space = make_logreg_focused_space(best_c=1.0, best_tol=1e-4)

    c_spec = space["C"]
    if c_spec["param_type"] == "float":
        assert c_spec["low"] < 1.0
        assert c_spec["high"] > 1.0

    tol_spec = space["tol"]
    if tol_spec["param_type"] == "float":
        assert tol_spec["low"] < 1e-4
        assert tol_spec["high"] > 1e-4


def test_make_logreg_focused_space_clamps_c() -> None:
    """make_logreg_focused_space clamps C to valid range."""
    space = make_logreg_focused_space(best_c=1e-5, best_tol=1e-4)
    c_spec = space["C"]
    if c_spec["param_type"] == "float":
        assert c_spec["low"] >= 1e-6

    space_high = make_logreg_focused_space(best_c=1e5, best_tol=1e-4)
    c_spec_high = space_high["C"]
    if c_spec_high["param_type"] == "float":
        assert c_spec_high["high"] <= 1e6


def test_make_logreg_focused_space_clamps_tol() -> None:
    """make_logreg_focused_space clamps tol to valid range."""
    space = make_logreg_focused_space(best_c=1.0, best_tol=1e-7)
    tol_spec = space["tol"]
    if tol_spec["param_type"] == "float":
        assert tol_spec["low"] >= 1e-8

    space_high = make_logreg_focused_space(best_c=1.0, best_tol=1e-2)
    tol_spec_high = space_high["tol"]
    if tol_spec_high["param_type"] == "float":
        assert tol_spec_high["high"] <= 1e-1  # min(1e-2, best*10)


def test_make_logreg_focused_space_max_iter_narrowed() -> None:
    """make_logreg_focused_space has narrower max_iter range."""
    space = make_logreg_focused_space(best_c=1.0, best_tol=1e-4)
    max_iter = space["max_iter"]
    if max_iter["param_type"] == "int":
        assert max_iter["low"] == 200
        assert max_iter["high"] == 500


def test_make_random_forest_default_space_returns_complete_space() -> None:
    """make_random_forest_default_space returns space with all required parameters."""
    space = make_random_forest_default_space()

    assert "n_estimators" in space
    assert "max_depth" in space
    assert "min_samples_split" in space
    assert "min_samples_leaf" in space
    assert "max_features" in space


def test_make_random_forest_default_space_param_types() -> None:
    """make_random_forest_default_space uses correct param types."""
    space = make_random_forest_default_space()

    assert space["n_estimators"]["param_type"] == "int"
    assert space["max_depth"]["param_type"] == "int"
    assert space["min_samples_split"]["param_type"] == "int"
    assert space["min_samples_leaf"]["param_type"] == "int"
    assert space["max_features"]["param_type"] == "categorical_str"


def test_make_random_forest_default_space_ranges() -> None:
    """make_random_forest_default_space has sensible default ranges."""
    space = make_random_forest_default_space()

    n_est = space["n_estimators"]
    if n_est["param_type"] == "int":
        assert n_est["low"] == 50
        assert n_est["high"] == 500

    max_depth = space["max_depth"]
    if max_depth["param_type"] == "int":
        assert max_depth["low"] == 3
        assert max_depth["high"] == 20


def test_make_random_forest_default_space_max_features_choices() -> None:
    """make_random_forest_default_space includes sqrt and log2."""
    space = make_random_forest_default_space()
    mf = space["max_features"]
    assert mf["param_type"] == "categorical_str"
    if mf["param_type"] == "categorical_str":
        assert "sqrt" in mf["choices"]
        assert "log2" in mf["choices"]


def test_make_random_forest_default_space_min_samples_ranges() -> None:
    """make_random_forest_default_space has valid min_samples ranges."""
    space = make_random_forest_default_space()

    split = space["min_samples_split"]
    if split["param_type"] == "int":
        assert split["low"] == 2
        assert split["high"] == 20

    leaf = space["min_samples_leaf"]
    if leaf["param_type"] == "int":
        assert leaf["low"] == 1
        assert leaf["high"] == 10


def test_make_random_forest_focused_space_narrows_around_best() -> None:
    """make_random_forest_focused_space narrows around best values."""
    space = make_random_forest_focused_space(best_max_depth=10, best_n_estimators=200)

    max_depth = space["max_depth"]
    if max_depth["param_type"] == "int":
        assert max_depth["low"] < 10
        assert max_depth["high"] > 10

    n_est = space["n_estimators"]
    if n_est["param_type"] == "int":
        assert n_est["low"] < 200
        assert n_est["high"] > 200


def test_make_random_forest_focused_space_clamps_depth() -> None:
    """make_random_forest_focused_space clamps max_depth to valid range."""
    space = make_random_forest_focused_space(best_max_depth=2, best_n_estimators=200)
    max_depth = space["max_depth"]
    if max_depth["param_type"] == "int":
        assert max_depth["low"] >= 2

    space_high = make_random_forest_focused_space(best_max_depth=24, best_n_estimators=200)
    max_depth_high = space_high["max_depth"]
    if max_depth_high["param_type"] == "int":
        assert max_depth_high["high"] <= 25


def test_make_random_forest_focused_space_clamps_n_estimators() -> None:
    """make_random_forest_focused_space clamps n_estimators to valid range."""
    space = make_random_forest_focused_space(best_max_depth=5, best_n_estimators=60)
    n_est = space["n_estimators"]
    if n_est["param_type"] == "int":
        assert n_est["low"] >= 50

    space_high = make_random_forest_focused_space(best_max_depth=5, best_n_estimators=750)
    n_est_high = space_high["n_estimators"]
    if n_est_high["param_type"] == "int":
        assert n_est_high["high"] <= 800


def test_make_random_forest_focused_space_narrower_min_samples() -> None:
    """make_random_forest_focused_space has narrower min_samples ranges."""
    space = make_random_forest_focused_space(best_max_depth=5, best_n_estimators=200)

    split = space["min_samples_split"]
    if split["param_type"] == "int":
        assert split["low"] == 2
        assert split["high"] == 15

    leaf = space["min_samples_leaf"]
    if leaf["param_type"] == "int":
        assert leaf["low"] == 1
        assert leaf["high"] == 8
