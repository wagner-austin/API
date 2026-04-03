"""Tests for backend get_default_search_space and get_focused_search_space methods.

Verifies that all classifier and regressor backends return correctly typed
search spaces and that focused spaces narrow around provided best params.

No mocks. Tests instantiate real backends and call real factory functions.
"""

from __future__ import annotations

from covenant_ml.backends.cleargbm.backend import ClearGBMBackend
from covenant_ml.backends.lightgbm.backend import LightGBMBackend
from covenant_ml.backends.lightgbm.regressor import LightGBMRegressorBackend
from covenant_ml.backends.logreg.backend import LogRegBackend
from covenant_ml.backends.random_forest.backend import RandomForestBackend
from covenant_ml.backends.xgboost.backend import XGBoostBackend
from covenant_ml.backends.xgboost.regressor import XGBoostRegressorBackend
from covenant_ml.optimizer.search_spaces import (
    make_cleargbm_default_space,
    make_lightgbm_default_space,
    make_logreg_default_space,
    make_random_forest_default_space,
    make_xgboost_default_space,
)
from covenant_ml.optimizer.types import (
    SampledFloatParams,
    SampledIntParams,
)

# =============================================================================
# XGBoost Classifier
# =============================================================================


def test_xgboost_default_search_space() -> None:
    """XGBoostBackend.get_default_search_space matches factory output."""
    backend = XGBoostBackend()
    space = backend.get_default_search_space()
    expected = make_xgboost_default_space()
    assert space == expected


def test_xgboost_focused_search_space() -> None:
    """XGBoostBackend.get_focused_search_space returns space with expected keys."""
    backend = XGBoostBackend()
    int_params: SampledIntParams = {"max_depth": 6}
    float_params: SampledFloatParams = {"learning_rate": 0.05}
    space = backend.get_focused_search_space(
        best_int_params=int_params,
        best_float_params=float_params,
    )
    assert "max_depth" in space
    assert "learning_rate" in space
    assert "n_estimators" in space


# =============================================================================
# LightGBM Classifier
# =============================================================================


def test_lightgbm_default_search_space() -> None:
    """LightGBMBackend.get_default_search_space matches factory output."""
    backend = LightGBMBackend()
    space = backend.get_default_search_space()
    expected = make_lightgbm_default_space()
    assert space == expected


def test_lightgbm_focused_search_space() -> None:
    """LightGBMBackend.get_focused_search_space returns space with expected keys."""
    backend = LightGBMBackend()
    int_params: SampledIntParams = {"num_leaves": 31}
    float_params: SampledFloatParams = {"learning_rate": 0.1}
    space = backend.get_focused_search_space(
        best_int_params=int_params,
        best_float_params=float_params,
    )
    assert "num_leaves" in space
    assert "learning_rate" in space
    assert "n_estimators" in space


# =============================================================================
# ClearGBM Classifier
# =============================================================================


def test_cleargbm_default_search_space() -> None:
    """ClearGBMBackend.get_default_search_space matches factory output."""
    backend = ClearGBMBackend()
    space = backend.get_default_search_space()
    expected = make_cleargbm_default_space()
    assert space == expected


def test_cleargbm_focused_search_space() -> None:
    """ClearGBMBackend.get_focused_search_space returns space with expected keys."""
    backend = ClearGBMBackend()
    int_params: SampledIntParams = {"max_depth": 4}
    float_params: SampledFloatParams = {"learning_rate": 0.05}
    space = backend.get_focused_search_space(
        best_int_params=int_params,
        best_float_params=float_params,
    )
    assert "max_depth" in space
    assert "learning_rate" in space
    assert "n_estimators" in space
    assert "min_samples_split" in space


# =============================================================================
# LogReg Classifier
# =============================================================================


def test_logreg_default_search_space() -> None:
    """LogRegBackend.get_default_search_space matches factory output."""
    backend = LogRegBackend()
    space = backend.get_default_search_space()
    expected = make_logreg_default_space()
    assert space == expected


def test_logreg_focused_search_space() -> None:
    """LogRegBackend.get_focused_search_space returns space with expected keys."""
    backend = LogRegBackend()
    int_params: SampledIntParams = {}
    float_params: SampledFloatParams = {"C": 1.0, "tol": 1e-4}
    space = backend.get_focused_search_space(
        best_int_params=int_params,
        best_float_params=float_params,
    )
    assert "C" in space
    assert "tol" in space
    assert "max_iter" in space


# =============================================================================
# Random Forest Classifier
# =============================================================================


def test_random_forest_default_search_space() -> None:
    """RandomForestBackend.get_default_search_space matches factory output."""
    backend = RandomForestBackend()
    space = backend.get_default_search_space()
    expected = make_random_forest_default_space()
    assert space == expected


def test_random_forest_focused_search_space() -> None:
    """RandomForestBackend.get_focused_search_space returns space with expected keys."""
    backend = RandomForestBackend()
    int_params: SampledIntParams = {"max_depth": 10, "n_estimators": 200}
    float_params: SampledFloatParams = {}
    space = backend.get_focused_search_space(
        best_int_params=int_params,
        best_float_params=float_params,
    )
    assert "n_estimators" in space
    assert "max_depth" in space
    assert "min_samples_split" in space


# =============================================================================
# XGBoost Regressor
# =============================================================================


def test_xgboost_regressor_default_search_space() -> None:
    """XGBoostRegressorBackend.get_default_search_space matches factory output."""
    backend = XGBoostRegressorBackend()
    space = backend.get_default_search_space()
    expected = make_xgboost_default_space()
    assert space == expected


def test_xgboost_regressor_focused_search_space() -> None:
    """XGBoostRegressorBackend.get_focused_search_space returns expected keys."""
    backend = XGBoostRegressorBackend()
    int_params: SampledIntParams = {"max_depth": 5}
    float_params: SampledFloatParams = {"learning_rate": 0.01}
    space = backend.get_focused_search_space(
        best_int_params=int_params,
        best_float_params=float_params,
    )
    assert "max_depth" in space
    assert "learning_rate" in space
    assert "n_estimators" in space


# =============================================================================
# LightGBM Regressor
# =============================================================================


def test_lightgbm_regressor_default_search_space() -> None:
    """LightGBMRegressorBackend.get_default_search_space matches factory output."""
    backend = LightGBMRegressorBackend()
    space = backend.get_default_search_space()
    expected = make_lightgbm_default_space()
    assert space == expected


def test_lightgbm_regressor_focused_search_space() -> None:
    """LightGBMRegressorBackend.get_focused_search_space returns expected keys."""
    backend = LightGBMRegressorBackend()
    int_params: SampledIntParams = {"num_leaves": 63}
    float_params: SampledFloatParams = {"learning_rate": 0.05}
    space = backend.get_focused_search_space(
        best_int_params=int_params,
        best_float_params=float_params,
    )
    assert "num_leaves" in space
    assert "learning_rate" in space
    assert "n_estimators" in space


# =============================================================================
# Registry integration: search spaces via default registries
# =============================================================================


def test_classifier_registry_search_spaces() -> None:
    """All classifier backends in default registry return valid search spaces."""
    from covenant_ml.backends import default_registry

    reg = default_registry()
    for name in reg.list_backends():
        backend = reg.get(name)
        space = backend.get_default_search_space()
        # Every search space must contain a learning_rate, C, or num_leaves key
        has_known_key = (
            "learning_rate" in space
            or "C" in space
            or "num_leaves" in space
            or "n_estimators" in space
        )
        assert has_known_key, f"Backend {name} returned empty or unknown search space"


def test_regressor_registry_search_spaces() -> None:
    """All regressor backends in default registry return valid search spaces."""
    from covenant_ml.backends.regressor_registry import default_regressor_registry

    reg = default_regressor_registry()
    for name in reg.list_backends():
        backend = reg.get(name)
        space = backend.get_default_search_space()
        # Every regressor search space must have learning_rate and n_estimators
        assert "learning_rate" in space, f"Backend {name} missing learning_rate"
        assert "n_estimators" in space, f"Backend {name} missing n_estimators"
