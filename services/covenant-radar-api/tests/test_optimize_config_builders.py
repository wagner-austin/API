"""Tests for the per-backend optimize config builders.

Split from test_model_saver_configs.py by role: these classes test the
pure sampled-params -> train-config builders; the original file keeps the
build_train_config dispatch and the save_best_model workflow.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import pytest
from covenant_ml.optimizer.types import (
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
)
from scripts.optimize._config_builders import (
    _build_cleargbm_config,
    _build_lightgbm_config,
    _build_logreg_config,
    _build_lstm_config,
    _build_mlp_config,
    _build_random_forest_config,
    _build_xgboost_config,
    _narrow_logreg_penalty,
    _narrow_logreg_solver,
)


class TestBuildXGBoostConfig:
    """Tests for _build_xgboost_config."""

    def test_xgboost_config(self) -> None:
        """XGBoost config has expected fields."""
        config = _build_xgboost_config(
            SampledIntParams(max_depth=6, n_estimators=100),
            SampledFloatParams(
                learning_rate=0.1,
                reg_alpha=0.01,
                reg_lambda=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
            ),
        )
        assert config["learning_rate"] == 0.1
        assert config["max_depth"] == 6
        assert config["n_estimators"] == 100
        assert config["subsample"] == 0.8
        assert config["colsample_bytree"] == 0.8
        assert config["reg_alpha"] == 0.01
        assert config["reg_lambda"] == 0.01
        assert config["device"] == "auto"
        assert config["early_stopping_rounds"] == 10


class TestBuildMLPConfig:
    """Tests for _build_mlp_config."""

    def test_mlp_config(self) -> None:
        """MLP config has hidden_sizes tuple from n_layers and hidden_size."""
        config = _build_mlp_config(
            SampledIntParams(n_layers=3, hidden_size=128, batch_size=64),
            SampledFloatParams(learning_rate=0.001, dropout=0.2),
        )
        assert config["hidden_sizes"] == (128, 128, 128)
        assert config["batch_size"] == 64
        assert config["dropout"] == 0.2
        assert config["learning_rate"] == 0.001
        assert config["precision"] == "fp32"
        assert config["optimizer"] == "adamw"
        assert config["n_epochs"] == 50
        assert config["early_stopping_patience"] == 10


class TestBuildLSTMConfig:
    """Tests for _build_lstm_config."""

    def test_lstm_config(self) -> None:
        """LSTM config has hidden_size, num_layers, dropout."""
        config = _build_lstm_config(
            SampledIntParams(hidden_size=64, num_layers=2, batch_size=32),
            SampledFloatParams(learning_rate=0.001, dropout=0.3),
        )
        assert config["hidden_size"] == 64
        assert config["num_layers"] == 2
        assert config["dropout"] == 0.3
        assert config["learning_rate"] == 0.001
        assert config["bidirectional"] is False
        assert config["sequence_length"] == 5
        assert config["n_epochs"] == 50


class TestBuildLightGBMConfig:
    """Tests for _build_lightgbm_config."""

    def test_lightgbm_config(self) -> None:
        """LightGBM config has num_leaves, min_child_samples."""
        config = _build_lightgbm_config(
            SampledIntParams(max_depth=-1, n_estimators=100, num_leaves=31, min_child_samples=20),
            SampledFloatParams(
                learning_rate=0.1,
                reg_alpha=0.01,
                reg_lambda=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
            ),
        )
        assert config["num_leaves"] == 31
        assert config["min_child_samples"] == 20
        assert config["max_depth"] == -1
        assert config["early_stopping_rounds"] == 10


class TestBuildClearGBMConfig:
    """Tests for _build_cleargbm_config."""

    def test_cleargbm_config(self) -> None:
        """ClearGBM config has min_samples_split, min_samples_leaf, max_bins."""
        config = _build_cleargbm_config(
            SampledIntParams(
                max_depth=5,
                n_estimators=100,
                min_samples_split=10,
                min_samples_leaf=5,
                max_bins=64,
            ),
            SampledFloatParams(
                learning_rate=0.1,
                subsample=1.0,
                reg_alpha=0.0,
                reg_lambda=1.0,
            ),
        )
        assert config["min_samples_split"] == 10
        assert config["min_samples_leaf"] == 5
        assert config["max_bins"] == 64
        assert config["monotonic_constraints"] is None
        assert config["n_jobs"] == -1
        assert config["growth_strategy"] == "depth_wise"
        assert config["num_leaves"] is None

    def test_unsampled_reg_defaults_to_the_trial_values(self) -> None:
        """Without sampled reg params the saved config matches the trials.

        The ClearGBM search space samples no regularization; every trial
        trains at 0.0/0.0. Requiring the keys here crashed save_best_model
        at the end of every ClearGBM sweep (KeyError: 'reg_alpha').
        """
        config = _build_cleargbm_config(
            SampledIntParams(
                max_depth=5,
                n_estimators=100,
                min_samples_split=10,
                min_samples_leaf=5,
                max_bins=64,
            ),
            SampledFloatParams(learning_rate=0.1, subsample=1.0),
        )
        assert config["reg_alpha"] == 0.0
        assert config["reg_lambda"] == 0.0


class TestBuildLogRegConfig:
    """Tests for _build_logreg_config."""

    def test_logreg_config(self) -> None:
        """LogReg config has solver, penalty, C, tol, l1_ratio."""
        config = _build_logreg_config(
            SampledFloatParams(C=1.0, tol=0.0001, l1_ratio=0.5),
            SampledStringParams(solver="saga", penalty="elasticnet"),
        )
        assert config["solver"] == "saga"
        assert config["penalty"] == "elasticnet"
        assert config["C"] == 1.0
        assert config["tol"] == 0.0001
        assert config["l1_ratio"] == 0.5
        assert config["max_iter"] == 1000
        assert config["class_weight_balanced"] is True

    def test_l2_lbfgs(self) -> None:
        """LogReg config with l2 penalty and lbfgs solver."""
        config = _build_logreg_config(
            SampledFloatParams(C=0.5, tol=0.001, l1_ratio=0.0),
            SampledStringParams(solver="lbfgs", penalty="l2"),
        )
        assert config["solver"] == "lbfgs"
        assert config["penalty"] == "l2"

    def test_l1_liblinear(self) -> None:
        """LogReg config with l1 penalty and liblinear solver."""
        config = _build_logreg_config(
            SampledFloatParams(C=10.0, tol=0.0001, l1_ratio=0.0),
            SampledStringParams(solver="liblinear", penalty="l1"),
        )
        assert config["solver"] == "liblinear"
        assert config["penalty"] == "l1"

    def test_none_penalty_newton_cg(self) -> None:
        """LogReg config with none penalty and newton-cg solver."""
        config = _build_logreg_config(
            SampledFloatParams(C=1.0, tol=0.0001, l1_ratio=0.0),
            SampledStringParams(solver="newton-cg", penalty="none"),
        )
        assert config["solver"] == "newton-cg"
        assert config["penalty"] == "none"

    def test_newton_cholesky(self) -> None:
        """LogReg config with newton-cholesky solver."""
        config = _build_logreg_config(
            SampledFloatParams(C=1.0, tol=0.0001, l1_ratio=0.0),
            SampledStringParams(solver="newton-cholesky", penalty="l2"),
        )
        assert config["solver"] == "newton-cholesky"

    def test_sag(self) -> None:
        """LogReg config with sag solver."""
        config = _build_logreg_config(
            SampledFloatParams(C=1.0, tol=0.0001, l1_ratio=0.0),
            SampledStringParams(solver="sag", penalty="l2"),
        )
        assert config["solver"] == "sag"


class TestBuildRandomForestConfig:
    """Tests for _build_random_forest_config."""

    def test_sqrt_features(self) -> None:
        """RandomForest config with max_features='sqrt'."""
        config = _build_random_forest_config(
            SampledIntParams(n_estimators=200, min_samples_split=5, min_samples_leaf=2),
            SampledFloatParams(),
            SampledStringParams(max_features="sqrt"),
        )
        assert config["max_features"] == "sqrt"
        assert config["n_estimators"] == 200
        assert config["min_samples_split"] == 5
        assert config["min_samples_leaf"] == 2
        assert config["bootstrap"] is True
        assert config["class_weight_balanced"] is True

    def test_log2_features(self) -> None:
        """RandomForest config with max_features='log2'."""
        config = _build_random_forest_config(
            SampledIntParams(n_estimators=100, min_samples_split=2, min_samples_leaf=1),
            SampledFloatParams(),
            SampledStringParams(max_features="log2"),
        )
        assert config["max_features"] == "log2"

    def test_float_features(self) -> None:
        """RandomForest config with float max_features via max_features_float."""
        config = _build_random_forest_config(
            SampledIntParams(n_estimators=100, min_samples_split=2, min_samples_leaf=1),
            SampledFloatParams(max_features_float=0.7),
            SampledStringParams(),
        )
        assert config["max_features"] == 0.7

    def test_default_features(self) -> None:
        """RandomForest config defaults to 'sqrt' when no max_features specified."""
        config = _build_random_forest_config(
            SampledIntParams(n_estimators=100, min_samples_split=2, min_samples_leaf=1),
            SampledFloatParams(),
            SampledStringParams(),
        )
        assert config["max_features"] == "sqrt"


class TestNarrowLogRegSolver:
    """Tests for _narrow_logreg_solver."""

    def test_all_valid_solvers(self) -> None:
        """All valid solver names are narrowed correctly."""
        solvers = ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
        for solver in solvers:
            assert _narrow_logreg_solver(solver) == solver

    def test_invalid_solver_raises(self) -> None:
        """Invalid solver name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid LogReg solver"):
            _narrow_logreg_solver("invalid_solver")


class TestNarrowLogRegPenalty:
    """Tests for _narrow_logreg_penalty."""

    def test_all_valid_penalties(self) -> None:
        """All valid penalty names are narrowed correctly."""
        penalties = ["l1", "l2", "elasticnet", "none"]
        for penalty in penalties:
            assert _narrow_logreg_penalty(penalty) == penalty

    def test_invalid_penalty_raises(self) -> None:
        """Invalid penalty name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid LogReg penalty"):
            _narrow_logreg_penalty("invalid_penalty")
