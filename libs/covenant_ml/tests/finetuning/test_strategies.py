"""Tests for fine-tuning strategy implementations.

Tests cover:
- StagedFineTuning
- WarmStartFineTuning
- IterativeRefinementFineTuning
- Strategy protocol compliance
- Capabilities reporting
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.finetuning.strategies import (
    create_iterative_refinement_finetuning,
    create_staged_finetuning,
    create_warm_start_finetuning,
)
from covenant_ml.finetuning.strategies.iterative import IterativeRefinementFineTuning
from covenant_ml.finetuning.strategies.staged import StagedFineTuning
from covenant_ml.finetuning.strategies.warm_start import WarmStartFineTuning
from covenant_ml.finetuning.types import (
    FineTuningConfig,
    StageConfig,
    WarmStartConfig,
    make_warm_start_config,
)
from covenant_ml.optimizer.types import (
    FloatRangeSpec,
    IntRangeSpec,
    OptimizationSummary,
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
    TrialResult,
    XGBoostSearchSpace,
)

# =============================================================================
# Test Helpers
# =============================================================================


def _make_features(n_samples: int, n_features: int) -> NDArray[np.float64]:
    """Create feature matrix."""
    rng = np.random.default_rng(42)
    result: NDArray[np.float64] = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    return result


def _make_labels(n_samples: int) -> NDArray[np.int64]:
    """Create binary label array."""
    rng = np.random.default_rng(42)
    result: NDArray[np.int64] = rng.integers(0, 2, size=n_samples, dtype=np.int64)
    return result


def _make_xgboost_search_space() -> XGBoostSearchSpace:
    """Create a simple XGBoost search space."""
    return XGBoostSearchSpace(
        max_depth=IntRangeSpec(param_type="int", low=3, high=6, log_scale=False),
        n_estimators=IntRangeSpec(param_type="int", low=50, high=100, log_scale=False),
        learning_rate=FloatRangeSpec(param_type="float", low=0.01, high=0.1, log_scale=True),
        reg_alpha=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
        reg_lambda=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
        subsample=FloatRangeSpec(param_type="float", low=0.6, high=1.0, log_scale=False),
        colsample_bytree=FloatRangeSpec(param_type="float", low=0.6, high=1.0, log_scale=False),
    )


def _make_finetuning_config(n_stages: int = 2, trials_per_stage: int = 3) -> FineTuningConfig:
    """Create fine-tuning config with specified stages."""
    stages: list[StageConfig] = []

    if n_stages >= 1:
        stages.append(
            StageConfig(
                stage_name="exploration",
                n_trials=trials_per_stage,
                search_radius=1.0,
                use_previous_best=False,
            )
        )

    if n_stages >= 2:
        stages.append(
            StageConfig(
                stage_name="refinement",
                n_trials=trials_per_stage,
                search_radius=0.5,
                use_previous_best=True,
            )
        )

    if n_stages >= 3:
        stages.append(
            StageConfig(
                stage_name="final",
                n_trials=trials_per_stage,
                search_radius=0.25,
                use_previous_best=True,
            )
        )

    return FineTuningConfig(
        stages=tuple(stages),
        random_state=42,
        early_stop_threshold=0.001,
        max_total_trials=n_stages * trials_per_stage + 10,
    )


def _make_prior_summary(best_value: float = 0.80) -> OptimizationSummary:
    """Create prior optimization summary for warm-start testing."""
    return OptimizationSummary(
        best_trial_number=0,
        best_value=best_value,
        best_int_params=SampledIntParams(max_depth=5, n_estimators=80),
        best_float_params=SampledFloatParams(
            learning_rate=0.05,
            reg_alpha=0.1,
            reg_lambda=0.5,
            subsample=0.8,
            colsample_bytree=0.8,
        ),
        best_string_params=SampledStringParams(),
        n_trials_total=10,
        n_trials_complete=10,
        n_trials_pruned=0,
        n_trials_failed=0,
        total_duration_seconds=5.0,
    )


def _dummy_objective(
    x_features: NDArray[np.float64],
    y_labels: NDArray[np.int64],
    feature_names: list[str],
    int_params: SampledIntParams,
    float_params: SampledFloatParams,
    string_params: SampledStringParams,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> float:
    """Dummy objective that returns a random value."""
    rng = np.random.default_rng(random_state + int_params.get("max_depth", 0))
    return float(rng.uniform(0.5, 0.9))


# =============================================================================
# StagedFineTuning Tests
# =============================================================================


class TestStagedFineTuning:
    """Tests for StagedFineTuning."""

    def test_strategy_name(self) -> None:
        """Strategy name is correct."""
        strategy = StagedFineTuning()
        assert strategy.strategy_name() == "staged"

    def test_capabilities(self) -> None:
        """Capabilities are correctly reported."""
        strategy = StagedFineTuning()
        caps = strategy.capabilities()

        assert caps["supports_warm_start"] is True
        assert caps["supports_staged"] is True
        assert caps["supports_early_stop"] is True
        assert caps["preserves_prior_params"] is True

    def test_fine_tune_runs_stages(self) -> None:
        """fine_tune runs configured stages."""
        strategy = StagedFineTuning()
        x = _make_features(100, 10)
        y = _make_labels(100)
        space = _make_xgboost_search_space()
        config = _make_finetuning_config(n_stages=2, trials_per_stage=3)

        result = strategy.fine_tune(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=_dummy_objective,
        )

        assert result["stages_completed"] == 2
        assert len(result["stage_results"]) == 2
        assert result["stage_results"][0]["stage_name"] == "exploration"
        assert result["stage_results"][1]["stage_name"] == "refinement"

    def test_fine_tune_returns_best_params(self) -> None:
        """fine_tune returns best parameters."""
        strategy = StagedFineTuning()
        x = _make_features(100, 10)
        y = _make_labels(100)
        space = _make_xgboost_search_space()
        config = _make_finetuning_config(n_stages=1, trials_per_stage=5)

        result = strategy.fine_tune(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=_dummy_objective,
        )

        assert "max_depth" in result["final_int_params"]
        assert "learning_rate" in result["final_float_params"]
        assert result["final_best_value"] > 0

    def test_fine_tune_with_warm_start(self) -> None:
        """fine_tune works with warm start."""
        strategy = StagedFineTuning()
        x = _make_features(100, 10)
        y = _make_labels(100)
        space = _make_xgboost_search_space()
        config = _make_finetuning_config(n_stages=1, trials_per_stage=3)

        prior_summary = _make_prior_summary()
        warm_start = make_warm_start_config(prior_summary, narrow_factor=0.5)

        result = strategy.fine_tune(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=_dummy_objective,
            warm_start=warm_start,
        )

        assert result["stages_completed"] >= 1
        assert result["total_trials"] >= 1

    def test_max_total_trials_limit(self) -> None:
        """Respects max_total_trials limit."""
        strategy = StagedFineTuning()
        x = _make_features(100, 10)
        y = _make_labels(100)
        space = _make_xgboost_search_space()

        config = FineTuningConfig(
            stages=(
                StageConfig(
                    stage_name="exploration",
                    n_trials=100,
                    search_radius=1.0,
                    use_previous_best=False,
                ),
            ),
            random_state=42,
            early_stop_threshold=0.0,
            max_total_trials=5,  # Limit to 5
        )

        result = strategy.fine_tune(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=_dummy_objective,
        )

        assert result["total_trials"] <= 5

    def test_max_total_trials_stops_before_stage(self) -> None:
        """Stops at max_total_trials before starting a new stage."""
        strategy = StagedFineTuning()
        x = _make_features(100, 10)
        y = _make_labels(100)
        space = _make_xgboost_search_space()

        # First stage uses all trials, so second stage should not run
        config = FineTuningConfig(
            stages=(
                StageConfig(
                    stage_name="exploration",
                    n_trials=5,
                    search_radius=1.0,
                    use_previous_best=False,
                ),
                StageConfig(
                    stage_name="refinement",
                    n_trials=5,
                    search_radius=0.5,
                    use_previous_best=True,
                ),
            ),
            random_state=42,
            early_stop_threshold=0.0,
            max_total_trials=5,  # Limit to exactly first stage
        )

        result = strategy.fine_tune(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=_dummy_objective,
        )

        # Should stop after first stage since max_total_trials is reached
        assert result["stages_completed"] == 1

    def test_factory_function(self) -> None:
        """Factory function creates correct strategy."""
        strategy = create_staged_finetuning()
        assert strategy.strategy_name() == "staged"
        caps = strategy.capabilities()
        assert caps["supports_staged"] is True


# =============================================================================
# WarmStartFineTuning Tests
# =============================================================================


class TestWarmStartFineTuning:
    """Tests for WarmStartFineTuning."""

    def test_strategy_name(self) -> None:
        """Strategy name is correct."""
        strategy = WarmStartFineTuning()
        assert strategy.strategy_name() == "warm_start"

    def test_capabilities(self) -> None:
        """Capabilities are correctly reported."""
        strategy = WarmStartFineTuning()
        caps = strategy.capabilities()

        assert caps["supports_warm_start"] is True
        assert caps["supports_staged"] is False
        assert caps["supports_early_stop"] is False
        assert caps["preserves_prior_params"] is True

    def test_fine_tune_single_stage(self) -> None:
        """fine_tune runs single stage."""
        strategy = WarmStartFineTuning()
        x = _make_features(100, 10)
        y = _make_labels(100)
        space = _make_xgboost_search_space()
        config = _make_finetuning_config(n_stages=1, trials_per_stage=3)

        result = strategy.fine_tune(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=_dummy_objective,
        )

        assert result["stages_completed"] == 1
        assert len(result["stage_results"]) == 1
        assert result["early_stopped"] is False

    def test_fine_tune_with_warm_start(self) -> None:
        """fine_tune narrows search with warm start."""
        strategy = WarmStartFineTuning()
        x = _make_features(100, 10)
        y = _make_labels(100)
        space = _make_xgboost_search_space()
        config = _make_finetuning_config(n_stages=1, trials_per_stage=3)

        prior_summary = _make_prior_summary()
        warm_start = make_warm_start_config(prior_summary, narrow_factor=0.5)

        result = strategy.fine_tune(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=_dummy_objective,
            warm_start=warm_start,
        )

        assert result["stages_completed"] == 1
        assert result["total_trials"] >= 1

    def test_empty_stages_raises(self) -> None:
        """Raises ValueError if no stages configured."""
        strategy = WarmStartFineTuning()
        x = _make_features(100, 10)
        y = _make_labels(100)
        space = _make_xgboost_search_space()

        config = FineTuningConfig(
            stages=(),
            random_state=42,
            early_stop_threshold=0.0,
            max_total_trials=10,
        )

        with pytest.raises(ValueError, match="at least one stage"):
            strategy.fine_tune(
                x_features=x,
                y_labels=y,
                feature_names=[f"f{i}" for i in range(10)],
                search_space=space,
                config=config,
                objective=_dummy_objective,
            )

    def test_factory_function(self) -> None:
        """Factory function creates correct strategy."""
        strategy = create_warm_start_finetuning()
        assert strategy.strategy_name() == "warm_start"
        caps = strategy.capabilities()
        assert caps["supports_warm_start"] is True


# =============================================================================
# IterativeRefinementFineTuning Tests
# =============================================================================


class TestIterativeRefinementFineTuning:
    """Tests for IterativeRefinementFineTuning."""

    def test_strategy_name(self) -> None:
        """Strategy name is correct."""
        strategy = IterativeRefinementFineTuning()
        assert strategy.strategy_name() == "iterative_refinement"

    def test_capabilities(self) -> None:
        """Capabilities are correctly reported."""
        strategy = IterativeRefinementFineTuning()
        caps = strategy.capabilities()

        assert caps["supports_warm_start"] is True
        assert caps["supports_staged"] is False
        assert caps["supports_early_stop"] is True
        assert caps["preserves_prior_params"] is True

    def test_default_parameters(self) -> None:
        """Default parameters are correct."""
        strategy = IterativeRefinementFineTuning()
        assert strategy.trials_per_iteration == 20
        assert strategy.max_iterations == 10
        assert strategy.radius_decay == 0.7

    def test_custom_parameters(self) -> None:
        """Can set custom parameters."""
        strategy = IterativeRefinementFineTuning(
            trials_per_iteration=10,
            max_iterations=5,
            radius_decay=0.5,
        )
        assert strategy.trials_per_iteration == 10
        assert strategy.max_iterations == 5
        assert strategy.radius_decay == 0.5

    def test_fine_tune_runs_iterations(self) -> None:
        """fine_tune runs iterations until convergence or max."""
        strategy = IterativeRefinementFineTuning(
            trials_per_iteration=3,
            max_iterations=3,
            radius_decay=0.5,
        )
        x = _make_features(100, 10)
        y = _make_labels(100)
        space = _make_xgboost_search_space()
        config = FineTuningConfig(
            stages=(
                StageConfig(
                    stage_name="exploration",
                    n_trials=3,
                    search_radius=1.0,
                    use_previous_best=False,
                ),
            ),
            random_state=42,
            early_stop_threshold=0.0,  # Disable early stop
            max_total_trials=100,
        )

        result = strategy.fine_tune(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=_dummy_objective,
        )

        assert result["stages_completed"] >= 1
        assert result["total_trials"] >= 3

    def test_fine_tune_with_warm_start(self) -> None:
        """fine_tune works with warm start."""
        strategy = IterativeRefinementFineTuning(
            trials_per_iteration=3,
            max_iterations=2,
            radius_decay=0.5,
        )
        x = _make_features(100, 10)
        y = _make_labels(100)
        space = _make_xgboost_search_space()
        config = FineTuningConfig(
            stages=(),
            random_state=42,
            early_stop_threshold=0.0,
            max_total_trials=100,
        )

        prior_summary = _make_prior_summary()
        warm_start = WarmStartConfig(
            prior_summary=prior_summary,
            narrow_factor=0.5,
            inherit_string_params=True,
        )

        result = strategy.fine_tune(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=_dummy_objective,
            warm_start=warm_start,
        )

        assert result["stages_completed"] >= 1

    def test_max_total_trials_limit(self) -> None:
        """Respects max_total_trials limit."""
        strategy = IterativeRefinementFineTuning(
            trials_per_iteration=10,
            max_iterations=100,
            radius_decay=0.5,
        )
        x = _make_features(100, 10)
        y = _make_labels(100)
        space = _make_xgboost_search_space()
        config = FineTuningConfig(
            stages=(),
            random_state=42,
            early_stop_threshold=0.0,
            max_total_trials=15,  # Limit total trials
        )

        result = strategy.fine_tune(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=_dummy_objective,
        )

        assert result["total_trials"] <= 15

    def test_early_stop_threshold(self) -> None:
        """Stops early when improvement below threshold."""

        # Use a constant objective so improvement is always 0
        def constant_objective(
            x_features: NDArray[np.float64],
            y_labels: NDArray[np.int64],
            feature_names: list[str],
            int_params: SampledIntParams,
            float_params: SampledFloatParams,
            string_params: SampledStringParams,
            train_ratio: float,
            val_ratio: float,
            test_ratio: float,
            random_state: int,
        ) -> float:
            return 0.75  # Constant value, no improvement

        strategy = IterativeRefinementFineTuning(
            trials_per_iteration=3,
            max_iterations=10,
            radius_decay=0.5,
        )
        x = _make_features(100, 10)
        y = _make_labels(100)
        space = _make_xgboost_search_space()
        config = FineTuningConfig(
            stages=(),
            random_state=42,
            early_stop_threshold=0.01,  # Will trigger since improvement is 0
            max_total_trials=100,
        )

        result = strategy.fine_tune(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=constant_objective,
        )

        # Should stop early due to insufficient improvement
        assert result["early_stopped"] is True
        # Should have at least 2 iterations (need one to compute improvement)
        assert result["stages_completed"] >= 2

    def test_stage_names_progression(self) -> None:
        """Stage names progress from exploration to refinement to final."""
        strategy = IterativeRefinementFineTuning(
            trials_per_iteration=3,
            max_iterations=3,
            radius_decay=0.5,
        )
        x = _make_features(100, 10)
        y = _make_labels(100)
        space = _make_xgboost_search_space()
        config = FineTuningConfig(
            stages=(),
            random_state=42,
            early_stop_threshold=0.0,  # Disable early stopping
            max_total_trials=100,
        )

        result = strategy.fine_tune(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=_dummy_objective,
        )

        # First iteration should be exploration
        assert result["stage_results"][0]["stage_name"] == "exploration"
        # Last iteration should be final
        assert result["stage_results"][-1]["stage_name"] == "final"

    def test_factory_function(self) -> None:
        """Factory function creates correct strategy."""
        strategy = create_iterative_refinement_finetuning()
        assert strategy.strategy_name() == "iterative_refinement"
        caps = strategy.capabilities()
        assert caps["supports_early_stop"] is True


# =============================================================================
# Trial Callback Tests
# =============================================================================


class TestTrialCallbacks:
    """Tests for trial callback functionality."""

    def test_staged_callback_called(self) -> None:
        """Staged strategy calls trial callback."""
        strategy = StagedFineTuning()
        x = _make_features(100, 10)
        y = _make_labels(100)
        space = _make_xgboost_search_space()
        config = _make_finetuning_config(n_stages=1, trials_per_stage=3)

        callback_count = 0

        def callback(result: TrialResult) -> None:
            nonlocal callback_count
            callback_count += 1

        strategy.fine_tune(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=_dummy_objective,
            trial_callback=callback,
        )

        assert callback_count >= 3

    def test_warm_start_callback_called(self) -> None:
        """Warm start strategy calls trial callback."""
        strategy = WarmStartFineTuning()
        x = _make_features(100, 10)
        y = _make_labels(100)
        space = _make_xgboost_search_space()
        config = _make_finetuning_config(n_stages=1, trials_per_stage=3)

        callback_count = 0

        def callback(result: TrialResult) -> None:
            nonlocal callback_count
            callback_count += 1

        strategy.fine_tune(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=_dummy_objective,
            trial_callback=callback,
        )

        assert callback_count >= 3

    def test_iterative_callback_called(self) -> None:
        """Iterative strategy calls trial callback."""
        strategy = IterativeRefinementFineTuning(
            trials_per_iteration=3,
            max_iterations=2,
            radius_decay=0.5,
        )
        x = _make_features(100, 10)
        y = _make_labels(100)
        space = _make_xgboost_search_space()
        config = FineTuningConfig(
            stages=(),
            random_state=42,
            early_stop_threshold=0.0,
            max_total_trials=100,
        )

        callback_count = 0

        def callback(result: TrialResult) -> None:
            nonlocal callback_count
            callback_count += 1

        strategy.fine_tune(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=_dummy_objective,
            trial_callback=callback,
        )

        assert callback_count >= 3
