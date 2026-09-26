"""Tests for fine-tuning testing utilities.

Tests cover:
- FakeFineTuningStrategy
- Factory functions
- Test registry creation
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.finetuning.protocol import FineTuningCapabilities, FineTuningStrategyName
from covenant_ml.finetuning.testing import (
    FakeFineTuningStrategy,
    make_fake_finetuning_strategy,
    make_test_finetuning_registry,
)
from covenant_ml.finetuning.types import (
    FineTuningConfig,
    FineTuningResult,
    FineTuningStage,
    StageConfig,
    StageResult,
)
from covenant_ml.optimizer.types import (
    FloatRangeSpec,
    IntRangeSpec,
    OptimizationSummary,
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
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


def _make_search_space() -> XGBoostSearchSpace:
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


def _make_config() -> FineTuningConfig:
    """Create fine-tuning config."""
    return FineTuningConfig(
        stages=(
            StageConfig(
                stage_name=FineTuningStage.EXPLORATION,
                n_trials=5,
                search_radius=1.0,
                use_previous_best=False,
            ),
        ),
        random_state=42,
        early_stop_threshold=0.001,
        max_total_trials=10,
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
    """Dummy objective."""
    return 0.85


# =============================================================================
# FakeFineTuningStrategy Tests
# =============================================================================


class TestFakeFineTuningStrategy:
    """Tests for FakeFineTuningStrategy."""

    def test_default_strategy_name(self) -> None:
        """Default strategy name is staged."""
        strategy = FakeFineTuningStrategy()
        assert strategy.strategy_name() is FineTuningStrategyName.STAGED

    def test_custom_strategy_name(self) -> None:
        """Can set custom strategy name."""
        strategy = FakeFineTuningStrategy(name=FineTuningStrategyName.WARM_START)
        assert strategy.strategy_name() is FineTuningStrategyName.WARM_START

    def test_default_capabilities(self) -> None:
        """Default capabilities are correct."""
        strategy = FakeFineTuningStrategy()
        caps = strategy.capabilities()

        assert caps["supports_warm_start"] is True
        assert caps["supports_staged"] is True
        assert caps["supports_early_stop"] is True
        assert caps["preserves_prior_params"] is True

    def test_custom_capabilities(self) -> None:
        """Can set custom capabilities."""
        custom_caps = FineTuningCapabilities(
            supports_warm_start=True,
            supports_staged=False,
            supports_early_stop=False,
            preserves_prior_params=True,
        )
        strategy = FakeFineTuningStrategy(capabilities=custom_caps)
        caps = strategy.capabilities()

        assert caps["supports_staged"] is False
        assert caps["supports_early_stop"] is False

    def test_fine_tune_returns_generated_result(self) -> None:
        """fine_tune returns generated fake result."""
        strategy = FakeFineTuningStrategy()
        x = _make_features(100, 10)
        y = _make_labels(100)
        space = _make_search_space()
        config = _make_config()

        result = strategy.fine_tune(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=_dummy_objective,
        )

        assert result["final_best_value"] == 0.85
        assert result["stages_completed"] == 1
        assert result["early_stopped"] is False

    def test_custom_result(self) -> None:
        """Can provide custom result."""
        stage_result = StageResult(
            stage_name=FineTuningStage.EXPLORATION,
            optimization_summary=OptimizationSummary(
                best_trial_number=0,
                best_value=0.99,
                best_int_params=SampledIntParams(max_depth=10),
                best_float_params=SampledFloatParams(learning_rate=0.05),
                best_string_params=SampledStringParams(),
                n_trials_total=20,
                n_trials_complete=20,
                n_trials_pruned=0,
                n_trials_failed=0,
                total_duration_seconds=2.0,
            ),
            improvement_over_previous=0.1,
            cumulative_trials=20,
        )

        custom_result = FineTuningResult(
            stage_results=(stage_result,),
            final_best_value=0.99,
            final_int_params=SampledIntParams(max_depth=10),
            final_float_params=SampledFloatParams(learning_rate=0.05),
            final_string_params=SampledStringParams(),
            total_trials=20,
            total_duration_seconds=2.0,
            stages_completed=1,
            early_stopped=False,
        )

        strategy = FakeFineTuningStrategy(result=custom_result)
        x = _make_features(100, 10)
        y = _make_labels(100)
        space = _make_search_space()
        config = _make_config()

        result = strategy.fine_tune(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=_dummy_objective,
        )

        assert result["final_best_value"] == 0.99
        assert result["total_trials"] == 20

    def test_fine_tune_call_count(self) -> None:
        """Tracks number of fine_tune calls."""
        strategy = FakeFineTuningStrategy()
        x = _make_features(100, 10)
        y = _make_labels(100)
        space = _make_search_space()
        config = _make_config()

        assert strategy.fine_tune_call_count == 0

        strategy.fine_tune(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=_dummy_objective,
        )
        assert strategy.fine_tune_call_count == 1

        strategy.fine_tune(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=_dummy_objective,
        )
        assert strategy.fine_tune_call_count == 2


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestMakeFakeFineTuningStrategy:
    """Tests for make_fake_finetuning_strategy factory."""

    def test_default_factory(self) -> None:
        """Factory creates strategy with defaults."""
        strategy = make_fake_finetuning_strategy()
        assert strategy.strategy_name() is FineTuningStrategyName.STAGED
        caps = strategy.capabilities()
        assert caps["supports_staged"] is True

    def test_factory_with_custom_name(self) -> None:
        """Factory creates strategy with custom name."""
        strategy = make_fake_finetuning_strategy(name=FineTuningStrategyName.ITERATIVE_REFINEMENT)
        assert strategy.strategy_name() is FineTuningStrategyName.ITERATIVE_REFINEMENT

    def test_factory_with_custom_value(self) -> None:
        """Factory creates strategy with custom best value."""
        strategy = make_fake_finetuning_strategy(best_value=0.95)
        x = _make_features(100, 10)
        y = _make_labels(100)
        space = _make_search_space()
        config = _make_config()

        result = strategy.fine_tune(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=_dummy_objective,
        )

        assert result["final_best_value"] == 0.95


# =============================================================================
# Test Registry Tests
# =============================================================================


class TestMakeTestFineTuningRegistry:
    """Tests for make_test_finetuning_registry factory."""

    def test_registry_has_all_strategies(self) -> None:
        """Test registry has expected fake strategies."""
        registry = make_test_finetuning_registry()
        strategies = registry.list_strategies()

        assert FineTuningStrategyName.STAGED in strategies
        assert FineTuningStrategyName.WARM_START in strategies
        assert FineTuningStrategyName.ITERATIVE_REFINEMENT in strategies

    def test_strategies_are_fake(self) -> None:
        """All strategies are FakeFineTuningStrategy instances."""
        registry = make_test_finetuning_registry()

        strategy = registry.get(FineTuningStrategyName.STAGED)
        assert strategy.strategy_name() is FineTuningStrategyName.STAGED

        strategy2 = registry.get(FineTuningStrategyName.WARM_START)
        assert strategy2.strategy_name() is FineTuningStrategyName.WARM_START

    def test_iterative_refinement_strategy(self) -> None:
        """Iterative refinement strategy is accessible and works."""
        registry = make_test_finetuning_registry()
        strategy = registry.get(FineTuningStrategyName.ITERATIVE_REFINEMENT)

        assert strategy.strategy_name() is FineTuningStrategyName.ITERATIVE_REFINEMENT
        caps = strategy.capabilities()
        assert caps["supports_staged"] is True

    def test_strategies_work(self) -> None:
        """Fake strategies produce valid results."""
        registry = make_test_finetuning_registry()
        strategy = registry.get(FineTuningStrategyName.STAGED)

        x = _make_features(100, 10)
        y = _make_labels(100)
        space = _make_search_space()
        config = _make_config()

        result = strategy.fine_tune(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=_dummy_objective,
        )

        assert result["stages_completed"] >= 1
        assert result["final_best_value"] > 0
