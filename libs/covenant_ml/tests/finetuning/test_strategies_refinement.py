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
from numpy.typing import NDArray

from covenant_ml.finetuning.strategies import (
    create_iterative_refinement_finetuning,
)
from covenant_ml.finetuning.strategies.iterative import IterativeRefinementFineTuning
from covenant_ml.finetuning.strategies.staged import StagedFineTuning
from covenant_ml.finetuning.strategies.warm_start import WarmStartFineTuning
from covenant_ml.finetuning.types import (
    FineTuningConfig,
    StageConfig,
    WarmStartConfig,
)
from covenant_ml.optimizer.types import (
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
    TrialResult,
)
from tests.finetuning._strategies_fixtures import (
    _dummy_objective,
    _make_features,
    _make_finetuning_config,
    _make_labels,
    _make_prior_summary,
    _make_xgboost_search_space,
)


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
