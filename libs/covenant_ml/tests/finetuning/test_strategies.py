"""Tests for fine-tuning strategy implementations.

Tests cover:
- StagedFineTuning
- WarmStartFineTuning
- IterativeRefinementFineTuning
- Strategy protocol compliance
- Capabilities reporting
"""

from __future__ import annotations

import pytest

from covenant_ml.finetuning.strategies import (
    create_staged_finetuning,
    create_warm_start_finetuning,
)
from covenant_ml.finetuning.strategies.staged import StagedFineTuning
from covenant_ml.finetuning.strategies.warm_start import WarmStartFineTuning
from covenant_ml.finetuning.types import (
    FineTuningConfig,
    StageConfig,
    make_warm_start_config,
)
from tests.finetuning._strategies_fixtures import (
    _dummy_objective,
    _make_features,
    _make_finetuning_config,
    _make_labels,
    _make_prior_summary,
    _make_xgboost_search_space,
)


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
