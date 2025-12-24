"""Tests for fine-tuning types.

Tests cover:
- StageConfig
- FineTuningConfig
- StageResult
- FineTuningResult
- WarmStartConfig
- Factory functions
"""

from __future__ import annotations

from covenant_ml.finetuning.types import (
    FineTuningConfig,
    FineTuningResult,
    StageConfig,
    StageResult,
    WarmStartConfig,
    make_default_finetuning_config,
    make_default_stage_config,
    make_warm_start_config,
)
from covenant_ml.optimizer.types import (
    OptimizationSummary,
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
)

# =============================================================================
# Test Helpers
# =============================================================================


def _make_optimization_summary(best_value: float = 0.85) -> OptimizationSummary:
    """Create an optimization summary for testing."""
    return OptimizationSummary(
        best_trial_number=0,
        best_value=best_value,
        best_int_params=SampledIntParams(max_depth=5, n_estimators=100),
        best_float_params=SampledFloatParams(learning_rate=0.1),
        best_string_params=SampledStringParams(),
        n_trials_total=10,
        n_trials_complete=10,
        n_trials_pruned=0,
        n_trials_failed=0,
        total_duration_seconds=1.0,
    )


# =============================================================================
# StageConfig Tests
# =============================================================================


class TestStageConfig:
    """Tests for StageConfig TypedDict."""

    def test_create_exploration_stage(self) -> None:
        """Can create exploration stage config."""
        config = StageConfig(
            stage_name="exploration",
            n_trials=50,
            search_radius=1.0,
            use_previous_best=False,
        )

        assert config["stage_name"] == "exploration"
        assert config["n_trials"] == 50
        assert config["search_radius"] == 1.0
        assert config["use_previous_best"] is False

    def test_create_refinement_stage(self) -> None:
        """Can create refinement stage config."""
        config = StageConfig(
            stage_name="refinement",
            n_trials=30,
            search_radius=0.5,
            use_previous_best=True,
        )

        assert config["stage_name"] == "refinement"
        assert config["use_previous_best"] is True

    def test_create_final_stage(self) -> None:
        """Can create final stage config."""
        config = StageConfig(
            stage_name="final",
            n_trials=20,
            search_radius=0.25,
            use_previous_best=True,
        )

        assert config["stage_name"] == "final"


# =============================================================================
# FineTuningConfig Tests
# =============================================================================


class TestFineTuningConfig:
    """Tests for FineTuningConfig TypedDict."""

    def test_create_config(self) -> None:
        """Can create fine-tuning config with stages."""
        stages = (
            StageConfig(
                stage_name="exploration",
                n_trials=50,
                search_radius=1.0,
                use_previous_best=False,
            ),
            StageConfig(
                stage_name="refinement",
                n_trials=30,
                search_radius=0.5,
                use_previous_best=True,
            ),
        )

        config = FineTuningConfig(
            stages=stages,
            random_state=42,
            early_stop_threshold=0.001,
            max_total_trials=100,
        )

        assert len(config["stages"]) == 2
        assert config["random_state"] == 42
        assert config["early_stop_threshold"] == 0.001
        assert config["max_total_trials"] == 100


# =============================================================================
# StageResult Tests
# =============================================================================


class TestStageResult:
    """Tests for StageResult TypedDict."""

    def test_create_stage_result(self) -> None:
        """Can create stage result."""
        summary = _make_optimization_summary()
        result = StageResult(
            stage_name="exploration",
            optimization_summary=summary,
            improvement_over_previous=0.0,
            cumulative_trials=10,
        )

        assert result["stage_name"] == "exploration"
        assert result["optimization_summary"]["best_value"] == 0.85
        assert result["improvement_over_previous"] == 0.0
        assert result["cumulative_trials"] == 10


# =============================================================================
# FineTuningResult Tests
# =============================================================================


class TestFineTuningResult:
    """Tests for FineTuningResult TypedDict."""

    def test_create_finetuning_result(self) -> None:
        """Can create complete fine-tuning result."""
        summary = _make_optimization_summary()
        stage_result = StageResult(
            stage_name="exploration",
            optimization_summary=summary,
            improvement_over_previous=0.0,
            cumulative_trials=10,
        )

        result = FineTuningResult(
            stage_results=(stage_result,),
            final_best_value=0.85,
            final_int_params=SampledIntParams(max_depth=5),
            final_float_params=SampledFloatParams(learning_rate=0.1),
            final_string_params=SampledStringParams(),
            total_trials=10,
            total_duration_seconds=1.5,
            stages_completed=1,
            early_stopped=False,
        )

        assert len(result["stage_results"]) == 1
        assert result["final_best_value"] == 0.85
        assert result["total_trials"] == 10
        assert result["stages_completed"] == 1
        assert result["early_stopped"] is False

    def test_early_stopped_result(self) -> None:
        """Result correctly indicates early stopping."""
        summary = _make_optimization_summary()
        stage_result = StageResult(
            stage_name="exploration",
            optimization_summary=summary,
            improvement_over_previous=0.0001,
            cumulative_trials=10,
        )

        result = FineTuningResult(
            stage_results=(stage_result,),
            final_best_value=0.85,
            final_int_params=SampledIntParams(),
            final_float_params=SampledFloatParams(),
            final_string_params=SampledStringParams(),
            total_trials=10,
            total_duration_seconds=1.0,
            stages_completed=1,
            early_stopped=True,
        )

        assert result["early_stopped"] is True


# =============================================================================
# WarmStartConfig Tests
# =============================================================================


class TestWarmStartConfig:
    """Tests for WarmStartConfig TypedDict."""

    def test_create_warm_start_config(self) -> None:
        """Can create warm start config."""
        summary = _make_optimization_summary()
        config = WarmStartConfig(
            prior_summary=summary,
            narrow_factor=0.5,
            inherit_string_params=True,
        )

        assert config["prior_summary"]["best_value"] == 0.85
        assert config["narrow_factor"] == 0.5
        assert config["inherit_string_params"] is True


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestMakeDefaultStageConfig:
    """Tests for make_default_stage_config factory."""

    def test_exploration_stage_defaults(self) -> None:
        """Exploration stage uses correct defaults."""
        config = make_default_stage_config("exploration")

        assert config["stage_name"] == "exploration"
        assert config["n_trials"] == 50
        assert config["search_radius"] == 0.5
        assert config["use_previous_best"] is False

    def test_refinement_stage_defaults(self) -> None:
        """Refinement stage uses correct defaults."""
        config = make_default_stage_config("refinement")

        assert config["stage_name"] == "refinement"
        assert config["use_previous_best"] is True

    def test_final_stage_defaults(self) -> None:
        """Final stage uses correct defaults."""
        config = make_default_stage_config("final")

        assert config["stage_name"] == "final"
        assert config["use_previous_best"] is True

    def test_custom_parameters(self) -> None:
        """Can override default parameters."""
        config = make_default_stage_config("exploration", n_trials=100, search_radius=0.8)

        assert config["n_trials"] == 100
        assert config["search_radius"] == 0.8


class TestMakeDefaultFineTuningConfig:
    """Tests for make_default_finetuning_config factory."""

    def test_default_config(self) -> None:
        """Default config has three stages."""
        config = make_default_finetuning_config()

        assert len(config["stages"]) == 3
        assert config["stages"][0]["stage_name"] == "exploration"
        assert config["stages"][1]["stage_name"] == "refinement"
        assert config["stages"][2]["stage_name"] == "final"
        assert config["random_state"] == 42
        assert config["early_stop_threshold"] == 0.001

    def test_custom_trial_counts(self) -> None:
        """Can customize trial counts for each stage."""
        config = make_default_finetuning_config(
            exploration_trials=100,
            refinement_trials=50,
            final_trials=25,
        )

        assert config["stages"][0]["n_trials"] == 100
        assert config["stages"][1]["n_trials"] == 50
        assert config["stages"][2]["n_trials"] == 25
        assert config["max_total_trials"] == 175

    def test_custom_random_state(self) -> None:
        """Can customize random state."""
        config = make_default_finetuning_config(random_state=123)

        assert config["random_state"] == 123


class TestMakeWarmStartConfig:
    """Tests for make_warm_start_config factory."""

    def test_default_factory(self) -> None:
        """Factory creates config with defaults."""
        summary = _make_optimization_summary()
        config = make_warm_start_config(summary)

        assert config["prior_summary"]["best_value"] == 0.85
        assert config["narrow_factor"] == 0.5
        assert config["inherit_string_params"] is True

    def test_custom_narrow_factor(self) -> None:
        """Can customize narrow factor."""
        summary = _make_optimization_summary()
        config = make_warm_start_config(summary, narrow_factor=0.3)

        assert config["narrow_factor"] == 0.3
