"""Tests for fine-tuning protocol.

Tests cover:
- FineTuningCapabilities TypedDict
- FineTuningStrategyProtocol compliance
- FineTuningStrategyFactory protocol
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.finetuning.protocol import (
    FineTuningCapabilities,
    FineTuningStrategyName,
    FineTuningStrategyProtocol,
)
from covenant_ml.finetuning.strategies import (
    create_iterative_refinement_finetuning,
    create_staged_finetuning,
    create_warm_start_finetuning,
)
from covenant_ml.finetuning.testing import FakeFineTuningStrategy
from covenant_ml.finetuning.types import FineTuningConfig, StageConfig
from covenant_ml.optimizer.types import (
    FloatRangeSpec,
    IntRangeSpec,
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
                stage_name="exploration",
                n_trials=3,
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
# FineTuningCapabilities Tests
# =============================================================================


class TestFineTuningCapabilities:
    """Tests for FineTuningCapabilities TypedDict."""

    def test_create_capabilities(self) -> None:
        """Can create capabilities TypedDict."""
        caps = FineTuningCapabilities(
            supports_warm_start=True,
            supports_staged=True,
            supports_early_stop=True,
            preserves_prior_params=True,
        )

        assert caps["supports_warm_start"] is True
        assert caps["supports_staged"] is True
        assert caps["supports_early_stop"] is True
        assert caps["preserves_prior_params"] is True

    def test_capabilities_all_false(self) -> None:
        """Can create capabilities with all False."""
        caps = FineTuningCapabilities(
            supports_warm_start=False,
            supports_staged=False,
            supports_early_stop=False,
            preserves_prior_params=False,
        )

        assert caps["supports_warm_start"] is False
        assert caps["supports_staged"] is False


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestProtocolCompliance:
    """Tests that implementations satisfy FineTuningStrategyProtocol."""

    def test_fake_strategy_is_protocol(self) -> None:
        """FakeFineTuningStrategy satisfies protocol."""
        strategy: FineTuningStrategyProtocol = FakeFineTuningStrategy()

        # Methods should work correctly
        name: FineTuningStrategyName = strategy.strategy_name()
        assert name in ("staged", "warm_start", "iterative_refinement")

        caps = strategy.capabilities()
        assert caps["supports_warm_start"] is True

    def test_staged_is_protocol(self) -> None:
        """StagedFineTuning satisfies protocol."""
        strategy: FineTuningStrategyProtocol = create_staged_finetuning()

        name = strategy.strategy_name()
        assert name == "staged"

        caps = strategy.capabilities()
        assert caps["supports_staged"] is True

    def test_warm_start_is_protocol(self) -> None:
        """WarmStartFineTuning satisfies protocol."""
        strategy: FineTuningStrategyProtocol = create_warm_start_finetuning()

        name = strategy.strategy_name()
        assert name == "warm_start"

        caps = strategy.capabilities()
        assert caps["supports_warm_start"] is True

    def test_iterative_is_protocol(self) -> None:
        """IterativeRefinementFineTuning satisfies protocol."""
        strategy: FineTuningStrategyProtocol = create_iterative_refinement_finetuning()

        name = strategy.strategy_name()
        assert name == "iterative_refinement"

        caps = strategy.capabilities()
        assert caps["supports_early_stop"] is True


# =============================================================================
# Factory Protocol Tests
# =============================================================================


class TestFactoryProtocol:
    """Tests for FineTuningStrategyFactory protocol."""

    def test_staged_factory_returns_protocol(self) -> None:
        """Staged factory returns protocol-compliant instance."""
        strategy: FineTuningStrategyProtocol = create_staged_finetuning()
        assert strategy.strategy_name() == "staged"
        assert strategy.capabilities()["supports_staged"] is True

    def test_warm_start_factory_returns_protocol(self) -> None:
        """Warm start factory returns protocol-compliant instance."""
        strategy: FineTuningStrategyProtocol = create_warm_start_finetuning()
        assert strategy.strategy_name() == "warm_start"
        assert strategy.capabilities()["supports_warm_start"] is True

    def test_iterative_factory_returns_protocol(self) -> None:
        """Iterative factory returns protocol-compliant instance."""
        strategy: FineTuningStrategyProtocol = create_iterative_refinement_finetuning()
        assert strategy.strategy_name() == "iterative_refinement"
        assert strategy.capabilities()["supports_early_stop"] is True


# =============================================================================
# Strategy Name Type Tests
# =============================================================================


class TestFineTuningStrategyName:
    """Tests for FineTuningStrategyName literal type."""

    def test_staged_returns_valid_name(self) -> None:
        """Staged strategy returns valid name."""
        strategy: FineTuningStrategyProtocol = create_staged_finetuning()
        name: FineTuningStrategyName = strategy.strategy_name()
        assert name == "staged"

    def test_warm_start_returns_valid_name(self) -> None:
        """Warm start strategy returns valid name."""
        strategy: FineTuningStrategyProtocol = create_warm_start_finetuning()
        name: FineTuningStrategyName = strategy.strategy_name()
        assert name == "warm_start"

    def test_iterative_returns_valid_name(self) -> None:
        """Iterative strategy returns valid name."""
        strategy: FineTuningStrategyProtocol = create_iterative_refinement_finetuning()
        name: FineTuningStrategyName = strategy.strategy_name()
        assert name == "iterative_refinement"


# =============================================================================
# Integration Tests
# =============================================================================


class TestProtocolIntegration:
    """Integration tests for protocol usage."""

    def test_can_use_any_strategy_via_protocol(self) -> None:
        """Can use any strategy implementation through protocol interface."""
        x = _make_features(100, 10)
        y = _make_labels(100)
        space = _make_search_space()
        config = _make_config()

        # Use protocol type for the variable
        strategy: FineTuningStrategyProtocol

        # Test with fake strategy
        strategy = FakeFineTuningStrategy()
        result = strategy.fine_tune(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=_dummy_objective,
        )
        assert result["stages_completed"] >= 1

        # Test with real strategy
        strategy = create_staged_finetuning()
        result = strategy.fine_tune(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=_dummy_objective,
        )
        assert result["stages_completed"] >= 1
