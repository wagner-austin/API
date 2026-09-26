"""Tests for CV strategy registry.

Tests cover:
- Registry creation
- Strategy registration
- Strategy retrieval
- Default registry population
- Error handling
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.validation import (
    CVSplitterRegistration,
    CVSplitterRegistry,
    CVStrategyName,
    default_cv_registry,
)
from covenant_ml.validation.strategies import StratifiedKFoldSplitter

# =============================================================================
# Test Helpers
# =============================================================================


def _make_labels(n_pos: int, n_neg: int) -> NDArray[np.int64]:
    """Create binary label array with specified class counts."""
    pos: NDArray[np.int64] = np.ones(n_pos, dtype=np.int64)
    neg: NDArray[np.int64] = np.zeros(n_neg, dtype=np.int64)
    result: NDArray[np.int64] = np.concatenate([pos, neg])
    return result


# =============================================================================
# CVSplitterRegistry Tests
# =============================================================================


class TestCVSplitterRegistry:
    """Tests for CVSplitterRegistry."""

    def test_empty_registry(self) -> None:
        """New registry has no registered strategies."""
        registry = CVSplitterRegistry()
        assert registry.list_strategies() == []

    def test_register_and_get(self) -> None:
        """Can register and retrieve a strategy."""
        registry = CVSplitterRegistry()

        def factory() -> StratifiedKFoldSplitter:
            return StratifiedKFoldSplitter()

        registration = CVSplitterRegistration(factory)
        registry.register(CVStrategyName.STRATIFIED_KFOLD, registration)

        splitter = registry.get(CVStrategyName.STRATIFIED_KFOLD)
        assert splitter.strategy_name() is CVStrategyName.STRATIFIED_KFOLD

    def test_list_strategies(self) -> None:
        """List returns all registered strategy names."""
        registry = CVSplitterRegistry()

        def factory() -> StratifiedKFoldSplitter:
            return StratifiedKFoldSplitter()

        registration = CVSplitterRegistration(factory)
        registry.register(CVStrategyName.STRATIFIED_KFOLD, registration)
        registry.register(CVStrategyName.SHUFFLE_SPLIT, registration)

        strategies = registry.list_strategies()
        assert CVStrategyName.STRATIFIED_KFOLD in strategies
        assert CVStrategyName.SHUFFLE_SPLIT in strategies
        assert len(strategies) == 2

    def test_duplicate_registration_raises(self) -> None:
        """Registering same name twice raises ValueError."""
        registry = CVSplitterRegistry()

        def factory() -> StratifiedKFoldSplitter:
            return StratifiedKFoldSplitter()

        registration = CVSplitterRegistration(factory)
        registry.register(CVStrategyName.STRATIFIED_KFOLD, registration)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(CVStrategyName.STRATIFIED_KFOLD, registration)

    def test_has_strategy_returns_true_when_registered(self) -> None:
        """has_strategy returns True for registered strategies."""
        registry = CVSplitterRegistry()

        def factory() -> StratifiedKFoldSplitter:
            return StratifiedKFoldSplitter()

        registration = CVSplitterRegistration(factory)
        registry.register(CVStrategyName.STRATIFIED_KFOLD, registration)

        assert registry.has_strategy(CVStrategyName.STRATIFIED_KFOLD) is True

    def test_has_strategy_returns_false_when_not_registered(self) -> None:
        """has_strategy returns False for unregistered strategies."""
        registry = CVSplitterRegistry()
        assert registry.has_strategy(CVStrategyName.STRATIFIED_KFOLD) is False

    def test_get_capabilities_returns_strategy_capabilities(self) -> None:
        """get_capabilities returns the strategy's capabilities."""
        registry = CVSplitterRegistry()

        def factory() -> StratifiedKFoldSplitter:
            return StratifiedKFoldSplitter()

        registration = CVSplitterRegistration(factory)
        registry.register(CVStrategyName.STRATIFIED_KFOLD, registration)

        caps = registry.get_capabilities(CVStrategyName.STRATIFIED_KFOLD)
        assert caps["preserves_class_ratio"] is True
        assert caps["supports_groups"] is False
        assert caps["supports_temporal"] is False
        assert caps["supports_shuffle"] is True


# =============================================================================
# CVSplitterRegistration Tests
# =============================================================================


class TestCVSplitterRegistration:
    """Tests for CVSplitterRegistration."""

    def test_capabilities_caches_result(self) -> None:
        """capabilities() caches the result after first call."""
        call_count = 0

        def counting_factory() -> StratifiedKFoldSplitter:
            nonlocal call_count
            call_count += 1
            return StratifiedKFoldSplitter()

        registration = CVSplitterRegistration(counting_factory)

        # First call should invoke factory
        caps1 = registration.capabilities()
        assert call_count == 1
        assert caps1["preserves_class_ratio"] is True

        # Second call should use cache
        caps2 = registration.capabilities()
        assert call_count == 1  # Still 1, not 2
        assert caps2["preserves_class_ratio"] is True

    def test_capabilities_returns_correct_values(self) -> None:
        """capabilities() returns the splitter's capabilities."""

        def factory() -> StratifiedKFoldSplitter:
            return StratifiedKFoldSplitter()

        registration = CVSplitterRegistration(factory)
        caps = registration.capabilities()

        assert caps["preserves_class_ratio"] is True
        assert caps["supports_shuffle"] is True


# =============================================================================
# Default Registry Tests
# =============================================================================


class TestDefaultCVRegistry:
    """Tests for the default CV registry."""

    def test_default_registry_has_strategies(self) -> None:
        """Default registry has expected strategies."""
        registry = default_cv_registry()
        strategies = registry.list_strategies()

        assert CVStrategyName.STRATIFIED_KFOLD in strategies
        assert CVStrategyName.GROUP_STRATIFIED_KFOLD in strategies
        assert CVStrategyName.SHUFFLE_SPLIT in strategies
        assert CVStrategyName.TIME_SERIES in strategies

    def test_stratified_kfold_works(self) -> None:
        """Stratified kfold from registry works correctly."""
        registry = default_cv_registry()
        splitter = registry.get(CVStrategyName.STRATIFIED_KFOLD)

        y = _make_labels(50, 50)
        split_info = splitter.split(y, n_folds=5, random_state=42)

        assert split_info["n_folds"] == 5
        assert len(split_info["folds"]) == 5

    def test_group_stratified_kfold_works(self) -> None:
        """Group stratified kfold from registry works correctly."""
        registry = default_cv_registry()
        splitter = registry.get(CVStrategyName.GROUP_STRATIFIED_KFOLD)

        assert splitter.strategy_name() is CVStrategyName.GROUP_STRATIFIED_KFOLD

    def test_shuffle_split_works(self) -> None:
        """Shuffle split from registry works correctly."""
        registry = default_cv_registry()
        splitter = registry.get(CVStrategyName.SHUFFLE_SPLIT)

        assert splitter.strategy_name() is CVStrategyName.SHUFFLE_SPLIT

    def test_time_series_works(self) -> None:
        """Time series from registry works correctly."""
        registry = default_cv_registry()
        splitter = registry.get(CVStrategyName.TIME_SERIES)

        assert splitter.strategy_name() is CVStrategyName.TIME_SERIES

    def test_each_call_returns_fresh_registry(self) -> None:
        """Each call to default_cv_registry returns a new instance."""
        registry1 = default_cv_registry()
        registry2 = default_cv_registry()

        # They should be different instances
        assert registry1 is not registry2

    def test_each_get_returns_fresh_instance(self) -> None:
        """Each get call returns a new splitter instance."""
        registry = default_cv_registry()

        splitter1 = registry.get(CVStrategyName.STRATIFIED_KFOLD)
        splitter2 = registry.get(CVStrategyName.STRATIFIED_KFOLD)

        assert splitter1 is not splitter2
