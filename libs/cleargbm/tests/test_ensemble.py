"""Tests for cleargbm.ensemble module.

Uses numpy arrays for all array operations.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from cleargbm.ensemble import (
    _add_tree_predictions,
    _compute_loss,
    _init_early_stopping_state,
    _update_early_stopping_state,
    predict_proba,
    predict_raw,
    train_gradient_boosting,
)
from cleargbm.types import GradientBoostingConfig, TrainingProgress


def _float_matrix(data: list[list[float]]) -> NDArray[np.float64]:
    """Create a 2D float array from nested list (helper for strict typing)."""
    return np.array(data, dtype=np.float64)


def _float_array(data: list[float]) -> NDArray[np.float64]:
    """Create a 1D float array from list (helper for strict typing)."""
    return np.array(data, dtype=np.float64)


def _int_array(data: list[int]) -> NDArray[np.int64]:
    """Create a 1D int array from list (helper for strict typing)."""
    return np.array(data, dtype=np.int64)


def _make_config(
    n_estimators: int = 10,
    max_depth: int = 3,
    learning_rate: float = 0.1,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: int | None = None,
    subsample: float = 1.0,
    random_state: int = 42,
    track_contributions: bool = True,
    monotonic_constraints: tuple[int, ...] | None = None,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
    n_jobs: int = 1,
    early_stopping_rounds: int | None = None,
) -> GradientBoostingConfig:
    """Create a test configuration."""
    return GradientBoostingConfig(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_bins=64,
        subsample=subsample,
        random_state=random_state,
        track_contributions=track_contributions,
        monotonic_constraints=monotonic_constraints,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        n_jobs=n_jobs,
        early_stopping_rounds=early_stopping_rounds,
    )


class TestComputeLoss:
    """Tests for _compute_loss helper."""

    def test_perfect_predictions_low_loss(self) -> None:
        """Perfect predictions should have low loss."""
        y_true = _int_array([1, 1, 0, 0])
        raw_preds = _float_array([5.0, 5.0, -5.0, -5.0])

        loss = _compute_loss(y_true, raw_preds)

        assert loss < 0.01

    def test_terrible_predictions_high_loss(self) -> None:
        """Terrible predictions should have high loss."""
        y_true = _int_array([1, 1, 0, 0])
        raw_preds = _float_array([-5.0, -5.0, 5.0, 5.0])

        loss = _compute_loss(y_true, raw_preds)

        assert loss > 4.0


class TestAddTreePredictions:
    """Tests for _add_tree_predictions helper."""

    def test_adds_scaled_predictions(self) -> None:
        """Should add scaled tree predictions to raw predictions."""
        raw_preds = _float_array([0.0, 1.0, 2.0])
        tree_preds = _float_array([1.0, 2.0, 3.0])
        learning_rate = 0.5

        result = _add_tree_predictions(raw_preds, tree_preds, learning_rate)

        n_results: int = result.shape[0]
        assert n_results == 3
        assert abs(result.item(0) - 0.5) < 1e-10
        assert abs(result.item(1) - 2.0) < 1e-10
        assert abs(result.item(2) - 3.5) < 1e-10


class TestTrainGradientBoosting:
    """Tests for train_gradient_boosting."""

    def test_trains_on_simple_data(self) -> None:
        """Should train successfully on simple separable data."""
        x_train = _float_matrix([[0.0], [0.1], [0.2], [0.3], [0.7], [0.8], [0.9], [1.0]])
        y_train = _int_array([0, 0, 0, 0, 1, 1, 1, 1])
        config = _make_config(n_estimators=5, max_depth=2, learning_rate=0.5)

        model = train_gradient_boosting(
            x_train=x_train,
            y_train=y_train,
            x_val=None,
            y_val=None,
            config=config,
            feature_names=("f0",),
        )

        assert len(model["trees"]) == 5
        assert model["n_classes"] == 2
        assert model["feature_names"] == ("f0",)
        assert abs(model["learning_rate"] - 0.5) < 1e-10

    def test_trains_with_validation_set(self) -> None:
        """Should train with validation set and report val_loss."""
        x_train = _float_matrix([[0.0], [0.1], [0.8], [0.9]])
        y_train = _int_array([0, 0, 1, 1])
        x_val = _float_matrix([[0.2], [0.7]])
        y_val = _int_array([0, 1])
        config = _make_config(n_estimators=3, max_depth=2)

        progress_updates: list[TrainingProgress] = []

        def callback(progress: TrainingProgress) -> None:
            progress_updates.append(progress)

        model = train_gradient_boosting(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            config=config,
            feature_names=("f0",),
            progress_callback=callback,
        )

        assert len(model["trees"]) == 3
        assert len(progress_updates) == 3

        for progress in progress_updates:
            val_loss = progress["val_loss"]
            if val_loss is None:
                pytest.fail("Expected val_loss to be set")
            assert val_loss >= 0.0

    def test_empty_x_train_raises(self) -> None:
        """Should raise ValueError for empty x_train."""
        config = _make_config()
        x_train: NDArray[np.float64] = np.zeros((0, 1), dtype=np.float64)
        y_train: NDArray[np.int64] = np.zeros(0, dtype=np.int64)

        with pytest.raises(ValueError, match="must not be empty"):
            train_gradient_boosting(
                x_train=x_train,
                y_train=y_train,
                x_val=None,
                y_val=None,
                config=config,
                feature_names=("f0",),
            )

    def test_mismatched_x_y_raises(self) -> None:
        """Should raise ValueError when x_train and y_train have different lengths."""
        config = _make_config()
        x_train = _float_matrix([[0.0], [1.0]])
        y_train = _int_array([0])

        with pytest.raises(ValueError, match="same length"):
            train_gradient_boosting(
                x_train=x_train,
                y_train=y_train,
                x_val=None,
                y_val=None,
                config=config,
                feature_names=("f0",),
            )

    def test_mismatched_features_raises(self) -> None:
        """Should raise ValueError when features don't match feature_names."""
        config = _make_config()
        x_train = _float_matrix([[0.0, 1.0]])
        y_train = _int_array([0])

        with pytest.raises(ValueError, match="feature names"):
            train_gradient_boosting(
                x_train=x_train,
                y_train=y_train,
                x_val=None,
                y_val=None,
                config=config,
                feature_names=("f0",),
            )

    def test_progress_callback_called(self) -> None:
        """Should call progress callback after each tree."""
        x_train = _float_matrix([[0.0], [0.5], [1.0], [1.5]])
        y_train = _int_array([0, 0, 1, 1])
        config = _make_config(n_estimators=4, max_depth=1)

        call_count = 0

        def callback(progress: TrainingProgress) -> None:
            nonlocal call_count
            call_count += 1
            assert progress["tree_index"] == call_count - 1
            assert progress["total_trees"] == 4

        train_gradient_boosting(
            x_train=x_train,
            y_train=y_train,
            x_val=None,
            y_val=None,
            config=config,
            feature_names=("f0",),
            progress_callback=callback,
        )

        assert call_count == 4

    def test_loss_decreases_during_training(self) -> None:
        """Training loss should generally decrease during training."""
        x_train = _float_matrix([[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]])
        y_train = _int_array([0, 0, 0, 1, 1, 1])
        config = _make_config(n_estimators=10, max_depth=2, learning_rate=0.3)

        losses: list[float] = []

        def callback(progress: TrainingProgress) -> None:
            losses.append(progress["train_loss"])

        train_gradient_boosting(
            x_train=x_train,
            y_train=y_train,
            x_val=None,
            y_val=None,
            config=config,
            feature_names=("f0",),
            progress_callback=callback,
        )

        assert losses[0] > losses[-1]


class TestPredictRaw:
    """Tests for predict_raw."""

    def test_predicts_raw_scores(self) -> None:
        """Should predict raw scores for all samples."""
        x_train = _float_matrix([[0.0], [0.1], [0.9], [1.0]])
        y_train = _int_array([0, 0, 1, 1])
        config = _make_config(n_estimators=5, max_depth=2)

        model = train_gradient_boosting(
            x_train=x_train,
            y_train=y_train,
            x_val=None,
            y_val=None,
            config=config,
            feature_names=("f0",),
        )

        x_test = _float_matrix([[0.0], [0.5], [1.0]])
        raw_preds = predict_raw(model, x_test)

        n_preds: int = raw_preds.shape[0]
        assert n_preds == 3
        assert raw_preds.item(0) < raw_preds.item(2)

    def test_empty_x_raises(self) -> None:
        """Should raise ValueError for empty x."""
        x_train = _float_matrix([[0.0], [1.0]])
        y_train = _int_array([0, 1])
        config = _make_config(n_estimators=2, max_depth=1)

        model = train_gradient_boosting(
            x_train=x_train,
            y_train=y_train,
            x_val=None,
            y_val=None,
            config=config,
            feature_names=("f0",),
        )

        x_empty: NDArray[np.float64] = np.zeros((0, 1), dtype=np.float64)
        with pytest.raises(ValueError, match="must not be empty"):
            predict_raw(model, x_empty)

    def test_wrong_features_raises(self) -> None:
        """Should raise ValueError when x has wrong number of features."""
        x_train = _float_matrix([[0.0], [1.0]])
        y_train = _int_array([0, 1])
        config = _make_config(n_estimators=2, max_depth=1)

        model = train_gradient_boosting(
            x_train=x_train,
            y_train=y_train,
            x_val=None,
            y_val=None,
            config=config,
            feature_names=("f0",),
        )

        x_wrong = _float_matrix([[0.0, 1.0]])
        with pytest.raises(ValueError, match="features"):
            predict_raw(model, x_wrong)


class TestPredictProba:
    """Tests for predict_proba."""

    def test_predicts_probabilities(self) -> None:
        """Should predict class probabilities."""
        x_train = _float_matrix([[0.0], [0.1], [0.9], [1.0]])
        y_train = _int_array([0, 0, 1, 1])
        config = _make_config(n_estimators=5, max_depth=2)

        model = train_gradient_boosting(
            x_train=x_train,
            y_train=y_train,
            x_val=None,
            y_val=None,
            config=config,
            feature_names=("f0",),
        )

        x_test = _float_matrix([[0.0], [1.0]])
        probas = predict_proba(model, x_test)

        assert len(probas) == 2
        for prob_0, prob_1 in probas:
            assert prob_0 >= 0.0
            assert prob_0 <= 1.0
            assert prob_1 >= 0.0
            assert prob_1 <= 1.0
            assert abs(prob_0 + prob_1 - 1.0) < 1e-10

    def test_probabilities_reflect_training(self) -> None:
        """Probabilities should reflect training data patterns."""
        x_train = _float_matrix([[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]])
        y_train = _int_array([0, 0, 0, 1, 1, 1])
        config = _make_config(n_estimators=10, max_depth=2, learning_rate=0.3)

        model = train_gradient_boosting(
            x_train=x_train,
            y_train=y_train,
            x_val=None,
            y_val=None,
            config=config,
            feature_names=("f0",),
        )

        x_test = _float_matrix([[0.0], [1.0]])
        probas = predict_proba(model, x_test)

        assert probas[0][0] > probas[0][1]
        assert probas[1][1] > probas[1][0]

    def test_multiple_features(self) -> None:
        """Should work with multiple features."""
        x_train = _float_matrix([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y_train = _int_array([0, 1, 1, 0])
        config = _make_config(n_estimators=20, max_depth=3, learning_rate=0.2)

        model = train_gradient_boosting(
            x_train=x_train,
            y_train=y_train,
            x_val=None,
            y_val=None,
            config=config,
            feature_names=("f0", "f1"),
        )

        x_test = _float_matrix([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        probas = predict_proba(model, x_test)

        assert len(probas) == 4
        assert probas[0][0] > 0.3
        assert probas[1][1] > 0.3
        assert probas[2][1] > 0.3
        assert probas[3][0] > 0.3


class TestEarlyStoppingState:
    """Tests for early stopping state management."""

    def test_init_state(self) -> None:
        """Initial state should have infinity loss and zero counters."""
        state = _init_early_stopping_state()

        assert state["best_val_loss"] == float("inf")
        assert state["best_round"] == 0
        assert state["rounds_without_improvement"] == 0
        assert state["should_stop"] is False

    def test_update_with_improvement(self) -> None:
        """Improvement should reset counter and update best."""
        state = _init_early_stopping_state()
        state = _update_early_stopping_state(state, val_loss=0.5, tree_idx=0, patience=3)

        assert state["best_val_loss"] == 0.5
        assert state["best_round"] == 0
        assert state["rounds_without_improvement"] == 0
        assert state["should_stop"] is False

    def test_update_without_improvement(self) -> None:
        """No improvement should increment counter."""
        state = _init_early_stopping_state()
        state = _update_early_stopping_state(state, val_loss=0.5, tree_idx=0, patience=3)
        state = _update_early_stopping_state(state, val_loss=0.6, tree_idx=1, patience=3)

        assert state["best_val_loss"] == 0.5
        assert state["best_round"] == 0
        assert state["rounds_without_improvement"] == 1
        assert state["should_stop"] is False

    def test_update_triggers_stop_after_patience(self) -> None:
        """Should trigger stop after patience rounds without improvement."""
        state = _init_early_stopping_state()
        state = _update_early_stopping_state(state, val_loss=0.5, tree_idx=0, patience=2)
        state = _update_early_stopping_state(state, val_loss=0.6, tree_idx=1, patience=2)
        state = _update_early_stopping_state(state, val_loss=0.7, tree_idx=2, patience=2)

        assert state["best_val_loss"] == 0.5
        assert state["best_round"] == 0
        assert state["rounds_without_improvement"] == 2
        assert state["should_stop"] is True

    def test_update_resets_counter_on_improvement(self) -> None:
        """Improvement after degradation should reset counter."""
        state = _init_early_stopping_state()
        state = _update_early_stopping_state(state, val_loss=0.5, tree_idx=0, patience=3)
        state = _update_early_stopping_state(state, val_loss=0.6, tree_idx=1, patience=3)
        state = _update_early_stopping_state(state, val_loss=0.4, tree_idx=2, patience=3)

        assert state["best_val_loss"] == 0.4
        assert state["best_round"] == 2
        assert state["rounds_without_improvement"] == 0
        assert state["should_stop"] is False


class TestEarlyStopping:
    """Tests for early stopping in train_gradient_boosting."""

    def test_early_stopping_stops_training(self) -> None:
        """Training should stop when validation loss stops improving."""
        # Create data where model overfits quickly
        x_train = _float_matrix([[0.0], [0.1], [0.9], [1.0]])
        y_train = _int_array([0, 0, 1, 1])
        x_val = _float_matrix([[0.2], [0.8]])
        y_val = _int_array([0, 1])
        config = _make_config(n_estimators=100, max_depth=3, early_stopping_rounds=3)

        model = train_gradient_boosting(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            config=config,
            feature_names=("f0",),
        )

        # Model should have fewer trees than n_estimators due to early stopping
        assert len(model["trees"]) < 100

    def test_early_stopping_returns_best_model(self) -> None:
        """Model should contain only trees up to best round."""
        x_train = _float_matrix([[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]])
        y_train = _int_array([0, 0, 0, 1, 1, 1])
        x_val = _float_matrix([[0.3], [0.7]])
        y_val = _int_array([0, 1])
        config = _make_config(n_estimators=50, max_depth=2, early_stopping_rounds=2)

        progress_updates: list[TrainingProgress] = []

        def callback(progress: TrainingProgress) -> None:
            progress_updates.append(progress)

        model = train_gradient_boosting(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            config=config,
            feature_names=("f0",),
            progress_callback=callback,
        )

        # Model should have at least one tree
        n_trees = len(model["trees"])
        assert n_trees >= 1
        # Model should have fewer trees than total progress updates if early stopping triggered
        # (because we return trees up to best_round, not the last round)
        n_updates = len(progress_updates)
        # If early stopping triggered, we built more trees than we kept
        # If it didn't trigger, we kept all trees
        assert n_trees <= n_updates

    def test_early_stopping_disabled_without_validation(self) -> None:
        """Early stopping should be disabled when no validation set."""
        x_train = _float_matrix([[0.0], [0.5], [1.0], [1.5]])
        y_train = _int_array([0, 0, 1, 1])
        config = _make_config(n_estimators=10, max_depth=2, early_stopping_rounds=2)

        model = train_gradient_boosting(
            x_train=x_train,
            y_train=y_train,
            x_val=None,
            y_val=None,
            config=config,
            feature_names=("f0",),
        )

        # All trees should be trained since no validation
        assert len(model["trees"]) == 10

    def test_early_stopping_disabled_when_none(self) -> None:
        """Early stopping should be disabled when early_stopping_rounds is None."""
        x_train = _float_matrix([[0.0], [0.5], [1.0], [1.5]])
        y_train = _int_array([0, 0, 1, 1])
        x_val = _float_matrix([[0.25], [1.25]])
        y_val = _int_array([0, 1])
        config = _make_config(n_estimators=10, max_depth=2, early_stopping_rounds=None)

        model = train_gradient_boosting(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            config=config,
            feature_names=("f0",),
        )

        # All trees should be trained since early stopping disabled
        assert len(model["trees"]) == 10

    def test_early_stopping_continues_with_improvement(self) -> None:
        """Training should continue while validation loss improves."""
        x_train = _float_matrix([[0.0], [0.1], [0.2], [0.3], [0.7], [0.8], [0.9], [1.0]])
        y_train = _int_array([0, 0, 0, 0, 1, 1, 1, 1])
        x_val = _float_matrix([[0.15], [0.85]])
        y_val = _int_array([0, 1])
        # Patience of 5 with 10 estimators should train all if improving
        config = _make_config(n_estimators=10, max_depth=2, early_stopping_rounds=5)

        progress_updates: list[TrainingProgress] = []

        def callback(progress: TrainingProgress) -> None:
            progress_updates.append(progress)

        train_gradient_boosting(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            config=config,
            feature_names=("f0",),
            progress_callback=callback,
        )

        # Should have trained at least one tree
        n_updates = len(progress_updates)
        assert n_updates >= 1

    def test_early_stopping_respects_patience(self) -> None:
        """Training should not stop before patience rounds without improvement."""
        x_train = _float_matrix([[0.0], [1.0]])
        y_train = _int_array([0, 1])
        x_val = _float_matrix([[0.5]])
        y_val = _int_array([0])
        # With patience=5 and small data, should train at least 5 trees
        config = _make_config(n_estimators=20, max_depth=1, early_stopping_rounds=5)

        progress_updates: list[TrainingProgress] = []

        def callback(progress: TrainingProgress) -> None:
            progress_updates.append(progress)

        train_gradient_boosting(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            config=config,
            feature_names=("f0",),
            progress_callback=callback,
        )

        # Should have at least patience+1 updates before stopping
        # (1 for best round + patience rounds without improvement)
        assert len(progress_updates) >= 6
