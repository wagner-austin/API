"""Tests for cleargbm.ensemble module.

Built from scratch - uses only Python stdlib (no numpy).
"""

from __future__ import annotations

import pytest

from cleargbm.ensemble import (
    _add_tree_predictions,
    _compute_loss,
    predict_proba,
    predict_raw,
    train_gradient_boosting,
)
from cleargbm.types import GradientBoostingConfig, TrainingProgress


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
    )


class TestComputeLoss:
    """Tests for _compute_loss helper."""

    def test_perfect_predictions_low_loss(self) -> None:
        """Perfect predictions should have low loss."""
        y_true = (1, 1, 0, 0)
        # High raw predictions for positive, low for negative
        raw_preds = (5.0, 5.0, -5.0, -5.0)

        loss = _compute_loss(y_true, raw_preds)

        assert loss < 0.01

    def test_terrible_predictions_high_loss(self) -> None:
        """Terrible predictions should have high loss."""
        y_true = (1, 1, 0, 0)
        # Wrong direction predictions
        raw_preds = (-5.0, -5.0, 5.0, 5.0)

        loss = _compute_loss(y_true, raw_preds)

        assert loss > 4.0


class TestAddTreePredictions:
    """Tests for _add_tree_predictions helper."""

    def test_adds_scaled_predictions(self) -> None:
        """Should add scaled tree predictions to raw predictions."""
        raw_preds = (0.0, 1.0, 2.0)
        tree_preds = (1.0, 2.0, 3.0)
        learning_rate = 0.5

        result = _add_tree_predictions(raw_preds, tree_preds, learning_rate)

        assert len(result) == 3
        assert abs(result[0] - 0.5) < 1e-10  # 0.0 + 0.5 * 1.0
        assert abs(result[1] - 2.0) < 1e-10  # 1.0 + 0.5 * 2.0
        assert abs(result[2] - 3.5) < 1e-10  # 2.0 + 0.5 * 3.0


class TestTrainGradientBoosting:
    """Tests for train_gradient_boosting."""

    def test_trains_on_simple_data(self) -> None:
        """Should train successfully on simple separable data."""
        # Simple separable data
        x_train: tuple[tuple[float, ...], ...] = (
            (0.0,),
            (0.1,),
            (0.2,),
            (0.3,),
            (0.7,),
            (0.8,),
            (0.9,),
            (1.0,),
        )
        y_train = (0, 0, 0, 0, 1, 1, 1, 1)
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
        x_train: tuple[tuple[float, ...], ...] = (
            (0.0,),
            (0.1,),
            (0.8,),
            (0.9,),
        )
        y_train = (0, 0, 1, 1)
        x_val: tuple[tuple[float, ...], ...] = ((0.2,), (0.7,))
        y_val = (0, 1)
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

        # Check that val_loss was reported for each tree
        for progress in progress_updates:
            val_loss = progress["val_loss"]
            if val_loss is None:
                pytest.fail("Expected val_loss to be set")
            assert val_loss >= 0.0

    def test_empty_x_train_raises(self) -> None:
        """Should raise ValueError for empty x_train."""
        config = _make_config()

        with pytest.raises(ValueError, match="must not be empty"):
            train_gradient_boosting(
                x_train=(),
                y_train=(),
                x_val=None,
                y_val=None,
                config=config,
                feature_names=("f0",),
            )

    def test_mismatched_x_y_raises(self) -> None:
        """Should raise ValueError when x_train and y_train have different lengths."""
        config = _make_config()

        with pytest.raises(ValueError, match="same length"):
            train_gradient_boosting(
                x_train=((0.0,), (1.0,)),
                y_train=(0,),  # Only 1 label but 2 samples
                x_val=None,
                y_val=None,
                config=config,
                feature_names=("f0",),
            )

    def test_mismatched_features_raises(self) -> None:
        """Should raise ValueError when features don't match feature_names."""
        config = _make_config()

        with pytest.raises(ValueError, match="feature names"):
            train_gradient_boosting(
                x_train=((0.0, 1.0),),  # 2 features
                y_train=(0,),
                x_val=None,
                y_val=None,
                config=config,
                feature_names=("f0",),  # But only 1 name
            )

    def test_progress_callback_called(self) -> None:
        """Should call progress callback after each tree."""
        x_train: tuple[tuple[float, ...], ...] = (
            (0.0,),
            (0.5,),
            (1.0,),
            (1.5,),
        )
        y_train = (0, 0, 1, 1)
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
        x_train: tuple[tuple[float, ...], ...] = (
            (0.0,),
            (0.1,),
            (0.2,),
            (0.8,),
            (0.9,),
            (1.0,),
        )
        y_train = (0, 0, 0, 1, 1, 1)
        config = _make_config(n_estimators=10, max_depth=2, learning_rate=0.3)

        losses: list[float] = []

        def callback(progress: TrainingProgress) -> None:
            train_loss = progress["train_loss"]
            losses.append(train_loss)

        train_gradient_boosting(
            x_train=x_train,
            y_train=y_train,
            x_val=None,
            y_val=None,
            config=config,
            feature_names=("f0",),
            progress_callback=callback,
        )

        # First loss should be higher than last loss
        assert losses[0] > losses[-1]


class TestPredictRaw:
    """Tests for predict_raw."""

    def test_predicts_raw_scores(self) -> None:
        """Should predict raw scores for all samples."""
        x_train: tuple[tuple[float, ...], ...] = (
            (0.0,),
            (0.1,),
            (0.9,),
            (1.0,),
        )
        y_train = (0, 0, 1, 1)
        config = _make_config(n_estimators=5, max_depth=2)

        model = train_gradient_boosting(
            x_train=x_train,
            y_train=y_train,
            x_val=None,
            y_val=None,
            config=config,
            feature_names=("f0",),
        )

        x_test: tuple[tuple[float, ...], ...] = ((0.0,), (0.5,), (1.0,))
        raw_preds = predict_raw(model, x_test)

        assert len(raw_preds) == 3
        # Negative samples should have lower raw scores
        assert raw_preds[0] < raw_preds[2]

    def test_empty_x_raises(self) -> None:
        """Should raise ValueError for empty x."""
        x_train: tuple[tuple[float, ...], ...] = ((0.0,), (1.0,))
        y_train = (0, 1)
        config = _make_config(n_estimators=2, max_depth=1)

        model = train_gradient_boosting(
            x_train=x_train,
            y_train=y_train,
            x_val=None,
            y_val=None,
            config=config,
            feature_names=("f0",),
        )

        with pytest.raises(ValueError, match="must not be empty"):
            predict_raw(model, ())

    def test_wrong_features_raises(self) -> None:
        """Should raise ValueError when x has wrong number of features."""
        x_train: tuple[tuple[float, ...], ...] = ((0.0,), (1.0,))
        y_train = (0, 1)
        config = _make_config(n_estimators=2, max_depth=1)

        model = train_gradient_boosting(
            x_train=x_train,
            y_train=y_train,
            x_val=None,
            y_val=None,
            config=config,
            feature_names=("f0",),
        )

        with pytest.raises(ValueError, match="features"):
            predict_raw(model, ((0.0, 1.0),))  # 2 features but model expects 1


class TestPredictProba:
    """Tests for predict_proba."""

    def test_predicts_probabilities(self) -> None:
        """Should predict class probabilities."""
        x_train: tuple[tuple[float, ...], ...] = (
            (0.0,),
            (0.1,),
            (0.9,),
            (1.0,),
        )
        y_train = (0, 0, 1, 1)
        config = _make_config(n_estimators=5, max_depth=2)

        model = train_gradient_boosting(
            x_train=x_train,
            y_train=y_train,
            x_val=None,
            y_val=None,
            config=config,
            feature_names=("f0",),
        )

        x_test: tuple[tuple[float, ...], ...] = ((0.0,), (1.0,))
        probas = predict_proba(model, x_test)

        assert len(probas) == 2
        # Each row should be (prob_class_0, prob_class_1)
        for prob_0, prob_1 in probas:
            assert prob_0 >= 0.0
            assert prob_0 <= 1.0
            assert prob_1 >= 0.0
            assert prob_1 <= 1.0
            assert abs(prob_0 + prob_1 - 1.0) < 1e-10

    def test_probabilities_reflect_training(self) -> None:
        """Probabilities should reflect training data patterns."""
        x_train: tuple[tuple[float, ...], ...] = (
            (0.0,),
            (0.1,),
            (0.2,),
            (0.8,),
            (0.9,),
            (1.0,),
        )
        y_train = (0, 0, 0, 1, 1, 1)
        config = _make_config(n_estimators=10, max_depth=2, learning_rate=0.3)

        model = train_gradient_boosting(
            x_train=x_train,
            y_train=y_train,
            x_val=None,
            y_val=None,
            config=config,
            feature_names=("f0",),
        )

        # Test on extreme values
        x_test: tuple[tuple[float, ...], ...] = ((0.0,), (1.0,))
        probas = predict_proba(model, x_test)

        # Sample at x=0.0 should have high prob for class 0
        assert probas[0][0] > probas[0][1]

        # Sample at x=1.0 should have high prob for class 1
        assert probas[1][1] > probas[1][0]

    def test_multiple_features(self) -> None:
        """Should work with multiple features."""
        x_train: tuple[tuple[float, ...], ...] = (
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (1.0, 1.0),
        )
        # XOR-like pattern
        y_train = (0, 1, 1, 0)
        config = _make_config(n_estimators=20, max_depth=3, learning_rate=0.2)

        model = train_gradient_boosting(
            x_train=x_train,
            y_train=y_train,
            x_val=None,
            y_val=None,
            config=config,
            feature_names=("f0", "f1"),
        )

        x_test: tuple[tuple[float, ...], ...] = (
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (1.0, 1.0),
        )
        probas = predict_proba(model, x_test)

        assert len(probas) == 4
        # Should learn XOR pattern to some degree
        # (0,0) -> class 0, (0,1) -> class 1, (1,0) -> class 1, (1,1) -> class 0
        assert probas[0][0] > 0.3  # Some confidence for class 0
        assert probas[1][1] > 0.3  # Some confidence for class 1
        assert probas[2][1] > 0.3  # Some confidence for class 1
        assert probas[3][0] > 0.3  # Some confidence for class 0
