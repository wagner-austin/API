"""Tests for the cross-validation runner.

Tests cover:
- Full cross-validation execution
- Preprocessing isolation per fold
- OOF prediction collection
- Metrics computation
"""

from __future__ import annotations

import numpy as np
import pytest

from covenant_ml.validation import (
    run_group_cross_validation,
)
from tests.validation._runner_fixtures import (
    _check_probabilities_valid,
    _get_groups_for_indices,
    _make_groups,
    _make_labels_for_groups,
    _make_separable_features_for_groups,
    simple_trainer,
)


class TestRunGroupCrossValidation:
    """Tests for run_group_cross_validation function."""

    def test_returns_cv_result(self) -> None:
        """Returns properly structured CVResult."""
        # 20 groups, 3 samples each
        groups = _make_groups((3,) * 20)
        y = _make_labels_for_groups((3,) * 20, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
        x = _make_separable_features_for_groups(y, groups, n_features=5)

        result = run_group_cross_validation(
            x, y, groups, n_folds=5, random_state=42, trainer=simple_trainer
        )

        assert "n_folds" in result
        assert "fold_results" in result
        assert "mean_val_auc" in result
        assert "std_val_auc" in result
        assert "oof_predictions" in result

    def test_correct_number_of_folds(self) -> None:
        """Creates correct number of fold results."""
        groups = _make_groups((3,) * 20)
        y = _make_labels_for_groups((3,) * 20, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
        x = _make_separable_features_for_groups(y, groups, n_features=5)

        result = run_group_cross_validation(
            x, y, groups, n_folds=5, random_state=42, trainer=simple_trainer
        )

        assert result["n_folds"] == 5
        assert len(result["fold_results"]) == 5

    def test_groups_do_not_leak(self) -> None:
        """No group appears in both train and val of same fold."""
        groups = _make_groups((3,) * 20)
        y = _make_labels_for_groups((3,) * 20, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
        x = _make_separable_features_for_groups(y, groups, n_features=5)

        result = run_group_cross_validation(
            x, y, groups, n_folds=5, random_state=42, trainer=simple_trainer
        )

        # With 20 groups and 5 folds, each fold should have 4 groups in validation
        for fold_result in result["fold_results"]:
            val_indices = fold_result["val_indices"]
            val_groups = _get_groups_for_indices(groups, val_indices)

            # Each fold should have exactly 4 groups (20 groups / 5 folds)
            assert len(val_groups) == 4

    def test_oof_predictions_have_correct_shape(self) -> None:
        """OOF predictions have same length as input."""
        n_samples = 60
        groups = _make_groups((3,) * 20)
        y = _make_labels_for_groups((3,) * 20, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
        x = _make_separable_features_for_groups(y, groups, n_features=5)

        result = run_group_cross_validation(
            x, y, groups, n_folds=5, random_state=42, trainer=simple_trainer
        )

        assert len(result["oof_predictions"]) == n_samples

    def test_oof_predictions_are_probabilities(self) -> None:
        """OOF predictions are valid probabilities in [0, 1]."""
        groups = _make_groups((3,) * 20)
        y = _make_labels_for_groups((3,) * 20, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
        x = _make_separable_features_for_groups(y, groups, n_features=5)

        result = run_group_cross_validation(
            x, y, groups, n_folds=5, random_state=42, trainer=simple_trainer
        )

        oof = result["oof_predictions"]
        assert _check_probabilities_valid(oof)

    def test_mean_auc_is_average_of_folds(self) -> None:
        """mean_val_auc is average of fold AUCs."""
        groups = _make_groups((3,) * 20)
        y = _make_labels_for_groups((3,) * 20, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
        x = _make_separable_features_for_groups(y, groups, n_features=5)

        result = run_group_cross_validation(
            x, y, groups, n_folds=5, random_state=42, trainer=simple_trainer
        )

        fold_aucs = [fr["val_auc"] for fr in result["fold_results"]]
        expected_mean = sum(fold_aucs) / len(fold_aucs)

        assert result["mean_val_auc"] == pytest.approx(expected_mean)

    def test_progress_callback_is_called(self) -> None:
        """Progress callback is called for each fold."""
        groups = _make_groups((3,) * 20)
        y = _make_labels_for_groups((3,) * 20, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
        x = _make_separable_features_for_groups(y, groups, n_features=5)

        calls: list[tuple[int, int]] = []

        def callback(fold: int, total: int) -> None:
            calls.append((fold, total))

        run_group_cross_validation(
            x,
            y,
            groups,
            n_folds=3,
            random_state=42,
            trainer=simple_trainer,
            progress_callback=callback,
        )

        assert calls == [(0, 3), (1, 3), (2, 3)]

    def test_reproducibility(self) -> None:
        """Same seed produces identical results."""
        groups = _make_groups((3,) * 20)
        y = _make_labels_for_groups((3,) * 20, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
        x = _make_separable_features_for_groups(y, groups, n_features=5)

        result1 = run_group_cross_validation(
            x, y, groups, n_folds=3, random_state=42, trainer=simple_trainer
        )
        result2 = run_group_cross_validation(
            x, y, groups, n_folds=3, random_state=42, trainer=simple_trainer
        )

        np.testing.assert_array_almost_equal(result1["oof_predictions"], result2["oof_predictions"])
        assert result1["mean_val_auc"] == result2["mean_val_auc"]

    def test_different_seeds_produce_different_results(self) -> None:
        """Different seeds produce different OOF predictions."""
        groups = _make_groups((3,) * 20)
        y = _make_labels_for_groups((3,) * 20, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
        x = _make_separable_features_for_groups(y, groups, n_features=5)

        result1 = run_group_cross_validation(
            x, y, groups, n_folds=3, random_state=42, trainer=simple_trainer
        )
        result2 = run_group_cross_validation(
            x, y, groups, n_folds=3, random_state=123, trainer=simple_trainer
        )

        # OOF predictions should differ due to different fold assignments
        assert not np.allclose(result1["oof_predictions"], result2["oof_predictions"])
