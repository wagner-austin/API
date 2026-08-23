"""Tests for the ``cleargbm.ensemble_ranking`` public API.

These exercise the ranking Python boundary in front of the Rust core:
end-to-end training on a small two-query dataset, scoring through the
shared :func:`cleargbm.ensemble.predict_raw` surface, and the boundary
rejections.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from cleargbm.ensemble import predict_proba, predict_raw
from cleargbm.ensemble_ranking import train_gradient_boosting_ranking
from cleargbm.types import GradientBoostingConfig


def _make_ranking_config(
    n_estimators: int = 50,
    early_stopping_rounds: int | None = None,
) -> GradientBoostingConfig:
    """Return a minimal valid lambdarank training config."""
    return GradientBoostingConfig(
        n_estimators=n_estimators,
        max_depth=2,
        learning_rate=0.3,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        colsample_bytree=None,
        categorical_features=None,
        n_classes=None,
        lambdarank_truncation_level=4,
        goss_top_rate=None,
        goss_other_rate=None,
        max_bins=16,
        subsample=1.0,
        random_state=42,
        monotonic_constraints=None,
        reg_alpha=0.0,
        reg_lambda=0.0,
        n_jobs=1,
        early_stopping_rounds=early_stopping_rounds,
        growth_strategy="depth_wise",
        num_leaves=None,
        objective="lambdarank",
        scale_pos_weight=None,
    )


def _ranking_data() -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Two queries of four documents; feature 0 tracks the relevance."""
    x: NDArray[np.float64] = np.zeros((8, 2), dtype=np.float64)
    y: NDArray[np.int64] = np.zeros(8, dtype=np.int64)
    for i in range(8):
        label = i % 4
        x[i, 0] = float(label) + 0.1 * float(i // 4)
        y[i] = label
    group: NDArray[np.int64] = np.zeros(2, dtype=np.int64)
    group[0] = 4
    group[1] = 4
    return x, y, group


class TestTrainRanking:
    """Training learns the within-query ordering; the boundary rejects."""

    def test_scores_ascend_with_relevance_within_each_query(self) -> None:
        """The raw score orders each query's documents by their grade."""
        x, y, group = _ranking_data()
        model = train_gradient_boosting_ranking(
            x, y, group, None, None, None, _make_ranking_config(), ("f0", "f1")
        )
        scores = predict_raw(model, x)
        for query in range(2):
            base = query * 4
            for i in range(3):
                lower = float(scores.flat[base + i].item())
                higher = float(scores.flat[base + i + 1].item())
                assert lower < higher, f"query {query} misordered at {i}: {scores!r}"

    def test_train_with_weights_and_a_validation_split_runs(self) -> None:
        """The fully-populated call: weights plus the validation triple."""
        x, y, group = _ranking_data()
        weights = np.ones(8, dtype=np.float64)
        model = train_gradient_boosting_ranking(
            x,
            y,
            group,
            x,
            y,
            group,
            _make_ranking_config(n_estimators=5, early_stopping_rounds=5),
            ("f0", "f1"),
            sample_weight=weights,
        )
        assert predict_raw(model, x).shape == (8,)

    def test_a_ranking_model_refuses_predict_proba(self) -> None:
        """Raw scores are ranking keys, not log-odds."""
        x, y, group = _ranking_data()
        model = train_gradient_boosting_ranking(
            x, y, group, None, None, None, _make_ranking_config(n_estimators=3), ("f0", "f1")
        )
        with pytest.raises(ValueError, match="ranking keys"):
            predict_proba(model, x)

    def test_train_empty_x_raises_at_boundary(self) -> None:
        """The Python boundary rejects an empty matrix before Rust sees it."""
        x = np.empty((0, 2), dtype=np.float64)
        y = np.empty((0,), dtype=np.int64)
        group = np.empty((0,), dtype=np.int64)
        with pytest.raises(ValueError, match="x_train must not be empty"):
            train_gradient_boosting_ranking(
                x, y, group, None, None, None, _make_ranking_config(), ("f0", "f1")
            )

    def test_a_partial_validation_triple_is_rejected(self) -> None:
        """x_val without y_val and val_group names the triple."""
        x, y, group = _ranking_data()
        with pytest.raises(ValueError, match="together"):
            train_gradient_boosting_ranking(
                x, y, group, x, None, None, _make_ranking_config(), ("f0", "f1")
            )

    def test_bad_group_sizes_propagate_the_rust_rejection(self) -> None:
        """Group sizes that do not partition the rows are refused."""
        x, y, _ = _ranking_data()
        bad_group: NDArray[np.int64] = np.zeros(2, dtype=np.int64)
        bad_group[0] = 4
        bad_group[1] = 3
        with pytest.raises(ValueError, match="sum to 7 but there are 8 rows"):
            train_gradient_boosting_ranking(
                x, y, bad_group, None, None, None, _make_ranking_config(), ("f0", "f1")
            )
