"""Tests for the experiment's scorer, driven against real scikit-learn."""

from __future__ import annotations

from covenant_ml.growth_policy.factory import make_metrics

from .numeric import floats, ints


class TestSklearnMetrics:
    """Each metric delegates to the right scikit-learn function."""

    def test_auc_roc_is_one_for_a_perfect_ranking(self) -> None:
        """A perfectly ordered score should give an AUC of exactly one."""
        metrics = make_metrics()
        labels = ints([0, 0, 1, 1])
        scores = floats([0.1, 0.2, 0.8, 0.9])

        assert metrics.auc_roc(labels, scores) == 1.0

    def test_auc_roc_is_a_half_for_a_constant_score(self) -> None:
        """A score carrying no information should sit at chance."""
        metrics = make_metrics()
        labels = ints([0, 1, 0, 1])
        scores = floats([0.5, 0.5, 0.5, 0.5])

        assert metrics.auc_roc(labels, scores) == 0.5

    def test_auc_pr_is_one_for_a_perfect_ranking(self) -> None:
        """A perfectly ordered score should give an average precision of one."""
        metrics = make_metrics()
        labels = ints([0, 0, 1, 1])
        scores = floats([0.1, 0.2, 0.8, 0.9])

        assert metrics.auc_pr(labels, scores) == 1.0

    def test_log_loss_rewards_confident_correct_predictions(self) -> None:
        """A confident correct prediction should score below a hedged one."""
        metrics = make_metrics()
        labels = ints([1, 1])

        confident = metrics.log_loss(labels, floats([0.99, 0.99]))
        hedged = metrics.log_loss(labels, floats([0.6, 0.6]))

        assert confident < hedged

    def test_log_loss_scores_a_single_class_fold(self) -> None:
        """Passing the label set is what lets a one-class fold score at all."""
        metrics = make_metrics()
        labels = ints([1, 1, 1])

        value = metrics.log_loss(labels, floats([0.8, 0.8, 0.8]))

        assert value > 0.0
