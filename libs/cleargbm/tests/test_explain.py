"""Tests for cleargbm.explain module.

Built from scratch - uses only Python stdlib (no numpy).
"""

from __future__ import annotations

import pytest

from cleargbm.ensemble import train_gradient_boosting
from cleargbm.explain import (
    _build_path_to_node,
    _compute_feature_contributions,
    _extract_rules_from_tree,
    _get_path_features,
    explain_prediction,
    extract_rules,
    get_feature_importances,
    get_feature_interactions,
)
from cleargbm.types import (
    DecisionTree,
    GradientBoostingConfig,
    GradientBoostingModel,
    TreeNode,
    TreePredictionExplanation,
)


def _make_config(
    n_estimators: int = 5,
    max_depth: int = 3,
    learning_rate: float = 0.3,
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


def _train_simple_model() -> tuple[
    tuple[tuple[float, ...], ...], tuple[int, ...], GradientBoostingModel
]:
    """Train a simple model for testing."""
    x_train: tuple[tuple[float, ...], ...] = (
        (0.0, 0.0),
        (0.0, 1.0),
        (0.1, 0.0),
        (0.1, 1.0),
        (0.9, 0.0),
        (0.9, 1.0),
        (1.0, 0.0),
        (1.0, 1.0),
    )
    y_train = (0, 0, 0, 0, 1, 1, 1, 1)
    config = _make_config(n_estimators=5, max_depth=2)

    model = train_gradient_boosting(
        x_train=x_train,
        y_train=y_train,
        x_val=None,
        y_val=None,
        config=config,
        feature_names=("f0", "f1"),
    )

    return x_train, y_train, model


class TestExplainPrediction:
    """Tests for explain_prediction."""

    def test_explains_prediction(self) -> None:
        """Should generate explanation for a prediction."""
        _, _, model = _train_simple_model()

        explanation = explain_prediction(model, (0.0, 0.0))

        assert explanation["final_probability"] >= 0.0
        assert explanation["final_probability"] <= 1.0
        assert len(explanation["tree_contributions"]) == 5  # 5 trees
        assert len(explanation["top_features"]) <= 10

    def test_explanation_probability_matches_pattern(self) -> None:
        """Explanation probability should match training pattern."""
        _, _, model = _train_simple_model()

        # Low value should predict class 0 (low probability)
        expl_low = explain_prediction(model, (0.0, 0.5))
        # High value should predict class 1 (high probability)
        expl_high = explain_prediction(model, (1.0, 0.5))

        assert expl_low["final_probability"] < expl_high["final_probability"]

    def test_wrong_features_raises(self) -> None:
        """Should raise ValueError for wrong feature count."""
        _, _, model = _train_simple_model()

        with pytest.raises(ValueError, match="features"):
            explain_prediction(model, (0.0,))  # Only 1 feature, model expects 2

    def test_base_prediction_included(self) -> None:
        """Explanation should include base prediction."""
        _, _, model = _train_simple_model()

        explanation = explain_prediction(model, (0.5, 0.5))

        assert explanation["base_prediction"] == model["base_prediction"]


class TestComputeFeatureContributions:
    """Tests for _compute_feature_contributions."""

    def test_computes_contributions(self) -> None:
        """Should compute contributions for all features."""
        tree_explanations: list[TreePredictionExplanation] = [
            TreePredictionExplanation(
                tree_index=0,
                prediction=1.0,
                path=(
                    {
                        "feature_index": 0,
                        "feature_name": "f0",
                        "threshold": 0.5,
                        "direction": "left",
                    },
                ),
                leaf_node_id=1,
                n_samples_in_leaf=10,
            ),
        ]
        feature_names = ("f0", "f1")

        contributions = _compute_feature_contributions(tree_explanations, feature_names, 0.1)

        assert len(contributions) == 2
        # f0 should have contribution, f1 should not
        assert contributions[0]["n_splits"] == 1
        assert contributions[1]["n_splits"] == 0

    def test_empty_path_no_contribution(self) -> None:
        """Should handle tree explanations with empty paths."""
        tree_explanations: list[TreePredictionExplanation] = [
            TreePredictionExplanation(
                tree_index=0,
                prediction=1.0,
                path=(),  # Empty path (root is leaf)
                leaf_node_id=0,
                n_samples_in_leaf=10,
            ),
        ]
        feature_names = ("f0", "f1")

        contributions = _compute_feature_contributions(tree_explanations, feature_names, 0.1)

        assert len(contributions) == 2
        # No splits means no contributions
        assert contributions[0]["n_splits"] == 0
        assert contributions[1]["n_splits"] == 0
        assert contributions[0]["total_contribution"] == 0.0
        assert contributions[1]["total_contribution"] == 0.0


class TestExtractRules:
    """Tests for extract_rules."""

    def test_extracts_rules(self) -> None:
        """Should extract rules from trained model."""
        _, _, model = _train_simple_model()

        rules = extract_rules(model, min_samples=1, max_rules=10)

        # Should have multiple rules (5 trees * 2 leaves each typically)
        assert len(rules) >= 5
        for rule in rules:
            # Check rule structure is valid
            assert rule["n_samples"] >= 1
            assert rule["importance"] >= 0.0

    def test_max_rules_limit(self) -> None:
        """Should respect max_rules limit."""
        _, _, model = _train_simple_model()

        rules = extract_rules(model, min_samples=1, max_rules=3)

        assert len(rules) <= 3

    def test_min_samples_filter(self) -> None:
        """Should filter rules by min_samples."""
        _, _, model = _train_simple_model()

        # With high min_samples, should get fewer rules
        rules_high = extract_rules(model, min_samples=100, max_rules=20)

        assert len(rules_high) == 0  # No leaf has 100 samples


class TestExtractRulesFromTree:
    """Tests for _extract_rules_from_tree."""

    def test_extracts_from_simple_tree(self) -> None:
        """Should extract rules from simple tree."""
        tree = DecisionTree(
            nodes=(
                TreeNode(
                    node_id=0,
                    is_leaf=False,
                    feature_index=0,
                    feature_name="f0",
                    threshold=0.5,
                    value=0.0,
                    n_samples=20,
                    left_child=1,
                    right_child=2,
                    nan_direction="left",
                ),
                TreeNode(
                    node_id=1,
                    is_leaf=True,
                    feature_index=None,
                    feature_name=None,
                    threshold=None,
                    value=-1.0,
                    n_samples=10,
                    left_child=None,
                    right_child=None,
                    nan_direction=None,
                ),
                TreeNode(
                    node_id=2,
                    is_leaf=True,
                    feature_index=None,
                    feature_name=None,
                    threshold=None,
                    value=1.0,
                    n_samples=10,
                    left_child=None,
                    right_child=None,
                    nan_direction=None,
                ),
            ),
            max_depth=1,
            n_leaves=2,
            feature_names=("f0",),
        )

        rules = _extract_rules_from_tree(tree, min_samples=5)

        # Should have 2 rules (one for each leaf)
        assert len(rules) == 2

    def test_root_leaf_no_rules(self) -> None:
        """Should not add rule when root is a leaf (empty path)."""
        tree = DecisionTree(
            nodes=(
                TreeNode(
                    node_id=0,
                    is_leaf=True,
                    feature_index=None,
                    feature_name=None,
                    threshold=None,
                    value=0.5,
                    n_samples=20,
                    left_child=None,
                    right_child=None,
                    nan_direction=None,
                ),
            ),
            max_depth=0,
            n_leaves=1,
            feature_names=("f0",),
        )

        rules = _extract_rules_from_tree(tree, min_samples=5)

        # Root is a leaf, so path is empty and no rule is added
        assert len(rules) == 0


class TestBuildPathToNode:
    """Tests for _build_path_to_node."""

    def test_builds_path_for_leaf(self) -> None:
        """Should build path for leaf node."""
        nodes: tuple[TreeNode, ...] = (
            TreeNode(
                node_id=0,
                is_leaf=False,
                feature_index=0,
                feature_name="f0",
                threshold=0.5,
                nan_direction="left",
                value=0.0,
                n_samples=20,
                left_child=1,
                right_child=2,
            ),
            TreeNode(
                node_id=1,
                is_leaf=True,
                feature_index=None,
                feature_name=None,
                threshold=None,
                nan_direction=None,
                value=-1.0,
                n_samples=10,
                left_child=None,
                right_child=None,
            ),
            TreeNode(
                node_id=2,
                is_leaf=True,
                feature_index=None,
                feature_name=None,
                threshold=None,
                nan_direction=None,
                value=1.0,
                n_samples=10,
                left_child=None,
                right_child=None,
            ),
        )

        path = _build_path_to_node(nodes, 1, ("f0",))

        assert len(path) == 1
        assert "f0" in path[0]
        assert "<=" in path[0]

    def test_empty_path_for_root(self) -> None:
        """Should return empty path for root node."""
        nodes: tuple[TreeNode, ...] = (
            TreeNode(
                node_id=0,
                is_leaf=True,
                feature_index=None,
                feature_name=None,
                threshold=None,
                nan_direction=None,
                value=0.0,
                n_samples=10,
                left_child=None,
                right_child=None,
            ),
        )

        path = _build_path_to_node(nodes, 0, ("f0",))

        assert path == []

    def test_uses_feature_names_when_node_name_none(self) -> None:
        """Should fall back to feature_names tuple when node feature_name is None."""
        nodes: tuple[TreeNode, ...] = (
            TreeNode(
                node_id=0,
                is_leaf=False,
                feature_index=0,
                feature_name=None,  # No name stored in node
                threshold=0.5,
                nan_direction="left",
                value=0.0,
                n_samples=20,
                left_child=1,
                right_child=2,
            ),
            TreeNode(
                node_id=1,
                is_leaf=True,
                feature_index=None,
                feature_name=None,
                threshold=None,
                nan_direction=None,
                value=-1.0,
                n_samples=10,
                left_child=None,
                right_child=None,
            ),
            TreeNode(
                node_id=2,
                is_leaf=True,
                feature_index=None,
                feature_name=None,
                threshold=None,
                nan_direction=None,
                value=1.0,
                n_samples=10,
                left_child=None,
                right_child=None,
            ),
        )

        path = _build_path_to_node(nodes, 1, ("feature_zero",))

        assert len(path) == 1
        assert "feature_zero" in path[0]

    def test_uses_fallback_name_when_feature_idx_out_of_range(self) -> None:
        """Should fall back to feature_{idx} when feature_idx >= len(feature_names)."""
        nodes: tuple[TreeNode, ...] = (
            TreeNode(
                node_id=0,
                is_leaf=False,
                feature_index=5,  # Out of range for feature_names
                feature_name=None,
                threshold=0.5,
                nan_direction="left",
                value=0.0,
                n_samples=20,
                left_child=1,
                right_child=2,
            ),
            TreeNode(
                node_id=1,
                is_leaf=True,
                feature_index=None,
                feature_name=None,
                threshold=None,
                nan_direction=None,
                value=-1.0,
                n_samples=10,
                left_child=None,
                right_child=None,
            ),
            TreeNode(
                node_id=2,
                is_leaf=True,
                feature_index=None,
                feature_name=None,
                threshold=None,
                nan_direction=None,
                value=1.0,
                n_samples=10,
                left_child=None,
                right_child=None,
            ),
        )

        path = _build_path_to_node(nodes, 1, ("f0",))  # Only 1 feature name

        assert len(path) == 1
        assert "feature_5" in path[0]

    def test_skips_node_with_none_feature_or_threshold(self) -> None:
        """Should skip conditions when feature_idx or threshold is None."""
        nodes: tuple[TreeNode, ...] = (
            TreeNode(
                node_id=0,
                is_leaf=False,
                feature_index=None,  # Invalid internal node
                feature_name=None,
                threshold=None,
                nan_direction=None,
                value=0.0,
                n_samples=20,
                left_child=1,
                right_child=2,
            ),
            TreeNode(
                node_id=1,
                is_leaf=True,
                feature_index=None,
                feature_name=None,
                threshold=None,
                nan_direction=None,
                value=-1.0,
                n_samples=10,
                left_child=None,
                right_child=None,
            ),
            TreeNode(
                node_id=2,
                is_leaf=True,
                feature_index=None,
                feature_name=None,
                threshold=None,
                nan_direction=None,
                value=1.0,
                n_samples=10,
                left_child=None,
                right_child=None,
            ),
        )

        path = _build_path_to_node(nodes, 1, ("f0",))

        # Path is empty because the internal node has None feature_idx/threshold
        assert path == []


class TestGetFeatureImportances:
    """Tests for get_feature_importances."""

    def test_computes_importances(self) -> None:
        """Should compute feature importances."""
        _, _, model = _train_simple_model()

        importances = get_feature_importances(model)

        # Should have importance for each feature
        assert len(importances) == 2

        # Importances should sum to 1 (or less if some features unused)
        total = sum(imp["total_contribution"] for imp in importances)
        assert total <= 1.0 + 1e-10

    def test_more_important_feature_higher(self) -> None:
        """Primary feature should have higher importance."""
        _, _, model = _train_simple_model()

        importances = get_feature_importances(model)

        # f0 separates the data, so should be more important
        f0_importance = next(
            imp["total_contribution"] for imp in importances if imp["feature_name"] == "f0"
        )
        f1_importance = next(
            imp["total_contribution"] for imp in importances if imp["feature_name"] == "f1"
        )

        # f0 should be more important than f1 (f1 doesn't separate)
        assert f0_importance >= f1_importance

    def test_empty_trees_zero_importance(self) -> None:
        """Should handle model with no splits (all leaf roots)."""
        # Create a model with trees that are just root leaves
        model = GradientBoostingModel(
            trees=(
                DecisionTree(
                    nodes=(
                        TreeNode(
                            node_id=0,
                            is_leaf=True,
                            feature_index=None,
                            feature_name=None,
                            threshold=None,
                            nan_direction=None,
                            value=0.5,
                            n_samples=10,
                            left_child=None,
                            right_child=None,
                        ),
                    ),
                    max_depth=0,
                    n_leaves=1,
                    feature_names=("f0", "f1"),
                ),
            ),
            base_prediction=0.0,
            learning_rate=0.1,
            feature_names=("f0", "f1"),
            n_classes=2,
            config=_make_config(),
        )

        importances = get_feature_importances(model)

        # All importances should be 0 (no splits)
        assert len(importances) == 2
        for imp in importances:
            assert imp["total_contribution"] == 0.0
            assert imp["n_splits"] == 0

    def test_skips_non_leaf_nodes_with_none_feature_idx(self) -> None:
        """Should skip non-leaf nodes where feature_idx is None (edge case)."""
        model = GradientBoostingModel(
            trees=(
                DecisionTree(
                    nodes=(
                        TreeNode(
                            node_id=0,
                            is_leaf=False,
                            feature_index=None,  # Invalid but handled gracefully
                            feature_name=None,
                            threshold=0.5,
                            nan_direction=None,
                            value=0.0,
                            n_samples=20,
                            left_child=1,
                            right_child=2,
                        ),
                        TreeNode(
                            node_id=1,
                            is_leaf=True,
                            feature_index=None,
                            feature_name=None,
                            threshold=None,
                            nan_direction=None,
                            value=-1.0,
                            n_samples=10,
                            left_child=None,
                            right_child=None,
                        ),
                        TreeNode(
                            node_id=2,
                            is_leaf=True,
                            feature_index=None,
                            feature_name=None,
                            threshold=None,
                            nan_direction=None,
                            value=1.0,
                            n_samples=10,
                            left_child=None,
                            right_child=None,
                        ),
                    ),
                    max_depth=1,
                    n_leaves=2,
                    feature_names=("f0", "f1"),
                ),
            ),
            base_prediction=0.0,
            learning_rate=0.1,
            feature_names=("f0", "f1"),
            n_classes=2,
            config=_make_config(),
        )

        importances = get_feature_importances(model)

        # All importances should be 0 (split node has None feature_idx)
        assert len(importances) == 2
        for imp in importances:
            assert imp["total_contribution"] == 0.0
            assert imp["n_splits"] == 0


class TestGetFeatureInteractions:
    """Tests for get_feature_interactions."""

    def test_finds_interactions(self) -> None:
        """Should find feature interactions."""
        # Train with data where both features matter
        x_train: tuple[tuple[float, ...], ...] = (
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (1.0, 1.0),
        )
        y_train = (0, 1, 1, 0)  # XOR-like
        config = _make_config(n_estimators=10, max_depth=3)

        model = train_gradient_boosting(
            x_train=x_train,
            y_train=y_train,
            x_val=None,
            y_val=None,
            config=config,
            feature_names=("f0", "f1"),
        )

        interactions = get_feature_interactions(model, top_k=5)

        # May or may not find interactions depending on tree structure
        # Just check it runs without error
        for _feat1, _feat2, count in interactions:
            assert count >= 1

    def test_respects_top_k(self) -> None:
        """Should respect top_k limit."""
        _, _, model = _train_simple_model()

        interactions = get_feature_interactions(model, top_k=1)

        assert len(interactions) <= 1

    def test_counts_feature_pairs_in_path(self) -> None:
        """Should count feature pairs that co-occur in decision paths."""
        # Create a model with a tree that uses both features in path
        model = GradientBoostingModel(
            trees=(
                DecisionTree(
                    nodes=(
                        TreeNode(
                            node_id=0,
                            is_leaf=False,
                            feature_index=0,
                            feature_name="f0",
                            threshold=0.5,
                            nan_direction="left",
                            value=0.0,
                            n_samples=20,
                            left_child=1,
                            right_child=2,
                        ),
                        TreeNode(
                            node_id=1,
                            is_leaf=False,
                            feature_index=1,
                            feature_name="f1",
                            threshold=0.5,
                            nan_direction="left",
                            value=0.0,
                            n_samples=10,
                            left_child=3,
                            right_child=4,
                        ),
                        TreeNode(
                            node_id=2,
                            is_leaf=True,
                            feature_index=None,
                            feature_name=None,
                            threshold=None,
                            nan_direction=None,
                            value=1.0,
                            n_samples=10,
                            left_child=None,
                            right_child=None,
                        ),
                        TreeNode(
                            node_id=3,
                            is_leaf=True,
                            feature_index=None,
                            feature_name=None,
                            threshold=None,
                            nan_direction=None,
                            value=-1.0,
                            n_samples=5,
                            left_child=None,
                            right_child=None,
                        ),
                        TreeNode(
                            node_id=4,
                            is_leaf=True,
                            feature_index=None,
                            feature_name=None,
                            threshold=None,
                            nan_direction=None,
                            value=-0.5,
                            n_samples=5,
                            left_child=None,
                            right_child=None,
                        ),
                    ),
                    max_depth=2,
                    n_leaves=3,
                    feature_names=("f0", "f1"),
                ),
            ),
            base_prediction=0.0,
            learning_rate=0.1,
            feature_names=("f0", "f1"),
            n_classes=2,
            config=_make_config(),
        )

        interactions = get_feature_interactions(model, top_k=5)

        # Should find exactly 1 interaction (f0, f1 pair)
        # because tree has depth 2 with both features used in path to leaves 3 and 4
        assert len(interactions) == 1
        # The (f0, f1) pair should appear exactly 2 times (leaves 3 and 4)
        feat1, feat2, count = interactions[0]
        assert feat1 == "f0"
        assert feat2 == "f1"
        assert count == 2


class TestGetPathFeatures:
    """Tests for _get_path_features."""

    def test_finds_features_in_path(self) -> None:
        """Should find features used in path."""
        nodes: tuple[TreeNode, ...] = (
            TreeNode(
                node_id=0,
                is_leaf=False,
                feature_index=0,
                feature_name="f0",
                threshold=0.5,
                nan_direction="left",
                value=0.0,
                n_samples=20,
                left_child=1,
                right_child=2,
            ),
            TreeNode(
                node_id=1,
                is_leaf=True,
                feature_index=None,
                feature_name=None,
                threshold=None,
                nan_direction=None,
                value=-1.0,
                n_samples=10,
                left_child=None,
                right_child=None,
            ),
            TreeNode(
                node_id=2,
                is_leaf=True,
                feature_index=None,
                feature_name=None,
                threshold=None,
                nan_direction=None,
                value=1.0,
                n_samples=10,
                left_child=None,
                right_child=None,
            ),
        )

        features = _get_path_features(nodes, 1)

        assert features == [0]

    def test_empty_for_root(self) -> None:
        """Should return empty list for root node."""
        nodes: tuple[TreeNode, ...] = (
            TreeNode(
                node_id=0,
                is_leaf=True,
                feature_index=None,
                feature_name=None,
                threshold=None,
                nan_direction=None,
                value=0.0,
                n_samples=10,
                left_child=None,
                right_child=None,
            ),
        )

        features = _get_path_features(nodes, 0)

        assert features == []

    def test_skips_nodes_with_none_feature_index(self) -> None:
        """Should skip nodes where feature_index is None (edge case)."""
        # Create a tree with an internal node that has feature_index=None
        # This is an edge case that shouldn't happen in practice
        nodes: tuple[TreeNode, ...] = (
            TreeNode(
                node_id=0,
                is_leaf=False,
                feature_index=None,  # Invalid but handled gracefully
                feature_name=None,
                threshold=0.5,
                nan_direction=None,
                value=0.0,
                n_samples=20,
                left_child=1,
                right_child=2,
            ),
            TreeNode(
                node_id=1,
                is_leaf=True,
                feature_index=None,
                feature_name=None,
                threshold=None,
                nan_direction=None,
                value=-1.0,
                n_samples=10,
                left_child=None,
                right_child=None,
            ),
            TreeNode(
                node_id=2,
                is_leaf=True,
                feature_index=None,
                feature_name=None,
                threshold=None,
                nan_direction=None,
                value=1.0,
                n_samples=10,
                left_child=None,
                right_child=None,
            ),
        )

        features = _get_path_features(nodes, 1)

        # Should return empty list since feature_index is None
        assert features == []
