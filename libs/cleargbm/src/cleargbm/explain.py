"""Interpretability features - rule extraction, contribution breakdown.

Built from scratch - uses only Python stdlib (no numpy).
"""

from __future__ import annotations

from operator import itemgetter

from cleargbm.losses import sigmoid
from cleargbm.tree import explain_tree_prediction
from cleargbm.types import (
    DecisionTree,
    FeatureContribution,
    FloatArray,
    GradientBoostingModel,
    PredictionExplanation,
    Rule,
    TreeNode,
    TreePredictionExplanation,
)


def explain_prediction(
    model: GradientBoostingModel,
    x_single: FloatArray,
) -> PredictionExplanation:
    """Generate full explanation for a single prediction.

    Args:
        model: Trained gradient boosting model.
        x_single: Single sample feature vector.

    Returns:
        Complete explanation with contributions from each tree.

    Raises:
        ValueError: If x_single has wrong number of features.
    """
    if len(x_single) != len(model["feature_names"]):
        raise ValueError(
            f"x_single has {len(x_single)} features but model expects {len(model['feature_names'])}"
        )

    base_prediction = model["base_prediction"]
    learning_rate = model["learning_rate"]

    # Get explanations from each tree
    tree_explanations: list[TreePredictionExplanation] = []
    for tree_idx, tree in enumerate(model["trees"]):
        explanation = explain_tree_prediction(tree, x_single, tree_idx)
        tree_explanations.append(explanation)

    # Compute feature contributions across all trees
    feature_contributions = _compute_feature_contributions(
        tree_explanations, model["feature_names"], learning_rate
    )

    # Compute final probability
    raw_prediction = base_prediction
    for expl in tree_explanations:
        raw_prediction += learning_rate * expl["prediction"]
    final_probability = sigmoid(raw_prediction)

    # Sort features by absolute contribution and take top ones
    def _get_abs_contribution(fc: FeatureContribution) -> float:
        return abs(fc["total_contribution"])

    sorted_contributions = sorted(
        feature_contributions,
        key=_get_abs_contribution,
        reverse=True,
    )

    return PredictionExplanation(
        final_probability=final_probability,
        base_prediction=base_prediction,
        tree_contributions=tuple(tree_explanations),
        top_features=tuple(sorted_contributions[:10]),
    )


def _compute_feature_contributions(
    tree_explanations: list[TreePredictionExplanation],
    feature_names: tuple[str, ...],
    learning_rate: float,
) -> list[FeatureContribution]:
    """Compute feature contributions from tree explanations.

    Args:
        tree_explanations: Explanations from each tree.
        feature_names: Names of features.
        learning_rate: Learning rate used in training.

    Returns:
        List of feature contributions.
    """
    # Aggregate by feature
    contributions: dict[int, float] = {}
    split_counts: dict[int, int] = {}

    for expl in tree_explanations:
        path = expl["path"]
        tree_pred = expl["prediction"] * learning_rate

        # Distribute tree prediction among features in path
        n_splits = len(path)
        if n_splits > 0:
            contribution_per_split = tree_pred / n_splits
            for split in path:
                feat_idx = split["feature_index"]
                contributions[feat_idx] = contributions.get(feat_idx, 0.0) + contribution_per_split
                split_counts[feat_idx] = split_counts.get(feat_idx, 0) + 1

    # Build result
    result: list[FeatureContribution] = []
    for feat_idx in range(len(feature_names)):
        result.append(
            FeatureContribution(
                feature_name=feature_names[feat_idx],
                feature_index=feat_idx,
                total_contribution=contributions.get(feat_idx, 0.0),
                n_splits=split_counts.get(feat_idx, 0),
            )
        )

    return result


def extract_rules(
    model: GradientBoostingModel,
    min_samples: int = 10,
    max_rules: int = 20,
) -> tuple[Rule, ...]:
    """Extract human-readable rules from model.

    Finds the most common/important decision paths and converts
    them to readable rule format.

    Args:
        model: Trained gradient boosting model.
        min_samples: Minimum samples a rule must cover.
        max_rules: Maximum number of rules to return.

    Returns:
        Tuple of extracted rules, sorted by importance.
    """
    rules: list[Rule] = []

    for tree in model["trees"]:
        tree_rules = _extract_rules_from_tree(tree, min_samples)
        rules.extend(tree_rules)

    # Sort by importance and take top rules
    sorted_rules = sorted(rules, key=itemgetter("importance"), reverse=True)

    return tuple(sorted_rules[:max_rules])


def _extract_rules_from_tree(
    tree: DecisionTree,
    min_samples: int,
) -> list[Rule]:
    """Extract rules from a single tree.

    Args:
        tree: Decision tree.
        min_samples: Minimum samples for a valid rule.

    Returns:
        List of rules from this tree.
    """
    rules: list[Rule] = []
    nodes = tree["nodes"]
    feature_names = tree["feature_names"]

    # Find leaf nodes and build paths to them
    for node in nodes:
        if node["is_leaf"] and node["n_samples"] >= min_samples:
            # Build path from root to this leaf
            path = _build_path_to_node(nodes, node["node_id"], feature_names)
            if path:
                rules.append(
                    Rule(
                        conditions=tuple(path),
                        prediction_contribution=node["value"],
                        n_samples=node["n_samples"],
                        importance=abs(node["value"]) * node["n_samples"],
                    )
                )

    return rules


def _build_path_to_node(
    nodes: tuple[TreeNode, ...],
    target_node_id: int,
    feature_names: tuple[str, ...],
) -> list[str]:
    """Build path of conditions from root to target node.

    Args:
        nodes: All nodes in tree.
        target_node_id: ID of target node.
        feature_names: Feature names.

    Returns:
        List of condition strings.
    """
    if target_node_id == 0:
        return []

    # Build parent mapping
    parent_map: dict[int, tuple[int, str]] = {}  # child_id -> (parent_id, direction)
    for node in nodes:
        node_id = node["node_id"]
        left_child = node["left_child"]
        right_child = node["right_child"]

        if left_child is not None:
            parent_map[left_child] = (node_id, "left")
        if right_child is not None:
            parent_map[right_child] = (node_id, "right")

    # Trace path from target to root
    conditions: list[str] = []
    current_id = target_node_id

    while current_id in parent_map:
        parent_id, direction = parent_map[current_id]
        parent_node = nodes[parent_id]

        feature_idx = parent_node["feature_index"]
        threshold = parent_node["threshold"]
        feature_name = parent_node["feature_name"]

        if feature_idx is not None and threshold is not None:
            if feature_name is not None:
                name = feature_name
            elif feature_idx < len(feature_names):
                name = feature_names[feature_idx]
            else:
                name = f"feature_{feature_idx}"

            if direction == "left":
                conditions.append(f"{name} <= {threshold:.4f}")
            else:
                conditions.append(f"{name} > {threshold:.4f}")

        current_id = parent_id

    # Reverse to get root-to-leaf order
    conditions.reverse()
    return conditions


def get_feature_importances(
    model: GradientBoostingModel,
) -> tuple[FeatureContribution, ...]:
    """Get aggregate feature importance scores.

    Importance is computed as the sum of gain from splits using each feature,
    weighted by the number of samples in each split.

    Args:
        model: Trained gradient boosting model.

    Returns:
        Feature contributions sorted by importance.
    """
    feature_names = model["feature_names"]
    n_features = len(feature_names)

    # Aggregate importance by feature
    importance: dict[int, float] = {}
    split_counts: dict[int, int] = {}

    for tree in model["trees"]:
        for node in tree["nodes"]:
            if not node["is_leaf"]:
                feature_idx = node["feature_index"]
                if feature_idx is not None:
                    n_samples = node["n_samples"]
                    importance[feature_idx] = importance.get(feature_idx, 0.0) + n_samples
                    split_counts[feature_idx] = split_counts.get(feature_idx, 0) + 1

    # Normalize by total
    total = sum(importance.values())
    if total > 0:
        for idx in importance:
            importance[idx] /= total

    # Build result
    result: list[FeatureContribution] = []
    for feat_idx in range(n_features):
        result.append(
            FeatureContribution(
                feature_name=feature_names[feat_idx],
                feature_index=feat_idx,
                total_contribution=importance.get(feat_idx, 0.0),
                n_splits=split_counts.get(feat_idx, 0),
            )
        )

    # Sort by contribution
    result.sort(key=lambda x: x["total_contribution"], reverse=True)

    return tuple(result)


def get_feature_interactions(
    model: GradientBoostingModel,
    top_k: int = 10,
) -> tuple[tuple[str, str, int], ...]:
    """Detect feature co-occurrences in decision paths.

    Args:
        model: Trained gradient boosting model.
        top_k: Number of top interactions to return.

    Returns:
        Tuple of (feature1, feature2, count) sorted by count.
    """
    feature_names = model["feature_names"]
    interactions: dict[tuple[int, int], int] = {}

    for tree in model["trees"]:
        # For each leaf, find which features appear in the path
        for node in tree["nodes"]:
            if node["is_leaf"]:
                path_features = _get_path_features(tree["nodes"], node["node_id"])
                # Count pairs
                for i, feat1 in enumerate(path_features):
                    for feat2 in path_features[i + 1 :]:
                        key = (min(feat1, feat2), max(feat1, feat2))
                        interactions[key] = interactions.get(key, 0) + 1

    # Sort by count and take top k
    sorted_interactions = sorted(interactions.items(), key=itemgetter(1), reverse=True)

    result: list[tuple[str, str, int]] = []
    for (feat1, feat2), count in sorted_interactions[:top_k]:
        result.append((feature_names[feat1], feature_names[feat2], count))

    return tuple(result)


def _get_path_features(
    nodes: tuple[TreeNode, ...],
    target_node_id: int,
) -> list[int]:
    """Get features used in path from root to target node.

    Args:
        nodes: All nodes in tree.
        target_node_id: ID of target node.

    Returns:
        List of feature indices in path.
    """
    if target_node_id == 0:
        return []

    # Build parent mapping
    parent_map: dict[int, int] = {}
    for node in nodes:
        node_id = node["node_id"]
        left_child = node["left_child"]
        right_child = node["right_child"]

        if left_child is not None:
            parent_map[left_child] = node_id
        if right_child is not None:
            parent_map[right_child] = node_id

    # Trace path from target to root
    features: list[int] = []
    current_id = target_node_id

    while current_id in parent_map:
        parent_id = parent_map[current_id]
        parent_node = nodes[parent_id]
        feature_idx = parent_node["feature_index"]
        if feature_idx is not None:
            features.append(feature_idx)
        current_id = parent_id

    return features


__all__ = [
    "explain_prediction",
    "extract_rules",
    "get_feature_importances",
    "get_feature_interactions",
]
