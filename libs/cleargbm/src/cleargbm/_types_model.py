"""Model-related type definitions for ClearGBM.

Provides GradientBoostingConfig, GradientBoostingModel, and TrainingProgress
TypedDicts with their encode/decode functions.

This module is private (underscore prefix) — not for external use.
"""

from __future__ import annotations

from typing import Literal, TypedDict, get_args

from cleargbm._types_json import (
    JSONDict,
    JSONTypeError,
    _as_json_dict,
    _get_optional_float,
    _get_optional_int,
    _require_bool,
    _require_float,
    _require_int,
    _require_str,
    require_n_jobs,
    require_non_negative_float,
    require_non_negative_int,
    require_positive_float,
    require_positive_int,
    require_unit_float,
)
from cleargbm._types_tree import (
    DecisionTree,
    decode_decision_tree,
    encode_decision_tree,
)

# =============================================================================
# Configuration
# =============================================================================

GrowthStrategy = Literal["depth_wise", "leaf_wise"]
"""Tree growth policy — the order in which nodes are chosen for splitting.

``depth_wise`` expands every node at one depth before the next, bounded by
``max_depth``. ``leaf_wise`` repeatedly splits the highest-gain leaf, bounded
by a leaf budget. The two build different trees from identical data, so this
is an algorithm parameter and not a fallback switch.

A closed literal on purpose: variants are enumerated here and rejected at the
Rust boundary if unimplemented, so a mistyped arm name fails rather than
quietly training the default policy.
"""

GROWTH_STRATEGIES: tuple[GrowthStrategy, ...] = get_args(GrowthStrategy)
"""Every accepted :data:`GrowthStrategy` value, for validation and iteration."""


class GradientBoostingConfig(TypedDict):
    """Configuration for gradient boosting training.

    Args:
        n_estimators: Number of boosting rounds (maximum if early stopping
            enabled).
        max_depth: Maximum tree depth.
        learning_rate: Shrinkage factor for updates.
        min_samples_split: Minimum samples to split a node.
        min_samples_leaf: Minimum samples in a leaf.
        max_features: Max features per split (None = all).
        max_bins: Number of histogram bins for split finding (64-256 typical).
        subsample: Row subsampling ratio.
        random_state: Random seed.
        track_contributions: Track feature contributions.
        monotonic_constraints: Per-feature constraints (-1, 0, +1).
        reg_alpha: L1 regularization term on leaf weights (default: 0.0).
        reg_lambda: L2 regularization term on leaf weights (default: 1.0).
        n_jobs: Number of parallel workers (1 = sequential, -1 = all cores).
        early_stopping_rounds: Stop training after this many rounds without
            improvement on validation loss. None disables early stopping.
            Requires validation data to be provided during training.
        growth_strategy: Tree growth policy, ``"depth_wise"`` or
            ``"leaf_wise"``. Required, with no implicit default: a benchmark
            arm that meant to name a policy and silently got another one is
            the failure this axis exists to prevent.
        num_leaves: Leaf budget for ``"leaf_wise"`` growth. Must be an int
            >= 2 under ``"leaf_wise"`` and ``None`` under ``"depth_wise"`` —
            paired with the policy rather than ignored under the wrong one,
            so a run can never report a knob it did not honour. Best-first
            growth has no depth to bound its shape, which is why the budget
            is mandatory there.

            This layer checks the field's own type and range. The *pairing*
            against ``growth_strategy`` is enforced once, at the Rust
            boundary, so the cross-field rule has a single owner rather than
            two copies that can drift.
        scale_pos_weight: Weight applied to positive samples in the loss,
            its gradients and the base score. Must be a finite positive
            number; ``1.0`` trains unweighted (and is bit-identical to the
            pre-weighting behavior). Required, with no implicit default,
            for the same reason ``growth_strategy`` is: the field exists
            because a backend once reported a weight it never applied.
    """

    n_estimators: int
    max_depth: int
    learning_rate: float
    min_samples_split: int
    min_samples_leaf: int
    max_features: int | None
    max_bins: int
    subsample: float
    random_state: int
    track_contributions: bool
    monotonic_constraints: tuple[int, ...] | None
    reg_alpha: float
    reg_lambda: float
    n_jobs: int
    early_stopping_rounds: int | None
    growth_strategy: GrowthStrategy
    num_leaves: int | None
    scale_pos_weight: float


def require_leaf_budget(value: int, name: str) -> int:
    """Validate a leaf budget.

    Args:
        value: Candidate budget.
        name: Field name, used in the error message.

    Returns:
        The validated budget.

    Raises:
        ValueError: If ``value`` is below 2. A budget of 1 cannot describe a
            tree containing a split, so it is rejected here rather than left
            to produce a confusing error inside the builder.
    """
    if value < 2:
        raise ValueError(f"{name} must be >= 2, got {value}")
    return value


def require_growth_strategy(value: str, name: str) -> GrowthStrategy:
    """Narrow a string to a :data:`GrowthStrategy`.

    Args:
        value: Candidate policy name.
        name: Field name, used in the error message.

    Returns:
        The value, narrowed to the literal type.

    Raises:
        ValueError: If ``value`` is not one of :data:`GROWTH_STRATEGIES`.
    """
    for strategy in GROWTH_STRATEGIES:
        if value == strategy:
            return strategy
    raise ValueError(f"{name} must be one of {list(GROWTH_STRATEGIES)}, got {value!r}")


def encode_gradient_boosting_config(
    config: GradientBoostingConfig,
) -> JSONDict:
    """Encode GradientBoostingConfig to JSON-serializable dict.

    Args:
        config: Config to encode.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "n_estimators": config["n_estimators"],
        "max_depth": config["max_depth"],
        "learning_rate": config["learning_rate"],
        "min_samples_split": config["min_samples_split"],
        "min_samples_leaf": config["min_samples_leaf"],
        "max_features": config["max_features"],
        "max_bins": config["max_bins"],
        "subsample": config["subsample"],
        "random_state": config["random_state"],
        "track_contributions": config["track_contributions"],
        "monotonic_constraints": (
            list(config["monotonic_constraints"])
            if config["monotonic_constraints"] is not None
            else None
        ),
        "reg_alpha": config["reg_alpha"],
        "reg_lambda": config["reg_lambda"],
        "n_jobs": config["n_jobs"],
        "early_stopping_rounds": config["early_stopping_rounds"],
        "growth_strategy": config["growth_strategy"],
        "num_leaves": config["num_leaves"],
        "scale_pos_weight": config["scale_pos_weight"],
    }


def decode_gradient_boosting_config(
    raw: JSONDict,
) -> GradientBoostingConfig:
    """Decode raw dict to GradientBoostingConfig.

    Args:
        raw: Raw dictionary from JSON.

    Returns:
        Validated GradientBoostingConfig.

    Raises:
        KeyError: If required key is missing.
        JSONTypeError: If value has wrong type.
        ValueError: If value fails validation.
    """
    n_estimators = require_positive_int(_require_int(raw, "n_estimators"), "n_estimators")
    max_depth = require_positive_int(_require_int(raw, "max_depth"), "max_depth")
    learning_rate = require_positive_float(_require_float(raw, "learning_rate"), "learning_rate")
    min_samples_split = require_positive_int(
        _require_int(raw, "min_samples_split"), "min_samples_split"
    )
    min_samples_leaf = require_positive_int(
        _require_int(raw, "min_samples_leaf"), "min_samples_leaf"
    )
    max_features = _get_optional_int(raw, "max_features")
    if max_features is not None:
        max_features = require_positive_int(max_features, "max_features")
    max_bins = require_positive_int(_require_int(raw, "max_bins"), "max_bins")
    subsample = require_unit_float(_require_float(raw, "subsample"), "subsample")
    random_state = _require_int(raw, "random_state")
    track_contributions = _require_bool(raw, "track_contributions")

    monotonic_constraints: tuple[int, ...] | None = None
    if "monotonic_constraints" in raw and raw["monotonic_constraints"] is not None:
        mc_raw = raw["monotonic_constraints"]
        if not isinstance(mc_raw, list):
            raise JSONTypeError(
                f"monotonic_constraints must be list or None, got {type(mc_raw).__name__}"
            )
        mc_list: list[int] = []
        for i, val in enumerate(mc_raw):
            if not isinstance(val, int) or isinstance(val, bool):
                raise JSONTypeError(
                    f"monotonic_constraints[{i}] must be int, got {type(val).__name__}"
                )
            if val not in (-1, 0, 1):
                raise ValueError(f"monotonic_constraints[{i}] must be -1, 0, or 1, got {val}")
            mc_list.append(val)
        monotonic_constraints = tuple(mc_list)

    reg_alpha = require_non_negative_float(_require_float(raw, "reg_alpha"), "reg_alpha")
    reg_lambda = require_non_negative_float(_require_float(raw, "reg_lambda"), "reg_lambda")
    n_jobs = require_n_jobs(_require_int(raw, "n_jobs"), "n_jobs")

    # early_stopping_rounds: None or positive int
    early_stopping_rounds = _get_optional_int(raw, "early_stopping_rounds")
    if early_stopping_rounds is not None:
        early_stopping_rounds = require_positive_int(early_stopping_rounds, "early_stopping_rounds")

    growth_strategy = require_growth_strategy(
        _require_str(raw, "growth_strategy"), "growth_strategy"
    )

    num_leaves = _get_optional_int(raw, "num_leaves")
    if num_leaves is not None:
        num_leaves = require_leaf_budget(num_leaves, "num_leaves")

    scale_pos_weight = require_positive_float(
        _require_float(raw, "scale_pos_weight"), "scale_pos_weight"
    )

    return GradientBoostingConfig(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_bins=max_bins,
        subsample=subsample,
        random_state=random_state,
        track_contributions=track_contributions,
        monotonic_constraints=monotonic_constraints,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        n_jobs=n_jobs,
        early_stopping_rounds=early_stopping_rounds,
        growth_strategy=growth_strategy,
        num_leaves=num_leaves,
        scale_pos_weight=scale_pos_weight,
    )


# =============================================================================
# Model
# =============================================================================


class GradientBoostingModel(TypedDict):
    """Trained gradient boosting model."""

    trees: tuple[DecisionTree, ...]
    base_prediction: float
    learning_rate: float
    feature_names: tuple[str, ...]
    n_classes: int
    config: GradientBoostingConfig


def encode_gradient_boosting_model(
    model: GradientBoostingModel,
) -> JSONDict:
    """Encode GradientBoostingModel to JSON-serializable dict.

    Args:
        model: Model to encode.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "trees": [encode_decision_tree(t) for t in model["trees"]],
        "base_prediction": model["base_prediction"],
        "learning_rate": model["learning_rate"],
        "feature_names": list(model["feature_names"]),
        "n_classes": model["n_classes"],
        "config": encode_gradient_boosting_config(model["config"]),
    }


def decode_gradient_boosting_model(
    raw: JSONDict,
) -> GradientBoostingModel:
    """Decode raw dict to GradientBoostingModel.

    Args:
        raw: Raw dictionary from JSON.

    Returns:
        Validated GradientBoostingModel.

    Raises:
        KeyError: If required key is missing.
        JSONTypeError: If value has wrong type.
        ValueError: If value fails validation.
    """
    trees_raw = raw["trees"]
    if not isinstance(trees_raw, list):
        raise JSONTypeError(f"trees must be list, got {type(trees_raw).__name__}")
    trees: list[DecisionTree] = []
    for i, tree_raw in enumerate(trees_raw):
        tree_dict = _as_json_dict(tree_raw, f"trees[{i}]")
        trees.append(decode_decision_tree(tree_dict))

    base_prediction = _require_float(raw, "base_prediction")
    learning_rate = require_positive_float(_require_float(raw, "learning_rate"), "learning_rate")
    n_classes = require_positive_int(_require_int(raw, "n_classes"), "n_classes")

    feature_names_raw = raw["feature_names"]
    if not isinstance(feature_names_raw, list):
        raise JSONTypeError(f"feature_names must be list, got {type(feature_names_raw).__name__}")
    feature_names: list[str] = []
    for i, name in enumerate(feature_names_raw):
        if not isinstance(name, str):
            raise JSONTypeError(f"feature_names[{i}] must be str, got {type(name).__name__}")
        feature_names.append(name)

    config_dict = _as_json_dict(raw["config"], "config")
    config = decode_gradient_boosting_config(config_dict)

    return GradientBoostingModel(
        trees=tuple(trees),
        base_prediction=base_prediction,
        learning_rate=learning_rate,
        feature_names=tuple(feature_names),
        n_classes=n_classes,
        config=config,
    )


# =============================================================================
# Training Progress
# =============================================================================


class TrainingProgress(TypedDict):
    """Progress update during training."""

    tree_index: int
    total_trees: int
    train_loss: float
    val_loss: float | None


def encode_training_progress(progress: TrainingProgress) -> JSONDict:
    """Encode TrainingProgress to JSON-serializable dict.

    Args:
        progress: Progress to encode.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "tree_index": progress["tree_index"],
        "total_trees": progress["total_trees"],
        "train_loss": progress["train_loss"],
        "val_loss": progress["val_loss"],
    }


def decode_training_progress(raw: JSONDict) -> TrainingProgress:
    """Decode raw dict to TrainingProgress.

    Args:
        raw: Raw dictionary from JSON.

    Returns:
        Validated TrainingProgress.

    Raises:
        KeyError: If required key is missing.
        TypeError: If value has wrong type.
        ValueError: If value fails validation.
    """
    tree_index = require_non_negative_int(_require_int(raw, "tree_index"), "tree_index")
    total_trees = require_positive_int(_require_int(raw, "total_trees"), "total_trees")
    train_loss = require_non_negative_float(_require_float(raw, "train_loss"), "train_loss")
    val_loss = _get_optional_float(raw, "val_loss")
    if val_loss is not None:
        val_loss = require_non_negative_float(val_loss, "val_loss")

    return TrainingProgress(
        tree_index=tree_index,
        total_trees=total_trees,
        train_loss=train_loss,
        val_loss=val_loss,
    )


__all__ = [
    "GROWTH_STRATEGIES",
    "GradientBoostingConfig",
    "GradientBoostingModel",
    "GrowthStrategy",
    "TrainingProgress",
    "decode_gradient_boosting_config",
    "decode_gradient_boosting_model",
    "decode_training_progress",
    "encode_gradient_boosting_config",
    "encode_gradient_boosting_model",
    "encode_training_progress",
    "require_growth_strategy",
    "require_leaf_budget",
]
