"""Model-related type definitions for ClearGBM.

Provides GradientBoostingConfig, GradientBoostingModel, and TrainingProgress
TypedDicts with their encode/decode functions.

This module is private (underscore prefix) — not for external use.
"""

from __future__ import annotations

from typing import TypedDict

from cleargbm._types_json import (
    JSONDict,
    JSONTypeError,
    _as_json_dict,
    _get_optional_float,
    _get_optional_int,
    _require_bool,
    _require_float,
    _require_int,
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
    "GradientBoostingConfig",
    "GradientBoostingModel",
    "TrainingProgress",
    "decode_gradient_boosting_config",
    "decode_gradient_boosting_model",
    "decode_training_progress",
    "encode_gradient_boosting_config",
    "encode_gradient_boosting_model",
    "encode_training_progress",
]
