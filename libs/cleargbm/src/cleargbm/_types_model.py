"""Model and training-progress type definitions for ClearGBM.

Provides the GradientBoostingModel and TrainingProgress TypedDicts with
their encode/decode functions. The training configuration lives in
``_types_config``.

This module is private (underscore prefix) — not for external use.
"""

from __future__ import annotations

from typing import TypedDict

from cleargbm._types_config import (
    GradientBoostingConfig,
    decode_gradient_boosting_config,
    encode_gradient_boosting_config,
)
from cleargbm._types_json import (
    JSONDict,
    JSONTypeError,
    _as_json_dict,
    _get_optional_float,
    _require_float,
    _require_int,
    require_non_negative_float,
    require_non_negative_int,
    require_positive_float,
    require_positive_int,
)
from cleargbm._types_tree import (
    DecisionTree,
    decode_decision_tree,
    encode_decision_tree,
)

# =============================================================================
# Model
# =============================================================================


class GradientBoostingModel(TypedDict):
    """Trained gradient boosting model.

    Everything objective-dependent — how a raw score reads, whether
    probabilities exist — is answered by ``config["objective"]``. The base
    score has two mutually exclusive spellings: ``base_prediction`` (a
    scalar) for the single-score objectives, ``class_base_predictions``
    (one log-prior per class) for ``"multiclass_softmax"``, whose trees
    are stored round-major (tree ``t`` belongs to class
    ``t % n_classes``).
    """

    trees: tuple[DecisionTree, ...]
    base_prediction: float | None
    class_base_predictions: tuple[float, ...] | None
    learning_rate: float
    feature_names: tuple[str, ...]
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
        "class_base_predictions": (
            list(model["class_base_predictions"])
            if model["class_base_predictions"] is not None
            else None
        ),
        "learning_rate": model["learning_rate"],
        "feature_names": list(model["feature_names"]),
        "config": encode_gradient_boosting_config(model["config"]),
    }


def _decode_class_base_predictions(raw: JSONDict) -> tuple[float, ...] | None:
    """Decode the optional per-class base scores.

    Args:
        raw: Raw dictionary from JSON.

    Returns:
        The scores as a tuple, or None when absent or null.

    Raises:
        JSONTypeError: If the value is not a list of numbers.
    """
    if "class_base_predictions" not in raw or raw["class_base_predictions"] is None:
        return None
    cb_raw = raw["class_base_predictions"]
    if not isinstance(cb_raw, list):
        raise JSONTypeError(
            f"class_base_predictions must be list or None, got {type(cb_raw).__name__}"
        )
    scores: list[float] = []
    for i, val in enumerate(cb_raw):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise JSONTypeError(
                f"class_base_predictions[{i}] must be float, got {type(val).__name__}"
            )
        scores.append(float(val))
    return tuple(scores)


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

    base_prediction = _get_optional_float(raw, "base_prediction")
    class_base_predictions = _decode_class_base_predictions(raw)
    if (base_prediction is None) == (class_base_predictions is None):
        raise ValueError("exactly one of base_prediction and class_base_predictions must be set")
    learning_rate = require_positive_float(_require_float(raw, "learning_rate"), "learning_rate")

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
        class_base_predictions=class_base_predictions,
        learning_rate=learning_rate,
        feature_names=tuple(feature_names),
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
    "GradientBoostingModel",
    "TrainingProgress",
    "decode_gradient_boosting_model",
    "decode_training_progress",
    "encode_gradient_boosting_model",
    "encode_training_progress",
]
