"""Training-configuration type definitions for ClearGBM.

Provides the GrowthStrategy / Objective literals, the GradientBoostingConfig
TypedDict, its field validators, and the config encode/decode pair. Model
and progress types live in ``_types_model``.

This module is private (underscore prefix) — not for external use.
"""

from __future__ import annotations

from typing import Literal, TypedDict, get_args

from cleargbm._types_json import (
    JSONDict,
    JSONTypeError,
    _get_optional_float,
    _get_optional_int,
    _require_float,
    _require_int,
    _require_str,
    require_n_jobs,
    require_non_negative_float,
    require_open_unit_float,
    require_positive_float,
    require_positive_int,
    require_unit_float,
)

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

Objective = Literal["binary_log_loss", "squared_error", "multiclass_softmax"]
"""Training objective — the loss whose gradients the trees descend.

``binary_log_loss`` is binary classification: 0/1 labels, a log-odds base
score, sigmoid probabilities. ``squared_error`` is regression: continuous
targets, a label-mean base score, raw scores that ARE the predictions.
``multiclass_softmax`` is K-class classification: integer class labels,
one log-prior base score per class, ``n_classes`` trees per boosting round,
softmax probabilities.

A closed literal for the same reason :data:`GrowthStrategy` is: a mistyped
objective fails at the Rust boundary rather than quietly training the wrong
loss.
"""

OBJECTIVES: tuple[Objective, ...] = get_args(Objective)
"""Every accepted :data:`Objective` value, for validation and iteration."""


class GradientBoostingConfig(TypedDict):
    """Configuration for gradient boosting training.

    Args:
        n_estimators: Number of boosting rounds (maximum if early stopping
            enabled).
        max_depth: Maximum tree depth.
        learning_rate: Shrinkage factor for updates.
        min_samples_split: Minimum samples to split a node.
        min_samples_leaf: Minimum samples in a leaf.
        max_features: Features each split may consider (None = all). A real
            per-split budget as of 2026-08-22; earlier configs carried the
            field but the trainer ignored it.
        colsample_bytree: Fraction of features each TREE may use (None =
            all), drawn once per boosting round; the per-split
            ``max_features`` draw then selects within the tree's set. Must
            lie strictly between 0 and 1 when set — ``None`` is the only
            spelling of "all features", so ``1.0`` is rejected rather than
            silently equivalent.
        categorical_features: Feature indices treated as categorical
            (None = every feature numeric). Strictly ascending when set —
            the one canonical spelling of a set. Values in those columns
            must be non-negative integer codes; splits partition categories
            by set membership (LightGBM's many-vs-many mechanism) rather
            than by threshold.
        n_classes: Class count, paired with the objective: an int >= 2
            under ``"multiclass_softmax"`` (each boosting round trains that
            many trees) and ``None`` under every other objective. The
            pairing itself is enforced at the Rust boundary.
        max_bins: Number of histogram bins for split finding (64-256 typical).
        subsample: Row subsampling ratio.
        random_state: Random seed.
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
        objective: The loss to train under: ``"binary_log_loss"``,
            ``"squared_error"``, or ``"multiclass_softmax"``. Required, with
            no implicit default: a run must name the loss it descends.
        scale_pos_weight: Weight applied to positive samples in the loss,
            its gradients and the base score. Paired with the objective:
            must be a finite positive float under ``"binary_log_loss"``
            (``1.0`` trains unweighted, bit-identical to the pre-weighting
            behavior) and ``None`` under every other objective — regression
            has no positive class to weight, and multiclass weighs rows via
            ``sample_weight``. As with ``num_leaves``, this layer checks the
            field's own type; the *pairing* is enforced once, at the Rust
            boundary.
    """

    n_estimators: int
    max_depth: int
    learning_rate: float
    min_samples_split: int
    min_samples_leaf: int
    max_features: int | None
    colsample_bytree: float | None
    categorical_features: tuple[int, ...] | None
    n_classes: int | None
    max_bins: int
    subsample: float
    random_state: int
    monotonic_constraints: tuple[int, ...] | None
    reg_alpha: float
    reg_lambda: float
    n_jobs: int
    early_stopping_rounds: int | None
    growth_strategy: GrowthStrategy
    num_leaves: int | None
    objective: Objective
    scale_pos_weight: float | None


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


def require_objective(value: str, name: str) -> Objective:
    """Narrow a string to an :data:`Objective`.

    Args:
        value: Candidate objective name.
        name: Field name, used in the error message.

    Returns:
        The value, narrowed to the literal type.

    Raises:
        ValueError: If ``value`` is not one of :data:`OBJECTIVES`.
    """
    for objective in OBJECTIVES:
        if value == objective:
            return objective
    raise ValueError(f"{name} must be one of {list(OBJECTIVES)}, got {value!r}")


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
        "colsample_bytree": config["colsample_bytree"],
        "categorical_features": (
            list(config["categorical_features"])
            if config["categorical_features"] is not None
            else None
        ),
        "n_classes": config["n_classes"],
        "max_bins": config["max_bins"],
        "subsample": config["subsample"],
        "random_state": config["random_state"],
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
        "objective": config["objective"],
        "scale_pos_weight": config["scale_pos_weight"],
    }


def _decode_categorical_features(raw: JSONDict) -> tuple[int, ...] | None:
    """Decode the optional categorical feature index list.

    Args:
        raw: Raw dictionary from JSON.

    Returns:
        The indices as a tuple, or None when absent or null.

    Raises:
        JSONTypeError: If the value is not a list of ints.
        ValueError: If the list is empty, holds a negative index, or is not
            strictly ascending — a set has one canonical spelling.
    """
    if "categorical_features" not in raw or raw["categorical_features"] is None:
        return None
    cf_raw = raw["categorical_features"]
    if not isinstance(cf_raw, list):
        raise JSONTypeError(
            f"categorical_features must be list or None, got {type(cf_raw).__name__}"
        )
    if not cf_raw:
        raise ValueError("categorical_features must be non-empty when set (null = all numeric)")
    indices: list[int] = []
    for i, val in enumerate(cf_raw):
        if not isinstance(val, int) or isinstance(val, bool):
            raise JSONTypeError(f"categorical_features[{i}] must be int, got {type(val).__name__}")
        if val < 0:
            raise ValueError(f"categorical_features[{i}] must be >= 0, got {val}")
        if indices and val <= indices[-1]:
            raise ValueError(
                f"categorical_features must be strictly ascending, got {val} after {indices[-1]}"
            )
        indices.append(val)
    return tuple(indices)


def _decode_monotonic_constraints(raw: JSONDict) -> tuple[int, ...] | None:
    """Decode the optional per-feature monotonic constraint list.

    Args:
        raw: Raw dictionary from JSON.

    Returns:
        The constraints as a tuple, or None when absent or null.

    Raises:
        JSONTypeError: If the value is not a list of ints.
        ValueError: If any constraint is outside {-1, 0, 1}.
    """
    if "monotonic_constraints" not in raw or raw["monotonic_constraints"] is None:
        return None
    mc_raw = raw["monotonic_constraints"]
    if not isinstance(mc_raw, list):
        raise JSONTypeError(
            f"monotonic_constraints must be list or None, got {type(mc_raw).__name__}"
        )
    mc_list: list[int] = []
    for i, val in enumerate(mc_raw):
        if not isinstance(val, int) or isinstance(val, bool):
            raise JSONTypeError(f"monotonic_constraints[{i}] must be int, got {type(val).__name__}")
        if val not in (-1, 0, 1):
            raise ValueError(f"monotonic_constraints[{i}] must be -1, 0, or 1, got {val}")
        mc_list.append(val)
    return tuple(mc_list)


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
    colsample_bytree = _get_optional_float(raw, "colsample_bytree")
    if colsample_bytree is not None:
        colsample_bytree = require_open_unit_float(colsample_bytree, "colsample_bytree")
    categorical_features = _decode_categorical_features(raw)
    n_classes = _get_optional_int(raw, "n_classes")
    if n_classes is not None and n_classes < 2:
        raise ValueError(f"n_classes must be >= 2 when set, got {n_classes}")
    max_bins = require_positive_int(_require_int(raw, "max_bins"), "max_bins")
    subsample = require_unit_float(_require_float(raw, "subsample"), "subsample")
    random_state = _require_int(raw, "random_state")

    monotonic_constraints = _decode_monotonic_constraints(raw)

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

    objective = require_objective(_require_str(raw, "objective"), "objective")

    scale_pos_weight = _get_optional_float(raw, "scale_pos_weight")
    if scale_pos_weight is not None:
        scale_pos_weight = require_positive_float(scale_pos_weight, "scale_pos_weight")

    return GradientBoostingConfig(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        colsample_bytree=colsample_bytree,
        categorical_features=categorical_features,
        n_classes=n_classes,
        max_bins=max_bins,
        subsample=subsample,
        random_state=random_state,
        monotonic_constraints=monotonic_constraints,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        n_jobs=n_jobs,
        early_stopping_rounds=early_stopping_rounds,
        growth_strategy=growth_strategy,
        num_leaves=num_leaves,
        objective=objective,
        scale_pos_weight=scale_pos_weight,
    )


__all__ = [
    "GROWTH_STRATEGIES",
    "OBJECTIVES",
    "GradientBoostingConfig",
    "GrowthStrategy",
    "Objective",
    "decode_gradient_boosting_config",
    "encode_gradient_boosting_config",
    "require_growth_strategy",
    "require_leaf_budget",
    "require_objective",
]
