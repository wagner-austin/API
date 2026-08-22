"""ClearGBM config recognition and translation into cleargbm's native forms.

The backend's config surface (`ClearGBMConfig`) speaks in caller terms —
name-keyed monotonic constraints, fractional max_features — while the
cleargbm library wants per-feature tuples and integer counts. This module
owns that boundary: the type guard that recognizes a ClearGBM config, the
two shape translations, and the class-weight derivation from labels.

Strict typing only: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

from typing import TypeGuard

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

from ...types import ClassifierTrainConfig, ClearGBMConfig

_log = get_logger(__name__)


def _is_cleargbm_config(cfg: ClassifierTrainConfig) -> TypeGuard[ClearGBMConfig]:
    """Check if config is ClearGBMConfig by looking for ClearGBM-specific keys.

    Args:
        cfg: Configuration to check.

    Returns:
        True if config is ClearGBMConfig.
    """
    return (
        isinstance(cfg, dict)
        and "min_samples_split" in cfg
        and "min_samples_leaf" in cfg  # LightGBM has min_child_samples instead
    )


def _resolve_monotonic_constraints(
    constraints: dict[str, int] | None,
    feature_names: tuple[str, ...],
) -> tuple[int, ...] | None:
    """Translate name-keyed constraints into cleargbm's per-feature tuple.

    Args:
        constraints: Mapping of feature name to +1 (increasing) or -1
            (decreasing), or None for unconstrained training.
        feature_names: Resolved feature names, in column order.

    Returns:
        One int per feature (0 = unconstrained), or None when no
        constraints were given.

    Raises:
        ValueError: If a constraint names a feature that does not exist —
            silently dropping it would train a different model than the
            caller stated.
    """
    if constraints is None:
        return None
    unknown = sorted(set(constraints) - set(feature_names))
    if unknown:
        raise ValueError(f"monotonic_constraints name unknown features: {', '.join(unknown)}")
    return tuple(constraints.get(name, 0) for name in feature_names)


def _resolve_max_features(max_features: int | float | None, n_features: int) -> int | None:
    """Translate the config's max_features into cleargbm's int-or-None form.

    Args:
        max_features: None (all features), an int count, or a float
            fraction of the feature count.
        n_features: Total number of features.

    Returns:
        A feature count, or None for all features.
    """
    if max_features is None:
        return None
    if isinstance(max_features, int):
        return max_features
    return max(1, int(max_features * n_features))


def _compute_class_weight(y_train: NDArray[np.int64]) -> float:
    """Compute scale_pos_weight from training labels.

    Args:
        y_train: Training labels.

    Returns:
        Weight for positive class.
    """
    pos_mask: NDArray[np.bool_] = y_train == 1
    neg_mask: NDArray[np.bool_] = y_train == 0
    n_positive = int(np.count_nonzero(pos_mask))
    n_negative = int(np.count_nonzero(neg_mask))
    if n_positive == 0:
        raise ValueError("Training set has no positive samples")
    computed = float(n_negative) / float(n_positive)
    _log.info(
        "Auto-calculated scale_pos_weight for ClearGBM",
        extra={
            "n_positive": n_positive,
            "n_negative": n_negative,
            "scale_pos_weight": computed,
        },
    )
    return computed
