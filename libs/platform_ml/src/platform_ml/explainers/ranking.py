"""Turning an array of importances into a ranked, named list.

Three copies of this existed: two inside this package -- gradient.py and
integrated_gradients.py, which sit in the same directory -- and one in
covenant_ml, which already depends on this package. The covenant_ml copy was
called ``_rank_features`` and the other two ``_rank_importances``, so a grep
for either name found two of the three.

Each also carried its own ``_get_importance_from_pair``, a one-line sort key
that exists because this package forbids lambdas in a sort under its typing
rules.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .types import FeatureImportanceScore


def _importance_of(pair: tuple[int, float]) -> float:
    """Read the importance out of an (index, importance) pair.

    A named function rather than a lambda because this package's typing rules
    refuse one in a sort key.

    Args:
        pair: Tuple of (feature index, importance score).

    Returns:
        The importance score.
    """
    return pair[1]


def rank_importances(
    feature_names: list[str],
    importances: NDArray[np.float64],
) -> list[FeatureImportanceScore]:
    """Rank features by importance, most important first.

    Args:
        feature_names: Names positionally matching ``importances``.
        importances: Importance scores, read through ``.flat`` so an array of
            shape (n_features,) and one of shape (1, n_features) both work --
            the callers produce both.

    Returns:
        One score per feature, sorted descending, each carrying its 1-based
        rank.
    """
    pairs: list[tuple[int, float]] = [
        (index, float(value.item())) for index, value in enumerate(importances.flat)
    ]
    ordered = sorted(pairs, key=_importance_of, reverse=True)
    return [
        {"name": feature_names[index], "importance": importance, "rank": rank}
        for rank, (index, importance) in enumerate(ordered, start=1)
    ]


__all__ = ["rank_importances"]
