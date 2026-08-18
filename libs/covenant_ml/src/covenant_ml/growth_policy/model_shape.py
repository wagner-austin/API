"""Reading tree shape out of a fitted XGBoost ensemble.

Leaf counts come from the booster's text dump rather than
``Booster.trees_to_dataframe``. The dataframe route requires ``pandas``, which
this library does not depend on -- it uses ``polars`` -- so that call cannot run
inside this package's environment. Counting leaf nodes in the dump reads the
same fitted trees.

LightGBM and ClearGBM already have leaf counters in
:mod:`covenant_ml.benchmarking.model_shape`; only XGBoost was missing one, so
only XGBoost is added here.
"""

from __future__ import annotations

from .types import ERR_NO_TREES

#: Token marking a leaf node in XGBoost's text dump, as in ``3:leaf=0.12``.
_LEAF_TOKEN = "leaf="


def mean_leaves_from_xgb_dump(dumps: list[str]) -> float:
    """Count the mean number of leaves per tree in an XGBoost text dump.

    Args:
        dumps: One text representation per boosted tree, as returned by
            ``Booster.get_dump()``.

    Returns:
        Mean leaves per tree across the ensemble.

    Raises:
        ValueError: If the ensemble holds no trees, or if any tree carries no
            leaf. Either means the dump is not the text format this reads, and
            returning a plausible number from it would put a wrong leaf count
            into a published table.
    """
    if len(dumps) == 0:
        raise ValueError(f"[{ERR_NO_TREES}] XGBoost dump contains no trees")
    counts: list[int] = []
    for index, tree in enumerate(dumps):
        leaves = tree.count(_LEAF_TOKEN)
        if leaves == 0:
            raise ValueError(f"[{ERR_NO_TREES}] XGBoost dump tree {index} contains no leaf node")
        counts.append(leaves)
    return float(sum(counts)) / len(counts)


__all__ = ["mean_leaves_from_xgb_dump"]
