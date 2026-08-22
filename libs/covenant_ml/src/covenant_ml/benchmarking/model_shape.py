"""Leaf counting for every learner, from its serialized model structure.

Wall-clock alone cannot compare a depth-wise learner with a leaf-wise one: at
a fixed ``max_depth`` ClearGBM grows a full balanced tree while LightGBM stops
at ``num_leaves``, so the two can differ in tree size by roughly two-to-one
and a raw ratio silently conflates "slower per unit of work" with "doing more
work per tree". These functions recover the tree size so results can be
normalized by it.

Every decoder validates before counting, so a change in any library's
serialization format surfaces as a traceable error rather than a wrong number.

One module for all three learners on purpose: a counter kept beside its own
consumer drifts from its siblings, and the three answer the same question
about the same benchmark.
"""

from __future__ import annotations

from platform_core.json_utils import JSONValue, load_json_str

from .types import (
    ERR_NO_TREES,
    _require_bool,
    _require_int,
    _require_list,
    _require_mapping,
)

#: Token marking a leaf node in XGBoost's text dump, as in ``3:leaf=0.12``.
_XGB_LEAF_TOKEN = "leaf="


def mean_leaves_from_cleargbm_json(raw: str) -> float:
    """Count ClearGBM's mean leaves per tree from its exported model JSON.

    ClearGBM serializes each tree as a flat node list, where a node is a leaf
    when its ``is_leaf`` flag is set.

    Args:
        raw: JSON document from ``cleargbm.ensemble.export_model_json``.

    Returns:
        Mean leaves per tree across the ensemble.

    Raises:
        ValueError: If the document is not a model export, or the ensemble
            contains no trees.
    """
    document = _require_mapping(load_json_str(raw), "model")
    trees = _require_list(document.get("trees"), "model.trees")
    if len(trees) == 0:
        raise ValueError(f"[{ERR_NO_TREES}] Field 'model.trees' must not be empty")

    total_leaves = 0
    for tree_index, tree_value in enumerate(trees):
        tree_field = f"model.trees[{tree_index}]"
        tree = _require_mapping(tree_value, tree_field)
        nodes = _require_list(tree.get("nodes"), f"{tree_field}.nodes")
        for node_index, node_value in enumerate(nodes):
            node_field = f"{tree_field}.nodes[{node_index}]"
            node = _require_mapping(node_value, node_field)
            if _require_bool(node.get("is_leaf"), f"{node_field}.is_leaf"):
                total_leaves += 1

    return total_leaves / len(trees)


def mean_leaves_from_lightgbm_dump(dump: dict[str, JSONValue]) -> float:
    """Count LightGBM's mean leaves per tree from its dumped model.

    LightGBM reports the leaf count per tree directly, under ``tree_info``.

    Args:
        dump: Mapping from ``Booster.dump_model``.

    Returns:
        Mean leaves per tree across the ensemble.

    Raises:
        ValueError: If the dump is missing ``tree_info``, or the ensemble
            contains no trees.
    """
    tree_info = _require_list(dump.get("tree_info"), "dump.tree_info")
    if len(tree_info) == 0:
        raise ValueError(f"[{ERR_NO_TREES}] Field 'dump.tree_info' must not be empty")

    total_leaves = 0
    for tree_index, tree_value in enumerate(tree_info):
        tree_field = f"dump.tree_info[{tree_index}]"
        tree = _require_mapping(tree_value, tree_field)
        total_leaves += _require_int(tree.get("num_leaves"), f"{tree_field}.num_leaves")

    return total_leaves / len(tree_info)


def mean_leaves_from_xgb_dump(dumps: list[str]) -> float:
    """Count XGBoost's mean leaves per tree from its booster text dump.

    Read from the text dump rather than ``Booster.trees_to_dataframe``: that
    route requires ``pandas``, which this library does not depend on -- it uses
    ``polars`` -- so it cannot run in this environment. The dump describes the
    same fitted trees.

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
        leaves = tree.count(_XGB_LEAF_TOKEN)
        if leaves == 0:
            raise ValueError(f"[{ERR_NO_TREES}] XGBoost dump tree {index} contains no leaf node")
        counts.append(leaves)
    return float(sum(counts)) / len(counts)


__all__ = [
    "mean_leaves_from_cleargbm_json",
    "mean_leaves_from_lightgbm_dump",
    "mean_leaves_from_xgb_dump",
]
