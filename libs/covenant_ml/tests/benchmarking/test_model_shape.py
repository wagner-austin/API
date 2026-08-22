"""Tests for leaf counting from each learner's serialized structure."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONValue, dump_json_str

from covenant_ml.benchmarking.model_shape import (
    mean_leaves_from_cleargbm_json,
    mean_leaves_from_lightgbm_dump,
    mean_leaves_from_xgb_dump,
)
from covenant_ml.benchmarking.types import ERR_NO_TREES, ERR_NOT_LIST, ERR_NOT_MAPPING


def cleargbm_document(leaf_flags_per_tree: list[list[bool]]) -> str:
    """Build a ClearGBM model export with the given leaf layout.

    Args:
        leaf_flags_per_tree: One ``is_leaf`` flag list per tree.

    Returns:
        The serialized document.
    """
    trees: list[JSONValue] = [
        {"nodes": [{"is_leaf": flag} for flag in flags]} for flags in leaf_flags_per_tree
    ]
    return dump_json_str({"trees": trees})


def test_cleargbm_counts_only_leaf_nodes() -> None:
    document = cleargbm_document([[False, True, True]])
    assert mean_leaves_from_cleargbm_json(document) == 2.0


def test_cleargbm_averages_across_trees() -> None:
    document = cleargbm_document([[True, True], [True, True, True, True]])
    assert mean_leaves_from_cleargbm_json(document) == 3.0


def test_cleargbm_rejects_empty_ensemble() -> None:
    with pytest.raises(ValueError, match=ERR_NO_TREES):
        mean_leaves_from_cleargbm_json(dump_json_str({"trees": []}))


def test_cleargbm_rejects_non_object_document() -> None:
    with pytest.raises(ValueError, match=ERR_NOT_MAPPING):
        mean_leaves_from_cleargbm_json(dump_json_str([1, 2]))


def test_cleargbm_rejects_missing_trees() -> None:
    with pytest.raises(ValueError, match=ERR_NOT_LIST):
        mean_leaves_from_cleargbm_json(dump_json_str({"other": 1}))


def test_cleargbm_reports_the_offending_node_path() -> None:
    document = dump_json_str({"trees": [{"nodes": [{"is_leaf": "yes"}]}]})
    with pytest.raises(ValueError, match=r"model\.trees\[0\]\.nodes\[0\]\.is_leaf"):
        mean_leaves_from_cleargbm_json(document)


def test_lightgbm_averages_reported_leaf_counts() -> None:
    dump: dict[str, JSONValue] = {"tree_info": [{"num_leaves": 31}, {"num_leaves": 29}]}
    assert mean_leaves_from_lightgbm_dump(dump) == 30.0


def test_lightgbm_rejects_empty_ensemble() -> None:
    dump: dict[str, JSONValue] = {"tree_info": []}
    with pytest.raises(ValueError, match=ERR_NO_TREES):
        mean_leaves_from_lightgbm_dump(dump)


def test_lightgbm_rejects_missing_tree_info() -> None:
    dump: dict[str, JSONValue] = {"other": 1}
    with pytest.raises(ValueError, match=ERR_NOT_LIST):
        mean_leaves_from_lightgbm_dump(dump)


def test_lightgbm_reports_the_offending_tree_path() -> None:
    dump: dict[str, JSONValue] = {"tree_info": [{"num_leaves": "many"}]}
    with pytest.raises(ValueError, match=r"dump\.tree_info\[0\]\.num_leaves"):
        mean_leaves_from_lightgbm_dump(dump)


#: A two-leaf tree in the text format ``Booster.get_dump()`` emits.
_TWO_LEAF_TREE = "0:[f0<0.5] yes=1,no=2\n\t1:leaf=-0.1\n\t2:leaf=0.2\n"

#: A four-leaf tree in the same format.
_FOUR_LEAF_TREE = (
    "0:[f0<0.5] yes=1,no=2\n"
    "\t1:[f1<0.5] yes=3,no=4\n"
    "\t\t3:leaf=-0.1\n"
    "\t\t4:leaf=0.2\n"
    "\t2:[f1<1.5] yes=5,no=6\n"
    "\t\t5:leaf=0.3\n"
    "\t\t6:leaf=-0.4\n"
)


class TestMeanLeavesFromXgbDump:
    """Counting leaves across an XGBoost ensemble."""

    def test_counts_a_single_tree(self) -> None:
        """One two-leaf tree should average two."""
        assert mean_leaves_from_xgb_dump([_TWO_LEAF_TREE]) == 2.0

    def test_averages_across_trees(self) -> None:
        """A two-leaf and a four-leaf tree should average three."""
        assert mean_leaves_from_xgb_dump([_TWO_LEAF_TREE, _FOUR_LEAF_TREE]) == 3.0

    def test_rejects_an_empty_ensemble(self) -> None:
        """No trees means no mean, so it must fail rather than return zero."""
        with pytest.raises(ValueError, match=ERR_NO_TREES):
            mean_leaves_from_xgb_dump([])

    def test_rejects_a_tree_with_no_leaf(self) -> None:
        """A dump in an unexpected format must fail, naming the offending tree."""
        with pytest.raises(ValueError, match=ERR_NO_TREES) as excinfo:
            mean_leaves_from_xgb_dump([_TWO_LEAF_TREE, "0:[f0<0.5] yes=1,no=2\n"])

        assert "tree 1" in str(excinfo.value)
