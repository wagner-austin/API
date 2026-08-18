"""Tests for reading tree shape out of an XGBoost text dump."""

from __future__ import annotations

import pytest

from covenant_ml.growth_policy.model_shape import mean_leaves_from_xgb_dump
from covenant_ml.growth_policy.types import ERR_NO_TREES

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
    """Counting leaves across an ensemble."""

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
