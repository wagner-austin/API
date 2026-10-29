"""Common rank semantics and class counts have one metric owner."""

from __future__ import annotations

import numpy as np

from covenant_ml.ranking import binary_counts, midranks
from tests._metrics_fixtures import _make_float_array, _make_int_array


def test_midranks_average_tied_ranks() -> None:
    ranks = midranks(_make_float_array([3.0, 1.0, 3.0, 2.0]))
    np.testing.assert_array_equal(ranks, _make_float_array([3.5, 1.0, 3.5, 2.0]))


def test_class_counts_preserve_imbalance_and_accept_ranking_scores() -> None:
    assert binary_counts(_make_int_array([0, 0, 1]), _make_float_array([-2.0, 3.0, 0.0])) == (1, 2)
