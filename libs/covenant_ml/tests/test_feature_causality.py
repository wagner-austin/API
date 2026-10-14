"""Feature definitions cannot learn from other rows or future feature distributions."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.features import compute_pairwise_products, compute_pairwise_ratios


@pytest.mark.parametrize("products", [False, True])
def test_future_rows_cannot_change_capped_feature_definition(products: bool) -> None:
    past_values: list[list[float]] = [
        [1.0, 1.0, 1.0, 1.0],
        [3.0, 2.0, 1.0, 1.0],
        [5.0, 3.0, 1.0, 1.0],
    ]
    future_values: list[list[float]] = [[1.0, 1.0, 100.0, 1000.0], [1.0, 1.0, -100.0, -1000.0]]
    past: NDArray[np.float64] = np.asarray(past_values, dtype=np.float64)
    future: NDArray[np.float64] = np.asarray(future_values, dtype=np.float64)
    transform = compute_pairwise_products if products else compute_pairwise_ratios
    before, names = transform(past, ["a", "b", "c", "d"], 2)
    after, new_names = transform(np.vstack((past, future)), ["a", "b", "c", "d"], 2)
    assert names == new_names == (["a*b", "a*c"] if products else ["a/b", "a/c"])
    assert np.array_equal(before, after[: len(past)])
    changed = past.copy()
    changed[0, 0] = 2.0
    control, control_names = transform(changed, ["a", "b", "c", "d"], 2)
    assert control_names == names
    observed: NDArray[np.float64] = control[0]
    original: NDArray[np.float64] = before[0]
    assert np.array_equal(observed, 2 * original)
    assert np.array_equal(control[1:], before[1:])
