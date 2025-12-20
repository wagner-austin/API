"""Tests for in-place sanitization of non-finite values.

Ensures the sanitization handles NaN/Inf robustly and in-place.
"""

from __future__ import annotations

import numpy as np

from covenant_ml.datasets.loaders._polars_utils import sanitize_array_inplace


class TestSanitizeInplace:
    """Unit tests for sanitize_array_inplace.

    Tests target both code paths: when non-finite values exist and when all
    values are already finite.
    """

    def test_sanitize_inplace_replaces_nan_and_inf(self) -> None:
        """Replaces NaN and +/-Inf with 0.0 and preserves finite values.

        Args:
            None

        Returns:
            None

        Raises:
            AssertionError: If sanitization fails to replace non-finite values
            or modifies finite values unexpectedly.
        """
        arr = np.zeros((3, 3), dtype=np.float64)
        arr[0, 1] = float("nan")
        arr[1, 0] = float("inf")
        arr[1, 1] = float("-inf")
        arr[0, 2] = 3.0
        arr[1, 2] = 5.0
        arr[2, 0] = 7.0
        arr[2, 1] = 8.0
        arr[2, 2] = 9.0

        sanitize_array_inplace(arr)

        expected = np.zeros((3, 3), dtype=np.float64)
        expected[0, 0] = 0.0
        expected[0, 1] = 0.0
        expected[0, 2] = 3.0
        expected[1, 0] = 0.0
        expected[1, 1] = 0.0
        expected[1, 2] = 5.0
        expected[2, 0] = 7.0
        expected[2, 1] = 8.0
        expected[2, 2] = 9.0

        assert np.array_equal(arr, expected)

    def test_sanitize_inplace_no_change_when_all_finite(self) -> None:
        """Leaves arrays with only finite values unchanged.

        Args:
            None

        Returns:
            None

        Raises:
            AssertionError: If finite-only arrays are modified.
        """
        arr = np.zeros((2, 3), dtype=np.float64)
        arr[0, 0] = 1.0
        arr[0, 1] = 2.0
        arr[0, 2] = 3.0
        arr[1, 0] = 4.0
        arr[1, 1] = 5.0
        arr[1, 2] = 6.0

        before = arr.copy()
        sanitize_array_inplace(arr)
        assert np.array_equal(arr, before)
