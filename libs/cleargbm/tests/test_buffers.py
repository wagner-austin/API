"""Tests for buffer classes.

Tests for FloatBuffer, IntBuffer, and HistogramBuffer.
Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from cleargbm.buffers import FloatBuffer, HistogramBuffer, IntBuffer


def _float_array(values: tuple[float, ...]) -> NDArray[np.float64]:
    """Create a float64 array from values.

    Args:
        values: Tuple of float values.

    Returns:
        Float64 numpy array.
    """
    arr: NDArray[np.float64] = np.zeros(len(values), dtype=np.float64)
    for i, v in enumerate(values):
        arr[i] = v
    return arr


def _int_array(values: tuple[int, ...]) -> NDArray[np.int64]:
    """Create an int64 array from values.

    Args:
        values: Tuple of int values.

    Returns:
        Int64 numpy array.
    """
    arr: NDArray[np.int64] = np.zeros(len(values), dtype=np.int64)
    for i, v in enumerate(values):
        arr[i] = v
    return arr


def _empty_float_array() -> NDArray[np.float64]:
    """Create an empty float64 array.

    Returns:
        Empty float64 numpy array.
    """
    arr: NDArray[np.float64] = np.zeros(0, dtype=np.float64)
    return arr


def _empty_int_array() -> NDArray[np.int64]:
    """Create an empty int64 array.

    Returns:
        Empty int64 numpy array.
    """
    arr: NDArray[np.int64] = np.zeros(0, dtype=np.int64)
    return arr


class TestFloatBuffer:
    """Tests for FloatBuffer class."""

    def test_init_creates_zero_filled_buffer(self) -> None:
        """Test buffer is initialized with zeros."""
        buf = FloatBuffer(5)
        assert len(buf) == 5
        for i in range(5):
            assert buf.get(i) == 0.0

    def test_init_raises_on_non_positive_size(self) -> None:
        """Test init raises ValueError for non-positive size."""
        with pytest.raises(ValueError, match="size must be positive"):
            FloatBuffer(0)
        with pytest.raises(ValueError, match="size must be positive"):
            FloatBuffer(-1)

    def test_get_returns_value(self) -> None:
        """Test get returns correct value."""
        buf = FloatBuffer(3)
        buf.set(1, 3.14)
        assert buf.get(1) == 3.14

    def test_get_raises_on_out_of_bounds(self) -> None:
        """Test get raises IndexError for out of bounds."""
        buf = FloatBuffer(3)
        with pytest.raises(IndexError, match="index 3 out of bounds"):
            buf.get(3)
        with pytest.raises(IndexError, match="index -1 out of bounds"):
            buf.get(-1)

    def test_set_stores_value(self) -> None:
        """Test set stores value at index."""
        buf = FloatBuffer(3)
        buf.set(0, 1.5)
        buf.set(2, 2.5)
        assert buf.get(0) == 1.5
        assert buf.get(2) == 2.5

    def test_set_raises_on_out_of_bounds(self) -> None:
        """Test set raises IndexError for out of bounds."""
        buf = FloatBuffer(3)
        with pytest.raises(IndexError, match="index 3 out of bounds"):
            buf.set(3, 1.0)
        with pytest.raises(IndexError, match="index -1 out of bounds"):
            buf.set(-1, 1.0)

    def test_add_increments_value(self) -> None:
        """Test add increments value at index."""
        buf = FloatBuffer(3)
        buf.set(1, 2.0)
        buf.add(1, 3.0)
        assert buf.get(1) == 5.0

    def test_add_raises_on_out_of_bounds(self) -> None:
        """Test add raises IndexError for out of bounds."""
        buf = FloatBuffer(3)
        with pytest.raises(IndexError, match="index 3 out of bounds"):
            buf.add(3, 1.0)
        with pytest.raises(IndexError, match="index -1 out of bounds"):
            buf.add(-1, 1.0)

    def test_fill_sets_all_values(self) -> None:
        """Test fill sets all elements to value."""
        buf = FloatBuffer(4)
        buf.fill(7.5)
        for i in range(4):
            assert buf.get(i) == 7.5

    def test_reset_zeros_all_values(self) -> None:
        """Test reset sets all elements to zero."""
        buf = FloatBuffer(3)
        buf.fill(5.0)
        buf.reset()
        for i in range(3):
            assert buf.get(i) == 0.0

    def test_as_array_returns_underlying_array(self) -> None:
        """Test as_array returns the numpy array."""
        buf = FloatBuffer(3)
        buf.set(0, 1.0)
        arr: NDArray[np.float64] = buf.as_array()
        assert arr.item(0) == 1.0
        # Modification through array affects buffer
        arr[1] = 2.0
        assert buf.get(1) == 2.0

    def test_to_tuple_returns_immutable_copy(self) -> None:
        """Test to_tuple returns tuple copy."""
        buf = FloatBuffer(3)
        buf.set(0, 1.0)
        buf.set(1, 2.0)
        buf.set(2, 3.0)
        t = buf.to_tuple()
        assert t == (1.0, 2.0, 3.0)
        # Modification of buffer doesn't affect tuple
        buf.set(0, 9.0)
        assert t == (1.0, 2.0, 3.0)

    def test_from_tuple_creates_buffer(self) -> None:
        """Test from_tuple creates buffer from tuple."""
        buf = FloatBuffer.from_tuple((1.0, 2.0, 3.0))
        assert len(buf) == 3
        assert buf.get(0) == 1.0
        assert buf.get(1) == 2.0
        assert buf.get(2) == 3.0

    def test_from_tuple_raises_on_empty(self) -> None:
        """Test from_tuple raises ValueError for empty tuple."""
        with pytest.raises(ValueError, match="data must not be empty"):
            FloatBuffer.from_tuple(())

    def test_from_array_creates_buffer(self) -> None:
        """Test from_array creates buffer from numpy array."""
        arr: NDArray[np.float64] = _float_array((1.0, 2.0, 3.0))
        buf = FloatBuffer.from_array(arr)
        assert len(buf) == 3
        assert buf.get(0) == 1.0
        assert buf.get(1) == 2.0
        assert buf.get(2) == 3.0
        # Modification of source array doesn't affect buffer
        arr[0] = 9.0
        assert buf.get(0) == 1.0

    def test_from_array_raises_on_empty(self) -> None:
        """Test from_array raises ValueError for empty array."""
        arr: NDArray[np.float64] = _empty_float_array()
        with pytest.raises(ValueError, match="array must not be empty"):
            FloatBuffer.from_array(arr)


class TestIntBuffer:
    """Tests for IntBuffer class."""

    def test_init_creates_zero_filled_buffer(self) -> None:
        """Test buffer is initialized with zeros."""
        buf = IntBuffer(5)
        assert len(buf) == 5
        for i in range(5):
            assert buf.get(i) == 0

    def test_init_raises_on_non_positive_size(self) -> None:
        """Test init raises ValueError for non-positive size."""
        with pytest.raises(ValueError, match="size must be positive"):
            IntBuffer(0)
        with pytest.raises(ValueError, match="size must be positive"):
            IntBuffer(-1)

    def test_get_returns_value(self) -> None:
        """Test get returns correct value."""
        buf = IntBuffer(3)
        buf.set(1, 42)
        assert buf.get(1) == 42

    def test_get_raises_on_out_of_bounds(self) -> None:
        """Test get raises IndexError for out of bounds."""
        buf = IntBuffer(3)
        with pytest.raises(IndexError, match="index 3 out of bounds"):
            buf.get(3)
        with pytest.raises(IndexError, match="index -1 out of bounds"):
            buf.get(-1)

    def test_set_stores_value(self) -> None:
        """Test set stores value at index."""
        buf = IntBuffer(3)
        buf.set(0, 10)
        buf.set(2, 20)
        assert buf.get(0) == 10
        assert buf.get(2) == 20

    def test_set_raises_on_out_of_bounds(self) -> None:
        """Test set raises IndexError for out of bounds."""
        buf = IntBuffer(3)
        with pytest.raises(IndexError, match="index 3 out of bounds"):
            buf.set(3, 1)
        with pytest.raises(IndexError, match="index -1 out of bounds"):
            buf.set(-1, 1)

    def test_add_increments_value(self) -> None:
        """Test add increments value at index."""
        buf = IntBuffer(3)
        buf.set(1, 5)
        buf.add(1, 3)
        assert buf.get(1) == 8

    def test_add_raises_on_out_of_bounds(self) -> None:
        """Test add raises IndexError for out of bounds."""
        buf = IntBuffer(3)
        with pytest.raises(IndexError, match="index 3 out of bounds"):
            buf.add(3, 1)
        with pytest.raises(IndexError, match="index -1 out of bounds"):
            buf.add(-1, 1)

    def test_fill_sets_all_values(self) -> None:
        """Test fill sets all elements to value."""
        buf = IntBuffer(4)
        buf.fill(7)
        for i in range(4):
            assert buf.get(i) == 7

    def test_reset_zeros_all_values(self) -> None:
        """Test reset sets all elements to zero."""
        buf = IntBuffer(3)
        buf.fill(5)
        buf.reset()
        for i in range(3):
            assert buf.get(i) == 0

    def test_as_array_returns_underlying_array(self) -> None:
        """Test as_array returns the numpy array."""
        buf = IntBuffer(3)
        buf.set(0, 1)
        arr: NDArray[np.int64] = buf.as_array()
        assert arr.item(0) == 1
        # Modification through array affects buffer
        arr[1] = 2
        assert buf.get(1) == 2

    def test_to_tuple_returns_immutable_copy(self) -> None:
        """Test to_tuple returns tuple copy."""
        buf = IntBuffer(3)
        buf.set(0, 1)
        buf.set(1, 2)
        buf.set(2, 3)
        t = buf.to_tuple()
        assert t == (1, 2, 3)
        # Modification of buffer doesn't affect tuple
        buf.set(0, 9)
        assert t == (1, 2, 3)

    def test_from_tuple_creates_buffer(self) -> None:
        """Test from_tuple creates buffer from tuple."""
        buf = IntBuffer.from_tuple((1, 2, 3))
        assert len(buf) == 3
        assert buf.get(0) == 1
        assert buf.get(1) == 2
        assert buf.get(2) == 3

    def test_from_tuple_raises_on_empty(self) -> None:
        """Test from_tuple raises ValueError for empty tuple."""
        with pytest.raises(ValueError, match="data must not be empty"):
            IntBuffer.from_tuple(())

    def test_from_array_creates_buffer(self) -> None:
        """Test from_array creates buffer from numpy array."""
        arr: NDArray[np.int64] = _int_array((1, 2, 3))
        buf = IntBuffer.from_array(arr)
        assert len(buf) == 3
        assert buf.get(0) == 1
        assert buf.get(1) == 2
        assert buf.get(2) == 3
        # Modification of source array doesn't affect buffer
        arr[0] = 9
        assert buf.get(0) == 1

    def test_from_array_raises_on_empty(self) -> None:
        """Test from_array raises ValueError for empty array."""
        arr: NDArray[np.int64] = _empty_int_array()
        with pytest.raises(ValueError, match="array must not be empty"):
            IntBuffer.from_array(arr)


class TestHistogramBuffer:
    """Tests for HistogramBuffer class."""

    def test_init_creates_zero_filled_histogram(self) -> None:
        """Test histogram is initialized with zeros."""
        hist = HistogramBuffer(5)
        assert hist.n_bins == 5
        for i in range(5):
            assert hist.get_gradient_sum(i) == 0.0
            assert hist.get_hessian_sum(i) == 0.0
            assert hist.get_count(i) == 0

    def test_init_raises_on_non_positive_n_bins(self) -> None:
        """Test init raises ValueError for non-positive n_bins."""
        with pytest.raises(ValueError, match="n_bins must be positive"):
            HistogramBuffer(0)
        with pytest.raises(ValueError, match="n_bins must be positive"):
            HistogramBuffer(-1)

    def test_get_gradient_sum_returns_value(self) -> None:
        """Test get_gradient_sum returns correct value."""
        hist = HistogramBuffer(3)
        hist.accumulate(1, 2.5, 1.0)
        assert hist.get_gradient_sum(1) == 2.5

    def test_get_gradient_sum_raises_on_out_of_bounds(self) -> None:
        """Test get_gradient_sum raises IndexError for out of bounds."""
        hist = HistogramBuffer(3)
        with pytest.raises(IndexError, match="bin_idx 3 out of bounds"):
            hist.get_gradient_sum(3)
        with pytest.raises(IndexError, match="bin_idx -1 out of bounds"):
            hist.get_gradient_sum(-1)

    def test_get_hessian_sum_returns_value(self) -> None:
        """Test get_hessian_sum returns correct value."""
        hist = HistogramBuffer(3)
        hist.accumulate(1, 2.5, 1.5)
        assert hist.get_hessian_sum(1) == 1.5

    def test_get_hessian_sum_raises_on_out_of_bounds(self) -> None:
        """Test get_hessian_sum raises IndexError for out of bounds."""
        hist = HistogramBuffer(3)
        with pytest.raises(IndexError, match="bin_idx 3 out of bounds"):
            hist.get_hessian_sum(3)
        with pytest.raises(IndexError, match="bin_idx -1 out of bounds"):
            hist.get_hessian_sum(-1)

    def test_get_count_returns_value(self) -> None:
        """Test get_count returns correct value."""
        hist = HistogramBuffer(3)
        hist.accumulate(1, 2.5, 1.0)
        hist.accumulate(1, 1.0, 0.5)
        assert hist.get_count(1) == 2

    def test_get_count_raises_on_out_of_bounds(self) -> None:
        """Test get_count raises IndexError for out of bounds."""
        hist = HistogramBuffer(3)
        with pytest.raises(IndexError, match="bin_idx 3 out of bounds"):
            hist.get_count(3)
        with pytest.raises(IndexError, match="bin_idx -1 out of bounds"):
            hist.get_count(-1)

    def test_accumulate_adds_to_bin(self) -> None:
        """Test accumulate adds sample to bin."""
        hist = HistogramBuffer(3)
        hist.accumulate(1, 2.0, 1.0)
        hist.accumulate(1, 3.0, 2.0)
        assert hist.get_gradient_sum(1) == 5.0
        assert hist.get_hessian_sum(1) == 3.0
        assert hist.get_count(1) == 2

    def test_accumulate_raises_on_out_of_bounds(self) -> None:
        """Test accumulate raises IndexError for out of bounds."""
        hist = HistogramBuffer(3)
        with pytest.raises(IndexError, match="bin_idx 3 out of bounds"):
            hist.accumulate(3, 1.0, 1.0)
        with pytest.raises(IndexError, match="bin_idx -1 out of bounds"):
            hist.accumulate(-1, 1.0, 1.0)

    def test_accumulate_batch_adds_multiple_samples(self) -> None:
        """Test accumulate_batch adds multiple samples."""
        hist = HistogramBuffer(3)
        bin_indices: NDArray[np.int64] = _int_array((0, 1, 1, 2))
        gradients: NDArray[np.float64] = _float_array((1.0, 2.0, 3.0, 4.0))
        hessians: NDArray[np.float64] = _float_array((0.5, 1.0, 1.5, 2.0))
        hist.accumulate_batch(bin_indices, gradients, hessians)
        assert hist.get_gradient_sum(0) == 1.0
        assert hist.get_gradient_sum(1) == 5.0
        assert hist.get_gradient_sum(2) == 4.0
        assert hist.get_hessian_sum(0) == 0.5
        assert hist.get_hessian_sum(1) == 2.5
        assert hist.get_hessian_sum(2) == 2.0
        assert hist.get_count(0) == 1
        assert hist.get_count(1) == 2
        assert hist.get_count(2) == 1

    def test_reset_zeros_all_bins(self) -> None:
        """Test reset zeros all bins."""
        hist = HistogramBuffer(3)
        hist.accumulate(0, 1.0, 1.0)
        hist.accumulate(1, 2.0, 2.0)
        hist.reset()
        for i in range(3):
            assert hist.get_gradient_sum(i) == 0.0
            assert hist.get_hessian_sum(i) == 0.0
            assert hist.get_count(i) == 0

    def test_subtract_into_computes_difference(self) -> None:
        """Test subtract_into computes parent - child."""
        parent = HistogramBuffer(3)
        parent.accumulate(0, 5.0, 4.0)
        parent.accumulate(0, 3.0, 2.0)
        parent.accumulate(1, 2.0, 1.0)

        child = HistogramBuffer(3)
        child.accumulate(0, 3.0, 2.0)

        sibling = HistogramBuffer(3)
        sibling.subtract_into(parent, child)

        assert sibling.get_gradient_sum(0) == 5.0
        assert sibling.get_hessian_sum(0) == 4.0
        assert sibling.get_count(0) == 1
        assert sibling.get_gradient_sum(1) == 2.0
        assert sibling.get_hessian_sum(1) == 1.0
        assert sibling.get_count(1) == 1

    def test_subtract_into_raises_on_mismatched_bins(self) -> None:
        """Test subtract_into raises ValueError for mismatched bins."""
        parent = HistogramBuffer(3)
        child = HistogramBuffer(4)
        sibling = HistogramBuffer(3)
        with pytest.raises(ValueError, match="Histogram bin counts must match"):
            sibling.subtract_into(parent, child)

    def test_copy_from_copies_all_bins(self) -> None:
        """Test copy_from copies all bins."""
        src = HistogramBuffer(3)
        src.accumulate(0, 1.0, 0.5)
        src.accumulate(1, 2.0, 1.0)

        dst = HistogramBuffer(3)
        dst.copy_from(src)

        assert dst.get_gradient_sum(0) == 1.0
        assert dst.get_hessian_sum(0) == 0.5
        assert dst.get_count(0) == 1
        assert dst.get_gradient_sum(1) == 2.0
        # Modification of src doesn't affect dst
        src.accumulate(0, 10.0, 5.0)
        assert dst.get_gradient_sum(0) == 1.0

    def test_copy_from_raises_on_mismatched_bins(self) -> None:
        """Test copy_from raises ValueError for mismatched bins."""
        src = HistogramBuffer(3)
        dst = HistogramBuffer(4)
        with pytest.raises(ValueError, match="Histogram bin counts must match"):
            dst.copy_from(src)

    def test_gradient_sums_array_returns_view(self) -> None:
        """Test gradient_sums_array returns numpy array view."""
        hist = HistogramBuffer(3)
        hist.accumulate(0, 1.0, 0.5)
        arr: NDArray[np.float64] = hist.gradient_sums_array()
        val: float = arr.item(0)
        assert val == 1.0
        # Modification through array affects histogram
        arr[1] = 2.0
        assert hist.get_gradient_sum(1) == 2.0

    def test_hessian_sums_array_returns_view(self) -> None:
        """Test hessian_sums_array returns numpy array view."""
        hist = HistogramBuffer(3)
        hist.accumulate(0, 1.0, 0.5)
        arr: NDArray[np.float64] = hist.hessian_sums_array()
        val: float = arr.item(0)
        assert val == 0.5
        # Modification through array affects histogram
        arr[1] = 2.0
        assert hist.get_hessian_sum(1) == 2.0

    def test_counts_array_returns_view(self) -> None:
        """Test counts_array returns numpy array view."""
        hist = HistogramBuffer(3)
        hist.accumulate(0, 1.0, 0.5)
        arr: NDArray[np.int64] = hist.counts_array()
        val: int = arr.item(0)
        assert val == 1
        # Modification through array affects histogram
        arr[1] = 5
        assert hist.get_count(1) == 5

    def test_gradient_sums_tuple_returns_copy(self) -> None:
        """Test gradient_sums_tuple returns immutable tuple copy."""
        hist = HistogramBuffer(3)
        hist.accumulate(0, 1.0, 0.5)
        hist.accumulate(1, 2.0, 1.0)
        t = hist.gradient_sums_tuple()
        assert t == (1.0, 2.0, 0.0)
        # Modification of histogram doesn't affect tuple
        hist.accumulate(0, 10.0, 5.0)
        assert t == (1.0, 2.0, 0.0)

    def test_hessian_sums_tuple_returns_copy(self) -> None:
        """Test hessian_sums_tuple returns immutable tuple copy."""
        hist = HistogramBuffer(3)
        hist.accumulate(0, 1.0, 0.5)
        hist.accumulate(1, 2.0, 1.0)
        t = hist.hessian_sums_tuple()
        assert t == (0.5, 1.0, 0.0)

    def test_counts_tuple_returns_copy(self) -> None:
        """Test counts_tuple returns immutable tuple copy."""
        hist = HistogramBuffer(3)
        hist.accumulate(0, 1.0, 0.5)
        hist.accumulate(1, 2.0, 1.0)
        hist.accumulate(1, 3.0, 1.5)
        t = hist.counts_tuple()
        assert t == (1, 2, 0)

    def test_from_tuples_creates_histogram(self) -> None:
        """Test from_tuples creates histogram from tuples."""
        hist = HistogramBuffer.from_tuples(
            gradient_sums=(1.0, 2.0, 3.0),
            hessian_sums=(0.5, 1.0, 1.5),
            counts=(1, 2, 3),
        )
        assert hist.n_bins == 3
        assert hist.get_gradient_sum(0) == 1.0
        assert hist.get_gradient_sum(1) == 2.0
        assert hist.get_gradient_sum(2) == 3.0
        assert hist.get_hessian_sum(0) == 0.5
        assert hist.get_hessian_sum(1) == 1.0
        assert hist.get_hessian_sum(2) == 1.5
        assert hist.get_count(0) == 1
        assert hist.get_count(1) == 2
        assert hist.get_count(2) == 3

    def test_from_tuples_raises_on_empty(self) -> None:
        """Test from_tuples raises ValueError for empty tuple."""
        with pytest.raises(ValueError, match="gradient_sums must not be empty"):
            HistogramBuffer.from_tuples(
                gradient_sums=(),
                hessian_sums=(),
                counts=(),
            )

    def test_from_tuples_raises_on_mismatched_hessians(self) -> None:
        """Test from_tuples raises ValueError for mismatched hessian length."""
        with pytest.raises(ValueError, match=r"hessian_sums length .* != gradient_sums length"):
            HistogramBuffer.from_tuples(
                gradient_sums=(1.0, 2.0),
                hessian_sums=(0.5,),
                counts=(1, 2),
            )

    def test_from_tuples_raises_on_mismatched_counts(self) -> None:
        """Test from_tuples raises ValueError for mismatched counts length."""
        with pytest.raises(ValueError, match=r"counts length .* != gradient_sums length"):
            HistogramBuffer.from_tuples(
                gradient_sums=(1.0, 2.0),
                hessian_sums=(0.5, 1.0),
                counts=(1,),
            )

    def test_from_arrays_creates_histogram(self) -> None:
        """Test from_arrays creates histogram from numpy arrays."""
        grad: NDArray[np.float64] = _float_array((1.0, 2.0, 3.0))
        hess: NDArray[np.float64] = _float_array((0.5, 1.0, 1.5))
        counts: NDArray[np.int64] = _int_array((1, 2, 3))
        hist = HistogramBuffer.from_arrays(grad, hess, counts)
        assert hist.n_bins == 3
        assert hist.get_gradient_sum(0) == 1.0
        assert hist.get_gradient_sum(1) == 2.0
        assert hist.get_gradient_sum(2) == 3.0
        # Modification of source arrays doesn't affect histogram
        grad[0] = 9.0
        assert hist.get_gradient_sum(0) == 1.0

    def test_from_arrays_raises_on_empty(self) -> None:
        """Test from_arrays raises ValueError for empty array."""
        grad: NDArray[np.float64] = _empty_float_array()
        hess: NDArray[np.float64] = _empty_float_array()
        counts: NDArray[np.int64] = _empty_int_array()
        with pytest.raises(ValueError, match="gradient_sums must not be empty"):
            HistogramBuffer.from_arrays(grad, hess, counts)

    def test_from_arrays_raises_on_mismatched_hessians(self) -> None:
        """Test from_arrays raises ValueError for mismatched hessian size."""
        grad: NDArray[np.float64] = _float_array((1.0, 2.0))
        hess: NDArray[np.float64] = _float_array((0.5,))
        counts: NDArray[np.int64] = _int_array((1, 2))
        with pytest.raises(ValueError, match=r"hessian_sums size .* != gradient_sums size"):
            HistogramBuffer.from_arrays(grad, hess, counts)

    def test_from_arrays_raises_on_mismatched_counts(self) -> None:
        """Test from_arrays raises ValueError for mismatched counts size."""
        grad: NDArray[np.float64] = _float_array((1.0, 2.0))
        hess: NDArray[np.float64] = _float_array((0.5, 1.0))
        counts: NDArray[np.int64] = _int_array((1,))
        with pytest.raises(ValueError, match=r"counts size .* != gradient_sums size"):
            HistogramBuffer.from_arrays(grad, hess, counts)
