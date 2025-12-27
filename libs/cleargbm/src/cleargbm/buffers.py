"""Mutable buffer classes for efficient histogram building.

Provides pre-allocated, reusable numpy array buffers to avoid allocation
overhead during tree construction. Buffers support in-place mutation and
can be converted to immutable tuples for serialization.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class FloatBuffer:
    """Pre-allocated mutable float buffer using numpy.

    Provides O(1) element access and in-place mutation without
    allocation overhead. Used for gradients, hessians, and
    intermediate computations during tree building.

    Attributes:
        _data: Internal numpy array storage.
        _size: Number of elements.
    """

    __slots__ = ("_data", "_size")

    def __init__(self, size: int) -> None:
        """Initialize buffer with given size.

        Args:
            size: Number of float elements to allocate.

        Raises:
            ValueError: If size is not positive.
        """
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        self._data: NDArray[np.float64] = np.zeros(size, dtype=np.float64)
        self._size = size

    def __len__(self) -> int:
        """Return buffer size.

        Returns:
            Number of elements in buffer.
        """
        return self._size

    def get(self, index: int) -> float:
        """Get element at index.

        Args:
            index: Element index.

        Returns:
            Float value at index.

        Raises:
            IndexError: If index out of bounds.
        """
        if index < 0 or index >= self._size:
            raise IndexError(f"index {index} out of bounds for size {self._size}")
        value: np.float64 = self._data[index]
        return float(value)

    def set(self, index: int, value: float) -> None:
        """Set element at index.

        Args:
            index: Element index.
            value: Float value to set.

        Raises:
            IndexError: If index out of bounds.
        """
        if index < 0 or index >= self._size:
            raise IndexError(f"index {index} out of bounds for size {self._size}")
        self._data[index] = value

    def add(self, index: int, value: float) -> None:
        """Add value to element at index.

        Args:
            index: Element index.
            value: Float value to add.

        Raises:
            IndexError: If index out of bounds.
        """
        if index < 0 or index >= self._size:
            raise IndexError(f"index {index} out of bounds for size {self._size}")
        current: float = self._data.item(index)
        self._data[index] = current + value

    def fill(self, value: float) -> None:
        """Fill buffer with constant value.

        Args:
            value: Value to fill with.
        """
        self._data.fill(value)

    def reset(self) -> None:
        """Reset all elements to zero."""
        self._data.fill(0.0)

    def as_array(self) -> NDArray[np.float64]:
        """Get underlying numpy array (mutable view).

        Returns:
            The underlying numpy array.
        """
        return self._data

    def to_tuple(self) -> tuple[float, ...]:
        """Convert to immutable tuple for serialization.

        Returns:
            Tuple copy of buffer contents.
        """
        float_list: list[float] = self._data.tolist()
        return tuple(float_list)

    @staticmethod
    def from_tuple(data: tuple[float, ...]) -> FloatBuffer:
        """Create buffer from tuple.

        Args:
            data: Source tuple.

        Returns:
            New FloatBuffer with copied data.

        Raises:
            ValueError: If data is empty.
        """
        if len(data) == 0:
            raise ValueError("data must not be empty")
        buf = FloatBuffer(len(data))
        for i, v in enumerate(data):
            buf._data[i] = v
        return buf

    @staticmethod
    def from_array(arr: NDArray[np.float64]) -> FloatBuffer:
        """Create buffer from numpy array.

        Args:
            arr: Source numpy array.

        Returns:
            New FloatBuffer with copied data.

        Raises:
            ValueError: If array is empty.
        """
        if arr.size == 0:
            raise ValueError("array must not be empty")
        buf = FloatBuffer(arr.size)
        np.copyto(buf._data, arr)
        return buf


class IntBuffer:
    """Pre-allocated mutable int buffer using numpy.

    Provides O(1) element access and in-place mutation without
    allocation overhead. Used for sample counts and bin indices.

    Attributes:
        _data: Internal numpy array storage.
        _size: Number of elements.
    """

    __slots__ = ("_data", "_size")

    def __init__(self, size: int) -> None:
        """Initialize buffer with given size.

        Args:
            size: Number of int elements to allocate.

        Raises:
            ValueError: If size is not positive.
        """
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        self._data: NDArray[np.int64] = np.zeros(size, dtype=np.int64)
        self._size = size

    def __len__(self) -> int:
        """Return buffer size.

        Returns:
            Number of elements in buffer.
        """
        return self._size

    def get(self, index: int) -> int:
        """Get element at index.

        Args:
            index: Element index.

        Returns:
            Int value at index.

        Raises:
            IndexError: If index out of bounds.
        """
        if index < 0 or index >= self._size:
            raise IndexError(f"index {index} out of bounds for size {self._size}")
        value: np.int64 = self._data[index]
        return int(value)

    def set(self, index: int, value: int) -> None:
        """Set element at index.

        Args:
            index: Element index.
            value: Int value to set.

        Raises:
            IndexError: If index out of bounds.
        """
        if index < 0 or index >= self._size:
            raise IndexError(f"index {index} out of bounds for size {self._size}")
        self._data[index] = value

    def add(self, index: int, value: int) -> None:
        """Add value to element at index.

        Args:
            index: Element index.
            value: Int value to add.

        Raises:
            IndexError: If index out of bounds.
        """
        if index < 0 or index >= self._size:
            raise IndexError(f"index {index} out of bounds for size {self._size}")
        current: int = self._data.item(index)
        self._data[index] = current + value

    def fill(self, value: int) -> None:
        """Fill buffer with constant value.

        Args:
            value: Value to fill with.
        """
        self._data.fill(value)

    def reset(self) -> None:
        """Reset all elements to zero."""
        self._data.fill(0)

    def as_array(self) -> NDArray[np.int64]:
        """Get underlying numpy array (mutable view).

        Returns:
            The underlying numpy array.
        """
        return self._data

    def to_tuple(self) -> tuple[int, ...]:
        """Convert to immutable tuple for serialization.

        Returns:
            Tuple copy of buffer contents.
        """
        int_list: list[int] = self._data.tolist()
        return tuple(int_list)

    @staticmethod
    def from_tuple(data: tuple[int, ...]) -> IntBuffer:
        """Create buffer from tuple.

        Args:
            data: Source tuple.

        Returns:
            New IntBuffer with copied data.

        Raises:
            ValueError: If data is empty.
        """
        if len(data) == 0:
            raise ValueError("data must not be empty")
        buf = IntBuffer(len(data))
        for i, v in enumerate(data):
            buf._data[i] = v
        return buf

    @staticmethod
    def from_array(arr: NDArray[np.int64]) -> IntBuffer:
        """Create buffer from numpy array.

        Args:
            arr: Source numpy array.

        Returns:
            New IntBuffer with copied data.

        Raises:
            ValueError: If array is empty.
        """
        if arr.size == 0:
            raise ValueError("array must not be empty")
        buf = IntBuffer(arr.size)
        np.copyto(buf._data, arr)
        return buf


class HistogramBuffer:
    """Pre-allocated histogram accumulator using numpy arrays.

    Stores gradient sums, hessian sums, and counts per bin.
    Supports in-place accumulation and subtraction.

    Attributes:
        _n_bins: Number of histogram bins.
        _grad_sums: Gradient sum per bin.
        _hess_sums: Hessian sum per bin.
        _counts: Sample count per bin.
    """

    __slots__ = ("_counts", "_grad_sums", "_hess_sums", "_n_bins")

    def __init__(self, n_bins: int) -> None:
        """Initialize histogram with given bin count.

        Args:
            n_bins: Number of bins (including NaN bin).

        Raises:
            ValueError: If n_bins is not positive.
        """
        if n_bins <= 0:
            raise ValueError(f"n_bins must be positive, got {n_bins}")
        self._n_bins = n_bins
        self._grad_sums: NDArray[np.float64] = np.zeros(n_bins, dtype=np.float64)
        self._hess_sums: NDArray[np.float64] = np.zeros(n_bins, dtype=np.float64)
        self._counts: NDArray[np.int64] = np.zeros(n_bins, dtype=np.int64)

    @property
    def n_bins(self) -> int:
        """Return number of bins.

        Returns:
            Number of bins in histogram.
        """
        return self._n_bins

    def get_gradient_sum(self, bin_idx: int) -> float:
        """Get gradient sum for a bin.

        Args:
            bin_idx: Bin index.

        Returns:
            Gradient sum for the bin.

        Raises:
            IndexError: If bin_idx out of bounds.
        """
        if bin_idx < 0 or bin_idx >= self._n_bins:
            raise IndexError(f"bin_idx {bin_idx} out of bounds for n_bins {self._n_bins}")
        value: np.float64 = self._grad_sums[bin_idx]
        return float(value)

    def get_hessian_sum(self, bin_idx: int) -> float:
        """Get hessian sum for a bin.

        Args:
            bin_idx: Bin index.

        Returns:
            Hessian sum for the bin.

        Raises:
            IndexError: If bin_idx out of bounds.
        """
        if bin_idx < 0 or bin_idx >= self._n_bins:
            raise IndexError(f"bin_idx {bin_idx} out of bounds for n_bins {self._n_bins}")
        value: np.float64 = self._hess_sums[bin_idx]
        return float(value)

    def get_count(self, bin_idx: int) -> int:
        """Get sample count for a bin.

        Args:
            bin_idx: Bin index.

        Returns:
            Sample count for the bin.

        Raises:
            IndexError: If bin_idx out of bounds.
        """
        if bin_idx < 0 or bin_idx >= self._n_bins:
            raise IndexError(f"bin_idx {bin_idx} out of bounds for n_bins {self._n_bins}")
        value: np.int64 = self._counts[bin_idx]
        return int(value)

    def accumulate(
        self,
        bin_idx: int,
        gradient: float,
        hessian: float,
    ) -> None:
        """Add sample to bin.

        Args:
            bin_idx: Target bin index.
            gradient: Sample gradient.
            hessian: Sample hessian.

        Raises:
            IndexError: If bin_idx out of bounds.
        """
        if bin_idx < 0 or bin_idx >= self._n_bins:
            raise IndexError(f"bin_idx {bin_idx} out of bounds for n_bins {self._n_bins}")
        current_grad: float = self._grad_sums.item(bin_idx)
        current_hess: float = self._hess_sums.item(bin_idx)
        current_count: int = self._counts.item(bin_idx)
        self._grad_sums[bin_idx] = current_grad + gradient
        self._hess_sums[bin_idx] = current_hess + hessian
        self._counts[bin_idx] = current_count + 1

    def accumulate_batch(
        self,
        bin_indices: NDArray[np.int64],
        gradients: NDArray[np.float64],
        hessians: NDArray[np.float64],
    ) -> None:
        """Accumulate multiple samples into histogram using vectorized operations.

        Args:
            bin_indices: Bin index for each sample.
            gradients: Gradient for each sample.
            hessians: Hessian for each sample.
        """
        np.add.at(self._grad_sums, bin_indices, gradients)
        np.add.at(self._hess_sums, bin_indices, hessians)
        np.add.at(self._counts, bin_indices, 1)

    def reset(self) -> None:
        """Reset all bins to zero."""
        self._grad_sums.fill(0.0)
        self._hess_sums.fill(0.0)
        self._counts.fill(0)

    def subtract_into(
        self,
        parent: HistogramBuffer,
        child: HistogramBuffer,
    ) -> None:
        """Compute self = parent - child (sibling subtraction).

        Used for the histogram trick: sibling = parent - smaller_child.

        Args:
            parent: Parent histogram.
            child: Child histogram to subtract.

        Raises:
            ValueError: If bin counts don't match.
        """
        if parent._n_bins != self._n_bins or child._n_bins != self._n_bins:
            raise ValueError(
                f"Histogram bin counts must match: "
                f"self={self._n_bins}, parent={parent._n_bins}, child={child._n_bins}"
            )
        np.subtract(parent._grad_sums, child._grad_sums, out=self._grad_sums)
        np.subtract(parent._hess_sums, child._hess_sums, out=self._hess_sums)
        np.subtract(parent._counts, child._counts, out=self._counts)

    def copy_from(self, other: HistogramBuffer) -> None:
        """Copy contents from another histogram buffer.

        Args:
            other: Source histogram buffer.

        Raises:
            ValueError: If bin counts don't match.
        """
        if other._n_bins != self._n_bins:
            raise ValueError(
                f"Histogram bin counts must match: self={self._n_bins}, other={other._n_bins}"
            )
        np.copyto(self._grad_sums, other._grad_sums)
        np.copyto(self._hess_sums, other._hess_sums)
        np.copyto(self._counts, other._counts)

    def gradient_sums_array(self) -> NDArray[np.float64]:
        """Get gradient sums as numpy array (mutable view).

        Returns:
            Numpy array of gradient sums.
        """
        return self._grad_sums

    def hessian_sums_array(self) -> NDArray[np.float64]:
        """Get hessian sums as numpy array (mutable view).

        Returns:
            Numpy array of hessian sums.
        """
        return self._hess_sums

    def counts_array(self) -> NDArray[np.int64]:
        """Get counts as numpy array (mutable view).

        Returns:
            Numpy array of sample counts.
        """
        return self._counts

    def gradient_sums_tuple(self) -> tuple[float, ...]:
        """Get gradient sums as immutable tuple.

        Returns:
            Tuple of gradient sums.
        """
        float_list: list[float] = self._grad_sums.tolist()
        return tuple(float_list)

    def hessian_sums_tuple(self) -> tuple[float, ...]:
        """Get hessian sums as immutable tuple.

        Returns:
            Tuple of hessian sums.
        """
        float_list: list[float] = self._hess_sums.tolist()
        return tuple(float_list)

    def counts_tuple(self) -> tuple[int, ...]:
        """Get counts as immutable tuple.

        Returns:
            Tuple of sample counts.
        """
        int_list: list[int] = self._counts.tolist()
        return tuple(int_list)

    @staticmethod
    def from_tuples(
        gradient_sums: tuple[float, ...],
        hessian_sums: tuple[float, ...],
        counts: tuple[int, ...],
    ) -> HistogramBuffer:
        """Create histogram buffer from tuples.

        Args:
            gradient_sums: Gradient sums per bin.
            hessian_sums: Hessian sums per bin.
            counts: Sample counts per bin.

        Returns:
            New HistogramBuffer with copied data.

        Raises:
            ValueError: If tuple lengths don't match or are empty.
        """
        n_bins = len(gradient_sums)
        if n_bins == 0:
            raise ValueError("gradient_sums must not be empty")
        if len(hessian_sums) != n_bins:
            raise ValueError(
                f"hessian_sums length {len(hessian_sums)} != gradient_sums length {n_bins}"
            )
        if len(counts) != n_bins:
            raise ValueError(f"counts length {len(counts)} != gradient_sums length {n_bins}")

        buf = HistogramBuffer(n_bins)
        for i in range(n_bins):
            buf._grad_sums[i] = gradient_sums[i]
            buf._hess_sums[i] = hessian_sums[i]
            buf._counts[i] = counts[i]
        return buf

    @staticmethod
    def from_arrays(
        gradient_sums: NDArray[np.float64],
        hessian_sums: NDArray[np.float64],
        counts: NDArray[np.int64],
    ) -> HistogramBuffer:
        """Create histogram buffer from numpy arrays.

        Args:
            gradient_sums: Gradient sums per bin.
            hessian_sums: Hessian sums per bin.
            counts: Sample counts per bin.

        Returns:
            New HistogramBuffer with copied data.

        Raises:
            ValueError: If array sizes don't match or are empty.
        """
        n_bins = gradient_sums.size
        if n_bins == 0:
            raise ValueError("gradient_sums must not be empty")
        if hessian_sums.size != n_bins:
            raise ValueError(
                f"hessian_sums size {hessian_sums.size} != gradient_sums size {n_bins}"
            )
        if counts.size != n_bins:
            raise ValueError(f"counts size {counts.size} != gradient_sums size {n_bins}")

        buf = HistogramBuffer(n_bins)
        np.copyto(buf._grad_sums, gradient_sums)
        np.copyto(buf._hess_sums, hessian_sums)
        np.copyto(buf._counts, counts)
        return buf


__all__ = [
    "FloatBuffer",
    "HistogramBuffer",
    "IntBuffer",
]
