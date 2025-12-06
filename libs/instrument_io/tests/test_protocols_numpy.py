"""Tests for _protocols.numpy module."""

from __future__ import annotations

from instrument_io._protocols.numpy import (
    DTypeProtocol,
    NdArray1DProtocol,
    NdArray2DProtocol,
    NdArrayProtocol,
)


class MockDType:
    """Mock dtype for testing."""

    @property
    def name(self) -> str:
        return "float64"


class MockNdArray1D:
    """Mock 1D array for testing Protocol compliance."""

    def __init__(self, data: list[float]) -> None:
        self._data = data

    @property
    def shape(self) -> tuple[int]:
        return (len(self._data),)

    @property
    def dtype(self) -> MockDType:
        return MockDType()

    @property
    def ndim(self) -> int:
        return 1

    @property
    def size(self) -> int:
        return len(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> float:
        return self._data[idx]

    def tolist(self) -> list[float]:
        return self._data.copy()


class MockNdArray2D:
    """Mock 2D array for testing Protocol compliance."""

    def __init__(self, data: list[list[float]]) -> None:
        self._data = data
        # Pre-create row objects for Protocol compliance
        self._rows = [MockNdArray1D(row) for row in data]

    @property
    def shape(self) -> tuple[int, int]:
        if not self._data:
            return (0, 0)
        return (len(self._data), len(self._data[0]))

    @property
    def dtype(self) -> MockDType:
        return MockDType()

    @property
    def ndim(self) -> int:
        return 2

    @property
    def size(self) -> int:
        if not self._data:
            return 0
        return len(self._data) * len(self._data[0])

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> MockNdArray1D:
        return self._rows[idx]

    def tolist(self) -> list[list[float]]:
        return [row.copy() for row in self._data]


class MockNdArrayGeneric1D:
    """Mock 1D array implementing NdArrayProtocol."""

    def __init__(self, data: list[float]) -> None:
        self._data = data

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self._data),)

    @property
    def dtype(self) -> MockDType:
        return MockDType()

    @property
    def ndim(self) -> int:
        return 1

    @property
    def size(self) -> int:
        return len(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> float:
        return self._data[idx]

    def tolist(self) -> list[float] | list[list[float]] | list[int] | list[list[int]]:
        return self._data.copy()


class MockNdArrayGeneric2D:
    """Mock 2D array implementing NdArrayProtocol."""

    def __init__(self, data: list[list[float]]) -> None:
        self._data = data

    @property
    def shape(self) -> tuple[int, ...]:
        if not self._data:
            return (0, 0)
        return (len(self._data), len(self._data[0]))

    @property
    def dtype(self) -> MockDType:
        return MockDType()

    @property
    def ndim(self) -> int:
        return 2

    @property
    def size(self) -> int:
        if not self._data:
            return 0
        return len(self._data) * len(self._data[0])

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> float:
        # NdArrayProtocol.__getitem__ returns float for 1D indexing
        # For 2D, we return first element of row as float
        return self._data[idx][0]

    def tolist(self) -> list[float] | list[list[float]] | list[int] | list[list[int]]:
        return [row.copy() for row in self._data]


def test_dtype_protocol_compliance() -> None:
    mock = MockDType()
    # Verify it matches the protocol
    dtype: DTypeProtocol = mock
    assert dtype.name == "float64"


def test_ndarray_1d_protocol_compliance() -> None:
    mock = MockNdArray1D([1.0, 2.0, 3.0])
    # Verify it matches the protocol
    arr: NdArray1DProtocol = mock
    assert arr.shape == (3,)
    assert len(arr) == 3
    assert arr[0] == 1.0
    assert arr.tolist() == [1.0, 2.0, 3.0]
    assert arr.dtype.name == "float64"


def test_ndarray_2d_protocol_compliance() -> None:
    mock = MockNdArray2D([[1.0, 2.0], [3.0, 4.0]])
    # Verify it matches the protocol
    arr: NdArray2DProtocol = mock
    assert arr.shape == (2, 2)
    assert len(arr) == 2
    # __getitem__ returns NdArray1DProtocol per the Protocol
    row: NdArray1DProtocol = arr[0]
    assert row.tolist() == [1.0, 2.0]
    assert arr.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_ndarray_protocol_1d() -> None:
    mock = MockNdArrayGeneric1D([1.0, 2.0])
    arr: NdArrayProtocol = mock
    assert len(arr) == 2
    assert arr.ndim == 1
    assert arr.size == 2


def test_ndarray_protocol_2d() -> None:
    mock = MockNdArrayGeneric2D([[1.0], [2.0]])
    arr: NdArrayProtocol = mock
    assert len(arr) == 2
    assert arr.ndim == 2
    assert arr.size == 2
