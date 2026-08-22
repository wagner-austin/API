"""Torch protocol shims for the LSTM classifier backend."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from platform_ml.torch_types import (
    DTypeProtocol,
    TensorIterable,
    TensorProtocol,
    TrainableModel,
)


class _SplitsProtocol(Protocol):
    """Protocol for data splits."""

    x_train: NDArray[np.float64]
    y_train: NDArray[np.int64]
    x_val: NDArray[np.float64]
    y_val: NDArray[np.int64]
    x_test: NDArray[np.float64]
    y_test: NDArray[np.int64]

    @property
    def n_train(self) -> int: ...

    @property
    def n_val(self) -> int: ...

    @property
    def n_test(self) -> int: ...

    @property
    def n_total(self) -> int: ...


class _SequenceTensorProto(Protocol):
    """Protocol for 3D sequence tensor that supports select indexing.

    Provides select() method for extracting specific timesteps.
    Shape: (batch, seq_len, hidden_size)
    """

    @property
    def shape(self) -> tuple[int, ...]: ...

    def select(self, dim: int, index: int) -> TensorProtocol:
        """Select a slice along a dimension, removing that dimension."""
        ...

    def detach(self) -> _SequenceTensorProto: ...

    def cpu(self) -> _SequenceTensorProto: ...


class _LSTMLayerProto(Protocol):
    """Protocol for nn.LSTM layer with tuple output."""

    def __call__(
        self, x: TensorProtocol
    ) -> tuple[_SequenceTensorProto, tuple[TensorProtocol, TensorProtocol]]: ...
    def train(self, mode: bool = True) -> _LSTMLayerProto: ...
    def eval(self) -> _LSTMLayerProto: ...
    def state_dict(self) -> dict[str, TensorProtocol]: ...
    def load_state_dict(self, state_dict: dict[str, TensorProtocol]) -> None: ...
    def parameters(self) -> TensorIterable: ...
    def to(self, device: str) -> _LSTMLayerProto: ...
    def cuda(self) -> _LSTMLayerProto: ...


class _LinearLayerProto(Protocol):
    """Protocol for nn.Linear layer."""

    def __call__(self, x: TensorProtocol) -> TensorProtocol: ...
    def train(self, mode: bool = True) -> _LinearLayerProto: ...
    def eval(self) -> _LinearLayerProto: ...
    def state_dict(self) -> dict[str, TensorProtocol]: ...
    def load_state_dict(self, state_dict: dict[str, TensorProtocol]) -> None: ...
    def parameters(self) -> TensorIterable: ...
    def to(self, device: str) -> _LinearLayerProto: ...
    def cuda(self) -> _LinearLayerProto: ...


class _NNLSTMCtor(Protocol):
    """Protocol for nn.LSTM constructor."""

    def __call__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        batch_first: bool,
        dropout: float,
        bidirectional: bool,
    ) -> _LSTMLayerProto: ...


class _NNLinearCtor(Protocol):
    """Protocol for nn.Linear constructor."""

    def __call__(self, in_features: int, out_features: int) -> _LinearLayerProto: ...


class _SoftmaxCtor(Protocol):
    """Protocol for Softmax constructor."""

    def __call__(self, *, dim: int) -> TrainableModel: ...


class _TensorCtor(Protocol):
    """Protocol for torch.tensor constructor."""

    def __call__(self, data: NDArray[np.float64], dtype: DTypeProtocol) -> TensorProtocol: ...


class _EnableGradFactory(Protocol):
    """Protocol for torch.enable_grad context manager factory."""

    def __call__(self) -> AbstractContextManager[None]: ...
