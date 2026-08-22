"""Torch protocol shims for the LSTM regressor backend."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol

import numpy as np
from covenant_ml.backends.protocol import BackendCapabilities
from numpy.typing import NDArray
from platform_ml.torch_types import (
    DTypeProtocol,
    TensorIterable,
    TensorProtocol,
)


class _SequenceTensorProto(Protocol):
    """Protocol for 3D sequence tensor that supports select indexing.

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


class _OptimizerProto(Protocol):
    """Protocol for optimizer."""

    def zero_grad(self) -> None: ...
    def step(self) -> None: ...


class _OptimizerCtor(Protocol):
    """Protocol for optimizer constructor."""

    def __call__(self, params: TensorIterable, lr: float) -> _OptimizerProto: ...


class _LossProto(Protocol):
    """Protocol for loss function (MSELoss)."""

    def __call__(self, input: TensorProtocol, target: TensorProtocol) -> TensorProtocol: ...


class _LossCtor(Protocol):
    """Protocol for MSELoss constructor (no arguments)."""

    def __call__(self) -> _LossProto: ...


class _GradScalerProto(Protocol):
    """Protocol for gradient scaler."""

    def scale(self, loss: TensorProtocol) -> TensorProtocol: ...
    def step(self, optimizer: _OptimizerProto) -> None: ...
    def update(self) -> None: ...


class _AutocastFactory(Protocol):
    """Protocol for autocast context manager factory."""

    def __call__(
        self, device_type: str, *, dtype: DTypeProtocol
    ) -> AbstractContextManager[None]: ...


class _CudnnConfigProto(Protocol):
    """Protocol for torch.backends.cudnn config."""

    deterministic: bool
    benchmark: bool


class _TensorCtor(Protocol):
    """Protocol for torch.tensor constructor."""

    def __call__(self, data: NDArray[np.float64], dtype: DTypeProtocol) -> TensorProtocol: ...


class _NoGradFactory(Protocol):
    """Protocol for torch.no_grad context manager factory."""

    def __call__(self) -> AbstractContextManager[None]: ...


# =============================================================================
# Constants and type guard
# =============================================================================


LSTM_REGRESSOR_CAPABILITIES: BackendCapabilities = {
    "supports_train": True,
    "supports_gpu": True,
    "supports_early_stopping": True,
    "supports_feature_importance": False,
    "model_format": "pt",
}
