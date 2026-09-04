"""Torch protocol shims for the MLP regressor backend."""

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
    TrainableModel,
)


class _OptimizerProto(Protocol):
    """Protocol for torch optimizer."""

    def zero_grad(self) -> None: ...
    def step(self) -> None: ...


class _OptimizerCtor(Protocol):
    """Protocol for torch optimizer constructor."""

    def __call__(self, params: TensorIterable, lr: float) -> _OptimizerProto: ...


class _LossProto(Protocol):
    """Protocol for a loss function (MSELoss)."""

    def __call__(self, input: TensorProtocol, target: TensorProtocol) -> TensorProtocol: ...


class _LossCtor(Protocol):
    """Protocol for MSELoss constructor (no arguments)."""

    def __call__(self) -> _LossProto: ...


class _AutocastFactory(Protocol):
    """Protocol for torch.amp.autocast factory."""

    def __call__(
        self, *, device_type: str, dtype: DTypeProtocol
    ) -> AbstractContextManager[None]: ...


class _NNLinearCtor(Protocol):
    """Protocol for nn.Linear constructor."""

    def __call__(self, in_features: int, out_features: int) -> TrainableModel: ...


class _NNBatchNorm1dCtor(Protocol):
    """Protocol for nn.BatchNorm1d constructor."""

    def __call__(self, num_features: int) -> TrainableModel: ...


class _NNReLUCtor(Protocol):
    """Protocol for nn.ReLU constructor."""

    def __call__(self) -> TrainableModel: ...


class _NNDropoutCtor(Protocol):
    """Protocol for nn.Dropout constructor."""

    def __call__(self, p: float) -> TrainableModel: ...


class _NNSequentialCtor(Protocol):
    """Protocol for nn.Sequential constructor."""

    def __call__(self, *modules: TrainableModel) -> TrainableModel: ...


class _TensorCtor(Protocol):
    """Protocol for torch.tensor constructor (float64 input only for regression)."""

    def __call__(self, data: NDArray[np.float64], dtype: DTypeProtocol) -> TensorProtocol: ...


class _NoGradFactory(Protocol):
    """Protocol for torch.no_grad context manager factory."""

    def __call__(self) -> AbstractContextManager[None]: ...


class _CudnnConfigProto(Protocol):
    """Protocol for torch.backends.cudnn config."""

    deterministic: bool
    benchmark: bool


# =============================================================================
# Constants and type guard
# =============================================================================


MLP_REGRESSOR_CAPABILITIES: BackendCapabilities = {
    "supports_train": True,
    "supports_gpu": True,
    "supports_early_stopping": True,
    "supports_feature_importance": False,
    "model_format": "pt",
}
