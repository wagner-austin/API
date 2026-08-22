"""Torch protocol shims for the MLP classifier backend."""

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
    """Protocol for data splits (DataSplits or PreprocessedDataSplits)."""

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


class _OptimizerProto(Protocol):
    def zero_grad(self) -> None: ...
    def step(self) -> None: ...


class _OptimizerCtor(Protocol):
    def __call__(self, params: TensorIterable, lr: float) -> _OptimizerProto: ...


class _LossProto(Protocol):
    def __call__(self, logits: TensorProtocol, targets: TensorProtocol) -> TensorProtocol: ...


class _WeightedLossCtor(Protocol):
    def __call__(self, weight: TensorProtocol) -> _LossProto: ...


class _LossCtor(Protocol):
    def __call__(self) -> _LossProto: ...


class _AutocastFactory(Protocol):
    def __call__(
        self, *, device_type: str, dtype: DTypeProtocol
    ) -> AbstractContextManager[None]: ...


class _GradScalerProto(Protocol):
    def scale(self, loss: TensorProtocol) -> TensorProtocol: ...
    def step(self, optimizer: _OptimizerProto) -> None: ...
    def update(self) -> None: ...


class _NNLinearCtor(Protocol):
    def __call__(self, in_features: int, out_features: int) -> TrainableModel: ...


class _NNBatchNorm1dCtor(Protocol):
    def __call__(self, num_features: int) -> TrainableModel: ...


class _NNReLUCtor(Protocol):
    def __call__(self) -> TrainableModel: ...


class _NNDropoutCtor(Protocol):
    def __call__(self, p: float) -> TrainableModel: ...


class _NNSequentialCtor(Protocol):
    def __call__(self, *modules: TrainableModel) -> TrainableModel: ...


class _SoftmaxCtor(Protocol):
    def __call__(self, *, dim: int) -> TrainableModel: ...


class _TensorCtor(Protocol):
    def __call__(
        self, data: NDArray[np.float64] | NDArray[np.int64], dtype: DTypeProtocol
    ) -> TensorProtocol: ...


class _NoGradFactory(Protocol):
    def __call__(self) -> AbstractContextManager[None]: ...


class _CudnnConfigProto(Protocol):
    """Protocol for torch.backends.cudnn config we rely on."""

    deterministic: bool
    benchmark: bool


class _EnableGradFactory(Protocol):
    """Protocol for torch.enable_grad context manager factory."""

    def __call__(self) -> AbstractContextManager[None]: ...
