from __future__ import annotations

from typing import Protocol


class PILImage(Protocol):
    """Protocol for PIL.Image.Image without importing PIL."""

    @property
    def mode(self) -> str: ...
    @property
    def size(self) -> tuple[int, int]: ...
    def convert(self, mode: str) -> PILImage: ...
    def save(self, fp: str, format: str | None = None) -> None: ...


class ImageClassificationDataset(Protocol):
    """Protocol for image classification datasets returning (image, label)."""

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> tuple[PILImage, int]: ...


class TensorProtocol(Protocol):
    """Protocol for torch.Tensor without importing torch."""

    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> DTypeProtocol: ...
    @property
    def device(self) -> DeviceProtocol: ...
    @property
    def grad(self) -> TensorProtocol | None: ...

    def numel(self) -> int: ...
    def element_size(self) -> int: ...
    def item(self) -> float | int: ...
    def tolist(self) -> list[float] | list[int] | list[list[float]] | list[list[int]]: ...
    def detach(self) -> TensorProtocol: ...
    def cpu(self) -> TensorProtocol: ...
    def clone(self) -> TensorProtocol: ...
    def cuda(self, device: int | None = None) -> TensorProtocol: ...
    def to(self, device: DeviceProtocol | str) -> TensorProtocol: ...
    def __add__(self, other: TensorProtocol | float | int) -> TensorProtocol: ...
    def __mul__(self, other: TensorProtocol | float | int) -> TensorProtocol: ...
    def __truediv__(self, other: TensorProtocol | float | int) -> TensorProtocol: ...


class DTypeProtocol(Protocol):
    """Protocol for torch.dtype."""

    ...


class DeviceProtocol(Protocol):
    """Protocol for torch.device."""

    @property
    def type(self) -> str: ...
    @property
    def index(self) -> int | None: ...


class TensorIterator(Protocol):
    """Protocol for iterating over tensors (e.g., model parameters)."""

    def __iter__(self) -> TensorIterator: ...
    def __next__(self) -> TensorProtocol: ...


class TensorIterable(Protocol):
    """Protocol for an iterable of tensors."""

    def __iter__(self) -> TensorIterator: ...


class TrainableModel(Protocol):
    """Protocol for a PyTorch model that can be trained."""

    def train(self) -> TrainableModel: ...
    def eval(self) -> TrainableModel: ...
    def __call__(self, x: TensorProtocol) -> TensorProtocol: ...
    def state_dict(self) -> dict[str, TensorProtocol]: ...
    def parameters(self) -> TensorIterable: ...


class ThreadConfig(Protocol):
    """Protocol for configs that provide thread count settings."""

    def __getitem__(self, key: str) -> int: ...


def configure_torch_threads(cfg: ThreadConfig) -> None:
    """Configure PyTorch thread settings from a config dict.

    Args:
        cfg: Config dict with "threads" key containing thread count.
             If threads <= 0, no configuration is performed.
    """
    torch_mod = __import__("torch")
    threads = int(cfg["threads"])
    if threads > 0:
        set_num_threads: _SetNumThreadsFn = torch_mod.set_num_threads
        set_num_threads(threads)


class _SetNumThreadsFn(Protocol):
    def __call__(self, num: int) -> None: ...


def set_manual_seed(seed: int) -> None:
    """Set PyTorch random seed for reproducibility."""
    torch_mod = __import__("torch")
    manual_seed: _ManualSeedFn = torch_mod.manual_seed
    manual_seed(seed)


class _ManualSeedFn(Protocol):
    def __call__(self, seed: int) -> TensorProtocol: ...


def get_num_threads() -> int:
    """Get current PyTorch thread count."""
    torch_mod = __import__("torch")
    get_threads: _GetNumThreadsFn = torch_mod.get_num_threads
    return get_threads()


class _GetNumThreadsFn(Protocol):
    def __call__(self) -> int: ...


__all__ = [
    "DTypeProtocol",
    "DeviceProtocol",
    "ImageClassificationDataset",
    "PILImage",
    "TensorIterable",
    "TensorIterator",
    "TensorProtocol",
    "ThreadConfig",
    "TrainableModel",
    "configure_torch_threads",
    "get_num_threads",
    "set_manual_seed",
]
