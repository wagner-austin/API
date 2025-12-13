"""Public test utilities for platform_ml.

Provides Protocol types and hooks for testing ML components against real code paths.
HTTP transport fakes are defined in tests/ to avoid httpx import in src/.

Usage:
    # For wandb publisher tests:
    from platform_ml.testing import hooks, reset_hooks, WandbModuleProtocol

    def fake_load_wandb() -> WandbModuleProtocol:
        return FakeWandbModule()

    hooks.load_wandb_module = fake_load_wandb
"""

from __future__ import annotations

from collections.abc import Mapping
from types import TracebackType
from typing import Final, Protocol

import numpy as np
from numpy.typing import NDArray
from platform_core.json_utils import JSONValue

from .torch_types import (
    DeviceProtocol,
    DTypeProtocol,
    TensorProtocol,
)

# ---------------------------------------------------------------------------
# Wandb hooks - external service, needs hooks
# ---------------------------------------------------------------------------


class WandbRunProtocol(Protocol):
    """Protocol for wandb.Run interface."""

    @property
    def id(self) -> str:
        """Run ID assigned by wandb."""
        ...


class WandbConfigProtocol(Protocol):
    """Protocol for wandb.config interface."""

    def update(self, d: Mapping[str, JSONValue]) -> None:
        """Update config with dictionary."""
        ...


class WandbTableProtocol(Protocol):
    """Protocol for wandb.Table result."""

    @property
    def columns(self) -> list[str]:
        """Column headers."""
        ...

    @property
    def data(self) -> list[list[float | int | str | bool]]:
        """Table data rows."""
        ...


class WandbTableCtorProtocol(Protocol):
    """Protocol for wandb.Table constructor."""

    def __call__(
        self,
        columns: list[str],
        data: list[list[float | int | str | bool]],
    ) -> WandbTableProtocol:
        """Create a new wandb Table."""
        ...


class WandbModuleProtocol(Protocol):
    """Protocol for wandb module interface."""

    @property
    def run(self) -> WandbRunProtocol | None:
        """Current active run, or None if not initialized."""
        ...

    @property
    def config(self) -> WandbConfigProtocol:
        """Config object for the current run."""
        ...

    @property
    def table_ctor(self) -> WandbTableCtorProtocol:
        """Table constructor for creating wandb Tables."""
        ...

    def init(self, *, project: str, name: str) -> WandbRunProtocol:
        """Initialize a new wandb run."""
        ...

    def log(self, data: Mapping[str, float | int | str | bool | WandbTableProtocol]) -> None:
        """Log metrics to the current run."""
        ...

    def finish(self) -> None:
        """Finish the current run."""
        ...


class LoadWandbModuleCallable(Protocol):
    """Protocol for loading wandb module."""

    def __call__(self) -> WandbModuleProtocol:
        """Load and return the wandb module."""
        ...


class ImportWandbCallable(Protocol):
    """Protocol for importing wandb module."""

    def __call__(self) -> WandbModuleProtocol:
        """Import and return raw wandb module."""
        ...


class CheckWandbAvailableCallable(Protocol):
    """Protocol for checking if wandb is available."""

    def __call__(self) -> bool:
        """Return True if wandb is available, False otherwise."""
        ...


class _Hooks:
    """Mutable container for test hooks.

    Only for external services that cannot be tested otherwise (wandb).
    """

    load_wandb_module: LoadWandbModuleCallable
    import_wandb: ImportWandbCallable
    check_wandb_available: CheckWandbAvailableCallable


def _production_import_wandb() -> WandbModuleProtocol:
    """Production implementation that imports real wandb."""
    raw_wandb: WandbModuleProtocol = __import__("wandb")
    return raw_wandb


def _production_check_wandb_available() -> bool:
    """Production implementation that checks if wandb is installed."""
    import importlib.util

    spec = importlib.util.find_spec("wandb")
    return spec is not None


class _WandbModuleAdapter:
    """Adapter that wraps real wandb module to match WandbModuleProtocol.

    Uses getattr for all attribute access to avoid mypy issues with
    dynamically loaded modules and PascalCase attributes like Table.
    """

    _TABLE_ATTR: Final[str] = "Table"

    def __init__(self, wandb_mod: WandbModuleProtocol) -> None:
        self._wandb = wandb_mod

    @property
    def run(self) -> WandbRunProtocol | None:
        result: WandbRunProtocol | None = getattr(self._wandb, "run", None)
        return result

    @property
    def config(self) -> WandbConfigProtocol:
        result: WandbConfigProtocol = self._wandb.config
        return result

    @property
    def table_ctor(self) -> WandbTableCtorProtocol:
        result: WandbTableCtorProtocol = getattr(self._wandb, self._TABLE_ATTR)
        return result

    def init(self, *, project: str, name: str) -> WandbRunProtocol:
        init_func = self._wandb.init
        result: WandbRunProtocol = init_func(project=project, name=name)
        return result

    def log(self, data: Mapping[str, float | int | str | bool | WandbTableProtocol]) -> None:
        log_func = self._wandb.log
        log_func(data)

    def finish(self) -> None:
        finish_func = self._wandb.finish
        finish_func()


def _production_load_wandb_module() -> WandbModuleProtocol:
    """Production implementation that loads real wandb."""
    from platform_ml.wandb_publisher import WandbUnavailableError

    if not hooks.check_wandb_available():
        raise WandbUnavailableError("wandb package is not installed")

    raw_wandb = hooks.import_wandb()
    return _WandbModuleAdapter(raw_wandb)


# Global hooks instance - only for wandb (external service)
hooks: Final[_Hooks] = _Hooks()


def set_production_hooks() -> None:
    """Set all hooks to production implementations."""
    hooks.check_wandb_available = _production_check_wandb_available
    hooks.import_wandb = _production_import_wandb
    hooks.load_wandb_module = _production_load_wandb_module


def reset_hooks() -> None:
    """Reset hooks to production implementations."""
    set_production_hooks()


# Initialize with production hooks by default
set_production_hooks()


# ---------------------------------------------------------------------------
# Torch test fakes - complete implementations for testing
# ---------------------------------------------------------------------------


class FakeDType:
    """Fake dtype that satisfies DTypeProtocol."""


class FakeDevice:
    """Fake device that satisfies DeviceProtocol."""

    def __init__(self, *, device_type: str = "cpu", index: int | None = None) -> None:
        self._type = device_type
        self._index = index

    @property
    def type(self) -> str:
        return self._type

    @property
    def index(self) -> int | None:
        return self._index


class FakeTensor:
    """Fake tensor that satisfies TensorProtocol.

    Provides complete implementation for all TensorProtocol methods.
    All operations return self for chaining.
    """

    def __init__(
        self,
        *,
        shape: tuple[int, ...] = (),
        device_type: str = "cpu",
    ) -> None:
        self._shape = shape
        self._device = FakeDevice(device_type=device_type)
        self._dtype = FakeDType()

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> DTypeProtocol:
        return self._dtype

    @property
    def device(self) -> DeviceProtocol:
        return self._device

    @property
    def grad(self) -> TensorProtocol | None:
        return None

    def numel(self) -> int:
        result = 1
        for dim in self._shape:
            result *= dim
        return result

    def element_size(self) -> int:
        return 4

    def item(self) -> float:
        return 0.0

    def tolist(self) -> list[float]:
        return []

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return FakeTensor(shape=self._shape, device_type="cpu")

    def clone(self) -> FakeTensor:
        return FakeTensor(shape=self._shape, device_type=self._device.type)

    def cuda(self, device: int | None = None) -> FakeTensor:
        return FakeTensor(shape=self._shape, device_type="cuda")

    def to(self, device: DeviceProtocol | str) -> FakeTensor:
        device_type = device if isinstance(device, str) else device.type
        return FakeTensor(shape=self._shape, device_type=device_type)

    def backward(self) -> None:
        pass

    def numpy(self) -> NDArray[np.float64]:
        return np.zeros(self._shape, dtype=np.float64)

    def argmax(self, dim: int | None = None) -> FakeTensor:
        return FakeTensor(shape=self._shape, device_type=self._device.type)

    def __add__(self, other: TensorProtocol | float | int) -> FakeTensor:
        return self

    def __mul__(self, other: TensorProtocol | float | int) -> FakeTensor:
        return self

    def __truediv__(self, other: TensorProtocol | float | int) -> FakeTensor:
        return self


class FakeNoGradContext:
    """Fake no_grad context manager."""

    def __enter__(self) -> None:
        pass

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass


class FakeCudaModule:
    """Fake cuda module that satisfies _CudaModuleProtocol.

    Configure cuda_available to control is_available() return value.
    """

    def __init__(self, *, cuda_available: bool = False) -> None:
        self._available = cuda_available
        self.is_available_call_count = 0

    def is_available(self) -> bool:
        self.is_available_call_count += 1
        return self._available


class FakeTorchModule:
    """Fake torch module that satisfies _TorchModuleProtocol.

    Configure via constructor parameters:
    - cuda_module: Optional FakeCudaModule instance for call verification
    - cuda_available: Controls cuda.is_available() return value (ignored if cuda_module provided)
    - num_threads: Controls get_num_threads() return value

    Records calls for verification:
    - set_num_threads_calls: List of arguments passed to set_num_threads
    - manual_seed_calls: List of seeds passed to manual_seed
    """

    def __init__(
        self,
        *,
        cuda_module: FakeCudaModule | None = None,
        cuda_available: bool = False,
        num_threads: int = 1,
    ) -> None:
        if cuda_module is not None:
            self._cuda = cuda_module
        else:
            self._cuda = FakeCudaModule(cuda_available=cuda_available)
        self._num_threads = num_threads
        self.set_num_threads_calls: list[int] = []
        self.manual_seed_calls: list[int] = []

    @property
    def cuda(self) -> FakeCudaModule:
        return self._cuda

    def set_num_threads(self, num: int) -> None:
        self.set_num_threads_calls.append(num)

    def manual_seed(self, seed: int) -> FakeTensor:
        self.manual_seed_calls.append(seed)
        return FakeTensor()

    def get_num_threads(self) -> int:
        return self._num_threads

    def tensor(
        self,
        data: NDArray[np.float64] | NDArray[np.int64] | NDArray[np.int32] | list[float] | list[int],
        dtype: DTypeProtocol | None = None,
        device: DeviceProtocol | str | None = None,
    ) -> FakeTensor:
        if isinstance(data, list):
            shape: tuple[int, ...] = (len(data),)
        else:
            # data is NDArray - get shape as tuple of ints
            shape = tuple(int(d) for d in data.shape)
        return FakeTensor(shape=shape)

    def zeros(
        self,
        *size: int,
        dtype: DTypeProtocol | None = None,
        device: DeviceProtocol | str | None = None,
    ) -> FakeTensor:
        return FakeTensor(shape=size)

    def from_numpy(self, ndarray: NDArray[np.float64]) -> FakeTensor:
        return FakeTensor(shape=tuple(ndarray.shape))

    def no_grad(self) -> FakeNoGradContext:
        return FakeNoGradContext()

    def save(self, obj: dict[str, TensorProtocol], f: str) -> None:
        pass

    def load(self, f: str) -> dict[str, TensorProtocol]:
        return {}

    @property
    def float32(self) -> FakeDType:
        return FakeDType()

    @property
    def float16(self) -> FakeDType:
        return FakeDType()

    @property
    def bfloat16(self) -> FakeDType:
        return FakeDType()

    @property
    def long(self) -> FakeDType:
        return FakeDType()

    @property
    def int64(self) -> FakeDType:
        return FakeDType()


__all__ = [
    "FakeCudaModule",
    "FakeDType",
    "FakeDevice",
    "FakeNoGradContext",
    "FakeTensor",
    "FakeTorchModule",
    "LoadWandbModuleCallable",
    "WandbConfigProtocol",
    "WandbModuleProtocol",
    "WandbRunProtocol",
    "WandbTableCtorProtocol",
    "WandbTableProtocol",
    "hooks",
    "reset_hooks",
    "set_production_hooks",
]
