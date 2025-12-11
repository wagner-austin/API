from __future__ import annotations

from platform_core import torch_types as torch_types_mod
from platform_core.torch_types import (
    DeviceProtocol,
    TensorProtocol,
    ThreadConfig,
    _TorchModuleProtocol,
    configure_torch_threads,
    get_num_threads,
    set_manual_seed,
)


class _MockThreadConfig:
    """Mock config that satisfies ThreadConfig protocol."""

    def __init__(self, threads: int) -> None:
        self._threads = threads

    def __getitem__(self, key: str) -> int:
        if key == "threads":
            return self._threads
        raise KeyError(key)


class _MockTensor:
    """Mock tensor returned by manual_seed."""

    @property
    def shape(self) -> tuple[int, ...]:
        return ()

    @property
    def dtype(self) -> _MockDType:
        return _MockDType()

    @property
    def device(self) -> _MockDevice:
        return _MockDevice()

    @property
    def grad(self) -> TensorProtocol | None:
        return None

    def numel(self) -> int:
        return 0

    def element_size(self) -> int:
        return 4

    def item(self) -> float:
        return 0.0

    def tolist(self) -> list[float]:
        return []

    def detach(self) -> TensorProtocol:
        return self

    def cpu(self) -> TensorProtocol:
        return self

    def clone(self) -> TensorProtocol:
        return self

    def cuda(self, device: int | None = None) -> TensorProtocol:
        return self

    def to(self, device: DeviceProtocol | str) -> TensorProtocol:
        return self

    def __add__(self, other: TensorProtocol | float | int) -> TensorProtocol:
        return self

    def __mul__(self, other: TensorProtocol | float | int) -> TensorProtocol:
        return self

    def __truediv__(self, other: TensorProtocol | float | int) -> TensorProtocol:
        return self


class _MockDType:
    """Mock dtype."""

    pass


class _MockDevice:
    """Mock device."""

    @property
    def type(self) -> str:
        return "cpu"

    @property
    def index(self) -> int | None:
        return None


class _MockTorchModule:
    """Mock torch module with set_num_threads, manual_seed, get_num_threads methods."""

    def __init__(self, num_threads: int = 4) -> None:
        self.set_num_threads_calls: list[int] = []
        self.manual_seed_calls: list[int] = []
        self._num_threads = num_threads

    def set_num_threads(self, num: int) -> None:
        self.set_num_threads_calls.append(num)

    def manual_seed(self, seed: int) -> TensorProtocol:
        self.manual_seed_calls.append(seed)
        return _MockTensor()

    def get_num_threads(self) -> int:
        return self._num_threads


def test_configure_torch_threads_positive_threads() -> None:
    """Test configure_torch_threads sets threads when positive."""
    cfg: ThreadConfig = _MockThreadConfig(4)
    mock_torch = _MockTorchModule()

    def _mock_import() -> _TorchModuleProtocol:
        return mock_torch

    torch_types_mod._import_torch = _mock_import

    configure_torch_threads(cfg)

    assert mock_torch.set_num_threads_calls == [4]


def test_configure_torch_threads_zero_threads() -> None:
    """Test configure_torch_threads does nothing when threads is 0."""
    cfg: ThreadConfig = _MockThreadConfig(0)
    mock_torch = _MockTorchModule()

    def _mock_import() -> _TorchModuleProtocol:
        return mock_torch

    torch_types_mod._import_torch = _mock_import

    configure_torch_threads(cfg)

    # With threads=0, we should not call set_num_threads
    assert mock_torch.set_num_threads_calls == []


def test_configure_torch_threads_negative_threads() -> None:
    """Test configure_torch_threads does nothing when threads is negative."""
    cfg: ThreadConfig = _MockThreadConfig(-1)
    mock_torch = _MockTorchModule()

    def _mock_import() -> _TorchModuleProtocol:
        return mock_torch

    torch_types_mod._import_torch = _mock_import

    configure_torch_threads(cfg)

    # With threads=-1, we should not call set_num_threads
    assert mock_torch.set_num_threads_calls == []


def test_thread_config_protocol() -> None:
    """Test that our mock satisfies ThreadConfig protocol."""

    def accepts_config(cfg: ThreadConfig) -> int:
        return cfg["threads"]

    test_cfg: ThreadConfig = _MockThreadConfig(8)
    result = accepts_config(test_cfg)
    assert result == 8


def test_set_manual_seed() -> None:
    """Test set_manual_seed calls torch.manual_seed."""
    mock_torch = _MockTorchModule()

    def _mock_import() -> _TorchModuleProtocol:
        return mock_torch

    torch_types_mod._import_torch = _mock_import

    set_manual_seed(42)

    assert mock_torch.manual_seed_calls == [42]


def test_get_num_threads() -> None:
    """Test get_num_threads returns torch.get_num_threads value."""
    mock_torch = _MockTorchModule(num_threads=8)

    def _mock_import() -> _TorchModuleProtocol:
        return mock_torch

    torch_types_mod._import_torch = _mock_import

    result = get_num_threads()

    assert result == 8


def test_default_import_torch_returns_real_torch() -> None:
    """Test _default_import_torch imports the real torch module."""
    from platform_core.torch_types import _default_import_torch

    torch_mod = _default_import_torch()
    # Verify it returns a torch module by calling get_num_threads
    threads = torch_mod.get_num_threads()
    assert threads >= 0
