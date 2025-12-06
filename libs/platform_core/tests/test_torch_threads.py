from __future__ import annotations

from typing import Protocol

from platform_core.torch_types import (
    TensorProtocol,
    ThreadConfig,
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

    def detach(self) -> _MockTensor:
        return self

    def cpu(self) -> _MockTensor:
        return self

    def cuda(self, device: int | None = None) -> _MockTensor:
        return self

    def to(self, device: _MockDevice | str) -> _MockTensor:
        return self

    def __add__(self, other: TensorProtocol | float | int) -> _MockTensor:
        return self

    def __mul__(self, other: TensorProtocol | float | int) -> _MockTensor:
        return self

    def __truediv__(self, other: TensorProtocol | float | int) -> _MockTensor:
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

    def manual_seed(self, seed: int) -> _MockTensor:
        self.manual_seed_calls.append(seed)
        return _MockTensor()

    def get_num_threads(self) -> int:
        return self._num_threads


def test_configure_torch_threads_positive_threads(
    monkeypatch: _MonkeypatchProto,
) -> None:
    """Test configure_torch_threads sets threads when positive."""
    cfg: ThreadConfig = _MockThreadConfig(4)

    mock_torch = _MockTorchModule()

    def mock_import(name: str) -> _MockTorchModule:
        if name == "torch":
            return mock_torch
        raise ImportError(f"No module named {name}")

    monkeypatch.setattr("builtins.__import__", mock_import)

    configure_torch_threads(cfg)

    assert mock_torch.set_num_threads_calls == [4]


def test_configure_torch_threads_zero_threads(
    monkeypatch: _MonkeypatchProto,
) -> None:
    """Test configure_torch_threads does nothing when threads is 0."""
    cfg: ThreadConfig = _MockThreadConfig(0)

    mock_torch = _MockTorchModule()

    def mock_import(name: str) -> _MockTorchModule:
        if name == "torch":
            return mock_torch
        raise ImportError(f"No module named {name}")

    monkeypatch.setattr("builtins.__import__", mock_import)

    configure_torch_threads(cfg)

    assert mock_torch.set_num_threads_calls == []


def test_configure_torch_threads_negative_threads(
    monkeypatch: _MonkeypatchProto,
) -> None:
    """Test configure_torch_threads does nothing when threads is negative."""
    cfg: ThreadConfig = _MockThreadConfig(-1)

    mock_torch = _MockTorchModule()

    def mock_import(name: str) -> _MockTorchModule:
        if name == "torch":
            return mock_torch
        raise ImportError(f"No module named {name}")

    monkeypatch.setattr("builtins.__import__", mock_import)

    configure_torch_threads(cfg)

    assert mock_torch.set_num_threads_calls == []


def test_thread_config_protocol() -> None:
    """Test that our mock satisfies ThreadConfig protocol."""

    def accepts_config(cfg: ThreadConfig) -> int:
        return cfg["threads"]

    test_cfg: ThreadConfig = _MockThreadConfig(8)
    result = accepts_config(test_cfg)
    assert result == 8


def test_set_manual_seed(
    monkeypatch: _MonkeypatchProto,
) -> None:
    """Test set_manual_seed calls torch.manual_seed."""
    mock_torch = _MockTorchModule()

    def mock_import(name: str) -> _MockTorchModule:
        if name == "torch":
            return mock_torch
        raise ImportError(f"No module named {name}")

    monkeypatch.setattr("builtins.__import__", mock_import)

    set_manual_seed(42)

    assert mock_torch.manual_seed_calls == [42]


def test_get_num_threads(
    monkeypatch: _MonkeypatchProto,
) -> None:
    """Test get_num_threads returns torch.get_num_threads value."""
    mock_torch = _MockTorchModule(num_threads=8)

    def mock_import(name: str) -> _MockTorchModule:
        if name == "torch":
            return mock_torch
        raise ImportError(f"No module named {name}")

    monkeypatch.setattr("builtins.__import__", mock_import)

    result = get_num_threads()

    assert result == 8


# Protocol for pytest monkeypatch


class _MockImportFn(Protocol):
    """Protocol for mock __import__ function used in tests."""

    def __call__(self, name: str) -> _MockTorchModule: ...


class _MonkeypatchProto(Protocol):
    def setattr(self, name: str, value: _MockImportFn) -> None: ...
