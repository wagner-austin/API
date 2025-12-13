"""Tests for platform_ml.torch_types thread and seed configuration.

Tests actual code paths using FakeTorchModule from testing.py.
"""

from __future__ import annotations

from platform_ml import torch_types as torch_types_mod
from platform_ml.testing import FakeTorchModule
from platform_ml.torch_types import (
    ThreadConfig,
    _TorchModuleProtocol,
    configure_torch_threads,
    get_num_threads,
    set_manual_seed,
)


class _FakeThreadConfig:
    """Fake config that satisfies ThreadConfig protocol."""

    def __init__(self, threads: int) -> None:
        self._threads = threads

    def __getitem__(self, key: str) -> int:
        if key == "threads":
            return self._threads
        raise KeyError(key)


def test_configure_torch_threads_positive_threads() -> None:
    """Test configure_torch_threads sets threads when positive."""
    cfg: ThreadConfig = _FakeThreadConfig(4)
    fake_torch = FakeTorchModule()

    def _fake_import() -> _TorchModuleProtocol:
        return fake_torch

    torch_types_mod._import_torch = _fake_import

    configure_torch_threads(cfg)

    assert fake_torch.set_num_threads_calls == [4]


def test_configure_torch_threads_zero_threads() -> None:
    """Test configure_torch_threads does nothing when threads is 0."""
    cfg: ThreadConfig = _FakeThreadConfig(0)
    fake_torch = FakeTorchModule()

    def _fake_import() -> _TorchModuleProtocol:
        return fake_torch

    torch_types_mod._import_torch = _fake_import

    configure_torch_threads(cfg)

    # With threads=0, we should not call set_num_threads
    assert fake_torch.set_num_threads_calls == []


def test_configure_torch_threads_negative_threads() -> None:
    """Test configure_torch_threads does nothing when threads is negative."""
    cfg: ThreadConfig = _FakeThreadConfig(-1)
    fake_torch = FakeTorchModule()

    def _fake_import() -> _TorchModuleProtocol:
        return fake_torch

    torch_types_mod._import_torch = _fake_import

    configure_torch_threads(cfg)

    # With threads=-1, we should not call set_num_threads
    assert fake_torch.set_num_threads_calls == []


def test_thread_config_protocol() -> None:
    """Test that our fake satisfies ThreadConfig protocol."""

    def accepts_config(cfg: ThreadConfig) -> int:
        return cfg["threads"]

    test_cfg: ThreadConfig = _FakeThreadConfig(8)
    result = accepts_config(test_cfg)
    assert result == 8


def test_set_manual_seed() -> None:
    """Test set_manual_seed calls torch.manual_seed."""
    fake_torch = FakeTorchModule()

    def _fake_import() -> _TorchModuleProtocol:
        return fake_torch

    torch_types_mod._import_torch = _fake_import

    set_manual_seed(42)

    assert fake_torch.manual_seed_calls == [42]


def test_get_num_threads() -> None:
    """Test get_num_threads returns torch.get_num_threads value."""
    fake_torch = FakeTorchModule(num_threads=8)

    def _fake_import() -> _TorchModuleProtocol:
        return fake_torch

    torch_types_mod._import_torch = _fake_import

    result = get_num_threads()

    assert result == 8


def test_default_import_torch_returns_real_torch() -> None:
    """Test _default_import_torch imports the real torch module."""
    from platform_ml.torch_types import _default_import_torch

    torch_mod = _default_import_torch()
    # Verify it returns a torch module by calling get_num_threads
    threads = torch_mod.get_num_threads()
    assert threads >= 0
